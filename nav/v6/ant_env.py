"""V6-owned Safe MAMuJoCo Ant-2x4 adapter.

Only the official simulator, reward and cost are reused.  The legacy V5 Ant
safety module import, QP target, warm-start actor and survival head are
intentionally absent.  The adapter exposes the V6 safety features:

* ``h``: a frozen scalar safety value, repeated to each agent;
* ``grad``: ``d h / d geometry`` (9-D), repeated to each agent;
* ``h_vec``: the four analytic margins, for diagnostics only;
* the official task reward and official safety cost separately.

The actor never receives the task value or the expert/QP target.  The h is
frozen and safety-only; the actor learns how to trade task reward against the
cost penalty.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .ant_safety import (
    GEOM_DIM,
    FixedAntSafetyProvider,
    geometry_from_env,
)


def _default_mamujoco_root() -> Path:
    configured = os.environ.get('SAFE_MAMUJOCO_ROOT')
    if configured:
        return Path(configured).expanduser()
    # Keep the historical sibling-checkout default, but make it portable.
    return Path(__file__).resolve().parents[2].parent / 'Safe-MARL-proposal_v2'


def _import_mujoco_multi():
    root = _default_mamujoco_root()
    package = root / 'third_party/mappo_lag/MAPPO-Lagrangian'
    for path in (str(root), str(package)):
        if path not in sys.path:
            sys.path.insert(0, path)
    from mappo_lagrangian.envs.safety_ma_mujoco.safety_multiagent_mujoco import (
        MujocoMulti,
    )
    return MujocoMulti


def _scalar(value: Any, default: float = 0.0) -> float:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    return float(array[0]) if array.size else float(default)


class Ant2x4Env:
    """Single V6 Ant-2x4 environment with h/grad features and official costs.

    ``safety_provider`` must expose ``value_and_grad(geom) -> (h, grad)``.
    When omitted, the analytic threshold-aligned margin is used.  The learned
    V6 h is passed in by the training script only after it has been frozen.
    """

    def __init__(
        self,
        seed: int = 0,
        episode_limit: int = 1000,
        agent_obsk: int = 1,
        *,
        h_kappa: float = 0.0,
        h_kappa_z: float = 0.0,
        upright_scale: float = 0.2,
        safety_provider: Optional[Any] = None,
        h_scale: float = 0.5,
        h_offset: float = 0.0,
    ):
        MujocoMulti = _import_mujoco_multi()
        self.env = MujocoMulti(env_args=dict(
            scenario='Ant-v2',
            agent_conf='2x4',
            agent_obsk=int(agent_obsk),
            episode_limit=int(episode_limit),
        ))
        self.num_agents = int(self.env.n_agents)
        self.observation_dim = int(self.env.observation_space[0].shape[0])
        self.action_dim = int(self.env.action_space[0].shape[0])
        self.max_step = int(episode_limit)
        # Match upstream MAPPO: the centralized state is the normalized
        # ``[official_obs, agent_id]`` vector (31-D for Ant-2x4), not the raw
        # 29-D simulator observation.  Using raw state here caused a large
        # value-scale mismatch with the V5 pure-MAPPO baseline.
        self.global_state_dim = int(len(self.env.get_state()[0]))
        self.seed = int(seed)
        self.step_count = 0

        self.h_kappa = float(h_kappa)
        self.h_kappa_z = float(h_kappa_z)
        self.upright_scale = float(upright_scale)
        self.provider = safety_provider or FixedAntSafetyProvider(
            scale=h_scale, offset=h_offset,
            kappa=h_kappa, kappa_z=h_kappa_z,
        )

        self._obs: Optional[np.ndarray] = None
        self._global_state: Optional[np.ndarray] = None
        self._geom: Optional[np.ndarray] = None
        self._margins = None
        self._h_scalar = 0.0
        self._h_vec = np.zeros(4, dtype=np.float32)
        self._grad = np.zeros(GEOM_DIM, dtype=np.float32)
        self._h_agent = np.zeros((self.num_agents, 1), dtype=np.float32)
        self._grad_agent = np.zeros(
            (self.num_agents, GEOM_DIM), dtype=np.float32)
        self._torso = np.zeros(3, dtype=np.float32)

    # ------------------------------------------------------------------
    # Safety feature refresh
    # ------------------------------------------------------------------
    def _refresh(self) -> None:
        geom, margins = geometry_from_env(
            self.env.env, kappa=self.h_kappa, kappa_z=self.h_kappa_z)
        h, grad = self.provider.value_and_grad(geom)
        grad = np.asarray(grad, dtype=np.float32).reshape(GEOM_DIM)
        self._geom = np.asarray(geom, dtype=np.float32).copy()
        self._margins = margins
        self._h_scalar = float(h)
        self._h_vec = np.asarray(margins.vector, dtype=np.float32).copy()
        self._grad = grad.copy()
        self._h_agent = np.full(
            (self.num_agents, 1), self._h_scalar, dtype=np.float32)
        self._grad_agent = np.repeat(
            grad[None, :], self.num_agents, axis=0)
        self._torso = np.asarray(
            self.env.env.get_body_com('torso'), dtype=np.float32).copy()

    def _global(self) -> np.ndarray:
        # ``MujocoMulti.get_state`` returns the upstream normalized
        # ``[state, agent_id]`` vector.  Keep the same representation as the
        # V5 pure-MAPPO reference; the critic must not see raw simulator state.
        return np.asarray(
            self.env.get_state()[0], dtype=np.float32).copy()

    # ------------------------------------------------------------------
    # Gym-like API
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.seed = int(seed)
        np.random.seed(self.seed)
        try:
            self.env.env.seed(self.seed)
        except Exception:
            pass
        obs, _share_obs, _avail = self.env.reset()
        self.step_count = 0
        self._obs = np.asarray(obs, dtype=np.float32).copy()
        self._global_state = self._global()
        self._refresh()
        return (
            self._obs.copy(),
            self._h_agent.copy(),
            self._grad_agent.copy(),
            self._h_vec.copy(),
            float(self._h_scalar),
            self._geom.copy(),
            self._global_state.copy(),
        )

    def step(self, actions: np.ndarray):
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.num_agents, self.action_dim):
            raise ValueError(
                f'expected actions {(self.num_agents, self.action_dim)}, '
                f'got {tuple(actions.shape)}')
        obs, _share_obs, rewards, dones, infos, _avail = self.env.step(
            [actions[i] for i in range(self.num_agents)])

        info = infos[0]
        cost = _scalar(info.get('cost', 0.0))
        wall = _scalar(info.get('cost_obj', 0.0))
        topple = _scalar(info.get('cost_done', 0.0))
        reward = _scalar(rewards)
        done = bool(np.all(dones))
        bad_transition = bool(info.get('bad_transition', False))
        next_step = self.step_count + 1
        truncated = bool(done and (bad_transition or next_step >= self.max_step))
        terminated = bool(done and not truncated)
        self.step_count = next_step

        self._obs = np.asarray(obs, dtype=np.float32).copy()
        self._global_state = self._global()
        self._refresh()

        info_out = dict(
            reward=float(reward),
            forward_reward=float(_scalar(info.get('reward_forward', 0.0))),
            ctrl_reward=float(_scalar(info.get('reward_ctrl', 0.0))),
            contact_reward=float(_scalar(info.get('reward_contact', 0.0))),
            survive_reward=float(_scalar(info.get('reward_survive', 0.0))),
            cost=float(cost),
            wall=float(wall),
            collision=float(wall),
            topple=float(topple),
            cost_obj=float(wall),
            cost_done=float(topple),
            safety=float(1.0 - cost),
            h=float(self._h_scalar),
            h_scalar=float(self._h_scalar),
            h_vec=self._h_vec.copy(),
            h_grad=self._grad.copy(),
            geom=self._geom.copy(),
            margins=self._margins.components,
            step=int(self.step_count),
            x=float(self._torso[0]),
            y=float(self._torso[1]),
            z=float(self._torso[2]),
            terminated=bool(terminated),
            truncated=bool(truncated),
        )
        return (
            self._obs.copy(),
            self._h_agent.copy(),
            self._grad_agent.copy(),
            self._h_vec.copy(),
            float(self._h_scalar),
            self._geom.copy(),
            self._global_state.copy(),
            float(reward),
            float(cost),
            bool(terminated),
            bool(truncated),
            info_out,
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def get_geometry(self) -> np.ndarray:
        if self._geom is None:
            self._global_state = self._global()
            self._refresh()
        return self._geom.copy()

    def get_margins(self):
        if self._margins is None:
            self._refresh()
        return self._margins

    def get_torso_xyz(self) -> np.ndarray:
        return self._torso.copy()

    def get_global_state(self) -> np.ndarray:
        if self._global_state is None:
            self._global_state = self._global()
        return self._global_state.copy()

    def get_env_info(self) -> dict[str, Any]:
        return dict(
            n_agents=int(self.num_agents),
            obs_dim=int(self.observation_dim),
            action_dim=int(self.action_dim),
            global_state_dim=int(self.global_state_dim),
            episode_limit=int(self.max_step),
        )

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass

    def __del__(self):
        self.close()


__all__ = ['Ant2x4Env']
