"""Official Safe MAMuJoCo Ant-2x4 adapter for V5 feasibility experiments.

The environment itself is the upstream ``MujocoMulti(Ant-v2, 2x4)`` stack from
the separate ``mappo_lag`` conda environment.  This wrapper deliberately does
not use the upstream MAPPO-Lag actor or QP: it exposes observations, official
reward/cost, and the fixed V5 Ant margins for a QP-free policy.

The bridge is intentionally isolated here while the Ant stack is evaluated;
the navigation V5 environment remains independent of MuJoCo.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from algos.safe_policy.ant_h import (
    finite_difference_grad,
    margins_from_env,
)


def _default_mamujoco_root() -> Path:
    configured = os.environ.get('SAFE_MAMUJOCO_ROOT')
    if configured:
        return Path(configured).expanduser()
    # Keep the historical sibling-checkout default, but make it portable.
    return Path(__file__).resolve().parents[3].parent / 'Safe-MARL-proposal_v2'


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


class Ant2x4Env:
    """Single-env adapter with official Ant cost and fixed safety margins."""

    def __init__(self, seed=50, episode_limit=1000, agent_obsk=1,
                 h_kappa=0.0, h_kappa_z=0.0, upright_scale=0.2,
                 safety_provider=None):
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
        self.step_count = 0
        self.seed = int(seed)
        self.h_kappa = float(h_kappa)
        self.h_kappa_z = float(h_kappa_z)
        self.upright_scale = float(upright_scale)
        self.safety_provider = safety_provider
        self._survival_risk = 0.0
        self._obs = None
        self._geometry = None
        self._h = None
        self._grad = None
        self._torso = None

    def _margins(self):
        geometry, fixed_margins = margins_from_env(
            self.env.env, kappa=self.h_kappa, kappa_z=self.h_kappa_z,
            upright_scale=self.upright_scale)
        self._torso = np.asarray(self.env.env.get_body_com('torso'), dtype=np.float32).copy()
        if self.safety_provider is not None:
            pred = self.safety_provider.value_and_local_grad(geometry)
            h_vec = np.asarray(
                pred.output.h_cost.detach().cpu(), dtype=np.float32).reshape(-1)
            if h_vec.size != 4:
                raise ValueError(
                    f'provider must expose 4 Ant cost margins, got {h_vec.size}')
            h_scalar = float(pred.output.h_cost_scalar.detach().cpu().reshape(-1)[0])
            grad = np.asarray(
                pred.grad_cost.detach().cpu(), dtype=np.float32).reshape(-1)
            self._survival_risk = float(
                pred.output.survival_risk.detach().cpu().reshape(-1)[0])
            return geometry.astype(np.float32), h_vec.astype(np.float32), h_scalar, grad
        self._survival_risk = 0.0
        return (
            geometry.astype(np.float32),
            fixed_margins.vector,
            float(fixed_margins.h),
            finite_difference_grad(
                geometry, kappa=self.h_kappa, kappa_z=self.h_kappa_z,
                upright_scale=self.upright_scale),
        )

    def _refresh(self):
        self._geometry, h_vec, h_scalar, grad = self._margins()
        self._h = np.repeat(h_vec[None, :], self.num_agents, axis=0)
        self._grad = np.repeat(grad[None, :], self.num_agents, axis=0)
        return h_vec, h_scalar, grad

    def reset(self, seed=None):
        if seed is not None:
            self.seed = int(seed)
        np.random.seed(self.seed)
        try:
            self.env.env.seed(self.seed)
        except Exception:
            pass
        obs, _share, _avail = self.env.reset()
        self.step_count = 0
        self._obs = np.asarray(obs, dtype=np.float32)
        h_vec, h_scalar, grad = self._refresh()
        return self._obs, self._h, self._grad, h_vec, h_scalar, grad

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.num_agents, self.action_dim):
            raise ValueError(f'expected {(self.num_agents, self.action_dim)} actions, '
                             f'got {actions.shape}')
        obs, _share, rewards, dones, infos, _avail = self.env.step(
            [actions[i] for i in range(self.num_agents)])
        info = infos[0]
        def scalar(key):
            value = info.get(key, 0.0)
            return float(np.asarray(value, dtype=np.float64).reshape(-1)[0])
        cost = scalar('cost')
        wall = scalar('cost_obj')
        topple = scalar('cost_done')
        reward = float(np.asarray(rewards, dtype=np.float64).reshape(-1)[0])
        self.step_count += 1
        self._obs = np.asarray(obs, dtype=np.float32)
        h_vec, h_scalar, grad = self._refresh()
        terminated = bool(np.all(dones)) and self.step_count < self.max_step
        truncated = bool(self.step_count >= self.max_step)
        info_out = dict(
            reward=reward,
            cost=cost,
            wall=wall,
            topple=topple,
            collision=wall,  # wall clearance is the official collision-like cost
            safety=1.0 - cost,
            h=h_scalar,
            h_vec=h_vec.copy(),
            h_grad=grad.copy(),
            survival_risk=float(self._survival_risk),
            step=self.step_count,
            x=float(self._torso[0]),
            y=float(self._torso[1]),
            z=float(self._torso[2]),
        )
        return (self._obs, self._h, self._grad, h_vec, h_scalar, grad,
                terminated, truncated, info_out)

    def action_gradient_h(self, actions, eps=0.02):
        """Finite-difference d h_{t+1} / d a_t in the actual simulator.

        This is a diagnostic / optional QP-free shield.  It saves and restores
        the MuJoCo state, so it does not advance the environment.
        """
        actions = np.asarray(actions, dtype=np.float64).reshape(-1)
        sim = self.env.env.sim
        state = sim.get_state()
        frame_skip = int(self.env.env.frame_skip)
        grad = np.zeros_like(actions)
        try:
            for i in range(actions.size):
                vals = []
                for sign in (-1.0, 1.0):
                    sim.set_state(state)
                    sim.forward()
                    a = actions.copy()
                    a[i] = np.clip(a[i] + sign * eps, -1.0, 1.0)
                    self.env.env.do_simulation(a, frame_skip)
                    _, m = margins_from_env(self.env.env)
                    vals.append(m.h)
                grad[i] = (vals[1] - vals[0]) / (2.0 * eps)
        finally:
            sim.set_state(state)
            sim.forward()
        return grad.astype(np.float32)

    def get_geometry(self):
        return np.asarray(self._geometry, dtype=np.float32).copy()

    def get_torso_xyz(self):
        return np.asarray(self._torso, dtype=np.float32).copy()

    def get_state(self):
        return np.concatenate([self._obs[i] for i in range(self.num_agents)],
                              axis=0).astype(np.float32)

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass


__all__ = ['Ant2x4Env']
