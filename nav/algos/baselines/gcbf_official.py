"""Evaluate vendored official GCBF/GCBF+ models on fixed scene keys.

The official ``gcbf_plus.py`` and CBF-QP modules import ``jaxproxqp`` at module
load time. GCBF+ inference in the official code uses the learned actor only
(``2 * actor + u_ref``); it does not call the QP solver. We therefore provide a
small import-only stub when the real ``jaxproxqp`` package is absent. The stub
is *not* a QP solver and will raise if a QP method is executed. This keeps the
official GCBF+ checkpoint/evaluation path usable while making the missing QP
dependency explicit.

For training, or for centralized/decentralized CBF-QP, a real solver is still
required.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

_VENDOR = Path(__file__).resolve().parents[3] / 'third_party' / 'gcbfplus'
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))


class _MissingQP:
    """Import-only sentinel; never a substitute for the real QP solver."""

    class QPModel:
        @staticmethod
        def create(*args, **kwargs):
            raise RuntimeError(
                'jaxproxqp is not installed; this path requires the real QP '
                'solver (centralized/dec_share_cbf or GCBF training).')

    class Settings:
        @staticmethod
        def default():
            raise RuntimeError('jaxproxqp is not installed')

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            'jaxproxqp is not installed; GCBF+ inference does not call the QP, '
            'but QP algorithms are unavailable.')


def _ensure_importable():
    try:
        import jaxproxqp  # noqa: F401
        return False
    except Exception:
        package = types.ModuleType('jaxproxqp')
        package.JaxProxQP = _MissingQP
        module = types.ModuleType('jaxproxqp.jaxproxqp')
        module.JaxProxQP = _MissingQP
        package.jaxproxqp = module
        sys.modules['jaxproxqp'] = package
        sys.modules['jaxproxqp.jaxproxqp'] = module
        return True


def load_official_model(path, step=None, num_agents=8, num_obstacles=8,
                        n_rays=32, max_step=256, area_size=4.0):
    """Load the official pretrained model and return (env, algo, config, step)."""
    import yaml

    _ensure_importable()
    from gcbfplus.algo import make_algo
    from gcbfplus.env import make_env

    path = Path(path)
    with open(path / 'config.yaml') as stream:
        config = yaml.load(stream, Loader=yaml.UnsafeLoader)
    env = make_env(
        env_id=config.env,
        num_agents=num_agents,
        num_obs=num_obstacles,
        area_size=area_size,
        max_step=max_step,
        n_rays=n_rays,
    )
    model_dir = path / 'models'
    if step is None:
        step = max(int(p.name) for p in model_dir.iterdir() if p.name.isdigit())
    algo = make_algo(
        algo=config.algo,
        env=env,
        node_dim=env.node_dim,
        edge_dim=env.edge_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        n_agents=env.num_agents,
        gnn_layers=config.gnn_layers,
        batch_size=config.batch_size,
        buffer_size=config.buffer_size,
        horizon=getattr(config, 'horizon', 32),
        lr_actor=config.lr_actor,
        lr_cbf=config.lr_cbf,
        alpha=config.alpha,
        eps=getattr(config, 'eps', 0.02),
        inner_epoch=getattr(config, 'inner_epoch', 8),
        loss_action_coef=config.loss_action_coef,
        loss_unsafe_coef=config.loss_unsafe_coef,
        loss_safe_coef=config.loss_safe_coef,
        loss_h_dot_coef=config.loss_h_dot_coef,
        max_grad_norm=2.0,
        seed=getattr(config, 'seed', 0),
    )
    algo.load(str(model_dir), int(step))
    return env, algo, config, int(step)


def evaluate_official(env, algo, seeds, max_step=256, reward_scale=20.0,
                      reach_bonus=5.0):
    """Official any-time metrics plus a common task return on fixed seeds."""
    import jax
    import jax.numpy as jnp
    import jax.random as jr

    from gcbfplus.utils.utils import jax_jit_np, jax_vmap, tree_index

    act_fn = jax.jit(algo.act)
    rollout_fn = jax_jit_np(env.rollout_fn(act_fn, max_step))
    collision_fn = jax_jit_np(jax_vmap(env.collision_mask))
    finish_fn = jax_jit_np(jax_vmap(env.finish_mask))
    warning_fn = jax_jit_np(jax_vmap(env.unsafe_mask))
    agent_states_fn = jax_vmap(
        lambda graph: graph.type_states(type_idx=0, n_type=env.num_agents))

    rows = []
    for seed in seeds:
        rollout = rollout_fn(jr.PRNGKey(int(seed)))
        collision = np.asarray(collision_fn(rollout.Tp1_graph), dtype=bool)
        reached = np.asarray(finish_fn(rollout.Tp1_graph), dtype=bool)
        warning = np.asarray(warning_fn(rollout.Tp1_graph), dtype=bool)
        # Tp1 includes the initial state; T_action/reward cover T transitions.
        t_steps = collision.shape[0] - 1
        collision = collision[1:]
        reached = reached[1:]
        warning = warning[1:]
        collision_any = collision.any(axis=0)
        reached_any = reached.any(axis=0)
        reached_final = reached[-1]
        collision_final = collision[-1]
        states = np.asarray(
            agent_states_fn(rollout.Tp1_graph), dtype=np.float32)
        goals = np.asarray(
            tree_index(rollout.Tp1_graph, 0).env_states.goal,
            dtype=np.float32)
        dist = np.linalg.norm(states[..., :2] - goals[None, :, :2], axis=-1)
        progress = dist[:-1] - dist[1:]
        task_return = progress * float(reward_scale) + float(reach_bonus) * reached
        collision_rate = float(collision.mean())
        collision_episode = bool(collision_any.any())
        rows.append(dict(
            episode_return=float(task_return.sum(axis=0).mean()),
            official_control_reward=float(np.asarray(rollout.T_reward).sum()),
            official_cost=float(np.asarray(rollout.T_cost).sum()),
            safety=1.0 - collision_rate,
            unsafe=collision_rate,
            safe_mask=float((~collision).mean()),
            warning=float(warning.mean()),
            collision_episode=float(collision_episode),
            safe_episode=float(not collision_episode),
            warning_episode=float(warning.any()),
            collision_final_any=float(collision_final.any()),
            reach=float(reached_any.mean()),
            reach_episode_any=float(reached_any.any()),
            reach_final=float(reached_final.mean()),
            reach_final_any=float(reached_final.any()),
            official_safety=float((~collision_any).mean()),
            official_reach=float(reached_any.mean()),
            official_success=float(((~collision_any) & reached_any).mean()),
            action_abs=float(np.abs(np.asarray(rollout.T_action)).mean()),
            steps=int(t_steps),
        ))
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


__all__ = [
    '_ensure_importable',
    'evaluate_official',
    'load_official_model',
]
