"""Official centralized CBF-QP baseline with an OSQP fallback solver.

The official GCBF+ repository uses ``jaxproxqp`` for this QP. That package is
not available in this environment. This module keeps the official
DoubleIntegrator CBF, control-affine dynamics, QP matrices, bounds, and
relaxation penalty, but solves the resulting convex QP with OSQP. It is an
external baseline only; V5 never calls this code.

The QP is the model-based ``centralized_cbf`` formulation from the vendored
official code:
    min 0.5 ||u||^2 + 0.5 * 10 ||r||^2 - u_ref^T u + 1e3 * sum(r)
    s.t. -Lg_h u - r <= Lf_h + alpha h,  r >= 0,  u in [-1, 1].
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def _official_cbf(env, k=3):
    from .gcbf_official import _ensure_importable
    _ensure_importable()
    from gcbfplus.algo.utils import get_pwise_cbf_fn
    return get_pwise_cbf_fn(env._env, k=int(k))


def build_qp_matrices(env, k=3):
    """Compile the official h/Jacobian/Lie-derivative computation."""
    import jax
    import jax.numpy as jnp

    n_agents = env.num_agents
    cbf = _official_cbf(env, k=k)
    agent_ids = jnp.arange(n_agents)

    @jax.jit
    def matrices(graph):
        def h_aug(new_agent_state):
            new_state = graph.states.at[agent_ids].set(new_agent_state)
            new_graph = env._env.add_edge_feats(graph, new_state)
            return cbf(new_graph)[0].reshape(-1)

        agent_state = graph.type_states(type_idx=0, n_type=n_agents)
        h = h_aug(agent_state)
        h_x = jax.jacfwd(h_aug)(agent_state)
        f, g = env._env.control_affine_dyn(agent_state)
        Lf = jnp.einsum('mjn,jn->m', h_x, f)
        Lg = jnp.einsum('mjn,jnu->mju', h_x, g).reshape(
            h.shape[0], n_agents * env.action_dim)
        u_ref = env._env.u_ref(graph).reshape(-1)
        return h, Lf, Lg, u_ref

    return matrices


def solve_centralized_qp(h, Lf, Lg, u_ref, alpha=1.0,
                         relax_penalty=1e3, eps_abs=1e-5, eps_rel=1e-5):
    """Solve the official centralized QP; return (action, relaxation, status)."""
    import osqp

    h = np.asarray(h, dtype=np.float64)
    Lf = np.asarray(Lf, dtype=np.float64)
    Lg = np.asarray(Lg, dtype=np.float64)
    u_ref = np.asarray(u_ref, dtype=np.float64)
    n_agents = u_ref.size // 2
    n_u = u_ref.size
    n_r = h.size
    n = n_u + n_r

    P = sp.diags(
        np.concatenate([np.ones(n_u), 10.0 * np.ones(n_r)]),
        format='csc')
    q = np.concatenate([-u_ref, relax_penalty * np.ones(n_r)])
    C = -np.concatenate([Lg, np.eye(n_r)], axis=1)
    b = Lf + float(alpha) * h

    # OSQP has no separate box bounds; stack an identity block.
    A = sp.vstack([
        sp.csc_matrix(C),
        sp.eye(n, format='csc'),
    ], format='csc')
    lower = np.concatenate([
        np.full(n_r, -np.inf),
        np.concatenate([np.full(n_u, -1.0), np.zeros(n_r)]),
    ])
    upper = np.concatenate([
        b,
        np.concatenate([np.full(n_u, 1.0), np.full(n_r, 1e9)]),
    ])
    solver = osqp.OSQP()
    solver.setup(P=P, q=q, A=A, l=lower, u=upper,
                 verbose=False, eps_abs=float(eps_abs),
                 eps_rel=float(eps_rel), max_iter=10000,
                 polish=True, warm_starting=True)
    result = solver.solve()
    status = str(result.info.status)
    if result.x is None or 'solved' not in status:
        return u_ref.reshape(n_agents, 2), np.zeros(n_r), status
    x = np.asarray(result.x)
    return x[:n_u].reshape(n_agents, 2), x[n_u:], status


def evaluate(env, matrices, seeds, alpha=1.0, reward_scale=20.0,
             reach_bonus=5.0):
    """Deterministic centralized CBF-QP evaluation on fixed scenes."""
    rows = []
    for seed in seeds:
        env.reset(seed=int(seed))
        total_return = np.zeros(env.num_agents, dtype=np.float64)
        safe_mask_values = []
        collision_values = []
        warning_values = []
        h_values = []
        action_abs = []
        reached_any = np.zeros(env.num_agents, dtype=bool)
        reached_final = np.zeros(env.num_agents, dtype=bool)
        collision_any = np.zeros(env.num_agents, dtype=bool)
        collision_final = np.zeros(env.num_agents, dtype=bool)
        warning_any = np.zeros(env.num_agents, dtype=bool)
        failures = 0
        for _ in range(env.max_step):
            graph = env.get_graph()
            h, Lf, Lg, u_ref = matrices(graph)
            action, _, status = solve_centralized_qp(
                h, Lf, Lg, u_ref, alpha=alpha)
            if 'solved' not in status:
                failures += 1
            action_abs.append(float(np.abs(action).mean()))
            h_values.append(float(np.asarray(h).mean()))
            _, reward, terminated, truncated, info = env.step(action)
            reached = np.asarray(info['reached'], dtype=bool)
            collision_step = np.asarray(info['collision'], dtype=bool)
            warning_step = np.asarray(info['unsafe'], dtype=bool)
            reached_any |= reached
            reached_final = reached
            collision_any |= collision_step
            collision_final = collision_step
            warning_any |= warning_step
            total_return += (reward * float(reward_scale)
                             + float(reach_bonus) * reached)
            safe_mask_values.append(float(info['safety']))
            collision_values.append(float(collision_step.mean()))
            warning_values.append(float(warning_step.mean()))
            if terminated or truncated:
                break
        collision_rate = float(np.mean(collision_values))
        collision_episode = bool(collision_any.any())
        rows.append(dict(
            episode_return=float(total_return.mean()),
            safety=1.0 - collision_rate,
            unsafe=collision_rate,
            safe_mask=float(np.mean(safe_mask_values)),
            warning=float(np.mean(warning_values)),
            collision_episode=float(collision_episode),
            safe_episode=float(not collision_episode),
            warning_episode=float(warning_any.any()),
            collision_final_any=float(collision_final.any()),
            reach=float(reached_any.mean()),
            reach_episode_any=float(reached_any.any()),
            reach_final=float(reached_final.mean()),
            reach_final_any=float(reached_final.any()),
            official_safety=float((~collision_any).mean()),
            official_reach=float(reached_any.mean()),
            official_success=float(((~collision_any) & reached_any).mean()),
            h_mean=float(np.mean(h_values)),
            h_min=float(np.min(h_values)),
            action_abs=float(np.mean(action_abs)),
            qp_failures=float(failures),
        ))
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


__all__ = ['build_qp_matrices', 'evaluate', 'solve_centralized_qp']
