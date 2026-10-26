"""Generate near-boundary safety samples for learned h.

Random-policy rollouts spend most time in open space, so unsafe samples are
rare.  This script builds hard states near obstacle edges and between agents,
then labels them with the environment's unsafe/collision masks.  It does not
use task reward, QP labels, official h, or an expert policy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import yaml

NAV = Path(__file__).resolve().parents[1]
ROOT = NAV.parent
if str(NAV) not in sys.path:
    sys.path.insert(0, str(NAV))

from envs.gcbfplus import OfficialDoubleIntegrator  # noqa: E402
from v6.graph_data import add_graph_to_row, stack_graph_rows  # noqa: E402
from v6.provenance import assert_independent_path  # noqa: E402


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _edge_point_and_normal(points: np.ndarray, rng: np.random.Generator):
    # points: [4,2] in edge order; pick one edge.
    k = int(rng.integers(0, 4))
    p0 = points[k]
    p1 = points[(k + 1) % 4]
    t = float(rng.uniform(0.0, 1.0))
    point = (1.0 - t) * p0 + t * p1
    tangent = p1 - p0
    tangent = tangent / (np.linalg.norm(tangent) + 1e-9)
    normal = np.array([-tangent[1], tangent[0]])
    center = points.mean(axis=0)
    if np.dot(normal, point - center) < 0:
        normal = -normal
    return point, normal


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='configs/v6/v6_nav_independent.yaml')
    ap.add_argument('--data-dir', default='outputs/v6/independent/data')
    ap.add_argument('--out', default='outputs/v6/independent/data/train_aug.npz')
    ap.add_argument('--scenes', type=int, default=8000)
    ap.add_argument('--seed', type=int, default=777)
    ap.add_argument('--reset-every', type=int, default=250)
    ap.add_argument('--obstacle-fraction', type=float, default=0.7)
    ap.add_argument('--min-offset', type=float, default=-0.08)
    ap.add_argument('--max-offset', type=float, default=0.18)
    args = ap.parse_args()

    assert_independent_path(
        _resolve(args.data_dir), kind='V6 h data directory')
    out_path = assert_independent_path(
        _resolve(args.out), kind='V6 h augmented data output')
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = NAV / cfg_path
    config = yaml.safe_load(cfg_path.read_text())
    env_cfg = config['env']
    env = OfficialDoubleIntegrator(
        num_agents=env_cfg['num_agents'],
        area_size=env_cfg['area_size'],
        max_step=env_cfg['max_step'],
        num_obstacles=env_cfg['num_obstacles'],
        n_rays=env_cfg['n_rays'],
        max_neighbors=env_cfg['max_neighbors'],
        seed=args.seed,
    )
    rng = np.random.default_rng(args.seed)
    n = env.num_agents
    half = 0.5 * env.area_size - 0.15
    rows = []
    n_unsafe = 0

    for scene in range(int(args.scenes)):
        if scene % int(args.reset_every) == 0:
            env.reset(seed=args.seed + scene // int(args.reset_every))
        goals = env.get_goal_states()
        obstacles = env.graph.env_states.obstacle
        points = np.asarray(obstacles.points, dtype=np.float32)
        states = np.zeros((n, 4), dtype=np.float32)
        states[:, :2] = rng.uniform(-half, half, (n, 2))
        states[:, 2:] = rng.normal(0.0, 0.25, (n, 2))

        if rng.random() < float(args.obstacle_fraction):
            i = int(rng.integers(0, n))
            k = int(rng.integers(0, len(points)))
            point, normal = _edge_point_and_normal(points[k], rng)
            offset = rng.uniform(args.min_offset, args.max_offset)
            states[i, :2] = point + offset * normal
            # Point the velocity toward the obstacle; this activates the
            # environment heading-dependent warning zone, not just collision.
            states[i, 2:] = -normal * rng.uniform(0.1, 0.6)
        else:
            # Put two agents close enough and head-on to produce hard states.
            i, j = rng.choice(n, size=2, replace=False)
            direction = rng.normal(size=2)
            direction = direction / (np.linalg.norm(direction) + 1e-9)
            dist = rng.uniform(0.02, 0.15)
            states[j, :2] = states[i, :2] + direction * dist
            speed = rng.uniform(0.1, 0.6)
            states[i, 2:] = direction * speed
            states[j, 2:] = -direction * speed

        states[:, :2] = np.clip(states[:, :2], -half, half)
        state = env._env.EnvState(
            jnp.asarray(states), env.graph.type_states(
                type_idx=1, n_type=env.num_agents),
            env.graph.env_states.obstacle)
        graph = env._env.get_graph(state)
        env.graph = graph
        unsafe = np.asarray(env._env.unsafe_mask(graph), dtype=bool)
        collision = np.asarray(env._env.collision_mask(graph), dtype=bool)
        # Augmented states are single snapshots: they have no future rollout.
        # Use them only as instantaneous negatives; non-unsafe snapshots are
        # ignored by the horizon label instead of being mislabeled positive.
        labels_instant = np.where(
            unsafe | collision, -1.0, 1.0).astype(np.float32)
        labels_horizon = np.where(
            unsafe | collision, -1.0, 0.0).astype(np.float32)
        row = dict(
            obs=env.observation().astype(np.float32),
            labels=labels_horizon,
            labels_instant=labels_instant,
            labels_horizon=labels_horizon,
        )
        add_graph_to_row(row, graph)
        rows.append(row)
        n_unsafe += int(np.sum(labels_horizon < 0))
        if (scene + 1) % 500 == 0:
            print(f'scenes={scene + 1}/{args.scenes} '
                  f'unsafe_rate={n_unsafe / ((scene + 1) * n):.4f}', flush=True)

    obs = np.stack([r['obs'] for r in rows])
    labels_horizon = np.stack([r['labels_horizon'] for r in rows])
    labels_instant = np.stack([r['labels_instant'] for r in rows])
    out = out_path
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        obs=obs,
        labels=labels_horizon,
        graph_labels=labels_horizon,
        graph_labels_horizon=labels_horizon,
        graph_labels_instant=labels_instant,
        graph_hdot_mask=np.zeros(len(labels_horizon), dtype=np.float32),
    )
    payload.update(stack_graph_rows(rows))
    np.savez_compressed(out, **payload)
    summary = dict(
        scenes=int(args.scenes), rows=int(obs.shape[0] * obs.shape[1]),
        unsafe_rate=float(np.mean(labels_horizon < 0)),
        ignored_rate=float(np.mean(labels_horizon == 0)),
        out=str(out),
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
