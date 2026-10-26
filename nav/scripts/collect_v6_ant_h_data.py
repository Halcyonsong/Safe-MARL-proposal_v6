"""Collect V6-owned Ant safety data for the 9-D h(s).

The collector uses only the official Ant simulator/reward/cost.  It stores
torso geometry, forward-window safety labels, a dense margin target, and the
target state-space gradient.  No task return, expert action, V5 checkpoint, or
official h artifact is read.

Usage (under the mappo_lag environment):
    LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin \
      python scripts/collect_v6_ant_h_data.py \
        --out ../outputs/v6/independent/ant/h_data \
        --episodes 200 --aug-rows 20000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

NAV = Path(__file__).resolve().parents[1]
ROOT = NAV.parent
if str(NAV) not in sys.path:
    sys.path.insert(0, str(NAV))

from v6.ant_env import Ant2x4Env  # noqa: E402
from v6.ant_safety import (  # noqa: E402
    FixedAntSafetyProvider,
    NullAntSafetyProvider,
    forward_safety_labels,
    geom_from_torso,
    horizon_risk_targets,
    official_cost_from_geom,
    safety_target,
    safety_target_grad,
)
from v6.ant_h import AntSafetyDataset  # noqa: E402


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _split_episodes(episode_ids: list[int], rng: np.random.Generator,
                    val_fraction: float, test_fraction: float):
    ids = np.asarray(sorted(episode_ids), dtype=np.int64)
    rng.shuffle(ids)
    n = len(ids)
    n_test = int(round(n * test_fraction))
    n_val = int(round(n * val_fraction))
    test = set(int(v) for v in ids[:n_test])
    val = set(int(v) for v in ids[n_test:n_test + n_val])
    train = set(int(v) for v in ids[n_test + n_val:])
    return train, val, test


def _augment_geom(rng: np.random.Generator, n: int) -> np.ndarray:
    """Sample geometry around the official walls/height/upright thresholds."""
    out = []
    for _ in range(int(n)):
        x = float(rng.uniform(0.0, 100.0))
        # Bias half of the samples toward a wall, and the rest broadly.
        if rng.random() < 0.5:
            y = float(rng.uniform(-8.0, 8.0))
        else:
            y = float(rng.uniform(-20.0, 20.0))
        z = float(rng.uniform(0.0, 1.2))
        z_rot = float(rng.uniform(-1.0, 1.0))
        vx = float(rng.uniform(-2.0, 2.0))
        vy = float(rng.uniform(-2.0, 2.0))
        vz = float(rng.uniform(-2.0, 2.0))
        out.append(geom_from_torso(x, y, z, vx, vy, vz, z_rot))
    if not out:
        return np.zeros((0, 9), dtype=np.float32)
    return np.asarray(out, dtype=np.float32)


def _perturb_rollout(geoms: np.ndarray, rng: np.random.Generator,
                     n: int | None = None,
                     scales: tuple[float, ...] | None = None) -> np.ndarray:
    if len(geoms) == 0:
        return np.zeros((0, geoms.shape[-1]), dtype=np.float32)
    n = int(n or len(geoms))
    idx = rng.integers(0, len(geoms), size=n)
    base = geoms[idx].astype(np.float64)
    if scales is None:
        scales = (2.0, 2.0, 0.4, 0.4, 0.10, 0.10, 0.20, 0.20, 0.20)
    noise = rng.normal(size=base.shape) * np.asarray(scales, dtype=np.float64)
    out = base + noise
    # Geometry entries 0/1 are distances; 4 is z.
    out[:, 0] = np.clip(out[:, 0], 0.0, 20.0)
    out[:, 1] = np.clip(out[:, 1], 0.0, 20.0)
    out[:, 4] = np.clip(out[:, 4], 0.0, 1.3)
    out[:, 5] = np.clip(out[:, 5], -1.0, 1.0)
    return out.astype(np.float32)


def _make_dataset(geom: np.ndarray, labels: np.ndarray,
                  episode: np.ndarray, source: np.ndarray,
                  h_scale: float, h_kappa: float = 0.0,
                  h_kappa_z: float = 0.0, target=None,
                  target_grad=None) -> AntSafetyDataset:
    if target is None:
        target = np.asarray(
            [safety_target(g, scale=h_scale, kappa=h_kappa,
                           kappa_z=h_kappa_z) for g in geom],
            dtype=np.float32)
    else:
        target = np.asarray(target, dtype=np.float32).reshape(-1)
    if target_grad is None:
        target_grad = np.stack(
            [safety_target_grad(g, scale=h_scale, kappa=h_kappa,
                                kappa_z=h_kappa_z) for g in geom], axis=0
        ) if len(geom) else np.zeros((0, geom.shape[-1]), dtype=np.float32)
    else:
        target_grad = np.asarray(target_grad, dtype=np.float32)
    return AntSafetyDataset(
        geom=geom, labels=labels, target=target, target_grad=target_grad,
        episode=episode, source=source, h_scale=h_scale,
        h_kappa=h_kappa, h_kappa_z=h_kappa_z)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default='../outputs/v6/independent/ant/h_data')
    ap.add_argument('--episodes', type=int, default=200)
    ap.add_argument('--episode-limit', type=int, default=1000)
    ap.add_argument('--horizon', type=int, default=32)
    ap.add_argument('--h-scale', type=float, default=0.5)
    ap.add_argument('--kappa', type=float, default=0.0)
    ap.add_argument('--kappa-z', type=float, default=0.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--action-mode',
                    choices=('random', 'zero', 'policy'),
                    default='random')
    ap.add_argument('--action-std', type=float, default=0.7)
    ap.add_argument('--policy-checkpoint', default=None,
                    help='V6 no-h policy used when --action-mode=policy')
    ap.add_argument('--policy-stochastic', action='store_true',
                    help='sample from the policy instead of deterministic acts')
    ap.add_argument('--target-mode', choices=('current', 'horizon'),
                    default='current',
                    help='current margin or future horizon risk target')
    ap.add_argument('--aug-rows', type=int, default=20000)
    ap.add_argument('--perturb-rows', type=int, default=0,
                    help='extra geometry perturbations of rollout rows')
    ap.add_argument('--val-fraction', type=float, default=0.15)
    ap.add_argument('--test-fraction', type=float, default=0.15)
    args = ap.parse_args()

    out = _resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))
    policy = None
    if args.action_mode == 'policy':
        if not args.policy_checkpoint:
            raise ValueError('--action-mode=policy requires --policy-checkpoint')
        from v6.ant_mappo import load_ant_mappo_model
        policy, _ = load_ant_mappo_model(
            _resolve(args.policy_checkpoint), device='cpu')
        if policy.use_h_features:
            raise ValueError(
                'predictive-data collector expects a no-h cost-only policy; '
                'the current policy consumes h features')
        policy.eval()
        provider = NullAntSafetyProvider()
    else:
        provider = FixedAntSafetyProvider(
            scale=args.h_scale, kappa=args.kappa, kappa_z=args.kappa_z)
    env = Ant2x4Env(
        seed=int(args.seed), episode_limit=int(args.episode_limit),
        safety_provider=provider, h_kappa=float(args.kappa),
        h_kappa_z=float(args.kappa_z))

    all_geom: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_target: list[np.ndarray] = []
    all_target_grad: list[np.ndarray] = []
    all_episode: list[np.ndarray] = []
    all_source: list[np.ndarray] = []
    episode_stats: list[dict] = []

    for ep in range(int(args.episodes)):
        obs, h, grad, _hvec, _hs, _geom, _state = env.reset(
            seed=int(args.seed) + ep)
        geoms = [env.get_geometry().copy()]
        unsafe = [False]
        ep_cost = 0.0
        ep_wall = 0.0
        ep_topple = 0.0
        ep_len = 0
        truncated = False
        terminated = False
        for _ in range(int(args.episode_limit)):
            if args.action_mode == 'zero':
                actions = np.zeros(
                    (env.num_agents, env.action_dim), dtype=np.float32)
            elif args.action_mode == 'policy':
                actions, _ = policy.act_single(
                    obs, h, grad,
                    deterministic=not bool(args.policy_stochastic),
                    device='cpu')
                actions = np.asarray(actions, dtype=np.float32)
            else:
                actions = rng.normal(
                    0.0, float(args.action_std),
                    size=(env.num_agents, env.action_dim)).astype(np.float32)
                actions = np.clip(actions, -1.0, 1.0)
            (obs, h, grad, _hvec, _hs, _geom, _state,
             reward, cost, terminated, truncated, info) = env.step(actions)
            ep_cost = max(ep_cost, float(cost))
            ep_wall = max(ep_wall, float(info.get('wall', 0.0)))
            ep_topple = max(ep_topple, float(info.get('topple', 0.0)))
            ep_len += 1
            geoms.append(env.get_geometry().copy())
            unsafe.append(bool(cost > 0.0 or terminated))
            if terminated or truncated:
                break

        geoms_arr = np.asarray(geoms, dtype=np.float32)
        unsafe_arr = np.asarray(unsafe, dtype=bool)
        labels = forward_safety_labels(unsafe_arr, horizon=int(args.horizon))
        if args.target_mode == 'horizon':
            # Keep every row, including the old ``0`` rows: they are the
            # near-future-unsafe states that were discarded before.  A
            # negative label means an unsafe event is inside the window; the
            # target is the minimum future margin (negative before the event).
            labels = labels.copy()
            labels[labels == 0.0] = -1.0
            target, _risk = horizon_risk_targets(
                geoms_arr, scale=float(args.h_scale),
                horizon=int(args.horizon), kappa=float(args.kappa),
                kappa_z=float(args.kappa_z))
            # Terminal/cost events that do not show a negative geometry margin
            # still need a negative target; otherwise labels and h conflict.
            target = np.where(
                labels < 0.0, np.minimum(target, -0.02), target
            ).astype(np.float32)
            target_grad = np.stack([safety_target_grad(
                g, scale=float(args.h_scale), kappa=float(args.kappa),
                kappa_z=float(args.kappa_z)) for g in geoms_arr], axis=0)
            keep = np.ones(len(geoms_arr), dtype=bool)
        else:
            keep = labels != 0.0
            target = np.asarray([safety_target(
                g, scale=float(args.h_scale), kappa=float(args.kappa),
                kappa_z=float(args.kappa_z)) for g in geoms_arr],
                dtype=np.float32)
            target_grad = np.stack([safety_target_grad(
                g, scale=float(args.h_scale), kappa=float(args.kappa),
                kappa_z=float(args.kappa_z)) for g in geoms_arr], axis=0)
        all_geom.append(geoms_arr[keep])
        all_labels.append(labels[keep])
        all_target.append(target[keep])
        all_target_grad.append(target_grad[keep])
        all_episode.append(np.full(int(keep.sum()), ep, dtype=np.int64))
        all_source.append(np.zeros(int(keep.sum()), dtype=np.int8))
        episode_stats.append(dict(
            episode=int(ep), length=int(ep_len), cost=float(ep_cost),
            wall=float(ep_wall), topple=float(ep_topple),
            terminated=bool(terminated), truncated=bool(truncated),
            rows=int(keep.sum()), unsafe_rows=int((labels < 0).sum()),
            safe_rows=int((labels > 0).sum()),
            ignored_rows=int((labels == 0).sum()),
        ))

    rollout_geom = (np.concatenate(all_geom, axis=0)
                    if all_geom else np.zeros((0, 9), dtype=np.float32))
    rollout_labels = (np.concatenate(all_labels, axis=0)
                      if all_labels else np.zeros(0, dtype=np.float32))
    rollout_target = (np.concatenate(all_target, axis=0)
                      if all_target else np.zeros(0, dtype=np.float32))
    rollout_target_grad = (np.concatenate(all_target_grad, axis=0)
                           if all_target_grad else np.zeros(
                               (0, 9), dtype=np.float32))
    rollout_episode = (np.concatenate(all_episode, axis=0)
                       if all_episode else np.zeros(0, dtype=np.int64))
    rollout_source = (np.concatenate(all_source, axis=0)
                      if all_source else np.zeros(0, dtype=np.int8))

    aug_geom = _augment_geom(rng, int(args.aug_rows))
    if len(aug_geom):
        aug_labels = np.asarray(
            [-1.0 if official_cost_from_geom(g) else 1.0 for g in aug_geom],
            dtype=np.float32)
    else:
        aug_labels = np.zeros(0, dtype=np.float32)
    aug_episode = np.full(len(aug_geom), -1, dtype=np.int64)
    aug_source = np.ones(len(aug_geom), dtype=np.int8)
    aug_target = np.asarray([safety_target(
        g, scale=float(args.h_scale), kappa=float(args.kappa),
        kappa_z=float(args.kappa_z)) for g in aug_geom], dtype=np.float32)
    aug_target_grad = np.stack([safety_target_grad(
        g, scale=float(args.h_scale), kappa=float(args.kappa),
        kappa_z=float(args.kappa_z)) for g in aug_geom], axis=0) \
        if len(aug_geom) else np.zeros((0, 9), dtype=np.float32)

    if int(args.perturb_rows) > 0 and len(rollout_geom):
        pert_geom = _perturb_rollout(
            rollout_geom, rng, n=int(args.perturb_rows))
        pert_labels = np.asarray(
            [-1.0 if official_cost_from_geom(g) else 1.0 for g in pert_geom],
            dtype=np.float32)
        pert_episode = np.full(len(pert_geom), -2, dtype=np.int64)
        pert_source = np.full(len(pert_geom), 2, dtype=np.int8)
        pert_target = np.asarray([safety_target(
            g, scale=float(args.h_scale), kappa=float(args.kappa),
            kappa_z=float(args.kappa_z)) for g in pert_geom],
            dtype=np.float32)
        pert_target_grad = np.stack([safety_target_grad(
            g, scale=float(args.h_scale), kappa=float(args.kappa),
            kappa_z=float(args.kappa_z)) for g in pert_geom], axis=0)
    else:
        pert_geom = np.zeros((0, 9), dtype=np.float32)
        pert_labels = np.zeros(0, dtype=np.float32)
        pert_target = np.zeros(0, dtype=np.float32)
        pert_target_grad = np.zeros((0, 9), dtype=np.float32)
        pert_episode = np.zeros(0, dtype=np.int64)
        pert_source = np.zeros(0, dtype=np.int8)

    geom = np.concatenate([rollout_geom, aug_geom, pert_geom], axis=0)
    labels = np.concatenate([rollout_labels, aug_labels, pert_labels], axis=0)
    target = np.concatenate(
        [rollout_target, aug_target, pert_target], axis=0)
    target_grad = np.concatenate(
        [rollout_target_grad, aug_target_grad, pert_target_grad], axis=0)
    episode = np.concatenate(
        [rollout_episode, aug_episode, pert_episode], axis=0)
    source = np.concatenate(
        [rollout_source, aug_source, pert_source], axis=0)
    if len(geom) == 0:
        raise RuntimeError('no Ant h rows collected')

    rollout_ids = sorted(set(int(v) for v in rollout_episode if v >= 0))
    train_ids, val_ids, test_ids = _split_episodes(
        rollout_ids, rng, args.val_fraction, args.test_fraction)
    # Rollout rows: split by episode.  Augmentation/perturbation: random split.
    masks = {
        'train': np.zeros(len(geom), dtype=bool),
        'val': np.zeros(len(geom), dtype=bool),
        'test': np.zeros(len(geom), dtype=bool),
    }
    rollout_mask = source == 0
    for key, ids in (('train', train_ids), ('val', val_ids), ('test', test_ids)):
        if ids:
            masks[key] |= rollout_mask & np.isin(
                episode, np.asarray(sorted(ids)))
    aux = source != 0
    aux_idx = rng.permutation(np.flatnonzero(aux))
    n_test = int(round(len(aux_idx) * float(args.test_fraction)))
    n_val = int(round(len(aux_idx) * float(args.val_fraction)))
    masks['test'][aux_idx[:n_test]] = True
    masks['val'][aux_idx[n_test:n_test + n_val]] = True
    masks['train'][aux_idx[n_test + n_val:]] = True

    summary = dict(
        schema='v6.ant_h_data.v1',
        out=str(out),
        seed=int(args.seed),
        episodes=int(args.episodes),
        episode_limit=int(args.episode_limit),
        horizon=int(args.horizon),
        h_scale=float(args.h_scale),
        h_kappa=float(args.kappa),
        h_kappa_z=float(args.kappa_z),
        action_mode=args.action_mode,
        action_std=float(args.action_std),
        policy_checkpoint=args.policy_checkpoint,
        policy_stochastic=bool(args.policy_stochastic),
        target_mode=args.target_mode,
        aug_rows=int(args.aug_rows),
        perturb_rows=int(args.perturb_rows),
        rows=int(len(geom)),
        rollout_rows=int(len(rollout_geom)),
        aug_rows_used=int(len(aug_geom)),
        perturb_rows_used=int(len(pert_geom)),
        rollout_episodes=int(len(rollout_ids)),
        label_unsafe=int((labels < 0).sum()),
        label_safe=int((labels > 0).sum()),
        label_ignored=int((labels == 0).sum()),
        target_mean=float(np.mean(target)) if len(target) else 0.0,
        target_negative=int((target < 0).sum()),
        source_counts={str(k): int((source == k).sum()) for k in (0, 1, 2)},
        episode_stats=episode_stats,
        splits={},
    )

    for split in ('train', 'val', 'test'):
        ds = _make_dataset(
            geom[masks[split]], labels[masks[split]],
            episode[masks[split]], source[masks[split]], args.h_scale,
            args.kappa, args.kappa_z, target=target[masks[split]],
            target_grad=target_grad[masks[split]])
        ds.save(out / f'{split}_data.npz')
        summary['splits'][split] = dict(
            rows=len(ds),
            unsafe=int((ds.labels < 0).sum()),
            safe=int((ds.labels > 0).sum()),
            rollout=int((ds.source == 0).sum()) if ds.source is not None else 0,
            augmented=int((ds.source != 0).sum()) if ds.source is not None else 0,
        )
        print(f'{split}: rows={len(ds)} unsafe={int((ds.labels < 0).sum())} '
              f'safe={int((ds.labels > 0).sum())}')

    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    env.close()
    print(json.dumps({k: summary[k] for k in (
        'rows', 'rollout_rows', 'aug_rows_used', 'label_unsafe',
        'label_safe', 'splits')}, indent=2))


if __name__ == '__main__':
    main()
