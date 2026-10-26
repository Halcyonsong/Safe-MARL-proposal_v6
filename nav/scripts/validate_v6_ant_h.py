"""Independent rollout validation gate for the V6 Ant safety h.

The training summaries mix rollout rows with geometry augmentation.  Before
starting MAPPO, this script checks the frozen h on *fresh* V6 Ant rollouts that
were never used to fit h, under several action regimes.  It reports both

* instantaneous official-cost classification, and
* the forward-window safe/unsafe/ignored labels used to train h,

plus the state-gradient direction quality and the h value at episode resets.
No task return, V5 artifact, expert action, QP target, or MuJoCo task policy is
loaded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

NAV = Path(__file__).resolve().parents[1]
ROOT = NAV.parent
if str(NAV) not in sys.path:
    sys.path.insert(0, str(NAV))

from v6.ant_env import Ant2x4Env  # noqa: E402
from v6.ant_h import AntSafetyProvider  # noqa: E402
from v6.ant_safety import (  # noqa: E402
    GEOM_DIM,
    forward_safety_labels,
    safety_target,
    safety_target_grad,
)


class _ZeroProvider:
    """Cheap placeholder; validation reads geometry and official cost only."""

    def value_and_grad(self, geom):
        return 0.0, np.zeros(GEOM_DIM, dtype=np.float32)


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _roc_auc(scores: np.ndarray, positive: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    positive = np.asarray(positive, dtype=bool).reshape(-1)
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(scores, kind='mergesort')
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    pos_rank_sum = ranks[positive].sum()
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) /
                 (n_pos * n_neg))


def _binary_metrics(
    h: np.ndarray,
    unsafe: np.ndarray,
    *,
    threshold: float = 0.0,
) -> dict[str, float]:
    h = np.asarray(h, dtype=np.float64).reshape(-1)
    unsafe = np.asarray(unsafe, dtype=bool).reshape(-1)
    if len(h) != len(unsafe):
        raise ValueError('h/unsafe length mismatch')
    pred = h < float(threshold)
    n_pos = int(unsafe.sum())
    n_neg = int((~unsafe).sum())
    tp = int(np.sum(pred & unsafe))
    fn = int(np.sum(~pred & unsafe))
    fp = int(np.sum(pred & ~unsafe))
    tn = int(np.sum(~pred & ~unsafe))
    unsafe_recall = tp / n_pos if n_pos else 0.0
    safe_recall = tn / n_neg if n_neg else 0.0
    return {
        'rows': int(len(h)),
        'unsafe_rows': n_pos,
        'safe_rows': n_neg,
        'unsafe_rate': float(unsafe.mean()) if len(unsafe) else 0.0,
        'accuracy': float((tp + tn) / len(h)) if len(h) else 0.0,
        'balanced_accuracy': 0.5 * (unsafe_recall + safe_recall),
        'unsafe_recall': float(unsafe_recall),
        'safe_recall': float(safe_recall),
        'false_alarm': float(fp / n_neg) if n_neg else 0.0,
        'miss': float(fn / n_pos) if n_pos else 0.0,
        'auroc': _roc_auc(-h, unsafe),
        'h_mean': float(h.mean()) if len(h) else 0.0,
        'h_std': float(h.std()) if len(h) else 0.0,
        'h_neg_rate': float(pred.mean()) if len(pred) else 0.0,
    }


def _gradient_metrics(
    model,
    geom: np.ndarray,
    *,
    h_scale: float,
    sample: int,
    device: torch.device,
) -> dict[str, float]:
    geom = np.asarray(geom, dtype=np.float32)
    if len(geom) == 0:
        return {'rows': 0, 'grad_valid': 0, 'grad_cosine': 0.0,
                'grad_cosine_min': 0.0, 'grad_mse': 0.0}
    if sample > 0 and len(geom) > int(sample):
        idx = np.linspace(0, len(geom) - 1, int(sample), dtype=np.int64)
        geom = geom[idx]
    x = torch.as_tensor(geom, dtype=torch.float32, device=device)
    x.requires_grad_(True)
    model.eval()
    with torch.enable_grad():
        h = model(x)
        grad = torch.autograd.grad(h.sum(), x, create_graph=False)[0]
    grad = grad.detach().cpu().numpy().astype(np.float32)
    target = np.stack(
        [safety_target_grad(g, scale=h_scale) for g in geom], axis=0)
    dot = np.sum(grad * target, axis=1)
    gn = np.linalg.norm(grad, axis=1)
    tn = np.linalg.norm(target, axis=1)
    valid = (gn > 1e-8) & (tn > 1e-8)
    cosine = dot[valid] / (gn[valid] * tn[valid])
    return {
        'rows': int(len(geom)),
        'grad_valid': int(valid.sum()),
        'grad_cosine': float(np.mean(cosine)) if len(cosine) else 0.0,
        'grad_cosine_min': float(np.min(cosine)) if len(cosine) else 0.0,
        'grad_mse': float(np.mean(np.sum((grad - target) ** 2, axis=1))),
    }


def _collect_regime(
    name: str,
    action_std: float,
    *,
    seed: int,
    episodes: int,
    episode_limit: int,
    horizon: int,
) -> dict:
    rng = np.random.default_rng(int(seed))
    env = Ant2x4Env(
        seed=int(seed), episode_limit=int(episode_limit),
        safety_provider=_ZeroProvider())
    geoms: list[np.ndarray] = []
    unsafe: list[bool] = []
    reset_rows: list[bool] = []
    episode_rows: list[int] = []
    episode_unsafe: list[int] = []
    episode_wall: list[int] = []
    episode_topple: list[int] = []
    episode_lengths: list[int] = []
    episode_truncated: list[int] = []
    try:
        for ep in range(int(episodes)):
            _obs, _h, _grad, _hvec, _hs, geom, _state = env.reset(
                seed=int(seed) + ep)
            geoms.append(np.asarray(geom, dtype=np.float32).copy())
            unsafe.append(False)
            reset_rows.append(True)
            ep_start = len(geoms) - 1
            ep_len = 0
            ep_unsafe = 0
            ep_wall = 0
            ep_topple = 0
            truncated = False
            for _ in range(int(episode_limit)):
                if action_std <= 0.0:
                    actions = np.zeros(
                        (env.num_agents, env.action_dim), dtype=np.float32)
                else:
                    actions = rng.normal(
                        0.0, float(action_std),
                        size=(env.num_agents, env.action_dim)).astype(
                            np.float32)
                    actions = np.clip(actions, -1.0, 1.0)
                (_obs, _h, _grad, _hvec, _hs, next_geom, _state,
                 _reward, cost, terminated, truncated, info) = env.step(actions)
                geoms.append(np.asarray(next_geom, dtype=np.float32).copy())
                is_unsafe = bool(cost > 0.0 or terminated)
                unsafe.append(is_unsafe)
                reset_rows.append(False)
                ep_len += 1
                ep_unsafe += int(is_unsafe)
                ep_wall += int(float(info.get('wall', 0.0)) > 0.0)
                ep_topple += int(float(info.get('topple', 0.0)) > 0.0)
                if terminated or truncated:
                    break
            episode_rows.append(len(geoms) - ep_start)
            episode_unsafe.append(ep_unsafe)
            episode_wall.append(ep_wall)
            episode_topple.append(ep_topple)
            episode_lengths.append(ep_len)
            episode_truncated.append(int(truncated))
    finally:
        env.close()

    geom_arr = np.asarray(geoms, dtype=np.float32)
    unsafe_arr = np.asarray(unsafe, dtype=bool)
    episode = np.repeat(np.arange(int(episodes)), episode_rows).astype(np.int64)
    # `forward_safety_labels` is defined per episode; apply it episode-wise.
    forward = np.zeros(len(geom_arr), dtype=np.float32)
    cursor = 0
    for ep in range(int(episodes)):
        n = int(episode_rows[ep])
        sl = slice(cursor, cursor + n)
        forward[sl] = forward_safety_labels(
            unsafe_arr[sl], horizon=int(horizon))
        cursor += n
    return {
        'name': name,
        'action_std': float(action_std),
        'seed': int(seed),
        'episodes': int(episodes),
        'rows': int(len(geom_arr)),
        'geom': geom_arr,
        'unsafe': unsafe_arr,
        'forward_labels': forward,
        'reset_rows': np.asarray(reset_rows, dtype=bool),
        'episode': episode,
        'episode_stats': {
            'length_mean': float(np.mean(episode_lengths)),
            'length_min': int(np.min(episode_lengths)),
            'length_max': int(np.max(episode_lengths)),
            'unsafe_episodes': int(sum(ep_unsafe > 0
                                      for ep_unsafe in episode_unsafe)),
            'wall_episodes': int(sum(ep_wall > 0
                                    for ep_wall in episode_wall)),
            'topple_episodes': int(sum(ep_topple > 0
                                      for ep_topple in episode_topple)),
            'truncated_episodes': int(sum(episode_truncated)),
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        '--h-checkpoint',
        default='outputs/v6/independent/ant/h/ant_h.pkl')
    ap.add_argument(
        '--out', default='outputs/v6/independent/ant/h_validation')
    ap.add_argument('--seed', type=int, default=30000)
    ap.add_argument('--episodes', type=int, default=50)
    ap.add_argument('--episode-limit', type=int, default=1000)
    ap.add_argument('--horizon', type=int, default=32)
    ap.add_argument('--grad-sample', type=int, default=4096)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    out = _resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    provider = AntSafetyProvider.from_checkpoint(
        _resolve(args.h_checkpoint), device=args.device)
    h_scale = float(provider.h_scale)
    h_offset = float(provider.h_offset)

    regimes = [
        ('zero', 0.0),
        ('random_0.1', 0.1),
        ('random_0.7', 0.7),
        ('random_1.0', 1.0),
    ]
    collected = {}
    results = {}
    for i, (name, std) in enumerate(regimes):
        data = _collect_regime(
            name, std,
            seed=int(args.seed) + i * 10000,
            episodes=int(args.episodes),
            episode_limit=int(args.episode_limit),
            horizon=int(args.horizon),
        )
        collected[name] = data
        h = provider.batch_value(data['geom'])
        data['h'] = h
        instant = _binary_metrics(h, data['unsafe'], threshold=0.0)
        fwd_mask = data['forward_labels'] != 0.0
        forward = _binary_metrics(
            h[fwd_mask], data['forward_labels'][fwd_mask] < 0.0,
            threshold=0.0)
        reset = _binary_metrics(
            h[data['reset_rows']],
            np.zeros(int(data['reset_rows'].sum()), dtype=bool),
            threshold=0.0)
        grad = _gradient_metrics(
            provider.model, data['geom'], h_scale=h_scale,
            sample=int(args.grad_sample), device=provider.device)
        target = np.asarray(
            [safety_target(g, scale=h_scale) for g in data['geom']],
            dtype=np.float32)
        results[name] = {
            'action_std': data['action_std'],
            'seed': data['seed'],
            'episodes': data['episodes'],
            'rows': data['rows'],
            'episode_stats': data['episode_stats'],
            'instant': instant,
            'forward': forward,
            'reset': {
                'rows': int(data['reset_rows'].sum()),
                'h_mean': reset['h_mean'],
                'h_std': reset['h_std'],
                'h_neg_rate': reset['h_neg_rate'],
            },
            'target_mse': float(np.mean((h - target) ** 2)),
            'gradient': grad,
        }
        print(
            f"{name:12s} rows={instant['rows']:5d} "
            f"unsafe={instant['unsafe_rows']:4d} "
            f"BA={instant['balanced_accuracy']:.3f} "
            f"recall={instant['unsafe_recall']:.3f} "
            f"FA={instant['false_alarm']:.3f} "
            f"fwd_BA={forward['balanced_accuracy']:.3f} "
            f"grad_cos={grad['grad_cosine']:.3f}")

    pooled_h = np.concatenate([collected[name]['h'] for name, _ in regimes])
    pooled_unsafe = np.concatenate(
        [collected[name]['unsafe'] for name, _ in regimes])
    pooled_forward_h = np.concatenate([
        collected[name]['h'][collected[name]['forward_labels'] != 0.0]
        for name, _ in regimes
    ])
    pooled_forward_unsafe = np.concatenate([
        collected[name]['forward_labels'][
            collected[name]['forward_labels'] != 0.0] < 0.0
        for name, _ in regimes
    ])
    pooled = _binary_metrics(pooled_h, pooled_unsafe, threshold=0.0)
    pooled_forward = _binary_metrics(
        pooled_forward_h, pooled_forward_unsafe, threshold=0.0)
    pooled_grad = _gradient_metrics(
        provider.model,
        np.concatenate([collected[name]['geom'] for name, _ in regimes]),
        h_scale=h_scale, sample=int(args.grad_sample),
        device=provider.device)

    regime_gate = {}
    for name, value in results.items():
        inst = value['instant']
        enough_unsafe = bool(inst['unsafe_rows'] >= 10)
        ba_ok = bool(inst['balanced_accuracy'] >= 0.85)
        recall_ok = bool(inst['unsafe_recall'] >= 0.85)
        false_alarm_ok = bool(inst['false_alarm'] <= 0.10)
        passed = bool(
            (not enough_unsafe) or
            (ba_ok and recall_ok and false_alarm_ok))
        regime_gate[name] = {
            'enough_unsafe': enough_unsafe,
            'ba_ok': ba_ok,
            'recall_ok': recall_ok,
            'false_alarm_ok': false_alarm_ok,
            'passed': passed,
        }

    reset_neg_rate = max(
        (v['reset']['h_neg_rate'] for v in results.values()), default=0.0)
    gate = {
        'overall_instant_ba_ge_0.95': bool(
            pooled['balanced_accuracy'] >= 0.95),
        'overall_instant_recall_ge_0.90': bool(
            pooled['unsafe_recall'] >= 0.90),
        'overall_instant_false_alarm_le_0.05': bool(
            pooled['false_alarm'] <= 0.05),
        'forward_ba_ge_0.90': bool(
            pooled_forward['balanced_accuracy'] >= 0.90),
        'forward_recall_ge_0.90': bool(
            pooled_forward['unsafe_recall'] >= 0.90),
        'grad_cosine_ge_0.80': bool(
            pooled_grad['grad_cosine'] >= 0.80),
        'reset_false_alarm_le_0.05': bool(reset_neg_rate <= 0.05),
        'each_unsafe_regime_passed': bool(
            all(v['passed'] for v in regime_gate.values())),
    }
    gate['passed'] = bool(all(gate.values()))

    summary = {
        'schema': 'v6.ant_h.validation.v1',
        'checkpoint': str(_resolve(args.h_checkpoint)),
        'h_offset': h_offset,
        'h_scale': h_scale,
        'seed': int(args.seed),
        'episodes_per_regime': int(args.episodes),
        'episode_limit': int(args.episode_limit),
        'horizon': int(args.horizon),
        'regimes': results,
        'regime_gate': regime_gate,
        'pooled_instant': pooled,
        'pooled_forward': pooled_forward,
        'pooled_gradient': pooled_grad,
        'reset_h_neg_rate_max': float(reset_neg_rate),
        'gate': gate,
    }
    (out / 'validation.json').write_text(
        json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: summary[k] for k in (
        'pooled_instant', 'pooled_forward', 'pooled_gradient',
        'reset_h_neg_rate_max', 'gate')}, indent=2))
    print(f'wrote {out / "validation.json"}')
    if not gate['passed']:
        print('Ant h validation gate FAILED', file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
