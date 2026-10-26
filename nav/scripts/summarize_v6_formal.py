#!/usr/bin/env python3
"""Aggregate matched-seed V6 evaluations (mean and sample std).

This script is intentionally read-only.  It expects one evaluation JSON per
seed, produced by ``evaluate_v6.py`` with the same episode count and evaluation
seed.  The primary V6 metrics are episode-level ``safe_episode`` and
``collision_episode``; ``official_*`` fields are retained as legacy JSON names
for per-agent any-time metrics and are not episode-level safety.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

METRICS = (
    'official_reach',
    'official_success',
    'official_safety',
    'safe_episode',
    'collision_episode',
    'reach_final',
    'episode_return',
    'h_mean',
    'h_min',
    'action_abs',
    'cbf_true_sat',
    'cbf_violation',
)


def _mean_std(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {'mean': None, 'std': None, 'n': 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        'mean': float(arr.mean()),
        'std': float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        'n': int(len(arr)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--pattern', default='seed*/eval_*ep_final.json')
    args = ap.parse_args()

    root = Path(args.dir)
    paths = sorted(root.glob(args.pattern))
    if not paths:
        raise SystemExit(f'no evaluation files matched {root / args.pattern}')

    runs = []
    for path in paths:
        payload = json.loads(path.read_text())
        metrics = payload['metrics']
        runs.append(dict(
            seed=payload.get('training_seed'),
            eval_seed=payload.get('eval_seed'),
            episodes=payload.get('episodes'),
            steps=payload.get('steps'),
            checkpoint=str(payload.get('checkpoint')),
            h_checkpoint=str(payload.get('h_checkpoint')),
            metrics={key: float(metrics[key]) for key in METRICS
                     if key in metrics},
            path=str(path),
        ))

    eval_seeds = {row['eval_seed'] for row in runs}
    episodes = {row['episodes'] for row in runs}
    training_seeds = [row['seed'] for row in runs]
    if len(eval_seeds) != 1:
        raise SystemExit(f'evaluation seeds are not matched: {eval_seeds}')
    if len(episodes) != 1:
        raise SystemExit(f'episode counts are not matched: {episodes}')
    if len(set(training_seeds)) != len(training_seeds):
        raise SystemExit(f'duplicate training seeds: {training_seeds}')

    aggregate = {
        key: _mean_std([row['metrics'][key] for row in runs
                        if key in row['metrics']])
        for key in METRICS
    }
    payload = dict(
        schema='v6-formal-aggregate.v1',
        root=str(root),
        runs=runs,
        aggregate=aggregate,
        primary_metrics=['safe_episode', 'collision_episode'],
        note=('safe_episode/collision_episode are episode-level; '
              'official_safety/official_success are per-agent any-time legacy '
              'field names and are not episode-level safety.'),
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + '\n')

    lines = [
        '| seed | steps | reach | success | safe_ep | collision_ep | '
        'official_safety |',
        '|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in runs:
        m = row['metrics']
        lines.append(
            f"| {row['seed']} | {row['steps']} | {m['official_reach']:.4f} | "
            f"{m['official_success']:.4f} | {m['safe_episode']:.3f} | "
            f"{m['collision_episode']:.3f} | {m['official_safety']:.4f} |")
    lines += [
        '',
        '| metric | mean | std | n |',
        '|---|---:|---:|---:|',
    ]
    for key in METRICS:
        agg = aggregate[key]
        if agg['n']:
            lines.append(
                f"| {key} | {agg['mean']:.4f} | {agg['std']:.4f} | "
                f"{agg['n']} |")
    markdown = '\n'.join(lines) + '\n'
    (out.with_suffix('.md')).write_text(markdown)
    print(markdown)
    print(f'saved: {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
