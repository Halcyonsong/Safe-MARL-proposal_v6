#!/usr/bin/env python3
"""Summarize V6 safety-screen runs against the h-only baseline.

Reads the 20-episode screen JSONs (and optional 100-episode promoted files)
under ``outputs/v6/independent/safety_opt``.  Primary metrics are episode-level
``safe_episode``/``collision_episode``; ``official_*`` fields remain per-agent
any-time legacy JSON names.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

METRICS = (
    'official_reach',
    'official_success',
    'official_safety',
    'safe_episode',
    'collision_episode',
    'episode_return',
    'h_mean',
    'action_abs',
    'cbf_true_sat',
    'cbf_violation',
)


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default='outputs/v6/independent/safety_opt')
    ap.add_argument('--baseline', default=(
        'outputs/v6/independent/single_seed_16k/h_only/summary.json'))
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    root = Path(args.root)
    baseline_path = Path(args.baseline)
    baseline = None
    if baseline_path.exists():
        summary = json.loads(baseline_path.read_text())
        history = summary.get('eval_history') or []
        if history:
            row = history[-1]
            baseline = dict(
                name='h_only_final_20ep',
                steps=row.get('steps'),
                episodes=20,
                metrics={key: float(row[key]) for key in METRICS
                         if key in row},
                path=str(baseline_path),
            )

    runs = []
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        screen = (load(run_dir / 'eval_20ep_final.json')
                  or load(run_dir / 'eval_40ep_final.json'))
        full = load(run_dir / 'eval_100ep_final.json')
        if screen is None and full is None:
            continue
        runs.append(dict(
            name=run_dir.name,
            screen=screen,
            full=full,
        ))

    lines = ['| run | episodes | reach | success | safe_ep | collision_ep | '
             'official_safety |', '|---|---:|---:|---:|---:|---:|---:|']
    for row in ([baseline] if baseline else []) + [
            dict(name=r['name'],
                 steps=(r['screen'] or r['full']).get('steps'),
                 episodes=(r['screen'] or r['full']).get('episodes'),
                 metrics=(r['screen'] or r['full'])['metrics'])
            for r in runs]:
        m = row['metrics']
        lines.append(
            f"| {row['name']} | {row.get('episodes', '')} | "
            f"{m['official_reach']:.4f} | {m['official_success']:.4f} | "
            f"{m['safe_episode']:.3f} | {m['collision_episode']:.3f} | "
            f"{m['official_safety']:.4f} |")
    markdown = '\n'.join(lines) + '\n'
    print(markdown)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(dict(
            baseline=baseline,
            runs=runs,
            markdown=markdown), indent=2) + '\n')
        print(f'saved: {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
