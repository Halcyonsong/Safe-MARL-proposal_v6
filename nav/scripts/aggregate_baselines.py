"""Combine V5 reference results and external baseline results into one table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _read(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _metric(payload, key):
    if payload is None:
        return None
    metrics = payload.get('metrics', payload)
    value = metrics.get(key)
    if isinstance(value, dict):
        value = value.get('mean')
    return None if value is None else float(value)


ROWS = [
    ('V5 full', 'CBF-informed safe MARL',
     'outputs/v5/seeds_16384_safety/aggregate_100ep.json'),
    ('V5 h-input', 'ablation',
     'outputs/v5/h_ablation/h_input/aggregate_100ep.json'),
    ('V5 no-h', 'ablation',
     'outputs/v5/h_ablation/no_h/aggregate_100ep.json'),
    ('V5 task-only same-reward', 'ablation',
     'outputs/v5/h_ablation/task_only/aggregate_100ep.json'),
    ('pure MAPPO final', 'Safe MARL',
     'outputs/v5/baselines/mappo_pure/aggregate_100ep_policy_final.json'),
    ('pure MAPPO step16k', 'Safe MARL',
     'outputs/v5/baselines/mappo_pure/aggregate_100ep_policy_step_16384.json'),
    ('MAPPO-Lag final', 'constrained MARL',
     'outputs/v5/baselines/mappo_lag/aggregate_100ep_policy_final.json'),
    ('GCBF+ official', 'CBF / learned',
     'outputs/v5/baselines/gcbfplus/eval_100ep.json'),
    ('centralized CBF-QP', 'CBF-QP',
     'outputs/v5/baselines/centralized_cbf_qp/eval_100ep.json'),
]

KEYS = [
    'official_safety', 'safe_episode', 'collision_episode',
    'official_reach', 'official_success', 'episode_return',
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='outputs/v5/baselines/comparison.json')
    parser.add_argument('--markdown', default='outputs/v5/baselines/comparison.md')
    args = parser.parse_args()

    rows = []
    for name, family, path in ROWS:
        payload = _read(path)
        rows.append(dict(
            method=name, family=family, source=path,
            available=payload is not None,
            metrics={key: _metric(payload, key) for key in KEYS},
        ))

    payload = dict(
        protocol=dict(
            environment='official GCBF+ DoubleIntegrator',
            num_agents=8, area_size=4.0, max_step=256,
            scenes='jax.random.PRNGKey(10000..10099)',
            deterministic=True,
        ),
        rows=rows,
    )
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(payload, indent=2))

    lines = [
        '| method | family | official safety | safe episode | collision episode | reach | success | task return |',
        '|---|---|---:|---:|---:|---:|---:|---:|',
    ]
    for row in rows:
        metrics = row['metrics']
        if not row['available']:
            values = ['TBD'] * len(KEYS)
        else:
            values = [
                ('TBD' if metrics[key] is None else f'{metrics[key]:.4f}')
                for key in KEYS
            ]
        lines.append(
            f"| {row['method']} | {row['family']} | " + ' | '.join(values) + ' |')
    markdown = ('\n'.join(lines) + '\n\n'
                '> `task return` is `20 * progress + 5 * reached`; it is not the official GCBF+ control reward.\n')
    md = ROOT / args.markdown
    md.write_text(markdown)
    print(markdown)


if __name__ == '__main__':
    main()
