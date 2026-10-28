#!/usr/bin/env python3
"""Aggregate the V6 nav 3-seed 16k/24k/28k table and apply rule 7.1."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

METRICS = (
    'safe_episode', 'official_safety', 'official_success',
    'reach', 'reach_final', 'episode_return',
)


def load(path: Path) -> dict:
    return json.loads(path.read_text())['metrics']


def summary(rows: list[dict]) -> dict:
    out = {}
    for key in METRICS:
        vals = [float(row[key]) for row in rows]
        out[key] = dict(
            mean=statistics.fmean(vals),
            std=statistics.stdev(vals),
            min=min(vals),
            max=max(vals),
            range=max(vals) - min(vals),
            per_seed=vals,
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default='outputs/v6/independent')
    ap.add_argument('--out', default='outputs/v6/independent/nav/'
                                 'nav_3seed_16k_24k_28k_table.json')
    args = ap.parse_args()
    root = Path(args.root)

    rows = {}
    for seed in (0, 1, 2):
        rows[seed] = {
            '16k': load(root / 'nav_stageb_repair_16k' / f'seed{seed}' /
                        'eval_100ep_final.json'),
            '24k': load(root / 'nav_stageb_repair_28k' / f'seed{seed}' /
                        'eval_100ep_step24576.json'),
            '28k': load(root / 'nav_stageb_repair_28k' / f'seed{seed}' /
                        'eval_100ep_final.json'),
        }

    summaries = {
        budget: summary([rows[s][budget] for s in (0, 1, 2)])
        for budget in ('16k', '24k', '28k')
    }
    paired = {}
    for seed in (0, 1, 2):
        paired[str(seed)] = {
            f'{new}_minus_{old}': {
                key: float(rows[seed][new][key] - rows[seed][old][key])
                for key in METRICS
            }
            for new, old in (('24k', '16k'), ('28k', '16k'), ('28k', '24k'))
        }

    s28 = summaries['28k']
    rule = dict(
        mean_safe_ge_0_6933=bool(s28['safe_episode']['mean'] >= 0.6933),
        mean_official_ge_0_9417=bool(
            s28['official_safety']['mean'] >= 0.9417),
        mean_success_ge_0_36=bool(
            s28['official_success']['mean'] >= 0.36),
        mean_return_ge_55=bool(s28['episode_return']['mean'] >= 55.0),
        worst_success_ge_0_20=bool(
            s28['official_success']['min'] >= 0.20),
        worst_safe_ge_0_30=bool(s28['safe_episode']['min'] >= 0.30),
        safe_range_le_0_20=bool(s28['safe_episode']['range'] <= 0.20),
        worst_safe_ge_0_55=bool(s28['safe_episode']['min'] >= 0.55),
    )
    rule['mean_and_task_guardrails_pass'] = bool(
        rule['mean_safe_ge_0_6933']
        and rule['mean_official_ge_0_9417']
        and rule['mean_success_ge_0_36']
        and rule['mean_return_ge_55']
        and rule['worst_success_ge_0_20']
        and rule['worst_safe_ge_0_30'])
    rule['spread_pass'] = bool(
        rule['safe_range_le_0_20'] and rule['worst_safe_ge_0_55'])
    rule['full_pass'] = bool(
        rule['mean_and_task_guardrails_pass'] and rule['spread_pass'])

    result = dict(
        schema='v6.nav.3seed_16k_24k_28k_table.v1',
        seeds=[0, 1, 2],
        summaries=summaries,
        paired=paired,
        rule_7_1=rule,
        interpretation=(
            '28k passes the pre-registered mean/task/variance rule; report '
            'budget sensitivity, not a promoted main row'
            if rule['full_pass'] else
            '28k fails rule 7.1; keep 16k promoted and report the budget '
            'trade-off/negative result'),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['summaries'], indent=2))
    print(json.dumps(result['rule_7_1'], indent=2))
    print(f'saved: {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
