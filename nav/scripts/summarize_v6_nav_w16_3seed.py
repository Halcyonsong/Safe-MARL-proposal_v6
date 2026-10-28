#!/usr/bin/env python3
"""Aggregate the 3-seed W16 task warm-start table and apply rule 7.1."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

METRICS = (
    'safe_episode', 'official_safety', 'official_success',
    'official_reach', 'episode_return',
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
    ap.add_argument('--w16-root',
                    default='outputs/v6/independent/nav_w16_task_12k')
    ap.add_argument('--control-root',
                    default='outputs/v6/independent/nav_stageb_repair_28k')
    ap.add_argument('--out', default='outputs/v6/independent/'
                                    'nav_w16_task_12k/table_3seed.json')
    args = ap.parse_args()
    w16_root = Path(args.w16_root)
    ctrl_root = Path(args.control_root)

    w16, ctrl = {}, {}
    for seed in (0, 1, 2):
        w16[seed] = load(w16_root / f'seed{seed}' / 'eval_100ep_final.json')
        ctrl[seed] = load(ctrl_root / f'seed{seed}' / 'eval_100ep_final.json')

    w16_sum = summary([w16[s] for s in (0, 1, 2)])
    ctrl_sum = summary([ctrl[s] for s in (0, 1, 2)])
    paired = {
        str(s): {k: float(w16[s][k] - ctrl[s][k]) for k in METRICS}
        for s in (0, 1, 2)
    }
    s = w16_sum
    rule = dict(
        mean_safe_ge_0_6933=bool(s['safe_episode']['mean'] >= 0.6933),
        mean_official_ge_0_9417=bool(
            s['official_safety']['mean'] >= 0.9417),
        mean_success_ge_0_36=bool(
            s['official_success']['mean'] >= 0.36),
        mean_return_ge_55=bool(s['episode_return']['mean'] >= 55.0),
        worst_success_ge_0_20=bool(
            s['official_success']['min'] >= 0.20),
        worst_safe_ge_0_30=bool(s['safe_episode']['min'] >= 0.30),
        safe_range_le_0_20=bool(s['safe_episode']['range'] <= 0.20),
        worst_safe_ge_0_55=bool(s['safe_episode']['min'] >= 0.55),
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
        schema='v6.nav.w16_task_3seed.v1',
        seeds=[0, 1, 2],
        w16_summary=w16_sum,
        control_summary=ctrl_sum,
        paired_w16_minus_control=paired,
        rule_7_1=rule,
        interpretation=(
            'W16 3-seed passes rule 7.1; propose a separately '
            'pre-registered promotion, do not swap the 16k row silently'
            if rule['full_pass'] else
            'W16 3-seed fails rule 7.1; keep 16k promoted and report the '
            'task/safety trade-off'),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['w16_summary'], indent=2))
    print(json.dumps(result['rule_7_1'], indent=2))
    print(f'saved: {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
