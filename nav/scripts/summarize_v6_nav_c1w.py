#!/usr/bin/env python3
"""Summarize the C1w actor1+h2 4k warm-start diagnostic.

C1w is a one-seed diagnostic adaptation arm.  It continues the fixed V6
actor1 seed-0 checkpoint for 4,096 steps with h2 seed 0.  There is **no**
h1-continued control, so changes cannot be attributed to h2 rather than to
the extra training.  See
``outputs/v6/independent/nav/STAGE_C1W_H2_ACTOR1_4K.md``.

The script reports two things separately:

1. whether the result meets the same five C1a guardrail thresholds;
2. whether it improves on the no-training actor1+h2 h-swap.

Neither is a formal C0/C1 pass.  The output is a diagnostic record.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

KEYS = (
    'official_safety',
    'safe_episode',
    'official_success',
    'official_reach',
    'episode_return',
    'collision_episode',
    'warning_episode',
)


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text())['metrics']


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--eval', required=True, help='C1w eval JSON')
    ap.add_argument('--baseline', required=True,
                    help='actor1 seed0 + h1 eval JSON')
    ap.add_argument('--swap', required=True,
                    help='post-hoc actor1 seed0 + h2 no-training eval JSON')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    cur = _load(args.eval)
    base = _load(args.baseline)
    swap = _load(args.swap)

    delta_base = {k: float(cur[k] - base[k]) for k in KEYS}
    delta_swap = {k: float(cur[k] - swap[k]) for k in KEYS}

    # Same one-seed guardrails used for C1a, relative to actor1+h1.
    thresholds = dict(
        official_safety_min=float(base['official_safety'] + 0.00425),
        safe_episode_min=float(base['safe_episode']),
        official_success_min=float(base['official_success'] - 0.0275),
        official_reach_min=float(base['official_reach'] - 0.035),
        episode_return_min=float(base['episode_return'] - 2.91),
    )
    checks = dict(
        official_safety=bool(cur['official_safety']
                             >= thresholds['official_safety_min']),
        safe_episode=bool(cur['safe_episode']
                          >= thresholds['safe_episode_min']),
        official_success=bool(cur['official_success']
                              >= thresholds['official_success_min']),
        official_reach=bool(cur['official_reach']
                            >= thresholds['official_reach_min']),
        episode_return=bool(cur['episode_return']
                            >= thresholds['episode_return_min']),
    )
    meets_c1a_thresholds = bool(all(checks.values()))

    # Did the short continuation improve on the no-training swap?
    beats_swap_safety = bool(
        cur['official_safety'] >= swap['official_safety']
        and cur['safe_episode'] >= swap['safe_episode'])
    task_not_worse_than_swap = bool(
        cur['official_success'] >= swap['official_success'] - 0.0275
        and cur['official_reach'] >= swap['official_reach'] - 0.035
        and cur['episode_return'] >= swap['episode_return'] - 2.91)
    promising_adaptation = bool(
        meets_c1a_thresholds and beats_swap_safety
        and task_not_worse_than_swap)

    if promising_adaptation:
        interpretation = (
            'promising adaptation: meets the C1a guardrails and improves on '
            'the no-training h-swap without task regression; one seed, no '
            'h1 control, so not causal')
    elif meets_c1a_thresholds and not beats_swap_safety:
        interpretation = (
            'meets C1a guardrails but does not improve on the no-training '
            'h-swap; the short h2 continuation did not make the pair better '
            'matched than simply swapping h')
    elif not meets_c1a_thresholds:
        interpretation = (
            'does not meet the C1a guardrails; the h2 continuation did not '
            'preserve the actor1 safety/task level')
    else:
        interpretation = (
            'improves on the h-swap safety but fails one or more task '
            'guardrails; possible task drift')

    result = dict(
        schema='v6.nav.c1w_h2_actor1_4k_diagnostic.v1',
        eval=str(Path(args.eval).resolve()),
        baseline=str(Path(args.baseline).resolve()),
        swap=str(Path(args.swap).resolve()),
        baseline_metrics=base,
        swap_metrics=swap,
        c1w_metrics=cur,
        delta_vs_actor1_h1=delta_base,
        delta_vs_no_train_h2_swap=delta_swap,
        thresholds=thresholds,
        checks=checks,
        meets_c1a_thresholds=meets_c1a_thresholds,
        beats_swap_safety=beats_swap_safety,
        task_not_worse_than_swap=task_not_worse_than_swap,
        promising_adaptation=promising_adaptation,
        interpretation=interpretation,
        caveats=[
            'one seed only',
            'no h1-continued control: extra 4096 steps are confounded with h2',
            'warm start from actor1, not from scratch',
            'not a formal C0/C1 result and not a CBF/formal-safety claim',
        ],
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
