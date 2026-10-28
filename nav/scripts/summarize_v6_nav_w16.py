#!/usr/bin/env python3
"""Apply the pre-registered seed-0 W16 task warm-start screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

KEYS = (
    'safe_episode', 'official_safety', 'official_success',
    'official_reach', 'episode_return', 'unsafe', 'cbf_violation',
)


def load(path: str) -> dict:
    return json.loads(Path(path).read_text())['metrics']


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--w16-24k', required=True)
    ap.add_argument('--w16-28k', required=True)
    ap.add_argument('--control-24k', required=True)
    ap.add_argument('--control-28k', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    w24, w28 = load(args.w16_24k), load(args.w16_28k)
    c24, c28 = load(args.control_24k), load(args.control_28k)

    task_benefit = bool(
        w28['official_success'] >= 0.30
        or w28['episode_return'] >= 65.0)
    safety_ok = bool(
        w28['safe_episode'] >= 0.80
        and w28['official_safety'] >= 0.95)
    cost_ok = bool(w28['unsafe'] <= c28['unsafe'] + 0.005)
    passed = bool(task_benefit and safety_ok and cost_ok)

    result = dict(
        schema='v6.nav.w16_task_warmstart_screen.v1',
        paths=dict(
            w16_24k=args.w16_24k,
            w16_28k=args.w16_28k,
            control_24k=args.control_24k,
            control_28k=args.control_28k,
        ),
        w16_24k={k: w24.get(k) for k in KEYS},
        w16_28k={k: w28.get(k) for k in KEYS},
        control_24k={k: c24.get(k) for k in KEYS},
        control_28k={k: c28.get(k) for k in KEYS},
        delta_28k={k: float(w28[k] - c28[k]) for k in KEYS
                   if k in w28 and k in c28},
        delta_24k={k: float(w24[k] - c24[k]) for k in KEYS
                   if k in w24 and k in c24},
        thresholds=dict(
            task_success_min=0.30,
            task_return_min=65.0,
            safe_episode_min=0.80,
            official_safety_min=0.95,
            cost_margin=0.005,
        ),
        task_benefit=task_benefit,
        safety_ok=safety_ok,
        cost_ok=cost_ok,
        passed=passed,
        interpretation=(
            'seed-0 screen passed; expand to seeds 1/2 with per-seed V_T '
            'before any promotion decision'
            if passed else
            'seed-0 screen did not pass; do not expand seeds or promote'),
        caveats=[
            'one seed; seed-0 V_T trained on seed-0 16k states',
            'procedure-level comparison to from-scratch 24k/28k',
            'no C16/C24 control; T0w no-V_T remains channel attribution',
            '24k is diagnostic and cannot rescue a 28k fade',
            'no test-set tuning or best-checkpoint selection',
        ],
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    print(f'saved: {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
