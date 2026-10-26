#!/usr/bin/env python3
"""Apply the pre-registered C1a seed-0 pilot rule.

C1a trains actor2 from scratch with C0 h2 seed 0.  The pilot is a one-seed
directional safety check with a task-collapse guardrail; it is not a 3-seed
claim.  See ``outputs/v6/independent/nav/STAGE_C1A_H2_ACTOR2_PILOT.md``.

The thresholds are expressed relative to the actor1 seed-0 baseline:
  official_safety >= baseline + 0.00425
  safe_episode   >= baseline
  official_success >= baseline - 0.0275
  official_reach >= baseline - 0.035
  episode_return >= baseline - 2.91
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text())['metrics']


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--eval', required=True, help='C1a eval JSON')
    ap.add_argument('--baseline', required=True,
                    help='actor1 seed0 eval JSON')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    cur = _load(args.eval)
    base = _load(args.baseline)
    deltas = {key: float(cur[key] - base[key])
              for key in ('official_safety', 'safe_episode',
                          'official_success', 'official_reach',
                          'episode_return')}
    checks = dict(
        official_safety_ge_baseline_plus_0p00425=(
            deltas['official_safety'] >= 0.00425),
        safe_episode_not_worse=(
            deltas['safe_episode'] >= 0.0),
        official_success_not_worse_by_0p0275=(
            deltas['official_success'] >= -0.0275),
        official_reach_not_worse_by_0p035=(
            deltas['official_reach'] >= -0.035),
        episode_return_not_worse_by_2p91=(
            deltas['episode_return'] >= -2.91),
    )
    result = dict(
        schema='v6.nav.c1a_h2_actor2_pilot.v1',
        eval=str(Path(args.eval).resolve()),
        baseline=str(Path(args.baseline).resolve()),
        baseline_metrics=base,
        c1a_metrics=cur,
        delta=deltas,
        rule=dict(
            official_safety_min=float(base['official_safety'] + 0.00425),
            safe_episode_min=float(base['safe_episode']),
            official_success_min=float(base['official_success'] - 0.0275),
            official_reach_min=float(base['official_reach'] - 0.035),
            episode_return_min=float(base['episode_return'] - 2.91),
        ),
        checks=checks,
        passed=bool(all(checks.values())),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
