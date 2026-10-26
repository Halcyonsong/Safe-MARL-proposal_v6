#!/usr/bin/env python3
"""Summarize the T0w V_T warm-start diagnostic.

The warm-start is a screen, not a main claim.  The no-V_T 4k control is run
only if this screen shows a task benefit without a safety collapse.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

KEYS = (
    'safe_episode', 'collision_episode', 'warning_episode',
    'official_safety', 'official_success', 'official_reach',
    'episode_return', 'action_abs', 'h_mean', 'h_min', 'cbf_violation',
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text())['metrics']


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--eval', required=True)
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    ev = _load(Path(args.eval))
    base = _load(Path(args.baseline))
    delta = {k: float(ev[k] - base[k]) for k in KEYS if k in ev and k in base}

    task_benefit = bool(
        ev['official_success'] >= 0.4775
        or ev['episode_return'] >= 65.91)
    safety_ok = bool(
        ev['safe_episode'] >= 0.80
        and ev['official_safety'] >= 0.97)
    has_benefit = bool(task_benefit and safety_ok)

    if has_benefit:
        interpretation = (
            'warm-start shows a task-benefit signal without a safety '
            'collapse; run the matched no-V_T 4k control before any '
            'from-scratch task-channel arm')
    elif task_benefit and not safety_ok:
        interpretation = (
            'warm-start shows a task signal but safety collapses; the task '
            'channel is not usable in this form; do not run the control')
    else:
        interpretation = (
            'warm-start does not show a task benefit at the pre-registered '
            'margin; stop the task-channel line; do not run the control')

    result = dict(
        schema='v6.nav.t0w_vt_warmstart.v1',
        eval=str(Path(args.eval).resolve()),
        baseline=str(Path(args.baseline).resolve()),
        eval_metrics=ev,
        baseline_metrics=base,
        delta=delta,
        thresholds=dict(
            task_success_min=0.4775,
            task_return_min=65.91,
            safe_episode_min=0.80,
            official_safety_min=0.97,
        ),
        task_benefit=task_benefit,
        safety_ok=safety_ok,
        has_benefit=has_benefit,
        run_no_vt_control=has_benefit,
        interpretation=interpretation,
        caveats=[
            'warm-start includes 4096 extra training steps; the no-V_T 4k '
            'control is required to isolate the task channel',
            'V_T is trained on actor1 seed-0 states; one seed only',
            'not a main V6 claim and not a from-scratch result',
            'the gate is off at evaluation; h remains safety-only',
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
