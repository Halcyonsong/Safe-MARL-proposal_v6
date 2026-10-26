#!/usr/bin/env python3
"""Summarize the T0w matched no-V_T 4k control.

The control isolates the frozen V_T task channel from the extra 4,096
training steps.  It is valid only if it keeps the actor1 safety guardrails;
otherwise the comparison is degenerate and nothing is concluded.

Pre-registered rule (fixed before seeing the control result):

- control_valid: control safe_episode >= 0.80 and official_safety >= 0.97;
- task_gain: warm-start official_success >= control + 0.05 OR
  episode_return >= control + 5.0;
- safety_not_worse: warm-start safe_episode >= control - 0.03 AND
  official_safety >= control - 0.01;
- vt_channel_supported = control_valid and task_gain and safety_not_worse.
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

TASK_SUCCESS_MARGIN = 0.05
TASK_RETURN_MARGIN = 5.0
SAFE_EPISODE_MARGIN = 0.03
OFFICIAL_SAFETY_MARGIN = 0.01
CONTROL_SAFE_MIN = 0.80
CONTROL_OFFICIAL_SAFETY_MIN = 0.97


def _load(path: Path) -> dict:
    return json.loads(path.read_text())['metrics']


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--control', required=True)
    ap.add_argument('--warmstart', required=True)
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    ctrl = _load(Path(args.control))
    warm = _load(Path(args.warmstart))
    base = _load(Path(args.baseline))

    delta_warm_ctrl = {
        k: float(warm[k] - ctrl[k]) for k in KEYS if k in warm and k in ctrl}
    delta_ctrl_base = {
        k: float(ctrl[k] - base[k]) for k in KEYS if k in ctrl and k in base}

    control_valid = bool(
        ctrl['safe_episode'] >= CONTROL_SAFE_MIN
        and ctrl['official_safety'] >= CONTROL_OFFICIAL_SAFETY_MIN)
    task_gain = bool(
        warm['official_success'] >= ctrl['official_success'] + TASK_SUCCESS_MARGIN
        or warm['episode_return'] >= ctrl['episode_return'] + TASK_RETURN_MARGIN)
    safety_not_worse = bool(
        warm['safe_episode'] >= ctrl['safe_episode'] - SAFE_EPISODE_MARGIN
        and warm['official_safety']
        >= ctrl['official_safety'] - OFFICIAL_SAFETY_MARGIN)
    vt_channel_supported = bool(control_valid and task_gain and safety_not_worse)

    if not control_valid:
        interpretation = (
            'matched control fails its own safety guardrails; the comparison '
            'is degenerate.  Report an invalid control, make no task-channel '
            'claim, and do not start the from-scratch arm')
    elif vt_channel_supported:
        interpretation = (
            'the frozen V_T task channel beats the matched no-V_T 4k control '
            'on task without a safety loss; the channel is supported beyond '
            'extra training as a one-seed diagnostic.  A from-scratch '
            'task-channel arm still needs a separate pre-registration')
    elif task_gain and not safety_not_worse:
        interpretation = (
            'the task gain comes with a safety loss versus the matched '
            'control; the task channel is not usable in this form and the '
            'from-scratch arm is not justified')
    else:
        interpretation = (
            'the warm-start does not beat the matched no-V_T 4k control on '
            'task; the apparent benefit is explained by the extra training '
            'steps.  Stop the task-channel line before any from-scratch arm')

    result = dict(
        schema='v6.nav.t0w_vt_control.v1',
        control=str(Path(args.control).resolve()),
        warmstart=str(Path(args.warmstart).resolve()),
        baseline=str(Path(args.baseline).resolve()),
        control_metrics=ctrl,
        warmstart_metrics=warm,
        baseline_metrics=base,
        delta_warmstart_minus_control=delta_warm_ctrl,
        delta_control_minus_baseline=delta_ctrl_base,
        thresholds=dict(
            task_success_margin=TASK_SUCCESS_MARGIN,
            task_return_margin=TASK_RETURN_MARGIN,
            safe_episode_margin=SAFE_EPISODE_MARGIN,
            official_safety_margin=OFFICIAL_SAFETY_MARGIN,
            control_safe_episode_min=CONTROL_SAFE_MIN,
            control_official_safety_min=CONTROL_OFFICIAL_SAFETY_MIN,
        ),
        control_valid=control_valid,
        task_gain=task_gain,
        safety_not_worse=safety_not_worse,
        vt_channel_supported=vt_channel_supported,
        run_from_scratch_arm=vt_channel_supported,
        interpretation=interpretation,
        caveats=[
            'one seed; this is a diagnostic screen, not a formal claim',
            'V_T is trained on actor1 seed-0 states; the control and '
            'warm-start share init, steps, seeds, h, gate, cost and reward',
            'the gate is off at evaluation; h remains safety-only',
            'a positive result justifies only a separately pre-registered '
            'from-scratch task-channel arm, not a main claim',
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
