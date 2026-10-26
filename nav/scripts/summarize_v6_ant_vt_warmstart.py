#!/usr/bin/env python3
"""Summarize the Ant V_T 1M warm-start transfer diagnostic.

Pre-registered rule (fixed before the control result):

- control_valid: control deterministic safe_episode >= 0.90 and
  cost_fraction <= 0.01;
- task_gain: warm deterministic forward_return >= control + 5% OR
  episode_return >= control + 5%;
- safety_not_worse: warm deterministic safe_episode >= control - 0.03,
  cost_fraction <= control + 0.005, wall_fraction <= control + 0.02,
  topple_fraction <= control + 0.02;
- vt_transfer_supported = control_valid and task_gain and safety_not_worse.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

KEYS = (
    'safe_episode', 'unsafe_episode', 'wall_episode', 'topple_episode',
    'episode_return', 'forward_return', 'ctrl_return', 'survive_return',
    'episode_length', 'survived', 'cost_rate', 'cost_fraction',
    'wall_fraction', 'topple_fraction', 'h_mean', 'h_min', 'h_neg_frac',
    'final_x', 'mean_x', 'action_abs',
)

CONTROL_SAFE_MIN = 0.90
CONTROL_COST_MAX = 0.01
TASK_REL_MARGIN = 0.05
SAFE_MARGIN = 0.03
COST_MARGIN = 0.005
WALL_MARGIN = 0.02
TOPPLE_MARGIN = 0.02


def _load(path: Path) -> dict:
    return json.loads(path.read_text())['metrics']


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--control-det', required=True)
    ap.add_argument('--control-stoch', required=True)
    ap.add_argument('--warmstart-det', required=True)
    ap.add_argument('--warmstart-stoch', required=True)
    ap.add_argument('--base-det', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    c_det = _load(Path(args.control_det))
    c_sto = _load(Path(args.control_stoch))
    w_det = _load(Path(args.warmstart_det))
    w_sto = _load(Path(args.warmstart_stoch))
    b_det = _load(Path(args.base_det))

    delta_w_c = {
        k: float(w_det[k] - c_det[k])
        for k in KEYS if k in w_det and k in c_det}
    delta_c_b = {
        k: float(c_det[k] - b_det[k])
        for k in KEYS if k in c_det and k in b_det}
    delta_w_c_sto = {
        k: float(w_sto[k] - c_sto[k])
        for k in KEYS if k in w_sto and k in c_sto}

    control_valid = bool(
        c_det['safe_episode'] >= CONTROL_SAFE_MIN
        and c_det['cost_fraction'] <= CONTROL_COST_MAX)
    task_gain = bool(
        w_det['forward_return']
        >= c_det['forward_return'] * (1.0 + TASK_REL_MARGIN)
        or w_det['episode_return']
        >= c_det['episode_return'] * (1.0 + TASK_REL_MARGIN))
    safety_not_worse = bool(
        w_det['safe_episode'] >= c_det['safe_episode'] - SAFE_MARGIN
        and w_det['cost_fraction'] <= c_det['cost_fraction'] + COST_MARGIN
        and w_det['wall_fraction'] <= c_det['wall_fraction'] + WALL_MARGIN
        and w_det['topple_fraction']
        <= c_det['topple_fraction'] + TOPPLE_MARGIN)
    vt_transfer_supported = bool(
        control_valid and task_gain and safety_not_worse)

    if not control_valid:
        interpretation = (
            'the matched no-V_T control fails its own safety guardrails; '
            'the comparison is degenerate.  Report an invalid control, make '
            'no V_T-transfer claim, and stop the Ant V_T line')
    elif vt_transfer_supported:
        interpretation = (
            'the frozen 1-D A_T channel beats the matched no-V_T 1M '
            'continuation on the deterministic Ant task without a '
            'pre-registered safety loss.  This is a one-seed cross-domain '
            'diagnostic; a 5-D/longer-budget/new-seed arm still needs a '
            'separate pre-registration')
    elif task_gain and not safety_not_worse:
        interpretation = (
            'the Ant task gain comes with a safety loss versus the matched '
            'control; the 1-D A_T channel is not usable in this form and the '
            'line stops')
    else:
        interpretation = (
            'the Ant warm-start does not beat the matched no-V_T 1M '
            'continuation on the task; the extra training explains the '
            'change and the Ant V_T line stops')

    result = dict(
        schema='v6.ant_vt_transfer.v1',
        control_deterministic=str(Path(args.control_det).resolve()),
        control_stochastic=str(Path(args.control_stoch).resolve()),
        warmstart_deterministic=str(Path(args.warmstart_det).resolve()),
        warmstart_stochastic=str(Path(args.warmstart_stoch).resolve()),
        base_deterministic=str(Path(args.base_det).resolve()),
        control_metrics=c_det,
        warmstart_metrics=w_det,
        control_stochastic_metrics=c_sto,
        warmstart_stochastic_metrics=w_sto,
        base_metrics=b_det,
        delta_warmstart_minus_control=delta_w_c,
        delta_control_minus_base=delta_c_b,
        delta_warmstart_minus_control_stochastic=delta_w_c_sto,
        thresholds=dict(
            control_safe_episode_min=CONTROL_SAFE_MIN,
            control_cost_fraction_max=CONTROL_COST_MAX,
            task_relative_margin=TASK_REL_MARGIN,
            safe_episode_margin=SAFE_MARGIN,
            cost_fraction_margin=COST_MARGIN,
            wall_fraction_margin=WALL_MARGIN,
            topple_fraction_margin=TOPPLE_MARGIN,
        ),
        control_valid=control_valid,
        task_gain=task_gain,
        safety_not_worse=safety_not_worse,
        vt_transfer_supported=vt_transfer_supported,
        run_from_scratch_arm=False,
        interpretation=interpretation,
        caveats=[
            'one seed; this is a diagnostic screen, not a formal Ant claim',
            'the promoted Ant 3-seed safety result is unchanged',
            'h, the cost penalty, the reward, the optimizer and the base '
            'checkpoint are shared by both arms; only the 1-D A_T input differs',
            'the V_T quality gate must have passed before this result is '
            'interpretable',
            'a positive result justifies only a separately pre-registered '
            'follow-up, not a main claim',
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
