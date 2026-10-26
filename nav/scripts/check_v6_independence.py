#!/usr/bin/env python3
"""Check that the main V6 path contains no legacy coupled artifacts.

This is a static guard, not a proof.  It scans the main config/script set for
V5/V4 paths, official-h checkpoints, and the old warm-start entry point.
Run from the project root or the ``nav`` directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

NAV = Path(__file__).resolve().parents[1]
ROOT = NAV.parent

MAIN_FILES = [
    NAV / 'configs/v6/v6_nav_independent.yaml',
    NAV / 'configs/v6/v6_nav_h_only_16k.yaml',
    NAV / 'configs/v6/v6_nav.yaml',
    NAV / 'configs/v6/v6_nav_mechanism.yaml',
    NAV / 'scripts/train_v6.py',
    NAV / 'scripts/evaluate_v6.py',
    NAV / 'scripts/run_v6_formal_3seed.sh',
    NAV / 'scripts/run_v6_safety_screen.sh',
    NAV / 'scripts/summarize_v6_formal.py',
    NAV / 'scripts/summarize_v6_safety.py',
    NAV / 'scripts/collect_v6_h_data.py',
    NAV / 'scripts/augment_v6_h_data.py',
    NAV / 'scripts/merge_v6_h_data.py',
    NAV / 'scripts/train_v6_official_h.py',
    NAV / 'scripts/run_v6_protocol.sh',
    NAV / 'v6/policy.py',
    NAV / 'v6/training.py',
    NAV / 'v6/ant_safety.py',
    NAV / 'v6/ant_h.py',
    NAV / 'v6/ant_env.py',
    NAV / 'v6/ant_mappo.py',
    NAV / 'configs/v6/v6_ant_h.yaml',
    NAV / 'configs/v6/v6_ant_h_only.yaml',
    NAV / 'configs/v6/v6_ant_cost_only.yaml',
    NAV / 'configs/v6/v6_ant_task_only.yaml',
    NAV / 'configs/v6/v6_ant_task_base.yaml',
    NAV / 'configs/v6/v6_ant_cost_base.yaml',
    NAV / 'configs/v6/v6_ant_h_base.yaml',
    NAV / 'configs/v6/v6_ant_cost_p1.yaml',
    NAV / 'configs/v6/v6_ant_h_p1.yaml',
    NAV / 'configs/v6/v6_ant_cost_p01.yaml',
    NAV / 'configs/v6/v6_ant_h_p01.yaml',
    NAV / 'configs/v6/v6_ant_cost_p01_10m.yaml',
    NAV / 'configs/v6/v6_ant_h_p01_10m.yaml',
    NAV / 'configs/v6/v6_ant_cost_p05_10m.yaml',
    NAV / 'configs/v6/v6_ant_h_p05_10m.yaml',
    NAV / 'configs/v6/v6_ant_h_p01s3_10m.yaml',
    NAV / 'configs/v6/v6_ant_cost_p02_10m.yaml',
    NAV / 'configs/v6/v6_ant_h_p02_10m.yaml',
    NAV / 'configs/v6/v6_ant_cost_ramp05_10m.yaml',
    NAV / 'configs/v6/v6_ant_h_ramp05_s3_10m.yaml',
    NAV / 'configs/v6/v6_ant_cost_ramp10_10m.yaml',
    NAV / 'configs/v6/v6_ant_h_ramp10_s3_10m.yaml',
    NAV / 'configs/v6/v6_ant_h_ramp05_s1_10m.yaml',
    NAV / 'configs/v6/v6_ant_h_ramp10_s1_10m.yaml',
    NAV / 'configs/v6/v6_ant_h_dynamic.yaml',
    NAV / 'configs/v6/v6_ant_h_dyn_2m.yaml',
    NAV / 'configs/v6/v6_ant_h_dyn_b2m.yaml',
    NAV / 'configs/v6/v6_ant_h_dyn_gate_2m.yaml',
    NAV / 'configs/v6/v6_ant_h_predictive.yaml',
    NAV / 'configs/v6/v6_ant_h_pred_gate_2m.yaml',
    NAV / 'scripts/collect_v6_ant_h_data.py',
    NAV / 'scripts/train_v6_ant_h.py',
    NAV / 'scripts/train_v6_ant_mappo.py',
    NAV / 'scripts/evaluate_v6_ant_mappo.py',
    NAV / 'scripts/run_v6_ant_smoke.sh',
    NAV / 'scripts/run_v6_ant_protocol.sh',
]

FORBIDDEN = (
    'Safe-MARL-proposal_v4',
    'data_v5h0',
    'h_mech',
    '--init-policy',
    'init_policy_from_v5',
    'third_party/gcbfplus/pretrained',
    'outputs/v6/official_h',
    'outputs/v6/h_data',
    'algos.safe_policy',
)


def main() -> int:
    violations: list[str] = []
    for path in MAIN_FILES:
        if not path.exists():
            violations.append(f'missing: {path}')
            continue
        text = '\n'.join(
            line for line in path.read_text().splitlines()
            if not line.lstrip().startswith('#'))
        for token in FORBIDDEN:
            if token in text:
                violations.append(f'{path.relative_to(ROOT)}: {token}')
    if violations:
        print('V6 independence check FAILED:')
        for item in violations:
            print('  -', item)
        return 1
    print(f'V6 independence check passed ({len(MAIN_FILES)} files).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
