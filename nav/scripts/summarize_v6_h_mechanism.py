"""Print legacy V5-coupled h-mechanism tables from saved JSON files.

Not part of the independent V6 main protocol; the default ``h_mech`` tree is
legacy.  Use ``outputs/v6/independent`` for new runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path):
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _fmt(value, digits=3):
    if value is None:
        return '-'
    try:
        return f'{float(value):.{digits}f}'
    except (TypeError, ValueError):
        return str(value)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default='outputs/v6/h_mech')
    args = ap.parse_args()
    root = Path(args.root)
    variants = ['instant', 'horizon', 'instant_hdot', 'horizon_hdot',
                'horizon_hdot_safe', 'horizon_hdot_detach',
                'horizon_hdot_margin', 'old_h0']
    print('## h supervision ablation\n')
    print('| variant | bal acc | AUROC | unsafe R | h mean | ref h r | ref grad cos | '
          'ref vel cos | grad_v/m | feasible | true CBF sat |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for name in variants:
        summary = _load(root / name / 'summary.json')
        test = summary.get('test', {})
        diag = _load(root / 'diag' / f'{name}.json')
        print('| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
            name,
            _fmt(test.get('balanced_accuracy')),
            _fmt(test.get('auroc')),
            _fmt(test.get('unsafe_recall')),
            _fmt(test.get('h_mean')),
            _fmt(diag.get('ref_h_pearson')),
            _fmt(diag.get('ref_grad_cosine')),
            _fmt(diag.get('ref_grad_vel_cosine')),
            _fmt(diag.get('grad_vel_scaled_norm')),
            _fmt(diag.get('cbf_feasible')),
            _fmt(diag.get('cbf_true_sat')),
        ))
    print('\n## Fixed-h V5 swap (diagnostic only)\n')
    print('| h | collision ep | safe ep | official safety | reach | success | '
          'pred CBF sat | true CBF sat | feasible |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for name in variants + ['official_h']:
        swap = _load(root / 'swap' / f'{name}.json')
        m = swap.get('metrics', {})
        print('| {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
            name,
            _fmt(m.get('collision_episode')),
            _fmt(m.get('safe_episode')),
            _fmt(m.get('official_safety')),
            _fmt(m.get('official_reach')),
            _fmt(m.get('official_success')),
            _fmt(m.get('cbf_pred_sat')),
            _fmt(m.get('cbf_true_sat')),
            _fmt(m.get('cbf_feasible')),
        ))
    print('\n## Actor CBF-loss ablation (fixed h, no task channel)\n')
    print('| arm | steps | collision ep | safe ep | official safety | reach | '
          'success | pred CBF sat | true CBF sat | feasible |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for name in ['baseline', 'cbf', 'baseline_margin', 'cbf_margin']:
        summary = _load(root / f'actor_{name}' / 'summary.json')
        eval_final = _load(root / f'actor_{name}' / 'eval_final.json')
        m = eval_final.get('metrics', summary.get('eval', {}))
        print('| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
            name,
            summary.get('steps', '-'),
            _fmt(m.get('collision_episode')),
            _fmt(m.get('safe_episode')),
            _fmt(m.get('official_safety')),
            _fmt(m.get('official_reach')),
            _fmt(m.get('official_success')),
            _fmt(m.get('cbf_pred_sat')),
            _fmt(m.get('cbf_true_sat')),
            _fmt(m.get('cbf_feasible')),
        ))


if __name__ == '__main__':
    main()
