#!/usr/bin/env python3
"""Aggregate the read-only h2/actor2 diagnostic audit.

The audit JSONs are produced by ``diagnose_v6_nav_h2.py`` for the fixed final
actor1 checkpoint of each seed.  This script only reads those files (and the
saved row arrays) and reports:

- cross-seed actor1 reproduction versus the formal 100ep evaluation;
- collision-state h1 / gate statistics;
- h1 behavior in the six steps before each new collision onset;
- h1 calibration on the actor1 state distribution, compared with the offline
  repaired-h validation/test reference;
- a pre-declared, descriptive classification of the bottleneck.

It does not train, select checkpoints, or change any gate/cost setting.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from diagnose_v6_nav_h2 import (  # noqa: E402
    _auc,
    _binary_metrics,
    _pearson,
    _quantiles,
)

ROOT = SCRIPTS.parents[1]

OFFLINE_H_SUMMARY = (
    ROOT / 'outputs/v6/independent/h_stageb_repair/summary.json')
DATA_REPAIR_SUMMARY = (
    ROOT / 'outputs/v6/independent/data_stageb_repair/summary.json')


def _load(path: Path) -> dict:
    with path.open() as stream:
        return json.load(stream)


def _seed_rows(payload: dict) -> dict:
    aggregate = payload['aggregate']
    collision = payload['collision']
    calibration = payload['calibration']
    return dict(
        episodes=payload['episodes'],
        safe_episode=aggregate['safe_episode'],
        official_safety=aggregate['official_safety'],
        official_success=aggregate['official_success'],
        official_reach=aggregate['official_reach'],
        episode_return=aggregate['episode_return'],
        collision_episodes=aggregate['collision_episodes'],
        collision_events=aggregate['collision_events'],
        collision_steps=aggregate['collision_steps'],
        agent_steps=aggregate['agent_steps'],
        h_mean=payload['h_all']['mean'],
        h_min=payload['h_all']['min'],
        h_q05=payload['h_all']['q05'],
        gate_open_frac=payload['gate_all']['open_frac'],
        gate_closed_frac=payload['gate_all']['closed_frac'],
        gate_fully_closed_frac=payload['gate_all']['fully_closed_frac'],
        collision_h_median=collision['h_cur'].get('q50'),
        collision_h_min=collision['h_cur'].get('min'),
        collision_h_max=collision['h_cur'].get('max'),
        collision_h_next_median=collision['h_next'].get('q50'),
        frac_h_cur_above_high=collision['frac_h_cur_above_high'],
        frac_h_cur_above_zero=collision['frac_h_cur_above_zero'],
        frac_h_cur_in_gate=collision['frac_h_cur_in_gate'],
        frac_h_cur_below_zero=collision['frac_h_cur_below_zero'],
        frac_h_next_above_high=collision['frac_h_next_above_high'],
        frac_gate_open=collision['frac_gate_open'],
        frac_gate_half=collision['frac_gate_half'],
        frac_gate_closed=collision['frac_gate_closed'],
        target_corr=calibration['target_corr'],
        target_corr_raw=calibration.get('target_corr_raw'),
        safe_target_corr=calibration['safe_target_corr'],
        unsafe_target_corr=calibration['unsafe_target_corr'],
        auroc_unsafe=calibration['auroc_unsafe'],
        balanced_accuracy=(calibration['binary_at_zero']
                           .get('balanced_accuracy')),
        safe_recall=(calibration['binary_at_zero'].get('safe_recall')),
        unsafe_recall=(calibration['binary_at_zero'].get('unsafe_recall')),
        false_alarm=(calibration['binary_at_zero'].get('false_alarm')),
        label_rates=calibration.get('label_rates'),
        future_collision_auroc=calibration['future_collision_auroc'],
        future_collision_count=calibration['future_collision_count'],
        future_unsafe_auroc=calibration['future_unsafe_auroc'],
        future_unsafe_count=calibration['future_unsafe_count'],
        pre_collision=payload['pre_collision'],
    )


def _region_metrics(h: np.ndarray, target: np.ndarray,
                    mask: np.ndarray) -> dict:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return dict(count=0)
    hh, tt = h[mask], target[mask]
    return dict(
        count=int(mask.sum()),
        h_mean=float(hh.mean()),
        target_mean=float(tt.mean()),
        mae=float(np.mean(np.abs(hh - tt))),
        corr=_pearson(hh, tt),
    )


def _npz_metrics(path: Path, gate_high: float) -> dict:
    with np.load(path) as data:
        h = data['h'].astype(np.float64)
        target = data['target'].astype(np.float64)
        current_unsafe = data['current_unsafe'].astype(bool)
        horizon_safe = data['horizon_safe'].astype(bool)
        future_collision = data['future_collision'].astype(bool)
    safe = horizon_safe
    return dict(
        target_mae=float(np.mean(np.abs(h - target))),
        target_rmse=float(np.sqrt(np.mean((h - target) ** 2))),
        target_mae_safe=float(np.mean(np.abs(h[safe] - target[safe]))),
        target_mean_pred=float(h.mean()),
        target_mean_true=float(target.mean()),
        h_std=float(h.std()),
        target_std=float(target.std()),
        safe_target_std=float(target[safe].std()),
        risk_regions=dict(
            target_lt_0p5=_region_metrics(h, target, target < 0.5),
            h_lt_high=_region_metrics(h, target, h < gate_high),
            future_collision=_region_metrics(h, target, future_collision),
            current_unsafe=_region_metrics(h, target, current_unsafe),
        ),
    )


def _weighted(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not mask.any():
        return None
    return float(np.sum(values[mask] * weights[mask])
                 / np.sum(weights[mask]))


def _pool_rows(paths: list[Path], gate_high: float) -> dict:
    arrays = []
    for path in paths:
        with np.load(path) as data:
            arrays.append({k: data[k] for k in data.files})
    if not arrays:
        return {}
    pooled = {k: np.concatenate([a[k] for a in arrays])
              for k in arrays[0]}
    h = pooled['h'].astype(np.float64)
    target = pooled['target'].astype(np.float64)
    target_raw = pooled['target_raw'].astype(np.float64)
    current_unsafe = pooled['current_unsafe'].astype(bool)
    horizon_safe = pooled['horizon_safe'].astype(bool)
    graded = current_unsafe | horizon_safe
    unsafe_label = current_unsafe[graded]
    ignored = ~current_unsafe & ~horizon_safe
    return dict(
        rows=int(h.size),
        episodes=int(np.unique(pooled['episode']).size),
        target_corr=_pearson(h, target),
        target_corr_raw=_pearson(h, target_raw),
        safe_target_corr=_pearson(h[horizon_safe], target[horizon_safe]),
        unsafe_target_corr=_pearson(h[current_unsafe],
                                    target[current_unsafe]),
        target_mae=float(np.mean(np.abs(h - target))),
        target_rmse=float(np.sqrt(np.mean((h - target) ** 2))),
        target_mae_safe=float(np.mean(np.abs(
            h[horizon_safe] - target[horizon_safe]))),
        target_mean_pred=float(h.mean()),
        target_mean_true=float(target.mean()),
        h_std=float(h.std()),
        target_std=float(target.std()),
        safe_target_std=float(target[horizon_safe].std()),
        risk_regions=dict(
            target_lt_0p5=_region_metrics(h, target, target < 0.5),
            h_lt_high=_region_metrics(h, target, h < gate_high),
            future_collision=_region_metrics(
                h, target, pooled['future_collision'].astype(bool)),
            current_unsafe=_region_metrics(h, target, current_unsafe),
        ),
        auroc_unsafe=_auc(-h[graded], unsafe_label),
        binary_at_zero=_binary_metrics(h[graded], unsafe_label, 0.0),
        label_rates=dict(
            current_unsafe=float(current_unsafe.mean()),
            horizon_safe=float(horizon_safe.mean()),
            ignored=float(ignored.mean()),
        ),
        h_by_label=dict(
            horizon_safe=_quantiles(h[horizon_safe]),
            current_unsafe=_quantiles(h[current_unsafe]),
            ignored=_quantiles(h[ignored]),
        ),
        future_collision_auroc=_auc(
            -h, pooled['future_collision'].astype(bool)),
        future_collision_count=int(pooled['future_collision'].sum()),
        future_unsafe_auroc=_auc(
            -h, pooled['future_unsafe'].astype(bool)),
        future_unsafe_count=int(pooled['future_unsafe'].sum()),
        gate=dict(
            open_frac=float(np.mean(pooled['gate_next'] >= 0.999)),
            closed_frac=float(np.mean(pooled['gate_next'] < 0.5)),
            fully_closed_frac=float(np.mean(pooled['gate_next'] <= 1e-6)),
        ),
    )


def _offline_reference() -> dict:
    summary = _load(OFFLINE_H_SUMMARY)
    data = _load(DATA_REPAIR_SUMMARY)
    return dict(
        val=dict(target_corr=summary['val']['target_corr'],
                 safe_target_corr=summary['val']['target_corr_safe'],
                 target_mae=summary['val']['target_mae'],
                 target_rmse=summary['val']['target_rmse'],
                 balanced_accuracy=summary['val']['balanced_accuracy'],
                 auroc_unsafe=summary['val']['auroc'],
                 grad_cosine=summary['val']['grad_cosine']),
        test=dict(target_corr=summary['test']['target_corr'],
                  safe_target_corr=summary['test']['target_corr_safe'],
                  target_mae=summary['test']['target_mae'],
                  target_rmse=summary['test']['target_rmse'],
                  target_std=data['splits']['test']['target_std'],
                  balanced_accuracy=summary['test']['balanced_accuracy'],
                  auroc_unsafe=summary['test']['auroc'],
                  grad_cosine=summary['test']['grad_cosine']),
    )


def _classify(per_seed: dict, pooled: dict, offline: dict) -> dict:
    """Descriptive classification declared before looking at the results.

    h-limited means collisions concentrate in states h1 calls safe and h1
    calibration on the actor1 distribution drops clearly relative to offline
    test.  Actor/action-limited means collisions concentrate in states h1
    already flags (gate closed) while calibration holds.  Everything else is
    mixed.
    """
    events = sum(row['collision_events'] for row in per_seed.values())
    high = _weighted(
        [row['frac_h_cur_above_high'] for row in per_seed.values()],
        [row['collision_events'] for row in per_seed.values()])
    open_ = _weighted(
        [row['frac_gate_open'] for row in per_seed.values()],
        [row['collision_events'] for row in per_seed.values()])
    below = _weighted(
        [row['frac_h_cur_below_zero'] for row in per_seed.values()],
        [row['collision_events'] for row in per_seed.values()])
    closed = _weighted(
        [row['frac_gate_closed'] for row in per_seed.values()],
        [row['collision_events'] for row in per_seed.values()])
    target_drop = (offline['test']['target_corr'] - pooled['target_corr']
                   if pooled.get('target_corr') is not None else None)
    safe_drop = (offline['test']['safe_target_corr']
                 - pooled['safe_target_corr']
                 if pooled.get('safe_target_corr') is not None else None)
    pooled_mae = pooled.get('target_mae')
    mae_ratio = (pooled_mae / offline['test']['target_mae']
                 if pooled_mae is not None else None)
    lag5_high = _weighted(
        [row['pre_collision']['lag_5']['frac_above_high']
         for row in per_seed.values()],
        [row['pre_collision']['lag_5']['count']
         for row in per_seed.values()])
    lag5_zero = _weighted(
        [row['pre_collision']['lag_5']['frac_above_zero']
         for row in per_seed.values()],
        [row['pre_collision']['lag_5']['count']
         for row in per_seed.values()])
    lag0_high = _weighted(
        [row['pre_collision']['lag_0']['frac_above_high']
         for row in per_seed.values()],
        [row['pre_collision']['lag_0']['count']
         for row in per_seed.values()])
    high_side = ((high is not None and high >= 0.5)
                 or (open_ is not None and open_ >= 0.5))
    low_side = ((below is not None and below >= 0.5)
                or (closed is not None and closed >= 0.5))
    # Correlation alone is attenuated by the narrow target range on actor1
    # states, so use absolute error as the primary calibration criterion.
    # 0.20 is 32% above the offline repaired test MAE (0.151).
    calibration_degrades = ((pooled_mae is not None and pooled_mae >= 0.20)
                            or (safe_drop is not None and safe_drop >= 0.10))
    if events == 0:
        label = 'no-collisions'
    elif high_side and calibration_degrades:
        label = 'h-limited: h2 justified'
    elif high_side and not calibration_degrades:
        label = 'h-safe-labeled: policy/target mismatch, h2 plausible'
    elif low_side and not calibration_degrades:
        label = 'actor/action-limited: h2 likely marginal'
    else:
        label = 'mixed'
    return dict(
        rule=(
            'h-limited if >=50% of new collision onsets have h_cur>=high '
            'or gate_next open, and actor1-state calibration degrades '
            '(pooled target MAE >=0.20 vs offline test 0.151, or safe corr '
            'drop >=0.10); h-safe-labeled if the same high/gate-open '
            'collision pattern holds but calibration does not degrade '
            '(policy/target mismatch, h2 possible but action conditioning '
            'may be needed); actor/action-limited if >=50% have h_cur<=0 or '
            'gate_next closed without calibration degradation; else mixed.'),
        collision_events=int(events),
        frac_h_cur_above_high=high,
        frac_gate_open=open_,
        frac_h_cur_below_zero=below,
        frac_gate_closed=closed,
        target_corr_drop_vs_test=target_drop,
        safe_target_corr_drop_vs_test=safe_drop,
        target_mae=pooled_mae,
        target_mae_ratio_vs_test=mae_ratio,
        pre_collision_lag5_frac_above_high=lag5_high,
        pre_collision_lag5_frac_above_zero=lag5_zero,
        pre_collision_lag0_frac_above_high=lag0_high,
        label=label,
    )


def _fmt(value, digits=4):
    if value is None:
        return 'n/a'
    if isinstance(value, float):
        return f'{value:.{digits}f}'
    return str(value)


def _markdown(summary: dict) -> str:
    per = summary['per_seed']
    pooled = summary['pooled']
    offline = summary['offline_reference']
    decision = summary['decision']
    lines = []
    lines.append('# V6 navigation h2/actor2 diagnostic audit\n')
    lines.append('Read-only audit of the fixed final actor1 checkpoints.  '
                 'No training, no checkpoint selection, no gate change.\n')
    lines.append('The classification below is the pre-registered '
                 '**impact-state** rule.  It does not test whether h1 warned '
                 'early enough.  See H2_ACTOR2_DIAGNOSTIC.md for the '
                 'temporal/data-source reinterpretation (overall diagnosis: '
                 'mixed).\n')
    lines.append('## Actor1 reproduction\n')
    lines.append('| seed | safe_ep | official_safety | success | reach | '
                 'return | collision ep | events |')
    lines.append('|---:|---:|---:|---:|---:|---:|---:|---:|')
    for seed, row in per.items():
        lines.append(
            f"| {seed} | {_fmt(row['safe_episode'],3)} | "
            f"{_fmt(row['official_safety'])} | "
            f"{_fmt(row['official_success'])} | "
            f"{_fmt(row['official_reach'])} | "
            f"{_fmt(row['episode_return'],2)} | "
            f"{row['collision_episodes']} | {row['collision_events']} |")
    lines.append('')
    lines.append('## Collision states (new collision onsets)\n')
    lines.append('| seed | events | h_cur med | h_cur >= high | h_cur > 0 | '
                 'gate next open | gate next closed | h_next med |')
    lines.append('|---:|---:|---:|---:|---:|---:|---:|---:|')
    for seed, row in per.items():
        lines.append(
            f"| {seed} | {row['collision_events']} | "
            f"{_fmt(row['collision_h_median'])} | "
            f"{_fmt(row['frac_h_cur_above_high'],3)} | "
            f"{_fmt(row['frac_h_cur_above_zero'],3)} | "
            f"{_fmt(row['frac_gate_open'],3)} | "
            f"{_fmt(row['frac_gate_closed'],3)} | "
            f"{_fmt(row['collision_h_next_median'])} |")
    lines.append('')
    lines.append('## h1 before a collision onset (state h, lag in steps)\n')
    lines.append('| seed | lag 0 | lag 1 | lag 2 | lag 3 | lag 4 | lag 5 | '
                 'lag5 >= high | lag5 > 0 |')
    lines.append('|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for seed, row in per.items():
        pre = row['pre_collision']
        means = [pre[f'lag_{lag}']['mean_h'] for lag in range(6)]
        lines.append(
            f"| {seed} | " + ' | '.join(_fmt(v) for v in means) +
            f" | {_fmt(pre['lag_5']['frac_above_high'],3)} | "
            f"{_fmt(pre['lag_5']['frac_above_zero'],3)} |")
    lines.append('')
    lines.append('## h1 calibration on actor1 states\n')
    lines.append('| source | target corr | safe target corr | target MAE | '
                 'target RMSE | BA | AUROC | future-collision AUROC |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|')
    lines.append(
        f"| offline repaired test | {_fmt(offline['test']['target_corr'])} | "
        f"{_fmt(offline['test']['safe_target_corr'])} | "
        f"{_fmt(offline['test']['target_mae'])} | "
        f"{_fmt(offline['test']['target_rmse'])} | "
        f"{_fmt(offline['test']['balanced_accuracy'])} | "
        f"{_fmt(offline['test']['auroc_unsafe'])} | n/a |")
    lines.append(
        f"| offline repaired val | {_fmt(offline['val']['target_corr'])} | "
        f"{_fmt(offline['val']['safe_target_corr'])} | "
        f"{_fmt(offline['val']['target_mae'])} | "
        f"{_fmt(offline['val']['target_rmse'])} | "
        f"{_fmt(offline['val']['balanced_accuracy'])} | "
        f"{_fmt(offline['val']['auroc_unsafe'])} | n/a |")
    lines.append(
        f"| actor1 states (pooled) | {_fmt(pooled.get('target_corr'))} | "
        f"{_fmt(pooled.get('safe_target_corr'))} | "
        f"{_fmt(pooled.get('target_mae'))} | "
        f"{_fmt(pooled.get('target_rmse'))} | "
        f"{_fmt((pooled.get('binary_at_zero') or {}).get('balanced_accuracy'))} | "
        f"{_fmt(pooled.get('auroc_unsafe'))} | "
        f"{_fmt(pooled.get('future_collision_auroc'))} |")
    lines.append('')
    lines.append(
        f"Target std: offline test "
        f"{_fmt(offline['test'].get('target_std'))}, actor1 pooled "
        f"{_fmt(pooled.get('target_std'))}.  A narrow actor1 target range "
        f"attenuates correlation; target MAE/RMSE are the primary calibration "
        f"comparison.")
    lines.append('')
    lines.append('## Calibration in risk regions (pooled)\n')
    lines.append('| region | rows | h mean | target mean | MAE | corr |')
    lines.append('|---|---:|---:|---:|---:|---:|')
    regions = pooled.get('risk_regions', {})
    for key, label in (
            ('target_lt_0p5', 'target < 0.5'),
            ('h_lt_high', 'h < gate high'),
            ('future_collision', 'future collision'),
            ('current_unsafe', 'current unsafe label')):
        row = regions.get(key, {})
        lines.append(
            f"| {label} | {row.get('count', 0)} | "
            f"{_fmt(row.get('h_mean'))} | {_fmt(row.get('target_mean'))} | "
            f"{_fmt(row.get('mae'))} | {_fmt(row.get('corr'))} |")
    lines.append('')
    lines.append('## Decision evidence\n')
    lines.append(f"- classification: **{decision['label']}**")
    lines.append(f"- new collision onsets: {decision['collision_events']}")
    lines.append(f"- fraction h_cur >= high: "
                 f"{_fmt(decision['frac_h_cur_above_high'],3)}")
    lines.append(f"- fraction gate_next open: "
                 f"{_fmt(decision['frac_gate_open'],3)}")
    lines.append(f"- fraction h_cur <= 0: "
                 f"{_fmt(decision['frac_h_cur_below_zero'],3)}")
    lines.append(f"- fraction gate_next closed: "
                 f"{_fmt(decision['frac_gate_closed'],3)}")
    lines.append(f"- target-corr drop vs offline test: "
                 f"{_fmt(decision['target_corr_drop_vs_test'],3)}")
    lines.append(f"- safe-corr drop vs offline test: "
                 f"{_fmt(decision['safe_target_corr_drop_vs_test'],3)}")
    lines.append(f"- target MAE on actor1 states: "
                 f"{_fmt(decision['target_mae'])} "
                 f"({_fmt(decision['target_mae_ratio_vs_test'],2)}x offline "
                 f"test MAE)")
    lines.append(f"- range restriction (target std ratio): "
                 f"{_fmt((pooled.get('target_std') or 0.0) / offline['test']['target_std'],2)}")
    risk = pooled.get('risk_regions', {}).get('future_collision', {})
    lines.append(f"- future-collision rows: {risk.get('count', 0)}, "
                 f"h mean {_fmt(risk.get('h_mean'))}, "
                 f"target mean {_fmt(risk.get('target_mean'))}, "
                 f"MAE {_fmt(risk.get('mae'))}")
    lines.append(f"- pre-collision lag5 h >= high: "
                 f"{_fmt(decision['pre_collision_lag5_frac_above_high'],3)}")
    lines.append(f"- pre-collision lag5 h > 0: "
                 f"{_fmt(decision['pre_collision_lag5_frac_above_zero'],3)}")
    lines.append('')
    lines.append(f"Rule: {decision['rule']}\n")
    return '\n'.join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dir', default='outputs/v6/independent/nav_h2_audit')
    ap.add_argument('--out-prefix', default='summary')
    args = ap.parse_args()

    root = Path(args.dir)
    if not root.is_absolute():
        root = ROOT / root
    json_paths = sorted(root.glob('seed*_100ep.json'))
    if not json_paths:
        raise SystemExit(f'no seed*_100ep.json under {root}')
    per_seed = {}
    npz_paths = []
    references = {}
    gate_high = None
    for path in json_paths:
        payload = _load(path)
        seed = int(payload['training_seed'])
        if gate_high is None:
            gate_high = float(payload.get('protocol', {}).get(
                'gate_high', 0.26726993918418884))
        per_seed[seed] = _seed_rows(payload)
        references[seed] = payload.get('verification')
        npz = Path(payload.get('data_npz', path.with_suffix('.npz')))
        if npz.exists():
            per_seed[seed].update(_npz_metrics(npz, gate_high))
            npz_paths.append(npz)
    pooled = _pool_rows(npz_paths, gate_high if gate_high is not None else 0.0)
    offline = _offline_reference()
    decision = _classify(per_seed, pooled, offline)
    summary = dict(
        schema='v6.nav.h2_audit.summary.v1',
        audit_dir=str(root),
        per_seed={str(k): v for k, v in per_seed.items()},
        pooled=pooled,
        offline_reference=offline,
        decision=decision,
        verification=references,
    )
    out_json = root / f'{args.out_prefix}.json'
    out_md = root / f'{args.out_prefix}.md'
    out_json.write_text(json.dumps(summary, indent=2) + '\n')
    markdown = _markdown(summary)
    out_md.write_text(markdown + '\n')
    print(markdown)
    print(f'saved: {out_json}')
    print(f'saved: {out_md}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
