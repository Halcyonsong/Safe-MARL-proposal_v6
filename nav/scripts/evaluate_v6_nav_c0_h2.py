#!/usr/bin/env python3
"""Evaluate h1 vs C0 h2 on held-out actor1 states.

This is an offline, read-only calibration screen.  It does not run the
environment, train an actor, or select checkpoints on the formal 100-episode
test set.  Both h checkpoints are evaluated on exactly the same actor1 rows
from the C0 merged test split (``data_source == 0``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

NAV = Path(__file__).resolve().parents[1]
ROOT = NAV.parent
if str(NAV) not in sys.path:
    sys.path.insert(0, str(NAV))
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from calibrate_v6_nav_gate import _make_graphs, _predict  # noqa: E402
from train_v6_official_h import (  # noqa: E402
    _metrics, _target_metrics)
from v6.official_h import (  # noqa: E402
    build_official_net, load_official_h_checkpoint)
from v6.provenance import assert_independent_path  # noqa: E402


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _load_h(path: Path):
    ckpt = load_official_h_checkpoint(path)
    if ckpt.get('schema') != 'v6-official-cbf-h':
        raise ValueError(f'not a V6 h checkpoint: {path}')
    net = build_official_net(int(ckpt.get('gnn_layers', 1)))
    return ckpt, net


def _predict_shifted(ckpt, net, data, n_agents: int) -> tuple[np.ndarray,
                                                              np.ndarray]:
    raw = _predict(ckpt['params'], net, data, 'graph_', n_agents)
    raw_next = _predict(ckpt['params'], net, data, 'next_graph_', n_agents)
    offset = float(ckpt.get('h_offset', 0.0))
    return raw + offset, raw_next + offset


def _region(h: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return dict(count=0)
    hh, tt = h[mask], target[mask]
    corr = 0.0
    if np.std(hh) > 1e-12 and np.std(tt) > 1e-12:
        corr = float(np.corrcoef(hh, tt)[0, 1])
    return dict(
        count=int(mask.sum()),
        h_mean=float(np.mean(hh)),
        target_mean=float(np.mean(tt)),
        mae=float(np.mean(np.abs(hh - tt))),
        corr=corr,
    )


def _episode_slices(episode_ids: np.ndarray, steps: np.ndarray):
    for episode in np.unique(episode_ids):
        idx = np.where(episode_ids == episode)[0]
        order = np.argsort(steps[idx], kind='mergesort')
        yield episode, idx[order]


def _time_to_collision(margin: np.ndarray) -> np.ndarray:
    """Distance to the next geometric collision (margin <= 0) per row.

    The recurrence must read the *next* row's distance.  Adding one to the
    current row's initial sentinel (10,000) never propagates the collision
    backwards and silently empties every 1..32-step bin.
    """
    n_rows, n_agents = margin.shape
    dist = np.full((n_rows, n_agents), 10_000, dtype=np.int32)
    for t in range(n_rows - 1, -1, -1):
        coll = margin[t] <= 0
        if t == n_rows - 1:
            dist[t] = np.where(coll, 0, 10_000)
        else:
            dist[t] = np.where(
                coll, 0, np.minimum(dist[t + 1] + 1, 10_000))
    return dist


def _lead_bins(h: np.ndarray, target: np.ndarray, margin: np.ndarray,
               gate_high: float, episode_ids: np.ndarray,
               steps: np.ndarray) -> dict:
    bins = ((0, 0), (1, 1), (2, 4), (5, 8), (9, 16), (17, 32),
            (33, 10_000))
    collected = {key: [] for key in bins}
    for _episode, idx in _episode_slices(episode_ids, steps):
        hh = h[idx]
        tt = target[idx]
        dist = _time_to_collision(margin[idx])
        for lo, hi in bins:
            mask = (dist >= lo) & (dist <= hi)
            if mask.any():
                collected[(lo, hi)].append((hh[mask], tt[mask]))
    out = {}
    for key, parts in collected.items():
        if not parts:
            out[str(key)] = dict(count=0)
            continue
        hh = np.concatenate([x[0] for x in parts])
        tt = np.concatenate([x[1] for x in parts])
        out[str(key)] = dict(
            count=int(hh.size),
            h_mean=float(np.mean(hh)),
            target_mean=float(np.mean(tt)),
            mae=float(np.mean(np.abs(hh - tt))),
            frac_h_neg=float(np.mean(hh < 0.0)),
            frac_h_below_gate=float(np.mean(hh < gate_high)),
        )
    return out


def _lead_time(h: np.ndarray, margin: np.ndarray, gate_high: float,
               episode_ids: np.ndarray, steps: np.ndarray) -> dict:
    last_high = []
    last_pos = []
    for _episode, idx in _episode_slices(episode_ids, steps):
        hh = h[idx]
        coll = margin[idx] <= 0
        for a in range(coll.shape[1]):
            prev = False
            for t in range(coll.shape[0]):
                onset = bool(coll[t, a]) and not prev
                prev = bool(coll[t, a])
                if not onset:
                    continue
                high = np.where(hh[:t + 1, a] >= gate_high)[0]
                pos = np.where(hh[:t + 1, a] > 0.0)[0]
                if high.size:
                    last_high.append(int(t - high[-1]))
                if pos.size:
                    last_pos.append(int(t - pos[-1]))
    def stats(values):
        if not values:
            return dict(count=0)
        arr = np.asarray(values, dtype=np.float64)
        return dict(
            count=int(arr.size),
            mean=float(np.mean(arr)),
            median=float(np.median(arr)),
            q25=float(np.quantile(arr, 0.25)),
            q75=float(np.quantile(arr, 0.75)),
            min=int(arr.min()),
            max=int(arr.max()),
            frac_ge_5=float(np.mean(arr >= 5)),
            frac_ge_10=float(np.mean(arr >= 10)),
            frac_ge_17=float(np.mean(arr >= 17)),
        )
    return dict(last_h_ge_gate=stats(last_high),
                last_h_positive=stats(last_pos))


def _metrics_for(h: np.ndarray, target: np.ndarray, labels: np.ndarray,
                 target_mask: np.ndarray, gate_high: float,
                 margin: np.ndarray, future_margin: np.ndarray,
                 episode_ids: np.ndarray, steps: np.ndarray) -> dict:
    target_mask = np.asarray(target_mask, dtype=bool)
    if (target_mask.ndim == 1
            and target_mask.size == target.shape[0]):
        target_mask = np.broadcast_to(
            target_mask[:, None], target.shape)
    out = _target_metrics(h, target, labels, target_mask)
    out['target_std'] = float(np.std(target[target_mask]))
    out.update(_metrics(h, labels, 0.0))
    out['risk_regions'] = dict(
        target_lt_0p5=_region(h, target, target < 0.5),
        h_lt_gate=_region(h, target, h < gate_high),
        future_collision=_region(h, target, future_margin <= 0),
        current_collision=_region(h, target, margin <= 0),
    )
    out['lead_bins'] = _lead_bins(
        h, target, margin, gate_high, episode_ids, steps)
    out['warning_lead'] = _lead_time(
        h, margin, gate_high, episode_ids, steps)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', required=True,
                    help='C0 merged test_data.npz')
    ap.add_argument('--h1', required=True)
    ap.add_argument('--h1-calibration', required=True)
    ap.add_argument('--h2', required=True)
    ap.add_argument('--h2-calibration', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    data_path = assert_independent_path(
        _resolve(args.data), kind='V6 C0 evaluation data')
    h1_path = assert_independent_path(_resolve(args.h1), kind='V6 C0 h1')
    h2_path = assert_independent_path(_resolve(args.h2), kind='V6 C0 h2')
    h1_cal = json.loads(_resolve(args.h1_calibration).read_text())
    h2_cal = json.loads(_resolve(args.h2_calibration).read_text())
    h1_high = float(h1_cal['recommended_high'])
    h2_high = float(h2_cal['recommended_high'])

    with np.load(data_path) as raw:
        data = {key: np.asarray(raw[key]) for key in raw.files}
    if 'data_source' in data:
        mask = data['data_source'] == 0
        data = {key: value[mask] for key, value in data.items()}
    if data['graph_labels'].shape[0] == 0:
        raise ValueError('no actor1 rows in the evaluation data')

    labels = np.asarray(data['graph_labels'], dtype=np.float32)
    target = np.asarray(data['graph_margin_targets'], dtype=np.float32)
    target_mask = (np.asarray(data['graph_target_mask'], dtype=np.float32)
                   if 'graph_target_mask' in data
                   else np.ones_like(target, dtype=np.float32))
    margin = np.asarray(data['graph_margin_values'], dtype=np.float32)
    future_margin = np.asarray(
        data['graph_future_margin'], dtype=np.float32)
    episode_ids = np.asarray(data['episode_ids'])
    steps = np.asarray(data['steps'])
    n_agents = int(labels.shape[1])

    h1_ckpt, h1_net = _load_h(h1_path)
    h2_ckpt, h2_net = _load_h(h2_path)
    h1, _ = _predict_shifted(h1_ckpt, h1_net, data, n_agents)
    h2, _ = _predict_shifted(h2_ckpt, h2_net, data, n_agents)

    common = dict(
        target=target, labels=labels, target_mask=target_mask,
        margin=margin, future_margin=future_margin,
        episode_ids=episode_ids, steps=steps)
    m1 = _metrics_for(h1, gate_high=h1_high, **common)
    m2 = _metrics_for(h2, gate_high=h2_high, **common)

    b1 = m1['lead_bins']['(17, 32)']
    b2 = m2['lead_bins']['(17, 32)']
    l1 = m1['warning_lead']['last_h_ge_gate'].get('median')
    l2 = m2['warning_lead']['last_h_ge_gate'].get('median')
    primary_reduction = (1.0 - b2['mae'] / b1['mae']
                         if b1.get('mae') and b2.get('mae') else None)
    lead_delta = (float(l2 - l1) if l1 is not None and l2 is not None
                  else None)
    checks = dict(
        primary_17_32_mae_reduction_ge_20pct=(
            primary_reduction is not None and primary_reduction >= 0.20),
        warning_lead_increase_ge_2_steps=(
            lead_delta is not None and lead_delta >= 2.0),
        global_mae_not_worse_by_0p01=(
            m2['target_mae'] <= m1['target_mae'] + 0.01),
        safe_corr_not_worse_by_0p02=(
            m2['target_corr_safe'] >= m1['target_corr_safe'] - 0.02),
        future_collision_auroc_not_worse_by_0p02=(
            _future_auroc(h2, future_margin)
            >= _future_auroc(h1, future_margin) - 0.02),
    )
    pilot_checks = dict(
        primary_17_32_mae_reduction_ge_15pct=(
            primary_reduction is not None and primary_reduction >= 0.15),
        warning_lead_increase_ge_1_step=(
            lead_delta is not None and lead_delta >= 1.0),
        global_mae_not_worse_by_0p02=(
            m2['target_mae'] <= m1['target_mae'] + 0.02),
        safe_corr_not_worse_by_0p03=(
            m2['target_corr_safe'] >= m1['target_corr_safe'] - 0.03),
    )
    result = dict(
        schema='v6.nav.c0_h2_eval.v1',
        data=str(data_path),
        h1=dict(path=str(h1_path), gate_high=h1_high, metrics=m1),
        h2=dict(path=str(h2_path), gate_high=h2_high, metrics=m2),
        delta=dict(
            target_mae=m2['target_mae'] - m1['target_mae'],
            target_rmse=m2['target_rmse'] - m1['target_rmse'],
            target_corr=m2['target_corr'] - m1['target_corr'],
            target_corr_safe=(m2['target_corr_safe']
                              - m1['target_corr_safe']),
            future_collision_mae=(m2['risk_regions']['future_collision']
                                  .get('mae', 0.0)
                                  - m1['risk_regions']['future_collision']
                                  .get('mae', 0.0)),
            target_lt_0p5_mae=(m2['risk_regions']['target_lt_0p5']
                               .get('mae', 0.0)
                               - m1['risk_regions']['target_lt_0p5']
                               .get('mae', 0.0)),
            lead_17_32_mae=(b2.get('mae', 0.0) - b1.get('mae', 0.0)),
            warning_lead_median=(lead_delta
                                 if lead_delta is not None else 0.0),
        ),
        pass_rule=dict(
            primary_17_32_mae_reduction=primary_reduction,
            warning_lead_delta_steps=lead_delta,
            checks=checks,
            passed=bool(all(checks.values())),
        ),
        pilot_rule=dict(
            primary_17_32_mae_reduction=primary_reduction,
            warning_lead_delta_steps=lead_delta,
            checks=pilot_checks,
            passed=bool(all(pilot_checks.values())),
        ),
    )
    out = _resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return 0


def _future_auroc(h: np.ndarray, future_collision: np.ndarray) -> float:
    score = -np.asarray(h, dtype=np.float64).reshape(-1)
    y = np.asarray(future_collision, dtype=bool).reshape(-1)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if not n_pos or not n_neg:
        return 0.5
    order = np.argsort(score, kind='mergesort')
    sorted_score = score[order]
    ranks = np.empty(len(score), dtype=np.float64)
    i = 0
    while i < len(score):
        j = i + 1
        while j < len(score) and sorted_score[j] == sorted_score[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + 1 + j)
        i = j
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0)
                 / (n_pos * n_neg))


if __name__ == '__main__':
    raise SystemExit(main())
