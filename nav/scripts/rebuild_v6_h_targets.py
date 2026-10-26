"""Rebuild V6 navigation h targets on an existing rollout dataset.

The original Stage-B collector stored ``graph_margin_targets`` but not the
target state gradient, and it included a wall margin that the graph h cannot
identify (the serialized graph has no absolute-position feature).  This script
recomputes, from the stored observations only:

* the current geometric safety margin,
* the future-min margin over the same forward horizon,
* ``tanh(future_min / scale)`` with unsafe/ignored rows forced negative,
* the diagonal target gradient ``d tanh(current margin / scale) / d state_i``
  for h-gradient alignment.

It does not read task reward, expert actions, official h, QP labels, or V5
artifacts.  Existing labels and graph arrays are preserved.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

NAV = Path(__file__).resolve().parents[1]
ROOT = NAV.parent
if str(NAV) not in sys.path:
    sys.path.insert(0, str(NAV))

from v6.provenance import assert_independent_path  # noqa: E402
from v6.safety import (  # noqa: E402
    dense_margin_state_gradients, dense_margin_values,
    forward_margin_sequence, horizon_margin_targets)


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _margin_array(obs: np.ndarray, cfg: dict, include_wall: bool) -> np.ndarray:
    n_rows, n_agents, _ = obs.shape
    return dense_margin_values(
        obs.reshape(-1, obs.shape[-1]),
        max_neighbors=int(cfg['max_neighbors']),
        n_rays=int(cfg['n_rays']),
        area_size=float(cfg['area_size']),
        comm_radius=float(cfg.get('comm_radius', 0.5)),
        include_wall=include_wall,
    ).reshape(n_rows, n_agents)


def _gradient_array(obs: np.ndarray, cfg: dict, include_wall: bool,
                    scale: float) -> np.ndarray:
    return dense_margin_state_gradients(
        obs,
        max_neighbors=int(cfg['max_neighbors']),
        n_rays=int(cfg['n_rays']),
        area_size=float(cfg['area_size']),
        comm_radius=float(cfg.get('comm_radius', 0.5)),
        scale=float(scale),
        include_wall=include_wall,
    )


def _future_targets(data: dict, margins: np.ndarray, labels: np.ndarray,
                    cfg: dict, horizon: int, scale: float,
                    include_wall: bool) -> tuple[np.ndarray, np.ndarray,
                                                 np.ndarray]:
    n_rows, n_agents = margins.shape
    target = np.tanh(margins / float(scale)).astype(np.float32)
    risk = margins.astype(np.float32)
    target_mask = np.ones((n_rows,), dtype=np.float32)
    if 'episode_ids' not in data or 'next_obs' not in data:
        return target, risk, target_mask

    episode_ids = np.asarray(data['episode_ids']).reshape(-1)
    steps = (np.asarray(data['steps']).reshape(-1)
             if 'steps' in data else np.arange(n_rows))
    next_obs = np.asarray(data['next_obs'], dtype=np.float32)
    for episode in np.unique(episode_ids):
        idx = np.where(episode_ids == episode)[0]
        idx = idx[np.argsort(steps[idx], kind='mergesort')]
        terminal = _margin_array(
            next_obs[idx[-1]][None], cfg, include_wall)[0]
        sequence = forward_margin_sequence(margins[idx], terminal)
        seq_target, seq_risk = horizon_margin_targets(
            sequence, scale=float(scale), horizon=int(horizon))
        tgt = seq_target[:len(idx)].copy()
        rsk = seq_risk[:len(idx)].copy()
        # Ignored rows (unsafe somewhere in the forward window) are exactly
        # the predictive rows.  They must not carry a positive target.
        invalid = labels[idx] <= 0.0
        tgt[invalid] = np.minimum(tgt[invalid], -0.1)
        target[idx] = tgt
        risk[idx] = rsk
    return target.astype(np.float32), risk.astype(np.float32), target_mask


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--config',
                    default='configs/v6/v6_nav_independent.yaml')
    ap.add_argument('--horizon', type=int, default=32)
    ap.add_argument('--margin-scale', type=float, default=0.2)
    ap.add_argument('--include-wall', action=argparse.BooleanOptionalAction,
                    default=False,
                    help='include the outer-wall margin (default: off; the '
                         'graph h cannot observe absolute position)')
    args = ap.parse_args()

    data_dir = assert_independent_path(
        _resolve(args.data_dir), kind='V6 h target source')
    out_dir = assert_independent_path(
        _resolve(args.out), kind='V6 h target output')
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = NAV / cfg_path
    config = yaml.safe_load(cfg_path.read_text())
    cfg = dict(config['env'])
    cfg['comm_radius'] = 0.5
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = dict(
        schema='v6.h_targets.rebuilt.v1',
        source=str(data_dir), out=str(out_dir),
        horizon=int(args.horizon), margin_scale=float(args.margin_scale),
        include_wall=bool(args.include_wall), splits={})
    for split in ('train', 'val', 'test'):
        path = data_dir / f'{split}_data.npz'
        if not path.exists():
            continue
        with np.load(path) as raw:
            data = {key: raw[key] for key in raw.files}
        obs = np.asarray(data['obs'], dtype=np.float32)
        labels = np.asarray(data['graph_labels'], dtype=np.float32)
        margins = _margin_array(obs, cfg, bool(args.include_wall))
        grads = _gradient_array(
            obs, cfg, bool(args.include_wall), float(args.margin_scale))
        target, risk, target_mask = _future_targets(
            data, margins, labels, cfg, int(args.horizon),
            float(args.margin_scale), bool(args.include_wall))
        data['graph_margin_values'] = margins
        data['graph_future_margin'] = risk
        data['graph_margin_targets'] = target
        data['graph_margin_grads'] = grads
        data['graph_target_mask'] = target_mask
        np.savez_compressed(out_dir / f'{split}_data.npz', **data)
        summary['splits'][split] = dict(
            rows=int(len(labels)),
            unsafe_rate=float(np.mean(labels < 0)),
            ignored_rate=float(np.mean(labels == 0)),
            target_negative_rate=float(np.mean(target < 0)),
            target_mean=float(np.mean(target)),
            margin_mean=float(np.mean(margins)),
            gradient_abs_mean=float(np.mean(np.abs(grads))),
            future_margin_mean=float(np.mean(risk)),
            future_margin_min=float(np.min(risk)),
            future_margin_max=float(np.max(risk)),
        )
        print(json.dumps(summary['splits'][split]), flush=True)
    (out_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
