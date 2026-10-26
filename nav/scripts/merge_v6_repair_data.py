"""Merge rebuilt V6 navigation h datasets from two data sources.

Both sources must already contain the wall-free future-min targets and target
gradients produced by ``rebuild_v6_h_targets.py``.  This script concatenates
the common row fields, offsets episode ids so they remain unique, records the
source, and writes ``train/val/test_data.npz``.  It reads no task reward,
expert action, official h, QP label, or V5 artifact.
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

from v6.provenance import assert_independent_path  # noqa: E402

REQUIRED = (
    'obs', 'next_obs', 'labels', 'graph_labels', 'graph_labels_horizon',
    'graph_labels_instant', 'graph_hdot_mask', 'unsafe', 'safe_horizon',
    'episode_ids', 'steps',
    'graph_nodes', 'graph_states', 'graph_edges', 'graph_receivers',
    'graph_senders', 'graph_node_type', 'graph_n_node', 'graph_n_edge',
    'next_graph_nodes', 'next_graph_states', 'next_graph_edges',
    'next_graph_receivers', 'next_graph_senders', 'next_graph_node_type',
    'next_graph_n_node', 'next_graph_n_edge',
    'graph_margin_values', 'graph_future_margin', 'graph_margin_targets',
    'graph_margin_grads', 'graph_target_mask',
)


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _load(path: Path) -> dict:
    with np.load(path) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _check(data: dict, name: str) -> None:
    missing = [key for key in REQUIRED if key not in data]
    if missing:
        raise ValueError(f'{name} is missing fields: {missing}')
    rows = int(data['graph_labels'].shape[0])
    for key in REQUIRED:
        if data[key].shape[0] != rows:
            raise ValueError(
                f'{name}.{key} has {data[key].shape[0]} rows, expected {rows}')


def _concat(parts: list[tuple[dict, str]]) -> dict:
    for data, name in parts:
        _check(data, name)
    out: dict[str, np.ndarray] = {}
    key_order = []
    for data, _ in parts:
        for key in data:
            if key not in key_order and all(key in d for d, _ in parts):
                key_order.append(key)
    episode_offset = 0
    for i, (data, name) in enumerate(parts):
        if i == 0:
            episode_offset = int(np.max(data['episode_ids'])) + 1
        else:
            local = np.asarray(data['episode_ids'], dtype=np.int64)
            data = dict(data)
            data['episode_ids'] = (local + episode_offset).astype(
                data['episode_ids'].dtype)
            data['data_source'] = np.full(
                (len(local),), i, dtype=np.int8)
            parts[i] = (data, name)
            episode_offset += int(np.max(local)) + 1
    for key in key_order:
        out[key] = np.concatenate([d[key] for d, _ in parts], axis=0)
    if 'data_source' not in out:
        out['data_source'] = np.zeros(
            (int(out['graph_labels'].shape[0]),), dtype=np.int8)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stageb-dir', required=True)
    ap.add_argument('--random-dir', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    stageb = assert_independent_path(
        _resolve(args.stageb_dir), kind='V6 Stage-B rebuilt h data')
    random = assert_independent_path(
        _resolve(args.random_dir), kind='V6 random rebuilt h data')
    out = assert_independent_path(
        _resolve(args.out), kind='V6 merged repair h data')
    out.mkdir(parents=True, exist_ok=True)

    summary = dict(schema='v6.h_repair_merge.v1', stageb=str(stageb),
                   random=str(random), out=str(out), splits={})
    for split in ('train', 'val', 'test'):
        stageb_data = _load(stageb / f'{split}_data.npz')
        random_data = _load(random / f'{split}_data.npz')
        merged = _concat([(stageb_data, f'stageb/{split}'),
                          (random_data, f'random/{split}')])
        np.savez_compressed(out / f'{split}_data.npz', **merged)
        labels = merged['graph_labels']
        target = merged['graph_margin_targets']
        margins = merged['graph_margin_values']
        summary['splits'][split] = dict(
            rows=int(labels.shape[0]),
            stageb_rows=int(stageb_data['graph_labels'].shape[0]),
            random_rows=int(random_data['graph_labels'].shape[0]),
            unsafe_rate=float(np.mean(labels < 0)),
            ignored_rate=float(np.mean(labels == 0)),
            safe_rate=float(np.mean(labels > 0)),
            target_mean=float(np.mean(target)),
            target_std=float(np.std(target)),
            target_negative_rate=float(np.mean(target < 0)),
            target_corr_current=float(np.corrcoef(
                target.reshape(-1), margins.reshape(-1))[0, 1]),
            gradient_abs_mean=float(np.mean(np.abs(
                merged['graph_margin_grads']))),
            stageb_episode_ids=int(np.unique(
                stageb_data['episode_ids']).size),
            random_episode_ids=int(np.unique(
                random_data['episode_ids']).size),
            merged_episode_ids=int(np.unique(
                merged['episode_ids']).size),
        )
        print(json.dumps({split: summary['splits'][split]}), flush=True)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
