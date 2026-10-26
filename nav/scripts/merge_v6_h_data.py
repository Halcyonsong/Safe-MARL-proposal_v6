"""Merge random-rollout h data with near-boundary augmented h data.

Graph arrays are preserved so the navigation h can be trained with the V6
full-graph safety-network schema.  Augmented rows have no next graph and are
masked out of h_dot; they only contribute the safe/unsafe classification term.
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

GRAPH_KEYS = (
    'nodes', 'states', 'edges', 'receivers', 'senders', 'node_type',
    'n_node', 'n_edge',
)


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', default='outputs/v6/independent/data')
    ap.add_argument('--out', default='outputs/v6/independent/data/train_merged.npz')
    args = ap.parse_args()

    data_dir = assert_independent_path(
        _resolve(args.data_dir), kind='V6 h data directory')
    out = assert_independent_path(
        _resolve(args.out), kind='V6 h merged data output')
    with np.load(data_dir / 'train_data.npz') as base:
        obs = base['obs'].reshape(-1, base['obs'].shape[-1])
        labels_2d = base['labels'].reshape(base['labels'].shape[0], -1)
        labels = labels_2d.reshape(-1)
        nxt = (base['next_obs'].reshape(-1, base['next_obs'].shape[-1])
               if 'next_obs' in base.files else None)
        graph = {key: base['graph_' + key] for key in GRAPH_KEYS}
        next_graph = {
            key: base['next_graph_' + key] for key in GRAPH_KEYS}
        if 'graph_hdot_mask' in base.files:
            base_hdot = np.asarray(
                base['graph_hdot_mask']).reshape(-1).astype(np.float32)
        elif 'hdot_mask' in base.files:
            base_hdot = np.asarray(
                base['hdot_mask']).reshape(-1).astype(np.float32)
        else:
            base_hdot = np.ones(len(obs), dtype=np.float32)
    with np.load(data_dir / 'train_aug.npz') as aug:
        aug_obs = aug['obs'].reshape(-1, aug['obs'].shape[-1])
        aug_labels_2d = aug['labels'].reshape(aug['labels'].shape[0], -1)
        aug_labels = aug_labels_2d.reshape(-1)
        aug_graph = {key: aug['graph_' + key] for key in GRAPH_KEYS}

    # ``obs``/``labels`` are flattened per agent for the MLP view, while the
    # graph arrays and hdot masks are per transition row.  Keep the row count
    # separate from the flattened row count.
    n_base = int(labels_2d.shape[0])
    n_aug = int(aug_labels_2d.shape[0])
    obs = np.concatenate([obs, aug_obs], axis=0)
    labels = np.concatenate([labels, aug_labels], axis=0)
    if nxt is None:
        raise ValueError(
            'base h data is missing next_obs; h_dot requires transitions')
    next_obs = np.concatenate([nxt, aug_obs], axis=0)
    if len(base_hdot) != n_base:
        raise ValueError(
            f'base hdot mask has {len(base_hdot)} rows, expected {n_base}')
    hdot_mask = np.concatenate(
        [base_hdot, np.zeros(n_aug, dtype=np.float32)], axis=0)
    graph_labels = np.concatenate([labels_2d, aug_labels_2d], axis=0)
    graph_hdot_mask = np.concatenate(
        [base_hdot, np.zeros(n_aug, dtype=np.float32)], axis=0)

    merged_graph = {}
    for key in GRAPH_KEYS:
        merged_graph['graph_' + key] = np.concatenate(
            [graph[key], aug_graph[key]], axis=0)
        merged_graph['next_graph_' + key] = np.concatenate(
            [next_graph[key], aug_graph[key]], axis=0)

    out.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(obs=obs, labels=labels, next_obs=next_obs,
                   hdot_mask=hdot_mask, graph_labels=graph_labels,
                   graph_hdot_mask=graph_hdot_mask)
    payload.update(merged_graph)
    np.savez_compressed(out, **payload)
    summary = dict(
        schema='v6.h_merge.v1',
        out=str(out),
        base_data=str(data_dir / 'train_data.npz'),
        aug_data=str(data_dir / 'train_aug.npz'),
        base_rows=int(n_base),
        aug_rows=int(n_aug),
        rows=int(n_base + n_aug),
        agent_rows=int(len(obs)),
        hdot_rows=int(hdot_mask.sum()),
        unsafe_rate=float(np.mean(labels < 0)),
    )
    (out.parent / 'merge_summary.json').write_text(
        json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
