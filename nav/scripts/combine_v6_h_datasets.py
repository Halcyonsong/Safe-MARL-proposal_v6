"""Combine a base h dataset with policy-distribution h data.

The base directory is the random/near-boundary graph dataset; the extra
directory contains graphs collected by the V5 behavior policy.  Only the
extra train split is used for training, while val/test keep both sources so
the classifier is checked on both distributions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _load(path: Path) -> dict:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _concat(base: dict, extra: dict, *, split: str) -> dict:
    n_agents = int(base['graph_labels'].shape[1])
    out = {}
    flat_keys = {'obs', 'labels', 'next_obs', 'hdot_mask'}
    graph_keys = [key for key in base if key.startswith('graph_')
                  or key.startswith('next_graph_')]
    graph_keys += [key for key in extra if (key.startswith('graph_')
                  or key.startswith('next_graph_')) and key not in graph_keys]
    for key in flat_keys | set(graph_keys):
        if key not in base:
            continue
        b = base[key]
        e = extra.get(key)
        if e is None:
            if key in ('hdot_mask',):
                e = np.ones(int(extra['graph_labels'].shape[0]) * n_agents,
                            dtype=np.float32)
            elif key == 'graph_hdot_mask':
                e = np.ones(int(extra['graph_labels'].shape[0]),
                            dtype=np.float32)
            else:
                continue
        if key in ('obs', 'next_obs'):
            b = np.reshape(b, (-1, b.shape[-1]))
            e = np.reshape(e, (-1, e.shape[-1]))
            out[key] = np.concatenate([b, e], axis=0)
        elif key == 'labels':
            b_flat = np.reshape(b, (-1,))
            e_flat = np.reshape(e, (-1,))
            out[key] = np.concatenate([b_flat, e_flat], axis=0)
        else:
            out[key] = np.concatenate([b, e], axis=0)
    out['graph_labels'] = np.concatenate(
        [base['graph_labels'], extra['graph_labels']], axis=0)
    out['graph_hdot_mask'] = np.concatenate(
        [base['graph_hdot_mask'], extra['graph_hdot_mask']], axis=0)
    n_rows = int(out['graph_labels'].shape[0])
    if 'hdot_mask' not in out:
        out['hdot_mask'] = np.ones(n_rows * n_agents, dtype=np.float32)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base-dir', required=True)
    ap.add_argument('--extra-dir', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    base = Path(args.base_dir)
    extra = Path(args.extra_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train = _concat(
        _load(base / 'train_merged.npz'),
        _load(extra / 'train_data.npz'), split='train')
    val = _concat(
        _load(base / 'val_data.npz'),
        _load(extra / 'val_data.npz'), split='val')
    test = _concat(
        _load(base / 'test_data.npz'),
        _load(extra / 'test_data.npz'), split='test')
    for name, data in [('train', train), ('val', val), ('test', test)]:
        np.savez_compressed(out / f'{name}_data.npz', **data)
        print(name, int(data['graph_labels'].shape[0]), 'graphs',
              int(data['labels'].size), 'agent rows')


if __name__ == '__main__':
    main()
