#!/usr/bin/env python3
"""Merge C0 actor1 h data with the fixed random coverage dataset.

C0 tests whether retraining h on the final actor1 distribution closes the
long-horizon gap.  Both inputs must already carry the wall-free 32-step
future-min target and target gradients produced by ``collect_v6_h_data.py``
(actor1) or ``rebuild_v6_h_targets.py`` (random).

The merge ratio matches the repaired h1 training mix: 80 actor1 episodes
(20,480 timesteps) plus 80 random episodes (20,480 timesteps), with 10+10
validation and 10+10 test episodes.  This isolates the data-source change
rather than changing the data volume or the random/on-policy ratio.
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

from merge_v6_repair_data import _check, _concat, _load  # noqa: E402
from v6.provenance import assert_independent_path  # noqa: E402


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--actor1-dir', required=True)
    ap.add_argument('--random-dir', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    actor1 = assert_independent_path(
        _resolve(args.actor1_dir), kind='V6 C0 actor1 h data')
    random = assert_independent_path(
        _resolve(args.random_dir), kind='V6 C0 random h data')
    out = assert_independent_path(
        _resolve(args.out), kind='V6 C0 merged h data')
    out.mkdir(parents=True, exist_ok=True)

    actor1_summary = json.loads((actor1 / 'summary.json').read_text())
    random_summary = json.loads((random / 'summary.json').read_text())
    summary = dict(
        schema='v6.c0_h2_data_merge.v1',
        actor1=str(actor1),
        random=str(random),
        out=str(out),
        actor1_behavior=actor1_summary.get('behavior'),
        actor1_h=actor1_summary.get('behavior_h'),
        actor1_stochastic=actor1_summary.get('stochastic_actions'),
        actor1_seed=actor1_summary.get('seed'),
        random_source=random_summary.get('source'),
        splits={},
    )
    for split in ('train', 'val', 'test'):
        actor1_data = _load(actor1 / f'{split}_data.npz')
        random_data = _load(random / f'{split}_data.npz')
        _check(actor1_data, f'actor1/{split}')
        _check(random_data, f'random/{split}')
        merged = _concat([(actor1_data, f'actor1/{split}'),
                          (random_data, f'random/{split}')])
        # ``_concat`` only assigns data_source to the second and later parts;
        # write the explicit 0=actor1 / 1=random marker for every row.
        n_actor1 = int(actor1_data['graph_labels'].shape[0])
        n_random = int(random_data['graph_labels'].shape[0])
        merged['data_source'] = np.concatenate([
            np.zeros(n_actor1, dtype=np.int8),
            np.ones(n_random, dtype=np.int8),
        ])
        np.savez_compressed(out / f'{split}_data.npz', **merged)
        labels = merged['graph_labels']
        target = merged['graph_margin_targets']
        source = merged['data_source'].astype(bool)
        summary['splits'][split] = dict(
            rows=int(labels.shape[0]),
            actor1_rows=int((~source).sum()),
            random_rows=int(source.sum()),
            actor1_episodes=int(np.unique(
                merged['episode_ids'][~source]).size),
            random_episodes=int(np.unique(
                merged['episode_ids'][source]).size),
            unsafe_rate=float(np.mean(labels < 0)),
            ignored_rate=float(np.mean(labels == 0)),
            target_mean=float(np.mean(target)),
            target_std=float(np.std(target)),
            target_negative_rate=float(np.mean(target < 0)),
            actor1_target_mean=float(np.mean(target[~source])),
            actor1_target_negative_rate=float(
                np.mean(target[~source] < 0)),
        )
        print(json.dumps({split: summary['splits'][split]}), flush=True)

    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
