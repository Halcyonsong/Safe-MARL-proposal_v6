"""Aggregate V5 evaluation JSON files (mean ± std over training seeds)."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pattern', required=True,
                        help='glob for eval_*.json files')
    parser.add_argument('--out')
    args = parser.parse_args()
    paths = sorted(glob.glob(args.pattern))
    if not paths:
        raise SystemExit(f'no files matched: {args.pattern}')
    rows = [json.loads(Path(path).read_text()) for path in paths]
    keys = sorted(rows[0]['metrics'])
    aggregate = {}
    for key in keys:
        values = np.array([row['metrics'][key] for row in rows], dtype=float)
        aggregate[key] = dict(mean=float(values.mean()),
                              std=float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                              values=values.tolist())
    payload = dict(
        files=paths,
        training_seeds=[row.get('training_seed') for row in rows],
        eval_seeds=[row.get('eval_seed', row.get('seed')) for row in rows],
        episodes=[row.get('episodes') for row in rows],
        metrics=aggregate)
    text = json.dumps(payload, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + '\n')


if __name__ == '__main__':
    main()
