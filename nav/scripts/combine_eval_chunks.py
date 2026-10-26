"""Combine equal-sized evaluation chunks into one seed-level eval JSON."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pattern', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    paths = sorted(glob.glob(args.pattern))
    if not paths:
        raise SystemExit(f'no files matched: {args.pattern}')
    rows = [json.loads(Path(path).read_text()) for path in paths]
    keys = sorted(rows[0]['metrics'])
    metrics = {key: float(np.mean([row['metrics'][key] for row in rows]))
               for key in keys}
    payload = dict(
        checkpoint=rows[0].get('checkpoint'),
        steps=rows[0].get('steps'),
        episodes=sum(int(row.get('episodes', 0)) for row in rows),
        training_seed=rows[0].get('training_seed'),
        eval_seed=min(row.get('eval_seed', row.get('seed')) for row in rows),
        seed=min(row.get('seed', row.get('eval_seed')) for row in rows),
        chunks=paths,
        metrics=metrics,
    )
    text = json.dumps(payload, indent=2)
    print(text)
    Path(args.out).write_text(text + '\n')


if __name__ == '__main__':
    main()
