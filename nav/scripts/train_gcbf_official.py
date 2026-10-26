"""Launch the vendored official GCBF/GCBF+ training script with import shims.

GCBF+ training needs the real jaxproxqp solver because it computes QP labels for
the actor. GCBF (v0) training does not call the QP and can run with the
import-only shim. This wrapper is intended for GCBF v0.

Example:
  python scripts/train_gcbf_official.py --algo gcbf --env DoubleIntegrator \
    -n 8 --area-size 4 --steps 1000 --log-dir ../outputs/v5/baselines/gcbf
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

NAV = Path(__file__).resolve().parents[1]
ROOT = NAV.parent
VENDOR = ROOT / 'third_party' / 'gcbfplus'
if str(NAV) not in sys.path:
    sys.path.insert(0, str(NAV))
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

from algos.baselines.gcbf_official import _ensure_importable


def main():
    _ensure_importable()
    sys.argv = [str(VENDOR / 'train.py'), *sys.argv[1:]]
    runpy.run_path(str(VENDOR / 'train.py'), run_name='__main__')


if __name__ == '__main__':
    main()
