#!/usr/bin/env python
"""Thin launcher for the upstream MAPPO-Lag Ant trainer with CUDA enabled.

The vendor ``config.py`` defines ``--cuda`` as ``store_false`` with
``default=False``, so the upstream entry point always trains on CPU even when a
GPU is present.  The existing upstream Ant runs used a small launcher that
forces ``cuda=True`` after argument parsing; this file reproduces that launcher
inside the V6 tree without editing the vendor package.

No V5/V4 weights, checkpoints, or safety modules are loaded here.  Training
starts from random initialization.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V2 = Path(os.environ.get(
    'SAFE_MAMUJOCO_ROOT',
    str(ROOT.parent / 'Safe-MARL-proposal_v2'))).expanduser()
PKG = V2 / 'third_party/mappo_lag/MAPPO-Lagrangian'
TRAIN_PY = PKG / 'mappo_lagrangian/scripts/train/train_mujoco.py'


def _load_official():
    spec = importlib.util.spec_from_file_location(
        'v6_upstream_ant_train_mujoco', TRAIN_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(args):
    if not TRAIN_PY.exists():
        raise FileNotFoundError(TRAIN_PY)
    sys.path.insert(0, str(PKG))
    train_mod = _load_official()
    original_parse = train_mod.parse_args

    def parse_args_force_cuda(raw_args, parser):
        parsed = original_parse(raw_args, parser)
        if os.environ.get('ANT_FORCE_CUDA', '1') not in ('0', 'false', 'False'):
            parsed.cuda = True
        return parsed

    train_mod.parse_args = parse_args_force_cuda
    train_mod.main(args)


if __name__ == '__main__':
    main(sys.argv[1:])
