"""Smoke test for the fixed official GCBF+ h bridge."""

import sys
from pathlib import Path

import numpy as np
import pytest

NAV = Path(__file__).resolve().parents[1]
ROOT = NAV.parent
if str(NAV) not in sys.path:
    sys.path.insert(0, str(NAV))

from envs.gcbfplus import OfficialDoubleIntegrator
from algos.safe_policy.fixed_h import OfficialFixedH

CKPT = (ROOT / 'third_party/gcbfplus/pretrained/DoubleIntegrator/'
        'gcbf+/models/1000/cbf.pkl')


@pytest.mark.skipif(not CKPT.exists(),
                    reason='official GCBF+ cbf.pkl is not shipped with this repo')
def test_fixed_official_h_value_and_grad():
    env = OfficialDoubleIntegrator(num_agents=8, area_size=4, max_step=256,
                                   num_obstacles=8, n_rays=32, seed=0)
    env.reset(seed=0)
    ckpt = CKPT
    hnet = OfficialFixedH(ckpt, env, n_agents=8, gnn_layers=1)
    h, grad = hnet.value_and_local_grad()
    h = np.asarray(h)
    grad = np.asarray(grad)
    assert h.shape == (8,)
    assert grad.shape == (8, 4)
    assert np.isfinite(h).all() and np.isfinite(grad).all()
    assert np.abs(h).max() <= 1.0 + 1e-5
    assert np.abs(grad).max() > 0
