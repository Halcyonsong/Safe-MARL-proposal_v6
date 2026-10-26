"""Unit tests for V6 Ant geometry, margins, labels and h direction."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

NAV = Path(__file__).resolve().parents[1]
if str(NAV) not in sys.path:
    sys.path.insert(0, str(NAV))

from v6.ant_safety import (  # noqa: E402
    FixedAntSafetyProvider,
    finite_difference_grad,
    forward_safety_labels,
    geom_from_torso,
    horizon_risk_margins,
    horizon_risk_targets,
    margins_from_geom,
    official_cost_from_geom,
    safety_target,
)


def _geom(x=0.0, y=0.0, z=0.8, z_rot=1.0, vx=0.0, vy=0.0, vz=0.0):
    return geom_from_torso(x, y, z, vx, vy, vz, z_rot)


def test_center_margins_and_cost():
    g = _geom()
    m = margins_from_geom(g)
    assert np.isclose(m.h_wall, 5.0 - 1.8, atol=1e-5)
    assert m.h_floor > 0
    assert np.isclose(m.h_ceil, 0.2, atol=1e-5)
    assert m.h_upright > 0
    assert np.isclose(m.h, 0.2, atol=1e-5)
    assert official_cost_from_geom(g) == 0
    assert m.active == 2


def test_official_cost_and_margin_sign_agree():
    rng = np.random.default_rng(0)
    for _ in range(500):
        g = geom_from_torso(
            rng.uniform(0, 100), rng.uniform(-20, 20), rng.uniform(0, 1.3),
            rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(-2, 2),
            rng.uniform(-1, 1))
        m = margins_from_geom(g)
        assert int(m.h < 0.0) == official_cost_from_geom(g)


def test_wall_height_and_upright_cost():
    assert margins_from_geom(_geom(y=3.5)).h_wall < 0
    assert official_cost_from_geom(_geom(z=0.1)) == 1
    assert official_cost_from_geom(_geom(z_rot=-0.8)) == 1
    assert margins_from_geom(_geom(z=0.1)).h_floor < 0
    assert margins_from_geom(_geom(z_rot=-0.8)).h_upright < 0


def test_margin_gradient_direction():
    g = _geom(z=0.8)  # active margin is the ceiling: h = 1 - z.
    grad = finite_difference_grad(g)
    assert grad.shape == (9,)
    assert np.isclose(grad[4], -1.0, atol=1e-3)
    assert np.allclose(grad[[0, 1, 5, 6, 7, 8]], 0.0, atol=1e-4)


def test_forward_window_labels():
    unsafe = np.array([0, 0, 0, 1, 0, 0, 0, 0], dtype=bool)
    labels = forward_safety_labels(unsafe, horizon=2)
    # t=0 sees no unsafe in t=1,2 -> safe; t=1 sees t=3 -> ignored;
    # t=2 sees t=3 -> ignored; t=3 current unsafe; t=4 onward safe.
    assert labels[0] == 1
    assert labels[1] == 0
    assert labels[2] == 0
    assert labels[3] == -1
    assert labels[4] == 1
    assert labels[-1] == 1


def test_horizon_risk_uses_future_minimum():
    # Far wall, then a wall violation 2 steps ahead.  The horizon risk at t=0
    # must be negative even though the current margin is positive.
    geoms = np.stack([
        _geom(y=0.0),
        _geom(y=2.0),
        _geom(y=3.5),
        _geom(y=0.0),
    ])
    risk = horizon_risk_margins(geoms, horizon=2)
    assert risk[0] < 0.0
    assert risk[2] < 0.0
    target, raw = horizon_risk_targets(geoms, scale=0.5, horizon=2)
    assert target[0] < 0.0
    assert np.allclose(target, np.tanh(raw / 0.5), atol=1e-6)


def test_fixed_provider_and_target():
    g = _geom()
    provider = FixedAntSafetyProvider(scale=0.5)
    h, grad = provider.value_and_grad(g)
    assert np.isclose(h, float(np.tanh(0.2 / 0.5)), atol=1e-6)
    assert grad.shape == (9,)
    assert np.isfinite(grad).all()
    assert np.isclose(safety_target(g), h, atol=1e-6)
