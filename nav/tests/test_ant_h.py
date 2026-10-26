"""Smoke checks for the fixed Safe MAMuJoCo Ant-2x4 margins."""

import numpy as np
import torch

from algos.safe_policy.ant_h import (
    D_SAFE,
    Z_MAX,
    Z_MIN,
    Z_ROT_MIN,
    finite_difference_grad,
    geom_from_torso,
    margins_from_geom,
    official_cost,
)
from algos.safe_policy.qp_free import QpFreePolicy
from algos.safe_policy.qp_free_ant import QpFreeAntEnsemble


def _state(x=0.0, y=0.0, z=0.8, z_rot=1.0):
    return geom_from_torso(x, y, z, 0.0, 0.0, 0.0, z_rot)


def test_ant_margin_safe_center():
    x_geom = _state()
    m = margins_from_geom(x_geom)
    assert m.h_wall == np.float32(5.0 - D_SAFE)
    assert m.h_floor > 0 and m.h_ceil > 0 and m.h_upright > 0
    assert m.h > 0
    assert official_cost(x_geom) == 0


def test_ant_margin_wall_height_and_upright():
    # y=4 gives signed wall offsets [-1, 5]; the nearest wall is < 1.8 m.
    assert official_cost(_state(y=4.0)) == 1
    assert margins_from_geom(_state(y=4.0)).h_wall < 0
    assert official_cost(_state(z=Z_MIN - 0.01)) == 1
    assert official_cost(_state(z=Z_MAX + 0.01)) == 1
    assert official_cost(_state(z_rot=Z_ROT_MIN - 0.01)) == 1


def test_ant_margin_gradient_shape():
    g = finite_difference_grad(_state(), eps=1e-5)
    assert g.shape == (9,)
    assert np.isfinite(g).all()


def test_ant_policy_shapes_and_losses():
    model = QpFreePolicy(
        31, hidden=32, action_dim=4, h_dim=4, grad_dim=9)
    obs = torch.randn(5, 31)
    h = torch.randn(5, 4)
    grad = torch.randn(5, 9)
    out = model(obs, h, grad)
    assert out['action'].shape == (5, 4)
    assert out['task'].shape == (5, 4)
    assert out['delta'].shape == (5, 4)
    assert out['value'].shape == (5,)

    sampled = model.sample(obs, h, grad)
    assert sampled['env_action'].shape == (5, 4)
    assert sampled['log_prob'].shape == (5,)

    from algos.safe_policy.qp_free import (
        barrier_loss,
        forward_invariance_loss,
    )
    h_train = torch.randn(5, 4, requires_grad=True)
    h_next = torch.randn(5, 4, requires_grad=True)
    loss = barrier_loss(h_train) + forward_invariance_loss(h_train, h_next)
    loss.backward()


def test_ant_ensemble_shapes():
    model = QpFreeAntEnsemble(2, obs_dim=31, action_dim=4, h_dim=4, grad_dim=9,
                              hidden=32)
    obs = torch.randn(2, 31)
    h = torch.randn(2, 4)
    grad = torch.randn(2, 9)
    out = model(obs, h, grad)
    assert out['action'].shape == (2, 4)
    assert out['value'].shape == (2,)
    sampled = model.sample(obs, h, grad)
    assert sampled['env_action'].shape == (2, 4)
    raw = sampled['raw_action']
    ev = model.evaluate_actions(obs, h, grad, raw)
    assert ev['log_prob'].shape == (2,)
