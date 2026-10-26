"""Contract tests for the QP-free V5 policy."""

import sys
from pathlib import Path

import torch

NAV = Path(__file__).resolve().parents[1]
if str(NAV) not in sys.path:
    sys.path.insert(0, str(NAV))

from algos.safe_policy.qp_free import (
    QpFreePolicy, forward_invariance_loss, gradient_direction_loss,
    correction_penalty, action_bound_penalty)


def test_policy_shapes_and_bounds():
    model = QpFreePolicy(obs_dim=12, hidden=32)
    obs = torch.randn(2, 5, 12)
    h = torch.randn(2, 5, 1)
    g = torch.randn(2, 5, 4)
    out = model(obs, h, g)
    assert out['action'].shape == (2, 5, 2)
    assert out['delta'].shape == (2, 5, 2)
    assert out['feasibility'].shape == (2, 5)
    assert torch.all(out['action'].abs() <= 1.0)
    assert torch.all(out['delta'].abs() <= model.correction_radius + 1e-6)


def test_stochastic_interface_shapes_and_finiteness():
    model = QpFreePolicy(obs_dim=12, hidden=32)
    obs = torch.randn(2, 5, 12)
    h = torch.randn(2, 5, 1)
    g = torch.randn(2, 5, 4)
    out = model.sample(obs, h, g)
    assert out['raw_action'].shape == (2, 5, 2)
    assert out['env_action'].shape == (2, 5, 2)
    assert out['log_prob'].shape == (2, 5)
    assert out['entropy'].shape == (2, 5)
    assert torch.isfinite(out['log_prob']).all()
    assert torch.all(out['env_action'].abs() <= 1.0)
    evaluated = model.evaluate_actions(obs, h, g, out['raw_action'])
    assert evaluated['log_prob'].shape == (2, 5)
    assert torch.isfinite(evaluated['log_prob']).all()


def test_safety_losses_are_zero_when_conditions_hold():
    h = torch.ones(4)
    h_next = torch.ones(4)
    assert forward_invariance_loss(h, h_next, alpha=.1).item() == 0.0
    grad = torch.zeros(4, 4)
    delta = torch.zeros(4, 4)
    assert gradient_direction_loss(h, grad, delta).item() == 0.0
    assert correction_penalty(torch.zeros(2, 2)).item() == 0.0
    assert action_bound_penalty(torch.zeros(2, 2)).item() == 0.0
