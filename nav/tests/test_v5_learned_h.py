"""Contract tests for the official learned-h V5 safety layer."""

from __future__ import annotations

import numpy as np
import torch

from v5.safety.checkpoint import LearnedSafetyProvider, save_learned_safety
from v5.safety.dataset import SafetyDataset
from v5.safety.learned import LearnedSafetyModel
from v5.safety.losses import safety_h_total_loss


def test_learned_safety_shapes_and_gradient():
    model = LearnedSafetyModel(
        state_dim=9,
        cost_names=('wall', 'floor', 'ceil', 'upright'),
        hidden=(32, 32),
        survival_hidden=(32,),
    )
    x = torch.randn(5, 9)
    out = model(x)
    assert out.h_cost.shape == (5, 4)
    assert out.survival_logit.shape == (5,)
    assert out.h_cost_scalar.shape == (5,)
    assert out.survival_risk.shape == (5,)
    assert out.active_name[0] in out.cost_names

    pred = model.value_and_local_grad(x)
    assert pred.output.h_cost.shape == (5, 4)
    assert pred.grad_cost.shape == (5, 9)
    assert torch.isfinite(pred.grad_cost).all()


def test_safety_loss_has_no_return_term_and_backprops():
    model = LearnedSafetyModel(9, hidden=(16,), survival_hidden=(16,))
    x = torch.randn(8, 9)
    next_x = x + 0.01 * torch.randn_like(x)
    out = model(x)
    h = out.h_cost_scalar
    h_next = model(next_x).h_cost_scalar
    labels = torch.randint(0, 2, (8, 4)).float()
    terminal = torch.randint(0, 2, (8,)).float()
    loss, metrics = safety_h_total_loss(
        out.h_cost, out.survival_logit, labels, terminal,
        h=h, h_next=h_next, barrier_coef=0.1, invariance_coef=0.1)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
    assert 'total' in metrics
    assert 'return' not in metrics


def test_dataset_and_checkpoint_roundtrip(tmp_path):
    n = 12
    rng = np.random.default_rng(0)
    np.savez(
        tmp_path / 'data.npz',
        states=rng.normal(size=(n, 9)).astype(np.float32),
        cost_labels=rng.integers(0, 2, size=(n, 4)).astype(np.float32),
        terminal=rng.integers(0, 2, size=(n,)).astype(np.float32),
        next_states=rng.normal(size=(n, 9)).astype(np.float32),
    )
    dataset = SafetyDataset.from_npz(tmp_path / 'data.npz')
    assert len(dataset) == n
    assert dataset.cost_dim == 4
    assert dataset.state_dim == 9

    model = LearnedSafetyModel(9, hidden=(16,), survival_hidden=(16,))
    path = save_learned_safety(tmp_path / 'h.pt', model, config={})
    provider = LearnedSafetyProvider.from_checkpoint(path, device='cpu')
    pred = provider.value_and_local_grad(np.zeros(9, dtype=np.float32))
    assert pred.output.h_cost.shape == (1, 4)
    assert pred.grad_cost.shape == (1, 9)
    assert provider.cost_names == ('wall', 'floor', 'ceil', 'upright')
