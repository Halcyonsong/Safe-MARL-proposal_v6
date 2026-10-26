"""Unit tests for the V6 Ant 9-D safety h dataset/model/provider."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

NAV = Path(__file__).resolve().parents[1]
if str(NAV) not in sys.path:
    sys.path.insert(0, str(NAV))

from v6.ant_h import (  # noqa: E402
    AntSafetyDataset,
    AntSafetyNet,
    AntSafetyProvider,
    evaluate_ant_safety,
    train_ant_safety_h,
)
from v6.ant_safety import (  # noqa: E402
    geom_from_torso,
    margins_from_geom,
    safety_target,
    safety_target_grad,
)


def _synthetic_dataset(n=320, seed=0):
    rng = np.random.default_rng(seed)
    geoms = []
    for _ in range(n):
        geoms.append(geom_from_torso(
            rng.uniform(0, 100), rng.uniform(-12, 12),
            rng.uniform(0, 1.2), rng.uniform(-1, 1), rng.uniform(-1, 1),
            rng.uniform(-1, 1), rng.uniform(-1, 1)))
    geom = np.stack(geoms).astype(np.float32)
    labels = np.asarray(
        [-1.0 if margins_from_geom(g).h < 0 else 1.0 for g in geom],
        dtype=np.float32)
    target = np.asarray([safety_target(g) for g in geom], dtype=np.float32)
    target_grad = np.stack([safety_target_grad(g) for g in geom], axis=0)
    return AntSafetyDataset(
        geom=geom, labels=labels, target=target, target_grad=target_grad,
        episode=np.arange(n) // 4, source=np.zeros(n, dtype=np.int8))


def test_dataset_roundtrip_and_shapes(tmp_path):
    ds = _synthetic_dataset(40)
    path = ds.save(tmp_path / 'data.npz')
    loaded = AntSafetyDataset.from_npz(path)
    assert loaded.geom.shape == (40, 9)
    assert loaded.target.shape == (40,)
    assert loaded.target_grad.shape == (40, 9)
    assert np.allclose(loaded.labels, ds.labels)


def test_dynamic_target_recomputes_velocity_gradient(tmp_path):
    ds = _synthetic_dataset(40)
    path = ds.save(tmp_path / 'static.npz')
    dyn = AntSafetyDataset.from_npz(path, h_kappa=0.5, h_kappa_z=0.2)
    assert abs(dyn.h_kappa - 0.5) < 1e-8
    assert abs(dyn.h_kappa_z - 0.2) < 1e-8
    assert not np.allclose(dyn.target, ds.target)
    # The dynamic margin must carry nonzero approach-speed information.
    velocity_grad = dyn.target_grad[:, [2, 3, 6, 7, 8]]
    assert np.any(np.abs(velocity_grad) > 1e-6)


def test_model_output_and_gradient():
    model = AntSafetyNet(hidden=(32, 32))
    x = torch.randn(7, 9, requires_grad=True)
    h = model(x)
    assert h.shape == (7,)
    assert torch.all(h.abs() <= 1.0)
    grad = torch.autograd.grad(h.sum(), x)[0]
    assert grad.shape == (7, 9)
    assert torch.isfinite(grad).all()


def test_train_and_provider(tmp_path):
    train = _synthetic_dataset(200, seed=1)
    val = _synthetic_dataset(60, seed=2)
    test = _synthetic_dataset(60, seed=3)
    model = AntSafetyNet(hidden=(32, 32))
    summary = train_ant_safety_h(
        model, train, val, test_data=test, out=tmp_path,
        device='cpu', epochs=3, batch_size=64, lr=1e-3,
        h_scale=0.5, horizon=8, seed=0)
    assert (tmp_path / 'ant_h.pkl').exists()
    assert 'val' in summary and 'test' in summary
    provider = AntSafetyProvider.from_checkpoint(
        tmp_path / 'ant_h.pkl', device='cpu')
    g = train.geom[0]
    h, grad = provider.value_and_grad(g)
    assert np.isfinite(h) and np.isfinite(grad).all()
    assert grad.shape == (9,)
    values = provider.batch_value(train.geom[:5])
    assert values.shape == (5,)


def test_evaluation_metrics_finite():
    ds = _synthetic_dataset(50)
    model = AntSafetyNet(hidden=(16,))
    metrics = evaluate_ant_safety(model, ds, 'cpu')
    for key in ('balanced_accuracy', 'unsafe_recall', 'safe_recall', 'auroc'):
        assert np.isfinite(metrics[key])
