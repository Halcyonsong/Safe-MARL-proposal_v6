"""Offline training/evaluation for learned safety heads.

Only safety labels are used.  There is no return loss, no QP target, and no
expert ``u_ref`` here.  The resulting checkpoint is frozen for the official
V5 policy stage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.utils.data

from .checkpoint import save_learned_safety
from .dataset import SafetyDataset
from .learned import LearnedSafetyModel
from .losses import safety_h_total_loss


def _device(name: str | torch.device) -> torch.device:
    if str(name) == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def _batch_loss(
    model: LearnedSafetyModel,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    x = batch['state'].to(device)
    cost = batch['cost_labels'].to(device)
    terminal = batch['terminal'].to(device)
    out = model(x)
    h = out.h_cost_scalar
    h_next = None
    if 'next_state' in batch:
        h_next = model(batch['next_state'].to(device)).h_cost_scalar
    loss, metrics = safety_h_total_loss(
        out.h_cost, out.survival_logit, cost, terminal,
        margin=float(cfg.get('margin', 0.05)),
        survival_coef=float(cfg.get('survival_coef', 1.0)),
        calibration_coef=float(cfg.get('calibration_coef', 0.1)),
        h=h,
        h_next=h_next,
        barrier_coef=float(cfg.get('barrier_coef', 0.5)),
        invariance_coef=float(cfg.get('invariance_coef', 0.5)),
        alpha=float(cfg.get('alpha', 0.1)),
        direction_coef=0.0,
    )
    return loss, metrics


@torch.no_grad()
def evaluate_learned_h(
    model: LearnedSafetyModel,
    dataset: SafetyDataset,
    device: str | torch.device = 'cpu',
    batch_size: int = 512,
) -> dict[str, float]:
    model.eval()
    device = _device(device)
    pred_neg = []
    labels = []
    costs = []
    for batch in dataset.loader(batch_size=batch_size, shuffle=False):
        out = model(batch['state'].to(device))
        pred_neg.append((out.h_cost < 0).cpu().numpy())
        labels.append((batch['cost_labels'] > 0).cpu().numpy())
        costs.append(batch['cost_labels'].cpu().numpy())
    pred = np.concatenate(pred_neg, axis=0)
    unsafe = np.concatenate(labels, axis=0)
    cost = np.concatenate(costs, axis=0)
    names = list(model.cost_names)
    metrics: dict[str, float] = {
        'false_alarm': float(np.mean(pred & ~unsafe)),
        'miss': float(np.mean(~pred & unsafe)),
        'h_neg_rate': float(np.mean(pred)),
        'cost_rate': float(np.mean(cost)),
    }
    for i, name in enumerate(names):
        metrics[f'{name}_false_alarm'] = float(np.mean(pred[:, i] & ~unsafe[:, i]))
        metrics[f'{name}_miss'] = float(np.mean(~pred[:, i] & unsafe[:, i]))
        metrics[f'{name}_h_neg_rate'] = float(np.mean(pred[:, i]))
    return metrics


def train_learned_h(
    model: LearnedSafetyModel,
    dataset: SafetyDataset,
    config: dict[str, Any],
    out: str | Path,
    device: str | torch.device = 'auto',
) -> dict[str, Any]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    device = _device(device)
    model.to(device)
    train_cfg = dict(config.get('train', {}))
    epochs = int(train_cfg.get('epochs', 50))
    batch_size = int(train_cfg.get('batch_size', 256))
    lr = float(train_cfg.get('lr', 1e-3))
    weight_decay = float(train_cfg.get('weight_decay', 1e-4))
    val_fraction = float(train_cfg.get('val_fraction', 0.1))
    seed = int(train_cfg.get('seed', 0))
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = len(dataset)
    n_val = max(1, int(n * val_fraction)) if n > 1 else 0
    indices = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    class _Subset(torch.utils.data.Dataset):
        def __init__(self, base, idx):
            self.base = base
            self.idx = np.asarray(idx, dtype=np.int64)
        def __len__(self):
            return len(self.idx)
        def __getitem__(self, i):
            return self.base[int(self.idx[i])]

    train_loader = torch.utils.data.DataLoader(
        _Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        _Subset(dataset, val_idx), batch_size=batch_size, shuffle=False) if n_val else None

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best = None
    best_state = None
    history = []
    for epoch in range(epochs):
        model.train()
        train_rows = []
        for batch in train_loader:
            loss, metrics = _batch_loss(model, batch, device, train_cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           float(train_cfg.get('max_grad_norm', 1.0)))
            opt.step()
            train_rows.append(dict(metrics, loss=float(loss.detach())))
        train_mean = {k: float(np.mean([r[k] for r in train_rows]))
                      for k in train_rows[0]}
        val_mean = {}
        if val_loader is not None:
            model.eval()
            rows = []
            with torch.no_grad():
                for batch in val_loader:
                    loss, metrics = _batch_loss(model, batch, device, train_cfg)
                    rows.append(dict(metrics, loss=float(loss.detach())))
            val_mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        score = val_mean.get('loss', train_mean['loss'])
        if best is None or score < best:
            best = score
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        row = dict(epoch=epoch + 1, train=train_mean, val=val_mean,
                   best=float(best))
        history.append(row)
        print(json.dumps(row), flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    metrics = evaluate_learned_h(model, dataset, device=device)
    ckpt = save_learned_safety(
        out / train_cfg.get('checkpoint_name', 'learned_h.pt'),
        model,
        config=dict(config, train=train_cfg),
    )
    summary = dict(
        schema='v5.safety_h.train.v1',
        checkpoint=str(ckpt),
        data_rows=n,
        train_rows=int(len(train_idx)),
        val_rows=int(n_val),
        best_loss=float(best) if best is not None else None,
        metrics=metrics,
        history=history,
        note=('cost + survival safety only; no return, no QP target, '
              'no expert u_ref'),
    )
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


__all__ = ['evaluate_learned_h', 'train_learned_h']
