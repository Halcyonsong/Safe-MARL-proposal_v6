"""V6-learned Ant safety h(s) and its state-space direction.

The Ant h is intentionally much smaller than the navigation graph h: the Ant
safety state is a single 9-D torso geometry vector, so an MLP is the faithful
architecture.  The training protocol borrows the navigation construction:

* safety-only labels from the official Ant cost/terminal signal,
* forward-window positive/negative/ignored labels,
* a dense threshold-margin target,
* a gradient-direction target so the actor receives a meaningful
  ``grad_s h`` feature,
* threshold calibration on a held-out V6 split,
* frozen h with no task return and no shared actor parameters.

No V5 h, expert action, QP target, or task return is read here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .ant_safety import (
    GEOM_DIM,
    MARGIN_NAMES,
    margins_from_geom,
    safety_target,
    safety_target_grad,
)


class AntSafetyNet(nn.Module):
    """MLP from 9-D torso geometry to a bounded scalar safety value."""

    arch = 'ant_mlp'

    def __init__(
        self,
        geom_dim: int = GEOM_DIM,
        hidden: tuple[int, ...] = (128, 128),
    ):
        super().__init__()
        self.geom_dim = int(geom_dim)
        self.hidden = tuple(int(v) for v in hidden)
        layers: list[nn.Module] = []
        width = self.geom_dim
        for width_h in self.hidden:
            layers.extend([
                nn.Linear(width, width_h),
                nn.LayerNorm(width_h),
                nn.SiLU(),
            ])
            width = width_h
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Linear(width, 1)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.head.weight, gain=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, geom: torch.Tensor) -> torch.Tensor:
        if geom.ndim == 1:
            geom = geom.unsqueeze(0)
        if geom.shape[-1] != self.geom_dim:
            raise ValueError(
                f'expected geometry [...,{self.geom_dim}], got {tuple(geom.shape)}')
        return torch.tanh(self.head(self.encoder(geom))).squeeze(-1)

    def model_config(self) -> dict[str, Any]:
        return dict(
            arch=self.arch,
            geom_dim=self.geom_dim,
            hidden=list(self.hidden),
            output_activation='tanh',
        )


class AntSafetyProvider:
    """Frozen h provider with scalar value and 9-D state gradient."""

    kind = 'learned_ant'

    def __init__(
        self,
        model: AntSafetyNet,
        *,
        device: str | torch.device = 'cpu',
        h_offset: float = 0.0,
        h_scale: float = 1.0,
        metadata: Optional[dict[str, Any]] = None,
    ):
        self.model = model.to(torch.device(device)).eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.device = torch.device(device)
        self.h_offset = float(h_offset)
        self.h_scale = float(h_scale)
        self.metadata = dict(metadata or {})

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = 'cpu',
    ) -> 'AntSafetyProvider':
        ckpt = torch.load(Path(path), map_location='cpu', weights_only=False)
        model_cfg = dict(ckpt.get('model_config', {}))
        if model_cfg.get('arch') not in (None, 'ant_mlp', 'mlp'):
            raise ValueError(f'unsupported Ant h architecture: {model_cfg}')
        model = AntSafetyNet(
            geom_dim=int(ckpt.get('geom_dim', model_cfg.get('geom_dim', GEOM_DIM))),
            hidden=tuple(model_cfg.get('hidden', (128, 128))),
        )
        model.load_state_dict(ckpt['model'])
        return cls(
            model,
            device=device,
            h_offset=float(ckpt.get('h_offset', 0.0)),
            h_scale=float(ckpt.get('h_scale', 1.0)),
            metadata=ckpt,
        )

    def value(self, geom: np.ndarray) -> float:
        with torch.no_grad():
            x = torch.as_tensor(
                np.asarray(geom, dtype=np.float32).reshape(1, -1),
                device=self.device)
            return float(self.model(x).item() + self.h_offset)

    def value_and_grad(self, geom: np.ndarray) -> tuple[float, np.ndarray]:
        # ``enable_grad`` keeps the h direction available even when the caller
        # wraps an environment step in ``torch.no_grad``.
        with torch.enable_grad():
            x = torch.as_tensor(
                np.asarray(geom, dtype=np.float32).reshape(1, -1),
                dtype=torch.float32, device=self.device,
            ).requires_grad_(True)
            h = self.model(x)
            grad = torch.autograd.grad(
                h.sum(), x, create_graph=False, retain_graph=False)[0]
        return (float(h.detach().item() + self.h_offset),
                grad.detach().cpu().numpy().reshape(-1).astype(np.float32))

    @torch.no_grad()
    def batch_value(self, geom: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(
            np.asarray(geom, dtype=np.float32),
            dtype=torch.float32, device=self.device)
        return (self.model(x) + self.h_offset).cpu().numpy()


@dataclass
class AntSafetyDataset:
    geom: np.ndarray
    labels: np.ndarray
    target: Optional[np.ndarray] = None
    target_grad: Optional[np.ndarray] = None
    episode: Optional[np.ndarray] = None
    source: Optional[np.ndarray] = None
    h_scale: float = 0.5
    h_kappa: float = 0.0
    h_kappa_z: float = 0.0

    def __post_init__(self):
        self.geom = np.asarray(self.geom, dtype=np.float32)
        self.labels = np.asarray(self.labels, dtype=np.float32).reshape(-1)
        if self.geom.ndim != 2 or self.geom.shape[-1] != GEOM_DIM:
            raise ValueError(
                f'geom must be [M,{GEOM_DIM}], got {self.geom.shape}')
        if len(self.geom) != len(self.labels):
            raise ValueError('geom/labels length mismatch')
        if self.target is None:
            self.target = np.asarray(
                [safety_target(g, scale=self.h_scale,
                               kappa=self.h_kappa,
                               kappa_z=self.h_kappa_z)
                 for g in self.geom],
                dtype=np.float32)
        else:
            self.target = np.asarray(self.target, dtype=np.float32).reshape(-1)
        if len(self.target) != len(self.geom):
            raise ValueError('target length mismatch')
        if self.target_grad is None:
            self.target_grad = (
                np.stack([safety_target_grad(
                    g, scale=self.h_scale, kappa=self.h_kappa,
                    kappa_z=self.h_kappa_z) for g in self.geom], axis=0)
                if len(self.geom)
                else np.zeros((0, GEOM_DIM), dtype=np.float32))
        else:
            self.target_grad = np.asarray(self.target_grad, dtype=np.float32)
        if self.target_grad.shape != self.geom.shape:
            raise ValueError(
                f'target_grad must match geom {self.geom.shape}, '
                f'got {self.target_grad.shape}')
        if self.episode is not None:
            self.episode = np.asarray(self.episode).reshape(-1)
        if self.source is not None:
            self.source = np.asarray(self.source).reshape(-1)

    def __len__(self) -> int:
        return int(len(self.geom))

    @classmethod
    def from_npz(
        cls, path: str | Path, *, h_kappa: Optional[float] = None,
        h_kappa_z: Optional[float] = None,
    ) -> 'AntSafetyDataset':
        with np.load(Path(path)) as data:
            stored_kappa = (
                float(data['h_kappa']) if 'h_kappa' in data.files else 0.0)
            stored_kappa_z = (
                float(data['h_kappa_z'])
                if 'h_kappa_z' in data.files else 0.0)
            kappa = stored_kappa if h_kappa is None else float(h_kappa)
            kappa_z = (stored_kappa_z if h_kappa_z is None
                       else float(h_kappa_z))
            # If the caller requests a different dynamic margin, recompute the
            # dense target and its gradient from the stored geometry.
            recompute = (
                abs(kappa - stored_kappa) > 1e-12
                or abs(kappa_z - stored_kappa_z) > 1e-12)
            return cls(
                geom=data['geom'],
                labels=data['labels'],
                target=(None if recompute or 'target' not in data.files
                        else data['target']),
                target_grad=(
                    None if recompute or 'target_grad' not in data.files
                    else data['target_grad']),
                episode=data['episode'] if 'episode' in data.files else None,
                source=data['source'] if 'source' in data.files else None,
                h_scale=float(data['h_scale'])
                if 'h_scale' in data.files else 0.5,
                h_kappa=kappa,
                h_kappa_z=kappa_z,
            )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(
            geom=self.geom,
            labels=self.labels,
            target=self.target,
            target_grad=self.target_grad,
            h_scale=float(self.h_scale),
            h_kappa=float(self.h_kappa),
            h_kappa_z=float(self.h_kappa_z),
        )
        if self.episode is not None:
            payload['episode'] = self.episode
        if self.source is not None:
            payload['source'] = self.source
        np.savez_compressed(path, **payload)
        return path


def _roc_auc(scores: np.ndarray, positive: np.ndarray) -> float:
    n_pos = int(np.sum(positive))
    n_neg = int(np.sum(~positive))
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(scores, kind='mergesort')
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    rank_sum = float(np.sum(ranks[positive]))
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


@torch.no_grad()
def evaluate_ant_safety(
    model: AntSafetyNet,
    dataset: AntSafetyDataset,
    device: str | torch.device,
    *,
    batch_size: int = 2048,
    threshold: float = 0.0,
) -> dict[str, float]:
    model.eval()
    device = torch.device(device)
    values = []
    for start in range(0, len(dataset), int(batch_size)):
        geom = torch.as_tensor(
            dataset.geom[start:start + batch_size], dtype=torch.float32,
            device=device)
        values.append(model(geom).cpu().numpy())
    h = np.concatenate(values) if values else np.zeros(0, dtype=np.float32)
    unsafe = dataset.labels < 0
    pred = h < float(threshold)
    unsafe_recall = float(np.mean(pred[unsafe])) if unsafe.any() else 0.0
    safe_recall = float(np.mean(~pred[~unsafe])) if (~unsafe).any() else 0.0
    return {
        'accuracy': float(np.mean(pred == unsafe)),
        'balanced_accuracy': 0.5 * (unsafe_recall + safe_recall),
        'unsafe_recall': unsafe_recall,
        'safe_recall': safe_recall,
        'false_alarm': float(np.mean(pred & ~unsafe)),
        'miss': float(np.mean(~pred & unsafe)),
        'auroc': _roc_auc(-h, unsafe),
        'unsafe_rate': float(np.mean(unsafe)),
        'h_mean': float(np.mean(h)) if len(h) else 0.0,
        'h_std': float(np.std(h)) if len(h) else 0.0,
        'h_neg_rate': float(np.mean(pred)) if len(h) else 0.0,
    }


def evaluate_ant_gradient_alignment(
    model: AntSafetyNet,
    dataset: AntSafetyDataset,
    device: str | torch.device,
    *,
    batch_size: int = 2048,
) -> dict[str, float]:
    """Measure whether ``grad_s h`` points along the geometric margin target."""
    device = torch.device(device)
    model.eval()
    cosines: list[np.ndarray] = []
    mses: list[np.ndarray] = []
    for start in range(0, len(dataset), int(batch_size)):
        end = min(len(dataset), start + int(batch_size))
        geom = torch.as_tensor(
            dataset.geom[start:end], dtype=torch.float32, device=device,
        ).requires_grad_(True)
        target_grad = torch.as_tensor(
            dataset.target_grad[start:end], dtype=torch.float32,
            device=device)
        h = model(geom)
        grad = torch.autograd.grad(
            h.sum(), geom, create_graph=False, retain_graph=False)[0]
        active = target_grad.norm(dim=-1) > 1e-6
        if bool(active.any()):
            cos = F.cosine_similarity(
                grad[active], target_grad[active], dim=-1, eps=1e-6)
            cosines.append(cos.detach().cpu().numpy())
            mses.append(((grad[active] - target_grad[active]) ** 2)
                        .mean(dim=-1).detach().cpu().numpy())
    if not cosines:
        return dict(grad_cosine=0.0, grad_mse=0.0)
    return dict(
        grad_cosine=float(np.concatenate(cosines).mean()),
        grad_mse=float(np.concatenate(mses).mean()),
    )



def calibrate_ant_threshold(
    model: AntSafetyNet,
    dataset: AntSafetyDataset,
    device: str | torch.device,
) -> tuple[float, float]:
    """Choose the h threshold that maximises balanced accuracy."""
    device = torch.device(device)
    model.eval()
    with torch.no_grad():
        values = []
        for start in range(0, len(dataset), 2048):
            geom = torch.as_tensor(
                dataset.geom[start:start + 2048], dtype=torch.float32,
                device=device)
            values.append(model(geom).cpu().numpy())
    h = np.concatenate(values) if values else np.zeros(0, dtype=np.float32)
    unsafe = dataset.labels < 0
    candidates = np.unique(np.quantile(h, np.linspace(0.01, 0.99, 99)))
    best_t, best_score = 0.0, -float('inf')
    for t in candidates:
        pred = h < float(t)
        recall = float(np.mean(pred[unsafe])) if unsafe.any() else 0.0
        safe_recall = float(np.mean(~pred[~unsafe])) if (~unsafe).any() else 0.0
        score = 0.5 * (recall + safe_recall)
        if score > best_score:
            best_t, best_score = float(t), score
    return best_t, best_score


def _gradient_alignment_loss(
    model: AntSafetyNet,
    geom: torch.Tensor,
    target_grad: torch.Tensor,
    safe_mask: torch.Tensor,
) -> torch.Tensor:
    """Cosine alignment between learned ``d h / d s`` and the safety target."""
    if not bool(safe_mask.any()):
        return geom.new_zeros(())
    x = geom.detach().clone().requires_grad_(True)
    h = model(x)
    grad = torch.autograd.grad(
        h.sum(), x, create_graph=True, retain_graph=True)[0]
    active = safe_mask & (target_grad.norm(dim=-1) > 1e-6)
    if not bool(active.any()):
        return geom.new_zeros(())
    cos = F.cosine_similarity(grad[active], target_grad[active], dim=-1, eps=1e-6)
    return (1.0 - cos).mean()


def _ant_safety_loss(
    model: AntSafetyNet,
    geom: torch.Tensor,
    labels: torch.Tensor,
    target: torch.Tensor,
    target_grad: torch.Tensor,
    *,
    margin: float = 0.05,
    margin_coef: float = 0.5,
    grad_coef: float = 0.1,
    unsafe_weight: float = 1.0,
    target_on_all: bool = False,
    grad_on_all: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    h = model(geom)
    safe = labels > 0
    unsafe = labels < 0
    zero = h.new_zeros(())
    safe_loss = torch.relu(float(margin) - h)[safe].mean() if safe.any() else zero
    unsafe_loss = (
        float(unsafe_weight) * torch.relu(h + float(margin))[unsafe].mean()
        if unsafe.any() else zero)
    target_mask = (labels != 0) if target_on_all else safe
    grad_mask = (labels != 0) if grad_on_all else safe
    target_loss = (
        F.smooth_l1_loss(h[target_mask], target[target_mask],
                         reduction='mean')
        if target_mask.any() else zero)
    grad_loss = _gradient_alignment_loss(
        model, geom, target_grad, grad_mask)
    total = safe_loss + unsafe_loss + float(margin_coef) * target_loss \
        + float(grad_coef) * grad_loss
    return total, {
        'loss': float(total.detach()),
        'safe': float(safe_loss.detach()),
        'unsafe': float(unsafe_loss.detach()),
        'target': float(target_loss.detach()),
        'grad': float(grad_loss.detach()),
        'safe_rows': int(safe.sum()),
        'unsafe_rows': int(unsafe.sum()),
    }


def save_ant_safety_checkpoint(
    path: str | Path,
    model: AntSafetyNet,
    *,
    config: Optional[dict[str, Any]] = None,
    metrics: Optional[dict[str, float]] = None,
    h_offset: float = 0.0,
    h_scale: float = 0.5,
    h_kappa: float = 0.0,
    h_kappa_z: float = 0.0,
    horizon: int = 32,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(
        schema='v6-ant-safety-h.v1',
        geom_dim=int(model.geom_dim),
        model_config=model.model_config(),
        model=model.state_dict(),
        config=config or {},
        metrics=metrics or {},
        h_offset=float(h_offset),
        h_scale=float(h_scale),
        h_kappa=float(h_kappa),
        h_kappa_z=float(h_kappa_z),
        horizon=int(horizon),
    ), path)
    return path


def train_ant_safety_h(
    model: AntSafetyNet,
    train_data: AntSafetyDataset,
    val_data: AntSafetyDataset,
    *,
    test_data: Optional[AntSafetyDataset] = None,
    out: str | Path,
    device: str | torch.device = 'auto',
    epochs: int = 50,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    margin: float = 0.05,
    margin_coef: float = 0.5,
    grad_coef: float = 0.1,
    unsafe_weight: float = 1.0,
    target_on_all: bool = False,
    grad_on_all: bool = False,
    max_grad_norm: float = 1.0,
    h_scale: float = 0.5,
    h_kappa: float = 0.0,
    h_kappa_z: float = 0.0,
    horizon: int = 32,
    seed: int = 0,
) -> dict[str, Any]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    device = (torch.device('cuda' if torch.cuda.is_available() else 'cpu')
              if str(device) == 'auto' else torch.device(device))
    for name, dataset in (('train', train_data), ('val', val_data),
                          ('test', test_data)):
        if dataset is None:
            continue
        if abs(float(dataset.h_scale) - float(h_scale)) > 1e-8:
            raise ValueError(
                f'{name} dataset h_scale={dataset.h_scale} does not match '
                f'training h_scale={h_scale}; recollect or recompute targets')
        if abs(float(dataset.h_kappa) - float(h_kappa)) > 1e-8:
            raise ValueError(
                f'{name} dataset h_kappa={dataset.h_kappa} does not match '
                f'training h_kappa={h_kappa}; recompute targets')
        if abs(float(dataset.h_kappa_z) - float(h_kappa_z)) > 1e-8:
            raise ValueError(
                f'{name} dataset h_kappa_z={dataset.h_kappa_z} does not '
                f'match training h_kappa_z={h_kappa_z}; recompute targets')

    model = model.to(device)
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    best_state = None
    best_score = -float('inf')
    history: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(seed))
    n = len(train_data)
    for epoch in range(int(epochs)):
        model.train()
        order = rng.permutation(n)
        rows = []
        for start in range(0, n, int(batch_size)):
            idx = order[start:start + int(batch_size)]
            geom = torch.as_tensor(
                train_data.geom[idx], dtype=torch.float32, device=device)
            labels = torch.as_tensor(
                train_data.labels[idx], dtype=torch.float32, device=device)
            target = torch.as_tensor(
                train_data.target[idx], dtype=torch.float32, device=device)
            target_grad = torch.as_tensor(
                train_data.target_grad[idx], dtype=torch.float32, device=device)
            loss, metrics = _ant_safety_loss(
                model, geom, labels, target, target_grad,
                margin=margin, margin_coef=margin_coef,
                grad_coef=grad_coef, unsafe_weight=unsafe_weight,
                target_on_all=target_on_all, grad_on_all=grad_on_all)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            rows.append(metrics)
        train_mean = {
            k: float(np.mean([r[k] for r in rows]))
            for k in ('loss', 'safe', 'unsafe', 'target', 'grad')}
        val = evaluate_ant_safety(model, val_data, device)
        row = dict(epoch=epoch + 1, train=train_mean, val=val)
        history.append(row)
        print(json.dumps(row), flush=True)
        if val['balanced_accuracy'] > best_score:
            best_score = float(val['balanced_accuracy'])
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    threshold, calibrated_score = calibrate_ant_threshold(model, val_data, device)
    val_metrics = evaluate_ant_safety(
        model, val_data, device, threshold=threshold)
    val_metrics.update(evaluate_ant_gradient_alignment(model, val_data, device))
    test_metrics = (
        evaluate_ant_safety(model, test_data, device, threshold=threshold)
        if test_data is not None else {})
    if test_data is not None:
        test_metrics.update(
            evaluate_ant_gradient_alignment(model, test_data, device))
    checkpoint = save_ant_safety_checkpoint(
        out / 'ant_h.pkl',
        model,
        config=dict(
            epochs=int(epochs), batch_size=int(batch_size), lr=float(lr),
            weight_decay=float(weight_decay), margin=float(margin),
            margin_coef=float(margin_coef), grad_coef=float(grad_coef),
            unsafe_weight=float(unsafe_weight),
            target_on_all=bool(target_on_all),
            grad_on_all=bool(grad_on_all),
            h_scale=float(h_scale),
            h_kappa=float(h_kappa), h_kappa_z=float(h_kappa_z),
            horizon=int(horizon), seed=int(seed),
        ),
        metrics=dict(val=val_metrics, test=test_metrics),
        h_offset=-float(threshold),
        h_scale=float(h_scale),
        h_kappa=float(h_kappa),
        h_kappa_z=float(h_kappa_z),
        horizon=int(horizon),
    )
    summary = dict(
        schema='v6.ant_safety_h.train.v1',
        checkpoint=str(checkpoint),
        rows=len(train_data),
        val_rows=len(val_data),
        test_rows=len(test_data) if test_data is not None else 0,
        threshold=float(threshold),
        h_offset=-float(threshold),
        raw_best_balanced_accuracy=float(best_score),
        calibrated_balanced_accuracy=float(calibrated_score),
        val=val_metrics,
        test=test_metrics,
        margin_names=list(MARGIN_NAMES),
        history=history,
    )
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


__all__ = [
    'AntSafetyDataset', 'AntSafetyNet', 'AntSafetyProvider',
    'calibrate_ant_threshold', 'evaluate_ant_gradient_alignment',
    'evaluate_ant_safety',
    'save_ant_safety_checkpoint', 'train_ant_safety_h',
]
