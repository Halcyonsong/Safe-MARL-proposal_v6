"""Learned navigation safety value h.

V6 replaces the official frozen checkpoint with a learned local safety net.
The safety target is trained only from safety labels (collision/warning), never
from task return, QP labels, or an expert action.  The network output is
``tanh`` in [-1, 1] to match the V5 h/barrier scale.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import nn

from .features import diagonal_state_grad, torch_local_obs
from .graph_nets import LocalGraphNet


def dense_margin_values(
    obs: np.ndarray,
    *,
    max_neighbors: int = 5,
    n_rays: int = 32,
    area_size: float = 4.0,
    comm_radius: float = 0.5,
    car_radius: float = 0.05,
    include_wall: bool = True,
) -> np.ndarray:
    """Return the raw dense safety margin from the compact observation.

    The margin is positive away from constraints and negative inside them.
    It uses only geometry already in the observation (neighbour offsets,
    LiDAR hit offsets, own position); no task reward or expert action is
    involved.  ``dense_margin_targets`` maps this value through ``tanh``.

    ``include_wall`` is kept for the old boundary diagnostic.  The predictive
    graph h uses ``include_wall=False``: the serialized graph has only
    relative agent/goal/LiDAR edges and no absolute-position feature, so a
    wall margin is not identifiable from the h input.  The official
    navigation cost also does not penalise the outer wall.
    """
    x = torch.as_tensor(obs, dtype=torch.float32)
    n = x.shape[0]
    pos = x[:, :2]
    neighbor = x[:, 8:8 + 4 * int(max_neighbors)].reshape(
        n, int(max_neighbors), 4)[..., :2]
    nd = torch.linalg.vector_norm(neighbor, dim=-1)
    nd = torch.where(nd < 1e-6, torch.full_like(nd, float('inf')), nd)
    agent_margin = nd.min(dim=1).values - 2.0 * float(car_radius)
    lidar = x[:, 8 + 4 * int(max_neighbors):].reshape(
        n, int(n_rays), 2)
    ld = torch.linalg.vector_norm(lidar, dim=-1)
    hit = ld > 1e-6
    ld = torch.where(hit, ld, torch.full_like(ld, float(comm_radius)))
    obstacle_margin = ld.min(dim=1).values - float(car_radius)
    margin = torch.minimum(obstacle_margin, agent_margin)
    if include_wall:
        wall_margin = torch.minimum(
            pos, float(area_size) - pos).min(dim=1).values
        margin = torch.minimum(margin, wall_margin)
    margin = torch.clamp(margin, -0.5, 0.5)
    return margin.numpy()


def dense_margin_targets(
    obs: np.ndarray,
    *,
    max_neighbors: int = 5,
    n_rays: int = 32,
    area_size: float = 4.0,
    comm_radius: float = 0.5,
    car_radius: float = 0.05,
    scale: float = 0.2,
    include_wall: bool = True,
) -> np.ndarray:
    """Build a dense safety-margin target from the compact local observation.

    The target is positive away from constraints and negative inside them.
    It uses only geometry already in the observation (neighbour offsets, LiDAR
    hit offsets, own position); no task reward or expert action is involved.
    """
    margin = dense_margin_values(
        obs, max_neighbors=max_neighbors, n_rays=n_rays,
        area_size=area_size, comm_radius=comm_radius,
        car_radius=car_radius, include_wall=include_wall)
    return np.tanh(margin / float(scale)).astype(np.float32)


def dense_margin_state_gradients(
    obs: np.ndarray,
    *,
    max_neighbors: int = 5,
    n_rays: int = 32,
    area_size: float = 4.0,
    comm_radius: float = 0.5,
    car_radius: float = 0.05,
    scale: float = 0.2,
    include_wall: bool = False,
) -> np.ndarray:
    """Return ``d tanh(margin/scale) / d state_i`` for every agent.

    The gradient is the diagonal local state gradient ``[..., 4]`` used by the
    actor's h-feature.  It is computed from the compact observation only:
    the agent's own position is perturbed while other agents' positions and
    the LiDAR hit points are held fixed.  This is the faithful navigation
    analogue of the Ant analytic ``safety_target_grad`` used for gradient
    alignment; it is not an action label, QP target, or expert direction.
    """
    x = torch.as_tensor(np.asarray(obs), dtype=torch.float32)
    if x.ndim != 3:
        raise ValueError(f'expected [n_graphs,N,obs_dim], got {tuple(x.shape)}')
    n, n_agents = x.shape[:2]
    state0 = x[..., :4].detach()
    state = state0.clone().requires_grad_(True)
    pos = state[..., :2]
    delta = state0[..., :2] - pos

    neighbor0 = x[..., 8:8 + 4 * int(max_neighbors)].reshape(
        n, n_agents, int(max_neighbors), 4)[..., :2].detach()
    self_edge = torch.linalg.vector_norm(neighbor0, dim=-1) < 1e-6
    neighbor = neighbor0 + delta[..., None, :]
    neighbor = torch.where(self_edge[..., None],
                           torch.zeros_like(neighbor), neighbor)
    nd = torch.linalg.vector_norm(neighbor, dim=-1)
    nd = torch.where(nd < 1e-6, torch.full_like(nd, float('inf')), nd)
    agent_margin = nd.min(dim=-1).values - 2.0 * float(car_radius)

    lidar0 = x[..., 8 + 4 * int(max_neighbors):].reshape(
        n, n_agents, int(n_rays), 2).detach()
    hit = torch.linalg.vector_norm(lidar0, dim=-1) > 1e-6
    lidar = torch.where(hit[..., None], lidar0 + delta[..., None, :],
                        torch.zeros_like(lidar0))
    ld = torch.linalg.vector_norm(lidar, dim=-1)
    ld = torch.where(hit, ld, torch.full_like(ld, float(comm_radius)))
    obstacle_margin = ld.min(dim=-1).values - float(car_radius)

    margin = torch.minimum(obstacle_margin, agent_margin)
    if include_wall:
        wall_margin = torch.minimum(
            pos, float(area_size) - pos).min(dim=-1).values
        margin = torch.minimum(margin, wall_margin)
    margin = torch.clamp(margin, -0.5, 0.5)
    target = torch.tanh(margin / float(scale))
    grad = torch.autograd.grad(
        target.sum(), state, create_graph=False, retain_graph=False)[0]
    return grad.detach().numpy().astype(np.float32)


def forward_margin_sequence(margins: np.ndarray,
                            terminal_margin: np.ndarray) -> np.ndarray:
    """Append the terminal next-state margin exactly once.

    A rollout stores margins for states ``0..T-1``.  The margin after the last
    action belongs to state ``T`` and is not a training row, but it must close
    the forward window.  Duplicating the rollout or wrapping around to its
    beginning would corrupt end-of-episode targets.
    """
    margins = np.asarray(margins, dtype=np.float32)
    if margins.ndim != 2:
        raise ValueError(f'expected [T,N] margins, got {margins.shape}')
    terminal = np.asarray(terminal_margin, dtype=np.float32).reshape(
        1, margins.shape[1])
    return np.concatenate([margins, terminal], axis=0)


def horizon_min_margin(margins: np.ndarray, horizon: int = 32) -> np.ndarray:
    """Minimum margin over the current and next ``horizon`` states.

    ``margins`` has shape ``[T, N]``.  The result at ``t`` is
    ``min(margins[t:t + horizon + 1])``; the end of the rollout uses the
    available suffix.  This is the same forward-minimum semantic used by the
    Ant predictive-h target.
    """
    margins = np.asarray(margins, dtype=np.float64)
    if margins.ndim != 2:
        raise ValueError(f'expected [T,N] margins, got {margins.shape}')
    horizon = max(0, int(horizon))
    risk = np.empty_like(margins)
    for t in range(len(margins)):
        risk[t] = np.min(margins[t:t + horizon + 1], axis=0)
    return risk.astype(np.float32)


def horizon_margin_targets(
    margins: np.ndarray,
    *,
    scale: float = 0.2,
    horizon: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``tanh(future_min_margin / scale)`` and the raw future risk."""
    risk = horizon_min_margin(margins, horizon=horizon)
    target = np.tanh(risk / float(scale)).astype(np.float32)
    return target, risk


class LocalSafetyNet(nn.Module):
    """MLP over the local graph observation, with a signed safety output."""

    arch = 'mlp'

    def __init__(self, obs_dim: int, hidden: tuple[int, ...] = (256, 256)):
        super().__init__()
        layers: list[nn.Module] = []
        width = int(obs_dim)
        for h in hidden:
            layers.extend([
                nn.Linear(width, int(h)),
                nn.LayerNorm(int(h)),
                nn.SiLU(),
            ])
            width = int(h)
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)
        self.obs_dim = int(obs_dim)
        self.hidden = tuple(int(v) for v in hidden)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs)).squeeze(-1)

    def value_and_local_grad(
        self,
        states: torch.Tensor,
        goals: torch.Tensor,
        lidar: torch.Tensor,
        *,
        max_neighbors: int,
        n_rays: int,
        comm_radius: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return h [N] and diagonal d h_i / d s_i [N,4]."""
        if not states.requires_grad:
            states = states.detach().clone().requires_grad_(True)
        obs = torch_local_obs(
            states, goals, lidar,
            max_neighbors=max_neighbors, n_rays=n_rays,
            comm_radius=comm_radius)
        h = self.forward(obs)
        return h, diagonal_state_grad(h, states)


class LocalSafetyProvider:
    """Environment-facing frozen h provider used by the V5 policy loop."""

    kind = 'learned_local'

    def __init__(
        self,
        model: LocalSafetyNet,
        env,
        *,
        device: str | torch.device = 'cpu',
        max_neighbors: int = 5,
        n_rays: int = 32,
        comm_radius: float | None = None,
        metadata: Optional[dict[str, Any]] = None,
        h_offset: float = 0.0,
    ):
        self.model = model.to(torch.device(device))
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.env = env
        self.device = torch.device(device)
        self.max_neighbors = int(max_neighbors)
        self.n_rays = int(n_rays)
        self.comm_radius = float(
            env.comm_radius if comm_radius is None else comm_radius)
        self.metadata = dict(metadata or {})
        self.h_offset = float(h_offset)

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        env,
        *,
        device: str | torch.device = 'cpu',
        max_neighbors: int = 5,
        n_rays: int = 32,
        comm_radius: float | None = None,
    ) -> 'LocalSafetyProvider':
        ckpt = torch.load(Path(path), map_location='cpu', weights_only=False)
        model_cfg = dict(ckpt.get('model_config', {}))
        if model_cfg.get('arch') == 'graph':
            model = LocalGraphNet(
                int(ckpt['obs_dim']),
                hidden=int(model_cfg.get('graph_hidden', 128)),
                layers=int(model_cfg.get('graph_layers', 2)),
                max_neighbors=int(model_cfg.get('max_neighbors', max_neighbors)),
                n_rays=int(model_cfg.get('n_rays', n_rays)),
                output_activation='tanh')
        else:
            model = LocalSafetyNet(
                int(ckpt['obs_dim']),
                tuple(model_cfg.get('hidden', (256, 256))))
        model.load_state_dict(ckpt['model'])
        return cls(
            model, env, device=device, max_neighbors=max_neighbors,
            n_rays=n_rays, comm_radius=comm_radius, metadata=ckpt,
            h_offset=float(ckpt.get('h_offset', 0.0)))

    def _inputs(self):
        states = torch.tensor(
            self.env.get_agent_states(), dtype=torch.float32,
            device=self.device).detach().clone().requires_grad_(True)
        goals = torch.tensor(
            self.env.get_goal_states(), dtype=torch.float32,
            device=self.device)
        lidar = torch.tensor(
            self.env.get_lidar(), dtype=torch.float32, device=self.device)
        return states, goals, lidar

    @torch.no_grad()
    def value(self):
        states, goals, lidar = self._inputs()
        obs = torch_local_obs(
            states, goals, lidar,
            max_neighbors=self.max_neighbors, n_rays=self.n_rays,
            comm_radius=self.comm_radius)
        return (self.model(obs) + self.h_offset).detach().cpu().numpy()

    def value_and_local_grad(self):
        states, goals, lidar = self._inputs()
        with torch.enable_grad():
            h, grad = self.model.value_and_local_grad(
                states, goals, lidar,
                max_neighbors=self.max_neighbors, n_rays=self.n_rays,
                comm_radius=self.comm_radius)
        return ((h + self.h_offset).detach().cpu().numpy(),
                grad.detach().cpu().numpy())


def save_safety_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    config: dict[str, Any] | None = None,
    metrics: dict[str, float] | None = None,
    h_offset: float = 0.0,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(model, 'graph_config'):
        model_config = model.graph_config()
    else:
        model_config = dict(
            arch='mlp', hidden=list(getattr(model, 'hidden', (256, 256))))
    torch.save(dict(
        schema='v6-local-safety',
        obs_dim=model.obs_dim,
        model_config=model_config,
        model=model.state_dict(),
        config=config or {},
        metrics=metrics or {},
        h_offset=float(h_offset),
    ), path)
    return path


@dataclass
class SafetyDataset:
    obs: np.ndarray
    labels: np.ndarray
    next_obs: Optional[np.ndarray] = None
    margin_targets: Optional[np.ndarray] = None
    # Rows with valid next_obs (base random rollouts); augmented rows have no
    # transition and must not contribute to the h_dot loss.
    hdot_mask: Optional[np.ndarray] = None

    def __post_init__(self):
        self.obs = np.asarray(self.obs, dtype=np.float32)
        self.labels = np.asarray(self.labels, dtype=np.float32).reshape(-1)
        if self.obs.ndim != 2:
            raise ValueError(f'obs must be [M,D], got {self.obs.shape}')
        if len(self.obs) != len(self.labels):
            raise ValueError('obs/labels length mismatch')
        if self.next_obs is not None:
            self.next_obs = np.asarray(self.next_obs, dtype=np.float32)
            if self.next_obs.shape != self.obs.shape:
                raise ValueError('next_obs must match obs shape')
        if self.hdot_mask is None:
            self.hdot_mask = np.ones(len(self.obs), dtype=np.float32) \
                if self.next_obs is not None else np.zeros(len(self.obs), dtype=np.float32)
        else:
            self.hdot_mask = np.asarray(
                self.hdot_mask, dtype=np.float32).reshape(-1)
            if len(self.hdot_mask) != len(self.obs):
                raise ValueError('hdot_mask length mismatch')
        if self.margin_targets is None:
            self.margin_targets = dense_margin_targets(self.obs)
        else:
            self.margin_targets = np.asarray(
                self.margin_targets, dtype=np.float32).reshape(-1)
            if len(self.margin_targets) != len(self.obs):
                raise ValueError('margin_targets length mismatch')

    @classmethod
    def from_npz(cls, path: str | Path) -> 'SafetyDataset':
        with np.load(Path(path)) as data:
            obs = data['obs'].reshape(-1, data['obs'].shape[-1])
            labels = data['labels'].reshape(-1)
            nxt = data['next_obs'] if 'next_obs' in data.files else None
            if nxt is not None:
                nxt = nxt.reshape(-1, nxt.shape[-1])
            hdot_mask = data['hdot_mask'] if 'hdot_mask' in data.files else None
        return cls(obs=obs, labels=labels, next_obs=nxt,
                   hdot_mask=hdot_mask)

    def __len__(self) -> int:
        return int(len(self.obs))


def _safety_loss(
    model: LocalSafetyNet,
    obs: torch.Tensor,
    labels: torch.Tensor,
    *,
    margin: float,
    safe_weight: torch.Tensor,
    unsafe_weight: torch.Tensor,
    margin_targets: Optional[torch.Tensor] = None,
    margin_coef: float = 0.0,
    next_obs: Optional[torch.Tensor] = None,
    hdot_mask: Optional[torch.Tensor] = None,
    hdot_coef: float = 0.0,
    hdot_alpha: float = 1.0,
    hdot_eps: float = 0.02,
    dt: float = 0.03,
    invariance_coef: float = 0.0,
    alpha: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    h = model(obs)
    safe = labels > 0
    unsafe = ~safe
    zero = h.new_zeros(())
    safe_loss = (safe_weight * torch.relu(float(margin) - h))[safe].mean() \
        if safe.any() else zero
    unsafe_loss = (unsafe_weight * torch.relu(h + float(margin)))[unsafe].mean() \
        if unsafe.any() else zero
    total = safe_loss + unsafe_loss
    metrics = {
        'safe': float(safe_loss.detach()),
        'unsafe': float(unsafe_loss.detach()),
    }
    if margin_targets is not None and margin_coef:
        # Only safe samples get the dense margin target.  Unsafe samples are
        # handled by the classification term; otherwise a heading-dependent
        # warning label can conflict with a positive geometric margin.
        margin_loss = (torch.nn.functional.smooth_l1_loss(
            h[safe], margin_targets[safe], reduction='none').mean()
            if safe.any() else zero)
        total = total + float(margin_coef) * margin_loss
        metrics['margin'] = float(margin_loss.detach())
    if next_obs is not None and hdot_coef:
        h_next = model(next_obs)
        hdot = (h_next - h) / float(dt)
        # Official-style CBF condition: h_dot + alpha*h >= eps.  We only
        # use actual transitions, never QP labels or the official h.
        viol = torch.relu(-hdot - float(hdot_alpha) * h + float(hdot_eps))
        if hdot_mask is not None:
            m = hdot_mask.to(viol.dtype)
            hdot_loss = (viol * m).sum() / m.sum().clamp_min(1.0)
        else:
            hdot_loss = viol.mean()
        total = total + float(hdot_coef) * hdot_loss
        metrics['h_dot'] = float(hdot_loss.detach())
    if next_obs is not None and invariance_coef:
        h_next = model(next_obs)
        inv = torch.relu((1.0 - float(alpha)) * h - h_next).mean()
        total = total + float(invariance_coef) * inv
        metrics['invariance'] = float(inv.detach())
    metrics['total'] = float(total.detach())
    return total, metrics


@torch.no_grad()
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


def evaluate_safety(
    model: LocalSafetyNet,
    dataset: SafetyDataset,
    device: str | torch.device,
    batch_size: int = 1024,
    threshold: float = 0.0,
) -> dict[str, float]:
    model.eval()
    device = torch.device(device)
    preds, labels, values = [], [], []
    for start in range(0, len(dataset), batch_size):
        obs = torch.as_tensor(
            dataset.obs[start:start + batch_size], dtype=torch.float32,
            device=device)
        h = model(obs).detach()
        preds.append((h < float(threshold)).cpu().numpy())
        labels.append((dataset.labels[start:start + batch_size] < 0))
        values.append(h.cpu().numpy())
    pred = np.concatenate(preds)
    unsafe = np.concatenate(labels)
    h = np.concatenate(values)
    false_alarm = float(np.mean(pred & ~unsafe))
    miss = float(np.mean(~pred & unsafe))
    unsafe_recall = float(np.mean(pred[unsafe])) if unsafe.any() else 0.0
    safe_recall = float(np.mean(~pred[~unsafe])) if (~unsafe).any() else 0.0
    return {
        'accuracy': float(np.mean(pred == unsafe)),
        'balanced_accuracy': 0.5 * (unsafe_recall + safe_recall),
        'unsafe_recall': unsafe_recall,
        'safe_recall': safe_recall,
        'false_alarm': false_alarm,
        'miss': miss,
        'auroc': _roc_auc(-h, unsafe),
        'unsafe_rate': float(np.mean(unsafe)),
        'h_mean': float(np.mean(h)),
        'h_std': float(np.std(h)),
        'h_neg_rate': float(np.mean(pred)),
    }


@torch.no_grad()
def _raw_h(
    model: LocalSafetyNet,
    dataset: SafetyDataset,
    device: torch.device,
    batch_size: int = 2048,
) -> np.ndarray:
    model.eval()
    values = []
    for start in range(0, len(dataset), batch_size):
        obs = torch.as_tensor(
            dataset.obs[start:start + batch_size], dtype=torch.float32,
            device=device)
        values.append(model(obs).cpu().numpy())
    return np.concatenate(values)


def calibrate_threshold(
    model: LocalSafetyNet,
    dataset: SafetyDataset,
    device: torch.device,
) -> tuple[float, float]:
    """Choose the h threshold that maximizes balanced accuracy on held-out data.

    The provider adds ``-threshold`` to h, so the policy always sees the
    calibrated boundary at h=0.  The gradient is unchanged by this shift.
    """
    h = _raw_h(model, dataset, device)
    unsafe = dataset.labels < 0
    candidates = np.unique(np.quantile(h, np.linspace(0.01, 0.99, 99)))
    best_t, best_score = 0.0, -float('inf')
    for t in candidates:
        pred = h < float(t)
        if unsafe.any():
            recall = float(np.mean(pred[unsafe]))
        else:
            recall = 0.0
        if (~unsafe).any():
            safe_recall = float(np.mean(~pred[~unsafe]))
        else:
            safe_recall = 0.0
        score = 0.5 * (recall + safe_recall)
        if score > best_score:
            best_t, best_score = float(t), float(score)
    return best_t, best_score


def train_safety_net(
    model: LocalSafetyNet,
    train_data: SafetyDataset,
    val_data: SafetyDataset,
    *,
    out: str | Path,
    device: str | torch.device = 'auto',
    epochs: int = 30,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    margin: float = 0.05,
    unsafe_weight: float = 1.0,
    margin_coef: float = 0.5,
    hdot_coef: float = 0.0,
    hdot_alpha: float = 1.0,
    hdot_eps: float = 0.02,
    dt: float = 0.03,
    invariance_coef: float = 0.1,
    alpha: float = 0.1,
    max_grad_norm: float = 1.0,
    seed: int = 0,
) -> dict[str, Any]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    device = (torch.device('cuda' if torch.cuda.is_available() else 'cpu')
              if str(device) == 'auto' else torch.device(device))
    model = model.to(device)
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_safe = int(np.sum(train_data.labels > 0))
    n_unsafe = int(np.sum(train_data.labels <= 0))
    # The loss already averages safe and unsafe samples separately, so each
    # class contributes equally.  Extra inverse-frequency weights over-correct
    # and collapse h to negative everywhere on imbalanced safety data.
    safe_weight = torch.tensor(1.0, device=device)
    unsafe_weight_t = torch.tensor(float(unsafe_weight), device=device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_score = -float('inf')
    history = []
    n = len(train_data)
    rng = np.random.default_rng(seed)
    for epoch in range(int(epochs)):
        model.train()
        order = rng.permutation(n)
        rows = []
        for start in range(0, n, int(batch_size)):
            idx = order[start:start + int(batch_size)]
            obs = torch.as_tensor(
                train_data.obs[idx], dtype=torch.float32, device=device)
            labels = torch.as_tensor(
                train_data.labels[idx], dtype=torch.float32, device=device)
            targets = torch.as_tensor(
                train_data.margin_targets[idx], dtype=torch.float32,
                device=device)
            nxt = None
            hdot_mask = None
            if train_data.next_obs is not None:
                nxt = torch.as_tensor(
                    train_data.next_obs[idx], dtype=torch.float32,
                    device=device)
                hdot_mask = torch.as_tensor(
                    train_data.hdot_mask[idx], dtype=torch.float32,
                    device=device)
            loss, metrics = _safety_loss(
                model, obs, labels, margin=margin,
                safe_weight=safe_weight, unsafe_weight=unsafe_weight_t,
                margin_targets=targets, margin_coef=margin_coef,
                next_obs=nxt, hdot_mask=hdot_mask,
                hdot_coef=hdot_coef, hdot_alpha=hdot_alpha,
                hdot_eps=hdot_eps, dt=dt,
                invariance_coef=invariance_coef, alpha=alpha)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            rows.append(metrics)
        train_mean = {
            k: float(np.mean([r.get(k, 0.0) for r in rows]))
            for k in rows[0]}
        val = evaluate_safety(model, val_data, device)
        row = dict(epoch=epoch + 1, train=train_mean, val=val)
        history.append(row)
        print(json.dumps(row), flush=True)
        if val['balanced_accuracy'] > best_score:
            best_score = val['balanced_accuracy']
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    threshold, calibrated_score = calibrate_threshold(model, val_data, device)
    metrics = evaluate_safety(
        model, val_data, device, threshold=threshold)
    path = save_safety_checkpoint(
        out / 'safety_h.pt', model,
        config=dict(epochs=epochs, batch_size=batch_size, lr=lr,
                    margin=margin, margin_coef=margin_coef,
                    hdot_coef=hdot_coef, hdot_alpha=hdot_alpha,
                    hdot_eps=hdot_eps, dt=dt,
                    invariance_coef=invariance_coef),
        metrics=metrics, h_offset=-float(threshold))
    summary = dict(
        schema='v6.safety_h.train.v1',
        checkpoint=str(path),
        rows=len(train_data),
        safe_rows=n_safe,
        unsafe_rows=n_unsafe,
        threshold=float(threshold),
        h_offset=-float(threshold),
        raw_best_balanced_accuracy=float(best_score),
        best_balanced_accuracy=float(calibrated_score),
        metrics=metrics,
        history=history,
    )
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary
