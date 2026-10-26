"""Learned task value V_T and its relative scalar/gradient channel.

V_T is trained only from task reward.  It never receives cost labels.  The
policy receives the normalized relative scalar ``A_T`` and the state-space
task gradient ``grad_s V_T``; raw V_T is not a policy input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import nn

from .features import diagonal_state_grad, torch_local_obs
from .graph_nets import LocalGraphNet


def build_task_value_net(
    config: dict,
    obs_dim: int,
    *,
    max_neighbors: int = 5,
    n_rays: int = 32,
) -> nn.Module:
    """Build the scenario-appropriate V_T network (graph by default)."""
    arch = str(config.get('arch', 'graph')).lower()
    if arch == 'graph':
        return LocalGraphNet(
            int(obs_dim),
            hidden=int(config.get('graph_hidden', 128)),
            layers=int(config.get('graph_layers', 2)),
            max_neighbors=int(max_neighbors),
            n_rays=int(n_rays),
            output_activation=None)
    if arch == 'mlp':
        return LocalTaskValueNet(
            int(obs_dim), hidden=tuple(config.get('hidden', (256, 256))))
    raise ValueError(f'unknown task value architecture: {arch}')


class LocalTaskValueNet(nn.Module):
    """Per-agent task value on the local V5 observation."""

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
        return self.net(obs).squeeze(-1)


@dataclass
class TaskAdvantageEMA:
    """Per-agent recent mean/variance for relative task values.

    ``decay`` is the exponential-decay factor.  A cumulative mean/variance is
    wrong here: the actor changes during training, so early-policy task values
    would dominate the baseline and make ``A_T`` systematically nonzero.
    """

    mean: torch.Tensor
    var: torch.Tensor
    count: float = 0.0
    warmup_count: float = 1.0
    clip: float = 5.0
    decay: float = 0.99
    var_floor: float = 1.0

    @classmethod
    def zeros(cls, n_agents: int, warmup_count: float = 1.0,
              clip: float = 5.0, decay: float = 0.99,
              var_floor: float = 1.0) -> 'TaskAdvantageEMA':
        return cls(
            mean=torch.zeros(n_agents, dtype=torch.float32),
            var=torch.ones(n_agents, dtype=torch.float32),
            count=0.0,
            warmup_count=float(warmup_count),
            clip=float(clip),
            decay=float(decay),
            var_floor=float(var_floor),
        )

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        x = values.detach().float().reshape(-1, self.mean.numel())
        if x.numel() == 0:
            return
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = float(x.shape[0])

        if self.count < self.warmup_count:
            # Exact cumulative initialization while the baseline is warming.
            total = self.count + batch_count
            delta = batch_mean - self.mean
            new_mean = self.mean + delta * batch_count / total
            m2_old = self.var * self.count
            m2_batch = batch_var * batch_count
            m2_new = (m2_old + m2_batch
                      + delta.square() * (self.count * batch_count / total))
            self.mean.copy_(new_mean)
            self.var.copy_((m2_new / total).clamp_min(self.var_floor))
        else:
            # Recency-weighted baseline.  The variance update uses the law of
            # total variance so a moving policy mean is not ignored.
            d = min(max(float(self.decay), 0.0), 0.9999)
            new_mean = d * self.mean + (1.0 - d) * batch_mean
            new_var = (
                d * (self.var + (self.mean - new_mean).square())
                + (1.0 - d) * (batch_var
                               + (batch_mean - new_mean).square())
            )
            self.mean.copy_(new_mean)
            self.var.copy_(new_var.clamp_min(self.var_floor))
        self.count += batch_count

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        if self.count < self.warmup_count:
            return torch.zeros_like(values)
        var = self.var.clamp_min(self.var_floor)
        a = (values - self.mean) / torch.sqrt(var + 1e-6)
        return torch.clamp(a, -self.clip, self.clip)

    def state_dict(self) -> dict:
        return dict(mean=self.mean.cpu(), var=self.var.cpu(),
                    count=self.count, warmup_count=self.warmup_count,
                    clip=self.clip, decay=self.decay,
                    var_floor=self.var_floor)

    @classmethod
    def from_state_dict(cls, state: dict) -> 'TaskAdvantageEMA':
        # EMA statistics are always kept on CPU; normalize() receives CPU
        # values in both rollout and evaluation.  torch.load with
        # map_location=cuda would otherwise move these tensors silently.
        return cls(
            mean=torch.as_tensor(state['mean']).detach().cpu().float().clone(),
            var=torch.as_tensor(state['var']).detach().cpu().float().clone(),
            count=float(state.get('count', 0.0)),
            warmup_count=float(state.get('warmup_count', 1.0)),
            clip=float(state.get('clip', 5.0)),
            decay=float(state.get('decay', 0.99)),
            var_floor=float(state.get('var_floor', 1.0)),
        )


def task_state_grad(
    model: LocalTaskValueNet,
    states: torch.Tensor,
    goals: torch.Tensor,
    lidar: torch.Tensor,
    *,
    max_neighbors: int,
    n_rays: int,
    comm_radius: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return V_T [N] and diagonal d V_T,i / d s_i [N,4]."""
    if not states.requires_grad:
        states = states.detach().clone().requires_grad_(True)
    obs = torch_local_obs(
        states, goals, lidar,
        max_neighbors=max_neighbors, n_rays=n_rays,
        comm_radius=comm_radius)
    value = model(obs)
    return value, diagonal_state_grad(value, states)


def normalize_gradient(
    grad: torch.Tensor,
    *,
    eps: float = 1e-6,
    max_norm: float = 1.0,
) -> torch.Tensor:
    """Unit-normalize each agent's task-gradient row."""
    norm = torch.linalg.vector_norm(grad, dim=-1, keepdim=True)
    return grad / norm.clamp_min(eps) * min(1.0, float(max_norm))


__all__ = [
    'LocalTaskValueNet', 'LocalGraphNet', 'build_task_value_net',
    'TaskAdvantageEMA', 'task_state_grad', 'normalize_gradient',
]
