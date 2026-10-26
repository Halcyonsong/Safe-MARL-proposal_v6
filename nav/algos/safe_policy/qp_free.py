"""QP-free V5 policy and safety objectives.

This module contains no QP solver, no expert u_ref, and no QP labels.  It learns
a task action plus a bounded safety correction; fixed official h(s) and its
gradient are optional inputs and are used only as safety features/losses.
"""

from __future__ import annotations

import torch
from torch import nn


class QpFreePolicy(nn.Module):
    """Task/correction/feasibility policy with optional safety critics."""

    def __init__(self, obs_dim, hidden=256, correction_radius=0.3,
                 initial_log_std=-1.0, action_dim=2, h_dim=1, grad_dim=4,
                 extra_dim=0):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.h_dim = int(h_dim)
        self.grad_dim = int(grad_dim)
        self.extra_dim = int(extra_dim)
        self.correction_radius = float(correction_radius)
        self.encoder = nn.Sequential(
            nn.Linear(self.obs_dim + self.h_dim + self.grad_dim
                      + self.extra_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.task_head = nn.Linear(hidden, self.action_dim)
        self.correction_head = nn.Linear(hidden, self.action_dim)
        self.feasibility_head = nn.Linear(hidden, 1)
        self.value_head = nn.Linear(hidden, 1)
        self.cost_head = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.full((self.action_dim,), float(initial_log_std)))

    def _std(self):
        return torch.exp(self.log_std).clamp(0.05, 1.0)

    def forward(self, obs, h=None, grad_h=None, extra=None):
        if h is None:
            h = torch.zeros(*obs.shape[:-1], self.h_dim,
                            device=obs.device, dtype=obs.dtype)
        if grad_h is None:
            grad_h = torch.zeros(*obs.shape[:-1], self.grad_dim,
                                 device=obs.device, dtype=obs.dtype)
        if extra is None:
            extra = torch.zeros(*obs.shape[:-1], self.extra_dim,
                                device=obs.device, dtype=obs.dtype)
        if (h.shape[-1] != self.h_dim or grad_h.shape[-1] != self.grad_dim
                or extra.shape[-1] != self.extra_dim):
            raise ValueError(
                f'expected h_dim={self.h_dim}, grad_dim={self.grad_dim}, '
                f'extra_dim={self.extra_dim}; got {tuple(h.shape)}, '
                f'{tuple(grad_h.shape)}, {tuple(extra.shape)}')
        x = torch.cat([obs, h, grad_h, extra], dim=-1)
        z = self.encoder(x)
        task = torch.tanh(self.task_head(z))
        delta = self.correction_radius * torch.tanh(self.correction_head(z))
        mean = task + delta
        action = torch.clamp(mean, -1.0, 1.0)
        feasibility = torch.sigmoid(self.feasibility_head(z)).squeeze(-1)
        value = self.value_head(z).squeeze(-1)
        cost = self.cost_head(z).squeeze(-1)
        return dict(action=action, mean=mean, task=task, delta=delta,
                    feasibility=feasibility, value=value, cost=cost)

    def distribution(self, outputs):
        return torch.distributions.Normal(outputs['mean'], self._std())

    def sample(self, obs, h=None, grad_h=None, extra=None):
        outputs = self.forward(obs, h, grad_h, extra)
        dist = self.distribution(outputs)
        raw_action = dist.rsample()
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return dict(outputs, raw_action=raw_action,
                    env_action=torch.clamp(raw_action, -1.0, 1.0),
                    log_prob=log_prob, entropy=entropy)

    def evaluate_actions(self, obs, h, grad_h, raw_actions, extra=None):
        outputs = self.forward(obs, h, grad_h, extra)
        dist = self.distribution(outputs)
        log_prob = dist.log_prob(raw_actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return dict(outputs, log_prob=log_prob, entropy=entropy)


def forward_invariance_loss(h, h_next, alpha=0.1, eps=0.0):
    """Soft empirical forward-invariance penalty.

    h: [...,] current value; h_next: same shape next value.
    """
    return torch.relu(eps + (1.0 - alpha) * h - h_next).mean()


def gradient_direction_loss(h, grad_h, delta_state, margin=0.3, eps=0.0):
    """Penalize state updates that decrease h near the boundary.

    grad_h: [...,4], delta_state: [...,4] (next state minus current state).
    Only states with h < margin are penalized.
    """
    directional = (grad_h * delta_state).sum(dim=-1)
    active = (h < margin).to(directional.dtype)
    return (active * torch.relu(eps - directional)).mean()


def barrier_loss(h, margin=0.1, eps=0.0):
    """Keep h above a soft margin; this is not a hard CBF guarantee."""
    return torch.relu(margin + eps - h).mean()


def correction_penalty(delta):
    return delta.square().mean()


def action_bound_penalty(action):
    return torch.relu(action.abs() - 1.0).square().mean()


def qp_free_losses(outputs, h, h_next, grad_h, delta_state,
                   alpha=0.1, eps=0.0, correction_coef=1e-3,
                   gradient_coef=1e-3, invariance_coef=1.0):
    """Convenience bundle; task/value/cost PPO losses are added by the trainer."""
    return dict(
        invariance=invariance_coef * forward_invariance_loss(h, h_next, alpha, eps),
        gradient=gradient_coef * gradient_direction_loss(h, grad_h, delta_state, eps=eps),
        correction=correction_coef * correction_penalty(outputs['delta']),
        bound=action_bound_penalty(outputs['action']),
    )
