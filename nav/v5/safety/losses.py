"""Safety-only learned-h losses.

No task return and no QP target appear in this module.  Cost labels come from
the official environment; survival labels are terminal/full-episode outcomes.
"""

from __future__ import annotations

from typing import Optional

import torch


def cost_margin_loss(
    h_cost: torch.Tensor,
    cost_labels: torch.Tensor,
    margin: float = 0.05,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Signed-margin classification loss for each cost component.

    ``cost_labels`` is [B,C] with 1 for an unsafe/costly transition.
    """
    if h_cost.shape != cost_labels.shape:
        raise ValueError(
            f'h_cost {tuple(h_cost.shape)} != cost_labels {tuple(cost_labels.shape)}')
    unsafe = cost_labels > 0
    safe = ~unsafe
    safe_loss = torch.relu(float(margin) - h_cost)[safe].mean() \
        if safe.any() else h_cost.new_zeros(())
    unsafe_loss = torch.relu(h_cost + float(margin))[unsafe].mean() \
        if unsafe.any() else h_cost.new_zeros(())
    return safe_loss + unsafe_loss, dict(safe=safe_loss, unsafe=unsafe_loss)


def survival_bce_loss(
    survival_logit: torch.Tensor,
    terminal: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Terminal-risk BCE; terminal=1 means survival failure."""
    target = terminal.to(survival_logit.dtype).reshape_as(survival_logit)
    return torch.nn.functional.binary_cross_entropy_with_logits(
        survival_logit, target, pos_weight=pos_weight)


def barrier_loss(h: torch.Tensor, margin: float = 0.05) -> torch.Tensor:
    return torch.relu(float(margin) - h).mean()


def forward_invariance_loss(
    h: torch.Tensor,
    h_next: torch.Tensor,
    alpha: float = 0.1,
    eps: float = 0.0,
) -> torch.Tensor:
    return torch.relu(float(eps) + (1.0 - float(alpha)) * h - h_next).mean()


def gradient_direction_loss(
    h: torch.Tensor,
    grad_h: torch.Tensor,
    delta_state: torch.Tensor,
    margin: float = 0.1,
    eps: float = 0.0,
) -> torch.Tensor:
    directional = (grad_h * delta_state).sum(dim=-1)
    active = (h < float(margin)).to(directional.dtype)
    return (active * torch.relu(float(eps) - directional)).mean()


def calibration_loss(
    survival_risk: torch.Tensor,
    terminal: torch.Tensor,
) -> torch.Tensor:
    """Brier score for terminal risk (diagnostic auxiliary)."""
    target = terminal.to(survival_risk.dtype).reshape_as(survival_risk)
    return (survival_risk - target).square().mean()


def safety_h_total_loss(
    h_cost: torch.Tensor,
    survival_logit: torch.Tensor,
    cost_labels: torch.Tensor,
    terminal: torch.Tensor,
    *,
    margin: float = 0.05,
    survival_coef: float = 1.0,
    calibration_coef: float = 0.1,
    h: Optional[torch.Tensor] = None,
    h_next: Optional[torch.Tensor] = None,
    grad_h: Optional[torch.Tensor] = None,
    delta_state: Optional[torch.Tensor] = None,
    barrier_coef: float = 0.5,
    invariance_coef: float = 0.5,
    direction_coef: float = 0.2,
    alpha: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """One learned-h objective with cost and survival in separate terms."""
    cost, cost_parts = cost_margin_loss(h_cost, cost_labels, margin)
    survival = survival_bce_loss(survival_logit, terminal)
    risk = torch.sigmoid(survival_logit)
    calibration = calibration_loss(risk, terminal)
    total = cost + float(survival_coef) * survival \
        + float(calibration_coef) * calibration
    metrics = {
        'cost_safe': float(cost_parts['safe'].detach()),
        'cost_unsafe': float(cost_parts['unsafe'].detach()),
        'survival': float(survival.detach()),
        'calibration': float(calibration.detach()),
    }
    if h is not None and barrier_coef:
        term = barrier_loss(h, margin)
        total = total + float(barrier_coef) * term
        metrics['barrier'] = float(term.detach())
    if h is not None and h_next is not None and invariance_coef:
        term = forward_invariance_loss(h, h_next, alpha)
        total = total + float(invariance_coef) * term
        metrics['invariance'] = float(term.detach())
    if h is not None and grad_h is not None and delta_state is not None and direction_coef:
        term = gradient_direction_loss(h, grad_h, delta_state)
        total = total + float(direction_coef) * term
        metrics['direction'] = float(term.detach())
    metrics['total'] = float(total.detach())
    return total, metrics


__all__ = [
    'barrier_loss', 'calibration_loss', 'cost_margin_loss',
    'forward_invariance_loss', 'gradient_direction_loss',
    'safety_h_total_loss', 'survival_bce_loss',
]
