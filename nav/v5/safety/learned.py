"""Torch learned safety heads: shared trunk, cost margins and survival risk.

The network is deliberately QP-label-free.  It is trained from official
cost/survival labels and barrier/invariance regularisers, then frozen before
the QP-free policy is trained.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
from torch import nn

from .types import SafetyOutput, SafetyPrediction


def _mlp(input_dim: int, hidden: Sequence[int], output_dim: int) -> nn.Sequential:
    layers = []
    width = int(input_dim)
    for width_next in hidden:
        layers.extend([
            nn.Linear(width, int(width_next)),
            nn.LayerNorm(int(width_next)),
            nn.SiLU(),
        ])
        width = int(width_next)
    layers.append(nn.Linear(width, int(output_dim)))
    return nn.Sequential(*layers)


class LearnedSafetyModel(nn.Module):
    """Shared safety trunk + cost-margin head + survival-risk head.

    ``h_cost`` is a signed vector with one entry per cost component.  The
    survival head is a terminal-risk classifier, not a CBF value.
    """

    def __init__(
        self,
        state_dim: int,
        cost_names: Sequence[str] = ('wall', 'floor', 'ceil', 'upright'),
        hidden: Sequence[int] = (128, 128),
        survival_hidden: Sequence[int] = (128,),
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.cost_names = tuple(cost_names)
        self.cost_dim = len(self.cost_names)
        self.hidden = tuple(int(v) for v in hidden)
        self.survival_hidden = tuple(int(v) for v in survival_hidden)
        if self.cost_dim < 1:
            raise ValueError('at least one cost head is required')
        self.trunk = _mlp(self.state_dim, self.hidden, int(self.hidden[-1]))
        self.cost_head = nn.Linear(int(self.hidden[-1]), self.cost_dim)
        self.survival_head = _mlp(int(self.hidden[-1]), self.survival_hidden, 1)

    def _batch(self, x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        if x.ndim == 1:
            return x.unsqueeze(0), True
        if x.ndim != 2:
            raise ValueError(f'expected state [D] or [B,D], got {tuple(x.shape)}')
        return x, False

    def forward(self, x: torch.Tensor) -> SafetyOutput:
        x, _ = self._batch(x)
        z = self.trunk(x)
        h_cost = self.cost_head(z)
        survival_logit = self.survival_head(z).squeeze(-1)
        return SafetyOutput(
            h_cost=h_cost,
            survival_logit=survival_logit,
            cost_names=self.cost_names,
        )

    def value_and_local_grad(
        self,
        x: torch.Tensor,
        create_graph: bool = False,
    ) -> SafetyPrediction:
        """Return safety output and d min(h_cost) / d state.

        The returned output is always batched ([1,C] for a single state).
        """
        x_in, _ = self._batch(x)
        x_req = x_in.detach().clone().requires_grad_(True)
        out = self.forward(x_req)
        grad = torch.autograd.grad(
            out.h_cost_scalar.sum(), x_req, create_graph=create_graph)[0]
        return SafetyPrediction(output=out, grad_cost=grad)

    def safety_loss_terms(
        self,
        x: torch.Tensor,
        cost_labels: torch.Tensor,
        terminal: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Convenience terms; the trainer owns the final weighting."""
        out = self.forward(x)
        safe = cost_labels <= 0
        unsafe = cost_labels > 0
        safe_loss = torch.relu(0.05 - out.h_cost)[safe].mean() if safe.any() else out.h_cost.new_zeros(())
        unsafe_loss = torch.relu(out.h_cost + 0.05)[unsafe].mean() if unsafe.any() else out.h_cost.new_zeros(())
        survival = nn.functional.binary_cross_entropy_with_logits(
            out.survival_logit, terminal.float())
        return dict(
            cost_safe=safe_loss,
            cost_unsafe=unsafe_loss,
            survival=survival,
            h_cost=out.h_cost,
            survival_logit=out.survival_logit,
        )


__all__ = ['LearnedSafetyModel']
