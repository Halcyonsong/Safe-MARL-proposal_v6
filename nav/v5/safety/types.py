"""Safety-output types shared by learned and fixed h providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch


@dataclass
class SafetyOutput:
    """A batch of safety-head outputs.

    ``h_cost`` is a signed margin vector: non-negative means locally safe for
    that cost component.  ``survival_logit`` is an unnormalised terminal-risk
    score; it is intentionally *not* treated as a CBF value.
    """

    h_cost: torch.Tensor                 # [B, C]
    survival_logit: torch.Tensor         # [B]
    cost_names: Sequence[str]

    def __post_init__(self):
        if self.h_cost.ndim != 2:
            raise ValueError(f'h_cost must be [B,C], got {tuple(self.h_cost.shape)}')
        if self.survival_logit.ndim != 1:
            raise ValueError(
                f'survival_logit must be [B], got {tuple(self.survival_logit.shape)}')
        if self.h_cost.shape[0] != self.survival_logit.shape[0]:
            raise ValueError('batch size mismatch between cost and survival heads')
        if len(self.cost_names) != self.h_cost.shape[1]:
            raise ValueError('cost_names length does not match h_cost width')

    @property
    def h_cost_scalar(self) -> torch.Tensor:
        return self.h_cost.min(dim=-1).values

    @property
    def survival_risk(self) -> torch.Tensor:
        return torch.sigmoid(self.survival_logit)

    @property
    def active_cost(self) -> torch.Tensor:
        return self.h_cost.argmin(dim=-1)

    @property
    def active_name(self) -> list[str]:
        return [self.cost_names[i] for i in self.active_cost.detach().cpu().tolist()]

    def features(self) -> torch.Tensor:
        """Concatenate cost margins and survival risk for a policy input."""
        return torch.cat([self.h_cost, self.survival_risk.unsqueeze(-1)], dim=-1)

    def detached(self) -> 'SafetyOutput':
        return SafetyOutput(
            h_cost=self.h_cost.detach(),
            survival_logit=self.survival_logit.detach(),
            cost_names=self.cost_names,
        )


@dataclass
class SafetyPrediction:
    """Safety output plus the local state gradient of the scalar cost margin."""

    output: SafetyOutput
    grad_cost: torch.Tensor             # [B, state_dim]
    note: Optional[str] = None


def to_numpy_safety(output: SafetyOutput) -> dict:
    """Small convenience for environment adapters and logging."""
    return dict(
        h_cost=np.asarray(output.h_cost.detach().cpu(), dtype=np.float32),
        h_cost_scalar=np.asarray(
            output.h_cost_scalar.detach().cpu(), dtype=np.float32),
        survival_logit=np.asarray(
            output.survival_logit.detach().cpu(), dtype=np.float32),
        survival_risk=np.asarray(
            output.survival_risk.detach().cpu(), dtype=np.float32),
        active=np.asarray(output.active_cost.detach().cpu(), dtype=np.int64),
    )
