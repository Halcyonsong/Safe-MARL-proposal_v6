"""Fixed/analytic safety providers.

These are baselines and ablations, not the official learned-h chain:

- ``FixedAntSafetyProvider``: the official-cost-derived analytic Ant margin;
- ``FixedNavigationSafetyProvider``: the frozen official GCBF+ navigation h.

Both expose the same ``value_and_local_grad`` shape as the learned provider so
the QP-free policy can be compared without changing its interface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from algos.safe_policy.ant_h import finite_difference_grad, margins_from_geom
from .types import SafetyOutput, SafetyPrediction


class FixedAntSafetyProvider:
    kind = 'fixed_ant'
    cost_names = ('wall', 'floor', 'ceil', 'upright')

    def __init__(self, upright_scale: float = 0.2, **_: Any):
        self.upright_scale = float(upright_scale)

    @torch.no_grad()
    def value_and_local_grad(self, geometry) -> SafetyPrediction:
        g = np.asarray(geometry, dtype=np.float64).reshape(-1, 9)
        h_rows = []
        grad_rows = []
        for row in g:
            m = margins_from_geom(row, upright_scale=self.upright_scale)
            h_rows.append(m.vector)
            grad_rows.append(finite_difference_grad(
                row, upright_scale=self.upright_scale))
        h = torch.as_tensor(np.stack(h_rows), dtype=torch.float32)
        grad = torch.as_tensor(np.stack(grad_rows), dtype=torch.float32)
        out = SafetyOutput(
            h_cost=h,
            survival_logit=torch.zeros(h.shape[0], dtype=torch.float32),
            cost_names=self.cost_names,
        )
        return SafetyPrediction(
            output=out,
            grad_cost=grad,
            note='fixed analytic Ant margin; survival risk unavailable',
        )


class FixedNavigationSafetyProvider:
    """Frozen official GCBF+ navigation h (baseline only)."""

    kind = 'fixed_navigation'
    cost_names = ('collision',)

    def __init__(
        self,
        checkpoint: str | Path,
        env,
        n_agents: int = 8,
        gnn_layers: int = 1,
    ):
        # Imported lazily so Ant-only environments do not need JAX.
        from algos.safe_policy.fixed_h import OfficialFixedH
        self.hnet = OfficialFixedH(
            checkpoint, env, n_agents=n_agents, gnn_layers=gnn_layers)

    def value_and_local_grad(self, graph=None) -> SafetyPrediction:
        h, grad = self.hnet.value_and_local_grad(graph)
        h = torch.as_tensor(np.asarray(h), dtype=torch.float32).reshape(-1, 1)
        grad = torch.as_tensor(np.asarray(grad), dtype=torch.float32)
        out = SafetyOutput(
            h_cost=h,
            survival_logit=torch.zeros(h.shape[0], dtype=torch.float32),
            cost_names=self.cost_names,
        )
        return SafetyPrediction(
            output=out,
            grad_cost=grad,
            note='fixed official GCBF+ navigation h; survival risk unavailable',
        )


__all__ = ['FixedAntSafetyProvider', 'FixedNavigationSafetyProvider']
