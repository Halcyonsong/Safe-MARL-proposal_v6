"""Checkpoint and provider plumbing for learned safety heads."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .learned import LearnedSafetyModel
from .types import SafetyPrediction


SCHEMA = 'v5.safety_h.v1'


def save_learned_safety(
    path: str | Path,
    model: LearnedSafetyModel,
    config: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(
        schema=SCHEMA,
        state_dict=model.state_dict(),
        state_dim=model.state_dim,
        cost_names=list(model.cost_names),
        model_config=dict(
            hidden=list(model.hidden),
            survival_hidden=list(model.survival_hidden)),
        config=config or {},
    ), path)
    return path


def load_learned_safety(
    path: str | Path,
    device: str | torch.device = 'cpu',
) -> tuple[LearnedSafetyModel, dict[str, Any]]:
    ckpt = torch.load(Path(path), map_location='cpu', weights_only=False)
    schema = ckpt.get('schema')
    if schema != SCHEMA:
        raise ValueError(f'expected {SCHEMA!r}, got {schema!r}')
    cfg = ckpt.get('config') or {}
    model_cfg = cfg.get('model', {}) if isinstance(cfg, dict) else {}
    saved_model = ckpt.get('model_config') or {}
    model = LearnedSafetyModel(
        state_dim=int(ckpt['state_dim']),
        cost_names=tuple(ckpt['cost_names']),
        hidden=saved_model.get('hidden', model_cfg.get('hidden', (128, 128))),
        survival_hidden=saved_model.get(
            'survival_hidden', model_cfg.get('survival_hidden', (128,))),
    )
    model.load_state_dict(ckpt['state_dict'])
    model.to(torch.device(device))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, ckpt


class LearnedSafetyProvider:
    """Frozen learned-h provider consumed by the policy/environment.

    ``value_and_local_grad`` expects a state vector (or batch of vectors) and
    returns the batched SafetyOutput plus d h_scalar / d state.  The provider
    never receives return gradients.
    """

    kind = 'learned'

    def __init__(
        self,
        model: LearnedSafetyModel,
        device: str | torch.device = 'cpu',
        metadata: dict[str, Any] | None = None,
    ):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.metadata = dict(metadata or {})
        self.cost_names = tuple(model.cost_names)

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        device: str | torch.device = 'cpu',
    ) -> 'LearnedSafetyProvider':
        model, ckpt = load_learned_safety(path, device=device)
        return cls(model, device=device, metadata=ckpt)

    @torch.no_grad()
    def value(self, state) -> torch.Tensor:
        x = torch.as_tensor(np.asarray(state), dtype=torch.float32,
                            device=self.device).reshape(1, -1)
        return self.model(x).h_cost_scalar[0]

    def value_and_local_grad(self, state) -> SafetyPrediction:
        x = torch.as_tensor(np.asarray(state), dtype=torch.float32,
                            device=self.device).reshape(1, -1)
        with torch.enable_grad():
            return self.model.value_and_local_grad(x)

    def predict(self, states) -> torch.Tensor:
        x = torch.as_tensor(np.asarray(states), dtype=torch.float32,
                            device=self.device)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        with torch.no_grad():
            return self.model(x).h_cost


__all__ = [
    'LearnedSafetyProvider', 'SCHEMA', 'load_learned_safety',
    'save_learned_safety',
]
