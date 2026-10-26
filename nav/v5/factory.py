"""Factory for the official learned-h provider and fixed-h baselines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .safety.checkpoint import LearnedSafetyProvider
from .safety.fixed import FixedAntSafetyProvider, FixedNavigationSafetyProvider


def build_safety_provider(
    kind: str,
    *,
    checkpoint: str | Path | None = None,
    device: str = 'cpu',
    **kwargs: Any,
):
    """Build the environment-facing safety provider.

    ``kind='learned'`` is the official V5 path.  Fixed providers are baselines
    and ablations only.
    """
    kind = str(kind).lower()
    if kind == 'learned':
        if checkpoint is None:
            raise ValueError('learned h requires checkpoint=...')
        return LearnedSafetyProvider.from_checkpoint(checkpoint, device=device)
    if kind == 'fixed_ant':
        return FixedAntSafetyProvider(**kwargs)
    if kind == 'fixed_navigation':
        if checkpoint is None:
            raise ValueError('fixed navigation h requires checkpoint=...')
        if 'env' not in kwargs:
            raise ValueError('fixed navigation h requires env=...')
        return FixedNavigationSafetyProvider(checkpoint, **kwargs)
    raise ValueError(f'unknown safety provider kind: {kind!r}')


__all__ = ['build_safety_provider']
