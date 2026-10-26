"""LEGACY V5 official learned-h safety package.

This package is the historical V5 safety chain:

    learned h (cost + survival) -> freeze -> QP-free policy

It is retained as a code/design reference only.  The standalone V6 main path
must not load V5 weights, data, checkpoints, or official h; see
``nav/README_V6.md`` and ``nav/v6/``.
"""

from .factory import build_safety_provider

__all__ = ['build_safety_provider']
