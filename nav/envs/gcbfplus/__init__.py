"""Official GCBF+ environment adapters (V5).

The source of truth is vendored under ``third_party/gcbfplus``. This package
only provides a small Python/PyTorch-friendly view of the official environment;
it does not re-implement dynamics, obstacles or LiDAR.
"""

from .double_integrator import OfficialDoubleIntegrator

__all__ = ['OfficialDoubleIntegrator']
