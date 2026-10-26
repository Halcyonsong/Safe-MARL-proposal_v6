"""V5 QP-free safe policy package.

The navigation-only JAX fixed-h bridge is imported lazily so the Ant/MuJoCo
environment can use the torch-only policy modules without installing JAX in the
MuJoCo conda environment.
"""

__all__ = ['OfficialFixedH', 'QpFreePolicy']


def __getattr__(name):
    if name == 'OfficialFixedH':
        from .fixed_h import OfficialFixedH
        return OfficialFixedH
    if name == 'QpFreePolicy':
        from .qp_free import QpFreePolicy
        return QpFreePolicy
    raise AttributeError(name)
