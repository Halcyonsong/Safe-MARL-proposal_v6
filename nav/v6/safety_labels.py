"""Safety-label and CBF-term helpers shared by V6 training and diagnostics.

The official GCBF+ label is *not* an instantaneous collision classifier:

- current-step unsafe -> negative;
- no unsafe event in the next ``horizon`` steps -> positive;
- otherwise (unsafe within the horizon, but not now) -> ignored (0).

The implementation below follows the forward-window semantics of
``gcbfplus.algo.gcbf_plus.GCBFPlus.safe_mask``.  A trajectory may pass one extra
terminal unsafe mask; it marks the preceding ``horizon`` states unsafe even
though the terminal state itself is not a training row.
"""

from __future__ import annotations

import numpy as np


def forward_safe_mask(unsafe_series: np.ndarray, horizon: int = 32) -> np.ndarray:
    """Return the horizon-safe mask for the first ``T`` rows of a rollout.

    Parameters
    ----------
    unsafe_series:
        Boolean array ``[T + 1, N]``.  Rows ``0..T-1`` are training states;
        row ``T`` is the terminal state reached after the last action (used
        only to mark the tail of the rollout).
    horizon:
        Number of forward steps that must remain safe.

    Returns
    -------
    safe:
        Boolean array ``[T, N]``.  ``safe[t, i]`` is true iff agent ``i`` has
        no unsafe event in ``[t, t + horizon]`` (bounded by ``T``).
    """
    unsafe = np.asarray(unsafe_series, dtype=bool)
    if unsafe.ndim != 2:
        raise ValueError(f'expected [T+1, N] masks, got {unsafe.shape}')
    if unsafe.shape[0] < 1:
        raise ValueError('empty unsafe series')
    horizon = max(0, int(horizon))
    t_rows = unsafe.shape[0] - 1
    safe = np.ones((t_rows, unsafe.shape[1]), dtype=bool)
    for i in range(unsafe.shape[0]):
        # An unsafe event at i invalidates the preceding `horizon` states
        # (including i when i is a real row).  This is the forward-window
        # label: state t is positive only if it survives to t + horizon.
        start = max(0, i - horizon)
        end = min(i, t_rows - 1)
        if end >= start:
            safe[start:end + 1] &= ~unsafe[i][None, :]
    return safe


def horizon_labels(unsafe_series: np.ndarray, horizon: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(labels, safe)`` for the first ``T`` rows of a rollout.

    Labels are ``-1`` (unsafe now), ``+1`` (horizon-safe), or ``0`` (ignored).
    """
    unsafe = np.asarray(unsafe_series, dtype=bool)
    unsafe_rows = unsafe[:-1]
    safe = forward_safe_mask(unsafe, horizon)
    labels = np.where(unsafe_rows, -1.0,
                      np.where(safe, 1.0, 0.0)).astype(np.float32)
    return labels, safe


def instant_labels(unsafe_series: np.ndarray) -> np.ndarray:
    """Instantaneous classifier labels, kept only for controlled ablations."""
    unsafe = np.asarray(unsafe_series, dtype=bool)
    return np.where(unsafe[:-1], -1.0, 1.0).astype(np.float32)


def cbf_terms_numpy(h, grad_h, states, action, mass=0.1,
                    alpha=1.0, eps=0.02):
    """Compute action-space CBF terms for numpy arrays.

    For the DoubleIntegrator ``x=(p,v)`` and ``u`` in the environment action
    space, ``f=[v,0]`` and ``g=[0,I/m]``, hence

        h_dot = grad_p(h) . v + grad_v(h) . u / m.

    Returns ``(h_dot, violation, feasible)`` where ``feasible`` asks whether
    *some* action in the box ``[-1, 1]`` can satisfy ``h_dot + alpha*h >= 0``.
    """
    h = np.asarray(h, dtype=np.float64)
    grad_h = np.asarray(grad_h, dtype=np.float64)
    states = np.asarray(states, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    v = states[..., 2:]
    grad_p = grad_h[..., :2]
    grad_v = grad_h[..., 2:] / float(mass)
    lf = np.sum(grad_p * v, axis=-1)
    h_dot = lf + np.sum(grad_v * action, axis=-1)
    violation = np.maximum(0.0, -h_dot - float(alpha) * h + float(eps))
    # max over u in [-1,1]^2 of grad_v . u = sum(abs(grad_v))
    best_h_dot = lf + np.sum(np.abs(grad_v), axis=-1)
    feasible = (best_h_dot + float(alpha) * h >= 0.0)
    return h_dot, violation, feasible


def cbf_violation_torch(h, grad_h, states, action, mass=0.1,
                        alpha=1.0, eps=0.02):
    """Torch version used by the PPO actor loss.

    All arguments may have arbitrary leading dimensions; the last dimensions
    are ``[... , 4]`` for states/gradients and ``[..., 2]`` for actions.
    ``h`` has shape ``[...]``.
    """
    import torch

    v = states[..., 2:]
    grad_p = grad_h[..., :2]
    grad_v = grad_h[..., 2:] / float(mass)
    h_dot = (grad_p * v).sum(dim=-1) + (grad_v * action).sum(dim=-1)
    violation = torch.relu(-h_dot - float(alpha) * h + float(eps))
    return h_dot, violation


__all__ = [
    'forward_safe_mask', 'horizon_labels', 'instant_labels',
    'cbf_terms_numpy', 'cbf_violation_torch',
]
