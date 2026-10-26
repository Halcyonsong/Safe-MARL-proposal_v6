"""V6-owned safety geometry for Safe MAMuJoCo Ant-2x4.

This module is deliberately independent of the V5 the legacy V5 Ant safety module
implementation.  It only re-states the *official* Ant cost geometry:

    wall clearance  < 1.8 m
    torso z         outside [0.2, 1.0]
    torso upright   z_rot < -0.7

The safety state is a 9-D torso geometry vector.  The four smooth margins are
``wall``, ``floor``, ``ceil`` and ``upright``; their minimum is the scalar
safety value used by the V6 h/actor path.  No task return, expert action, QP
target, or V5 checkpoint is used here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TAN30 = float(np.tan(30.0 / 360.0 * 2.0 * np.pi))
YWALL = np.array([-5.0, 5.0], dtype=np.float64)

D_SAFE = 1.8
Z_MIN = 0.2
Z_MAX = 1.0
Z_ROT_MIN = -0.7

# Convert the dimensionless upright margin to a length-like scale so that the
# min over wall/height/upright is not dominated by different units.
UPRIGHT_SCALE = 0.2

GEOM_DIM = 9
MARGIN_DIM = 4
MARGIN_NAMES = ('wall', 'floor', 'ceil', 'upright')


@dataclass(frozen=True)
class AntMargins:
    h_wall: float
    h_floor: float
    h_ceil: float
    h_upright: float
    h: float
    active: int  # 0=wall, 1=floor, 2=ceil, 3=upright

    @property
    def vector(self) -> np.ndarray:
        return np.asarray(
            [self.h_wall, self.h_floor, self.h_ceil, self.h_upright],
            dtype=np.float32,
        )

    @property
    def components(self) -> dict[str, float]:
        return {
            'wall': float(self.h_wall),
            'floor': float(self.h_floor),
            'ceil': float(self.h_ceil),
            'upright': float(self.h_upright),
        }


def wall_y_dist(x: float, y: float) -> np.ndarray:
    """Signed wall offsets used by the official Ant cost.

    This mirrors the official ``ant.py`` branch conditions exactly, including
    the boundary behaviour at x=20, 60 and 100.
    """
    x = float(x)
    y = float(y)
    if x < 20.0:
        return y - x * TAN30 + YWALL
    if 20.0 < x < 60.0:
        return y + (x - 40.0) * TAN30 - YWALL
    if 60.0 < x < 100.0:
        return y - (x - 80.0) * TAN30 + YWALL
    return y - 20.0 * TAN30 + YWALL


def wall_y_dist_dot(x: float, vx: float, vy: float) -> np.ndarray:
    """Time derivative of :func:`wall_y_dist` along the torso velocity."""
    x = float(x)
    vx = float(vx)
    vy = float(vy)
    if x < 20.0:
        return np.array([vy - vx * TAN30, vy - vx * TAN30], dtype=np.float64)
    if 20.0 < x < 60.0:
        return np.array([vy + vx * TAN30, vy + vx * TAN30], dtype=np.float64)
    if 60.0 < x < 100.0:
        return np.array([vy - vx * TAN30, vy - vx * TAN30], dtype=np.float64)
    return np.array([vy, vy], dtype=np.float64)


def geom_from_torso(
    x: float,
    y: float,
    z: float,
    vx: float,
    vy: float,
    vz: float,
    z_rot: float,
) -> np.ndarray:
    """Build the 9-D safety geometry.

    Order: ``d_left, d_right, v_n_left, v_n_right, z, z_rot, vx, vy, vz``.
    The first four entries are the two wall-distance/speed pairs (the two
    labels are interchangeable; the nearest wall is selected by the margin).
    """
    yd = wall_y_dist(x, y)
    ydot = wall_y_dist_dot(x, vx, vy)
    # ``abs(yd)`` is the distance to one of the two walls.  The derivative of
    # that distance is sign(yd) * ydot; the approach speed is its negative.
    v_n_left = float(-np.sign(yd[0]) * ydot[0]) if yd[0] != 0 else float(-ydot[0])
    v_n_right = float(-np.sign(yd[1]) * ydot[1]) if yd[1] != 0 else float(-ydot[1])
    return np.asarray(
        [
            abs(float(yd[0])),
            abs(float(yd[1])),
            v_n_left,
            v_n_right,
            float(z),
            float(z_rot),
            float(vx),
            float(vy),
            float(vz),
        ],
        dtype=np.float32,
    )


def margins_from_geom(
    geom: np.ndarray,
    *,
    kappa: float = 0.0,
    kappa_z: float = 0.0,
    upright_scale: float = UPRIGHT_SCALE,
) -> AntMargins:
    """Compute the four threshold-aligned safety margins.

    Positive margin = safe side; negative = official cost side.  ``kappa`` and
    ``kappa_z`` add optional approach-speed prediction.  The default is the
    threshold-aligned margin used by the first V6 Ant pilot (``kappa=0``).
    """
    g = np.asarray(geom, dtype=np.float64).reshape(-1)
    if g.size != GEOM_DIM:
        raise ValueError(f'expected {GEOM_DIM}-D geometry, got {g.size}')

    d_left, d_right = float(g[0]), float(g[1])
    v_n_left, v_n_right = float(g[2]), float(g[3])
    z, z_rot, vz = float(g[4]), float(g[5]), float(g[8])

    nearest_left = d_left <= d_right
    d_wall = d_left if nearest_left else d_right
    v_n = v_n_left if nearest_left else v_n_right

    h_wall = d_wall - D_SAFE - float(kappa) * max(v_n, 0.0)
    h_floor = (z - Z_MIN) - float(kappa_z) * max(-vz, 0.0)
    h_ceil = Z_MAX - z
    h_upright = float(upright_scale) * (z_rot - Z_ROT_MIN)

    values = np.asarray([h_wall, h_floor, h_ceil, h_upright],
                        dtype=np.float64)
    active = int(np.argmin(values))
    return AntMargins(
        h_wall=float(h_wall),
        h_floor=float(h_floor),
        h_ceil=float(h_ceil),
        h_upright=float(h_upright),
        h=float(values.min()),
        active=active,
    )


def geometry_from_env(env, *, kappa: float = 0.0, kappa_z: float = 0.0):
    """Read the official Ant torso state and return ``(geom, margins)``.

    ``env`` is the underlying MuJoCo Ant environment (the object exposed as
    ``MujocoMulti.env``), not the V5 wrapper.  It needs ``get_body_com``,
    ``sim.data.qvel`` and ``data.get_body_xquat``.
    """
    torso = np.asarray(env.get_body_com('torso'), dtype=np.float64).reshape(-1)
    qvel = np.asarray(env.sim.data.qvel, dtype=np.float64).reshape(-1)
    quat = np.asarray(env.data.get_body_xquat('torso'), dtype=np.float64).reshape(-1)
    z_rot = float(1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2))
    geom = geom_from_torso(
        torso[0], torso[1], torso[2],
        qvel[0], qvel[1], qvel[2], z_rot,
    )
    margins = margins_from_geom(
        geom, kappa=kappa, kappa_z=kappa_z)
    return geom, margins


def margins_from_env(env, **kwargs) -> tuple[np.ndarray, AntMargins]:
    """Read the official Ant torso state and return ``(geom, margins)``."""
    return geometry_from_env(env, **kwargs)


def finite_difference_grad(
    geom: np.ndarray,
    *,
    eps: float = 1e-4,
    kappa: float = 0.0,
    kappa_z: float = 0.0,
    upright_scale: float = UPRIGHT_SCALE,
) -> np.ndarray:
    """Finite-difference gradient of the scalar min margin w.r.t. geometry."""
    x = np.asarray(geom, dtype=np.float64).reshape(-1)
    if x.size != GEOM_DIM:
        raise ValueError(f'expected {GEOM_DIM}-D geometry, got {x.size}')
    grad = np.zeros_like(x)
    h0 = margins_from_geom(
        x, kappa=kappa, kappa_z=kappa_z,
        upright_scale=upright_scale).h
    for i in range(x.size):
        xp = x.copy()
        xp[i] += float(eps)
        hi = margins_from_geom(
            xp, kappa=kappa, kappa_z=kappa_z,
            upright_scale=upright_scale).h
        grad[i] = (hi - h0) / float(eps)
    return grad.astype(np.float32)


def safety_target(
    geom: np.ndarray,
    *,
    scale: float = 0.5,
    kappa: float = 0.0,
    kappa_z: float = 0.0,
    upright_scale: float = UPRIGHT_SCALE,
) -> float:
    """Bounded h target: ``tanh(min_margin / scale)``."""
    h = margins_from_geom(
        geom, kappa=kappa, kappa_z=kappa_z,
        upright_scale=upright_scale).h
    return float(np.tanh(h / float(scale)))


def safety_target_grad(
    geom: np.ndarray,
    *,
    scale: float = 0.5,
    eps: float = 1e-5,
    kappa: float = 0.0,
    kappa_z: float = 0.0,
    upright_scale: float = UPRIGHT_SCALE,
) -> np.ndarray:
    """Finite-difference gradient of :func:`safety_target`.

    This is a safety-only target direction; it is used to regularise the
    learned h gradient, not as an action-space controller.
    """
    x = np.asarray(geom, dtype=np.float64).reshape(-1)
    y0 = safety_target(
        x, scale=scale, kappa=kappa, kappa_z=kappa_z,
        upright_scale=upright_scale)
    grad = np.zeros_like(x)
    for i in range(x.size):
        xp = x.copy()
        xp[i] += float(eps)
        yp = safety_target(
            xp, scale=scale, kappa=kappa, kappa_z=kappa_z,
            upright_scale=upright_scale)
        grad[i] = (yp - y0) / float(eps)
    return grad.astype(np.float32)


def official_cost_from_geom(geom: np.ndarray) -> int:
    """Instantaneous official Ant cost condition (wall/height/upright)."""
    g = np.asarray(geom, dtype=np.float64).reshape(-1)
    if g.size != GEOM_DIM:
        raise ValueError(f'expected {GEOM_DIM}-D geometry, got {g.size}')
    d_wall = min(float(g[0]), float(g[1]))
    z = float(g[4])
    z_rot = float(g[5])
    return int(d_wall < D_SAFE or z < Z_MIN or z > Z_MAX or z_rot < Z_ROT_MIN)


def forward_safety_labels(
    unsafe: np.ndarray,
    *,
    horizon: int = 32,
) -> np.ndarray:
    """Forward-window labels for one episode.

    ``unsafe[t]`` is the official cost (wall or fall) at state ``t``.  A state
    is negative if unsafe now, positive if no unsafe event occurs in the next
    ``horizon`` states, and ignored (0) if an unsafe event is in the window
    but not now.  This mirrors the navigation V6 forward-safe semantics while
    allowing Ant's terminal/fall cost to mark the preceding window.
    """
    unsafe = np.asarray(unsafe, dtype=bool).reshape(-1)
    horizon = max(1, int(horizon))
    labels = np.zeros(len(unsafe), dtype=np.float32)
    for t in range(len(unsafe)):
        if unsafe[t]:
            labels[t] = -1.0
            continue
        end = min(len(unsafe), t + horizon + 1)
        if not unsafe[t + 1:end].any():
            labels[t] = 1.0
    return labels


def horizon_risk_margins(
    geoms: np.ndarray,
    *,
    horizon: int = 32,
    kappa: float = 0.0,
    kappa_z: float = 0.0,
    upright_scale: float = UPRIGHT_SCALE,
) -> np.ndarray:
    """Minimum future margin over ``horizon`` steps for one episode.

    ``geoms[t]`` is the current torso geometry.  The returned value at ``t``
    is ``min(margin[t], ..., margin[t + horizon])``; it therefore uses only
    the official simulator geometry, never a task return or expert action.
    Values at the end of an episode use the available suffix.
    """
    g = np.asarray(geoms, dtype=np.float64)
    if g.ndim != 2 or g.shape[-1] != GEOM_DIM:
        raise ValueError(
            f'expected geoms [T,{GEOM_DIM}], got {g.shape}')
    horizon = max(1, int(horizon))
    margins = np.asarray([
        margins_from_geom(
            row, kappa=kappa, kappa_z=kappa_z,
            upright_scale=upright_scale).h
        for row in g
    ], dtype=np.float64)
    risk = np.empty_like(margins)
    for t in range(len(margins)):
        risk[t] = float(np.min(margins[t:t + horizon + 1]))
    return risk.astype(np.float32)


def horizon_risk_targets(
    geoms: np.ndarray,
    *,
    scale: float = 0.5,
    horizon: int = 32,
    kappa: float = 0.0,
    kappa_z: float = 0.0,
    upright_scale: float = UPRIGHT_SCALE,
) -> tuple[np.ndarray, np.ndarray]:
    """Bounded future-risk target and raw horizon margin.

    The target is ``tanh(min_future_margin / scale)``.  A state whose future
    window contains an official unsafe event is forced to a negative target so
    that the scalar h predicts risk before the event, not merely after it.
    """
    risk = horizon_risk_margins(
        geoms, horizon=horizon, kappa=kappa, kappa_z=kappa_z,
        upright_scale=upright_scale)
    target = np.tanh(risk / float(scale)).astype(np.float32)
    return target, risk


class FixedAntSafetyProvider:
    """Analytic safety reference used for data collection/diagnostics.

    This is not a learned h and is not the main V6 h arm.  It gives the
    environment a consistent h/grad signal while V6 h data is collected.
    """

    kind = 'fixed_ant'

    def __init__(self, scale: float = 0.5, offset: float = 0.0,
                 kappa: float = 0.0, kappa_z: float = 0.0):
        self.scale = float(scale)
        self.offset = float(offset)
        self.kappa = float(kappa)
        self.kappa_z = float(kappa_z)

    def value_and_grad(self, geom: np.ndarray) -> tuple[float, np.ndarray]:
        h = safety_target(
            geom, scale=self.scale, kappa=self.kappa,
            kappa_z=self.kappa_z)
        grad = safety_target_grad(
            geom, scale=self.scale, kappa=self.kappa,
            kappa_z=self.kappa_z)
        return float(h + self.offset), grad

    def value(self, geom: np.ndarray) -> float:
        return float(safety_target(
            geom, scale=self.scale, kappa=self.kappa,
            kappa_z=self.kappa_z) + self.offset)


class NullAntSafetyProvider:
    """Zero safety features for no-h baselines.

    The official cost and geometry diagnostics remain available; only the
    learned/analytic h and its gradient are replaced by zeros.  This avoids
    paying the finite-difference h cost in ``task_only``/``cost_only``.
    """

    kind = 'none'

    def value_and_grad(self, geom: np.ndarray) -> tuple[float, np.ndarray]:
        return 0.0, np.zeros(GEOM_DIM, dtype=np.float32)

    def value(self, geom: np.ndarray) -> float:
        return 0.0


__all__ = [
    'AntMargins', 'D_SAFE', 'GEOM_DIM', 'MARGIN_DIM', 'MARGIN_NAMES',
    'NullAntSafetyProvider',
    'UPRIGHT_SCALE', 'YWALL', 'Z_MAX', 'Z_MIN', 'Z_ROT_MIN',
    'FixedAntSafetyProvider', 'finite_difference_grad', 'forward_safety_labels',
    'geom_from_torso', 'geometry_from_env', 'horizon_risk_margins',
    'horizon_risk_targets', 'margins_from_env',
    'margins_from_geom', 'official_cost_from_geom', 'safety_target',
    'safety_target_grad', 'wall_y_dist', 'wall_y_dist_dot',
]
