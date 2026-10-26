"""Fixed safety margins for Safe MAMuJoCo Ant-2x4.

The official cost in ``ant.py`` is binary:

    cost = 1 if (wall clearance < 1.8 m) or (torso z outside [0.2, 1.0])
           or (z_rot < -0.7), else 0.

This module keeps those thresholds unchanged.  It exposes a fixed, smooth
margin vector and a scalar risk margin for V5.  The margin is not a CBF
certificate; it is a fixed safety feature/training signal.

The default V5 margin deliberately omits the approach-speed term.  On the
base Ant policy, ``kappa=0.3`` made h negative on ~18% of steps with cost=0;
the threshold-aligned margin is much better calibrated.  Velocity can still be
passed as a separate risk feature by a caller that wants a predictive margin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

TAN30 = float(np.tan(30.0 / 360.0 * 2.0 * np.pi))
YWALL = np.array([-5.0, 5.0], dtype=np.float64)
D_SAFE = 1.8
Z_MIN = 0.2
Z_MAX = 1.0
Z_ROT_MIN = -0.7

# Convert the dimensionless upright margin to a length-like scale so that
# min() over wall/height/upright margins does not mix units.
UPRIGHT_SCALE = 0.2


@dataclass
class AntMargins:
    h_wall: float
    h_floor: float
    h_ceil: float
    h_upright: float
    h: float
    active: int  # 0=wall, 1=floor, 2=ceil, 3=upright

    @property
    def vector(self) -> np.ndarray:
        return np.array(
            [self.h_wall, self.h_floor, self.h_ceil, self.h_upright],
            dtype=np.float32,
        )


def wall_y_dist(x: float, y: float) -> np.ndarray:
    """Return the two signed wall offsets used by the official AntEnv."""
    x = float(x)
    y = float(y)
    if x < 20.0:
        return y - x * TAN30 + YWALL
    if x < 60.0:
        return y + (x - 40.0) * TAN30 - YWALL
    if x < 100.0:
        return y - (x - 80.0) * TAN30 + YWALL
    return y - 20.0 * TAN30 + YWALL


def wall_y_dist_dot(x: float, vx: float, vy: float) -> np.ndarray:
    """Time derivative of ``wall_y_dist``."""
    x = float(x)
    if x < 20.0:
        return np.array([vy - vx * TAN30, vy - vx * TAN30])
    if x < 60.0:
        return np.array([vy + vx * TAN30, vy + vx * TAN30])
    if x < 100.0:
        return np.array([vy - vx * TAN30, vy - vx * TAN30])
    return np.array([vy, vy])


def geom_from_torso(
    x: float, y: float, z: float,
    vx: float, vy: float, vz: float,
    z_rot: float,
) -> np.ndarray:
    """Build the 9-D geometric state used by the fixed Ant margins.

    Order: d_left, d_right, v_n_left, v_n_right, z, z_rot, vx, vy, vz.
    """
    yd = wall_y_dist(x, y)
    ydot = wall_y_dist_dot(x, vx, vy)
    d_left = float(abs(yd[0]))
    d_right = float(abs(yd[1]))
    v_n_left = float(-np.sign(yd[0]) * ydot[0]) if yd[0] != 0 else float(-ydot[0])
    v_n_right = float(-np.sign(yd[1]) * ydot[1]) if yd[1] != 0 else float(-ydot[1])
    return np.array(
        [d_left, d_right, v_n_left, v_n_right,
         float(z), float(z_rot), float(vx), float(vy), float(vz)],
        dtype=np.float32,
    )


def margins_from_geom(
    x_geom: np.ndarray,
    kappa: float = 0.0,
    kappa_z: float = 0.0,
    upright_scale: float = UPRIGHT_SCALE,
) -> AntMargins:
    """Compute threshold-aligned margins from the 9-D geometric state."""
    g = np.asarray(x_geom, dtype=np.float64).reshape(-1)
    if g.size != 9:
        raise ValueError(f'expected 9-D x_geom, got {g.size}')
    d_left, d_right = g[0], g[1]
    v_n_left, v_n_right = g[2], g[3]
    z, z_rot, vz = g[4], g[5], g[8]
    nearest_left = d_left <= d_right
    d_wall = d_left if nearest_left else d_right
    v_n = v_n_left if nearest_left else v_n_right

    h_wall = d_wall - D_SAFE - float(kappa) * max(float(v_n), 0.0)
    h_floor = (z - Z_MIN) - float(kappa_z) * max(-float(vz), 0.0)
    h_ceil = Z_MAX - z
    h_upright = float(upright_scale) * (z_rot - Z_ROT_MIN)
    vals = np.array([h_wall, h_floor, h_ceil, h_upright], dtype=np.float64)
    active = int(np.argmin(vals))
    return AntMargins(
        h_wall=float(h_wall),
        h_floor=float(h_floor),
        h_ceil=float(h_ceil),
        h_upright=float(h_upright),
        h=float(vals.min()),
        active=active,
    )


def margins_from_env(env, **kwargs) -> tuple[np.ndarray, AntMargins]:
    """Read the official Ant torso state and return (x_geom, margins)."""
    torso = env.get_body_com('torso')
    qvel = env.sim.data.qvel
    quat = env.data.get_body_xquat('torso')
    z_rot = float(1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2))
    x_geom = geom_from_torso(
        torso[0], torso[1], torso[2],
        qvel[0], qvel[1], qvel[2], z_rot,
    )
    return x_geom, margins_from_geom(x_geom, **kwargs)


def finite_difference_grad(
    x_geom: np.ndarray,
    eps: float = 1e-4,
    **kwargs,
) -> np.ndarray:
    """Finite-difference d h / d x_geom.

    The margin is piecewise smooth because it is a min of four terms.  The
    returned gradient is one valid local subgradient for almost every state.
    """
    x = np.asarray(x_geom, dtype=np.float64).reshape(-1)
    grad = np.zeros_like(x)
    h0 = margins_from_geom(x, **kwargs).h
    for i in range(x.size):
        xp = x.copy()
        xp[i] += eps
        grad[i] = (margins_from_geom(xp, **kwargs).h - h0) / eps
    return grad.astype(np.float32)


def official_cost(
    x_geom: np.ndarray,
    z_rot: Optional[float] = None,
) -> int:
    """Official binary cost condition (debug/evaluation only)."""
    g = np.asarray(x_geom, dtype=np.float64).reshape(-1)
    d_wall = min(float(g[0]), float(g[1]))
    z = float(g[4])
    zr = float(g[5]) if z_rot is None else float(z_rot)
    return int(d_wall < D_SAFE or z < Z_MIN or z > Z_MAX or zr < Z_ROT_MIN)


__all__ = [
    'AntMargins', 'D_SAFE', 'UPRIGHT_SCALE', 'YWALL', 'Z_MAX', 'Z_MIN',
    'Z_ROT_MIN', 'finite_difference_grad', 'geom_from_torso',
    'margins_from_env', 'margins_from_geom', 'official_cost',
    'wall_y_dist', 'wall_y_dist_dot',
]
