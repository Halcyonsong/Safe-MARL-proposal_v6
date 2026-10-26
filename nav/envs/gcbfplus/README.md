# Official GCBF+ environment (V5 target)

This directory contains the official-aligned DoubleIntegrator adapter.
`double_integrator.py` delegates reset, dynamics, rectangular obstacles,
LiDAR and safety masks to the vendored official JAX environment; it does not
re-implement them.

The adapter deliberately does **not** call the official `step()` reward branch,
because that branch evaluates an LQR `u_ref`. V5 uses task progress instead; the
adapter reuses only the official state transition/collision/LiDAR components.

The source of truth is the official GCBF+ implementation:

- https://github.com/MIT-REALM/gcbfplus
- `DoubleIntegrator` with rectangular obstacles and LiDAR
- `dt=0.03`, `max_step=256`, `car_radius=0.05`, `comm_radius=0.5`,
  `n_rays=32`, `n_obs=8`, `obs_len_range=[0.1,0.5]`, `mass=0.1`
- action bounds `[-1,1]`, velocity limits `[-0.5,0.5]`, `area_size=4`

The adapter exposes a 92-dim per-agent policy view:

- own state (4), goal-relative state (4);
- 5 nearest agents relative position/velocity (20);
- 32 LiDAR beams relative hit position (64).

Official no-hit LiDAR beams use a `1e6` sentinel. Like the official graph mask,
the adapter zeroes beams with hit distance `>= comm_radius - 0.1`; feeding the
sentinel to the policy would destroy value/cost learning.

The adapter returns official `safe_mask`, `unsafe_mask`, `finish_mask` and
official collision cost in `info`. It exposes no public `u_ref` method.

The legacy `envs/navigation` adapter must not be used for V5 main results.
Do not re-design the environment; vendor the official implementation or port it
with fixed-seed trajectory parity tests.
