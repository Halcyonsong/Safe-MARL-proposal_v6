# V6 Ant dynamic-h closed-loop gate (terminated)

Date: 2026-10-24.  All numbers are matched seed 0, eval seed 30000, 100
episodes, 1000-step limit.  This is a **negative** follow-up, not a new safe
result.

## What changed

1. The h target now includes approach speed (dynamic margin):
   `d - 1.8 - kappa * max(v_n, 0)` for walls and `z - 0.2 - kappa_z *
   max(-vz, 0)` for the floor, with `kappa=0.5`, `kappa_z=0.2`.  The dynamic
   target was recomputed from the existing V6 h geometry; no V5 data, label,
   action or checkpoint was used.
2. A soft h barrier was added to the MAPPO reward:
   `- h_barrier_coef * max(0, h_barrier_margin - h_next)` with coefficient 2.0
   and margin 0.1.  This is a fixed reward term; there is no QP, action
   correction or Lagrangian multiplier.

Future-unsafe labels were left unchanged (still ignored) in this follow-up.
The dynamic target and explicit barrier failed before the future-window label
variable was worth testing.

The dynamic h network remains good offline: test balanced accuracy 0.9612,
unsafe recall 0.9666, false-alarm 0.0275, gradient cosine 0.870.  Increasing
both approach-speed inputs by 2 changed h from `0.135` to `-0.281`
(`delta=-0.415`), and the velocity-dimension gradient is nonzero.  The
velocity blindness of the old static h was fixed.

## 1M matched gate (100 episodes)

| arm | mode | forward | final_x | safe | wall | topple | cost_frac | survived |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| cost-only | det | 344.73 | 17.24 | 0.29 | 0.70 | 0.01 | 0.0784 | 0.99 |
| static h | det | 243.06 | 12.16 | 0.77 | 0.23 | 0.00 | 0.0782 | 1.00 |
| dynamic h | det | 509.06 | 25.46 | 0.09 | 0.91 | 0.00 | 0.2666 | 1.00 |
| dynamic h + barrier | det | -2.05 | -0.10 | 1.00 | 0.00 | 0.00 | 0.0000 | 1.00 |
| cost-only | sto | 151.78 | 7.59 | 0.04 | 0.76 | 0.44 | 0.2279 | 0.56 |
| static h | sto | 173.15 | 8.66 | 0.00 | 0.86 | 0.51 | 0.2567 | 0.49 |
| dynamic h | sto | 231.46 | 11.58 | 0.00 | 0.81 | 0.50 | 0.2357 | 0.50 |
| dynamic h + barrier | sto | 54.72 | 2.74 | 0.17 | 0.54 | 0.50 | 0.1691 | 0.50 |

Dynamic h alone learned forward faster, but its per-step cost was 3.4x the
matched cost-only control and its deterministic `safe_episode` fell to 0.09.
The barrier eliminated cost only by learning a standstill (`forward=-2.05`).
Neither satisfies “improve safety without collapsing task”.

## 1.5M check (20 deterministic episodes)

| arm | forward | final_x | safe | wall | cost_frac |
|---|---:|---:|---:|---:|---:|
| cost-only, same schedule | 476.07 | 23.81 | 0.40 | 0.45 | 0.0464 |
| dynamic h | 844.52 | 42.23 | 0.00 | 1.00 | 0.3514 |
| dynamic h + barrier | 13.61 | 0.68 | 1.00 | 0.00 | 0.0000 |

The no-barrier arm reached task parity with the final cost-only control while
wall contact became universal; the barrier arm remained stationary.  The runs
were stopped at 1.5M/2.0M rather than spending three seeds or 10M on a
mechanism that had already failed both sides of the gate.

## Decision

Terminate the dynamic-h and h-barrier branch.  Do not run 3 seeds, 10M, or the
navigation task-joint experiment with this mechanism.  The validated
task/safety control remains the cost-only scheduled penalty (2M checkpoint:
`forward=840.11`, `safe=0.90`, `cost_fraction=0.0044`; 10M:
`forward=4055.37`, `safe=0.10`, `cost_fraction=0.0222`).
