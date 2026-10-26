# Predictive-h gate: 2M / 5M / 10M retention

Date: 2026-10-25.  Seed 0 retention (2M/5M/10M) plus the completed 3-seed
10M sweep (seeds 0/1/2).  All formal evaluations use 100 episodes, the
protocol eval seed `20000 + 100*seed`, and episode limit 1000.  The
predictive-h gate is trained for 10M steps; the matched cost-only control is
the 10M scheduled-penalty run.  Gate is training only; evaluation uses the
official task reward and official cost.

## Deterministic retention

| steps | arm | forward | mean_x | final_x | safe | wall | topple | cost_fraction | survived |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2M | cost-only | 840.11 | | 42.01 | 0.90 | 0.10 | 0.01 | 0.0044 | 0.99 |
| 2M | predictive gate | 572.13 | | 28.61 | 0.95 | 0.05 | 0.00 | 0.0069 | 1.00 |
| 5M | cost-only | 2284.00 | | | 0.02 | 0.97 | 0.11 | 0.0700 | 0.89 |
| 5M | predictive gate | 1792.69 | 48.94 | 89.64 | 0.99 | 0.01 | 0.00 | 0.00004 | 1.00 |
| 10M | cost-only | 4055.37 | 95.78 | 202.77 | 0.10 | 0.89 | 0.02 | 0.0222 | 0.98 |
| 10M | predictive gate | 3723.09 | 86.98 | 186.16 | 0.97 | 0.03 | 0.00 | 0.00176 | 1.00 |

At 10M the gate retains **91.8%** of cost-only forward while reducing
`cost_fraction` by **92%** and raising `safe_episode` from `0.10` to `0.97`.
Relative to the pure-task 10M run (`forward=4496.85`, `cost_fraction=0.1323`,
`safe=0.0`), the gate retains **82.8%** of task progress with almost no
official cost.

## Stochastic retention

| steps | arm | forward | safe | wall | topple | cost_fraction | survived |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2M | cost-only | 496.91 | 0.14 | 0.57 | 0.54 | 0.0721 | 0.46 |
| 2M | predictive gate | 402.24 | 0.27 | 0.27 | 0.53 | 0.0306 | 0.47 |
| 5M | cost-only | 1334.04 | 0.00 | 0.80 | 0.52 | 0.0695 | 0.48 |
| 5M | predictive gate | 1207.69 | 0.54 | 0.14 | 0.34 | 0.00525 | 0.66 |
| 10M | cost-only | 3067.39 | 0.04 | 0.91 | 0.19 | 0.0508 | 0.81 |
| 10M | predictive gate | 2923.73 | 0.78 | 0.05 | 0.17 | 0.00419 | 0.83 |

At 10M stochastic the gate retains **95.3%** of cost-only forward, reduces
cost by **92%**, and raises `safe_episode` from `0.04` to `0.78`.

## 3-seed 10M sweep

Seeds 1/2 completed on 2026-10-25 (10M steps each, matched protocol).  Seed 0
is repeated for comparison.

Deterministic, 100 episodes:

| seed | arm | forward | final_x | safe | wall | topple | cost_fraction | survived |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | cost-only | 4055.37 | 202.77 | 0.10 | 0.89 | 0.02 | 0.0222 | 0.98 |
| 0 | predictive gate | 3723.09 | 186.16 | 0.97 | 0.03 | 0.00 | 0.0018 | 1.00 |
| 1 | cost-only | 4111.44 | 205.57 | 0.08 | 0.89 | 0.08 | 0.0264 | 0.92 |
| 1 | predictive gate | 3700.54 | 185.03 | 0.95 | 0.03 | 0.02 | 0.0018 | 0.98 |
| 2 | cost-only | 4192.30 | 209.62 | 0.01 | 0.99 | 0.04 | 0.0401 | 0.96 |
| 2 | predictive gate | 3890.90 | 194.55 | 0.99 | 0.00 | 0.01 | 0.0000 | 0.99 |
| mean | cost-only | 4119.70 | 205.99 | 0.063 | 0.923 | 0.047 | 0.0296 | 0.953 |
| mean | predictive gate | **3771.51** | 188.58 | **0.97** | **0.02** | **0.01** | **0.0012** | 0.99 |

Stochastic, 100 episodes:

| seed | arm | forward | final_x | safe | wall | topple | cost_fraction | survived |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | cost-only | 3067.39 | 153.38 | 0.04 | 0.91 | 0.19 | 0.0508 | 0.81 |
| 0 | predictive gate | 2923.73 | 146.19 | 0.78 | 0.05 | 0.17 | 0.0042 | 0.83 |
| 1 | cost-only | 3094.38 | 154.72 | 0.02 | 0.94 | 0.32 | 0.0548 | 0.69 |
| 1 | predictive gate | 2936.00 | 146.80 | 0.85 | 0.01 | 0.14 | 0.0011 | 0.86 |
| 2 | cost-only | 3457.07 | 172.86 | 0.00 | 1.00 | 0.19 | 0.0798 | 0.81 |
| 2 | predictive gate | 3172.47 | 158.63 | 0.89 | 0.01 | 0.10 | 0.0007 | 0.90 |
| mean | cost-only | 3206.28 | 160.32 | 0.020 | 0.950 | 0.233 | 0.0618 | 0.770 |
| mean | predictive gate | **3010.73** | 150.54 | **0.84** | **0.023** | **0.137** | **0.0020** | 0.863 |

Deterministic retention: **91.6%** forward, **96.0%** cost reduction,
`safe_episode 0.063 -> 0.97`.  Stochastic retention: **93.9%** forward,
**96.8%** cost reduction, `safe_episode 0.02 -> 0.84`.  Every seed is safe
(>= 0.95 deterministic) while every cost-only seed is unsafe (<= 0.10).

## Verdict

- Safety is retained from 2M to 10M; there is no late collapse like the
  cost-only 10M control.
- Task grows over training and remains close to the cost-only control at 10M.
- The 10M retention check **passes for all three seeds**: 91.6% deterministic
  and 93.9% stochastic forward retention, with a 0.97/0.84 safe-episode rate
  versus 0.063/0.02 for cost-only.
- The navigation task-joint run is still open; the 3-seed 2M gate is already
  complete.

## Open items

- No navigation task-joint run yet.
- No explicit actor `hdot` feature and no separate cost critic; the predictive
  horizon h and fixed cost penalty were sufficient for this retention pass.
- Seed 0 2M deterministic cost is still slightly higher than its own cost-only
  control (`0.0069 vs 0.0044`), while safe/wall improve; the 5M/10M checkpoints
  do not show that regression.
