# Safety-gated forward reward: 2M three-seed gate

Date: 2026-10-24.  Matched seed 0/1/2, 2,000,000 steps, 100 evaluation
episodes, eval seed 20000, episode limit 1000.  The gate is applied only during
training and only to the forward component:

```text
forward_gate = clip((h_next - 0.0) / (0.3 - 0.0), 0, 1)
r_train = ctrl + contact + survive + forward_gate * forward
          - cost_penalty * official_cost
```

There is no h barrier, QP, Lagrangian, expert action or action correction.
The actor input is still `obs31 + h + grad9`.  The h checkpoint is the dynamic
h (`kappa=0.5`, `kappa_z=0.2`) from `h_dynamic/ant_h.pkl`.

## Per-seed 100ep deterministic

| seed | arm | forward | final_x | safe | wall | topple | cost_fraction | survived |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | cost-only | 840.11 | 42.01 | 0.90 | 0.10 | 0.00 | 0.0044 | 0.99 |
| 0 | gate | 891.96 | 44.60 | 0.97 | 0.03 | 0.00 | 0.00024 | 1.00 |
| 1 | cost-only | 888.47 | 44.43 | 0.24 | 0.74 | 0.02 | 0.0430 | 0.98 |
| 1 | gate | 801.14 | 40.06 | 0.12 | 0.71 | 0.18 | 0.0665 | 0.82 |
| 2 | cost-only | 988.67 | 49.44 | 0.58 | 0.42 | 0.00 | 0.0124 | 1.00 |
| 2 | gate | 706.35 | 35.32 | 0.99 | 0.01 | 0.00 | 0.00001 | 1.00 |
| **mean** | **cost-only** | **905.75** | | **0.573** | **0.420** | **0.010** | **0.0199** | **0.990** |
| **mean** | **gate** | **799.82** | | **0.693** | **0.250** | **0.060** | **0.0223** | **0.940** |

## Per-seed 100ep stochastic

| seed | arm | forward | safe | wall | topple | cost_fraction | survived |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | cost-only | 496.91 | 0.14 | 0.61 | 0.55 | 0.0721 | 0.47 |
| 0 | gate | 576.75 | 0.24 | 0.46 | 0.38 | 0.0375 | 0.62 |
| 1 | cost-only | 513.46 | 0.02 | 0.63 | 0.61 | 0.1165 | 0.39 |
| 1 | gate | 459.73 | 0.06 | 0.46 | 0.64 | 0.0593 | 0.36 |
| 2 | cost-only | 565.77 | 0.04 | 0.61 | 0.54 | 0.1075 | 0.47 |
| 2 | gate | 444.71 | 0.24 | 0.39 | 0.45 | 0.0328 | 0.55 |
| **mean** | **cost-only** | **525.38** | **0.067** | **0.603** | **0.563** | **0.0987** | **0.443** |
| **mean** | **gate** | **493.73** | **0.180** | **0.437** | **0.490** | **0.0432** | **0.513** |

## Verdict

This is a **partial positive**, not a robust method:

- Seed 0: gate improves both safety and forward (`+6%` forward, cost 18x lower).
- Seed 2: gate is very safe but loses `29%` forward.
- Seed 1: gate is **worse than cost-only** on deterministic safety
  (`cost_fraction 0.0665 vs 0.0430`, `safe 0.12 vs 0.24`) and loses `10%`
  forward; stochastic cost is lower, but forward also drops.
- Deterministic mean: gate lowers wall episodes and raises safe episodes, but
  the mean cost is slightly higher and forward is `11.7%` lower.  The
  mean does not meet “lower cost without forward collapse”.
- Stochastic mean: gate reduces cost by ~56% and raises safe episodes from
  0.067 to 0.180, with `6%` lower forward.  This is the strongest support for
  the coupling hypothesis, but it is not enough to override the deterministic
  seed-1 failure.

The gate therefore does not pass the pre-agreed robust gate.  Do not run 10M,
navigation task-joint, or a 3-seed scaling claim from this result.  The next
mechanism change should address the remaining temporal/late-warning problem:
predictive risk h (keep future-unsafe rows, horizon target, hdot) or an
action-conditioned cost critic.  Repeating the gate-coefficient sweep would
only fit the seed variance.

## Follow-up

The current-h gate above did not pass.  The predictive-h follow-up (retained
future-unsafe rows, 32-step minimum-margin target, gate threshold 0.05) passes
the three-seed 2M gate: deterministic cost `0.0199 -> 0.00232`, safe
`0.573 -> 0.977`, forward `905.75 -> 756.60`; stochastic cost
`0.0987 -> 0.0264`, safe `0.067 -> 0.317`.  See
`PREDICTIVE_H_GATE_2M.md`.

The predictive follow-up also passed the seed-0 10M retention check:
deterministic `forward=3723.09, safe=0.97, cost=0.00176` versus cost-only
`4055.37, 0.10, 0.0222`; stochastic `2923.73, 0.78, 0.00419` versus
`3067.39, 0.04, 0.0508`.  See `RETENTION_10M.md`.
