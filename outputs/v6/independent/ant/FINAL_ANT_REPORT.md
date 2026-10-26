# V6 Ant report

Date: started 2026-10-24, updated 2026-10-25 with the completed 10M 3-seed
sweep (Asia/Shanghai).  This report is intentionally self-contained;
the full experimental log and the V5/V6 comparison are in
`PILOT_DIAGNOSIS.md`.

## 1. What was wrong with the old V6 baseline

The old 1M pilot was not a V5-equivalent pure MAPPO baseline.  The fixes were:

- the critic now sees the upstream normalized 31-D `[state, agent_id]` state,
  not raw 29-D simulator state;
- the PPO log-probability shape bug is fixed (`old_log_probs` was broadcasting
  to `(B, B)`), and `act_single()` returns one log-prob per agent;
- ValueNorm, feature/LayerNorm, ReLU, Gaussian std≈0.365, 5 PPO epochs,
  Huber value loss, separate actor/critic optimizers and the V5 learning rates
  are enabled;
- 8 parallel environments x 2048 steps = 16,384 transitions/update;
- no-h `task_only`/`cost_only` arms use `NullAntSafetyProvider` (no
  finite-difference h);
- checkpoint loading restores the saved activation (old checkpoints fall back
  to the policy config);
- the actor/critic keep independent gradient clipping and optimizer steps.

## 2. Corrected pure-task baseline: healthy

`mappo_task_base/seed0/policy_final.pth`, 10M steps, 100 deterministic
episodes (seed 20000):

| metric | value |
|---|---:|
| forward_return | 4496.85 |
| mean_x | 111.69 |
| final_x | 224.85 |
| episode_return | 5041.77 |
| wall_episode | 0.99 |
| topple_episode | 0.05 |
| cost_fraction | 0.1323 |
| survived | 0.95 |
| action_abs | 0.2708 |

Stochastic 100-episode: `forward_return=3620.85`, `final_x=181.05`,
`cost_fraction=0.1533`, `survived=0.75`.  This matches the V5 pure-task forward
level (V5: 4477 / x=203) and is the valid control.  The wall-episode rate is
higher than the V5 reference, so safety is still unresolved for the pure task.

## 3. Static/dynamic h branch: terminated; predictive follow-up passes 2M/10M

The h safety network itself remains good offline (validation BA 0.97+,
unsafe recall 0.97, grad cosine 0.926).  The closed-loop h/grad actor features
did not produce a robust benefit:

- h scale 1, p0.1: at 1M deterministic `safe=0.91, cost_fraction=0.015,
  forward=280.7`, but by 2.5M `safe=0, cost_fraction=0.456`;
- h scale 3, p0.1: at 0.5M `safe=0.85, cost_fraction=0.009`, but at 1M
  `safe=0.10, cost_fraction=0.722`;
- scheduled cost penalties (0.1 -> 0.5/1.0 by 2M) with scale 3 were worse than
  the matched cost-only control;
- alternating stronger penalties made scale-1 h stand still.

Decisive matched 2M comparison, 100 deterministic episodes:

| arm | forward_return | final_x | safe_episode | wall_episode | cost_fraction |
|---|---:|---:|---:|---:|---:|
| task_only | 1003.66 | 50.19 | 0.00 | 1.00 | 0.2646 |
| cost_only schedule 1.0 | 840.11 | 42.01 | 0.90 | 0.10 | 0.0044 |
| h(scale1) schedule 0.5 | 814.45 | 40.73 | 0.00 | 1.00 | 0.5651 |
| h(scale1) schedule 1.0 | 957.99 | 47.91 | 0.38 | 0.62 | 0.0256 |

The best h arm buys 14% more forward than the cost-only control, but with 6x
the per-step cost and much lower `safe_episode` (0.38 vs 0.90).  Under
stochastic actions the same ordering holds (cost-only `safe=0.14,
cost_fraction=0.0721`; h schedule 1.0 `safe=0.04, cost_fraction=0.1035`).

Therefore the h branch is stopped.  No 3-seed h scaling and no navigation
task-joint run are justified by this evidence.  The cost-only scheduled
penalty is a safe **control**, not an h result.

### Dynamic-h / h-barrier follow-up (also terminated)

The static-margin audit showed that h ignored approach velocity and that
future-unsafe labels were ignored.  The follow-up recomputed the V6 h target
with `kappa=0.5` and `kappa_z=0.2`, and added a fixed soft barrier
`2.0 * max(0, 0.1 - h_next)`.  The dynamic h is offline good (test BA 0.9612,
unsafe recall 0.9666, false alarm 0.0275, gradient cosine 0.870; the velocity
gradient is nonzero), but the closed-loop result is worse:

| 1M, 100 episodes | mode | forward | final_x | safe_episode | wall_episode | cost_fraction |
|---|---|---:|---:|---:|---:|---:|
| cost-only | det | 344.73 | 17.24 | 0.29 | 0.70 | 0.0784 |
| static h | det | 243.06 | 12.16 | 0.77 | 0.23 | 0.0782 |
| dynamic h | det | 509.06 | 25.46 | 0.09 | 0.91 | 0.2666 |
| dynamic h + barrier | det | -2.05 | -0.10 | 1.00 | 0.00 | 0.0000 |
| cost-only | sto | 151.78 | 7.59 | 0.04 | 0.76 | 0.2279 |
| dynamic h | sto | 231.46 | 11.58 | 0.00 | 0.81 | 0.2357 |
| dynamic h + barrier | sto | 54.72 | 2.74 | 0.17 | 0.54 | 0.1691 |

At 1.5M (20 deterministic episodes) dynamic h reached `forward=844.52`,
`cost_fraction=0.3514`, `wall_episode=1.00`, while the matched cost-only
control was `forward=476.07`, `cost_fraction=0.0464`, `safe=0.40`.  The barrier
arm was safe only by standing still (`forward=13.61`).  The runs were stopped:
no 3-seed, 10M or navigation task-joint test is justified.  Full details are in
`DYNAMIC_H_GATE.md` and `EFFECT_ANALYSIS.md`.

### Cost-only scheduled control at 10M

The cost-only control (no h) completes at 10M and is the matched no-h control
for the predictive gate:

| mode | forward_return | mean_x | final_x | safe_episode | wall_episode | cost_fraction | survived |
|---|---:|---:|---:|---:|---:|---:|---:|
| deterministic | 4055.37 | 95.78 | 202.77 | 0.10 | 0.89 | 0.0222 | 0.98 |
| stochastic | 3067.39 | 72.97 | 153.38 | 0.04 | 0.91 | 0.0508 | 0.81 |

It retains 90% of pure-task forward with 6x lower per-step cost.  `safe_episode`
is still low because any wall contact makes an episode unsafe; `cost_fraction`
is the more informative per-step measure.

### Safety-gated forward reward (partial, not robust)

A final feedback-coupling experiment multiplied only the forward component by
`s(h_next) = clip(h_next/0.3, 0, 1)` during training, keeping the official
cost penalty and no h barrier.  Three matched seeds at 2M, 100 episodes:

| mode | arm | forward | safe | wall | topple | cost_fraction |
|---|---|---:|---:|---:|---:|---:|
| det mean | cost-only | 905.75 | 0.573 | 0.420 | 0.010 | 0.0199 |
| det mean | forward gate | 799.82 | 0.693 | 0.250 | 0.060 | 0.0223 |
| sto mean | cost-only | 525.38 | 0.067 | 0.603 | 0.563 | 0.0987 |
| sto mean | forward gate | 493.73 | 0.180 | 0.437 | 0.490 | 0.0432 |

Seed 0 is a clear win (forward +6%, cost 18x lower), seed 2 is much safer but
29% slower, and seed 1 is worse than cost-only on deterministic safety
(`cost 0.0665 vs 0.0430`, `safe 0.12 vs 0.24`).  The deterministic mean cost is
not lower, and forward is 11.7% lower, so the robust gate fails.  Stochastic
cost is substantially lower, which supports the coupling hypothesis but is not
enough to scale.  Do not run 10M or navigation task-joint from the current-h
gate result.  Details: `GATE_2M_RESULTS.md`.

### Predictive-h gate follow-up (passes the 2M robust gate)

The previous gate used current dynamic h and still failed one seed.  The
follow-up retained the old future-unsafe `0` rows, trained h on a 32-step
future minimum margin, and gated forward with the predictive h.

Predictive h offline: test balanced accuracy `0.9145`, unsafe recall `0.8793`,
false alarm `0.0364`, AUROC `0.9511`, gradient cosine `0.9225`.  The h data
contains 114,523 rows (29,791 future/current-unsafe, 84,732 far-safe) and the
old `0` rows are no longer discarded.

Three matched seeds at 2M, 100 episodes:

| mode | arm | forward | safe | wall | topple | cost_fraction |
|---|---|---:|---:|---:|---:|---:|
| det mean | cost-only | 905.75 | 0.573 | 0.420 | 0.010 | 0.0199 |
| det mean | predictive gate | 756.60 | **0.977** | **0.023** | **0.000** | **0.00232** |
| sto mean | cost-only | 525.38 | 0.067 | 0.603 | 0.563 | 0.0987 |
| sto mean | predictive gate | 472.22 | **0.317** | **0.267** | **0.470** | **0.0264** |

The deterministic mean cost falls 88% and safe episodes rise 0.573 -> 0.977,
while forward retains 83.5% of the matched cost-only control.  Stochastic cost
falls 73% and safe episodes rise 0.067 -> 0.317, with 90% forward retained.
All stochastic seeds improve; deterministic seeds 1 and 2 improve both safety
and cost.  Seed 0 is the caveat: its deterministic cost is slightly higher than
its own cost-only control (`0.0069 vs 0.0044`), though safe/wall improve.  The
mean is a clear robust safety win and the seed-1 failure of the current-h gate
is gone.  Details: `PREDICTIVE_H_GATE_2M.md`.

### 10M retention (seed 0, passes)

| steps | mode | arm | forward | safe | wall | topple | cost_fraction |
|---:|---|---|---:|---:|---:|---:|---:|
| 2M | det | cost-only | 840.11 | 0.90 | 0.10 | 0.01 | 0.0044 |
| 2M | det | predictive gate | 572.13 | 0.95 | 0.05 | 0.00 | 0.0069 |
| 5M | det | cost-only | 2284.00 | 0.02 | 0.97 | 0.11 | 0.0700 |
| 5M | det | predictive gate | 1792.69 | 0.99 | 0.01 | 0.00 | 0.00004 |
| 10M | det | cost-only | 4055.37 | 0.10 | 0.89 | 0.02 | 0.0222 |
| 10M | det | predictive gate | 3723.09 | 0.97 | 0.03 | 0.00 | 0.00176 |
| 2M | sto | cost-only | 496.91 | 0.14 | 0.57 | 0.54 | 0.0721 |
| 2M | sto | predictive gate | 402.24 | 0.27 | 0.27 | 0.53 | 0.0306 |
| 5M | sto | cost-only | 1334.04 | 0.00 | 0.80 | 0.52 | 0.0695 |
| 5M | sto | predictive gate | 1207.69 | 0.54 | 0.14 | 0.34 | 0.00525 |
| 10M | sto | cost-only | 3067.39 | 0.04 | 0.91 | 0.19 | 0.0508 |
| 10M | sto | predictive gate | 2923.73 | 0.78 | 0.05 | 0.17 | 0.00419 |

At 10M deterministic the gate retains 91.8% of cost-only forward, cuts cost
by 92%, and raises safe episodes from 0.10 to 0.97.  Stochastic retains 95.3%
of forward, cuts cost by 92%, and raises safe episodes from 0.04 to 0.78.
There is no late safety collapse.  Details: `RETENTION_10M.md`.

## 4. Decision

1. Keep the corrected pure-task baseline as the task reference.
2. Do not scale the old static/dynamic h+grad actor integration, the dynamic-h
   barrier, or the current-h gate.
3. The **predictive-h + safety-gated forward** follow-up passes the three-seed
   2M gate and is the only h mechanism eligible for the next stage.
4. The predictive-h gate **passes the seed-0 and 3-seed 10M retention
   checks**; it is the first h mechanism that remains safe while approaching
   the cost-only task level.  The completed 3-seed sweep gives deterministic
   forward `3771.51` (`91.6%` retention) with safe `0.97` versus cost-only
   `4119.70 / 0.063`, and stochastic forward `3010.73` (`93.9%`) with safe
   `0.84` versus `3206.28 / 0.020`.  See `MULTISEED_10M_PLAN.md`.
5. The dynamic-target + soft-barrier hypothesis remains failed; do not tune it.
6. Cost-only remains the safe control, not an h result.
7. No navigation task-joint run has been started from this result.
8. A post-hoc single-seed task-channel diagnostic (frozen 1-D `A_T` in a 1M
   warm-start from the seed-0 10M gate policy, matched no-V_T control) is
   **closed null**: the V_T gate passed and the control was valid, but the
   warm-start did not beat control on deterministic forward
   (`3960.83` vs `4014.03`) or return (`4574.53` vs `4630.37`), with equal
   safety (`0.96`).  `vt_transfer_supported = false`; the line is stopped and
   the promoted 3-seed result is unchanged.  See
   `STAGE_ANT_VT_TRANSFER.md`.

## 5. Key files

- `nav/v6/ant_mappo.py`: normalized-state MAPPO, Gaussian/bounded actions,
  ValueNorm, separate optimizers, log-prob fix, scheduled penalty support.
- `nav/v6/ant_safety.py`: `NullAntSafetyProvider` for no-h controls.
- `nav/v6/ant_env.py`: 31-D normalized centralized state.
- `nav/configs/v6/v6_ant_task_base.yaml`: corrected pure-task baseline.
- `nav/configs/v6/v6_ant_h_*`, `v6_ant_cost_*`: h/cost experiments and controls.
- `nav/configs/v6/v6_ant_h_dynamic.yaml`, `v6_ant_h_dyn_2m.yaml`,
  `v6_ant_h_dyn_b2m.yaml`: dynamic-h target/barrier follow-up (terminated).
- `nav/configs/v6/v6_ant_h_dyn_gate_2m.yaml`: safety-gated forward follow-up
  (3-seed partial result; see `GATE_2M_RESULTS.md`).
- `nav/configs/v6/v6_ant_h_predictive.yaml`: predictive-h data/target/training.
- `nav/configs/v6/v6_ant_h_pred_gate_2m.yaml`: predictive-h + forward gate
  (3-seed 2M robust pass; see `PREDICTIVE_H_GATE_2M.md`).
- `outputs/v6/independent/ant/h_predictive/`: predictive h checkpoint and
  offline metrics.
- `outputs/v6/independent/ant/PREDICTIVE_H_GATE_2M.md`: three-seed predictive
  gate result.
- `outputs/v6/independent/ant/RETENTION_10M.md`: 2M/5M/10M seed-0 retention;
  the 10M check passes.
- `outputs/v6/independent/ant/STAGE_ANT_VT_TRANSFER.md`: closed-null 1-D `A_T`
  warm-start diagnostic; not part of the promoted claim.
- `outputs/v6/independent/ant/mappo_h_pred_gate_10m/seed0/`: 10M predictive-h
  gate checkpoint.
- `outputs/v6/independent/ant/DYNAMIC_H_GATE.md`: dynamic-h gate details.
- `outputs/v6/independent/ant/GATE_2M_RESULTS.md`: three-seed forward-gate result.
- `outputs/v6/independent/ant/mappo_task_base/seed0/`: pure-task 10M result.
- `outputs/v6/independent/ant/mappo_cost_ramp10_10m/seed0/`: cost-only 10M
  control; the 2M checkpoint is the strongest validated safe checkpoint.
- `outputs/v6/independent/ant/mappo_h_ramp*`: h attempts, stopped.
