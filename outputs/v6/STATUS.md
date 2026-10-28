# V6 project status

Last updated: 2026-10-28 (Asia/Shanghai).  This is the canonical status
snapshot.  Detailed experiment records are linked below; raw artifacts stay in
their original run directories.

## Bottom line

The promoted pre-registered experiments are complete.  The 2026-10-28
follow-ups close the budget and task questions as **trade-offs, not free
improvements**: A-only 28k loses task and is not promoted; the W16 task
warm-start recovers task strongly but fails the safety/spread gate and is not
promoted; the Ant V_T gradient interface is null on task (with a safety side
effect); neither fixed MAPPO penalty rescues the matched baseline.  The
density screen finds no 28k robustness benefit.  T0w/W16 are retained as
task-first trade-off points, not safety-non-inferiority passes.

- **Ant:** the predictive-h + safety-gated forward mechanism passes the
  matched 3-seed 10M retention check.  A from-scratch 10M MAPPO-Lag external
  screen (single seed) matches deterministic episode safety but is ~19% lower
  on forward return; stochastic V6 is better on both safety and task.  This
  is a reference, not a promoted row.  The Ant V_T gradient follow-up is
  **null on task** (det forward `+0.14%` vs control) although it improves
  safety/cost in the single seed.
- **Navigation:** the repaired Stage-B predictive-h gate passes the
  pre-registered 3-seed guardrails.  Safety improves clearly; task does not
  collapse on average.  The A-only 28k extension is task-costly (mean success
  `0.2767 < 0.36`) and not promoted.  The W16 task warm-start recovers task
  (mean success `0.6175`) but fails mean safety/official and spread; it is a
  task-first trade-off, not a promoted row.
- **Navigation attribution (P1, completed 2026-10-27):** the repaired-h 2x2
  shows actor h input is load-bearing at 16k from scratch; the gate is
  ineffective without it; within the repaired-h 2x2, only repaired h + gate
  combines safety with retained task.  The headline `+24.3 pp` safe delta decomposes descriptively
  into `+18.3 pp` from repaired h (gate off, but at a task cost) and
  `+6.0 pp` from the progress gate (which recovers task).
- **Baselines:** the navigation arm has been compared with the internal
  h-only/gate ablations and with archived V5/CBF/Safe-MARL references.
- **h2/actor2 diagnostic:** complete and read-only.  The impact-state rule
  gives actor/action-limited, but risk-region analysis shows a temporal h
  limitation consistent with h1's actor0-data mismatch.  Overall diagnosis:
  mixed.  Stage C0 (h2-only temporal-calibration screen) was run as a
  seed-0 pilot and **failed**: 17-32-step MAE 0.826 versus h1 0.709 (+16.5%
  worse), and every risk-region metric degraded.  C0b was not run.  The
  separately approved **C1a actor2-from-scratch pilot also failed**: 100ep
  `safe_episode 0.510` vs actor1 `0.840`, `official_safety 0.89375` vs
  `0.97375`, `success 0.33125` vs `0.42750`, `return 52.22` vs `60.91`; all
  five pre-registered checks failed.  The from-scratch h2/actor2 branch is
  stopped.  A diagnostic 4k warm-start of actor1 with h2 (C1w) was run and
  also **failed**: safe `0.840` (= actor1), official safety `0.97625` (below
  the no-training swap `0.985`), success `0.2025` vs `0.4275`, return `28.85`
  vs `60.91`.  The pre-registered h1-continued control was therefore not run
  and the h2 warm-start branch is stopped.  An evaluation-only C1s replication
  (h2 seed 0 on actor1 seeds 1/2) is complete and **`consistent_positive`**:
  mean `Δsafe +0.030`, `Δofficial_safety +0.00875`, `Δsuccess +0.01375`,
  `Δreach +0.00813`, `Δreturn -0.05`; but the safe mean is exactly at the
  pre-registered `+0.03` boundary and the gain is seed-1 driven.  C1s is
  off-policy and neither C1w nor C1s is a C1a pass.

- **Nav task channel (`V_T`):** the T0w 4k warm-start produced a large task
  gain (success `0.67875` vs matched control `0.380`, return `161.88` vs
  `69.67`) but missed the pre-registered scene-safety non-inferiority margin
  (safe `0.890` vs control `0.930`; threshold `0.900`, i.e. one episode), so
  `vt_channel_supported = false` under the safety-first rule.  Task-first
  reading: T0w is a **task-heavy Pareto point** with a small, statistically
  unresolved safety discount (official safety `-0.75 pp`; Wilson intervals
  overlap), not a safety-non-inferiority pass.  The strict safety-first line
  is stopped and no from-scratch arm is authorized yet; a 3-seed from-scratch
  `h1 + gate + V_T` replication is the next step if the task-first objective
  is chosen.  One seed, diagnostic only.

- **Ant task channel (`V_T`):** a single-seed 1M warm-start from the existing
  seed-0 10M h-only base with a frozen 1-D `A_T` was compared with a matched
  no-V_T 1M control.  The V_T gate passed (val EV `0.4838`, test Pearson
  `0.6597`) and the control was valid (det safe `0.96`, cost_fraction
  `0.000187`), but the warm-start did not beat control on the deterministic
  task: forward `3960.83` vs `4014.03` (`-1.33%`), return `4574.53` vs
  `4630.37` (`-1.21%`); safety was equal (`0.96` vs `0.96`).  Stochastic was
  also worse (forward `3247.81` vs `3440.15`, safe `0.73` vs `0.78`).
  `task_gain = false`, `safety_not_worse = true`, so
  `vt_transfer_supported = false`; the **1-D `A_T` interface** is stopped.
  The extra 1M training, not V_T, explains the improvement over base.  One
  seed, diagnostic only; the promoted Ant 3-seed claim is unchanged.  Record:
  `independent/ant/STAGE_ANT_VT_TRANSFER.md`.

- **Ant task-channel gradient follow-up (completed, 2026-10-28):** the 1-D
  null does not test the navigation interface, so a 250k warm-start from the
  same seed-0 10M base fed `[A_T, grad_s V_T]` (32-D: one scalar plus the
  unit-normalized 31-D state gradient) through a matched no-V_T control.
  Deterministic forward `3816.50` vs `3811.20` (`+0.14%`), return `4458.09`
  vs `4449.88` (`+0.18%`), safe `0.99` vs `0.89`, cost_fraction `0.00094`
  vs `0.00661`; stochastic forward `2968.53` vs `2856.21`, safe `0.78` vs
  `0.62`.  The pre-registered `1.05x` task-gain rule fails, so
  `vt_grad_supported=false`; the gradient interface removed the 1-D task
  harm and improved safety in this seed, but did not create a task gain.
  Single seed, diagnostic only; not a promoted claim and not a substitute
  for the nav W16 task gate.  Outputs: `independent/ant_vt_grad_250k/` and
  `independent/ant_vt_grad_control_250k/`; record:
  `independent/ant/STAGE_ANT_VT_TRANSFER.md`.

This is **not** a claim of formal/CBF-certified safety, and it is **not** a
navigation task-channel result.  Navigation task success remains below V5
full, CBF-QP, and GCBF+.

## Promoted results

### Navigation: repaired Stage B, 3 seeds x 100 episodes

Protocol: 16,384 steps/seed, frozen V6 repaired h, gate `low=0.0`,
`high=0.267270`, 100 deterministic episodes, eval seed 10000, fixed final
checkpoint.

| seed | safe_ep | collision_ep | official_safety | success | reach | return |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.840 | 0.160 | 0.9738 | 0.4275 | 0.4350 | 60.91 |
| 1 | 0.510 | 0.490 | 0.9000 | 0.4575 | 0.4963 | 70.92 |
| 2 | 0.730 | 0.270 | 0.9513 | 0.3450 | 0.3575 | 68.74 |
| **mean** | **0.693** | **0.307** | **0.9417** | **0.4100** | **0.4296** | **66.85** |
| **std** | 0.168 | 0.168 | 0.0378 | 0.0583 | 0.0695 | 5.26 |

Matched h-only 3-seed: `safe_episode=0.450`, `official_safety=0.8783`,
`official_success=0.4067`, `official_reach=0.4450`, `episode_return=61.73`.

Delta: safe `+24.3 pp`, official safety `+6.3 pp`, success `+0.3 pp`, reach
`-1.5 pp`, return `+8.3%`.

**P0 correction (2026-10-27):** the original-h + broad-gate 3-seed row reaches
`safe_episode=0.6433`, `official_safety=0.9325`, `official_success=0.4263`,
`official_reach=0.4525`, `episode_return=70.33` (P0).  P0 - h-only is
`+19.3 pp` safe / `+5.4 pp` official safety; A - P0 is `+5.0 pp` safe but
`-1.6 pp` success / `-4.9%` return, with per-seed safe deltas
`+14/+2/-1 pp`.  The repaired-h row does not dominate the original-h gate row
on task.  A and P0 also differ in gate scope/threshold, so A - P0 is not a
clean h-retrain estimate.

**P1 repaired-h 2x2 (completed 2026-10-27):** B = repaired h, gate off;
C = no actor h input, gate on; D = no actor h input, gate off.  Means:
B `0.6333/0.9267/0.3229/55.67`; C `0.0200/0.4987/0.0367/-6.63`;
D `0.0333/0.5479/0.0367/7.21` (safe / official safety / success / return).
Clean contrasts: B - H0 `+18.3 pp` safe but `-8.4 pp` success and `-6.1`
return; A - B `+6.0 pp` safe / `+8.7 pp` success / `+11.2` return; A - C and
B - D `+67.3/+60.0 pp` safe; C - D `-1.3 pp` safe / `0.0 pp` success.  Actor
h input is load-bearing at 16k from scratch (C/D collapse across all six
seeds; D is consistent with the archived no-h fixed-penalty baseline), the gate is
ineffective without h input, and within the repaired-h 2x2 only A combines
safety with retained task (P0 is a separate original-h broad-gate row).
The `+24.3 pp` vs h-only decomposes descriptively as `+18.3 pp` (repaired h,
gate off) + `6.0 pp` (progress gate, h on); "mostly the gate" applies to the
broad gate on the original h, not to the repaired-h row.  `n=3`; no
policy-level significance claim.  Details:
[`independent/nav/BASELINE_COMPARISON.md`](independent/nav/BASELINE_COMPARISON.md)
and [`independent/nav/P1_REPAIRED_H_FACTORIAL.md`](independent/nav/P1_REPAIRED_H_FACTORIAL.md).

Guardrails: all four aggregate thresholds and the no-catastrophic-seed rule
pass.  Records:
[`independent/nav/STAGE_B_REPAIR_MULTISEED.md`](independent/nav/STAGE_B_REPAIR_MULTISEED.md).

Caveats:

- `n=3`; seed-level paired differences are not statistically separated from
  zero (safe `p=0.216`, success `p=0.957`).  The claim is "passes the fixed
  pre-registered guardrails", not "policy-level proven superiority".
- The safety gain is carried by seeds 0/2; seed 1 is nearly neutral
  (`safe +0.02`, `official_safety -0.0025`).
- Task shifts are heterogeneous: seed 0 loses `10.5 pp` success, seeds 1/2
  gain `4.0/7.5 pp`.
- The 3-seed mean falls just below the archived V5 safety target
  (`official_safety >= 0.95`, `safe_episode >= 0.70`); seed 0 alone meets it.
- The formal 16k h-off 3-seed cell (D) is complete: `safe 0.0333`,
  `official_safety 0.5479`, `success 0.0367`, `return 7.21`.  It matches the
  archived no-h fixed-penalty baseline and shows that actor h input is
  load-bearing at this training budget; see
  [`independent/nav/P1_REPAIRED_H_FACTORIAL.md`](independent/nav/P1_REPAIRED_H_FACTORIAL.md).

### Navigation: A-only 28k budget extension (negative task result)

Protocol: A from scratch to 28,672 steps, 3 seeds, same frozen repaired h,
gate, cost, reward and optimizer as the 16k row; 100 deterministic episodes
on eval seeds `10000..10099`; fixed final checkpoint.  The 16k rows are the
same-run checkpoints and are bit-identical to the promoted 16k run.

| budget | mean safe | mean official | mean success | mean return |
|---|---:|---:|---:|---:|
| 16k | 0.6933 | 0.9417 | 0.4100 | 66.85 |
| 24k | 0.7000 | 0.9479 | 0.2704 | 59.97 |
| 28k | 0.7800 | 0.9604 | 0.2767 | 63.45 |

28k per seed: safe `0.86/0.70/0.78`, official `0.9750/0.9450/0.9613`,
success `0.19875/0.40375/0.22750`, return `55.52/84.53/50.29`.

Rule 7.1: mean safe, mean official, mean return, worst safe, safe range
(`0.16`) and worst-seed safe (`0.70`) pass; mean success (`0.2767` vs
`0.36`) and worst-seed success (`0.19875` vs `0.20`) fail.
**28k is not promoted; 16k remains the promoted row.**  Paired 28k-vs-16k:
seed 0 safe `+0.02`, success `-0.2288`, return `-5.39`; seed 1 safe `+0.19`,
success `-0.0538`, return `+13.61`; seed 2 safe `+0.05`, success `-0.1175`,
return `-18.45`.  Mean deltas: safe `+8.7 pp`, official `+1.9 pp`, success
`-13.3 pp`, return `-3.41`.

The task loss is already present at 24k (mean success `0.2704`), while 28k
recovers most of the 24k safety dip (`0.7000 -> 0.7800`) without recovering
task.  This is therefore a **task-costly budget extension / negative task
result**, not a clean monotonic safety-for-task trade at the endpoint.  No
extra 28k seeds, no endpoint switch, and no 24k/28k promotion.  Table:
[`independent/nav/nav_3seed_16k_24k_28k_table.json`](independent/nav/nav_3seed_16k_24k_28k_table.json).
Stage record:
[`independent/nav/STAGE_28K_BUDGET_EXTENSION.md`](independent/nav/STAGE_28K_BUDGET_EXTENSION.md).

### Navigation: W16 task warm-start (3 seeds, task recovered, safety trade-off)

Protocol: 16k checkpoint from the same 28k run, 12,288 steps to 28,672 with
the frozen V_T task channel (`use_task_scalar/grad`, zero-init task block),
formal 100ep at 24k and 28k.  Seeds 1/2 use per-seed V_T trained on that
seed's 16k rollouts.  No C16/C24 control; the from-scratch 24k/28k
checkpoints are the procedure-level control.

| arm | safe | official | success | reach | return |
|---|---:|---:|---:|---:|---:|
| W16 24k mean | 0.7233 | 0.9504 | 0.5958 | 0.6208 | 143.73 |
| W16 28k mean | 0.6633 | 0.9329 | 0.6175 | 0.6525 | 198.57 |
| from-scratch 24k mean | 0.7000 | 0.9479 | 0.2704 | 0.2796 | 59.97 |
| from-scratch 28k mean | 0.7800 | 0.9604 | 0.2767 | 0.2879 | 63.45 |

W16 28k per seed: safe `0.83/0.56/0.60`, official
`0.9725/0.9075/0.9187`, success `0.62125/0.67375/0.55750`, return
`237.36/217.40/140.95`.  Paired W16 28k - from-scratch 28k: seed 0 safe
`-3.0 pp`, success `+42.25 pp`; seed 1 safe `-14.0 pp`, success
`+27.0 pp`; seed 2 safe `-18.0 pp`, success `+33.0 pp`.

Rule 7.1 on the 28k row: mean success (`0.6175`), mean return (`198.57`),
worst success (`0.5575`), worst safe (`0.56`) and cost pass; **mean safe
`0.6633 < 0.6933`, mean official `0.9329 < 0.9417`, and safe range
`0.27 > 0.20` fail**.  Full rule fails.  The 24k diagnostic passes mean
safety/official/task but its safe range is `0.38` and worst seed is `0.55`,
so it also fails the spread rule.

**W16 is a task-first trade-off, not a promotion.**  It recovers task
strongly (`success 0.6175` vs `0.4100` at 16k and `0.2767` at
from-scratch 28k) but loses mean safety relative to 16k and is more
seed-variable, especially seeds 1/2.  Vs promoted 16k: success `+20.75 pp`,
safe `-3.0 pp`, official `-0.88 pp`, return `+131.7`.  Keep 16k promoted;
do not switch to 28k or 24k post hoc.  A safety-regularized task channel or a 5-seed
confirmatory pre-registration is a separate proposal.  Record:
[`independent/nav/STAGE_W16_TASK_WARMSTART.md`](independent/nav/STAGE_W16_TASK_WARMSTART.md).

### Navigation density screen (seed 0, 20 episodes, first block)

4 conditions x A28/A16/H0.  A16 retains substantially more task than A28 in
every condition (`success 0.34-0.39` vs `0.19-0.22`) with comparable or
better all-agent safety in 8/8, 4 obstacles and 16 agents; A28 is safer only
at 16 obstacles (`0.60` vs `0.50`).  H0 has A16-like task but much lower
safety at 16 obstacles/16 agents (`safe 0.25/0.05`).  **No A28
density-robustness benefit; report the task boundary.**  Descriptive
20-episode screens only.  Record:
[`independent/nav/DENSITY_STRESS_TEST.md`](independent/nav/DENSITY_STRESS_TEST.md)
and `independent/nav_density/seed0/summary.json`.

### Ant: predictive-h gate, 3 seeds x 10M

Protocol: 10M steps/seed, 100 deterministic and 100 stochastic episodes,
eval seed `20000 + 100*seed`, matched cost-only control.

| mode | arm | forward | safe_ep | cost_fraction | retention |
|---|---|---:|---:|---:|---:|
| deterministic | cost-only | 4119.70 | 0.063 | 0.0296 | 100% |
| deterministic | predictive gate | **3771.51** | **0.97** | **0.0012** | **91.6%** |
| stochastic | cost-only | 3206.28 | 0.020 | 0.0618 | 100% |
| stochastic | predictive gate | **3010.73** | **0.84** | **0.0020** | **93.9%** |

Every gate seed is safe (`0.97/0.95/0.99` deterministic); every cost-only seed
is unsafe (`0.10/0.08/0.01`).  Record:
[`independent/ant/RETENTION_10M.md`](independent/ant/RETENTION_10M.md).

**External baseline screen (not promoted):** a from-scratch 10M upstream
MAPPO-Lagrangian run (seed 0, 625/625 updates, eval seeds `20000..20099`)
matches V6 seed-0 deterministic episode safety but not task efficiency:

| mode | arm | forward | safe_ep | cost_fraction |
|---|---|---:|---:|---:|
| deterministic | MAPPO-Lag 10M | 3011.02 | 0.97 | 0.0001 |
| deterministic | V6 predictive gate seed 0 | **3723.09** | 0.97 | 0.0018 |
| stochastic | MAPPO-Lag 10M | 2305.82 | 0.66 | 0.0024 |
| stochastic | V6 predictive gate seed 0 | **2923.73** | **0.78** | 0.0042 |

This is a single-seed external screen, not a main-table row: deterministic
safety is equal, V6 keeps `+19.1%` forward; stochastic V6 is better on both
safety (`+12 pp`) and forward (`+21.1%`).  MAPPO-Lag has the lower continuous
cost fraction in both modes.  Record:
[`independent/ant/STAGE_ANT_MAPPO_LAG_10M.md`](independent/ant/STAGE_ANT_MAPPO_LAG_10M.md).

## Baseline positioning (navigation)

| method | official_safety | safe_ep | success | return | controlled? |
|---|---:|---:|---:|---:|---|
| GCBF+ official | 0.9988 | 0.990 | 0.8613 | 480.05 | no (pretrained teacher) |
| centralized CBF-QP | 0.9788 | 0.900 | 0.7813 | 406.82 | no (online QP, full state) |
| V5 full 3-seed | 0.9500 | 0.7467 | 0.5121 | 81.13 | no (official h, warm start) |
| V5 h-input 3-seed | 0.9533 | 0.7633 | 0.2779 | 47.35 | no (official h) |
| V5 no-h 3-seed | 0.4179 | 0.0133 | 0.0475 | 9.98 | no (archived) |
| pure MAPPO 16k 3-seed (no penalty) | 0.3079 | 0.0033 | 0.0571 | 32.17 | no (archived) |
| MAPPO + fixed penalty 10, 16k 3-seed | 0.5192 | 0.0333 | 0.0142 | -30.35 | **yes (matched 16k)** |
| MAPPO + fixed penalty 1, 16k 3-seed | 0.4096 | 0.0100 | 0.1325 | 59.36 | **yes (matched 16k)** |
| MAPPO + fixed penalty 100, 16k 3-seed | 0.6737 | 0.0733 | 0.0158 | -59.48 | **yes (matched 16k)** |
| MAPPO-Lag 16k 3-seed | 0.3217 | 0.0000 | 0.0863 | 39.55 | **yes (matched 16k)** |
| MAPPO-Lag 100k 3-seed | 0.4325 | 0.0033 | 0.1363 | 50.36 | no (generous reference) |
| **V6 repaired 3-seed** | **0.9417** | **0.6933** | **0.4100** | **66.85** | **yes (V6 protocol)** |

Reading:

- Safety is close to CBF-QP (`-3.7 pp`) and V5 full (`-0.8 pp`).
- Task is below V5 full (`-10.2 pp` success), CBF-QP (`-37.1 pp`), and GCBF+
  (`-45.1 pp`).
- V6 beats V5 h-input on success (`+13.2 pp`) and return (`+41%`) with slightly
  lower safety.
- V6 dominates V5 no-h, pure MAPPO, and both matched 16k collision-aware
  baselines (fixed penalty and MAPPO-Lag) on all reported axes.
- The matched 16k collision-aware baselines were added on 2026-10-27.  The
  fixed-penalty arm collapses task (`reach=0.043`, return `-30.35`), and the
  16k MAPPO-Lag arm has no collision-free episode.  The 2026-10-28
  fixed-penalty sweep fills the nearby penalties: `1` remains unsafe
  (`safe 0.01`, official `0.410`, success `0.133`) and `100` shifts toward
  safety (`0.073`, `0.674`) but collapses task (`success 0.016`, return
  `-59.5`).  Neither rescues the MAPPO baseline on the matched budget; V6
  remains the only matched row above `0.69` safe with task above `0.36`.
  The 100k MAPPO-Lag row is a generous reference, not compute-matched.

Full table and caveats:
[`independent/nav/BASELINE_COMPARISON.md`](independent/nav/BASELINE_COMPARISON.md).

## h2/actor2 diagnostic (complete, read-only)

Protocol: fixed final actor1 checkpoints, frozen repaired h, 3 seeds x 100
deterministic episodes, eval seed 10000.  The audit reproduces the formal
`safe_episode`, `official_safety`, `official_success`, `official_reach`, and
`episode_return` exactly.

- **155/155 new collision onsets** occur with `h_cur < 0` and
  `gate_next = 0`; none occur with `h_cur >= gate_high` or the gate open.
- **Impact-state label: actor/action-limited.**  h1 is not blind at the
  collision state; the actor collides after h1 has already warned and the
  diagnostic gate has already closed.
- **But the h1 training data are off-policy for final actor1.**  h1 was
  trained on h-only seed0 and Stage-A gate-progress seed0 rollouts plus random
  coverage, not on final repaired Stage-B actor1.  The risk-region evidence
  shows the resulting temporal gap: at 17-32 steps before collision, h mean
  `+0.242` versus target `-0.222`; future-collision-region MAE `0.394`.
  Pooled MAE looks good (`0.0976` versus offline test `0.1510`) because
  far-safe rows dominate it; it is not sufficient for the 32-step warning.
- **Overall diagnosis: mixed** -- temporal h limitation before impact, plus
  actor/action limitation at impact.  The iterative h2-on-actor1-data
  hypothesis is therefore justified.

Stage C0 was run as **C0a (seed 0 pilot)**: 100 training episodes + 101
held-out evaluation episodes, 40 epochs, same target/arch/loss/calibration,
from scratch.  It **failed** the pre-registered pilot rule: 17-32-step MAE
rose from 0.709 to 0.826 (+16.5% worse); 9-16-step MAE rose from 0.442 to
0.567; future-collision MAE rose from 0.375 to 0.478.  Global MAE and
safe-row correlation improved, so the failure is specifically temporal/
risk-region calibration, not global fit.  The nominal +11-step warning lead
is a gate-threshold artifact; threshold-free last `h > 0` is unchanged at
8 steps.  C0b (seeds 1/2) was not run and the original C1 was not authorized
at that stage.

A post-hoc, read-only seed-0 h-swap of h2 into actor1 was then run: safe
episode `0.90` vs `0.84`, official safety `0.985` vs `0.974`, success
`0.439` vs `0.428`, return `61.40` vs `60.91`.  The differences are small,
one-seed, and not statistically separated (safe-episode `z≈1.27`); the swap
is off-policy because actor1 was trained with h1, and h2 raises the h input.
This diagnostic does **not** reproduce the offline risk-region degradation,
but it does not change the pre-registered C0a failure or authorize the
original C1.  Following explicit approval, a new **C1a seed-0 actor2 pilot**
(h2 seed 0, from scratch, same cost/steps/eval) was run.  It **failed all
five pre-registered checks**: `safe_episode 0.510` vs `0.840`,
`official_safety 0.89375` vs `0.97375`, `official_success 0.33125` vs
`0.42750`, `official_reach 0.36250` vs `0.43500`, `episode_return 52.22` vs
`60.91`; collisions rose from `0.16` to `0.49`.  The training gate was almost
always open (`task_gate_mean ≈ 0.99`, closed `≈ 0.2%`), so the failure is not
gate suppression.  It is consistent with h2's tail miscalibration plus
one-seed from-scratch optimization, but the causal mechanism is not proven.
The from-scratch h2/actor2 branch is stopped and C1b is not authorized.  A
separate **C1w 4k warm-start of actor1 with h2** (no h1 control) was run as a
diagnostic adaptation arm and **failed the C1a guardrails**: `safe_episode
0.840`, `official_safety 0.97625`, `official_success 0.20250`, `official_reach
0.20625`, `episode_return 28.85`.  It matches actor1 on safe episodes and
collisions, does not beat the no-training swap on safety, and collapses the
task; `promising_adaptation = false`.  The pre-registered staged rule was
applied: the h1-continued control was **not run** and the h2 warm-start branch
is stopped.  It is not a C1a pass and cannot revive C1b.  A new evaluation-only
**C1s** replication (h2 seed 0 on fixed actor1 seeds 1/2) is complete and
classified **`consistent_positive`**: both seeds are non-negative on safe and
official safety; mean `Δsafe +0.030` (exactly at the pre-registered `+0.03`
threshold), `Δofficial_safety +0.00875`, `Δsuccess +0.01375`, `Δreach
+0.00813`, `Δreturn -0.05`; no seed violates the catastrophic-task guard.
Seed 1 carries most of the gain; seed 2 is marginal.  It is off-policy, uses
one h2 checkpoint, and is not a from-scratch result or a C0/C1 pass.  Record:
[`independent/nav/STAGE_C0_H2_TEMPORAL.md`](independent/nav/STAGE_C0_H2_TEMPORAL.md),
[`independent/nav/STAGE_C1A_H2_ACTOR2_PILOT.md`](independent/nav/STAGE_C1A_H2_ACTOR2_PILOT.md),
[`independent/nav/STAGE_C1W_H2_ACTOR1_4K.md`](independent/nav/STAGE_C1W_H2_ACTOR1_4K.md),
and
[`independent/nav/STAGE_C1S_H2_ACTOR_SEED_SWAP.md`](independent/nav/STAGE_C1S_H2_ACTOR_SEED_SWAP.md).
Diagnostic:
[`independent/nav/H2_ACTOR2_DIAGNOSTIC.md`](independent/nav/H2_ACTOR2_DIAGNOSTIC.md).

## Promoted vs archived

Promoted navigation artifacts:

- `independent/h/v6_h.pkl` — original V6 navigation h (h-only baseline).
- `independent/h_stageb_repair/v6_h_stageb_repair.pkl` — repaired h for the
  promoted gate.
- `independent/nav_stageb_repair_16k/seed{0,1,2}/` — promoted 3-seed actor and
  formal evaluations.
- `independent/nav_stageb_repair_16k/aggregate_100ep.{json,md}` — aggregate.
- `independent/nav_stageb_repair_16k/manifest.json` — config/h/runner hashes and
  protocol manifest.
- `independent/formal_3seed/h_only/` — matched h-only baseline.

Promoted Ant artifacts:

- `independent/ant/h_predictive/` — predictive h.
- `independent/ant/mappo_h_pred_gate_10m/seed{0,1,2}/` — promoted gate.
- `independent/ant/mappo_cost_ramp10_10m/seed{0,1,2}/` — matched cost-only.
- `independent/ant/RETENTION_10M.md`, `MULTISEED_10M_PLAN.md`.

Archived / not promoted (kept for the record; do not use as main evidence):

- `independent/nav_h_gate_16k/` — broad positive-reward gate
  (safety/task tradeoff).
- `independent/nav_gate_progress_16k/` — Stage A (failed safety).
- `independent/nav_stageb_16k/` — original Stage B (catastrophic).
- `independent/nav_stageb_gateoff_16k/` — gate-off diagnostic for original
  Stage B.
- `independent/single_seed_16k/`, `independent/runs/` — early single-seed
  screens/pilots (`outputs/v6/protocol/` is a top-level legacy run).
- `independent/safety_opt/` — barrier/CBF-loss screens; one 24.5k fine-tune
  is a reference only.
- `independent/_archive_smoke_20261026/` — archived smoke/debug runs
  (formerly `_smoke_*`, `h_stageb_repair_smoke*`, `*_nanfix_test`,
  `*_bench_grad`); not evidence.

Legacy / non-independent (do **not** use in the V6 main path):

- `outputs/v6/{co_train*,frozen_v6*,h_data*,h_mech,h_swap,official_h*,
  safety_h,gnn,ablation,protocol,smoke_co_train}` — early coupled experiments,
  many configured with `arch: official_cbf` / official h checkpoints.
- `data_v5h0`, `h_mech` checkpoints, and any V5/V4 artifact.

## What is still open / recently closed

- **Closed (2026-10-27):** formal 16k h-off 3-seed navigation baseline (P1 D:
  `safe 0.0333`, `official_safety 0.5479`, `success 0.0367`, `return 7.21`).
- Optional nav 100ep stochastic evaluation of the promoted checkpoints.
- Nav task channel / `V_T`: **strict safety-first rule not met; task-heavy
  trade-off documented.**  T0w actor1 + frozen finite-horizon V_T 4k
  warm-start passed its own screen (100ep: safe `0.890`, official safety
  `0.98250`, success `0.67875`, reach `0.685`, return `161.88`; actor1
  success `0.42750`, return `60.91`), and V_T passed its quality gate
  (val EV `0.2270`, test Pearson `0.4991`).  The matched no-V_T 4k control
  reached safe `0.930`, official safety `0.99000`, success `0.380`, return
  `69.67`.  `task_gain = true` (success `+0.29875`, return `+92.20`) but
  `safety_not_worse = false` (safe `0.890 < 0.900`; one episode) ->
  `vt_channel_supported = false` **under the pre-registered safety-first
  rule**.  Task-first reading: a task-heavy Pareto point, with a large task
  gain at a small, statistically unresolved scene-safety discount
  (official safety `-0.75 pp`; Wilson intervals overlap).  This is not a
  safety-non-inferiority pass.  The strict safety-first line is stopped;
  a 3-seed from-scratch `h1 + gate + V_T` replication is the optional next
  step if the task-first objective is chosen, and is not authorized yet.
  One seed; not a main claim.  Record:
  `independent/nav/STAGE_T0W_VT_WARMSTART.md`.
- h2/actor2 Stage C: C0a seed-0 h2-only pilot **failed**; h-only retraining
  on actor1 data did not improve the 17-32-step risk region.  C0b was not run
  and the original C1 is not authorized.  A post-hoc seed-0 h2+actor1 h-swap
  was slightly better but one-seed and not significant; it is a diagnostic,
  not a promotion criterion.  The separately approved **C1a seed-0 actor2
  pilot failed all five checks** (`safe 0.510`, `official_safety 0.89375`,
  `success 0.33125`, `reach 0.36250`, `return 52.22`); the from-scratch
  h2/actor2 branch is stopped and C1b is not authorized.  A **C1w 4k
  warm-start of actor1 with h2** (no h1 control) was run as a diagnostic
  adaptation arm and **failed** (safe `0.840`, official safety `0.97625`,
  success `0.2025`, reach `0.20625`, return `28.85`); it did not beat the
  no-training swap and collapsed the task.  The h1-continued control was not
  run; the h2 warm-start branch is stopped.  It is not a C1a pass.  A new
  **C1s** evaluation-only replication (h2 seed 0 on fixed actor1 seeds 1/2,
  100ep each) is complete and **`consistent_positive`** (mean `Δsafe +0.03`
  exactly at threshold, `Δofficial_safety +0.00875`, no catastrophic task
  regression).  This is a thin robustness signal for h2 seed 0 as a frozen
  input: seed 1 carries most of the gain, seed 2 is marginal, the swap is
  off-policy, and h2 seed 0 is distribution-shifted for seeds 1/2.  It is not
  a main claim and does not revive C1b.  A matched h2-seed study (C0b) is a
  discussion item requiring a new pre-registration and explicit approval; it
  is not launched.  **Current decision (2026-10-26): user chose to stop after
  C1s; no C0b and no new controlled arm are launched.**  If work resumes, the
  next controlled arm must be safety coupling or action conditioning (one axis
  only).  Do not combine with a gate/cost/task change.
- Optional Ant 10M multi-checkpoint (2M/5M) evaluation; retention is already
  established at 10M.
- **Ant 10M from-scratch MAPPO-Lag external baseline: complete, single-seed
  screen (2026-10-27 19:17).**  Fresh experiment
  `v6_lag_10m_seed0_restart_20261027`, seed 0, exactly 10M steps, unchanged
  upstream recipe, no checkpoint selection.  Evaluated once at 100
  deterministic + 100 stochastic episodes (eval seeds `20000..20099`).
  Deterministic: `safe 0.97`, `forward 3011.02`; stochastic: `safe 0.66`,
  `forward 2305.82`.  This is an external single-seed screen, not a V6 main
  claim; no 3-seed Ant expansion is launched.  Record:
  `independent/ant/STAGE_ANT_MAPPO_LAG_10M.md`.
- **Ant V_T transfer diagnostic: closed null (single seed).**  1-D `A_T`
  warm-start from the existing seed-0 10M h-only base, 1M extra steps, with
  a matched no-V_T 1M control.  h/cost/reward were frozen; the Ant V_T was
  trained offline on task return only and passed its gate.  The warm-start
  did not beat control on deterministic forward (`3960.83` vs `4014.03`) or
  return (`4574.53` vs `4630.37`); safety was equal (`0.96`).
  `vt_transfer_supported = false`; the line is stopped and no 5-D/longer/new-
  seed follow-up is launched.  Diagnostic only; the promoted Ant 3-seed
  claim is unchanged.  Record:
  `independent/ant/STAGE_ANT_VT_TRANSFER.md`.
- No `task_gate_high` sweep.
- Any action-conditioned filter/candidate screening requires explicit
  approval.

## Document map

- Proposal: `docs/PROJECT_PROPOSAL_V6.md` (historical design snapshot).
- Detailed design: `docs/DETAILED_DESIGN_V6.md` (historical design snapshot).
- Output/artifact index: `README.md` in this directory.
- Consolidated record: `V6_COMPLETE_RECORD.md`.
- TODO: `TODO_V6.md`.
- Navigation runbook: `nav/README_V6.md`.
- Navigation design/result docs: `independent/nav/README.md`.
- Navigation baselines: `independent/nav/BASELINE_COMPARISON.md`.
- h2/actor2 diagnostic: `independent/nav/H2_ACTOR2_DIAGNOSTIC.md`.
- Stage C0 pre-registration: `independent/nav/STAGE_C0_H2_TEMPORAL.md`.
- C1a/C1w/C1s follow-ups: `independent/nav/STAGE_C1A_H2_ACTOR2_PILOT.md`,
  `independent/nav/STAGE_C1W_H2_ACTOR1_4K.md`,
  `independent/nav/STAGE_C1S_H2_ACTOR_SEED_SWAP.md`.
- Ant records index: `independent/ant/README.md`.
- Ant summary: `independent/ant/FINAL_ANT_REPORT.md`.
- Ant 10M retention: `independent/ant/RETENTION_10M.md`.

## Reproduction shortcut

No retraining is needed to reproduce the tables.

```bash
# static independence check
python nav/scripts/check_v6_independence.py

# re-aggregate the existing nav formal evaluations
python nav/scripts/summarize_v6_formal.py \
  --dir outputs/v6/independent/nav_stageb_repair_16k \
  --out outputs/v6/independent/nav_stageb_repair_16k/aggregate_100ep.json

# re-evaluate one promoted checkpoint (example: seed 0)
python nav/scripts/evaluate_v6.py \
  --checkpoint outputs/v6/independent/nav_stageb_repair_16k/seed0/policy_final.pth \
  --h-checkpoint outputs/v6/independent/h_stageb_repair/v6_h_stageb_repair.pkl \
  --num-episodes 100 --seed 10000 --device cuda \
  --out outputs/v6/independent/nav_stageb_repair_16k/seed0/eval_100ep_rerun.json
```

Repaired Stage-B hashes:

- config `nav/configs/v6/v6_nav_stageb_repair_16k.yaml`:
  `4ee69c9e99a427708d17f1157991786d41a60e2624721e7e478b90a0aabebd16`
- h `outputs/v6/independent/h_stageb_repair/v6_h_stageb_repair.pkl`:
  `0f67eaeec99912881017231234aa10846c5edfda767118da2157b2da7df7fb13`

## Claim boundary

Safe claims:

- Ant predictive-h gate: strong 3-seed 10M empirical safety/task retention.
- Navigation repaired Stage-B gate: passes the pre-registered 3-seed
  guardrails; safety improves and task does not collapse on average.
- Navigation attribution (descriptive 2x2, `n=3`): actor h input is required
  for the 16k from-scratch V6 pipeline; the training-time gate is ineffective
  without it; repaired h + gate is the only cell combining safety with
  retained task.

Do not claim:

- formal/CBF-certified safety for either task;
- navigation task-channel safety non-inferiority or from-scratch success;
  T0w is a single-seed task-heavy trade-off, not a safety-non-inferiority pass;
- policy-level statistical superiority from `n=3`;
- that the gate alone improves safety or task without actor h input, or that
  repaired h alone is a strict improvement over the original h (it trades
  task for safety in the gate-off cell);
- V5/official-h equivalence or CBF-QP/GCBF+ task parity.
