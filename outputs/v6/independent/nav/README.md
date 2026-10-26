# V6 navigation records

Canonical project status: [`../../STATUS.md`](../../STATUS.md).

## Current promoted result

Repaired Stage-B predictive-h gate, 3 seeds x 100 deterministic episodes,
16,384 steps/seed, fixed final checkpoint, eval seed 10000:

| seed | safe_ep | collision_ep | official_safety | success | reach | return |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.840 | 0.160 | 0.9738 | 0.4275 | 0.4350 | 60.91 |
| 1 | 0.510 | 0.490 | 0.9000 | 0.4575 | 0.4963 | 70.92 |
| 2 | 0.730 | 0.270 | 0.9513 | 0.3450 | 0.3575 | 68.74 |
| **mean** | **0.693** | **0.307** | **0.9417** | **0.4100** | **0.4296** | **66.85** |
| **std** | 0.168 | 0.168 | 0.0378 | 0.0583 | 0.0695 | 5.26 |

Matched h-only 3-seed: `safe 0.450`, `official_safety 0.8783`,
`success 0.4067`, `return 61.73`.  All pre-registered aggregate guardrails
pass.  `n=3`; seed-level differences are not statistically separated.

Raw artifacts: `../nav_stageb_repair_16k/` (see `manifest.json` for config/h/
runner hashes); baseline h-only: `../formal_3seed/h_only/`.

## Document index

| document | role |
|---|---|
| [`STAGE_B_REPAIR_MULTISEED.md`](STAGE_B_REPAIR_MULTISEED.md) | pre-registered protocol, per-seed results, guardrails, caveats |
| [`BASELINE_COMPARISON.md`](BASELINE_COMPARISON.md) | matched internal arms, h-only reference, CBF/Safe-MARL references |
| [`H2_ACTOR2_DIAGNOSTIC.md`](H2_ACTOR2_DIAGNOSTIC.md) | read-only h2/actor2 diagnostic: collision-state h/gate, warning lead, actor1-state calibration, decision |
| [`STAGE_C0_H2_TEMPORAL.md`](STAGE_C0_H2_TEMPORAL.md) | h2-only temporal-calibration screen: pre-registration, C0a result (failed), pass rule |
| [`STAGE_C1A_H2_ACTOR2_PILOT.md`](STAGE_C1A_H2_ACTOR2_PILOT.md) | new seed-0 actor2 pilot with h2: protocol, baselines, pilot rule |
| [`STAGE_C1W_H2_ACTOR1_4K.md`](STAGE_C1W_H2_ACTOR1_4K.md) | diagnostic 4k warm-start of actor1 with h2; no h1 control; pre-registered |
| [`STAGE_C1S_H2_ACTOR_SEED_SWAP.md`](STAGE_C1S_H2_ACTOR_SEED_SWAP.md) | evaluation-only h2 seed-0 swap on actor1 seeds 1/2; pre-registered consistency rule |
| [`STAGE_T0W_VT_WARMSTART.md`](STAGE_T0W_VT_WARMSTART.md) | task-channel diagnostic: actor1 + frozen finite-horizon V_T 4k warm-start; strict safety-first rule missed by 1 episode, task-heavy trade-off |
| [`STAGE_B_H_DATA_TARGET.md`](STAGE_B_H_DATA_TARGET.md) | h data/target failure, repair, offline metrics, 3-seed actor result |
| [`PREDICTIVE_GATE.md`](PREDICTIVE_GATE.md) | gate history: broad gate, Stage A, Stage B, repaired Stage B |
| [`STAGE_A_GATE_PROGRESS.md`](STAGE_A_GATE_PROGRESS.md) | Stage A progress-only gate; archived, not promoted |
| [`LEARNABLE_DESIGN_POINTS.md`](LEARNABLE_DESIGN_POINTS.md) | design notes from the Ant comparison; not a result record |

## Status labels

- **Promoted:** repaired Stage-B 3-seed gate.
- **Matched baseline:** h-only 3-seed.
- **Read-only diagnostic (complete):** h2/actor2 audit; impact-state label
  actor/action-limited, overall diagnosis mixed because h1's training data are
  off-policy for final actor1 and the 17-32-step risk region is poorly
  calibrated.  No h2/actor2 training was launched at that stage.
- **Stage C0a (complete, failed):** h2-only temporal-calibration screen on
  seed 0.  Global MAE improved (`0.084` vs `0.092`) but the 17-32-step
  risk-region MAE worsened (`0.826` vs `0.709`, +16.5%), as did 9-16-step and
  future-collision MAE.  C0b was not run; no actor2 was trained.
- **Post-hoc seed-0 h2+actor1 h-swap (diagnostic):** safe episode `0.90` vs
  `0.84`, official safety `0.985` vs `0.974`, success `0.439` vs `0.428`,
  return `61.40` vs `60.91`; one seed, not significant, off-policy because
  actor1 was trained with h1.  It does not change C0a or authorize C1.
  Record: [`../nav_c0_h2_actor1_swap/README.md`](../nav_c0_h2_actor1_swap/README.md).
- **C1a seed-0 actor2 pilot (complete, failed):** h2 seed 0, actor2 from
  scratch, 16,384 steps, 100ep eval seed 10000.  All five pre-registered
  checks failed: safe `0.510` vs `0.840`, official safety `0.89375` vs
  `0.97375`, success `0.33125` vs `0.42750`, reach `0.36250` vs `0.43500`,
  return `52.22` vs `60.91`; collisions `0.16 -> 0.49`.  The gate was almost
  always open, so this is not gate suppression.  From-scratch h2/actor2
  branch stopped; C1b not authorized.  Record:
  [`STAGE_C1A_H2_ACTOR2_PILOT.md`](STAGE_C1A_H2_ACTOR2_PILOT.md).
- **C1w actor1+h2 4k warm-start (complete, failed):** diagnostic adaptation
  only, no h1-continued control.  100ep safe `0.840`, official safety
  `0.97625`, success `0.20250`, reach `0.20625`, return `28.85`;
  `promising_adaptation = false`.  The h1-continued control was not run and
  the h2 warm-start branch is stopped.  Not a C1a pass.  Record:
  [`STAGE_C1W_H2_ACTOR1_4K.md`](STAGE_C1W_H2_ACTOR1_4K.md).
- **C1s h2 seed-0 + actor1 seed replication (complete,
  `consistent_positive`, evaluation only):** seed 1 safe `0.56` vs `0.51`,
  official safety `0.91250` vs `0.90000`; seed 2 safe `0.74` vs `0.73`,
  official safety `0.95625` vs `0.95125`.  Mean `Δsafe +0.030` (exactly the
  `+0.03` boundary), `Δofficial_safety +0.00875`; all eight checks hold.
  Thin and seed-1-driven.  This is an off-policy swap using one h2
  checkpoint; h2 seed 0 is distribution-shifted for seeds 1/2.  It is not h2
  seed reproducibility, not a from-scratch result, and not a main claim.
  Record:
  [`STAGE_C1S_H2_ACTOR_SEED_SWAP.md`](STAGE_C1S_H2_ACTOR_SEED_SWAP.md).
- **T0w task-channel warm-start (strict safety-first rule missed; task-heavy
  trade-off):** actor1 + frozen finite-horizon V_T, 4k continuation,
  h1/gate/cost unchanged.  V_T passed its quality gate (val EV `0.2270`,
  test Pearson `0.4991`).  100ep warm-start: safe `0.890`, official safety
  `0.98250`, success `0.67875`, reach `0.685`, return `161.88`; matched
  no-V_T 4k control: safe `0.930`, official safety `0.99000`, success
  `0.380`, return `69.67`.  `task_gain = true` (success `+0.29875`,
  return `+92.20`) but `safety_not_worse = false` (safe `0.890 < 0.900`,
  one episode) -> `vt_channel_supported = false` under the pre-registered
  safety-first rule.  Task-first reading: a task-heavy Pareto point with a
  large task gain and a small, statistically unresolved safety discount;
  not a safety-non-inferiority pass.  No from-scratch arm is authorized yet;
  a 3-seed `h1 + gate + V_T` replication is the optional next step.  One
  seed, diagnostic only.  Record:
  [`STAGE_T0W_VT_WARMSTART.md`](STAGE_T0W_VT_WARMSTART.md).
- **Archived/not promoted:** broad gate, Stage A, original Stage B,
  gate-off diagnostic, barrier/CBF-loss screens.
- **Still open:** formal 16k h-off 3-seed baseline; optional stochastic
  100ep evaluation; optional 3-seed task-first `h1 + gate + V_T` replication;
  next controlled arm (safety coupling or action conditioning) pending
  approval.
