# V6 complete experiment record

Last updated: 2026-10-26.  This is the consolidated V6 record.  It summarizes
the multi-agent navigation branch and the Ant branch; raw artifacts are linked
by path.  It is deliberately conservative: pilot, single-seed, and formal
three-seed results are labeled separately.

## 0. Current status at a glance

Canonical status snapshot and promoted/archived artifact map:
[`STATUS.md`](STATUS.md).

- **Ant:** predictive-h + safety-gated forward passes 3 seeds x 10M.
  Deterministic forward retention `91.6%`, safe `0.97` vs cost-only `0.063`;
  stochastic retention `93.9%`, safe `0.84` vs `0.020`.
- **Navigation:** repaired Stage-B predictive-h gate passes the pre-registered
  3-seed guardrails: `safe_episode=0.693 +/- 0.168`,
  `official_safety=0.9417 +/- 0.0378`, `success=0.4100 +/- 0.0583`,
  `return=66.85 +/- 5.26`, versus h-only `0.450/0.8783/0.4067/61.73`.
- **Positioning:** navigation safety is near CBF-QP/V5 full, but task remains
  below V5 full/CBF-QP/GCBF+; this is not a formal safety certificate.
- **Closed null diagnostic:** Ant V_T 1-D transfer (single seed); the
  warm-start did not beat the matched no-V_T control, so the line is stopped
  and the promoted claims are unchanged.  See section 4.5.
- **Open:** formal 16k h-off 3-seed navigation baseline; navigation task
  channel; optional stochastic evaluation; Ant multi-checkpoint evaluation.

## 1. Scope and independence

V6 uses only:

- the official simulator/environment, task reward and safety/collision cost;
- V6-collected data and V6-trained h/actor checkpoints;
- generic PPO/MAPPO/network/training code.

V6 does **not** use:

- official h / `cbf.pkl`, QP/`u_ref`, expert actions/labels;
- V5/V4 weights, data, or checkpoints;
- `data_v5h0`, `h_mech` checkpoints, or V5 warm-start.

The static independence guard passed 57 files on 2026-10-24.

## 2. Artifact map

| area | record | raw artifacts |
|---|---|---|
| navigation main | `nav/README_V6.md` | `outputs/v6/independent/` |
| navigation formal 3-seed (h-only) | `outputs/v6/independent/formal_3seed/h_only/aggregate_100ep.md` | `.../h_only/seed{0,1,2}/` |
| navigation repaired Stage B 3-seed | `outputs/v6/independent/nav/STAGE_B_REPAIR_MULTISEED.md` | `.../nav_stageb_repair_16k/seed{0,1,2}/` |
| navigation baseline comparison | `outputs/v6/independent/nav/BASELINE_COMPARISON.md` | archived V5/CBF references |
| navigation h | `outputs/v6/independent/h/summary.json` | `.../h/v6_h.pkl` |
| navigation data | `outputs/v6/independent/data/summary.json` | `.../data/*.npz` |
| Ant summary | `outputs/v6/independent/ant/FINAL_ANT_REPORT.md` | `.../ant/` |
| Ant predictive h | `outputs/v6/independent/ant/PREDICTIVE_H_GATE_2M.md` | `.../ant/h_predictive/` |
| Ant 10M retention | `outputs/v6/independent/ant/RETENTION_10M.md` | `.../ant/mappo_h_pred_gate_10m/seed0/` |
| V5 Ant reference | `Safe-MARL-proposal_v4/outputs/v5/ant/baselines/comparison_100ep.md` | V5 outputs (reference only) |

## 3. Multi-agent navigation branch

### 3.1 Protocol

- 8 agents, double-integrator dynamics;
- area 4.0 x 4.0, 8 obstacles, 32 LiDAR rays, up to 5 neighbors;
- episode limit 256 steps;
- V6 h is a full-graph GNN safety function, not the Ant 9-D MLP;
- actor has task/correction/feasibility heads in the generic policy code;
- main h-only config: `nav/configs/v6/v6_nav_independent.yaml`.

### 3.2 V6 navigation h data and offline h

From `outputs/v6/independent/data/summary.json`:

- 100 random episodes, 25,600 rows;
- horizon 32; unsafe rate 5.68%; horizon-safe 86.30%; ignored 8.02%;
- 25,500 hdot rows;
- merged train: 28,480 rows (20,480 base + 8,000 augmentation), 227,840 agent rows;
- h checkpoint: `outputs/v6/independent/h/v6_h.pkl`.

From `outputs/v6/independent/h/summary.json`:

| metric | validation | test |
|---|---:|---:|
| balanced accuracy | 0.9867 | 0.9853 |
| unsafe recall | 0.9967 | 0.9988 |
| safe recall | 0.9766 | 0.9719 |
| false alarm | 0.0223 | 0.0256 |
| AUROC | 0.9986 | 0.9986 |
| threshold | 0.8621 | |
| h offset | -0.8621 | |

So the navigation h itself is very strong offline.  The limitation is the
closed-loop use of h, not the offline classifier.

### 3.3 Pilot: 8192 steps, 20 episodes, seed 0

| arm | safe_episode | collision_episode | official_safety | reach | success | episode_return |
|---|---:|---:|---:|---:|---:|---:|
| h-on | 0.60 | 0.40 | 0.9375 | 0.1313 | 0.1188 | 29.21 |
| h-off | 0.05 | 0.95 | 0.5188 | 0.0500 | 0.0188 | -31.73 |

This is a single-seed pilot only.  It showed that h-on was much better than
h-off, but task success was still low.

### 3.4 Formal h-only: 16384 steps, 3 seeds, 100 episodes

Source: `outputs/v6/independent/formal_3seed/h_only/aggregate_100ep.md`.

| seed | reach | success | safe_episode | collision_episode | official_safety | episode_return |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.5763 | 0.5325 | 0.620 | 0.380 | 0.9287 | 69.78 |
| 1 | 0.4487 | 0.4175 | 0.490 | 0.510 | 0.9025 | 65.74 |
| 2 | 0.3100 | 0.2700 | 0.240 | 0.760 | 0.8037 | 49.67 |
| **mean** | **0.4450** | **0.4067** | **0.4500** | **0.5500** | **0.8783** | **61.73** |

Standard deviations across seeds: reach `0.1332`, success `0.1316`, safe
`0.1931`, official safety `0.0659`, return `10.64`.

This is a **mixed** result: h-on is better than the no-h pilot, but seed 2 is
weak and the mean success is only 0.407.  A formal 16k h-off three-seed
baseline is not present yet; the h-off evidence is the 8k single-seed pilot.

### 3.5 Navigation safety screen

Single-seed 20/100 episode screens, from
`outputs/v6/independent/safety_opt/screen_summary.json`:

| variant | episodes | reach | success | safe_ep | collision_ep | official_safety | return |
|---|---:|---:|---:|---:|---:|---:|---:|
| h-only baseline | 20 | 0.5688 | 0.5125 | 0.50 | 0.50 | 0.8813 | 68.68 |
| cost | 20 | 0.1500 | 0.1375 | 0.30 | 0.70 | 0.8250 | 19.47 |
| CBF loss | 20 | 0.4813 | 0.4688 | 0.65 | 0.35 | 0.9438 | 63.75 |
| barrier | 20 | 0.3375 | 0.3375 | 1.00 | 0.00 | 1.0000 | 68.70 |
| barrier fine-tune | 100 | 0.3263 | 0.3188 | 0.73 | 0.27 | 0.9500 | 65.77 |
| barrier + both | 20 | 0.1063 | 0.1063 | 0.85 | 0.15 | 0.9688 | 23.12 |

Interpretation: the dense barrier improves safety but loses task, and the
combined arm loses too much task.  These screens are single-seed references;
the promoted 3-seed navigation gate is the repaired Stage-B result in section
3.9, not a barrier/cost variant.

### 3.6 Predictive-h broad task gate (archived: safety/task tradeoff)

Implementation: `nav/v6/training.py` now has a positive-task gate:
`gate_i = clip((h_next_i - low)/(high - low), 0, 1)`, applied only to positive
progress/reach reward.  The official collision cost is never gated.  Config:
`nav/configs/v6/v6_nav_h_gate.yaml`; design: 
`outputs/v6/independent/nav/PREDICTIVE_GATE.md`.

Calibration uses the V6 navigation h distribution, not the Ant threshold:
`low=0.0`, `high=0.13`, healthy h plateau ~0.138.  Evaluation disables the
gate.  Three seeds x 16384 steps were trained deterministically, but only
seed 0 was evaluated for 100 episodes; the remaining evaluations were stopped
after the tradeoff below.

| seed-0, 100ep | h-only | h + gate |
|---|---:|---:|
| safe_episode | 0.620 | 0.700 |
| collision_episode | 0.380 | 0.300 |
| official_safety | 0.9287 | 0.9400 |
| success | 0.5325 | 0.3725 |
| episode_return | 69.78 | 64.96 |

The gate improves episode-level safety but reduces success by 16 pp.  It is
recorded as an exploratory **safety/task tradeoff**, not a promoted nav result.
No three-seed aggregate is claimed.

### 3.7 Navigation Stage A: progress-only gate (archived: failed safety)

The first gate scaled all positive task reward.  Stage A tests the Ant-style
decomposition: gate only dense positive progress, leave reach/terminal
bonuses un-gated, freeze h, and train the actor from scratch.  Config:
`nav/configs/v6/v6_nav_gate_progress_16k.yaml`; design:
`outputs/v6/independent/nav/STAGE_A_GATE_PROGRESS.md`.

The 8k monitor was very poor (`safe_episode=0.100`, `collision=0.900`), but the
fixed final 16k checkpoint recovered in the 20ep monitor
(`safe=0.700`, `success=0.5813`, `return=80.19`).  The formal 100ep result did
not confirm a safety gain:

| seed-0, 100ep | h-only | Stage A |
|---|---:|---:|
| safe_episode | 0.620 | 0.530 |
| collision_episode | 0.380 | 0.470 |
| official_safety | 0.9287 | 0.9113 |
| official_success | 0.5325 | 0.5075 |
| official_reach | 0.5763 | 0.5475 |
| episode_return | 69.78 | 76.85 |

Stage A recovers task/return relative to the broad gate but loses 9 pp
episode-level safety relative to h-only.  It is **not promoted**, and seeds
1/2 are not run.  The 20ep monitor was optimistic.  The broad gate's safety
gain cannot be attributed to progress gating alone; the on-policy h gate is
weakly active (~5% closed at the end), so the next controlled change is h
quality/temporal target (Stage B), not another gate-coefficient sweep.

### 3.8 Navigation Stage B: on-policy h + future-min target (original: failed)

Stage B changes h data and h target only.  V6-owned h data was collected from
the h-only and Stage-A seed-0 policies with stochastic actions (100 episodes,
50/50, 25,600 rows).  The target is
`tanh(min geometric margin over the next 32 states / 0.2)`, with ignored rows
included in the target regression.  h and actor were both retrained from
scratch.

Offline h (validation-selected epoch 1):

| split | balanced acc | unsafe recall | false alarm | AUROC |
|---|---:|---:|---:|---:|
| val | 0.8714 | 0.8667 | 0.1229 | 0.9406 |
| test | 0.9029 | 0.9070 | 0.1005 | 0.9673 |

The pre-registered gate rule gave `low=0.0`, `high=max(0.02, q10_safe_shifted)`
`=0.02` (the safe lower tail was negative).  On validation this gate closed
12.8% of safe, 58.7% of ignored, and 87.3% of unsafe states.

Formal 100ep deterministic result, fixed final checkpoint:

| seed-0, 100ep | h-only | Stage A | Stage B |
|---|---:|---:|---:|
| safe_episode | 0.620 | 0.530 | **0.000** |
| collision_episode | 0.380 | 0.470 | **1.000** |
| official_safety | 0.9287 | 0.9113 | **0.4400** |
| official_success | 0.5325 | 0.5075 | **0.0375** |
| official_reach | 0.5763 | 0.5475 | **0.0700** |
| episode_return | 69.78 | 76.85 | **-23.60** |

This is a catastrophic actor failure, not a safety/task tradeoff.  The 8k and
16k monitors were already `safe_episode=0.0`.  Diagnosis: at 2,048 actor
steps the Stage-B h produced `h_neg_rate=40.4%` and the gate was 40.5% closed,
with `task_reward_mean=-0.135`; at the same step Stage A had 6.3% closed and
`task_reward_mean=+0.105`.  The new h has a broad negative lower tail on the
early from-scratch actor's states, so the gate suppressed progress before the
actor learned to reach.  The validation-calibrated `high=0.02` is a floor
caused by that negative tail, not a swept value.  The actor never recovered.

Deeper audit: the h did not actually learn the future-min target.  Its
correlation with the target is `0.079` overall and `0.013` on safe rows,
versus `0.811`/`0.427` for the Ant predictive h.  The nav target loss used
`Huber(delta=0.1)` with 10 epochs/lr `1e-4` and an hdot term, while Ant used
`SmoothL1(beta=1)` for 40 epochs/lr `1e-3` plus an explicit gradient-alignment
loss and no hdot term.  The nav h is a binary classifier whose safe lower tail
is uncalibrated, so the gate floors at `high=0.02`, closes 16.6% of top-decile
progress transitions and 8.6% of near-goal steps, and suppresses the dense
progress signal before the from-scratch actor learns to reach.  Nav h data also
has only 0.6% unsafe rows versus 26.1% for the Ant predictive data.

Gate-off control with the same h and actor budget (`task_gate_high=0.0`, 100ep
deterministic): `safe_episode=0.03`, `official_reach=0.135`,
`official_success=0.0763`, `episode_return=7.34`.  It is also far below h-only
(`0.620/0.576/0.5325/69.78`), so the Stage-B h features are harmful even
without gate suppression.  This is a diagnostic, not a promoted arm.

**Decision:** Stage B seed 0 fails both primary safety and task guardrails.
Do not run seeds 1/2 and do not claim a multi-seed Stage-B result.  The h
classifier itself is usable offline (AUROC 0.967), but this h/gate/actor
coupling is not.  The next controlled experiment must fix the h target
learning/data coverage or the gate coupling, not sweep the threshold.

### 3.9 Navigation Stage B repair: predictive future-min h (3-seed pass)

The repair changed only the h objective/data and the gate calibration derived
from the new h.  The repaired h was trained from scratch for 40 epochs on
`data_stageb_repair` (best epoch 26): val target corr `0.800`, safe-row corr
`0.768`, gradient cosine `0.911`, BA `0.994`, AUROC `0.9997`; test target corr
`0.794`, safe-row corr `0.736`, gradient cosine `0.911`, BA `0.993`, AUROC
`0.9997`.  The pre-registered gate calibration gives `low=0.0`,
`high=0.267270`; on validation the gate is fully closed on 100% of unsafe
rows, mean `0.95` on safe rows, and only `4.6%` of safe rows fall below 0.5.

The actor was retrained from scratch for 16,384 steps with the frozen
repaired h and `task_gate_low/high=0.0/0.267270`, then evaluated on the fixed
final checkpoint over 100 deterministic episodes (eval seed 10000):

| seed-0, 100ep | h-only | old Stage B | repaired Stage B |
|---|---:|---:|---:|
| safe_episode | 0.620 | 0.000 | **0.840** |
| collision_episode | 0.380 | 1.000 | **0.160** |
| official_safety | 0.9287 | 0.4400 | **0.9738** |
| official_success | 0.5325 | 0.0375 | **0.4275** |
| official_reach | 0.5763 | 0.0700 | **0.4350** |
| episode_return | 69.78 | -23.60 | **60.91** |

`reach_episode_any=0.97`, `h_mean=0.836`, `h_min=0.648`.  The 16k 20ep
monitor was already `safe_episode=0.90`, `official_success=0.4875`, return
`64.74`, so this was not a late collapse.  The old Stage-B failure modes
(harmful h features, gate suppression) are gone.

**Verdict (seed 0):** the safety guardrail passes clearly (+0.22 safe
episodes, +0.045 official safety, collision episodes 38% -> 16%).  Task
retention is partial at seed 0: success is 80.3% and return 87.3% of
matched-seed h-only; success `0.4275` is below the stricter `0.45`
matched-seed floor but above the formal h-only 3-seed mean `0.407`.

Seeds 1/2 then completed under a pre-registered protocol (same frozen h,
gate, config, 16,384 steps, 100 fixed eval scenes).  Final 3-seed result:

| seed | safe_ep | collision_ep | official_safety | success | reach | return |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.840 | 0.160 | 0.9738 | 0.4275 | 0.4350 | 60.91 |
| 1 | 0.510 | 0.490 | 0.9000 | 0.4575 | 0.4963 | 70.92 |
| 2 | 0.730 | 0.270 | 0.9513 | 0.3450 | 0.3575 | 68.74 |
| **mean** | **0.693** | **0.307** | **0.9417** | **0.4100** | **0.4296** | **66.85** |
| **std** | 0.168 | 0.168 | 0.0378 | 0.0583 | 0.0695 | 5.26 |

All pre-registered guardrails pass: mean safe `>=0.55`, safety `>=0.93`,
success `>=0.36`, return `>=55`, and no seed below safe `0.30` / success
`0.20`.  Versus matched h-only 3-seed, safety is `+24.3 pp` and return
`+8.3%`, while success is unchanged (`+0.3 pp`).  This is the first promoted
V6 navigation 3-seed gate result.

Caveats: `n=3`, seed-level paired differences are not statistically
separated from zero (safe `p=0.216`, success `p=0.957`); the safety gain is
carried by seeds 0/2 and seed 1 is nearly neutral.  Against archived external
baselines on the same official environment and metric definitions (different
training/information assumptions), the V6 3-seed safety `0.942` is close to
centralized CBF-QP `0.979` and V5 full 3-seed `0.950`, but V6 success `0.410`
is below V5 full `0.512` and far below CBF-QP `0.781` and GCBF+ `0.861`.
Full comparison, including the h-only 3-seed reference and the unmatched
barrier/h-off references:
`outputs/v6/independent/nav/BASELINE_COMPARISON.md`.  Protocol and per-seed
caveats: `outputs/v6/independent/nav/STAGE_B_REPAIR_MULTISEED.md`.

### 3.10 Read-only h2/actor2 diagnostic

Before authorizing another h/actor iteration, the fixed final actor1
checkpoints were audited for 3 seeds x 100 deterministic episodes (eval seed
10000) with the frozen repaired h.  The audit reproduces the formal
`safe_episode`, `official_safety`, `success`, `reach`, and `return` exactly.

Key result: **155/155 new collision onsets occur with `h_cur < 0` and
`gate_next = 0`**; none occur with `h_cur >= gate_high` or the diagnostic gate
open.  h1 is not blind at impact.  Calibration on actor1 states does not
degrade by the pre-registered rule: pooled target MAE `0.0976` vs offline
repaired test `0.1510`; safe-row correlation drop `0.040` (threshold `0.10`).
Pre-registered impact-state label: **actor/action-limited (h2 likely
marginal)**.

Temporal finding: h1's effective warning horizon is shorter than the 32-step
target.  In the 17-32 steps before collision, h mean is `+0.242` while the
target mean is `-0.222`; the future-collision-region MAE is `0.394`.  h
crosses below `gate_high` a median of 7 steps before collision and below zero
a median of 2 steps before collision.  The cause is a real data-source
mismatch: h1 was trained on h-only seed0 and Stage-A gate-progress seed0
rollouts plus random coverage, not on final repaired Stage-B actor1.  Pooled
MAE (`0.0976`) is dominated by far-safe rows and hides this.

Overall diagnosis: **mixed** -- temporal h limitation before impact plus
actor/action limitation at impact.  The h2-on-actor1-data hypothesis was
tested in **Stage C0a** (100 training episodes + 101 held-out evaluation
episodes, 40 epochs, h2 from scratch, same target/arch/loss/calibration).
It **failed**: 17-32-step risk-region MAE `0.826` vs h1 `0.709` (+16.5%
worse), 9-16-step MAE `0.567` vs `0.442`, future-collision MAE `0.478` vs
`0.375`, despite global MAE improving (`0.084` vs `0.092`).  The nominal
+11-step warning lead is a gate-threshold artifact; threshold-free last
`h > 0` is unchanged at 8 steps.  C0b was not run and Stage C1 actor2 was
not trained.  A post-hoc seed-0 h-swap of h2 into actor1 was slightly better
than h1 (safe episode `0.90` vs `0.84`, official safety `0.985` vs `0.974`,
success `0.439` vs `0.428`), but it is one seed, not significant, and
off-policy; it does not change the C0a decision.  Following explicit
approval, a new **C1a seed-0 actor2 pilot** with h2 seed 0 (from scratch,
same cost/steps, 100ep eval seed 10000) was run.  It **failed all five
pre-registered checks**: `safe_episode 0.510` vs `0.840`,
`official_safety 0.89375` vs `0.97375`, `official_success 0.33125` vs
`0.42750`, `official_reach 0.36250` vs `0.43500`, `episode_return 52.22` vs
`60.91`; collisions rose `0.16 -> 0.49`.  The training gate was almost
always open (`task_gate_mean≈0.99`), so this is not gate suppression.  The
from-scratch h2/actor2 branch is stopped and C1b is not authorized.  A
separate **C1w 4k warm-start of actor1 with h2** (no h1 control) was run as a
diagnostic adaptation arm and **failed the C1a guardrails**: 100ep safe
`0.840`, official safety `0.97625`, success `0.20250`, reach `0.20625`, return
`28.85`.  It matches actor1 on safe episodes and collisions, stays below the
no-training h2 swap on safety, and collapses the task.  The pre-registered
staged rule applies: the h1-continued control was not run and the h2
warm-start branch is stopped.  It is not a C1a pass.  A new evaluation-only
**C1s** replication (h2 seed 0 on fixed actor1 seeds 1/2, 100ep each) is
complete and **`consistent_positive`**: seed 1 safe `0.56` vs `0.51`,
official safety `0.91250` vs `0.90000`, return `74.59` vs `70.92`; seed 2
safe `0.74` vs `0.73`, official safety `0.95625` vs `0.95125`, return `64.97`
vs `68.74`.  Mean `Δsafe +0.030` (exactly at the pre-registered `+0.03`
boundary), `Δofficial_safety +0.00875`, `Δsuccess +0.01375`, `Δreach
+0.00813`, `Δreturn -0.05`; all eight checks hold.  This is a thin,
seed-1-driven robustness signal for h2 seed 0 as a frozen input, not a main
claim.  The swap is off-policy and h2 seed 0 is distribution-shifted for
seeds 1/2.  The h-only data/target axis is stopped; the next controlled arm
after C1w/C1s, if needed, must be safety coupling or action conditioning
(actor/action-limited impact).  Result record:
`outputs/v6/independent/nav/STAGE_C0_H2_TEMPORAL.md`; diagnostic:
`outputs/v6/independent/nav/H2_ACTOR2_DIAGNOSTIC.md`.

## 4. Ant branch

### 4.1 Ant h data and offline h

Predictive h data (`outputs/v6/independent/ant/h_data_predictive/summary.json`):

- 120 episodes from the V6 cost-only 2M behavior policy, stochastic actions;
- 84,523 rollout rows + 30,000 augmentation = 114,523 rows;
- 29,791 current/future-unsafe rows, 84,732 far-safe rows;
- old future-window `0` rows are retained and become negative labels;
- horizon 32, `h_scale=0.5`, `kappa=0.5`, `kappa_z=0.2`.

Predictive h test (`h_predictive/summary.json`):

| metric | value |
|---|---:|
| balanced accuracy | 0.9145 |
| unsafe recall | 0.8793 |
| false alarm | 0.0364 |
| AUROC | 0.9511 |
| gradient cosine | 0.9225 |

### 4.2 Ant 2M three-seed robustness

100 episodes, eval seed 20000, predictive h + forward gate:

| mode | arm | forward | safe | wall | topple | cost_fraction | survived |
|---|---|---:|---:|---:|---:|---:|---:|
| det | cost-only mean | 905.75 | 0.573 | 0.420 | 0.010 | 0.0199 | 0.990 |
| det | predictive gate mean | 756.60 | **0.977** | **0.023** | **0.000** | **0.00232** | **1.000** |
| sto | cost-only mean | 525.38 | 0.067 | 0.603 | 0.563 | 0.0987 | 0.443 |
| sto | predictive gate mean | 472.22 | **0.317** | **0.267** | **0.470** | **0.0264** | **0.530** |

See `PREDICTIVE_H_GATE_2M.md`.

### 4.3 Ant 10M retention (seed 0 and 3-seed sweep)

| steps | mode | arm | forward | safe | wall | topple | cost_fraction | survived |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 2M | det | cost-only | 840.11 | 0.90 | 0.10 | 0.01 | 0.0044 | 0.99 |
| 2M | det | predictive gate | 572.13 | 0.95 | 0.05 | 0.00 | 0.0069 | 1.00 |
| 5M | det | cost-only | 2284.00 | 0.02 | 0.97 | 0.11 | 0.0700 | 0.89 |
| 5M | det | predictive gate | 1792.69 | 0.99 | 0.01 | 0.00 | 0.00004 | 1.00 |
| 10M | det | cost-only | 4055.37 | 0.10 | 0.89 | 0.02 | 0.0222 | 0.98 |
| 10M | det | predictive gate | 3723.09 | 0.97 | 0.03 | 0.00 | 0.00176 | 1.00 |
| 2M | sto | cost-only | 496.91 | 0.14 | 0.57 | 0.54 | 0.0721 | 0.46 |
| 2M | sto | predictive gate | 402.24 | 0.27 | 0.27 | 0.53 | 0.0306 | 0.47 |
| 5M | sto | cost-only | 1334.04 | 0.00 | 0.80 | 0.52 | 0.0695 | 0.48 |
| 5M | sto | predictive gate | 1207.69 | 0.54 | 0.14 | 0.34 | 0.00525 | 0.66 |
| 10M | sto | cost-only | 3067.39 | 0.04 | 0.91 | 0.19 | 0.0508 | 0.81 |
| 10M | sto | predictive gate | 2923.73 | 0.78 | 0.05 | 0.17 | 0.00419 | 0.83 |

At 10M deterministic the gate retains 91.8% of cost-only forward and 94.3%
of its return, while cutting cost by 92% and raising safe episodes from 0.10
to 0.97.  Relative to V5 MAPPO-Lag it retains 97.7% of return.  Stochastic
retains 95.3% forward and 95.3% return, cuts cost by 92%, and raises safe
episodes from 0.04 to 0.78.

**Seeds 1/2 completed (3-seed 10M sweep).**  Formal 100-episode results:

| mode | arm | forward | safe | wall | cost_fraction |
|---|---|---:|---:|---:|---:|
| det | cost-only mean | 4119.70 | 0.063 | 0.923 | 0.0296 |
| det | predictive gate mean | **3771.51** | **0.97** | **0.02** | **0.0012** |
| sto | cost-only mean | 3206.28 | 0.020 | 0.950 | 0.0618 |
| sto | predictive gate mean | **3010.73** | **0.84** | **0.023** | **0.0020** |

Deterministic retention is 91.6% with 96.0% cost reduction; stochastic
retention is 93.9% with 96.8% cost reduction.  Every seed is safe under the
gate (det `0.97/0.95/0.99`) while every cost-only seed is unsafe (det
`0.10/0.08/0.01`).  See `RETENTION_10M.md` and
`MULTISEED_10M_PLAN.md`.

### 4.4 V5 reference comparison (Ant)

V5 reference: `Safe-MARL-proposal_v4/outputs/v5/ant/baselines/comparison_100ep.md`;
V5 h 3-seed: `.../v5/official/ant/comparison_3seed.md`.

| method | return | cost rate | wall rate | safe ep | survived | x_final |
|---|---:|---:|---:|---:|---:|---:|
| V5 MAPPO 10M | 4477.2 | 31.57% | 31.54% | 0% | 89% | 203.4 |
| V5 MAPPO-Lag 2M | 4471.2 | 2.18% | 2.16% | 0% | 95% | 203.2 |
| V5 learned h 3-seed | 4484.0 ± 123.5 | 2.06% | ~2.0% | 0% | 94.3% | 203.5 |
| **V6 predictive gate 10M (seed 0 row)** | **4366.6** | **0.176%** | **0.176%** | **97%** | **100%** | **186.2** |

Caveat: V5 evaluation uses seed 10000; V6 Ant uses seed 20000.  V5 MAPPO-Lag
is a 2M constrained fine-tune; V5 h uses the MAPPO-Lag prior.  V6 is from
scratch.  The V6 row above is the seed-0 reference for the V5 table; the
completed 3-seed 10M sweep gives deterministic forward `3771.51` (91.6%
retention), safe `0.97`, and stochastic forward `3010.73` (93.9%), safe
`0.84`; see `outputs/v6/independent/ant/RETENTION_10M.md`.  Treat this as a
strong reference comparison, not an exact head-to-head.

### 4.5 Ant V_T transfer diagnostic (closed null, single seed)

Pre-registration and full record:
`outputs/v6/independent/ant/STAGE_ANT_VT_TRANSFER.md`.

Question: on the existing seed-0 10M h-only base, does a frozen task value
`V_T` fed as a 1-D relative scalar `A_T` improve the task beyond a matched
no-V_T 1M continuation without losing the h-only safety effect?

Protocol: 1M extra steps (`10M -> 11M`), frozen h, cost penalty `0.1`,
reward/optimizer/seed shared by both arms; the only difference is the 42nd
actor input.  V_T is a separate 128x128 MLP over the 31-D agent observation,
trained offline on the ungated official task return (`K=64`,
`gamma=0.99`); it passed the gate (val EV `0.4838`, test Pearson `0.6597`).
Both arms started from the same checkpoint with a fresh Adam optimizer; the
new actor column is zero-initialized after the input LayerNorm, so step 0 is
exactly the base policy.

| arm | mode | safe_ep | forward | return | cost_fraction |
|---|---|---:|---:|---:|---:|
| matched no-V_T control | det | 0.96 | 4014.03 | 4630.37 | 0.000187 |
| 1-D `A_T` warm-start | det | 0.96 | 3960.83 | 4574.53 | 0.000747 |
| matched no-V_T control | stoch | 0.78 | 3440.15 | 3885.84 | 0.00319 |
| 1-D `A_T` warm-start | stoch | 0.73 | 3247.81 | 3670.66 | 0.00294 |

Decision: `control_valid = true`, `task_gain = false` (forward `-1.33%`,
return `-1.21%`), `safety_not_worse = true`, so
`vt_transfer_supported = false`.  The line is stopped; no 5-D,
longer-budget, or new-seed arm is launched.

Both arms improved over the 10M base (control forward `+7.8%`, warm-start
`+6.4%`), so the 1M continuation, not V_T, explains the gain.  The `A_T`
columns learned nonzero weights (norm `0.086`/`0.098`), so the channel was not
silently disconnected; it simply produced no measurable task benefit under
this protocol.  This closes only this 1-D screen; it is not evidence about a
5-D gradient variant.  Single seed, diagnostic only; the promoted Ant 3-seed
claim is unchanged.

## 5. What the Ant success actually teaches

The successful Ant mechanism is **not** h alone.  It is:

```text
predictive future risk h
        + forward-only safety gate
        + fixed official cost penalty
```

with no QP, no Lagrangian, no action filter, no expert labels.

The key mechanism:

```text
that gate = clip((h_next - low) / (high - low), 0, 1)
r_train = non_forward_reward
          + gate * forward_reward
          - cost_penalty * official_cost
```

- h is trained on future-window risk, retaining the old `0` rows;
- h is negative before the unsafe event, not only at it;
- when h is low, forward reward no longer pays;
- when h is healthy, the gate is open and task can grow;
- evaluation has the gate off and uses official metrics.

This is a **soft state-dependent reward constraint**, not a formal hard
constraint.  It is strong empirically (97% safe at 10M deterministic, 92%
cost reduction) but not a CBF certificate.

## 6. Is this enough to enter task navigation?

Two different questions:

### A. Enough to start a navigation safety-gate prototype?

**Yes.**  The Ant result gives a concrete, cheap hypothesis: add a
predictive-h forward/progress gate to the existing 8-agent navigation h-only
policy.  This is a natural first transfer experiment and does not require
V5/official h/QP.

### B. Enough to enter the full navigation task-channel experiment?

**Not yet.**  The navigation gate prototype now has a matched 3-seed
comparison against h-only:

```text
                 h-only            repaired gate
success          0.407 ± 0.132     0.410 ± 0.058
safe_episode     0.450 ± 0.193     0.693 ± 0.168
official_safety  0.878 ± 0.066     0.942 ± 0.038
return           61.73 ± 10.64     66.85 ± 5.26
```

That supports the gate itself, but the formal 16k h-off baseline is still not
present, and the task-channel effect would be confounded with the gate change.
Do not open the task channel/V_T until the h-off baseline exists and a separate
pre-registered task-channel experiment is approved.

## 7. Proposed navigation transfer

Do not copy Ant weights, thresholds, MLP, or 9-D state.  Transfer only the
mechanism.

### 7.1 Use the existing navigation h first

The navigation graph h already has horizon labels, stopping-clearance margin,
and hdot training; offline test BA is 0.985.  Do not retrain it before testing
the gate.  The missing piece is the **task-reward gate**, not another h.

### 7.2 Per-agent progress gate

The navigation environment already exposes per-agent `newly_reached`,
`dist_to_goal`, `collision`, and every agent has its own h/grad.  Define:

```text
gate_i = clip((h_next_i - low) / (high - low), 0, 1)
progress_i = old_distance_i - distance_i
r_task_i = gate_i * (reward_scale * progress_i + reach_bonus * newly_reached_i)
r_i = r_task_i - unsafe_penalty * collision_i
```

Use each agent's own gate for its progress, so one cautious agent does not
freeze the team.  For a shared terminal/reach bonus, use:

```text
team_gate = min_i gate_i
```

or a soft mean, so a team-level success bonus is not collected while another
agent is in a predicted-risk state.

### 7.3 Keep the cost un-gated

Never gate the official collision/unsafe cost.  The gate is only on task
progress/reach reward.  Keep the collision penalty and cost critic unchanged.
This is the same separation that worked in Ant.

### 7.4 Calibrate the nav gate on nav h, not Ant h

Ant uses `low=0, high=0.05` because its predictive h has a different scale.
Navigation h is centered by `h_offset=-0.8621`; its evaluation h mean is about
0.11 and h_min is negative.  Calibrate `low/high` on the V6 navigation
validation split, e.g. low at the operating threshold and high at a low
quantile of safe h.  Do not copy 0.05.

### 7.5 Multi-agent details

- Use per-agent h for local progress; use a team min/soft aggregate for shared
  success bonus.
- Do not gate on a single global h for all agents; that can create a
  standstill when one agent is near an obstacle.
- Keep h/grad as actor features; do not add action projection/QP.
- If the current h is too late, add hdot to the gate:
  `gate_i = f(h_next_i + beta * (h_next_i - h_i))`; test only if needed.

### 7.6 Evaluation

Compare, same seeds/steps/eval seed:

1. h-off;
2. h-only (current formal baseline);
3. h + task gate;
4. h + task gate + task channel (only after 3 passes).

Primary: `safe_episode`, `collision_episode`, `official_safety`; guardrail:
`success`, `reach`, `episode_return`.  Require no success collapse, and run
3 seeds x 100 episodes before a claim.

## 8. Pending TODO

- [x] Ant 10M seeds 1 and 2 (overnight batch).
- [x] Matched cost-only 10M seeds 1 and 2.
- [x] Ant 10M final 100-episode deterministic/stochastic evaluation, all
      three seeds.
- [ ] Ant 10M multi-checkpoint evaluation at 2M/5M.
- [x] **Ant V_T transfer diagnostic (single seed, closed null):** 1-D `A_T`
      warm-start from the seed-0 10M h-only base, 1M extra steps, matched
      no-V_T 1M control; frozen h/cost/reward, Ant V_T trained offline on
      task return only.  V_T gate passed; control valid; warm-start did not
      beat control on deterministic forward (`3960.83` vs `4014.03`) or
      return (`4574.53` vs `4630.37`); safety equal (`0.96`).
      `vt_transfer_supported = false`; line stopped.  Record:
      `outputs/v6/independent/ant/STAGE_ANT_VT_TRANSFER.md`.
- [ ] Navigation h-off formal 16k baseline, 3 seeds x 100 episodes.
- [x] Navigation predictive-h task-gate pilot, 1 seed x 100 episodes.
- [x] Navigation Stage A (progress-only gate), 1 seed x 100 episodes.
- [x] Navigation Stage B (on-policy h + future-min target), 1 seed x 100
      episodes.  Failed: `safe_episode=0.0`, `success=0.0375`.
- [x] Navigation Stage B repair (predictive future-min h), seed 0: safety
      passes (`safe_episode=0.840`, `official_safety=0.9738`) with partial
      task retention (`success=0.4275`, `return=60.91`).
- [x] Repaired Stage B seeds 1/2 under the pre-registered protocol.
      **3-seed pass**: `safe_episode=0.693 +/- 0.168`,
      `official_safety=0.9417 +/- 0.0378`, `official_success=0.4100 +/-
      0.0583`, `episode_return=66.85 +/- 5.26`.  All aggregate guardrails
      and the no-catastrophic-seed rule pass.  `n=3` caveat: seed-level
      differences are not statistically separated; seed 1 is nearly neutral.
      Record: `outputs/v6/independent/nav/STAGE_B_REPAIR_MULTISEED.md`.
- [x] Do not sweep `task_gate_high`; it was not swept.
- [x] Read-only h2/actor2 diagnostic, 3 seeds x 100 episodes.  Impact-state
      label: actor/action-limited (155/155 collision onsets at `h_cur < 0`,
      `gate_next = 0`).  Overall diagnosis: mixed, because h1 was trained on
      h-only/Stage-A rollouts, not final actor1, and the 17-32-step risk
      region is poorly calibrated (MAE `0.394`).  No h2/actor2 training was
      launched at that stage.  Record:
      `outputs/v6/independent/nav/H2_ACTOR2_DIAGNOSTIC.md`.
- [x] Stage C0a h2-only temporal-calibration screen: **failed** (17-32-step
      risk-region MAE `0.826` vs h1 `0.709`).  C0b not run; C1 actor2 not
      trained.  The h-only data/target axis is stopped.
- [x] Post-hoc seed-0 h2+actor1 closed-loop h-swap: slightly better than h1
      (safe `0.90` vs `0.84`, official safety `0.985` vs `0.974`), but one
      seed and not significant; diagnostic only.  Record:
      `outputs/v6/independent/nav_c0_h2_actor1_swap/README.md`.
- [x] C1a seed-0 actor2 pilot with h2: **failed all five checks**.  100ep
      safe `0.510` vs `0.840`, official safety `0.89375` vs `0.97375`,
      success `0.33125` vs `0.42750`, reach `0.36250` vs `0.43500`, return
      `52.22` vs `60.91`; collisions `0.16 -> 0.49`.  Gate was almost always
      open (`task_gate_mean≈0.99`).  From-scratch h2/actor2 branch stopped;
      C1b not authorized.  Record:
      `outputs/v6/independent/nav/STAGE_C1A_H2_ACTOR2_PILOT.md`.
- [x] C1w actor1+h2 4k warm-start: diagnostic adaptation only, no h1
      control.  **Failed**: safe `0.840`, official safety `0.97625`, success
      `0.20250`, reach `0.20625`, return `28.85`; `promising_adaptation =
      false`.  The h1-continued control was not run; the h2 warm-start
      branch is stopped.  Not a C1a pass.  Record:
      `outputs/v6/independent/nav/STAGE_C1W_H2_ACTOR1_4K.md`.
- [x] C1s evaluation-only h2 seed-0 swap on actor1 seeds 1/2:
      **`consistent_positive`** (all eight checks).  Mean `Δsafe +0.030`
      (exactly at the `+0.03` threshold), `Δofficial_safety +0.00875`,
      `Δsuccess +0.01375`, `Δreach +0.00813`, `Δreturn -0.05`.  Thin and
      seed-1-driven; off-policy; one h2 checkpoint.  Not a from-scratch
      result and not a main claim.  Record:
      `outputs/v6/independent/nav/STAGE_C1S_H2_ACTOR_SEED_SWAP.md`.
- [ ] C0b matched h2-seed study: discussion item only; needs a new
      pre-registration and explicit approval.  Do not launch implicitly.
- [ ] Next controlled arm, if approved: safety coupling or action
      conditioning (one axis only).  Do not combine with a gate/cost/task
      change.
- [ ] Formal 16k h-off 3-seed baseline remains absent.
- [x] **T0w task-channel line: strict safety-first rule missed; task-heavy
      trade-off documented.**  Actor1 + frozen finite-horizon V_T 4k
      warm-start; h1/gate/cost unchanged.  V_T passed its quality gate
      (val EV `0.2270`, test Pearson `0.4991`).  Warm-start 100ep: safe
      `0.890`, official safety `0.98250`, success `0.67875`, reach `0.685`,
      return `161.88` (actor1 success `0.42750`, return `60.91`).  Matched
      no-V_T 4k control: safe `0.930`, official safety `0.99000`, success
      `0.380`, return `69.67`.  `control_valid` and `task_gain` true
      (success `+0.29875`, return `+92.20`), but `safety_not_worse = false`
      (safe `0.890 < 0.900`, one episode); `vt_channel_supported = false`
      under the pre-registered safety-first rule.  Task-first reading: a
      task-heavy Pareto point with a large task gain and a small,
      statistically unresolved safety discount; not a safety-non-inferiority
      pass.  No from-scratch arm is authorized yet; a 3-seed
      `h1 + gate + V_T` replication is the optional next step.  One seed;
      not a main claim.  Record:
      `outputs/v6/independent/nav/STAGE_T0W_VT_WARMSTART.md`.
- [x] Record the 3-seed 10M result in this file,
  `outputs/v6/independent/ant/RETENTION_10M.md`, and
  `outputs/v6/independent/ant/MULTISEED_10M_PLAN.md`.

## 9. Claim boundary

Safe claims today:

- V6 Ant predictive-h + gate: strong empirical safety/task result at 2M
  three-seed and 10M three-seed (det 0.97 safe at 91.6% forward retention;
  stochastic 0.84 safe at 93.9% retention).
- V6 navigation h-only: offline h is strong; closed-loop h-only is mixed
  (formal 3-seed success 0.407, safe 0.450).
- V6 navigation safety gate: broad gate is a safety/task tradeoff; Stage A
  failed the safety guardrail; the original Stage B failed both guardrails.
  The repaired predictive-h Stage B **passes the pre-registered 3-seed
  guardrails** (seeds 0/1/2): `safe_episode=0.693 +/- 0.168`,
  `official_safety=0.9417 +/- 0.0378`, `success=0.4100 +/- 0.0583`,
  `return=66.85 +/- 5.26`.  Versus matched h-only, safety is `+24.3 pp` and
  return `+8.3%` with success unchanged.  This is a promoted 3-seed
  navigation gate result, not a CBF certificate and not a task-channel
  result.  `n=3`; seed-level differences are not statistically separated
  and the gain is seed-0/2 driven.

Do not claim:

- formal/certified safety for either task;
- navigation task-channel safety non-inferiority or from-scratch success;
  T0w is a single-seed task-heavy trade-off, not a safety-non-inferiority pass;
- V5/official-h equivalence;
- that the h2/actor2 from-scratch iteration fixes the remaining collisions.
  C0a was run and failed: h2 improved global MAE but worsened the 17-32-step
  risk region.  C1a was then run and **failed all five closed-loop checks**:
  actor2+h2 was worse than actor1+h1 on safety and task (safe `0.510` vs
  `0.840`, official safety `0.89375` vs `0.97375`).  C0b and C1b were not
  run.  A post-hoc seed-0 h2+actor1 swap was slightly better but
  one-seed/not significant and off-policy; it is not evidence that h2 would
  help a newly trained actor2.  C1w is a diagnostic actor1+h2 warm-start
  with no h1 control and it failed (task collapse, no safety gain over h1 and
  below the no-training swap); C1s is an evaluation-only, off-policy seed
  replication whose thin `consistent_positive` result (mean `Δsafe` exactly
  at `+0.03`) does not establish h2 seed reproducibility or a from-scratch
  benefit.  Neither can support a causal h2 claim.
