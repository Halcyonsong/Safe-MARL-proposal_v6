# V6 navigation baseline comparison

Canonical project status: [`../../STATUS.md`](../../STATUS.md).

Status: 3-seed comparison, updated 2026-10-27.  The main V6 row is the
repaired Stage-B actor (seeds 0/1/2).  Seed-0 matched rows are retained for
ablation comparison.  External V5/CBF rows are archived references, not
reruns.  The two collision-aware MAPPO rows added on 2026-10-27 are new
matched 16k runs, not archived V5 results.

## Protocol

Matched V6 rows use:

- 8 agents, 4x4 area, 8 obstacles, 256 steps;
- 16,384 training steps, seeds 0/1/2;
- separate 100-episode deterministic evaluation, eval seed 10000;
- fixed final checkpoint (not the monitor-best checkpoint);
- frozen V6-owned repaired h; no official h, QP, expert action, or V5 artifact.

`official_success` is the per-agent any-time success rate (4 agents x 100
episodes = 400 trajectories per seed); `safe_episode` is the fraction of the
100 episodes with no collision.  `official_safety` is the environment safety
metric, higher is better.

## V6 repaired Stage B: 3 seeds x 100 episodes

| seed | safe_ep | collision_ep | official_safety | success | reach | return |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.840 | 0.160 | 0.9738 | 0.4275 | 0.4350 | 60.91 |
| 1 | 0.510 | 0.490 | 0.9000 | 0.4575 | 0.4963 | 70.92 |
| 2 | 0.730 | 0.270 | 0.9513 | 0.3450 | 0.3575 | 68.74 |
| **mean** | **0.693** | **0.307** | **0.9417** | **0.4100** | **0.4296** | **66.85** |
| **std** | 0.168 | 0.168 | 0.0378 | 0.0583 | 0.0695 | 5.26 |

Source: `outputs/v6/independent/nav_stageb_repair_16k/aggregate_100ep.json`.

Matched h-only reference (same training seeds and eval scenes):

| h-only 3-seed | safe_ep | collision_ep | official_safety | success | reach | return |
|---|---:|---:|---:|---:|---:|---:|
| mean | 0.450 | 0.550 | 0.8783 | 0.4067 | 0.4450 | 61.73 |
| std | 0.1931 | 0.1931 | 0.0659 | 0.1316 | 0.1332 | 10.64 |

Delta (repaired - h-only): safe `+24.3 pp`, official safety `+6.3 pp`,
success `+0.3 pp`, reach `-1.5 pp`, return `+8.3%`.

### The +24.3 pp is mostly not an h-retrain effect (P0, added 2026-10-27)

The h-only reference has **no task gate**, while the repaired Stage-B row has
a training-time gate.  The delta above therefore mixes two changes: h
checkpoint quality and gate activation.  P0 completes the original-h +
broad-positive-reward gate row (`task_gate_high=0.13`, scope
`all_positive`), whose seed 0 existed but seeds 1/2 were not evaluated.

| original-h broad gate | safe_ep | collision_ep | official_safety | success | reach | return |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.700 | 0.300 | 0.9400 | 0.3725 | 0.3913 | 64.96 |
| 1 | 0.490 | 0.510 | 0.9000 | 0.4838 | 0.5275 | 80.91 |
| 2 | 0.740 | 0.260 | 0.9575 | 0.4225 | 0.4387 | 65.11 |
| **mean** | **0.6433** | **0.3567** | **0.9325** | **0.4263** | **0.4525** | **70.33** |
| **std** | 0.1343 | 0.1343 | 0.0295 | 0.0557 | 0.0692 | 9.17 |

Three-row comparison (all 3 seeds x 100 episodes, eval seed 10000):

| row | h checkpoint | training gate | safe_ep | official_safety | success | reach | return |
|---|---|---|---:|---:|---:|---:|---:|
| H0 formal h-only | original | off | 0.4500 | 0.8783 | 0.4067 | 0.4450 | 61.73 |
| **P0 original h + broad gate** | original | on | **0.6433** | **0.9325** | **0.4263** | **0.4525** | **70.33** |
| A repaired h + progress gate | repaired | on | 0.6933 | 0.9417 | 0.4100 | 0.4296 | 66.85 |

- **P0 - H0:** safe `+19.3 pp`, official safety `+5.4 pp`, success `+2.0 pp`,
  reach `+0.8 pp`, return `+13.9%`.  The broad gate alone accounts for most of
  the gain in this original-h comparison.  The clean repaired-h factorial
  below revises the attribution: with the repaired h, h retrain alone
  contributes `+18.3 pp` safe (but costs task), and the progress gate adds
  `+6.0 pp` safe while recovering task.  "Gate-dominated" is specific to the
  broad gate on the original h, not a universal decomposition.
- **A - P0:** safe `+5.0 pp`, official safety `+0.9 pp`, success `-1.6 pp`,
  reach `-2.3 pp`, return `-4.9%`.  Per-seed safe deltas are
  `+14/+2/-1 pp`: the repaired-h advantage over the original-h gate row is
  small, seed-dependent, and comes with a task cost.  A does not dominate P0.
- This comparison is still not a clean h-retrain estimate because P0 and A
  also differ in gate scope/threshold (`all_positive`, `0.13` vs
  `progress_only`, `0.26727`).  The clean single-factor h-retrain comparison is
  **B (repaired h, gate off) vs H0 (original h, gate off)** in the P1
  factorial below.

P0 source: `outputs/v6/independent/nav_h_gate_16k/seed{0,1,2}/eval_100ep_final.json`.

## P1: repaired-h 2x2 factorial (2026-10-27)

Pre-registration: `outputs/v6/independent/nav/P1_REPAIRED_H_FACTORIAL.md`.
All cells use the same frozen repaired h, 16,384 steps, seeds 0/1/2, fixed
unsafe penalty 10, from-scratch actors, final checkpoint, and 100
deterministic episodes on scenes `10000..10099`.  Aggregation:
`outputs/v6/independent/nav_repair_factorial/aggregate_100ep.{json,md}`.

### Cell means

| cell | actor h/grad input | training gate | safe_ep | official_safety | success | reach | return |
|---|---|:---:|---:|---:|---:|---:|---:|
| A repaired h + gate | on | on | 0.6933 | 0.9417 | 0.4100 | 0.4296 | 66.85 |
| B repaired h, gate off | on | off | 0.6333 | 0.9267 | 0.3229 | 0.3517 | 55.67 |
| C no actor h input, gate on | off | on | 0.0200 | 0.4987 | 0.0367 | 0.1008 | -6.63 |
| D no actor h input, gate off | off | off | 0.0333 | 0.5479 | 0.0367 | 0.0708 | 7.21 |
| H0 reference (original h, not a 2x2 cell) | on | off | 0.4500 | 0.8783 | 0.4067 | 0.4450 | 61.73 |

### Per-seed values for the new cells

| cell | seed | safe_ep | official_safety | success | reach | return |
|---|---:|---:|---:|---:|---:|---:|
| B | 0 | 0.64 | 0.9425 | 0.3738 | 0.3925 | 56.54 |
| B | 1 | 0.39 | 0.8625 | 0.4625 | 0.5250 | 78.58 |
| B | 2 | 0.87 | 0.9750 | 0.1325 | 0.1375 | 31.90 |
| C | 0 | 0.06 | 0.7275 | 0.0163 | 0.0250 | -58.13 |
| C | 1 | 0.00 | 0.4500 | 0.0513 | 0.1000 | 5.27 |
| C | 2 | 0.00 | 0.3187 | 0.0425 | 0.1775 | 32.96 |
| D | 0 | 0.03 | 0.5563 | 0.0325 | 0.0762 | -37.26 |
| D | 1 | 0.07 | 0.5813 | 0.0375 | 0.0775 | 29.65 |
| D | 2 | 0.00 | 0.5062 | 0.0400 | 0.0588 | 29.24 |

### Contrasts

Deltas are 3-seed mean differences; pp for rates, absolute return for money.
Per-seed safe deltas are in the last column.  Paired `p` is a two-sided t-test
over the three training seeds, **descriptive only** (pre-registration makes no
significance claim at `n=3`).

| contrast | estimand | safe_ep | official_safety | success | reach | return | safe p | per-seed safe |
|---|---|---:|---:|---:|---:|---:|---:|---|
| B - H0 | repaired-h effect, gate off | +18.3 pp | +4.8 pp | -8.4 pp | -9.3 pp | -6.1 | 0.502 | +2/-10/+63 |
| A - B | gate effect, h input on | +6.0 pp | +1.5 pp | +8.7 pp | +7.8 pp | +11.2 | 0.618 | +20/+12/-14 |
| A - C | h-input effect, gate on | +67.3 pp | +44.3 pp | +37.3 pp | +32.9 pp | +73.5 | 0.015 | +78/+51/+73 |
| B - D | h-input effect, gate off | +60.0 pp | +37.9 pp | +28.6 pp | +28.1 pp | +48.5 | 0.064 | +61/+32/+87 |
| C - D | gate effect, h input off | -1.3 pp | -4.9 pp | 0.0 pp | +3.0 pp | -13.8 | 0.697 | +3/-7/0 |
| (A-B)-(C-D) | h-input x gate interaction | +7.3 pp | +6.4 pp | +8.7 pp | +4.8 pp | +25.0 | — | — |

Pooled episode check (descriptive): pooling the 300 evaluation episodes per
row gives safe counts H0 `135/300`, P0 `193/300`, A `208/300`, B `190/300`,
C `6/300`, D `10/300`.  Wilson 95% intervals for A `[0.639, 0.743]` and B
`[0.577, 0.686]` overlap, while A/C and B/D are disjoint.  This matches the
seed-level reading: the h-input collapse is large, A - B is not resolved.
Episode-level intervals ignore training-seed clustering and are optimistic.

### Reading

1. **Actor h input is load-bearing at this training budget.**  Removing it
   collapses both gate-on (C) and gate-off (D) cells: safe episodes `2.0% /
   3.3%`, success `3.7%` in both, across all six seeds with no exception.
   The collapse is already visible during training (negative mean reward,
   higher unsafe rate), so it is not an evaluation artifact.  D is also
   consistent with the archived matched no-h fixed-penalty baseline (safe
   `0.033` vs `0.033`, official safety `0.548` vs `0.519`, success `0.037`
   vs `0.014`), which corroborates that V6's 16k from-scratch actor does not
   learn this task without h features.  This is a statement about the 16k V6 budget, not a
   claim that h is necessary at every budget.

2. **The gate cannot substitute for h input.**  C - D is flat on safe
   episodes (`-1.3 pp`), success (`0.0 pp`), and reach (`+3.0 pp`), and
   negative on return (`-13.8`).  Gate-only training without actor h does not
   recover task performance.

3. **With h input, the progress gate improves every endpoint
   directionally.**  A - B is `+6.0 pp` safe, `+1.5 pp` official safety,
   `+8.7 pp` success, `+7.8 pp` reach, `+11.2` return.  This is the clean
   training-time gate effect because A and B share the repaired h and differ
   only in gate.  It is not statistically separated (`p=0.618`) and is
   seed-heterogeneous on safety (`+20/+12/-14 pp`); it is a direction, not a
   demonstrated effect.

4. **Repaired h alone is a safety/task trade, not a strict improvement.**
   B - H0 is `+18.3 pp` safe and `+4.8 pp` official safety, but `-8.4 pp`
   success, `-9.3 pp` reach, and `-6.1` return.  The task loss is concentrated
   in B seed 2 (success `0.1325`, return `31.90`), while seeds 0/1 retain
   `0.37/0.46` success.  The repaired-h-only cell is therefore conservative,
   not dominant.

5. **A's profile requires both factors.**  A is the only repaired-h 2x2 cell
   that combines high safety (`0.693`) with retained success (`0.410`) and
   return (`66.85`).  P0 also combines safety with task, but it uses the
   original h and the broad gate, and its safety is lower (`0.643`).
   Descriptively, A dominates B on every endpoint, and A - H0 decomposes
   exactly as `(B - H0) + (A - B) = +18.3 + 6.0 = +24.3 pp` safe.  In this
   chain most of the safe gain comes from the repaired h with the gate off,
   while the gate adds a smaller safe increment and recovers the task
   performance that the h-only cell loses.  This refines the P0-based reading:
   the broad gate on the *original* h accounts for `+19.3 pp` vs H0, but the
   clean repaired-h factorial attributes `+18.3 pp` to h retrain and
   `+6.0 pp` to the progress gate.  The two decompositions differ because P0
   and A use different h checkpoints and gate scopes; neither is a universal
   attribution.

6. **Supported claim.**  The factorial supports a mechanism statement:
   actor h input is required for this 16k from-scratch pipeline, the gate is
   ineffective without it, and the repaired-h + gate package is the only cell
   with both safety and task.  It does **not** establish policy-level
   superiority: `n=3`, no multiplicity correction.  The only `p<0.05`
   single-factor contrast is the gate-on h-input effect (A - C safe
   `p=0.015`), and it is driven by the collapsed C cell; the gate-off
   counterpart (B - D) is `p=0.064`.  The interaction return difference
   (`+25.0`) has paired `p=0.034`, but it is not interpreted because both C
   and D are collapsed and return is heavy-tailed.  The promoted claim
   remains "passes the pre-registered 3-seed guardrails".

### Ablation scope

`--no-use-h-features` zeros the policy's h/grad channels but does not remove
the frozen repaired h from the gate; C is "gate-only, no h policy input", not
"no h network at all".  The V6 policy shares a trunk between the actor and
value heads, so the ablation affects both; "actor h input" in the tables is
shorthand for this policy-level input.  `evaluate_v6.py` loads the config
embedded in each checkpoint, so C/D keep `use_features=false` at evaluation;
there is no train/eval h-input mismatch.  `h_barrier_coef=0` in all cells, so
the ablation does not also remove a dense h-barrier cost.  The gate is
disabled at evaluation in every cell; gate on/off is a training-time
reward-shaping difference, not an inference-time filter.

## A-only 28k budget extension (2026-10-28)

Same frozen repaired h, gate, cost, reward and optimizer as the 16k row;
100 deterministic episodes on `10000..10099`; fixed final checkpoint.  The
16k rows are the same-run checkpoints (bit-identical to the promoted 16k
run).

| budget | mean safe | mean official | mean success | mean return |
|---|---:|---:|---:|---:|
| 16k | 0.6933 | 0.9417 | 0.4100 | 66.85 |
| 24k | 0.7000 | 0.9479 | 0.2704 | 59.97 |
| 28k | 0.7800 | 0.9604 | 0.2767 | 63.45 |

28k per seed: safe `0.86/0.70/0.78`, official `0.9750/0.9450/0.9613`,
success `0.19875/0.40375/0.22750`, return `55.52/84.53/50.29`.  Rule 7.1
passes safety/return/spread but fails mean success (`0.2767 < 0.36`) and
worst-seed success (`0.19875 < 0.20`).  Keep 16k promoted; 28k is a
task-costly budget extension, not a new main row.  Source:
`outputs/v6/independent/nav/nav_3seed_16k_24k_28k_table.json`.

## W16 task warm-start: 3 seeds x 100 episodes (2026-10-28)

| arm | safe | official | success | reach | return |
|---|---:|---:|---:|---:|---:|
| W16 24k mean | 0.7233 | 0.9504 | 0.5958 | 0.6208 | 143.73 |
| W16 28k mean | 0.6633 | 0.9329 | 0.6175 | 0.6525 | 198.57 |
| from-scratch 24k mean | 0.7000 | 0.9479 | 0.2704 | 0.2796 | 59.97 |
| from-scratch 28k mean | 0.7800 | 0.9604 | 0.2767 | 0.2879 | 63.45 |
| promoted 16k mean | 0.6933 | 0.9417 | 0.4100 | 0.4296 | 66.85 |

W16 28k per seed: safe `0.83/0.56/0.60`, official
`0.9725/0.9075/0.9187`, success `0.62125/0.67375/0.55750`, return
`237.36/217.40/140.95`.  The task channel recovers task strongly, but
rule 7.1 fails on mean safe (`0.6633<0.6933`), mean official
(`0.9329<0.9417`) and safe range (`0.27>0.20`), while task/return/
worst-seed bounds pass.  The 24k diagnostic passes mean safety/task but its
range is `0.38`.  **Task-first trade-off; 16k remains promoted.**  Source:
`outputs/v6/independent/nav_w16_task_12k/table_3seed.json`.

## Matched 16k collision-aware baselines (new, 2026-10-27)

These two families close the fairness gap left by the archived rows: pure
MAPPO 16k used no collision penalty, while MAPPO-Lag was only available at
100k.  Both new families use the official DoubleIntegrator, 16,384 training
steps, seeds 0/1/2, the final checkpoint, and 100 deterministic episodes on
scenes `10000..10099`.  Neither uses h, `cbf.pkl`, QP, `u_ref`, or V5/V4
weights.

**MAPPO + fixed collision penalty 10 (task reward, no Lagrangian):**

| seed | safe_ep | collision_ep | official_safety | success | reach | return |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.000 | 1.000 | 0.3513 | 0.0187 | 0.0737 | -2.72 |
| 1 | 0.040 | 0.960 | 0.5837 | 0.0138 | 0.0238 | -43.05 |
| 2 | 0.060 | 0.940 | 0.6225 | 0.0100 | 0.0325 | -45.28 |
| **mean** | **0.0333** | **0.9667** | **0.5192** | **0.0142** | **0.0433** | **-30.35** |
| **std** | 0.0306 | 0.0306 | 0.1467 | 0.0044 | 0.0267 | 23.95 |

**MAPPO + fixed collision penalty 1 (2026-10-28):**

| seed | safe_ep | collision_ep | official_safety | success | reach | return |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.000 | 1.000 | 0.4588 | 0.1938 | 0.3812 | 83.10 |
| 1 | 0.000 | 1.000 | 0.2663 | 0.0800 | 0.3038 | 46.97 |
| 2 | 0.030 | 0.970 | 0.5037 | 0.1237 | 0.2188 | 48.03 |
| **mean** | **0.0100** | **0.9900** | **0.4096** | **0.1325** | **0.3013** | **59.36** |
| **std** | 0.0173 | 0.0173 | 0.1246 | 0.0577 | 0.0816 | 20.16 |

**MAPPO + fixed collision penalty 100 (2026-10-28):**

| seed | safe_ep | collision_ep | official_safety | success | reach | return |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.030 | 0.970 | 0.6287 | 0.0200 | 0.0263 | -48.93 |
| 1 | 0.100 | 0.900 | 0.6687 | 0.0163 | 0.0325 | -58.33 |
| 2 | 0.090 | 0.910 | 0.7237 | 0.0112 | 0.0175 | -71.17 |
| **mean** | **0.0733** | **0.9267** | **0.6737** | **0.0158** | **0.0254** | **-59.48** |
| **std** | 0.0379 | 0.0379 | 0.0477 | 0.0044 | 0.0076 | 11.25 |

**MAPPO-Lag + official collision cost (`cost_limit=0.01`):**

| seed | safe_ep | collision_ep | official_safety | success | reach | return |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.000 | 1.000 | 0.3075 | 0.0788 | 0.2587 | 43.24 |
| 1 | 0.000 | 1.000 | 0.4000 | 0.0550 | 0.1362 | 21.59 |
| 2 | 0.000 | 1.000 | 0.2575 | 0.1250 | 0.4238 | 53.83 |
| **mean** | **0.0000** | **1.0000** | **0.3217** | **0.0863** | **0.2729** | **39.55** |
| **std** | 0.0000 | 0.0000 | 0.0723 | 0.0356 | 0.1443 | 16.43 |

Summary against the promoted V6 row:

| arm | safe_ep | official_safety | success | reach | return |
|---|---:|---:|---:|---:|---:|
| MAPPO + fixed penalty 1, 16k | 0.0100 | 0.4096 | 0.1325 | 0.3013 | 59.36 |
| MAPPO + fixed penalty 10, 16k | 0.0333 | 0.5192 | 0.0142 | 0.0433 | -30.35 |
| MAPPO + fixed penalty 100, 16k | 0.0733 | 0.6737 | 0.0158 | 0.0254 | -59.48 |
| MAPPO-Lag, 16k | 0.0000 | 0.3217 | 0.0863 | 0.2729 | 39.55 |
| **V6 repaired Stage B, 16k** | **0.6933** | **0.9417** | **0.4100** | **0.4296** | **66.85** |

Reading:

- The fixed-penalty sweep confirms the trade-off at 16k.  Penalty `1`
  remains unsafe (`safe 0.01`, official `0.410`) but keeps some task
  (`success 0.133`, return `59.36`); penalty `10` suppresses task almost
  completely (`reach 0.043`, return `-30.35`) and is still unsafe
  (`safe 0.033`); penalty `100` shifts to `official 0.674`/`safe 0.073`
  but collapses task (`success 0.016`, return `-59.48`).  No fixed penalty
  reaches the safety/task region of V6.
- MAPPO-Lag moves more (`reach=0.273`) but never produces a collision-free
  episode at 16k.  Its learned multiplier is around `0.20-0.21` at the end of
  training.
- Neither matched baseline approaches V6's `safe_ep=0.693` /
  `official_safety=0.942` at the same 16k budget.  The older MAPPO-Lag 100k
  row remains a **generous external reference**, not the matched row.

Sources:

- `outputs/v6/independent/nav_baselines_16k/mappo_penalty1/aggregate_100ep_policy_final.json`
- `outputs/v6/independent/nav_baselines_16k/mappo_same_reward/aggregate_100ep_policy_final.json`
- `outputs/v6/independent/nav_baselines_16k/mappo_penalty100/aggregate_100ep_policy_final.json`
- `outputs/v6/independent/nav_baselines_16k/mappo_lag/aggregate_100ep_policy_final.json`

## Matched 16k / 100ep / seed-0 arms

| arm | safe_ep | collision_ep | official_safety | success | reach | return | success retention |
|---|---:|---:|---:|---:|---:|---:|---:|
| h-only (matched seed 0) | 0.620 | 0.380 | 0.9287 | **0.5325** | **0.5763** | **69.78** | 100.0% |
| broad positive-reward gate | 0.700 | 0.300 | 0.9400 | 0.3725 | 0.3913 | 64.96 | 69.9% |
| Stage A progress-only gate | 0.530 | 0.470 | 0.9113 | 0.5075 | 0.5475 | 76.85 | 95.3% |
| old Stage B (bad future-min h) | 0.000 | 1.000 | 0.4400 | 0.0375 | 0.0700 | -23.60 | 7.0% |
| old Stage B, gate off (diagnostic) | 0.030 | 0.970 | 0.4850 | 0.0763 | 0.1350 | 7.34 | 14.3% |
| **repaired Stage B seed 0** | **0.840** | **0.160** | **0.9738** | 0.4275 | 0.4350 | 60.91 | **80.3%** |

These are seed-0 ablations; only the repaired Stage-B arm has a formal
3-seed result.  Do not compare their single-seed retention percentages to the
3-seed mean as if they were equally sampled.

## Other references (not matched)

| arm | budget / episodes | safe_ep | official_safety | success | reach | return |
|---|---|---:|---:|---:|---:|---:|
| h-off pilot | 8k / 20ep | 0.050 | 0.5188 | 0.0188 | 0.0500 | -31.73 |
| barrier fine-tune | 24.5k / 100ep | 0.730 | 0.9500 | 0.31875 | 0.32625 | 65.77 |
| barrier screen | 16k / 20ep | 1.000 | 1.0000 | 0.3375 | 0.3375 | 68.70 |
| CBF-loss screen | 16k / 20ep | 0.650 | 0.9438 | 0.4688 | 0.4813 | 63.75 |
| barrier + both screen | 24.5k / 20ep | 0.850 | 0.9688 | 0.1063 | 0.1063 | 23.12 |

The h-off pilot is retained only as a historical reference; the formal 16k
h-off 3-seed cell is P1's D (`safe 0.0333`, `official_safety 0.5479`,
`success 0.0367`, `return 7.21`), which confirms the no-h collapse at the
matched budget.  The barrier fine-tune uses a 24,576-step budget, so it is a
reference, not a matched baseline.

## External CBF-family and Safe-MARL reference

These are archived V5 results, read-only.  They use the same official
DoubleIntegrator, the same 100 fixed scenes (`10000..10099`), deterministic
actors, and the same metric definitions, but they are **not rerun** and are
**not V6 main-table baselines**: V5 uses the frozen official h and a safety
fine-tune from 8k; GCBF+ is a pretrained actor; CBF-QP is a model-based
controller with an online QP.  V6 is from scratch with V6-owned h and no QP.
Compare as a reference, not as a controlled experiment.

| method | family | budget / seeds | QP? | official safety | safe_ep | success | reach | return |
|---|---|---|---:|---:|---:|---:|---:|---:|
| GCBF+ official | CBF / learned | pretrained step 1000 | no (actor-only) | 0.9988 | 0.990 | 0.8613 | 0.8625 | 480.05 |
| centralized CBF-QP | CBF-QP | model-based | yes (OSQP) | 0.9788 | 0.900 | 0.7813 | 0.7888 | 406.82 |
| V5 full | CBF-informed Safe MARL | 16k, 3 seeds, from 8k | no | 0.9500 ± 0.0308 | 0.7467 ± 0.1159 | 0.5121 ± 0.1352 | 0.5363 ± 0.1490 | 81.13 ± 6.00 |
| V5 h-input | ablation | 16k, 3 seeds | no | 0.9533 ± 0.0402 | 0.7633 ± 0.1872 | 0.2779 ± 0.0567 | 0.2875 ± 0.0579 | 47.35 ± 9.61 |
| V5 no-h | ablation | 16k, 3 seeds | no | 0.4179 ± 0.0657 | 0.0133 ± 0.0058 | 0.0475 ± 0.0142 | 0.1125 ± 0.0344 | 9.98 ± 14.09 |
| pure MAPPO 16k | task-only MARL | 16k, 3 seeds | no | 0.3079 ± 0.0995 | 0.0033 ± 0.0058 | 0.0571 ± 0.0281 | 0.1713 ± 0.0435 | 32.17 ± 4.80 |
| MAPPO-Lag 100k | constrained MARL | 100k, 3 seeds | no | 0.4325 ± 0.0379 | 0.0033 ± 0.0058 | 0.1363 ± 0.0508 | 0.3088 ± 0.0928 | 50.36 ± 17.85 |
| **V6 repaired Stage B** | predictive h + gate | 16k, 3 seeds | no | 0.9417 ± 0.0378 | 0.6933 ± 0.1680 | 0.4100 ± 0.0583 | 0.4296 ± 0.0695 | 66.85 ± 5.26 |

`return` is the task return `20 * progress + 5 * reached`, not the native
GCBF+ control reward.  The `MAPPO-Lag 100k` row is a **generous external
reference**, not a compute-matched row; the matched 16k MAPPO-Lag result is in
the new section above.

Coverage note: GCBF v0, decentralized CBF-QP, a distinct HandCBF
implementation, and MACPO are not available in this workspace.  Do not invent
rows for them.  The centralized CBF-QP row is the hand-crafted CBF-QP
reference; it uses the official formulation with OSQP instead of the
unavailable `jaxproxqp`.

Matched seed-0 reference for the two V5 learned-policy arms:

| method | official safety | safe_ep | success | reach | return |
|---|---:|---:|---:|---:|---:|
| V5 full seed 0 | 0.9788 | 0.870 | 0.5525 | 0.5650 | 85.84 |
| V5 h-input seed 0 | 0.9825 | 0.900 | 0.3113 | 0.3125 | 55.44 |
| V6 repaired seed 0 | 0.9738 | 0.840 | 0.4275 | 0.4350 | 60.91 |

Positioning:

- GCBF+ is the strongest reference on both axes, but it is an official
  pretrained teacher with actor-only inference; V6 cannot copy its weights or
  h.  CBF-QP is also far ahead on task, with an online QP and full-state
  information.
- V6 3-seed safety `0.942` is close to CBF-QP `0.979` and V5 full 3-seed
  `0.950`, but V6 success `0.410` is below V5 full `0.512` and far below
  CBF-QP `0.781` / GCBF+ `0.861`.  V6 is not a replacement for the full
  Safe-MARL or CBF-QP baselines on task.
- V6 beats V5 h-input on task and return with slightly lower safety: success
  `0.410` vs `0.278` (`+13.2 pp`), reach `0.430` vs `0.288`, return `66.85`
  vs `47.35` (`+41%`), while safe episodes are `0.693` vs `0.763` and
  official safety `0.942` vs `0.953`.
- V6 dominates the unsafe MARL baselines: vs V5 no-h, pure MAPPO, and
  MAPPO-Lag it is higher on safety, safe episodes, success, reach, and return.
- V5 full is the strongest learned Safe-MARL reference: `0.950/0.747/0.512/
  81.13` over 3 seeds.  V6 loses `5.3 pp` safe episodes, `10.2 pp` success,
  and `17.6%` return while being within `0.8 pp` official safety.  V5 full
  uses official h and a warm start from 8k, so this is a reference gap, not an
  equal-information comparison.

## Reading

1. **Safety is the clearest 3-seed result.**  Mean `safe_episode` rises from
   `0.450` to `0.693` (`+24.3 pp`) and official safety from `0.878` to `0.942`
   (`+6.3 pp`).  All three seed-level safety deltas are non-negative
   (`+0.22/+0.02/+0.49`), but seed 1 is nearly neutral and the gain is carried
   by seeds 0/2.  With `n=3`, the paired seed-level difference is not
   statistically separated from zero (`p=0.216`); the pre-registered
   aggregate rule passes, but the claim is not "policy-level proven".
2. **Task does not collapse on average.**  Mean success is `0.410` vs h-only
   `0.407` (`+0.3 pp`) and return is `+8.3%`; per-seed success shifts are
   `-10.5/+4.0/+7.5 pp`, so retention is heterogeneous rather than uniform.
   The 3-seed success standard deviation also falls from `0.132` to `0.058`.
3. **It is stronger than the earlier gate designs at seed 0.**  Repaired
   seed 0 has `safe_episode=0.840` vs broad gate `0.700` and Stage A `0.530`,
   with success `0.4275` vs broad `0.3725`; Stage A retains more task at seed 0
   (`0.5075`) but loses safety.  Old Stage B and its gate-off control remain
   catastrophic and confirm that the old h features were harmful.
4. **It compares well with the barrier-loss alternative.**  At 16k seed 0 the
   repaired arm has higher safety and success than the 24.5k barrier
   fine-tune, with lower return.  The 20ep barrier screen reaches
   `safe_episode=1.0` but at `success=0.3375`; the repaired 16k 20ep monitor
   was `safe_episode=0.90`, `success=0.4875`, return `64.74` (optimistic vs
   the final 100ep).
5. **Recorded V5 safety target.**  Seed 0 meets the archived target of
   environment safety `>=0.95` and safe episode `>=0.70` (`0.9738/0.840`).
   The 3-seed mean falls just short (`0.9417/0.693`).  The target does not
   specify a numeric success floor, so this is a safety-target check, not an
   official-h reproduction.

## Claim boundary

- The repaired Stage-B arm passes the pre-registered 3-seed guardrails; this
  is a 3-seed navigation gate result.
- `n=3` only.  Seed-level paired differences are not statistically separated
  from zero; do not claim policy-level superiority over h-only.
- Do not claim superiority over V5 full, CBF-QP, or GCBF+ on task; V6 is
  still `10.2 pp` below V5 full and `37.1 pp` below CBF-QP success.
- Do not sweep `task_gate_high` to close the remaining gap.
- The formal 16k h-off 3-seed baseline is complete (P1 D: `safe 0.0333`,
  `official_safety 0.5479`, `success 0.0367`, `return 7.21`).  It collapses,
  so h-off is not a competitive arm at this budget.
- No action-conditioned filter/QP was used or evaluated here.
