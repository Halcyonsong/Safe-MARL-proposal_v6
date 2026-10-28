# Pending navigation 28k experiments

Status: **planning only — no 28k navigation run has been launched.**
Created 2026-10-27.  The Ant 10M MAPPO-Lag external baseline is running;
navigation 28k work starts only after it finishes and after this plan is
frozen.

Canonical status: [`../../STATUS.md`](../../STATUS.md).
16k factorial: [`P1_REPAIRED_H_FACTORIAL.md`](P1_REPAIRED_H_FACTORIAL.md).
Baseline table: [`BASELINE_COMPARISON.md`](BASELINE_COMPARISON.md).

## 1. Questions this plan must answer

1. Does the promoted V6 navigation package (repaired h + progress gate)
   improve safety or reduce seed variance when trained from scratch for
   28,672 steps instead of 16,384?
2. If it improves, is the gain a pure budget effect or does it change the
   h/gate attribution established at 16k?
3. Which baselines must be budget-matched at 28k, and which can remain
   non-matched references?

## 2. Fixed 16k facts (do not rerun)

All rows are 3 seeds, 100 deterministic episodes, eval seeds `10000..10099`,
final checkpoint.

| row | budget | safe_ep | official_safety | success | return |
|---|---:|---:|---:|---:|---:|
| H0 h-only (original h0, no gate; actor h input ON) | 16k | 0.4500 | 0.8783 | 0.4067 | 61.73 |
| A repaired h + progress gate | 16k | 0.6933 | 0.9417 | 0.4100 | 66.85 |
| B repaired h, gate off | 16k | 0.6333 | 0.9267 | 0.3229 | 55.67 |
| C no actor h, gate on | 16k | 0.0200 | 0.4987 | 0.0367 | -6.63 |
| D no actor h, gate off | 16k | 0.0333 | 0.5479 | 0.0367 | 7.21 |
| MAPPO + fixed penalty 10 | 16k | 0.0333 | 0.5192 | 0.0142 | -30.35 |
| MAPPO-Lag + official collision cost | 16k | 0.0000 | 0.3217 | 0.0863 | 39.55 |
| MAPPO-Lag 100k (archived) | 100k | 0.0033 | 0.4325 | 0.1363 | 50.36 |
| V5 full (archived) | 16k from 8k | 0.7467 | 0.9500 | 0.5121 | 81.13 |
| GCBF+ official (pretrained) | step 1000 | 0.9900 | 0.9988 | 0.8613 | 480.05 |
| centralized CBF-QP (model-based) | no training | 0.9000 | 0.9788 | 0.7813 | 406.82 |

The 16k P1 2x2 is the mechanism result: actor h input is load-bearing, the
gate cannot substitute for it, and A is the only cell combining safety with
retained task.  Do not overwrite that claim with a 28k run.

Naming note: **H0 is not h-off.**  H0 is the original-h h-only policy
baseline: the `v6_h.pkl` checkpoint was trained offline on 100 random-policy
episodes plus augmentation, then frozen; the actor receives h/grad features
and is trained from scratch with no gate.  The h-off cells are C/D in the
repaired-h 2x2.  "Original h0" refers to the h checkpoint's random-policy
training data, not to a random behavior policy at evaluation.

## 3. Experiment matrix

Priority: **P0 = required for a budget-matched 28k claim; P1 = needed for
gate or external attribution at 28k; P2 = only if 28k becomes the main table;
P3 = not planned.**

| ID | arm | budget | seeds | purpose | priority | status |
|---|---|---|---:|---|---|---|
| N28-A | V6 repaired h + progress gate | 28,672 | 0/1/2 | promoted package budget curve | **P0** | approved, waiting for Ant |
| N28-H0 | h-only (original h0, no gate; not h-off) | 28,672 | 0/1/2 | key budget-matched baseline | **P0** | proposed |
| N28-H0G | h0 + **progress gate** (new cell, not P0) | 28,672 | 0/1/2 | h quality with gate on; clean h0 x gate interaction | P1/P2 | optional |
| N28-B | repaired h, gate off | 28,672 | 0/1/2 | gate effect A-B at 28k | **P1** | proposed |
| N28-C | no actor h, gate on | 28,672 | 0/1/2 | h necessity at 28k | P3 | not planned by default |
| N28-D | no actor h, gate off | 28,672 | 0/1/2 | h necessity at 28k | P3 | not planned by default |
| N28-MAPPO | MAPPO + fixed penalty 10 | 28,672 | 0/1/2 | collision-aware task baseline | P1/P2 | optional |
| N28-MAPPO-Lag | MAPPO-Lag + official collision cost | 28,672 | 0/1/2 | collision-aware constrained baseline | P1/P2 | optional |
| N28-MAPPO-Lag-100k | archived MAPPO-Lag 100k | 100k | 0/1/2 | generous external reference | — | no rerun |
| N28-GCBF+ | GCBF+ official | pretrained | — | external CBF-family reference | — | no rerun |
| N28-CBF-QP | centralized CBF-QP | model-based | — | model-based safety reference | — | no rerun |
| N28-V5 | V5 full/h-input/no-h | 16k from 8k | 0/1/2 | archived Safe-MARL reference | — | no rerun |

### Minimal defensible 28k suite

If 28k is a **budget-sensitivity / robustness** result:

- N28-A only.  Compare A@28k to A@16k; no baseline retraining is needed for
  the budget statement.
- Keep the existing 16k matched baseline table unchanged.

If 28k is to support a **budget-matched safety claim**:

- N28-A + N28-H0.  Without H0@28k, A@28k vs H0@16k is confounded by budget.
- Add N28-B if the gate effect at 28k is part of the claim.
- N28-C/D are not required unless the claim explicitly says "h input is
  still necessary at 28k"; the 16k collapse already establishes the mechanism
  at the promoted budget.

If 28k needs a **clean h0/h1 x gate-on/off attribution**:

- N28-A + N28-B + N28-H0 + N28-H0G.  A and H0G use the same
  progress-only gate; H0 and B use gate off.  This gives h quality at gate
  off (B-H0), h quality at gate on (A-H0G), gate effect at h0 (H0G-H0), gate
  effect at h1 (A-B), and the interaction.
- Do not use P0 (original h0 + **broad** gate) as the h0+gate cell: its gate
  scope/threshold differs from A, so A-P0 is not a clean h-quality contrast.

If 28k is to **replace the 16k main table**:

- N28-A + N28-H0 + N28-B.
- N28-MAPPO and N28-MAPPO-Lag for the external rows, or explicitly demote
  their 16k rows to non-matched references.
- Do not promote 28k as a main table until these rows exist.

## 4. h-on/off and gate-on/off at 28k

The 16k 2x2 is sufficient for the **16k mechanism claim**.  It is not
automatically sufficient for a 28k mechanism claim.

| cell | 16k result | 28k needed? | reason |
|---|---|---|---|
| A h1 on, gate on | 0.6933 safe | **yes** | promoted package budget curve |
| B h1 on, gate off | 0.6333 safe | **P1** | needed only for gate effect A-B at 28k |
| H0 h0 on, gate off | 0.4500 safe | **P0** | budget-matched h-only baseline; B-H0 is h quality with gate off |
| H0G h0 on, **progress gate on** | not run | **P1/P2** | new cell; A-H0G is h quality with gate on; H0G-H0 is gate effect at h0 |
| P0 h0 on, **broad gate on** | 0.6433 safe | optional | different gate scope; P0-H0 is broad-gate effect, not h quality |
| C h1 checkpoint but actor h input off, gate on | 0.0200 safe | **no by default** | 16k collapse is already decisive; 28k only if claiming h necessity at 28k |
| D h1 checkpoint but actor h input off, gate off | 0.0333 safe | **no by default** | same |

If N28-A shows an unexpected safety/task regime change, revisit C/D.  Do not
pre-emptively spend 6 runs on cells that already collapsed at 16k.

## 5. Baseline-specific decisions

### GCBF+

GCBF+ official is a **pretrained external actor** (step 1000) from a different
CBF/learned family.  It is not a from-scratch V6 baseline and does not share
the V6 training pipeline.  Retraining it for 28k would require reproducing
its own training recipe and would no longer be the archived official
reference.  **Do not run a 28k GCBF+ row.**  Report it as non-matched.

If a reviewer demands a compute-matched CBF-family row, that is a separate
reproduction project, not a V6 baseline run.

### Centralized CBF-QP

CBF-QP is a **model-based online controller** with a hand-designed CBF and an
OSQP solve at each step.  It has no training budget, so a 28k row is
meaningless.  **Do not run a 28k CBF-QP row.**  Report it as a model-based
reference with no learning budget.

### MAPPO budgets

- Pure MAPPO task-only: 16,384 steps, 3 seeds (archived V5 comparison row).
- MAPPO + fixed collision penalty 10: 16,384 steps, 3 seeds
  (`nav_baselines_16k/mappo_same_reward`).
- MAPPO-Lag + official collision cost: 16,384 steps, 3 seeds
  (`nav_baselines_16k/mappo_lag`).
- MAPPO-Lag archived external reference: 100,000 steps, 3 seeds.

### MAPPO + fixed collision penalty 10

The 16k row already exists (`nav_baselines_16k/mappo_same_reward`).  Its 16k
result is very weak (`safe 0.033`, `success 0.014`, `return -30.35`).

- If it remains in the 28k headline table: run N28-MAPPO, 3 seeds.
- If 28k is only a within-V6 budget study: keep the 16k row as a 16k
  reference and do not compare it to A@28k as equal budget.
- Do not use the 100k MAPPO-Lag row as a substitute for MAPPO.

### MAPPO-Lag + official collision cost

The 16k row already exists (`nav_baselines_16k/mappo_lag`): `safe 0.000`,
`official safety 0.3217`, `success 0.0863`, `return 39.55`.

- If it remains in the 28k headline table: run N28-MAPPO-Lag, 3 seeds.
- Otherwise use the archived 100k MAPPO-Lag 3-seed row as a **generous
  external reference** (more budget than 28k), clearly labelled non-matched.
- Never compare the 16k MAPPO-Lag row to A@28k as equal budget.

### MAPPO-Lag 100k

Archived, 3 seeds, 100k steps.  It can be used **directly** as a
longer-budget external reference: 100k > 28k, so if V6 A@28k matches or beats
it, that is a conservative comparison.  It is **not** a same-budget 28k row;
label it "generous longer-budget reference".  Do not retrain at 28k unless a
strict 28k external table is required.  If its own curve is still improving
at 100k, do not call it "converged" either; call it "longer-budget reference".

### Fixed-penalty sweep for MAPPO (new, 16k fairness check)

The single `collision_penalty=10` row is not enough to rule out an
under-tuned baseline.  Add a small log-scale sweep at the matched 16k budget:

- penalties: `1`, `10`, `100`; `0` (pure MAPPO) and `10` already exist, so
  only `1` and `100` are new runs; `0` and `10` are the endpoints/reference;
- 16,384 steps, seeds 0/1/2, 100 deterministic episodes on
  `10000..10099`, fixed final checkpoint;
- same actor/critic/reward as `mappo_double_integrator_same_reward.yaml`;
  only `training.collision_penalty` changes;
- output: `outputs/v6/independent/nav_baselines_16k/mappo_penalty{p}/`;
- report the full safety/task Pareto frontier, every seed, mean/std/range;
- if the best penalty still cannot produce a competitive `safe_episode`, the
  baseline is robustly weak; if a tuned penalty closes the gap, do not claim
  a V6 advantage at 16k.

Optional companion: MAPPO-Lag `cost_limit` sweep
`{0.005, 0.01, 0.05}` under the same 16k/3-seed/100ep protocol
(`mappo_lag_cl{...}`).  This is P1, not a blocker.  **Scope decision
2026-10-27: defer this sweep; this round runs the fixed-penalty arms only.**
No cost-limit configs are created yet.

Implementation status (2026-10-27): explicit config variants are added, so
the existing `same_reward` config is untouched and each penalty has its own
hash:

| variant | config | sha256 |
|---|---|---|
| penalty 1 | `nav/configs/baselines/mappo_double_integrator_penalty1.yaml` | `6424e094b6ad5d9f055450b79f7ce017b1d15e328cd0513b41d40ed6a9d5189c` |
| penalty 100 | `nav/configs/baselines/mappo_double_integrator_penalty100.yaml` | `1d98e292815ca8c8034000e3bbb8a27059336a8cfd32c433aecaa8be80f96074` |

Only `seed`, comments, and `training.collision_penalty` differ from
`mappo_double_integrator_same_reward.yaml`; the 16k budget and all other
hyperparameters are unchanged.  `run_mappo_baseline.sh` and
`evaluate_mappo_baseline.sh` now accept `penalty1` / `penalty100`.  Planned
commands, run from `nav/`:

```bash
SEEDS="0 1 2" STEPS=16384 VARIANTS="penalty1 penalty100" \
  OUT_ROOT=../outputs/v6/independent/nav_baselines_16k \
  bash scripts/run_mappo_baseline.sh
SEEDS="0 1 2" VARIANTS="penalty1 penalty100" \
  OUT_ROOT=../outputs/v6/independent/nav_baselines_16k \
  bash scripts/evaluate_mappo_baseline.sh
```

Not launched as of 2026-10-27 20:50 CST: the 28k training/evaluation is
using the machine.  The optional MAPPO-Lag `cost_limit` sweep is still a
separate scope decision; no cost-limit configs are added yet.

Do not rerun the sweep at 28k by default.  If 28k becomes the main table,
rerun only the best penalty/cost-limit setting at 28,672 with 3 seeds.

### V5 / CBF-QP / GCBF+

Archived, different budgets/methods, and some use official h/QP.  Keep as
read-only references.  Do not rerun inside V6.

### broad gate / Stage A / old Stage B

Historical gate ablations.  Do not extend to 28k unless a separate gate-budget
question is approved.

## 5A. V6 density stress test (evaluation-only)

Purpose: test whether the promoted A package and its 28k extension hold up
outside the fixed 8-agent / 8-obstacle / 4x4 training distribution.  This is
an evaluation-only test; do not retrain h or the actor for the density
conditions.

Conditions (one factor at a time; default 8 agents / 8 obstacles / area 4.0):

| axis | levels | note |
|---|---|---|
| obstacles | 4, 8, 12, 16 | 8 is the default/shared row |
| agents | 8, 12, 16 | `max_neighbors=5`; agent count must be >= 5 |
| optional combined | 12 agents + 12 obstacles | only if the one-factor curves are informative |

**Scope decision 2026-10-27:** run the **4 key conditions first** — default
8/8, 4 obstacles, 16 obstacles, and 16 agents — as a seed-0, 20-episode
screen across the three policies.  Confirm only the endpoints that show a
non-trivial effect with 100 episodes x 3 seeds.  The 12-agent and 12/12
combined rows are deferred follow-ups, not part of the first block.

Policies:

- A@28k final (seeds 0/1/2);
- A@16k checkpoint from the same 28k run (paired budget control);
- H0@16k final (seeds 0/1/2), the promoted h-only baseline;
- add H0@28k / B@28k if those rows are approved.

Protocol:

- 100 deterministic episodes per condition, held-out scenes
  `30000..30099`; also report the default 8/8 condition on
  `10000..10099` to connect to the main table;
- fixed final checkpoint; no best-checkpoint or seed selection;
- report per-condition mean/std/range and every seed for `safe_episode`,
  `official_safety`, `official_success`, `official_reach`,
  `episode_return`;
- add per-agent collision rate if the evaluator exposes it.

Interpretation guardrail: `safe_episode` is an all-agent metric, so it
mechanically becomes harder as the agent count grows.  Compare methods
**within** a condition; do not read a drop from 8 to 16 agents as a safety
regression.  Use `official_safety` and collision rate for
density-normalized comparisons.

Implementation constraints:

- `evaluate_v6.py` now supports `--num-agents`, `--num-obstacles`,
  `--area-size`, and `--max-step`; the resolved env and the explicit
  overrides are written to the output JSON.  `max_neighbors` / `n_rays` stay
  fixed: they define the 92-dim observation and must match the checkpoint.
- The JAX h provider now accepts an evaluation-time `n_agents` override so
  the same permutation-invariant full-graph h can be applied to 12/16 agents;
  the default path (8 agents) is unchanged.
- The chunked evaluator passes optional `NUM_AGENTS`, `NUM_OBSTACLES`,
  `AREA_SIZE`, and `MAX_STEP` environment variables to the evaluator.
- `max_neighbors=5` includes self; with fewer than 5 agents the neighbor
  tensor is shorter than 92 dims and the policy will fail.  Use >= 5 agents,
  or add zero-padding to the environment first.  The evaluator now refuses
  `num_agents < max_neighbors` explicitly.
- **Cost warning (measured 2026-10-27):** the h Jacobian cost grows with the
  agent count and the evaluator is CPU-bound.  A 12-agent cold 1-episode run
  took 11.5 min while three 28k trainings were active; a 5-episode run
  exceeded 40 min.  A 16-agent cold run did not finish within 10 min under
  the same contention.  Do **not** run the density block concurrently with
  training.  Re-benchmark after the GPU/CPU is free; if the cost persists,
  run a 1-seed, 20-episode screen at intermediate densities and reserve
  100-episode, 3-seed evaluation for the endpoints.
- Keep the density test out of the main 16k/28k table; report it as a
  robustness/boundary section.

Decision: if A@28k's advantage over A@16k/H0 grows with density, report a
robustness benefit.  If it shrinks or reverses, report the boundary and do
not claim density generalization.

## 6. Evaluation protocol

For every new 28k row:

- from scratch, same frozen h/gate/cost/reward as the corresponding 16k arm;
- `total_steps=28672`, `save_interval=4096`;
- save checkpoints at 16,384 / 20,480 / 24,576 / 28,672;
- 100 deterministic episodes, eval seeds `10000..10099`, final checkpoint;
- no `policy_best` selection; no seed selection; no checkpoint selection;
- report mean, std, range, and every seed;
- task guardrails: mean `success >= 0.36`, mean `return >= 55`, no seed below
  the existing catastrophic thresholds;
- the 16k checkpoint from the same from-scratch run is a reproducibility
  check against the promoted 16k row, not a replacement for it.

For the budget curve, evaluate 20,480 / 24,576 / 28,672 and report all three.
The primary endpoint is the fixed 28,672 final checkpoint.  If the curve
peaks earlier, do not switch the primary post hoc; propose a separate
confirmatory run.

## 7. Decision gates

1. **After N28-A:** apply the pre-registered rule in section 7.1.  Does the
   3-seed mean improve or at least not degrade on `safe_episode`,
   `official_safety`, `success`, and `return`?  Does the observed
   between-seed range shrink below `0.20`, and does the worst seed rise above
   `0.55`?
2. **After N28-H0:** is A@28k still better than the budget-matched h-only
   baseline?  If not, no 28k safety claim over h-only.
3. **After N28-B:** does the gate still add safety/task at 28k?  If the
   A-B contrast disappears, report a package effect only, not a gate effect.
4. **Before promoting 28k:** decide whether the external 28k rows
   (MAPPO/MAPPO-Lag) are required.  If not, keep 28k as a secondary
   budget-sensitivity result.  The 16k penalty sweep is the fairness check
   for the 16k table; rerun only the best setting at 28k if 28k becomes the
   main table.
5. **After the density stress test:** does A@28k's advantage over A@16k/H0
   grow, hold, or shrink with density?  Report a robustness benefit only if
   it holds; otherwise report the boundary.
6. **C/D 28k:** only if N28-A shows an unexpected h-off recovery or a
   reviewer requires a 28k h-necessity claim.

### 7.1 Variance decision rule (pre-register before launch)

Baseline 16k A: `safe_episode 0.6933 ± 0.1680`, per-seed
`0.84/0.51/0.73`, range `0.33`, worst seed `0.51`; task
`success 0.4100`, `return 66.85`.

For the 28k final (3 seeds), apply these rules in order:

1. **Mean not degraded:** mean `safe_episode >= 0.6933` and mean
   `official_safety >= 0.9417`.
2. **Task guardrails:** mean `official_success >= 0.36`, mean
   `episode_return >= 55`, worst seed `official_success >= 0.20`,
   worst seed `safe_episode >= 0.30`.
3. **Observed spread narrowed:** between-seed range
   `max - min <= 0.20` (baseline `0.33`) and worst seed
   `safe_episode >= 0.55` (baseline `0.51`).
4. **Paired direction:** compare 28k vs the 16k checkpoint from the same
   from-scratch run, per seed.  Report the three deltas; do not use only the
   mean.

Outcomes:

- **1-3 pass:** report "28k maintained/improved mean safety and narrowed the
  observed between-seed spread."  This is a descriptive budget-sensitivity
  claim, not a statistical proof of variance reduction.
- **1-2 pass, 3 fails:** report "higher/equal mean, similar or high
  variance."  Add seeds (total 5) for **both A and H0** if H0@28k is in
  scope (and B/H0G if those rows are run) before any formal variance claim.
  Do not add seeds to A alone.  If H0@28k is not run, the variance claim is
  A-only and cannot be budget-matched to h-only.
- **1 or 2 fails:** report a negative or task-costly budget result; do not
  add seeds to rescue it.  Keep 16k as the promoted row.

When adding seeds, prefer using the **same 28k runs' 16k checkpoints** as the
5-seed 16k control instead of training separate 16k seeds.  This keeps the
A@28k vs A@16k variance comparison paired and avoids an extra 16k batch.

Statistical caveat: with `n=3`, the sample std/range is very unstable.  A
small observed range is only *consistent with* reduced variance, not proof;
a large observed range is stronger evidence of heterogeneity.  A formal
"variance reduced" claim needs at least 5 seeds, preferably 8-10.  At `n=3`,
always report every seed and the range, not just mean ± std.

If the 28k curve peaks before 28,672, keep the 28,672 final as the primary
endpoint and report the curve; do not switch the primary post hoc.

## 8. Compute estimate

Prior 16k 3-seed navigation training took about 56 minutes, but most of that
was the 20-episode monitor evaluations at 8k/16k, not the PPO updates.  The
28k A run therefore includes monitor evaluations at 8k/16k/24k and is expected
to take roughly 2-3 hours, plus about 1 hour for the six formal 100-episode
evaluations (final + paired 16k, 3 seeds).  The 28k run is currently using the
GPU.

Additional evaluation/training blocks:

- MAPPO fixed-penalty sweep: 6 new 16k CPU runs (`penalty=1` and `100`, 3
  seeds each; `penalty=10` already exists), plus the optional 6 new MAPPO-Lag
  cost-limit runs (`0.005` and `0.05`; `0.01` already exists).  These are
  CPU-only; run them after the 3-seed 28k table and the task warm-start
  feasibility block, not alongside the three concurrent JAX/PPO trainers.
- Density stress test: the full 6 conditions x 3 policies x 3 seeds x 100
  episodes is **not** one cheap overnight block.  The h Jacobian is CPU-bound
  and grows with the agent count; a 12-agent cold 1-episode run took 11.5 min
  under 28k-training contention, and a 5-episode run exceeded 40 min.  Plan a
  1-seed, 20-episode screen at the intermediate densities first, then confirm
  the endpoints with 100 episodes and all 3 seeds only if the screen is
  informative.  Re-benchmark after the machine is free.
- If the variance rule fails: 2 extra 28k seeds for A and H0 (4 runs total)
  plus 100ep evaluation, roughly another 1.5-2 hours.

The Ant 10M baseline finished on 2026-10-27 19:17 and its 200-episode screen
is recorded.  **Queue order was changed on 2026-10-27** to put the task
question ahead of the baseline/density blocks:

1. **Completed:** the **3-seed 100ep table** for A@16k/24k/28k.  Means:
   16k `0.6933/0.9417/0.4100/66.85`; 24k
   `0.7000/0.9479/0.2704/59.97`; 28k
   `0.7800/0.9604/0.2767/63.45` (safe/official/success/return).  28k fails
   mean success (`0.2767<0.36`) and worst-seed success
   (`0.19875<0.20`); 16k remains promoted.  The seed 1/2 16k rows were
   reused from `nav_stageb_repair_16k` (checkpoint tensors and EMA
   bit-identical to the 28k run's 16k checkpoints).  Table:
   `independent/nav/nav_3seed_16k_24k_28k_table.json`.
2. **Task warm-start feasibility**: one W16 arm (16k -> 28k, 12,288 task
   steps), evaluate at **24k and 28k**; compare each to the from-scratch
   24k/28k checkpoints (same seed and 16k prefix, same total budget).  No
   separate C16/C24 control is required for this procedure-level
   comparison; the T0w 16k+4k no-V_T control remains the channel-attribution
   evidence.  Add a short 24k -> 28k arm only if the 16k -> 28k gain fades at
   28k.  Seed-0 screen first; 3 seeds if the task gain holds with safety not
   worse.  Protocol and thresholds are frozen in
   [`STAGE_W16_TASK_WARMSTART.md`](STAGE_W16_TASK_WARMSTART.md).  Seed 0
   passed on 2026-10-28: 24k `0.93/0.98875/0.57/152.42`; 28k
   `0.83/0.9725/0.62125/237.36` (safe/official/success/return), versus
   from-scratch 28k `0.86/0.975/0.19875/55.52`.  **3-seed result:** W16 28k
   mean `safe 0.6633`, `official 0.9329`, `success 0.6175`, `return
   198.57`; the 24k diagnostic mean is `0.7233/0.9504/0.5958/143.73`.
   Rule 7.1 fails on 28k mean safe/official and safe range (`0.27`; 24k
   range `0.38`), while task/return/worst-seed bounds pass.  **Task-first
   trade-off, not promoted; keep 16k promoted.**  No post-hoc 24k/28k
   switch.  MAPPO was preempted for this block and restarted after.
3. **Ant V_T failure diagnosis** (conditional): only if step 2 passes with a
   3-seed task gain and safety not worse.  If the nav task arm fails, skip
   this and keep the Ant null as a limitation.  New pre-registration; one
   cause at a time.  The separate 250k `[A_T, grad_s V_T]` interface
   diagnostic finished null on task (forward `3816.50` vs control `3811.20`,
   `+0.14%`; safety improved `0.99` vs `0.89`).  It is not a promoted row,
   does not replace this gate, and is not expanded to 3 seeds.  Record:
   `independent/ant/STAGE_ANT_VT_TRANSFER.md`.
4. **MAPPO fixed-penalty 16k sweep — completed 2026-10-28.**  Means:
   penalty 1 `official 0.4096`, `safe 0.0100`, `success 0.1325`, `return
   59.36`; penalty 100 `0.6737/0.0733/0.0158/-59.48`.  Neither approaches
   V6; penalty 1 remains unsafe, penalty 100 collapses task.  `penalty=0/10`
   already existed; MAPPO-Lag `cost_limit` remains deferred.
5. **Density 4-condition seed-0 20ep screen — completed 2026-10-28.**
   A16 keeps more task than A28 in every condition (`success 0.34-0.39` vs
   `0.19-0.22`) with comparable/better safety in 8/8, 4 obstacles and 16
   agents; A28 is safer only at 16 obstacles.  No A28 density-robustness
   benefit.  H0 has A16-like task but much lower safety at 16 obstacles/16
   agents.  Descriptive screen only; no 100ep x 3-seed confirmation.
   Records: `independent/nav/DENSITY_STRESS_TEST.md`,
   `independent/nav_density/seed0/summary.json`.

Do not run the task block, MAPPO, and density concurrently.  The 15 GB machine
OOM-killed one 28k trainer when three JAX/PPO trainers plus final evals were
resident; one heavy block at a time.  The Ant V_T failure is a conditional
follow-up (see `TODO_V6.md`), not part of this queue until the nav task arm
passes.

## 9. Explicitly not planned

- GCBF+ 28k retraining.
- CBF-QP 28k retraining (no training budget).
- V5 full/h-input/no-h 28k retraining.
- broad gate / Stage A / old Stage B 28k.
- C/D 28k by default.
- seed0-only 8k versus seeds1/2 4k.
- `policy_best` as the primary checkpoint.
- any gate-threshold sweep at 28k.
- new algorithm families (QMIX / MADDPG / MATD3 / CPO / MACPO) or new
  benchmarks (Safety-Gymnasium / MPE / SocialNav) in this round.
- 4-agent density without adding observation zero-padding; `max_neighbors=5`
  requires at least 5 agents in the current env.
- a test-time h-shield is a useful mechanism control but is not a 28k
  blocker; schedule it separately if the gate's training-time value is
  challenged.

## 10. Freeze checklist before launch

- [x] Decide whether 28k is budget-sensitivity only or a candidate main row:
      **budget-sensitivity only** for the A-only run.
- [x] Freeze N28-A; **N28-H0 / N28-B / external 28k rows are not approved**
      and are absent.  Do not make a budget-matched A-vs-H0 claim.
- [x] Freeze MAPPO penalties `{1, 10, 100}`; `penalty=0/10` already exist and
      only `1/100` are new.  MAPPO-Lag cost limits remain an optional scope
      decision.
- [x] Freeze the first density block: **4 key conditions** (8/8, 4
      obstacles, 16 obstacles, 16 agents), seed-0 20-episode screen, held-out
      scenes `30000..30019`; confirm endpoints later at 100 episodes x 3
      seeds.  12-agent and 12/12 combined rows are deferred.
- [x] Add and test `evaluate_v6.py` env overrides; record code hashes:
      `evaluate_v6.py 7cb07a13...`, `official_h.py ffdc838c...`,
      `eval_v6_nav_100ep_chunked.sh c9054801...`.
- [x] Write pre-registration with primary endpoint = 28,672 final:
      [`STAGE_28K_BUDGET_EXTENSION.md`](STAGE_28K_BUDGET_EXTENSION.md).
- [x] Freeze task guardrails and variance criteria (section 7.1).
- [x] Confirm H0@28k is **not** approved; the 28k claim is A-only.
- [x] Record h checkpoint (`0f67eae...`), config hash (`153e705...`), runner
      hash (`7189190a...`), and eval seeds `10000..10099`.
- [x] Confirm Ant 10M has finished (2026-10-27 19:17) and GPU is free.
- [x] Launch A from scratch; do not resume the 16k checkpoints for the
      primary 28k row.
