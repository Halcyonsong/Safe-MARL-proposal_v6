# V6 navigation density stress test (evaluation-only)

Status: **first block complete** (2026-10-28 06:41).  The 4 key conditions
were run as a seed-0, 20-episode screen across A28/A16/H0.  The 12-agent /
12-obstacle rows remain deferred.  The screen shows no A28 density-robustness
benefit; see the result table and decision below.

Canonical status: [`../../STATUS.md`](../../STATUS.md).
Planning matrix: [`PENDING_28K_EXPERIMENTS.md`](PENDING_28K_EXPERIMENTS.md).

## Purpose

Test whether the promoted A package (repaired h + progress gate) and its 28k
extension hold up outside the fixed 8-agent / 8-obstacle / 4x4 training
distribution.  This is an evaluation-only robustness test: do **not** retrain
h or the actor for any density condition, and do not mix these rows into the
main 16k/28k table.

## Conditions

Default: 8 agents, 8 obstacles, area 4.0, 256 steps.  Change one factor at a
time unless the combined row is explicitly approved.

**Frozen first block (4 key conditions):** default 8/8, 4 obstacles, 16
obstacles, 16 agents.  Seed 0, 20 episodes, scenes `30000..30019`, three
policies.  Confirm only the endpoints with a non-trivial effect at 100
episodes x 3 seeds on `30000..30099`.

Deferred follow-ups: 12 obstacles, 12 agents, and the 12/12 combined row.
The evaluator supports them, but they are not in the first block.

| axis | levels in first block | note |
|---|---|---|
| obstacles | 4, 8, 16 | 8/8 is the shared default row |
| agents | 8, 16 | `max_neighbors=5`; agent count must be >= 5 |
| optional combined | 12 agents + 12 obstacles | deferred until the one-factor curves are informative |

Held-out scenes: `30000..30099` for confirmatory runs (never used for training
or model selection); the screen uses `30000..30019`.  Also evaluate the
default 8/8 condition on the main-table scenes `10000..10099` to connect the
stress test to the promoted row.

## Policies

- A@28k final (seeds 0/1/2), from the approved 28k run;
- A@16k checkpoint from the **same** 28k run (paired budget control);
- H0@16k final (seeds 0/1/2), the promoted h-only baseline;
- add H0@28k / B@28k only if those rows are separately approved.

## Metrics and comparison rules

Report per condition, per policy, per seed: `safe_episode`,
`official_safety`, `official_success`, `official_reach`, `episode_return`,
and per-step collision rate if available.  Report mean/std/range and every
seed.

`safe_episode` is an all-agent metric, so it mechanically becomes harder as
the agent count grows.  Compare methods **within** a condition; do not read a
drop from 8 to 16 agents as a safety regression.  Use `official_safety` and
per-step collision rate for density-normalized comparisons.

## Implementation (implemented and smoke-tested 2026-10-27)

- `evaluate_v6.py` accepts `--num-agents`, `--num-obstacles`, `--area-size`,
  and `--max-step`; the resolved env and explicit overrides are written to
  the output JSON.
- `eval_v6_nav_100ep_chunked.sh` passes optional `NUM_AGENTS`,
  `NUM_OBSTACLES`, `AREA_SIZE`, and `MAX_STEP` environment variables.
- `official_h.provider_from_checkpoint` accepts an evaluation-time
  `n_agents` override, so the same permutation-invariant full-graph h is
  applied to 12/16 agents.  The 8-agent path is unchanged.
- `max_neighbors` and `n_rays` are never changed: they define the 92-dim
  observation and must match the checkpoint.
- The evaluator refuses `num_agents < max_neighbors` (4 agents would produce
  a truncated observation; zero-padding is not implemented).
- Code hashes at freeze: `evaluate_v6.py
  7cb07a137fa9108fda1de8a935ebbe07012198d86071f155893ef65fd150ef3a`,
  `official_h.py
  ffdc838c182adcdc815b0787fe9816bfa230e215a70116ae2add6c0847e3de09`,
  `eval_v6_nav_100ep_chunked.sh
  c9054801965f71ff9547d8aeecc74e64508bdd06593d9bfb174906823bbdc60b`.

Smoke checks: 12 agents / 12 obstacles completed a 1-episode run and wrote
the overrides correctly; 8 agents completed 1 episode.  A 16-agent cold run
did not finish within 10 min while three 28k trainers were active.  These are
load-path checks, not density results.

## Cost and tiering

The h Jacobian is CPU-bound and grows with the agent count.  Under 28k-training
contention, a 12-agent cold 1-episode run took 11.5 min and a 5-episode run
exceeded 40 min.  Do not launch the full grid concurrently with training.
Re-benchmark on a free machine; if the cost persists, use:

1. **Stage 1 screen:** seed 0, 20 episodes per condition, 4 key conditions
   (default 8/8; 4 obstacles; 16 obstacles; 16 agents) x 3 policies.
2. **Stage 2 confirm:** 100 episodes x 3 seeds for the default and the
   endpoint conditions that showed a non-trivial effect; do not promote a
   screen-only difference.

The full 6-condition x 3-policy x 3-seed x 100-episode grid is not assumed to
fit in one overnight block.

## Screen results (seed 0, 20 episodes, scenes 30000..30019)

| condition | policy | safe_ep | official_safety | success | reach | return |
|---|---|---:|---:|---:|---:|---:|
| 8x8 | A28 | 0.75 | 0.9500 | 0.2125 | 0.2188 | 60.11 |
| 8x8 | A16 | 0.75 | 0.9563 | 0.3563 | 0.3625 | 55.45 |
| 8x8 | H0 | 0.65 | 0.9313 | 0.3563 | 0.3875 | 70.00 |
| 4 obstacles | A28 | 0.70 | 0.9500 | 0.1875 | 0.2000 | 55.06 |
| 4 obstacles | A16 | 0.80 | 0.9688 | 0.3875 | 0.3937 | 54.50 |
| 4 obstacles | H0 | 0.60 | 0.9187 | 0.4125 | 0.4437 | 65.65 |
| 16 obstacles | A28 | 0.60 | 0.9000 | 0.1875 | 0.1938 | 51.48 |
| 16 obstacles | A16 | 0.50 | 0.9062 | 0.3438 | 0.3563 | 56.29 |
| 16 obstacles | H0 | 0.25 | 0.8438 | 0.3438 | 0.3750 | 64.85 |
| 16 agents | A28 | 0.30 | 0.9187 | 0.2188 | 0.2344 | 61.03 |
| 16 agents | A16 | 0.45 | 0.9219 | 0.3594 | 0.3781 | 61.52 |
| 16 agents | H0 | 0.05 | 0.8531 | 0.3094 | 0.3531 | 65.07 |

Reading: A16 keeps substantially more task than A28 in every density
condition (success `0.34-0.39` vs `0.19-0.22`) with comparable or better
all-agent safety in 8/8, 4 obstacles and 16 agents; A28 is safer only in the
16-obstacle condition (`0.60` vs `0.50`).  H0 has task comparable to A16 but
much lower safety at 16 obstacles/16 agents (`safe 0.25/0.05`).  There is no
evidence that A28's 16k->28k safety advantage grows with density; the screen
shows a task boundary, not a robustness benefit.  These are descriptive
20-episode screens; confirmatory 100ep x 3-seed endpoints were not run.
Summary: `outputs/v6/independent/nav_density/seed0/summary.json`.

## Decision rule

If A@28k's advantage over A@16k / H0 grows with density, report a robustness
benefit.  If it shrinks or reverses, report the boundary and do not claim
density generalization.  Screen results are descriptive; confirmatory claims
require the 100-episode, 3-seed endpoints.

**Decision (2026-10-28):** the A28 advantage does **not** grow with density;
it reverses on task and is mixed on safety.  Report the task boundary and
do not claim density generalization for 28k.  No 100ep x 3-seed confirmation
is justified by this screen.
