# V6 Navigation W16 Task Warm-Start (seed-0 screen)

Status: completed 2026-10-28.  The 3-seed W16 arm recovers task strongly
(`success 0.6175` at 28k vs `0.4100` at 16k) but fails the 7.1 mean-safety
and spread gates.  It is a task-first trade-off; the 16k row remains
promoted.

## Question

The A-only 28k run improves mean safety but loses task success relative to
16k.  Can the frozen task-value channel recover the task level when it is
turned on for the last 12,288 steps of the same 16k -> 28k trajectory?

This is a **procedure-level** follow-up, not a new algorithm family and not a
post-hoc replacement for the pre-registered 28k primary endpoint.

## Protocol

- Base: seed-0 `policy_step_16384.pth` from
  `outputs/v6/independent/nav_stageb_repair_28k/`.
- Train from 16,384 to 28,672 steps (12,288 steps) with:
  - the same frozen repaired h (`h_stageb_repair`);
  - the same frozen V_T (`t0w_vt/v6_task_value.pkl`);
  - `use_task_scalar=true`, `use_task_grad=true`,
    `task_grad_scale=0.5`, `zero_init_block=true`;
  - the same gate, cost, reward, optimizer, and seed as the A-only run.
- Checkpoints at 24,576 and 28,672; formal 100 deterministic episodes on
  eval seeds `10000..10099`.
- Compare each checkpoint to the from-scratch A-only checkpoint at the same
  step.  No C16/C24 control is run: the from-scratch 16k -> 28k trajectory is
  the budget-matched control for this procedure.  The T0w 16k+4k no-V_T arm
  remains the channel-attribution evidence, not a second control here.
- Seed-0 screen first.  Expand to seeds 1/2 only if the screen passes.

## Pre-registered seed-0 screen

Primary point: 28k (28,672).  24k (24,576) is diagnostic; a gain that exists
at 24k but fades at 28k does not pass.

Task benefit (28k, either):
- `official_success >= 0.30` (at least +10.1 pp over the from-scratch 28k
  seed-0 value `0.19875`), or
- `episode_return >= 65.0` (at least +9.48 over `55.52`).

Safety not collapsed (28k, both):
- `safe_episode >= 0.80` (from-scratch seed-0 `0.860`, margin `-0.06`);
- `official_safety >= 0.95` (from-scratch seed-0 `0.975`, margin `-0.025`).

Cost sanity:
- `unsafe` (the evaluator's per-step cost-rate field, the navigation
  analogue of `cost_fraction`) `<= from-scratch + 0.005`.

The 24k diagnostic uses the same task direction and safety floors; a task
gain at 24k is not enough if 28k returns to the from-scratch level.

## Outcomes

- **Pass:** run seeds 1/2 with the same protocol and apply the section 7.1
  variance rule to the 3-seed W16 final.  Promotion still requires a new
  pre-registration; the 28k A-only row is not silently replaced.
- **Task gain but safety below floor:** report a task/safety trade-off; do
  not promote and do not add seeds.
- **No task gain:** stop the navigation task-channel line.  Do not tune the
  gate, V_T scale, or checkpoint on the test episodes.

## Explicit non-goals

- No C16/C24 control in this block.
- No `task_gate_high` sweep.
- No test-set tuning or best-checkpoint selection.
- No change to the promoted 16k row or the 28k primary endpoint.
- No claim that the Ant 1-D null is reversed by this navigation result.

## Seed-0 result (2026-10-28)

Formal 100ep:

| checkpoint | safe | official | success | reach | return | unsafe |
|---|---:|---:|---:|---:|---:|---:|
| W16 24k | 0.930 | 0.98875 | 0.5700 | 0.5725 | 152.42 | 0.00081 |
| W16 28k | 0.830 | 0.97250 | 0.6213 | 0.6363 | 237.36 | 0.00219 |
| from-scratch 24k | 0.650 | 0.93250 | 0.3688 | 0.3800 | 73.17 | 0.00362 |
| from-scratch 28k | 0.860 | 0.97500 | 0.1988 | 0.2063 | 55.52 | 0.00136 |

28k W16 - from-scratch: safe `-3.0 pp`, official `-0.25 pp`, success
`+42.25 pp`, reach `+43.0 pp`, return `+181.84`.  The pre-registered screen
passes: task benefit (`success 0.6213 >= 0.30`), safety floor
(`safe 0.83 >= 0.80`, `official 0.9725 >= 0.95`), and cost sanity all pass.

The 24k checkpoint is even stronger on safety (`0.93`) and still has
`success 0.57`; the 28k arm trades `10 pp` safe for `+5.1 pp` success and a
large return gain relative to 24k.  This is a real task-channel signal, not a
budget-only effect, but it is one seed and uses a seed-0 V_T.

Decision: run seeds 1/2 with per-seed V_T (same protocol), then apply the
section 7.1 variance rule to the 3-seed W16 28k row.  Do not promote W16 or
replace the 16k row before that.  No test-set tuning, no `task_gate_high`
sweep, and no best-checkpoint selection.

## 3-seed result (2026-10-28)

Seeds 1/2 used per-seed V_T trained on that seed's 16k rollouts (100
episodes, seeds `30000..30099`; val EV seed 0 `0.227`, seed 1 `0.161`,
seed 2 `0.294`).  Seed 1 has the weakest V_T and also the largest safety
loss; this is a plausible contributor to the seed heterogeneity, not a
post-hoc excuse.  The same 12,288-step W16 protocol was used; formal 100ep
at 24k and 28k.

W16 28k per seed: safe `0.83/0.56/0.60`, official
`0.9725/0.9075/0.9187`, success `0.62125/0.67375/0.55750`, return
`237.36/217.40/140.95`.  Means: safe `0.6633`, official `0.9329`, success
`0.6175`, reach `0.6525`, return `198.57`.

W16 24k per seed: safe `0.93/0.55/0.69`, official
`0.98875/0.91250/0.95000`, success `0.5700/0.6325/0.5850`, return
`152.42/156.31/122.47`.  Means: safe `0.7233`, official `0.9504`, success
`0.5958`, return `143.73`.

Paired W16 28k - from-scratch 28k: seed 0 safe `-3.0 pp`, official
`-0.25 pp`, success `+42.25 pp`, return `+181.84`; seed 1 safe `-14.0 pp`,
official `-3.75 pp`, success `+27.0 pp`, return `+132.87`; seed 2 safe
`-18.0 pp`, official `-4.25 pp`, success `+33.0 pp`, return `+90.66`.

Rule 7.1 on the 28k row: mean success/return, worst success, worst safe
and cost pass; **mean safe `0.6633 < 0.6933`, mean official
`0.9329 < 0.9417`, and safe range `0.27 > 0.20` fail**.  Full rule fails.
The 24k diagnostic passes mean safety/official/task but its safe range is
`0.38` and worst seed is `0.55`, so it also fails the spread rule.

**Decision: W16 is a task-first trade-off, not a promotion.**  It recovers
task strongly (`success 0.6175` vs `0.4100` at 16k and `0.2767` at
from-scratch 28k) but loses mean safety relative to 16k and is more
seed-variable, especially seeds 1/2.  Keep 16k as the promoted row; do not
switch to 28k or 24k post hoc.  A safety-regularized task channel or a
5-seed confirmatory pre-registration is a separate proposal.

## Records

- Config: `nav/configs/v6/v6_nav_w16_task_12k.yaml`
- Runner: `nav/scripts/run_v6_nav_w16_task_12k.sh`
- Output: `outputs/v6/independent/nav_w16_task_12k/seed0/`
- Seed-0 screen decision: `.../seed0/screen_decision.json`
- Per-seed V_T data/training: `outputs/v6/independent/nav_w16_task_12k/vt_data/`
