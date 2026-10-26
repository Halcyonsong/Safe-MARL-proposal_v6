# V6 navigation: predictive-h task gate

Status: the first broad gate was stopped after seed-0 100ep (2026-10-24)
because it produced a safety/task tradeoff.  After Stage A and two Stage-B h
repairs, the repaired Stage-B predictive-h gate passed the pre-registered
3-seed navigation guardrails (2026-10-25): mean `safe_episode=0.693`,
`official_safety=0.942`, `official_success=0.410`, `episode_return=66.85`
(seeds 0/1/2, 100ep each).  This is the promoted navigation gate result.  The
earlier broad-gate checkpoints remain archived and are not promoted.

## Hypothesis

The Ant branch succeeded when a predictive safety h was used as a **reward
gate**, not only as an actor feature.  The navigation h is already strong
offline (test balanced accuracy 0.985), but the formal h-only actor still
collides in 55% of episodes.  This experiment tests the same reward coupling
on the 8-agent navigation task.

The gate is training-only and is disabled at evaluation.

## Implementation

Changed:

- `nav/v6/training.py`: `safety_task_gate`, `apply_safety_task_gate`,
  `V6Rollout.task_gate`, gate logging.
- `nav/scripts/train_v6.py`: `--task-gate-low`, `--task-gate-high`,
  `--task-gate-positive-only`, gate diagnostics.
- `nav/configs/v6/v6_nav_h_gate.yaml`.
- `nav/tests/test_v6.py`: gate unit test.

Gate:

```text
gate_i = clip((h_next_i - low) / (high - low), 0, 1)
gate_i = 1 if high <= low
```

Only positive task progress/reach reward is gated:

```text
task_reward_i = progress_i * reward_scale
              + reach_bonus * reached_i
              + terminal_reach_bonus * reached_i * terminated

task_reward_i = gate_i * task_reward_i   if task_reward_i > 0
              = task_reward_i           otherwise

train_reward_i = task_reward_i - unsafe_penalty * collision_i
```

The official collision cost is never gated.  No QP, action projection, task
channel, expert action, or official h is used.

## Calibration

- frozen V6 nav h: `outputs/v6/independent/h/v6_h.pkl`
- h is centered by the V6 calibration (`h_offset=-0.8621`);
- observed healthy h plateau is about 0.138; negative h is about 2-3% of
  on-policy states;
- gate low = 0.0, high = 0.13;
- this is **not** the Ant 0.05 threshold.

## Protocol

- 8 agents, 4x4 area, 8 obstacles, 256 steps;
- frozen V6 h, from-scratch MAPPO, no task channel;
- 16384 steps, seeds 0/1/2, deterministic training to match the formal
  h-only baseline;
- evaluation is a separate 100-episode deterministic run with eval seed
  10000 (same range for every seed);
- baseline: existing h-only formal 3-seed, same 16384 steps, fixed eval range;
- final checkpoint is used, not the 20-episode training-eval best.

## Seed-0 result (100 episodes, deterministic)

| metric | h-only | h + gate | delta |
|---|---:|---:|---:|
| `safe_episode` | 0.620 | 0.700 | +8.0 pp |
| `collision_episode` | 0.380 | 0.300 | -8.0 pp |
| `official_safety` | 0.9287 | 0.9400 | +1.1 pp |
| `success` | 0.5325 | 0.3725 | **-16.0 pp** |
| `reach` | 0.5763 | 0.3913 | -18.5 pp |
| `episode_return` | 69.78 | 64.96 | -6.9% |
| `action_abs` | 0.2873 | 0.2003 | lower, not zero |

20ep quick check agrees: safe `0.50 -> 0.65`, success `0.5125 -> 0.3813`.

This is not a clean success.  Safety improves, but the task cost is too large
for the current gate.  The 100ep evaluations of seeds 1 and 2 were stopped
after seed 0; no three-seed aggregate is claimed.  The checkpoints remain
available if a later re-evaluation is useful.

## Guardrails

Primary:

- `safe_episode`
- `collision_episode`
- `official_safety`

Task guardrail:

- `success`
- `reach`
- `episode_return`

The gate is not accepted if safety improves only by collapsing success toward
standstill.  A positive seed-0 result must be repeated with seeds 1 and 2
before any three-seed claim.

If the first gate setting is null, do not sweep gate coefficients.  The next
controlled change is either (a) a quantile-calibrated low/high pair that gates
the pre-collision h tail, or (b) an explicit hdot lookahead
`h_next + beta * (h_next - h_prev)`.  Both are changes to safety coupling /
temporal prediction, not a coefficient sweep.

## Stage A follow-up

The first result showed the coupling structure was too broad: all positive
reward, including reach and terminal success, was gated.  Stage A keeps the
same frozen h and thresholds but gates **only dense progress**, following the
Ant forward-reward design.  See
[`STAGE_A_GATE_PROGRESS.md`](STAGE_A_GATE_PROGRESS.md).  The first gate
checkpoints remain archived as the broad-coupling control.

Stage A's 100-episode seed-0 result did **not** improve safety: `safe_episode`
0.530 vs h-only 0.620, `success` 0.5075 vs 0.5325.  The 20-episode monitor
(0.700/0.581) was optimistic.  Stage A is therefore not promoted; seeds 1/2
were not run.  The broad gate remains exploratory.  Next is Stage B: on-policy
h data and a future-min-margin target, followed by from-scratch h and actor
training.


## Stage B status (2026-10-25)

Stage B is the controlled h-quality experiment: V6-owned on-policy data from
the h-only and Stage-A seed-0 policies, stochastic actions, and the
future-minimum geometric margin target.  The first collection attempt was
interrupted by a server restart at episode 23/100 and saved no arrays; it was
restarted from scratch at 00:21 with the same protocol.  The new h is trained
from scratch and the actor is then trained from scratch with the frozen new h.
Gate thresholds will be calibrated from the new h distribution, not copied
from the old `0.0/0.13` pair.  See
[`STAGE_B_H_DATA_TARGET.md`](STAGE_B_H_DATA_TARGET.md).

Stage B later failed (result below).  The Ant 10M seeds 1/2 and matched
cost-only seeds 1/2 completed on 2026-10-25: the predictive gate keeps
`0.97/0.95/0.99` deterministic safe episodes across seeds at 91.6% of
cost-only forward progress, while cost-only is `0.10/0.08/0.01` safe.  See
`outputs/v6/independent/ant/MULTISEED_10M_PLAN.md`.  The Ant mechanism is
therefore validated; the open problem is the navigation h/data transfer, not
the gate concept itself.

## Stage B result (seed 0): failed, stopped

Stage B retrained h from scratch on 100 on-policy episodes with the
future-min-margin target, then retrained the actor from scratch with the new
frozen h and the pre-registered gate `low/high=0.0/0.02`.  The fixed final
checkpoint scored, over 100 deterministic episodes:

```text
safe_episode       0.000
collision_episode  1.000
official_safety    0.4400
official_success   0.0375
official_reach     0.0700
episode_return    -23.60
```

This is worse than h-only and Stage A on every axis.  The 8k and 16k monitors
were already zero-safe, so this is not a 100ep evaluation artifact.  At 2,048
actor steps the new h flagged 40.4% of rollout states negative and the gate
closed 40.5% of them, versus 6.3% closed in Stage A at the same step.  The
new h's negative lower tail on early from-scratch actor states suppressed
progress before the actor learned to reach.

Decision: the original Stage B fails; its seeds 1/2 were not run and no
multi-seed result is claimed for that h.  Do not sweep the gate threshold.
The next controlled experiment must address the distribution shift / gate
coupling.  (The repaired h variant later passed the 3-seed guardrails; see
the 3-seed result below.)

## Stage B repair (2026-10-25)

The failure was traced to the h objective, not an h offset/sign bug: the new
h never learned the future-min target (`corr(h,target)=0.079` vs Ant `0.811`;
safe-row corr `0.013`).  The repaired data keeps the same V6 sources but uses
wall-free future-min margins, SmoothL1(beta=1.0), 40 epochs at 1e-3, explicit
local state-gradient alignment, and no hdot term.  Graph rebuild for the
gradient is exact (valid-edge diff `0.0`, unchanged-state h diff `0.0`), and
its gradient convention matches the actor provider (mean cosine `0.993`).
Training is running from scratch as seed 0; actor retraining and gate
recalibration wait for the offline guardrails (target corr > 0.6, positive
safe-row corr, gradient cosine > 0.8, acceptable BA/AUROC).

The first repaired run hit a NaN at batch start 22,528 of epoch 1.  The loss was
finite and the FD gradient was finite; the NaN came from two separate issues:
the analytic `jacfwd` training path was second-order, and the cosine loss used
`jnp.linalg.norm`, whose derivative at an exactly-zero gradient is `0/0` even
when the row is masked.  Training now uses central finite differences and a
differentiable `sqrt(pred_sq*target_sq + 1e-12)` denominator, plus Ant-style
global gradient clipping (`max_grad_norm=1.0`).  A full-data 1-epoch stability
run completes: target corr `0.685`, safe-row corr `0.684`, gradient cosine
`0.724`, BA `0.914`, AUROC `0.980`.

The old Stage-B gate-off control (same h, `task_gate_high=0.0`) finished at
`safe_episode=0.03`, `official_reach=0.135`, `official_success=0.0763`,
`episode_return=7.34` (100ep deterministic), confirming that the old h
features are harmful even without the gate.

The repaired 40-epoch h finished (best epoch 26).  Val: target corr `0.800`,
safe-row corr `0.768`, gradient cosine `0.911`, BA `0.994`, AUROC `0.9997`.
Test: target corr `0.794`, safe-row corr `0.736`, gradient cosine `0.911`,
BA `0.993`, AUROC `0.9997`.  The pre-registered gate calibration gives
`low=0.0`, `high=0.267270`; on validation the gate is fully closed on all
unsafe rows, mean `0.95` on safe rows, and only `4.6%` of safe rows are below
0.5.

The repaired actor seed 0 completed the formal 100ep evaluation:
`safe_episode=0.840` (h-only `0.620`), `collision_episode=0.160` (h-only
`0.380`), `official_safety=0.9738` (h-only `0.9287`),
`official_success=0.4275` (h-only `0.5325`), `official_reach=0.4350`
(h-only `0.5763`), `episode_return=60.91` (h-only `69.78`).

### 3-seed result (promoted)

Seeds 1/2 then completed with the same frozen repaired h, gate, config,
16,384-step budget, and 100 fixed eval scenes:

| seed | safe_ep | collision_ep | official_safety | success | reach | return |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.840 | 0.160 | 0.9738 | 0.4275 | 0.4350 | 60.91 |
| 1 | 0.510 | 0.490 | 0.9000 | 0.4575 | 0.4963 | 70.92 |
| 2 | 0.730 | 0.270 | 0.9513 | 0.3450 | 0.3575 | 68.74 |
| **mean** | **0.693** | **0.307** | **0.9417** | **0.4100** | **0.4296** | **66.85** |
| **std** | 0.168 | 0.168 | 0.0378 | 0.0583 | 0.0695 | 5.26 |

Against the matched h-only 3-seed mean (`safe=0.450`, `safety=0.8783`,
`success=0.4067`, `return=61.73`), the gate adds `+24.3 pp` safe episodes,
`+6.3 pp` official safety, and `+8.3%` return, with success unchanged
(`+0.3 pp`).  All pre-registered guardrails pass.  The gain is not uniform:
seed 1 is nearly neutral on safety; task shifts are `-10.5/+4.0/+7.5 pp`
success for seeds 0/1/2.  With `n=3`, seed-level paired differences are not
statistically separated from zero.

Full protocol, per-seed table, and caveats:
[`STAGE_B_REPAIR_MULTISEED.md`](STAGE_B_REPAIR_MULTISEED.md).  Do not sweep
`task_gate_high`; any further arm needs a new pre-registration and approval.
