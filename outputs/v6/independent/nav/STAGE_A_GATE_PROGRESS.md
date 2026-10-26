# V6 navigation Stage A: progress-only safety gate

Status: seed-0 100ep complete (2026-10-24).  Stage A did **not** improve
safety over h-only; it is not promoted and seeds 1/2 are not run.  This is the
first controlled change after the broad positive-reward gate produced a
safety/task tradeoff.  (The later repaired Stage-B predictive-h gate passed the
3-seed guardrails; Stage A remains archived and is not part of that result.
See [`STAGE_B_REPAIR_MULTISEED.md`](STAGE_B_REPAIR_MULTISEED.md).)

## Motivation

The first navigation gate scaled **all positive task reward**:

```text
gate * (dense_progress + reach_bonus + terminal_success_bonus)
```

Ant's predictive-h gate instead scaled only the forward/progress component.
The broad navigation gate improved seed-0 safety but reduced success from
0.5325 to 0.3725, so it was not promoted and its remaining 100-episode
evaluations were stopped.

Stage A isolates the coupling structure:

```text
gate_i = clip((h_next_i - low) / (high - low), 0, 1)

task_reward_i = gate_i * positive_progress_i
              + reach_bonus_i
              + terminal_success_bonus_i

train_reward_i = task_reward_i - unsafe_penalty * official_cost_i
```

- Only positive dense progress is gated.
- Negative progress remains a penalty.
- Reach and terminal/success bonuses are not gated in this first version.
- Official collision/unsafe cost is never gated.
- The gate is training-only and disabled at evaluation.
- If the un-gated success bonus causes dangerous goal rushing, the next
  controlled variant is a team gate on the shared reach/terminal components.
  Do not add that variant before observing Stage A.

## What is held fixed

- frozen V6 navigation h: `outputs/v6/independent/h/v6_h.pkl`;
- same h thresholds as the first gate: `low=0.0`, `high=0.13`;
- same 8-agent scene, 16,384 steps, 128-step rollouts;
- same fixed cost penalty (`unsafe_penalty=10`);
- same actor architecture and from-scratch initialization;
- no h data/target change, no task channel, no h co-training;
- no QP, projection, action filter, expert action, or official h.

This is deliberately **not** an all-Ant-features bundle.  h quality, temporal
target, cost schedule, team risk, and task channel are separate later stages.

## Protocol

Config: `nav/configs/v6/v6_nav_gate_progress_16k.yaml`

- seed 0 first;
- monitor-only 20-episode deterministic evaluations at 8k and 16k;
- final checkpoint is used for the formal result, not the monitor-best
  checkpoint;
- separate 100-episode deterministic evaluation with eval seeds
  10000..10099;
- compare against the matched h-only seed-0 100-episode result;
- if Stage A passes, repeat seeds 1 and 2 before any three-seed claim.

## Acceptance guardrail

Safety must improve, but not by collapsing task performance.  For the seed-0
pilot, compare against h-only:

- primary: `safe_episode` and `collision_episode`;
- task guardrail: `success` should remain within roughly 5 percentage points
  of h-only, and `episode_return` should not collapse;
- diagnostic: `action_abs`, `reach`, and `official_safety`.

These are decision rules for whether to spend the three-seed budget, not
post-hoc metric selection.  The final claim uses the fixed final checkpoint
and the fixed eval seed range.

## Code changes

- `nav/v6/training.py`: `apply_task_gate_components` and
  `task_gate_scope`;
- `nav/scripts/train_v6.py`: config/CLI scope selection and logging;
- `nav/configs/v6/v6_nav_gate_progress_16k.yaml`;
- `nav/tests/test_v6.py`: Stage-A scope unit test.

## Results

### Interim 8k monitor (20 episodes, deterministic)

| metric | h-only, 8k | Stage A, 8k |
|---|---:|---:|
| `safe_episode` | 0.600 | **0.100** |
| `collision_episode` | 0.400 | **0.900** |
| `official_success` | 0.1188 | 0.0813 |
| `official_reach` | 0.1313 | 0.1125 |
| `episode_return` | 29.21 | 25.86 |
| `action_abs` | 0.1399 | 0.2083 |
| `h_mean` | not logged here | 0.0767 |

This is worse than the matched h-only monitor on both safety and task at 8k;
it is not the intended "safety up, task down" tradeoff.  The final 16k
checkpoint was still being trained.  Do not spend the 100-episode budget until
the final 20-episode monitor shows a recovery; the 100-episode result is only
for the fixed final checkpoint.

### Final 16k monitor (20 episodes, deterministic)

| metric | h-only, 16k | Stage A, 16k |
|---|---:|---:|
| `safe_episode` | 0.500 | **0.700** |
| `collision_episode` | 0.500 | **0.300** |
| `official_safety` | not logged here | 0.9438 |
| `official_success` | 0.5125 | **0.5813** |
| `official_reach` | 0.5688 | 0.6125 |
| `episode_return` | 68.68 | 80.19 |
| `action_abs` | 0.2784 | 0.2193 |
| `h_mean` | not logged here | 0.1214 |

Unlike the 8k checkpoint, the final checkpoint recovers on both axes in this
20-episode monitor.  This is promising but not yet a formal result: the
monitor uses only 20 episodes and the fixed checkpoint was selected before the
run, not from monitor-best.

### Formal 100ep deterministic result (fixed final checkpoint)

| metric | h-only | Stage A | delta |
|---|---:|---:|---:|
| `safe_episode` | 0.620 | **0.530** | -9.0 pp |
| `collision_episode` | 0.380 | **0.470** | +9.0 pp |
| `official_safety` | 0.9287 | 0.9113 | -1.7 pp |
| `official_success` | 0.5325 | 0.5075 | -2.5 pp |
| `official_reach` | 0.5763 | 0.5475 | -2.9 pp |
| `episode_return` | 69.78 | **76.85** | +10.1% |
| `action_abs` | 0.2873 | 0.2326 | -19.0% |
| `h_mean` | 0.1155 | 0.1134 | -0.0021 |

The 20-episode monitor was optimistic and did not survive the 100-episode
evaluation.  The 8k checkpoint was bad; the final checkpoint still collided in
47% of episodes, so the collapsed 8k result was not merely a transient
mid-run artifact for the formal metric.

### Interpretation

- Progress-only coupling recovered most of the task loss of the broad gate
  (`success` 0.3725 -> 0.5075), but did **not** recover safety; it is worse
  than h-only by 9 pp episode-level safety.
- The broad gate's safety gain came with a 16 pp success collapse.  Therefore
  the safety effect cannot be attributed to progress gating alone.
- The h gate was weakly active on-policy (`gate_mean ~= 0.95`, only ~5% closed
  at the end), so this result is also consistent with h on-policy saturation.
- Compared with the broad gate, Stage A is a different tradeoff: much better
  task, much worse safety.  It is not a clean success.

**Decision:** Stage A fails the primary safety guardrail.  Do not run seeds 1
and 2 for this arm, and do not claim a three-seed navigation gate result.  The
next controlled change should address h quality / temporal target (Stage B)
before adding another gate-scope variant.
