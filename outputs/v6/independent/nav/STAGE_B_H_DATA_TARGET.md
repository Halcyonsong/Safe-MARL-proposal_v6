# V6 navigation Stage B: on-policy h data + future-min-margin target

Status: the **original** Stage B h/data-target **failed at actor seed 0**;
its actor seeds 1/2 were not run.  The **repaired** future-min h variant then
passed the pre-registered 3-seed guardrails (seeds 0/1/2, 2026-10-25).  Data
collection restarted 2026-10-25 00:21 after a server restart (the first
attempt was interrupted at episode 23/100 and saved no arrays).  Stage B
changes h data and h target only.  It does not add a new gate scope, cost
schedule, task channel, or action filter.  Multi-seed protocol and results:
[`STAGE_B_REPAIR_MULTISEED.md`](STAGE_B_REPAIR_MULTISEED.md).

## Why

Stage A (progress-only gate) recovered task performance but did not improve
safety over h-only in 100 episodes.  The on-policy h gate was weakly active
(`gate_mean≈0.95`; only about 5% closed at the end).  The existing V6 h was
trained on random-policy data and binary horizon labels, while the task policy
visits a different state distribution.

Stage B therefore retrains h from scratch on **V6 policy states** with a
continuous **future-minimum geometric margin** target, then retrains the actor
from scratch with the new frozen h.

## Data collection

Config: `nav/configs/v6/v6_nav_h_stageb.yaml`.

Behavior sources (both V6-owned, no V5/V4/official weights):

- `outputs/v6/independent/formal_3seed/h_only/seed0/policy_final.pth`
- `outputs/v6/independent/nav_gate_progress_16k/seed0/policy_final.pth`

Protocol:

- 100 episodes, 50 per behavior source, 256 steps;
- stochastic actions from the behavior policy;
- behavior h for the actor's h/grad features: old V6 h
  `outputs/v6/independent/h/v6_h.pkl` (only a behavior input; not a target);
- seed 6200;
- splits 80/10/10 by episode;
- output: `outputs/v6/independent/data_stageb/`.

Only official environment collision/unsafe masks and V6-generated states are
used.  No task reward, expert action, QP/u_ref, or official h enters the h
labels.

## Target

For each agent at each state:

```text
margin_i(t) = dense geometric clearance from the current observation
              (nearest neighbor / LiDAR obstacle / wall)

future_margin_i(t) = min(margin_i(t), ..., margin_i(t + 32))

target_i(t) = tanh(future_margin_i(t) / 0.2)
```

Implementation:

- `nav/v6/safety.py`: `dense_margin_values`, `horizon_min_margin`,
  `horizon_margin_targets`;
- collect script appends the terminal next state so a final-step collision
  closes the preceding window;
- states with an official unsafe/warning event inside the horizon are forced
  to target `<= -0.1`, because the environment warning zone is
  heading-dependent and is not fully represented by pure geometry;
- binary horizon labels are retained for calibration/evaluation and for the
  existing safe/unsafe hinge terms;
- target regression is applied to **all** rows, including ignored rows
  (`labels == 0`, i.e. unsafe somewhere in the forward window but not now).
  Ignored rows are exactly where the future-min target carries predictive
  information; excluding them would make `target_on_all` ineffective.  The
  hinge terms still use only safe/unsafe labels;
- target scale `0.2` is a V6 navigation geometry scale, **not** the Ant `0.5`
  scale.

## h retraining

Config: `nav/configs/v6/v6_nav_h_stageb.yaml`.

- full-graph V6 h, from scratch (same architecture family as the old V6 h);
- 10 epochs, batch 128, lr 1e-4, weight decay 1e-4;
- safe/unsafe hinge losses plus target regression (`target_coef=0.5`,
  target on all rows including ignored); a synthetic all-ignored split gives
  nonzero target loss with `target_on_all=true` and zero with `false`,
  confirming the semantics;
- hdot consistency term as before;
- threshold calibrated only on the new V6 validation split;
- output: `outputs/v6/independent/h_stageb/v6_h_stageb.pkl`.

No old h parameters are loaded; this is a from-scratch h retrain.

## Actor retraining

Config: `nav/configs/v6/v6_nav_stageb_16k.yaml`.

- new h frozen;
- actor from scratch;
- same progress-only gate coupling as Stage A;
- gate thresholds will be calibrated from the **new h distribution** on the
  Stage B validation data; do not copy the old h's `low=0.0/high=0.13`.
  Specifically, `low=0.0` (the calibrated h=0 safety boundary) and
  `high=max(0.02, q10_safe_h - validation_threshold)`, where the last term is
  the 10th percentile of horizon-safe policy-h values.  In checkpoint space,
  `h_shifted = h_raw - threshold`, so this is
  `high=max(0.02, q10(h_shifted | label > 0))`.  The diagnostic
  `nav/scripts/calibrate_v6_nav_gate.py` computes it on validation next-state
  graphs and reports gate activation separately for safe/ignored/unsafe
  states.  This closes the gate on the lower tail without a threshold sweep;
- 16,384 steps, monitor-only 20ep at 8k and 16k, fixed final checkpoint;
- 100ep deterministic at eval seed 10000;
- no seeds 1/2 unless the seed-0 100ep safety and task guardrails hold (this
  condition was later met by the repaired variant).

## Guardrails

- h must not collapse on the new validation set: report balanced accuracy,
  unsafe recall, false alarm, AUROC, and h quantiles.
- Gate activation on behavior data must be reported; a gate that is almost
  always open is not a safety mechanism.
- Actor must improve `safe_episode`/`collision_episode` without a large
  `success`/`return` collapse.
- No threshold sweep: one data-calibrated threshold pair, then test.
- No action-conditioned filter/QP; no official h; no Ant weights.

## Stage-B h result (seed 0, 2026-10-25)

Trained from scratch with `nav/configs/v6/v6_nav_h_stageb.yaml` and
`target_on_all=true`.  Validation selects epoch 1; later epochs increase
false alarms and lower balanced accuracy.

| split | balanced acc | unsafe recall | safe recall | false alarm | AUROC |
|---|---:|---:|---:|---:|---:|
| val | 0.8714 | 0.8667 | 0.8761 | 0.1229 | 0.9406 |
| test | 0.9029 | 0.9070 | 0.8989 | 0.1005 | 0.9673 |

Checkpoint calibration:

- raw threshold `0.584499`, `h_offset=-0.584499`;
- provider-space `q10(h | label>0) = -0.056727`;
- pre-registered gate rule gives `low=0.0`, `high=max(0.02, -0.056727)=0.02`.

Gate activation on the validation graphs (current/next nearly identical):

| group | n | gate mean | closed (<0.5) | fully closed |
|---|---:|---:|---:|---:|
| all | 20,480 | 0.857 | 14.3% | 13.9% |
| safe | 19,923 | 0.871 | 12.8% | 12.4% |
| ignored | 392 | 0.415 | 58.7% | 57.7% |
| unsafe | 165 | 0.128 | 87.3% | 86.7% |

This is a much more active gate than the old h (`~5%` closed): it closes most
ignored/unsafe states while keeping most horizon-safe states open.  The
`high=0.02` floor comes from the negative safe-h lower tail, not from a
threshold sweep.

Artifacts:

- `outputs/v6/independent/h_stageb/v6_h_stageb.pkl`
- `outputs/v6/independent/h_stageb/summary.json`
- `outputs/v6/independent/h_stageb/gate_calibration.json`

## Stage-B actor result (seed 0) and stop decision

The actor was trained from scratch for 16,384 steps with the frozen Stage-B h
and `task_gate_low/high=0.0/0.02`, then evaluated on the fixed final
checkpoint over 100 deterministic episodes (eval seed 10000).

| metric | h-only | Stage A | Stage B |
|---|---:|---:|---:|
| `safe_episode` | 0.620 | 0.530 | **0.000** |
| `collision_episode` | 0.380 | 0.470 | **1.000** |
| `official_safety` | 0.9287 | 0.9113 | **0.4400** |
| `official_success` | 0.5325 | 0.5075 | **0.0375** |
| `official_reach` | 0.5763 | 0.5475 | **0.0700** |
| `episode_return` | 69.78 | 76.85 | **-23.60** |

The 8k and 16k monitors were already `safe_episode=0.0`, so this is not a
100ep-only surprise.

### Diagnosis

- At actor step 2,048 the new h had `h_neg_rate=40.4%` and the gate was
  `40.5%` closed, with `task_reward_mean=-0.135`.
- At the same step Stage A (old h) had `6.3%` closed and
  `task_reward_mean=+0.105`.
- The new h has a broad negative lower tail on the *early from-scratch actor's
  states*.  Those states are not represented in the h data, which came from
  the already-trained h-only/Stage-A policies.  The gate therefore suppressed
  progress before the actor learned to reach.
- The validation-calibrated `high=max(0.02, q10_safe_shifted)=0.02` is a floor
  caused by that negative tail, not a swept threshold.

### Decision

Stage B fails both the primary safety guardrail and the task guardrail.  Do
not run actor seeds 1/2; do not claim a multi-seed Stage-B result.  The h
classifier is usable offline (test AUROC 0.967), but this h/gate/actor coupling
is not.

The next controlled change should address the h/actor distribution shift or
the gate coupling (for example, a gate-off control with the same h, or h data
that covers early from-scratch actor states).  Do not sweep `task_gate_high`.

## Deeper implementation audit (2026-10-25)

The catastrophic actor result is not explained by an h offset/sign bug or a
config typo.  The Stage A vs Stage B config diff changes only the h
checkpoint, h-training metadata, target keys, and gate high.  The provider
applies `h_offset` exactly once; the gate and actor features use the same
shifted h; the gate is applied only to positive dense progress; the official
cost is never gated.  The actor feature construction is identical to
h-only/Stage A.

The real problem is that the new h did **not** learn the future-min target:

| metric | Ant predictive h | Nav Stage-B h |
|---|---:|---:|
| corr(h, target), all rows | **0.811** | **0.079** |
| corr(h, target), safe rows | **0.427** | **0.013** |
| safe-h vs target | close (safe mean 0.065 vs target 0.077) | far (shifted safe mean 0.248 vs target 0.513) |
| h data unsafe rate | **26.1%** | **0.6%** |
| gradient alignment | cosine loss to analytic target grad, `grad_coef=0.1`, cosine 0.92 | none |
| output behavior | graded | near-constant ~0.35 shifted on safe rows, non-monotonic vs target |

### Why the target regression failed

The Ant h is trained as a supervised value:

- `F.smooth_l1_loss` with **default beta=1.0** (gradient up to 1.0);
- 40 epochs, batch 512, lr `1e-3`;
- margin_coef `0.5`;
- gradient-alignment loss to the analytic safety-target gradient
  (`grad_coef=0.1`, `grad_on_all=true`);
- **no hdot/CBF consistency term**.

The nav Stage-B h instead uses:

- `optax.huber_loss(delta=0.1)`, which caps the target gradient at `0.1`;
- target_coef `0.5`, so the maximum target gradient is `0.05`;
- 10 epochs, batch 128, lr `1e-4`;
- an additional hdot term with coefficient `0.2` on safe transitions;
- no gradient-alignment loss.

The target signal is therefore much weaker than the hinge/hdot terms and
cannot shape the safe-state h distribution.  Training confirms this: target
loss is `0.0354` at epoch 1 and `0.0408` at epoch 10, while validation
balanced accuracy is best at **epoch 1** and later h saturates.  The h is a
binary safe/unsafe classifier, not a calibrated future-min value.

### Why the gate then hurts

Because the safe h lower tail is not calibrated, the pre-registered rule gives
`q10_safe_shifted=-0.0567` and floors `high` at `0.02`.  The gate behaves like
a binary classifier boundary.  On the Stage-B validation graphs:

- 14.3% of all agent-steps closed; 12.8% of safe, 58.7% of ignored, 87.3% of
  unsafe;
- 16.6% of the top-decile progress transitions closed;
- 8.6% of near-goal steps (`goal distance < 0.2`) closed, versus 0.8% for the
  old h.

Early actor training is the clearest failure mode: at step 2,048 the new h
flagged 40.4% of rollout states negative and the gate closed 40.5%, with
`task_reward_mean=-0.135`; Stage A had 6.3% closed and
`task_reward_mean=+0.105`.  The gate suppresses the dense progress signal
before the from-scratch actor learns to reach.

There is also a task-coupling mismatch: Ant gates a **forward** reward in open
locomotion, where moving forward is always available.  Nav gates
**goal-directed progress**, and goals often sit near obstacles; closing
progress near constraints can remove the only dense learning signal.  The
current result is consistent with this mismatch.

### Data coverage

The nav behavior policies (h-only/Stage A, stochastic) produce only 0.6%
unsafe and 1.9% ignored rows.  Ant predictive-h data comes from a cost-only
policy and contains 26.1% unsafe rows.  The nav h therefore has very few
near-collision examples, and its lower tail on early from-scratch actor states
is poorly estimated.  This is not a threshold problem; it is a data/target
problem.

### Gate-off diagnostic

Gate-off control (`task_gate_high=0.0`, same Stage-B h and actor budget/seed;
`outputs/v6/independent/nav_stageb_gateoff_16k/seed0`):

| metric | Stage B gated | Stage B gate-off | h-only reference |
|---|---:|---:|---:|
| safe_episode | 0.000 | **0.030** | 0.620 |
| official_reach | 0.070 | **0.135** | 0.576 |
| official_success | 0.0375 | **0.0763** | 0.5325 |
| episode_return | -23.60 | **7.34** | 69.78 |

Disabling the gate recovers a little return but the actor still collapses.
The Stage-B h features themselves are therefore harmful, not only gate
suppression.  This is a diagnostic, not a promoted arm and not a threshold
sweep.

## Repair (2026-10-25)

The Stage-B h objective was repaired to match the Ant predictive-h recipe:
wall-free future-min targets, `SmoothL1(beta=1.0)`, 40 epochs at `1e-3`,
explicit local state-gradient alignment, no hdot term, and risk-enriched data.
Graph rebuild for the alignment gradient is exact, and its convention matches
the actor's h provider (mean cosine `0.993`).

Two numerical issues had to be fixed in the nav h trainer before the repair
could train:

1. the analytic `jacfwd` local gradient made the alignment loss second-order
   and produced NaN parameter gradients on finite losses; training now uses
   central finite differences (`grad_fd_eps=1e-3`) and the exact Jacobian only
   for diagnostics;
2. the cosine normalization used `jnp.linalg.norm`; its derivative at an exact
   zero predicted gradient is `0/0`, which produced NaN parameter gradients
   even when the row was masked out.  The denominator now uses
   `sqrt(pred_sq * target_sq + 1e-12)` with the mask applied afterward.

Global gradient clipping (`max_grad_norm=1.0`) was also added to match Ant.
A full-data 1-epoch stability run now completes with val target corr `0.685`,
safe-row corr `0.684`, gradient cosine `0.724`, BA `0.914`, and AUROC `0.980`.

The 40-epoch repaired h completed from scratch as seed 0 (best epoch 26):

| split | target corr | safe corr | grad cosine | BA | AUROC |
|---|---:|---:|---:|---:|---:|
| val | 0.8004 | 0.7683 | 0.9111 | 0.9936 | 0.9997 |
| test | 0.7940 | 0.7364 | 0.9108 | 0.9926 | 0.9997 |

This passes the offline guardrails (target corr > 0.6, positive safe-row corr,
gradient cosine > 0.8, strong BA/AUROC).  The pre-registered gate calibration
on the repaired validation split gives `threshold_raw=0.061149`, `low=0.0`,
`high=0.267270`.  On validation the gate is fully closed on 100% of unsafe
rows, has mean `0.95` on safe rows, and closes only `4.6%` of safe rows below
0.5.

### Repaired actor result (seed 0)

The repaired actor was trained from scratch for 16,384 steps with the frozen
repaired h and `task_gate_low/high=0.0/0.267270`, then evaluated on the fixed
final checkpoint over 100 deterministic episodes (eval seed 10000):

| metric | h-only seed 0 | repaired Stage B seed 0 |
|---|---:|---:|
| `safe_episode` | 0.620 | **0.840** |
| `collision_episode` | 0.380 | **0.160** |
| `official_safety` | 0.9287 | **0.9738** |
| `official_success` | 0.5325 | **0.4275** |
| `official_reach` | 0.5763 | **0.4350** |
| `episode_return` | 69.78 | **60.91** |

`reach_episode_any=0.97` (some agent reaches in 97/100 episodes),
`h_mean=0.836`, `h_min=0.648`.  The 16k 20ep monitor was consistent:
`safe_episode=0.90`, `official_success=0.4875`, `official_reach=0.4938`,
return `64.74`; unlike the old Stage B, it was not a late collapse.

**Verdict (seed 0):** the safety guardrail passes clearly: `safe_episode`
rises `0.62 -> 0.84`, `official_safety` `0.929 -> 0.974`, and collision
episodes fall `38% -> 16%`.  Task retention is partial at seed 0: success is
`80.3%` of the matched-seed h-only value and return `87.3%`; success is below
the stricter `0.45` matched-seed floor, although it is above the formal h-only
3-seed mean `0.407`.

### Repaired actor result (3 seeds, final)

Seeds 1/2 were then run with the same frozen h, gate, config, step budget, and
evaluation scenes.

| seed | safe_ep | collision_ep | official_safety | success | reach | return |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.840 | 0.160 | 0.9738 | 0.4275 | 0.4350 | 60.91 |
| 1 | 0.510 | 0.490 | 0.9000 | 0.4575 | 0.4963 | 70.92 |
| 2 | 0.730 | 0.270 | 0.9513 | 0.3450 | 0.3575 | 68.74 |
| **mean** | **0.693** | **0.307** | **0.9417** | **0.4100** | **0.4296** | **66.85** |
| **std** | 0.168 | 0.168 | 0.0378 | 0.0583 | 0.0695 | 5.26 |

Matched h-only 3-seed mean: `safe_episode=0.450`, `official_safety=0.8783`,
`success=0.4067`, `reach=0.4450`, `return=61.73`.

All pre-registered guardrails pass:

- mean `safe_episode=0.693 >= 0.55`;
- mean `official_safety=0.9417 >= 0.93`;
- mean `official_success=0.410 >= 0.36`;
- mean `episode_return=66.85 >= 55`;
- worst seed `safe_episode=0.510 >= 0.30` and `success=0.345 >= 0.20`.

**Verdict (3 seeds): pass.**  The repaired Stage B is promoted as the first
V6 navigation 3-seed gate result.  Safety improves by `+24.3 pp` safe
episodes and `+6.3 pp` official safety, while mean success is unchanged
(`0.410` vs `0.407`) and return is `+8.3%`.  The gain is not uniform: seed 1
is nearly neutral on safety and seed 2 carries most of it; seed 0 loses
10.5 pp success while seeds 1/2 gain 4.0/7.5 pp.  With `n=3`, the seed-level
paired differences are not statistically separated from zero.  The gate
threshold was not swept.  Do not combine this result with a new gate scope,
cost-schedule, or task-channel change without a new pre-registration and
approval.

Full multi-baseline table and sampling caveats:
[`BASELINE_COMPARISON.md`](BASELINE_COMPARISON.md).
