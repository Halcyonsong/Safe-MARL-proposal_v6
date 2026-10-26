# Why the h policy became faster but not safer

Date: 2026-10-24.  Scope: the 1M dynamic-h policy
(`mappo_h_dyn_2m/seed0/policy_001015808.pth`).  This is an explanation of the
negative closed-loop result, not a new training result.

## Short answer

`h(s)` and `grad_s h` are **observations**, not a safety constraint.  They can
make the task easier to optimize while the policy still chooses wall contact.
The fixed cost penalty is a scalar objective, not a constraint; under that
objective the dynamic-h policy can have a *higher* shaped return than the
cost-only policy even though its cost is larger.

## Evidence

### h is valid and the actor does use it

- On a 1000-step deterministic trajectory, `corr(h, wall clearance)=0.71`;
  h is not a broken safety signal.
- First-layer per-input weight norms are comparable: observation mean `1.429`,
  h `1.435`, gradient mean `1.418`.  The h/grad channels are not ignored.
- Zeroing h/grad changes the deterministic mean action by `0.024` on average
  (action scale `0.113`; cosine similarity `0.978`).  The actor uses them, but
  they are not dominant.

### Changing the h sign does not make the policy safe

10-episode deterministic counterfactual at the same checkpoint and seed 30000
(the policy is evaluated with replaced safety inputs; this is a diagnostic, not
an in-distribution control):

| h/grad input | forward | final_x | safe | wall | cost_fraction |
|---|---:|---:|---:|---:|---:|
| original dynamic h | 525.74 | 26.28 | 0.00 | 1.00 | 0.2611 |
| h=+1, grad=0 | 501.53 | 25.07 | 0.00 | 1.00 | 0.3251 |
| h=0, grad=0 | 474.58 | 23.72 | 0.20 | 0.80 | 0.3380 |
| h=-1, grad=0 | 473.01 | 23.64 | 0.10 | 0.90 | 0.3884 |
| h unchanged, grad=0 | 505.70 | 25.28 | 0.00 | 1.00 | 0.2583 |

The policy remains unsafe for every h sign.  It is using h/grad as part of a
task representation, not as a safety switch.

### The actual scalar objective prefers the faster/unsafe policy

`episode_return` is the task return before the cost penalty.  The approximate
shaped return is `episode_return - cost_penalty * cost_fraction * length` using
the scheduled penalty at that checkpoint:

| checkpoint | arm | episode_return | cost_fraction | shaped return | safe |
|---|---|---:|---:|---:|---:|
| 1M, 100 ep | cost-only | 1267.55 | 0.0784 | ~1236 | 0.29 |
| 1M, 100 ep | dynamic h | 1424.29 | 0.2666 | ~1318 | 0.09 |
| 1.5M, 20 ep | cost-only | 1263.40 | 0.0464 | ~1235 | 0.40 |
| 1.5M, 20 ep | dynamic h | 1674.46 | 0.3514 | ~1438 | 0.00 |

So the dynamic-h arm is not “failing to optimize” the training objective; it
is finding a higher-return aggressive gait.  The cost-only arm is safer but
has a lower shaped return at the same step.  Cost-only safety is an
optimization outcome, not an enforced property: its 10M checkpoint drops to
`safe_episode=0.10` while keeping `cost_fraction=0.0222`.

## Reward asymmetry and the missing future labels

The task is not literally forward-only: the official Ant reward contains
forward, control and survive terms.  Safety, however, enters only as a
**negative cost penalty**; there is no positive safety return in the policy
objective.  PPO therefore has a strong dense task gradient and only a
negative/sparse safety gradient.  A safe action that gives up forward return
has a negative advantage even if it avoids a future cost.

This asymmetry is real, but adding a positive safety score alone would not fix
it: a constant safety reward is equivalent to a cost penalty plus a baseline,
and a state reward such as `+h` would favour standing still.  Safety still has
to be action-conditioned or constrained.

The h supervision is also incomplete.  `forward_safety_labels` does create
`-1` (unsafe now), `+1` (no cost in the next 32 states) and `0` (cost inside the
window but not now).  The collector then does:

```python
labels = forward_safety_labels(unsafe_arr, horizon=horizon)
keep = labels != 0.0
all_labels.append(labels[keep])
```

so the `0` rows — exactly the near-future-unsafe states — are **discarded**.
The h network thus sees only "unsafe now" and "far-safe" states.  Its target
is still the current geometric margin, and the dynamic follow-up only added
`kappa=0.5 s` velocity lookahead.  It is a current-state classifier, not a
future-risk predictor.  The barrier also uses `h_next`, with no hdot or horizon
risk.

## Root causes

1. **State information is not an action constraint.**  A scalar `h(s)` and a
   state-space gradient do not specify which joint torques keep Ant safe.  The
   actor must learn an action mapping, but nothing enforces `h(s') >= 0`.
   `grad_s h` is not a control-barrier action direction.
2. **The fixed cost penalty is too weak relative to task forward reward.**  The
   shaped returns above show that wall contact is worth it for the dynamic-h
   gait under the scheduled coefficient.  Raising the coefficient until h is
   safe would trade away forward and eventually produce standstill; that is a
   hyperparameter problem, not a safety guarantee.
3. **h is myopic.**  The target is the current geometric margin (with only
   `kappa=0.5 s` velocity lookahead in the dynamic variant).  Future-unsafe
   states are still ignored by the labels.  The warning arrives too late for
   the Ant's turning/braking dynamics, especially as the policy gets faster.
4. **The h barrier penalizes a state, not an unsafe action.**  `2 * relu(0.1 -
   h_next)` makes any low-h state expensive.  The easiest optimum is to stand
   still in a high-h state (`h≈0.55`, cost 0, forward -2).  It has no
   task-preserving feasible-action mechanism and no hdot/future-risk term.
5. **The extra features can help the task.**  Dynamic h adds a nonlinear,
   velocity-sensitive torso/clearance representation to a 31-D observation.
   That is useful for locomotion and can improve forward learning; it is not
   evidence that the safety objective is being enforced.  The task effect is
   consistent across 1M/1.5M and deterministic/stochastic, but this is still a
   single seed.
6. **One global h for two agents.**  The h/grad vector is repeated to both
   agents and contains no leg/contact/agent-specific information, so it cannot
   assign responsibility for safety or provide a per-agent correction.

## What not to do

- Do not run 3 seeds, 10M, or navigation task-joint on this h mechanism.
- Do not keep sweeping h feature scale, cost penalty, or barrier coefficient
  and call the best checkpoint a safety method.
- Do not treat the cost-only 2M checkpoint as a robust safe policy until its
  retention across seeds and later steps is checked; its 10M checkpoint is
  already much less safe.

## Why navigation h is not a counterexample

The navigation policy in `nav/v6/policy.py` has separate task, correction and
feasibility heads; h is not merely concatenated to a task actor.  Its h is
also graph/local LiDAR geometry, not a single shared 9-D torso summary.  A
navigation h result therefore does not establish that a state-only h feature
is sufficient for Ant.  Before using nav as evidence, ablate the
correction/feasibility heads; otherwise the safety gain may come from the
action layer, not from h.

## Next directions

### A. Current-h gate: tested, partial only

A three-seed 2M gate tested `forward * clip(h_next/0.3, 0, 1)` with the official
cost penalty.  Stochastic cost fell from `0.0987` to `0.0432` and safe episodes
rose from `0.067` to `0.180`, but deterministic seed 1 was worse than
cost-only (`cost 0.0665 vs 0.0430`, `safe 0.12 vs 0.24`) and the deterministic
mean forward fell 11.7%.  The deterministic mean cost also did not fall
(`0.0223 vs 0.0199`).  See `GATE_2M_RESULTS.md`.

### B. Predictive h + gate: tested, passes the 2M robust gate

The collector now retains the old future-unsafe `0` rows, h is trained on the
32-step future minimum margin, and the gate uses that predictive h.  Offline h
test: balanced accuracy `0.9145`, unsafe recall `0.8793`, false alarm `0.0364`,
AUROC `0.9511`, gradient cosine `0.9225`.  Three-seed 2M, 100 episodes:

| mode | arm | forward | safe | wall | topple | cost |
|---|---|---:|---:|---:|---:|---:|
| det mean | cost-only | 905.75 | 0.573 | 0.420 | 0.010 | 0.0199 |
| det mean | predictive gate | 756.60 | 0.977 | 0.023 | 0.000 | 0.00232 |
| sto mean | cost-only | 525.38 | 0.067 | 0.603 | 0.563 | 0.0987 |
| sto mean | predictive gate | 472.22 | 0.317 | 0.267 | 0.470 | 0.0264 |

This is the first h mechanism that passes the three-seed 2M safety gate:
cost falls 88% deterministic / 73% stochastic, safe episodes rise strongly,
and forward retains 83.5% / 90% (no standstill).  Seed 0 is the caveat
(deterministic cost `0.0069 vs 0.0044`, but safe/wall improve and forward is
lower).  Details: `PREDICTIVE_H_GATE_2M.md`.

### C. 10M retention: tested seed 0, passes

The predictive-h gate was trained to 10M and evaluated on matched 100ep
checkpoints.  Deterministic: forward `3723.09`, safe `0.97`, wall `0.03`,
cost `0.00176` at 10M, versus cost-only `4055.37 / 0.10 / 0.89 / 0.0222`.
Stochastic: forward `2923.73`, safe `0.78`, cost `0.00419`, versus
`3067.39 / 0.04 / 0.0508`.  The gate retains 91.8% deterministic and 95.3%
stochastic forward while cutting cost by 92%; there is no late safety collapse.
See `RETENTION_10M.md`.

The 10M seeds 1/2 and matched cost-only controls completed on 2026-10-25.
The three-seed deterministic gate mean is forward `3771.51` (`91.6%`
retention), safe `0.97`, cost_fraction `0.0012`, versus cost-only
`4119.70 / 0.063 / 0.0296`; stochastic is `3010.73` (`93.9%`), safe `0.84`,
versus `3206.28 / 0.020`.  Every gate seed is safe while every cost-only seed
is unsafe.  No seed/cost-critic retry is needed for the Ant claim; see
`MULTISEED_10M_PLAN.md`.  Navigation remains the open transfer target.

### D. Still not done: explicit hdot and action-conditioned safety

No explicit actor `hdot` feature was added; the future-window h target provides
the temporal signal.  `hdot` remains a diagnostic/fallback if 10M temporal
failures appear.  No action-conditioned `Q_cost(s,a)`/feasibility head was
trained; it requires relaxing the current “h is an actor feature, not an action
correction” rule.

### E. If 10M fails: baseline and pivot

The cost-only 2M baseline is itself seed-sensitive: deterministic
`cost_fraction` is `0.0044 / 0.0430 / 0.0124`, and `safe` is
`0.90 / 0.24 / 0.58`; the 10M control is less safe.  If predictive h does not
retain safety at 10M, close the Ant h branch and report the 2M predictive
result as a positive but non-retained boundary; pivot to nav only with an
explicit action/feasibility layer, not state-only h.
