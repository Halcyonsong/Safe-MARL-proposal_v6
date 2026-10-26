# V6 navigation: Ant reference and learnable design points

Date: 2026-10-24; status note added 2026-10-25.

Status note: the recommended order below has partly been executed.  The
repaired Stage-B predictive-h gate now passes the pre-registered 3-seed
guardrails; see [`STAGE_B_REPAIR_MULTISEED.md`](STAGE_B_REPAIR_MULTISEED.md)
and [`../../STATUS.md`](../../STATUS.md).  This file remains a design note, not
a result record.

This note separates three different things that are easy to conflate:

1. **evaluation** of safety during training;
2. **learning signal** and reward coupling;
3. **learnable modules** and data/target design.

The Ant result came from the second and third, not from evaluating more often.

## 1. What Ant does that nav currently does not

### 1.1 Safety evaluation during training

Ant evaluates every 250k steps with 20 episodes (eval seed 20000) and writes
per-checkpoint metrics.  The evaluation is diagnostic: it does not update the
actor or h, and the trainer does not select a "best" checkpoint from it.  The
periodic checkpoints support later 2M/5M/10M retention checks.

The final Ant report uses:

- 100 deterministic episodes;
- 100 stochastic episodes;
- `safe_episode`, `unsafe_episode`, `wall_episode`, `topple_episode`,
  `cost_rate`/`cost_fraction`;
- task `forward_return`, `mean_x`, `final_x`, `survived`.

### 1.2 Nav current state

The nav trainer (`nav/scripts/train_v6.py`) already supports periodic
`evaluate_v6` calls and a `policy_best` checkpoint.  However, the formal nav
h-only and gate runs intentionally used:

```text
eval_interval = 0
save_interval = 0
```

so there is only a final 100-episode deterministic evaluation.  The gate run
also has no intermediate checkpoint retention.

The nav evaluator now supports a separate stochastic mode:

```bash
python nav/scripts/evaluate_v6.py ... --stochastic
```

This is an evaluation-only change; it does not require retraining.

### 1.3 Important caveat

Adding safety evaluation does **not** make the policy safer.  It only makes
failures visible and allows retention/early-warning analysis.  Ant's safety
came from the predictive h, the task/cost coupling, and sufficient training.

## 1.4 Exact Ant gate details worth copying

These are the details that made the Ant gate behave like a safety constraint
rather than another actor feature:

- h target: future-minimum geometry margin over 32 steps;
- h data: matched V6 cost-only 2M policy, stochastic actions;
- gate input: `h_next`, not the current h;
- gate:
  ```text
  clip((h_next - low) / (high - low), 0, 1)
  ```
- Ant calibration: `low=0.0`, `high=0.05`; safe behavior median h≈0.066,
  future-unsafe h negative;
- gate only the **forward** reward;
- control/contact/survive components are left unchanged;
- official cost is added separately with a fixed scheduled penalty
  (roughly `0.1 -> 1.0`), never by Lagrangian/QP;
- training gate activation was ~0.76–0.92, so it actually bound;
- evaluation is gate-off, 100 deterministic + 100 stochastic episodes;
- periodic 20-episode safety eval and 2M/5M/10M retention;
- no action projection/filter.

Nav current gate differs in three important ways:

- h data is random, not the matched task policy;
- the gate is applied to **all positive progress/reach** rather than one task
  component, and there is no separate survive/control component;
- gate activation is ~0.94–0.99, i.e. it is mostly open.

## 2. Larger gaps than evaluation

### 2.1 h behavior data

This is the largest missing reference point.

- Ant predictive h is trained from the **matched V6 cost-only 2M policy**,
  stochastic actions, so the h sees states the task policy actually visits.
- Nav h data (`outputs/v6/independent/data/summary.json`) has
  `"behavior": "random"`: 100 random episodes, 25,600 rows.
- Nav h is very strong offline (test BA 0.985), but the on-policy h
  distribution is saturated (healthy h plateau about 0.138; only 2-4% negative
  states).  Random data may not cover the near-boundary states of a trained
  nav policy.

`nav/scripts/collect_v6_h_data.py` already accepts a V6-owned
`--policy-checkpoint` and optional `--h-checkpoint`.  A proper next iteration
is to collect from the V6 h-only or gate policy, then retrain the nav h.  This
is a **from-scratch V6 h retrain**, not an Ant/V5 weight transfer.

### 2.2 h target

- Ant target:
  `tanh(min_{k<=32} future_margin_{t+k} / 0.5)`, a continuous future-risk
  target with old zero rows retained.
- Nav target: binary horizon labels (unsafe within 32 steps), plus a
  stopping-clearance margin and a hdot consistency term.

The nav h has a temporal label but not the same explicit future-minimum-margin
regression target.  A continuous target may produce a less saturated,
better-calibrated gate signal.  This changes h training data/target, so both h
and the actor must be retrained from scratch.

### 2.3 Gate coupling

Current nav gate:

```text
gate_i = clip((h_next_i - low) / (high - low), 0, 1)
positive task progress/reach_i *= gate_i
```

This produced seed-0 safety improvement but a substantial success drop.  The
next changes should be coupling changes, not a coefficient sweep:

- gate only terminal/reach bonus, not dense progress;
- soft gate that preserves some progress;
- team-level risk for shared success;
- hdot lookahead: `h_next + beta * (h_next - h_prev)`.

### 2.4 Cost-penalty schedule

Ant uses a fixed schedule (for example `0.1 -> 1.0` over a defined ramp).
Nav uses a fixed `unsafe_penalty=10` and a separate cost critic.  A schedule is
worth testing as a control, but it is a safety-coupling/curriculum choice, not
a new network.

## 3. Learnable design points

| component | Ant | Nav current | Learnable? | Main gap / next step |
|---|---|---|---|---|
| safety h network | 9-D MLP, shared global | full-graph GNN, per-agent output | yes | current h works offline; data/target are the gap |
| h input | torso geometry | graph nodes/edges/LiDAR | yes | graph attention / neighbor risk |
| h target | future-min margin regression | binary horizon label | yes | continuous future-min target |
| h temporal term | horizon target | hdot consistency | yes | hdot/lookahead gate |
| h data policy | V6 cost-only 2M, stochastic | random rollouts | yes | collect from V6 h-only/gate policy |
| h calibration | behavior-data threshold | V6 validation threshold | yes | gate low/high from behavior quantiles |
| actor | per-agent MAPPO | from-scratch MAPPO | yes | current |
| centralized critic | yes | yes | yes | current |
| cost critic | cost-penalized reward | separate cost critic | yes | current |
| task value V_T | not used | implemented, disabled | yes | later task-channel stage |
| task gradient | not used | disabled | yes | later, avoid task leakage into h |
| safety gate | forward-only fixed gate | positive progress/reach fixed gate | fixed function | improve coupling, not learn a new label |
| cost penalty | fixed/ramped | fixed | schedule | test ramp/curriculum |
| multi-agent risk | 2 agents, shared h | 8 agents, per-agent h | yes | team/neighbor risk aggregator |
| credit assignment | MAPPO | MAPPO + cost advantage | yes | team success vs per-agent safety |
| uncertainty | not used | not used | yes | h ensemble/OOD/calibration |
| memory/recurrence | not used | not used | yes | temporal risk state, only if justified |
| action-conditioned safety | not used | not used | **blocked** | Q_cost(s,a)/CBF/QP/filter needs explicit approval |

There is also a non-learnable evaluation layer: deterministic/stochastic eval,
episode-level safe/collision, warning rate, per-agent collision, reach time,
checkpoint retention, and multi-seed reporting.  These are important for
claims but are not training signals.

## 4. From-scratch answer

- **Evaluate an existing checkpoint with more safety metrics:** no retraining.
- **Add stochastic evaluation:** no retraining; evaluator option only.
- **Periodic monitoring and intermediate checkpoints:** rerun training to
  produce the checkpoints, but evaluation itself does not change the gradient.
- **New h data/target or new h architecture:** yes, retrain h from scratch on
  V6-owned data, then retrain the actor from scratch with the new h.
- **New learnable actor/critic/uncertainty/team-risk module:** yes, train it
  from scratch with V6 data; do not initialize from Ant/V5.
- **Current gate actor:** already trained from scratch; h is the frozen V6 h.

## 5. Recommended order

1. (Done) Finish the current 3-seed gate evaluation; do not add mechanisms
   mid-run.
2. Re-evaluate the same checkpoints with 100 stochastic episodes.
3. Add monitor-only safety eval and intermediate checkpoints for the next run.
4. If the gate still trades too much success for safety, change the coupling
   (terminal-only/soft/team/hdot), not the gate threshold alone.
5. If the gate signal is saturated or too weak, upgrade h data (on-policy V6
   rollouts) and target (future-min margin), then retrain h and actor from
   scratch.
6. Only after nav gate/retention is stable, consider task channel, team risk,
   uncertainty, or action-conditioned safety (the last needs approval).
