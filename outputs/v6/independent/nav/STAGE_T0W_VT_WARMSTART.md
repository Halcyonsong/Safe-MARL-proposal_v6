# V6 navigation T0w: actor1 + frozen finite-horizon V_T 4k warm-start

Status: **pre-registered 2026-10-26; launching as a diagnostic only.**  The
safety line is frozen at the promoted 3-seed Stage-B result.  This arm tests
whether the actor1 policy can adapt to the proposal's task channel (`A_T`,
`grad_s V_T`) without task collapse, before spending on a from-scratch
task-channel arm.

This is **not** a main V6 claim, not a from-scratch result, and not a causal
task-channel result.  The user explicitly chose to defer the matched
no-V_T 4k control until this warm-start shows a benefit.

## 1. Question

> Can actor1, warm-started for 4,096 additional steps with a frozen
> finite-horizon task value `V_T`, keep its safety level and show a preliminary
> task benefit?

If yes, run a matched `actor1 + 4k, task block zero` control to separate the
task channel from the extra training.  If no, stop the task-channel line
before any from-scratch run.

## 2. V_T definition

`V_T` is a separate graph network trained offline on actor1 seed-0 rollouts.
It is **not** a critic and has no shared parameters with h or the PPO critic.

Target: finite-horizon ungated task return

```text
G_t^(K) = sum_{k=0}^{K-1} gamma^k * r_task_ungated(t+k)
r_task_ungated = 20 * progress + 5 * reach + 20 * terminal_reach
```

- `K = 64`, `gamma = 0.99`;
- no h gate;
- no safety cost;
- no official h, QP, expert action, or V5/V4 artifact.

The actor receives the existing 5-D task feature:

```text
[A_T, grad_s V_T (4)]
```

`A_T` is an EMA-normalized relative task value, not a PPO advantage.  The PPO
advantage still comes from the main critic, which is unchanged.

## 3. Data

- behavior policy: `outputs/v6/independent/nav_stageb_repair_16k/seed0/policy_final.pth`
- h: `outputs/v6/independent/h_stageb_repair/v6_h_stageb_repair.pkl` (frozen)
- stochastic actor1 rollouts, 100 episodes, 256 steps, 8 agents;
- environment seeds `30000..30099`;
- split by episode: 80 train / 10 validation / 10 test;
- the formal evaluation seeds `10000..10099` are not used for V_T training or
  selection.

## 4. V_T training

- separate `LocalGraphNet`, hidden 128, 2 graph layers, linear output;
- input: the 92-D base observation only; no h, no cost;
- MSE regression to `G_t^(K)`;
- AdamW, lr `3e-4`, weight decay `1e-4`, batch 256, 40 epochs,
  early stopping on validation MSE, gradient clip 1.0;
- save the best validation checkpoint;
- compute the per-agent EMA mean/var from the train+validation predictions and
  freeze it with the network.

No hyperparameter sweep.  The target, horizon, and optimizer are fixed before
training.

**V_T quality gate before the warm-start:** validation explained variance
`>= 0.20` and Pearson correlation `>= 0.40` on the held-out test split.  If
either fails, report a V_T quality failure and fix V_T offline once; do not
interpret the actor result as a task-channel test.

## 5. Warm-start

- initialize from actor1 seed-0 final (16,384 steps);
- continue exactly 4,096 steps (16,384 -> 20,480);
- frozen h1, frozen gate (`low=0`, `high=0.26726993918418884`), frozen cost,
  same reward, same optimizer/seed as Stage-B;
- enable `use_task_scalar=true`, `use_task_grad=true`;
- `task_grad_scale=0.5`, `task_grad_start_steps=1024` (global step, active);
- zero-initialize the actor's first-layer columns for the 5-D task block, so
  the initial policy is exactly actor1;
- V_T and its EMA are frozen; the task network is not updated during the
  warm-start.

Monitor evaluation: 20 episodes at 18,432 and 20,480, seed 10000.  Final
evaluation: 100 deterministic episodes, seeds `10000..10099`, `policy_final`
at 20,480, gate off.

## 6. Decision rule

Primary comparison: actor1 seed-0 formal 100ep:

| metric | actor1 seed 0 |
|---|---:|
| safe_episode | 0.840 |
| official_safety | 0.97375 |
| official_success | 0.42750 |
| official_reach | 0.43500 |
| episode_return | 60.91 |

Define:

- **task_benefit**: `official_success >= 0.4775` (+0.05) **or**
  `episode_return >= 65.91` (+5.0);
- **safety_ok**: `safe_episode >= 0.80` and
  `official_safety >= 0.97`.

`has_benefit = task_benefit and safety_ok`.

- **has_benefit = true:** run the matched no-V_T 4k control.  Only if the
  warm-start also beats that control on task without a safety loss is a
  from-scratch task-channel arm justified.
- **has_benefit = false:** stop the task-channel line.  Do not run the control
  and do not start the from-scratch arm.

### 6b. Matched no-V_T 4k control (rule fixed before the control result)

The control reuses actor1 seed-0, the same 4,096 extra steps (16,384 ->
20,480), and the same h1/gate/cost/reward/optimizer/seed and 100ep eval
(seeds `10000..10099`).  The 5-D task feature is held at zero and no V_T
checkpoint is loaded; nothing else changes.

- **control_valid:** control `safe_episode >= 0.80` and
  `official_safety >= 0.97`.  If it fails, the comparison is degenerate:
  report an invalid control and make no task-channel claim.
- **task_gain:** warm-start `official_success >= control + 0.05` **or**
  `episode_return >= control + 5.0`.
- **safety_not_worse:** warm-start `safe_episode >= control - 0.03` and
  `official_safety >= control - 0.01`.
- **vt_channel_supported = control_valid and task_gain and
  safety_not_worse.**

If supported, the frozen V_T channel beats extra training alone in this
one-seed diagnostic; a from-scratch arm still needs a separate
pre-registration.  If not supported, stop the task-channel line.

A 20-episode monitor collapse (`safe < 0.70` or `success < 0.30`) stops the
run early; the formal 100ep is still the decision if the monitor is not
catastrophic.

## 7. Caveats

- The warm-start has 4,096 extra training steps; any change is confounded with
  extra training until the no-V_T control is run.
- V_T is trained on actor1 seed-0 states; the warm-start stays close to that
  distribution, but the from-scratch arm may be OOD.
- One seed.  This is a diagnostic screen, not a 3-seed claim.
- No gate/cost/task-channel sweep; no QP or action filter.
- The task channel is not a runtime safety mechanism; h remains safety-only.

## 8. Result (warm-start, 100ep, final checkpoint 20,480, gate off)

V_T quality gate (held-out splits, 20,480 rows each):

| metric | threshold | validation | test |
|---|---:|---:|---:|
| explained variance | >= 0.20 | **0.2270** | **0.2446** |
| Pearson | >= 0.40 | **0.4860** | **0.4991** |

V_T passes the gate with limited margin on validation explained variance
(best epoch 5, early stop at 15); the actor result is therefore interpretable
as a task-channel diagnostic.

| metric | actor1 seed 0 | T0w warm-start | delta |
|---|---:|---:|---:|
| safe_episode | 0.840 | **0.890** | +0.050 |
| collision_episode | 0.160 | 0.110 | -0.050 |
| official_safety | 0.97375 | **0.98250** | +0.00875 |
| official_success | 0.42750 | **0.67875** | +0.25125 |
| official_reach | 0.43500 | **0.68500** | +0.25000 |
| episode_return | 60.91 | **161.88** | +100.97 |
| action_abs | 0.2769 | 0.2473 | -0.0296 |
| cbf_violation | 0.04464 | 0.04289 | -0.00175 |

- `task_benefit = true` (success `0.67875 >= 0.4775`, return
  `161.88 >= 65.91`);
- `safety_ok = true` (safe episode `0.890 >= 0.80`, official safety
  `0.98250 >= 0.97`);
- `has_benefit = true` -> the matched no-V_T 4k control was then run
  (section 9).

The 20ep monitor at 18,432 was near baseline (return 65.06, success 0.44375);
the 20,480 monitor jumped to return 179.69 / success 0.7375.  The formal
100ep is the decision.  The gain is confounded with the extra 4,096 training
steps until the control finishes; this remains one seed and is not a main
claim.

## 9. Result (matched no-V_T 4k control, 100ep)

The control finished at 20,480 steps with the same actor1 seed-0 init, same
h1/gate/cost/reward/optimizer/seed, and the 5-D task feature held at zero
(no V_T checkpoint).  All numbers are deterministic 100ep, eval seeds
`10000..10099`.

| metric | actor1 init | control 4k | warm-start 4k | warm - control |
|---|---:|---:|---:|---:|
| safe_episode | 0.840 | **0.930** | 0.890 | **-0.040** |
| collision_episode | 0.160 | 0.070 | 0.110 | +0.040 |
| official_safety | 0.97375 | **0.99000** | 0.98250 | -0.00750 |
| official_success | 0.42750 | 0.38000 | **0.67875** | **+0.29875** |
| official_reach | 0.43500 | 0.38125 | **0.68500** | **+0.30375** |
| episode_return | 60.91 | 69.67 | **161.88** | **+92.20** |
| action_abs | 0.2769 | 0.2326 | 0.2473 | +0.0148 |
| cbf_violation | 0.04464 | 0.03027 | 0.04289 | +0.01262 |

Pre-registered decision:

- `control_valid = true` (safe `0.930 >= 0.80`, official safety
  `0.99000 >= 0.97`);
- `task_gain = true` (success `+0.29875 >= 0.05`, return `+92.20 >= 5.0`);
- `safety_not_worse = false`: warm-start safe episode `0.890` is below
  `0.930 - 0.03 = 0.900`, i.e. **one episode out of 100**.  Official safety
  passes (`0.98250 >= 0.980`);
- `vt_channel_supported = false` under the pre-registered safety-first rule
  -> `run_from_scratch_arm = false` for now.

### 9b. Task-first interpretation (post-hoc, not a pre-registered pass)

The strict safety-first rule is not met.  If the project objective is
task-first, T0w is a **task-heavy Pareto point**: success `+29.9 pp` and
return `+92.2` at a small, statistically unresolved safety discount
(safe `-4.0 pp`, official safety `-0.75 pp`; the scene-level difference is
one episode and the Wilson intervals overlap).  This is a valid trade-off
statement, not a safety-non-inferiority pass.

Decision under the task-first objective: keep T0w as a promising pilot and
require a 3-seed from-scratch `h1 + gate + V_T` replication before any main
claim.  Decision under the pre-registered safety-first objective: stop the
strict line; the from-scratch arm is not authorized by the old rule.  The two
statements are consistent: the data buy task at a small safety cost, and the
old rule was designed to reject exactly that trade-off.

The honest claim is "large task gain with a small, statistically unresolved
scene-safety discount," not "V_T is unsafe" and not "the channel passes
safety non-inferiority."

Context for reading the table:

- Both arms improve scene safety over the actor1 init (control `0.93`,
  warm-start `0.89`, init `0.84`); the control improves it more.
- The control slightly regresses task versus the init (success `0.38` vs
  `0.4275`, return `69.67` vs `60.91`), so the warm-start's task gain is not
  explained by the extra 4,096 steps.
- The 20ep control monitor (`safe 0.95`) was optimistic; the 100ep value is
  `0.93`.  Do not rank on 20ep safety.

## 10. Artifacts

- data: `outputs/v6/independent/t0w_vt/data/actor1_seed0_100ep.npz`
- V_T checkpoint: `outputs/v6/independent/t0w_vt/v6_task_value.pkl`
- warm-start: `outputs/v6/independent/nav_t0w_vt_warmstart_4k/seed0/`
- matched no-V_T control: `outputs/v6/independent/nav_t0w_control_4k/seed0/`
- collector: `nav/scripts/collect_v6_task_value_data.py`
- V_T trainer: `nav/scripts/train_v6_task_value.py`
- runner: `nav/scripts/run_v6_nav_t0w_vt_warmstart_4k.sh`
- control runner: `nav/scripts/run_v6_nav_t0w_control_4k.sh`
- control config: `nav/configs/v6/v6_nav_t0w_control_4k.yaml`
- summarizer: `nav/scripts/summarize_v6_nav_t0w.py`
- control summarizer: `nav/scripts/summarize_v6_nav_t0w_control.py`

## 11. Status

- [x] Protocol pre-registered.
- [x] V_T data collection.
- [x] V_T offline training and validation (quality gate passed).
- [x] Warm-start 4k training.
- [x] 100ep evaluation and diagnostic decision (`has_benefit = true`).
- [x] Matched no-V_T 4k control and 100ep evaluation.
- [x] Pre-registered control decision: `control_valid = true`,
      `task_gain = true`, `safety_not_worse = false`,
      `vt_channel_supported = false`.
- [x] Strict safety-first line stopped; from-scratch arm **not** justified
      under the old rule and not launched.
- [ ] Optional task-first follow-up: 3-seed from-scratch `h1 + gate + V_T`
      replication, requiring a new pre-registration and approval.
