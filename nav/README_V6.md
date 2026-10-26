# V6 navigation: standalone learned h(s) + MAPPO

Canonical status: `outputs/v6/STATUS.md`.  Consolidated experiment record:
`outputs/v6/V6_COMPLETE_RECORD.md`.  Navigation records index:
`outputs/v6/independent/nav/README.md`.

Current promoted navigation result: the repaired Stage-B predictive-h gate
passed the pre-registered 3-seed guardrails (seeds 0/1/2, 16,384 steps,
100 deterministic episodes each): `safe_episode=0.693 ± 0.168`,
`official_safety=0.9417 ± 0.0378`, `official_success=0.4100 ± 0.0583`,
`episode_return=66.85 ± 5.26`.  Matched h-only is `0.450/0.8783/0.4067/61.73`.
See `outputs/v6/independent/nav/STAGE_B_REPAIR_MULTISEED.md` and
`outputs/v6/independent/nav/BASELINE_COMPARISON.md`.

V6 is an **independent** safe multi-agent navigation pipeline.  It replaces
the official h / expert `u_ref` / QP design with a safety function `h(s)` that
V6 collects data for and trains from scratch.  The main experiment is:

```text
V6 random rollouts + near-boundary augmentation
        -> V6 h(s) from scratch (safety-only)
        -> frozen h(s) + grad_s h(s) as actor features
        -> MAPPO/PPO actor from scratch (task reward)
        -> evaluate h-on vs h-off
```

There is **no official h checkpoint, no `cbf.pkl`, no QP/u_ref/expert label,
no V5/V4 weight or data, and no warm-start from V5** in the main path.  The
official navigation scene (dynamics, observations, collision/unsafe masks) is
used as the task definition; generic code/design such as PPO/MAPPO and a graph
encoder may be reused.  The actor/loss code is copied into the V6-owned
`nav/v6/policy.py`; the main V6 training path does not runtime-import V5
`algos.safe_policy`.

## Independence rules

Allowed:

- the official navigation scene and its collision/unsafe labels;
- generic network, PPO/MAPPO, GAE, reward, and training-code design;
- V6 self-collected data and V6 self-trained checkpoints;
- V6 self-warm-start produced by a from-scratch V6 stage (optional, later).

Forbidden:

- official h / `cbf.pkl` / any pretrained h weights;
- official h values, gradients, teacher labels, or distillation targets;
- QP labels/actions, `u_ref`, expert actions or projection;
- V5/V4 weights, data, checkpoints, or runtime warm-start;
- the current `h_mech` h checkpoint and `data_v5h0` data;
- margins/offsets calibrated by matching official h.

Run the static guard before a new experiment:

```bash
python nav/scripts/check_v6_independence.py
```

## Main protocol

Main config: `nav/configs/v6/v6_nav_independent.yaml`.  The commands below
assume the project root as the working directory; the wrapper computes its own
paths.

Stage A: collect V6 h data.

```bash
# From the project root.
python nav/scripts/collect_v6_h_data.py \
  --config configs/v6/v6_nav_independent.yaml \
  --out outputs/v6/independent/data \
  --episodes 100 --steps 256 --seed 2026
```

Do **not** pass `--policy-checkpoint` for the independent default.  The random
behavior policy is sufficient for the first pilot; a V6 self-trained behavior
policy can be added later, but V5 checkpoints/data are forbidden.

Stage B: near-boundary augmentation and merge.

```bash
python nav/scripts/augment_v6_h_data.py \
  --config configs/v6/v6_nav_independent.yaml \
  --data-dir outputs/v6/independent/data \
  --out outputs/v6/independent/data/train_aug.npz \
  --scenes 8000 --seed 777

python nav/scripts/merge_v6_h_data.py \
  --data-dir outputs/v6/independent/data \
  --out outputs/v6/independent/data/train_merged.npz
```

Stage C: train V6 h from scratch.

```bash
python nav/scripts/train_v6_official_h.py \
  --config configs/v6/v6_nav_independent.yaml \
  --train-data outputs/v6/independent/data/train_merged.npz \
  --val-data outputs/v6/independent/data/val_data.npz \
  --test-data outputs/v6/independent/data/test_data.npz \
  --out outputs/v6/independent/h \
  --tag v6_h
```

The checkpoint is `outputs/v6/independent/h/v6_h.pkl`.  Its threshold/offset is
calibrated on V6 validation data only.  If you sweep `--safe-margin` or
`hdot_eps`, use V6 validation only; do not calibrate a margin by comparing to
official h or by an official-h h-swap.

The script/module names `train_v6_official_h.py` and `v6/official_h.py` are
legacy compatibility labels.  They train/load the V6-owned h only; no official
h/`cbf.pkl` is read.

Stage D: from-scratch actor, h-on and h-off.

```bash
# h-on (main arm)
python nav/scripts/train_v6.py \
  --config configs/v6/v6_nav_independent.yaml \
  --h-checkpoint outputs/v6/independent/h/v6_h.pkl \
  --seed 0 --total-steps 8192 --eval-episodes 20 \
  --out outputs/v6/independent/runs/seed0_h

# h-off control, same seed/steps/eval seeds
python nav/scripts/train_v6.py \
  --config configs/v6/v6_nav_independent.yaml \
  --h-checkpoint outputs/v6/independent/h/v6_h.pkl \
  --no-use-h-features \
  --seed 0 --total-steps 8192 --eval-episodes 20 \
  --out outputs/v6/independent/runs/seed0_no_h
```

The actor is randomly initialized in both arms.  `--init-policy` and V5
checkpoints are not part of the main path.

Stage D2: Ant-style predictive-h task gate (exploration).

```bash
python nav/scripts/train_v6.py \
  --config configs/v6/v6_nav_h_gate.yaml \
  --seed 0 --eval-seed 10000 \
  --out outputs/v6/independent/nav_h_gate_16k/seed0
```

The gate uses the frozen V6 graph h at the post-transition state:

```text
gate_i = clip((h_next_i - low) / (high - low), 0, 1)
```

Only positive task progress/reach reward is multiplied by the gate.  The
official collision cost is unchanged, and the gate is disabled during
evaluation.  The default V6 nav calibration is `low=0.0`, `high=0.13`
(healthy h plateau is about 0.138); this is not the Ant threshold.  Set
`--task-gate-high 0` to disable the gate.  See
`outputs/v6/independent/nav/PREDICTIVE_GATE.md`.

Stage E: evaluate with the matching V6 h.

```bash
python nav/scripts/evaluate_v6.py \
  --checkpoint outputs/v6/independent/runs/seed0_h/policy_final.pth \
  --h-checkpoint outputs/v6/independent/h/v6_h.pkl \
  --num-episodes 20 --seed 10000 \
  --out outputs/v6/independent/runs/seed0_h/eval_20ep.json
```

The no-h arm uses its own saved config (`h.use_features: false`) and the same
h checkpoint only because the trainer/evaluator still constructs the provider;
the features are zeroed by the config.

Or run the wrapper:

```bash
# Single-seed pilot (default)
bash nav/scripts/run_v6_protocol.sh

# Formal comparison after the pilot shows a clear direction
SEEDS="0 1 2" TRAIN_STEPS=16384 EPISODES=100 \
  bash nav/scripts/run_v6_protocol.sh
```

### Reproducible 3-seed baseline

The formal h-only baseline uses the final actor checkpoint, a fixed evaluation
seed range, and a manifest with V6-owned h/data/config hashes:

```bash
# Uses V6 h only; no task channel, official h, QP, or V5 artifact.
REUSE_SEED0=1 SEEDS="0 1 2" TRAIN_STEPS=16384 \
  bash nav/scripts/run_v6_formal_3seed.sh
```

Outputs are under `outputs/v6/independent/formal_3seed/h_only/`, including
`manifest.json`, per-seed `eval_100ep_final.json`, and
`aggregate_100ep.json` / `.md`.  Set `REUSE_SEED0=0` to retrain seed 0 as well.
Do not select a checkpoint by the 20-episode training eval; the formal report
uses the pre-specified final checkpoint.

### Safety optimization screen

`nav/scripts/run_v6_safety_screen.sh` tests safety-only changes with frozen V6 h
and no task channel:

```bash
VARIANTS="cost cost20 barrier cbf" SEED=0 \
  bash nav/scripts/run_v6_safety_screen.sh
```

The screen uses 20 episodes per variant.  Promote a variant to 100 episodes
only if `safe_episode` improves without a large success collapse; then run the
same variant over 3 seeds.  `barrier` uses a dense cost from V6 `h`; `cbf` uses
a soft action-space CBF loss.  Neither is a QP/projection and neither updates
or gates h.

## Pilot decision rule

Use one seed first:

- primary: collision episode rate and safe episode rate;
- guardrail: success/reach must not collapse;
- compare h-on vs h-off with the same seed, steps, and eval seeds;
- a single seed is a go/no-go pilot, not a final claim.

If the direction is clear, run 3 seeds × 100 episodes.  The repaired Stage-B
arm completed this: mean safety `0.9417`, safe episode `0.693`, success `0.410`,
which is just below the archived V5 target on safety/safe-episode but does not
collapse task.  The safety-focused target is the archived V5 16k
MAPPO+official-h reference: roughly environment safety >= 0.95 and safe
episode >= 0.70, with success not collapsing.
The historical JSON field `official_safety` is the environment safety metric;
it is not official h.  V5 numbers are recorded targets only; do not run official
h to reproduce them.

A read-only h2/actor2 diagnostic followed the promoted Stage-B result.  On the
fixed final actor1 checkpoints (3 seeds x 100 episodes), all 155 collision
onsets occurred with `h_cur < 0` and the diagnostic gate closed, so the
impact-state label is **actor/action-limited**.  But h1 was trained on h-only
seed0 and Stage-A seed0 rollouts, not final actor1; the risk-region evidence
shows the resulting temporal gap (17-32 steps before collision: h mean
`+0.242` vs target `-0.222`; future-collision MAE `0.394`).  Overall diagnosis:
**mixed**.  Stage C0a (seed-0 h2-only retrain on actor1 data) was run and
**failed**: 17-32-step risk-region MAE `0.826` vs h1 `0.709` (+16.5% worse),
while global MAE improved (`0.084` vs `0.092`).  C0b was not run and the
original C1 was not authorized.  A post-hoc seed-0 h2+actor1 h-swap was slightly
better than h1 (safe `0.90` vs `0.84`, official safety `0.985` vs `0.974`),
but it is one seed, not significant, and off-policy; it does not change the
C0a decision.  The separately approved **C1a seed-0 actor2 pilot** with h2
seed 0 (from scratch, same cost/steps) was then run and **failed all five
pre-registered checks**: safe `0.510` vs `0.840`, official safety `0.89375`
vs `0.97375`, success `0.33125` vs `0.42750`, reach `0.36250` vs `0.43500`,
return `52.22` vs `60.91`; collisions `0.16 -> 0.49`.  The training gate was
almost always open (`task_gate_mean≈0.99`), so this is not gate suppression.
The from-scratch h2/actor2 branch is stopped and C1b is not authorized.  A
**C1w 4k warm-start of actor1 with h2** (no h1 control) was run and
**failed**: 100ep safe `0.840`, official safety `0.97625`, success `0.20250`,
reach `0.20625`, return `28.85`.  It matches actor1 on safe episodes and
collisions, stays below the no-training h2 swap on safety, and collapses the
task.  `promising_adaptation = false`; the h1-continued control was not run
and the h2 warm-start branch is stopped.  It is not a C1a pass.  A new
evaluation-only **C1s** replication (h2 seed 0 on fixed actor1 seeds 1/2,
100ep each) is complete and **`consistent_positive`**: mean `Δsafe +0.030`
(exactly at the pre-registered `+0.03` boundary),
`Δofficial_safety +0.00875`, `Δsuccess +0.01375`, `Δreach +0.00813`,
`Δreturn -0.05`; all eight checks hold.  The gain is thin and seed-1-driven;
the swap is off-policy and h2 seed 0 is distribution-shifted for seeds 1/2.
It does not establish h2 seed reproducibility or a from-scratch benefit and
is not a main claim.  The h-only data/target axis is stopped.  The next
controlled arm after C1w/C1s, if needed, must be safety coupling or action
conditioning (one axis only), because the impact-time failure is
actor/action-limited.  Do not combine it with a gate-scope, cost, or
task-channel change.  Result records:
`../outputs/v6/independent/nav/STAGE_C0_H2_TEMPORAL.md`,
`../outputs/v6/independent/nav/STAGE_C1A_H2_ACTOR2_PILOT.md`,
`../outputs/v6/independent/nav/STAGE_C1W_H2_ACTOR1_4K.md`, and
`../outputs/v6/independent/nav/STAGE_C1S_H2_ACTOR_SEED_SWAP.md`; diagnostic:
`../outputs/v6/independent/nav/H2_ACTOR2_DIAGNOSTIC.md`.

`h-swap` is only a diagnostic (same V6 actor, V6 h vs zero).  It is not a
substitute for PPO/MAPPO h-on/h-off training evidence, and it must never use
official h as the reference.

## Task channel and Ant

The h-only from-scratch result is established (3-seed), and the repaired
Stage-B gate now adds a promoted 3-seed safety result.  The task channel
(`V_T`, `A_T`, `grad_s V_T`) is being screened by **T0w**: actor1 + frozen
finite-horizon `V_T` 4k warm-start, h1/gate/cost unchanged.  This is a
diagnostic only; the matched no-V_T control and any from-scratch task-channel
arm are gated on a task benefit.  Do not combine it with a gate-scope or
cost-schedule change.  `V_T` is trained from V6-collected, ungated task
returns, independent of h and safety gating.  Record:
`../outputs/v6/independent/nav/STAGE_T0W_VT_WARMSTART.md`.

Ant is a second scenario, not a copy of navigation results.  Reuse the generic
h/actor design if useful, but collect Ant data and train Ant h/actor
independently.  Navigation and Ant do not need identical h or MAPPO settings.
The Ant predictive-h gate is already validated at 3 seeds x 10M.

## Legacy artifacts

The old V5-warm-start runs, `data_v5h0`, `h_mech` checkpoints, and all results
under those trees are **legacy diagnostics**.  They are not V6 main evidence.
Archive them under `archive/v5_coupled/` when cleaning the workspace; do not
load them in the independent protocol.
