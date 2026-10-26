# V6 Ant: from-scratch MAPPO with a 9-D safety h(s)

Consolidated experiment record: `outputs/v6/V6_COMPLETE_RECORD.md`.

This is the V6-clean Ant branch.  It reuses only the official Safe MAMuJoCo
simulator, task reward and official safety cost.  It does **not** import the
legacy V5 Ant safety module, does not load V5 actors/h/checkpoints, and does
not use QP/`u_ref`/expert actions.  The h(s) is V6-collected, V6-trained and
frozen before actor training.

## Current status (2026-10-24)

The static/dynamic-h actor-only and h-barrier mechanisms are terminated.  The
current-h forward gate was also partial: it improved two seeds but failed
seed 1 deterministically.  The **predictive-h + safety-gated forward**
follow-up now passes the three-seed 2M gate:

- h retains the old future-unsafe `0` rows and predicts the 32-step minimum
  future margin; offline test BA `0.9145`, unsafe recall `0.8793`, false alarm
  `0.0364`, AUROC `0.9511`, gradient cosine `0.9225`;
- forward gate `clip(h_next/0.05, 0, 1)`, training only;
- deterministic 100ep mean: forward `756.60`, safe `0.977`, wall `0.023`,
  cost `0.00232` vs matched cost-only `905.75 / 0.573 / 0.420 / 0.0199`;
- stochastic 100ep mean: forward `472.22`, safe `0.317`, cost `0.0264` vs
  `525.38 / 0.067 / 0.0987`.

The 10M seed-0 retention check also passes: deterministic forward `3723.09`,
safe `0.97`, cost `0.00176`; cost-only 10M is `4055.37 / 0.10 / 0.0222`.
Stochastic is `2923.73 / 0.78 / 0.00419` versus `3067.39 / 0.04 / 0.0508`.
See `outputs/v6/independent/ant/PREDICTIVE_H_GATE_2M.md` and
`outputs/v6/independent/ant/RETENTION_10M.md`.  The remaining check is 10M
seeds 1/2 (or an explicitly accepted single-seed nav gate).  The matched
cost-only 2M control remains `forward=840.11`, `safe=0.90`,
`cost_fraction=0.0044` at seed 0.

## 1. Safety state, h and direction

The safety state is the 9-D torso geometry

```text
[d_left, d_right, v_n_left, v_n_right, z, z_rot, vx, vy, vz]
```

The four threshold-aligned margins are

```text
h_wall    = min(d_left, d_right) - 1.8       # official wall cost
h_floor   = z - 0.2                          # official fall/height cost
h_ceil    = 1.0 - z
h_upright = 0.2 * (z_rot + 0.7)             # official topple cost
h(s)      = tanh(min(h_wall, h_floor, h_ceil, h_upright) / scale)

# Dynamic margins used by the current h data (the old h-barrier is terminated).
h_wall_dyn  = h_wall  - kappa * max(v_n, 0.0)
h_floor_dyn = h_floor - kappa_z * max(-vz, 0.0)
```

The h network is a small MLP `9 -> 128 -> 128 -> 1`.  It is safety-only:
labels come from the official cost/terminal signal, never from task return.
The actor receives `h` and the 9-D state-space direction `grad_s h`.  The
gradient is a **feature**, not an action correction.  There is no QP, CBF
projection, `u_ref`, or MAPPO-Lagrangian multiplier; the actor learns the
task/safety tradeoff through the fixed cost penalty.

The training loss includes a gradient-alignment term between `grad_s h` and
the finite-difference gradient of the geometric margin.  This is a
safety-only regularizer; it gives the actor a meaningful direction instead of
an arbitrary scalar-only h.  The h summary reports `grad_cosine` and
`grad_mse` on validation/test, so the direction quality is checked separately
from the safe/unsafe classification metrics.

## 2. Forward-window labels (borrowed from navigation V6)

The default Ant horizon is 32 environment steps (`dt=0.05`, about 1.6 s).
There are two target modes:

- `current` (old static/dynamic h): current official cost gives `-1`, no
  event in the window gives `+1`, and near-future-unsafe rows are `0` and were
  dropped by the old collector;
- `horizon` (predictive h): every row is retained.  A current or future unsafe
  event gives `label=-1` (the old `0` rows become `-1`), a far-safe window
  gives `+1`, and the regression target is
  `tanh(min_{k<=32} margin_{t+k} / scale)`.

The terminal fall state is marked unsafe, so the preceding window is not
mistaken for safe.  Task return, forward progress, control cost and the
survive reward are **not** used as h labels.  The predictive config uses
`target_on_all=true` and `grad_on_all=true`; the target gradient remains the
current geometric-margin direction.

## 3. What is safety and what is task?

Safety optimization (official / h):

* wall clearance (`cost_obj`, `h_wall`);
* fall/topple (`cost_done`, `h_floor`, `h_ceil`, `h_upright`);
* forward-window unsafe risk and the optional dense h barrier;
* `topple_episode`, `wall_episode`, `unsafe_episode`, `safe_episode`.

Task optimization (actor only):

* official forward reward;
* control cost and contact cost (they are part of the official task reward);
* survive bonus, episode return, forward x progress;
* episode length and survival-to-time-limit are reported as task/continuation
  diagnostics.  A fall itself is safety (`cost_done`), but "staying alive for
  the 1000-step limit" is not an h label.

This separation keeps h safety-only and prevents the task return from leaking
into the safety supervision.

## 4. Commands

The Ant simulator lives in a separate `mappo_lag` Python environment and
MuJoCo 2.1.0.  Activate that environment (or set `PYTHON=/path/to/python`)
and point `SAFE_MAMUJOCO_ROOT` at the upstream checkout when it is not the
sibling `../Safe-MARL-proposal_v2`.

End-to-end smoke:

```bash
bash nav/scripts/run_v6_ant_smoke.sh
```

V6-only data/h training plus the three from-scratch arms:

```bash
# default: task-only baseline first; seed 0; 10M steps; 8 envs
bash nav/scripts/run_v6_ant_protocol.sh
# after the task baseline is healthy:
ARMS="cost_only h_only" bash nav/scripts/run_v6_ant_protocol.sh
```

The protocol runner now runs the fresh-rollout h validation gate after h
training and aborts before MAPPO if it fails.  Set `SKIP_H_VALIDATION=1` only
when reusing an already accepted h explicitly.

Manual pieces:

```bash
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin
export PYTHONPATH=$PWD/nav

# 1) V6 random/perturbed data + geometric augmentation; labels are safety-only
python nav/scripts/collect_v6_ant_h_data.py --out outputs/v6/independent/ant/h_data

# 2) train/validate/freeze 9-D h
python nav/scripts/train_v6_ant_h.py --data-dir outputs/v6/independent/ant/h_data \
  --out outputs/v6/independent/ant/h

# 3) from-scratch MAPPO on the corrected V5-matched base
python nav/scripts/train_v6_ant_mappo.py \
  --config configs/v6/v6_ant_h_p01_10m.yaml \
  --h-checkpoint outputs/v6/independent/ant/h/ant_h.pkl \
  --out outputs/v6/independent/ant/mappo_h_p01_10m/seed0

# 4) predictive h: retain future-unsafe rows, use a 32-step horizon target
python nav/scripts/collect_v6_ant_h_data.py \
  --out outputs/v6/independent/ant/h_data_predictive \
  --action-mode policy \
  --policy-checkpoint outputs/v6/independent/ant/mappo_cost_ramp10_10m/seed0/policy_002015232.pth \
  --policy-stochastic --target-mode horizon --horizon 32 \
  --kappa 0.5 --kappa-z 0.2 --episodes 120 --aug-rows 30000
python nav/scripts/train_v6_ant_h.py --config configs/v6/v6_ant_h_predictive.yaml

# 5) predictive-h + forward gate (training gate only; eval is official)
python nav/scripts/train_v6_ant_mappo.py \
  --config configs/v6/v6_ant_h_pred_gate_2m.yaml \
  --out outputs/v6/independent/ant/mappo_h_pred_gate_2m/seed0 --seed 0
```

The corrected base configs are `v6_ant_task_base.yaml`,
`v6_ant_cost_base.yaml` and `v6_ant_h_base.yaml`.  They use the normalized
31-D centralized state, ValueNorm, a V5-style 64-wide ReLU/LayerNorm actor and
critic, a Gaussian policy with std≈0.365, separate actor/critic learning rates,
5 PPO epochs, 40-style minibatches and 10M steps.

`h_only` feeds h+grad; `cost_only` keeps the fixed cost penalty without h
features; `task_only` has neither.  The no-h arms use a 31-D actor input; the h
arm uses 41-D.  The main cost/h pilot configs use `cost_penalty=0.1`
(`v6_ant_cost_p01.yaml`, `v6_ant_h_p01.yaml`): penalty 10 and penalty 1 both
collapsed to a safe standstill at 0.5–1M, so they are not a useful task/safety
tradeoff.  The p0.1 pilot must show forward progress; only then check whether h
reduces wall/fall cost relative to cost-only.

Evaluate the deterministic mean policy and, separately, the stochastic policy:

```bash
python nav/scripts/evaluate_v6_ant_mappo.py \
  --checkpoint outputs/v6/independent/ant/mappo_task_base/seed0/policy_final.pth \
  --episodes 100 --seed 20000
python nav/scripts/evaluate_v6_ant_mappo.py \
  --checkpoint outputs/v6/independent/ant/mappo_task_base/seed0/policy_final.pth \
  --episodes 100 --seed 20000 --stochastic
```

### h acceptance gate before h-only

The training split mixes rollout rows with geometry augmentation, so it is
not sufficient by itself.  Before starting the 1M h-only pilot, validate the
frozen h on fresh V6 rollouts that were not used for fitting:

```bash
python nav/scripts/validate_v6_ant_h.py \
  --h-checkpoint outputs/v6/independent/ant/h/ant_h.pkl \
  --out outputs/v6/independent/ant/h_validation \
  --episodes 50 --horizon 32
```

The script evaluates zero-action, small-random, training-random and
high-random rollouts, plus the forward-window labels and the state-gradient
direction.  The formal gate is stored in `validation.json` under `gate`:
pooled instantaneous BA >= 0.95, unsafe recall >= 0.90, false alarm <= 0.05,
forward BA/recall >= 0.90, gradient cosine >= 0.80, reset false alarm <= 0.05,
and every regime with at least 10 unsafe states must have BA/recall >= 0.85 and
false alarm <= 0.10.  The current V6 Ant h passes this gate; the result file is
`outputs/v6/independent/ant/h_validation/validation.json`.  The wall-specific
test split has unsafe recall 0.981; its false alarms are confined to states
within about 0.07 m of the wall threshold, while the false-alarm rate is
1.8% above 0.05 m and zero above 0.10 m.

## 5. Corrected baseline (2026-10-23)

The first 1M pilot was stopped after 100k–600k evaluations showed standstill
behavior in all three arms.  It was not a V5-equivalent pure-MAPPO baseline:
the V6 critic saw raw 29-D state instead of the normalized 31-D state, ValueNorm
was disabled, the actor/critic shared one small-gradient-clipped optimizer, the
actor had no feature/LayerNorm and used a tanh-squashed Gaussian, training used
one environment / 2048-step updates for only 1M steps, and the PPO update had a
log-probability broadcasting bug that made the policy ratio `(B, B)` instead of
per-sample.  Checkpoint loading also forgot to restore ReLU activations.  These
are fixed in the corrected path.

The V5 pure-MAPPO reference (10M, 100 fixed scenes) is `x_final=203.4`,
`return=4477.2`, `cost_rate=31.57%`, `survived=89%`; its training return is only
about 409 at 1M and 608 at 2M.  Therefore 1M is far too early even for the
reference.  The corrected V6 configs match the upstream recipe as closely as
the independent code allows; see `outputs/v6/independent/ant/PILOT_DIAGNOSIS.md`.
The old 1M results remain useful only as a negative diagnostic.  The corrected
task-only run at 1M already evaluates at `forward_return=328.9`,
`final_x=16.4`, `wall_episode=1.0` (deterministic) and `forward_return=288.8`,
`final_x=14.4`, `wall_episode=0.95` (stochastic), i.e. the expected
forward-but-unsafe pure-task pattern.  Do not confuse this with the safe
standing-still behavior of the old pilot or of the cost-penalized arms at 0.5M.

The matched 1M p0.1 pilot is more informative:

| arm | mode | forward_return | final_x | cost_fraction | safe_episode |
|---|---|---:|---:|---:|---:|
| task_only | deterministic | 328.9 | 16.4 | 0.185 | 0.0 |
| cost_only | deterministic | 222.9 | 11.1 | 0.566 | 0.1 |
| h_only | deterministic | 299.1 | 15.0 | 0.000 | 1.0 |
| task_only | stochastic | 288.8 | 14.4 | 0.366 | 0.0 |
| cost_only | stochastic | 168.0 | 8.4 | 0.209 | 0.0 |
| h_only | stochastic | 185.5 | 9.3 | 0.269 | 0.0 |

The deterministic h-only result is encouraging, but the stochastic h-only
result is not robustly safe.  Treat it as a 1M signal only; the matched 10M
runs are the formal comparison.

At 100 deterministic episodes the 1M checkpoints give `h_only forward=280.7,
final_x=14.0, safe_episode=0.91, cost_fraction=0.015`, versus task-only
`283.3 / 14.2 / 0.00 / 0.177` and cost-only `233.5 / 11.7 / 0.03 / 0.630`.
Under stochastic actions the same h checkpoint is `forward=191.2,
cost_fraction=0.253, safe_episode=0.03`, versus task-only
`255.0 / 0.288 / 0.00` and cost-only `158.0 / 0.226 / 0.05`.  The
deterministic safety gain is strong; the stochastic robustness is unresolved.


## 6. Current result and h decision (2026-10-24)

The corrected pure-task baseline is healthy at 10M and 100 deterministic
episodes: `forward_return=4496.85`, `mean_x=111.69`, `final_x=224.85`,
`cost_fraction=0.1323`, `wall_episode=0.99`, `survived=0.95` (stochastic:
`forward_return=3620.85`, `final_x=181.05`, `cost_fraction=0.1533`).  This
matches the V5 pure-task forward level; the V5 reference is 4477 / x=203.

The h branch was **terminated** after matched 100-episode tests.  At 2M the
cost-only scheduled-penalty control (0.1 -> 1.0 by 2M) gives
`forward_return=840.11`, `final_x=42.01`, `safe_episode=0.90`,
`cost_fraction=0.0044`, while the best h variant gives
`forward_return=957.99`, `final_x=47.91`, `safe_episode=0.38`,
`cost_fraction=0.0256`.  The h variant buys 14% more forward at 6x the cost and
much lower safety; the other h variants are worse.  Stochastic results show
the same ordering.  No 3-seed h run and no navigation task-joint run were
launched.  The offline h remains good; the failure is the closed-loop use of
h/grad by the MAPPO actor.  See
`outputs/v6/independent/ant/PILOT_DIAGNOSIS.md`.

The matched cost-only control completes at 10M with deterministic
`forward_return=4055.37`, `final_x=202.77`, `cost_fraction=0.0222`,
`safe_episode=0.10`, `survived=0.98`; stochastic `forward_return=3067.39`,
`cost_fraction=0.0508`, `safe_episode=0.04`.  It is the best task/safety trade
found and is a **no-h control**, so it does not rescue the h hypothesis.

## 7. Metrics

Primary safety: `safe_episode`, `unsafe_episode`, `wall_episode`,
`topple_episode`.  Task: `forward_return`, `mean_x`, `final_x`, plus
`episode_return`, `ctrl_return`, `survive_return`, `episode_length`,
`survived`.  `h_mean`, `h_min` and `h_neg_frac` are diagnostics.  Do not treat
`official_*` navigation fields as Ant episode safety; they do not exist in
this path.  Report both deterministic mean-policy evaluation and
`--stochastic` evaluation; a deterministic standing policy can look safe while
the stochastic training policy still falls.

Important: the official Ant reward includes a `survive_reward` of 1 per step,
so a standing-still policy can have a large `episode_return` (~1000) while
`forward_return` is zero or negative.  Use `forward_return`, `mean_x` and
`final_x` to decide whether the task is actually being learned; do not use
`episode_return` alone.  The survive bonus is task reward, not an h label.
