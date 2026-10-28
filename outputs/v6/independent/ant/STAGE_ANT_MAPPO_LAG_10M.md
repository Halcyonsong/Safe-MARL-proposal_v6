# Ant 10M from-scratch MAPPO-Lagrangian external baseline

Status: **complete, external single-seed screen** (2026-10-27 19:17 CST).
Trained under fresh experiment `v6_lag_10m_seed0_restart_20261027` after the
Nav P0/P1 ablations closed.  The earlier short partial run was stopped and
archived; it is not a result and must not be resumed or evaluated.  This is an
**external baseline**, not a V6 main-method run and not a V6 claim.

Canonical status: [`../../STATUS.md`](../../STATUS.md).

## Why this run exists

There is no 10M from-scratch MAPPO-Lagrangian Ant checkpoint in the V6 asset
set.  The two existing upstream references are not replacements:

| artifact | what it is | why it is not the matched baseline |
|---|---|---|
| `mappo_paper/run1` | 10M pure MAPPO, `lamda_lagr=0` | no collision constraint |
| `lag_ft_mappo/run1` | 2M Lag fine-tune from the 10M pure actor | not from scratch; ~12M total actor steps; 2M constrained |

This run trains the upstream `mappo_lagr` recipe from random initialization for
exactly 10M environment steps.  It is a **single-seed screen** (seed 0).  A
three-seed baseline would require separate approval after seeing the screen.

## Protocol (frozen before launch)

- Trainer: upstream `MAPPO-Lagrangian/mappo_lagrangian/scripts/train/train_mujoco.py`,
  launched through `nav/scripts/train_ant_lag_upstream.py` (thin wrapper that
  forces `cuda=True`; the vendor parser otherwise defaults to CPU).
- Environment: official `Ant-v2`, `agent_conf=2x4`, `agent_obsk=1`,
  `episode_length=1000`, 16 rollout threads.
- Steps: 10,000,000 exactly (625 updates x 16 envs x 1000 steps).
- Seed: 0.  Fresh experiment name `v6_lag_10m_seed0`; the runner refuses to
  reuse an existing upstream output directory and never resumes.
- Hyperparameters (upstream paper recipe):
  `lr=9e-5`, `critic_lr=5e-3`, `std_x_coef=1`, `std_y_coef=0.5`,
  `num_mini_batch=40`, `ppo_epoch=5`, `use_value_active_masks`,
  `add_center_xy`, `use_state_agent`, `safety_bound=0.2`,
  `lamda_lagr=0.78`, `lagrangian_coef_rate=1e-7`.
- No V5/V4 weights, no V6 h, no QP, no expert action, no warm start.

Implementation note: upstream `config.py` defines `--use_value_active_masks`
with `action='store_false', default=True`, so passing the flag records
`use_value_active_masks=False` in the executed config (the same value as the
reference `mappo_paper` run).  Policy active masks remain `True`.  The
launcher is left unchanged for comparability with the existing upstream
reference; this is an upstream flag-semantics quirk, not a post-launch
protocol change.

Runner:

```bash
cd /home/xiang/workspace/python-project/Safe-MARL-proposal_v6
EXPERIMENT=v6_lag_10m_seed0_restart_20261027 \
OUT=outputs/v6/independent/ant/mappo_lag_10m/seed0_restart_20261027 \
TRAIN_SEED=0 STEPS=10000000 bash nav/scripts/run_ant_lag_10m.sh
```

Raw training output is copied to:

```text
outputs/v6/independent/ant/mappo_lag_10m/seed0_restart_20261027/
  launcher.log
  train.log
  command.txt
  upstream_run_dir.txt
  upstream_summary.json
  models/actor_agent{0,1}.pt
```

## Evaluation protocol

Adapter: `nav/scripts/evaluate_ant_upstream_lag.py`.  It loads the final
`actor_agent{i}.pt` files and reports the same metrics as the V6 Ant evaluator
(`safe_episode`, `forward_return`, `episode_return`, `cost_fraction`,
`wall_fraction`, `topple_fraction`, `survived`, `final_x`, ...).

- 100 deterministic episodes and 100 stochastic episodes;
- evaluation seeds `20000..20099`, identical to the V6 Ant evaluation;
- no checkpoint selection: the last saved model is used;
- run with the `mappo_lag` Python environment and MuJoCo 210:

```bash
LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin \
PYTHONPATH=$PWD/nav \
/home/xiang/miniconda3/envs/mappo_lag/bin/python \
  nav/scripts/evaluate_ant_upstream_lag.py \
  --models-dir outputs/v6/independent/ant/mappo_lag_10m/seed0_restart_20261027/models \
  --num-episodes 100 --seed 20000 --device cuda --training-seed 0 \
  --out outputs/v6/independent/ant/mappo_lag_10m/seed0_restart_20261027/eval_100ep_deterministic.json
```

Repeat with `--stochastic` for the stochastic row.

## Result (seed 0, 100 deterministic + 100 stochastic)

Raw files:

```text
outputs/v6/independent/ant/mappo_lag_10m/seed0_restart_20261027/
  eval_100ep_deterministic.json
  eval_100ep_stochastic.json
  eval_100ep_deterministic.log
  eval_100ep_stochastic.log
```

| mode | arm | forward | return | safe_ep | cost_fraction | survived | final_x |
|---|---|---:|---:|---:|---:|---:|---:|
| deterministic | upstream MAPPO-Lag 10M | 3011.02 | 3603.84 | 0.97 | 0.0001 | 0.99 | 150.56 |
| stochastic | upstream MAPPO-Lag 10M | 2305.82 | 2624.36 | 0.66 | 0.0024 | 0.81 | 115.30 |

Matched V6 seed-0 reference (same eval scenes `20000..20099`, same
`safe_episode` definition; from
[`RETENTION_10M.md`](RETENTION_10M.md)):

| mode | arm | forward | safe_ep | cost_fraction | survived |
|---|---|---:|---:|---:|---:|
| deterministic | V6 predictive gate | 3723.09 | 0.97 | 0.0018 | 1.00 |
| stochastic | V6 predictive gate | 2923.73 | 0.78 | 0.0042 | 0.83 |

Screen reading:

- **Deterministic safety is matched** at the episode level (`0.97` each), while
  V6 retains **+19.1%** forward return (`3723.09` vs `3011.02`).  MAPPO-Lag has
  the lower continuous `cost_fraction` (`0.0001` vs `0.0018`).
- **Stochastic V6 is better on both safety and task**: safe `0.78` vs `0.66`
  (`+12 pp`) and forward `2923.73` vs `2305.82` (`+21.1%`); MAPPO-Lag again has
  the lower continuous `cost_fraction`.
- This is **not** a weak straw-man baseline.  The deterministic safety matches
  the promoted V6 gate, so the external screen supports a **task-efficiency**
  advantage at equal deterministic episode safety, not a safety-superiority
  claim.  The stochastic row supports a joint advantage on this seed.
- Single seed only; compare seed 0 to seed 0, not to the 3-seed V6 mean.  Do
  not promote this row into the V6 main table, and do not claim "V6 beats
  MAPPO-Lag" on one seed.

## Reporting rules

1. Report both deterministic and stochastic rows, even if the actor is worse
   than the pure-task baseline or fails to walk.
2. Label the result **external baseline, single-seed screen**.  Do not promote
   it into the V6 main table and do not claim "V6 beats MAPPO-Lag" on one seed.
3. Keep `lag_ft_mappo` (2M constrained fine-tune, ~12M total) as a separate
   reference; never present it as a 10M from-scratch baseline.
4. No seed shopping, no retry with different hyperparameters, no checkpoint
   selection on evaluation episodes.
5. If the screen is competitive, propose a 3-seed run as a separate,
   explicitly approved task.  If it is not competitive, record it and stop.

## Status log

- 2026-10-27 10:09: pre-registration written; 32k smoke on GPU passed.
- 2026-10-27 10:51: matched 16k navigation baselines finished; CPU freed.
- 2026-10-27 10:57: full 10M run launched on GPU (`cuda=True`, 16 rollout
  envs, experiment `v6_lag_10m_seed0`).
- 2026-10-27 11:06: run **stopped** after ~8 minutes by user decision so the
  Nav P0/P1 ablations could use the machine.  The partial upstream output was
  moved to
  `outputs/v6/independent/ant/mappo_lag_10m/partial_stopped_seed0_upstream/`
  and marked do-not-resume.  No evaluation was run, no metrics are reported.
  The Ant 10M baseline is **postponed**; any restart must use a fresh
  experiment name.
- 2026-10-27 17:19: **restarted** after the Nav P0/P1 ablations closed.
  Fresh experiment name `v6_lag_10m_seed0_restart_20261027`, fresh output
  dir `outputs/v6/independent/ant/mappo_lag_10m/seed0_restart_20261027/`.
  Protocol unchanged: 10,000,000 steps, seed 0, upstream paper recipe,
  16 rollout envs, from-scratch.  The archived partial
  `partial_stopped_seed0_upstream/` remains untouched and is not resumed,
  not evaluated, and not reported.  No checkpoint selection; the final
  models will be evaluated once at 100 deterministic + 100 stochastic
  episodes, eval seeds `20000..20099`.
- 2026-10-27 17:21: startup verified.  Launcher PID 19182, trainer PID
  19191; `cuda=True`, 16 rollout workers alive, upstream `run1/` created,
  first model save written, GPU allocation active (552 MiB).  Steady-state
  cadence measured at ~10.5 s/update (625 updates total), so expected
  completion is roughly 1.8-2 h after launch, subject to machine load.  No
  changes to the pre-registered protocol; no checkpoint selection.
- 2026-10-27 17:31: evaluation adapter sanity check passed on the unrelated
  32k smoke checkpoint (1 deterministic episode, seed 20000, CPU).  This is
  a load-path check only, not a baseline result; the smoke checkpoint is not
  part of the reported screen.
- 2026-10-27 19:17: training **completed**, 625/625 updates
  (`9,936,000/10,000,000` logged; the final update completes the 10M budget).
  Final `models/actor_agent{0,1}.pt` and `critic_agent{0,1}.pt` copied; no
  resume and no checkpoint selection.
- 2026-10-27 19:23: frozen 100 deterministic + 100 stochastic evaluation
  completed on eval seeds `20000..20099`.  Deterministic
  `safe 0.97`, `forward 3011.02`, `return 3603.84`, `cost_fraction 0.0001`;
  stochastic `safe 0.66`, `forward 2305.82`, `return 2624.36`,
  `cost_fraction 0.0024`.  Result recorded above.  External single-seed
  screen only; no 3-seed Ant expansion is launched.
