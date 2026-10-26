# Ant V_T transfer diagnostic: frozen h + 1-D A_T warm-start (single seed)

Status: **completed 2026-10-26; diagnostic only; result null.**  The promoted
Ant 3-seed 10M safety result is unchanged and is not re-opened.  This arm asked
whether the navigation task-value channel transfers to Ant, not whether Ant
safety can be improved.

**Result:** the V_T quality gate passed and the matched control was valid, but
the 1-D `A_T` warm-start did **not** beat the no-V_T 1M control on the task
(deterministic forward `-1.33%`, return `-1.21%`) while safety was equal.  The
pre-registered rule gives `vt_transfer_supported = false`; the Ant V_T line is
stopped and no follow-up arm is launched.

This is **not** a main V6 claim, **not** a from-scratch result, and **not** a
replacement for the navigation from-scratch task-channel arm.  It is a
single-seed screen; it is not scaled without a robust benefit.

## 1. Question

> On the existing Ant 10M h-only seed-0 policy, does a frozen task value
> `V_T`, fed to the actor as a 1-D relative task scalar `A_T`, improve the
> task beyond a matched no-V_T 1M continuation without losing the h-only
> safety effect?

The first step is a base sanity check: re-evaluate the existing 10M h-only
seed-0 checkpoint and confirm the recorded deterministic/stochastic safety.
No h-only retraining is required for this screen.

## 2. Base policy and h

- Actor: `outputs/v6/independent/ant/mappo_h_pred_gate_10m/seed0/policy_010000000.pth`
- h: `outputs/v6/independent/ant/h_predictive/ant_h.pkl`
- Recorded 100ep deterministic: `safe_episode 0.97`, `forward 3723.09`,
  `cost_fraction 0.00176`, `wall 0.03`, `topple 0.00`, `survived 1.00`.
- Recorded 100ep stochastic: `safe_episode 0.78`, `forward 2923.73`,
  `cost_fraction 0.00419`.
- 3-seed deterministic `safe_episode`: 0.97 / 0.95 / 0.99; stochastic:
  0.78 / 0.85 / 0.89.

The base is re-evaluated with the standard Ant protocol (`100` episodes,
deterministic and stochastic, eval seed `20000`) before any V_T work.  A
gross mismatch with the recorded values stops the line.

## 3. V_T definition

`V_T` is a separate local task value trained offline on Ant rollouts.  It is
**not** a critic: the centralized PPO critic is unchanged.  It has no h input
and never sees the safety cost.

Target: finite-horizon ungated official task return

```text
G_t^(K) = sum_{k=0}^{K-1} gamma^k * r_task(t+k)
```

- `r_task`: the official Ant task reward returned by the environment
  (`reward`), not the shaped PPO reward and not the cost.
- `K = 64`, `gamma = 0.99`.
- The sum stops at the episode boundary; there is no bootstrapping past a
  terminal step.
- No h gate, no safety cost, no PPO critic role.

Network: a local MLP over the 31-D per-agent observation
(`LocalTaskValueNet`, hidden `(128, 128)`), producing one scalar per agent.
The navigation graph net is **not** reused; Ant has no lidar/goal graph.

The relative scalar is

```text
A_T = clip((V_T - EMA_mean) / sqrt(EMA_var + 1e-6), -5, 5)
```

The EMA is frozen with the V_T after offline training.

## 4. V_T data and quality gate

- Data: 100 episodes from the frozen 10M h-only base, stochastic actions,
  eval/collection seed `30000` (the same convention as the nav T0w
  collector).
- Split by episode: 80 train / 10 validation / 10 test.
- Quality gate (fixed before training the actor):
  - validation explained variance `>= 0.20`;
  - test Pearson `>= 0.40`.
- If either fails, report a V_T quality failure, fix V_T offline once, and
  re-check.  If it still fails, stop: do not interpret the actor warm-start
  as a task-channel test.

## 5. Actor feature

This pilot uses the **1-D relative scalar** `A_T` only.  The navigation 5-D
feature `[A_T, grad_s V_T]` is a separate follow-up; the 1-D pilot isolates
whether the V_T signal itself transfers.

Actor input becomes `obs31 + h1 + grad_h9 + A_T1 = 42-D`.  The base path
(`use_task_value = false`) remains 41-D and loads the existing checkpoint
unchanged.

### Norm-after-concat (important)

The Ant actor uses `LayerNorm(obs_dim)` on its input.  Adding a zero column
before that norm would change the LayerNorm statistics and would not preserve
the base policy.  The implementation therefore:

- applies the input LayerNorm only to the original 41-D base features;
- concatenates the 1-D task feature **after** the norm;
- zero-initializes the new first-layer column.

At step 0 the warm-start policy is therefore exactly the base policy.  The
new column is the only parameter difference; the V_T and its EMA are frozen.

## 6. Warm-start and matched control

Both arms start from the same 10M h-only seed-0 checkpoint, use the same h,
cost penalty (`0.1`), reward, optimizer settings, seed, and env seeds.

- **Warm-start:** 1M additional steps (`10,000,000 -> 11,000,000`) with the
  frozen 1-D `A_T` feature active.  h/gate/cost/reward are unchanged.
- **Matched control:** the same checkpoint, the same 1M additional steps, and
  the task feature off/zero.  This isolates the V_T channel from the extra
  training; comparing only base vs warm-start would confound the 1M steps.
- The PPO critic and its centralized state are unchanged.  No second critic
  head, no QP, no action filter, no task gate, no MAPPO-Lagrangian.
- Both arms start a fresh Adam optimizer (the base checkpoint has no optimizer
  state); the optimization budget is matched.

## 7. Evaluation

- In-training monitors: 20 episodes every 250k steps for both arms.
- Formal evaluation: 100 episodes, deterministic and stochastic, eval seed
  `20000`, same protocol for both arms and the base.
- No 20ep safety ranking; the 100ep result is the decision.

## 8. Pre-registered decision rule

The control must first be valid:

- **control_valid:** control 100ep deterministic `safe_episode >= 0.90` and
  `cost_fraction <= 0.01`.  If false, the comparison is degenerate: report an
  invalid control and make no V_T-transfer claim.
- **task_gain:** warm-start 100ep deterministic `forward_return >= control +
  5%` **or** `episode_return >= control + 5%`.
- **safety_not_worse:** warm-start 100ep deterministic
  - `safe_episode >= control - 0.03`,
  - `cost_fraction <= control + 0.005`,
  - `wall_fraction <= control + 0.02`,
  - `topple_fraction <= control + 0.02`.
- **vt_transfer_supported = control_valid and task_gain and
  safety_not_worse.**

If supported, report a one-seed cross-domain diagnostic and propose (do not
launch) a separately pre-registered 5-D / longer-budget / new-seed arm.  If
not supported, stop the Ant V_T line.

## 9. Scope and non-claims

- Single seed; not a formal Ant result and not a main V6 claim.
- The promoted Ant 3-seed safety result is untouched; this arm uses a frozen
  h and does not tune safety.
- No Ant h retraining is part of this screen; the existing 10M h-only base is
  reused.
- No navigation V_T weights are copied; the Ant V_T is trained from scratch
  on Ant data.
- No from-scratch Ant task-channel arm is opened by this protocol.

## 10. Artifacts

- data: `outputs/v6/independent/ant_vt/data/ant_h10m_seed0_100ep.npz`
- V_T checkpoint: `outputs/v6/independent/ant_vt/ant_task_value.pkl`
- V_T summary: `outputs/v6/independent/ant_vt/summary.json`
- base re-evaluation: `outputs/v6/independent/ant_vt/base_eval/`
- warm-start: `outputs/v6/independent/ant_vt_warmstart_1m/seed0/`
- control: `outputs/v6/independent/ant_vt_control_1m/seed0/`
- decision: `outputs/v6/independent/ant_vt/decision.json`
- chain log: `outputs/v6/independent/ant_vt/chain.log`
- collector: `nav/scripts/collect_v6_ant_task_value_data.py`
- V_T trainer: `nav/scripts/train_v6_ant_task_value.py`
- master runner: `nav/scripts/run_v6_ant_vt_transfer.sh`
- summarizer: `nav/scripts/summarize_v6_ant_vt_warmstart.py`

## 11. Status

Run started 2026-10-26 12:54 UTC via
`nav/scripts/run_v6_ant_vt_transfer.sh` (single seed, `CONFIRM=1`); completed
14:09 UTC.

- [x] Protocol pre-registered.
- [x] Base h-only sanity re-evaluation.
- [x] V_T data collection.
- [x] V_T offline training and quality gate (passed).
- [x] Warm-start 1M training.
- [x] Matched no-V_T 1M control.
- [x] 100ep deterministic + stochastic evaluation.
- [x] Pre-registered decision: `vt_transfer_supported = false`; line stopped.

## 12. Result (2026-10-26)

### Base sanity re-evaluation

Standard protocol: 100 episodes, eval seed `20000`, existing seed-0 10M
h-only checkpoint.  Both modes match the recorded base within seed noise, so
the line was not stopped.

| mode | safe_ep | forward | episode_return | cost_fraction |
|---|---:|---:|---:|---:|
| deterministic | 0.97 | 3723.09 | 4366.62 | 0.00176 |
| stochastic | 0.80 | 3017.92 | 3457.05 | 0.00275 |

Recorded stochastic was `safe 0.78`, `forward 2923.73`; the re-evaluation is
slightly better but not a protocol mismatch.

### V_T quality gate

100 episodes, stochastic base actions, collection seed `30000`; split
80 train / 10 validation / 10 test by episode; target `K=64`, `gamma=0.99`,
ungated official task return.

| split | rows | explained variance | Pearson | RMSE |
|---|---:|---:|---:|---:|
| train | 150,004 | 0.5437 | 0.7386 | 23.07 |
| validation | 19,144 | 0.4838 | 0.6971 | 24.10 |
| test | 18,034 | 0.4047 | 0.6597 | 26.05 |

Gate: validation EV `0.4838 >= 0.20`, test Pearson `0.6597 >= 0.40` ->
**passed** on the first attempt; no V_T fix was needed.

### Formal 100ep evaluation

| arm | mode | safe_ep | forward | episode_return | cost_fraction | wall_ep | topple_ep |
|---|---|---:|---:|---:|---:|---:|---:|
| matched control | deterministic | 0.96 | 4014.03 | 4630.37 | 0.000187 | 0.02 | 0.02 |
| 1-D `A_T` warm-start | deterministic | 0.96 | 3960.83 | 4574.53 | 0.000747 | 0.02 | 0.02 |
| matched control | stochastic | 0.78 | 3440.15 | 3885.84 | 0.00319 | 0.16 | 0.09 |
| 1-D `A_T` warm-start | stochastic | 0.73 | 3247.81 | 3670.66 | 0.00294 | 0.12 | 0.16 |

Deterministic warm-start minus control: forward `-53.20` (`-1.33%`),
episode return `-55.84` (`-1.21%`), safe `0.00`, cost_fraction `+0.00056`,
wall `0.00`, topple `0.00`.  Stochastic warm-start is also worse on the task
(forward `-192.34`, `-5.59%`; safe `-0.05`).

### Decision

- `control_valid = true` (safe `0.96 >= 0.90`, cost_fraction
  `0.000187 <= 0.01`).
- `task_gain = false` (forward and episode return are both below control).
- `safety_not_worse = true`.
- `vt_transfer_supported = false`.

Per the pre-registered rule, the Ant V_T line is **stopped**; no 5-D,
longer-budget, or new-seed arm is launched.

### Why the base improved at all

Both 1M continuations are better than the 10M base: control forward
`+290.94` (`+7.8%`) and warm-start forward `+237.74` (`+6.4%`); control also
has a much lower cost_fraction (`0.00176 -> 0.000187`).  Extra training, not
the V_T channel, explains the improvement.

The A_T input was live, not silently disconnected: both new actor columns
learned nonzero weights (norm `0.086` / `0.098`), and a post-hoc probe on
random observations shows a small but nonzero action effect.  The channel was
therefore available and used weakly; under this 1M 1-D protocol it still did
not produce a task gain.  This does not establish that a 5-D / longer-budget
variant would fail; it only closes the pre-registered 1-D screen, and any
follow-up needs a separate pre-registration and approval.
