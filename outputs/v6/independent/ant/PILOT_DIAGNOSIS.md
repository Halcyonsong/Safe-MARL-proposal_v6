# V6 Ant pilot diagnosis: current task baseline is not V5-equivalent

> Historical note (added 2026-10-25): this document records the 1M pilot and
> the termination of the static/dynamic h+grad branch.  The later
> **predictive-h + safety-gated forward** mechanism is a different branch and
> it passed the 3-seed 2M and 10M checks.  See `FINAL_ANT_REPORT.md`,
> `PREDICTIVE_H_GATE_2M.md`, and `RETENTION_10M.md`.  Statements below such as
> "the h branch is terminated" refer only to the static/dynamic h experiments.

The current short V6 Ant 1M pilot was stopped after the 100k–600k intermediate
evaluations showed no forward progress.  This is a **baseline/setup mismatch**,
not evidence that h is the problem.

## Overnight decision (2026-10-24)

**The h branch is terminated.**  After fixing the MAPPO baseline and evaluating
matched 100-episode checkpoints, no h configuration produced a robust
safety/task improvement over the cost-only control.  The corrected pure-task
baseline is healthy at 10M, and a cost-only scheduled-penalty control is a
strong safe baseline at 2M, but the frozen h+grad channels do not improve it;
they either stand still (scale 1 with stronger penalties) or become unsafe
(scale 3, or scale 1 at longer training).  No 3-seed h run and no navigation
task-joint run were launched.  Details and the decisive 2M comparison are
below.

## Reference: V5 Ant pure task MAPPO

`/home/xiang/workspace/python-project/Safe-MARL-proposal_v4/outputs/v5/ant/baselines/mappo_10m_eval100.json`
(V5 `mappo_paper`, lambda=0, 100 fixed scenes, deterministic):

| metric | value |
|---|---:|
| training steps | 10M |
| return | 4477.2 |
| cost rate | 31.57% |
| wall rate | 31.54% |
| topple rate | 0.036% |
| survived 1000 | 89% |
| x_final | 203.4 |

This is the expected “forward is good, safety is poor” pure task baseline.  The
V5 training log shows why 1M is too early: average episode reward is about 409
at 1M, 608 at 2M, 2605 at 8M, and 3406 at 10M.

## Current V6 task-only arm (stopped at ~600k)

20ep deterministic eval at 600,064 steps:

```text
safe_episode   1.0
forward_return -1.45
mean_x         -0.073
final_x        -0.070
episode_length 1000
survived       1.0
```

The policy stands still and collects `survive_reward`; it does not reproduce
the V5 pure MAPPO forward behavior.  The h-only and cost-only arms show the
same standstill.  Therefore the current pilot cannot answer whether h helps.

## Material setup differences

| item | V5 pure MAPPO | current V6 task-only |
|---|---|---|
| training budget | 10M | 1M |
| parallel envs / rollout | 16 threads, ~16k per update | 1 env, 2048 per update |
| actor | 64 hidden, ReLU, feature LayerNorm + MLP LayerNorm | 64x64 tanh, no feature/LayerNorm |
| critic | 64 hidden, normalized centralized state, ValueNorm/PopArt | 128x128 tanh, **raw 29-D global state**, no value norm |
| action distribution | Gaussian, fixed std ≈0.365, no tanh squash | tanh-squashed Gaussian, log_std -0.5 (std ≈0.607) |
| actor / critic lr | 9e-5 / 5e-3 | 3e-4 / 3e-4 (single optimizer) |
| PPO | 5 epochs, 40 minibatches, entropy .01, grad clip 10 | 4 epochs, 8 minibatches, entropy .001, grad clip .5 |
| value loss | clipped/Huber, value coef 1 | clipped squared, value coef .5 |
| global state | normalized `[state, agent_id]` (31-D) | raw `_get_obs()` (29-D) |
| no-h arm h overhead | none | still computes analytic h/grad every step for diagnostics |

The largest implementation risks are the raw centralized state, missing
ValueNorm/value scaling, the much smaller rollout/parallel sample diversity,
and the different actor/action parameterization.  The h dimension itself is
not the cause: `task_only` has no h features and also fails.

## Confirmed implementation bugs fixed

1. **PPO log-probability broadcasting bug.**  `old_log_probs[idx, i]` had
   shape `(B, 1)` while the new log probability was squeezed to `(B,)`.  The
   subtraction broadcast to `(B, B)`, so the PPO ratio and policy loss were
   computed over the wrong pairs (the smoke log showed `policy_loss≈98`).
   `act_single()` now returns one log-prob per agent and the update explicitly
   squeezes the stored value.  After the fix the same smoke update gives
   `policy_loss≈-0.005`.
2. **Critic saw raw state.**  The V6 env now returns the upstream normalized
   `[state, agent_id]` 31-D centralized state.
3. **No-h arms paid for finite-difference h.**  `task_only`/`cost_only` now use
   `NullAntSafetyProvider`; h gradients are only computed for the h arm.
4. **Separate actor/critic optimizers.**  In the old code the critic's much
   larger value loss shared one `max_grad_norm=0.5` clip with the actor; the
   corrected path uses actor lr `9e-5`, critic lr `5e-3`, and independent clip.
5. **Checkpoint activation mismatch.**  `load_ant_mappo_model()` reconstructed
   the actor/critic with default tanh even when the checkpoint config used
   ReLU.  Training-time evaluation was correct, but standalone checkpoint
   evaluation was not comparable.  Activations are now stored in
   `model_config` and restored (with policy-config fallback for old
   checkpoints).  Task-base 1M checkpoint re-evaluation then reproduced the
   training evaluation (`forward_return≈328.9`, `final_x≈16.4`).
6. **Value normalization / network recipe.**  The corrected configs enable
   ValueNorm, feature/LayerNorm, ReLU, Gaussian std≈0.365, 5 PPO epochs and
   Huber value loss.

## Current corrected baseline run

`outputs/v6/independent/ant/mappo_task_base/seed0/` is the first 10M-step
corrected task-only run (8 envs, 16,384 transitions/update, CPU).  At 1M steps
it already shows the expected pure-task pattern:

| 1M evaluation | forward_return | mean_x | final_x | wall_episode | survived | cost_fraction |
|---|---:|---:|---:|---:|---:|---:|
| deterministic, 20 ep | 328.9 | 12.4 | 16.4 | 100% | 100% | 18.5% |
| stochastic, 20 ep | 288.8 | 7.9 | 14.4 | 95% | 55% | 36.6% |

This is not V5-final performance, but it is clearly no longer the old
standing-still failure.  The same run continues to improve: at 2.5M the 20-ep
deterministic evaluation is `forward_return=1524.0`, `mean_x=46.0`,
`final_x=76.2`, `wall_episode=1.0`, `survived=1.0`.  This is now a credible
pure-task MAPPO baseline; the remaining question is the final 10M level.

The old penalty-10 cost-only/h-only arms are still safe-but-stationary at
0.5M; do not interpret their `safe_episode=1.0` as task success.

### Cost penalty 10 is a standing local optimum

The first corrected cost-only/h-only runs used `cost_penalty=10`.  At 1M their
deterministic evaluations were:

```text
cost_only: forward_return 2.82, final_x 0.14, cost_fraction 0.0
h_only:    forward_return -0.58, final_x -0.03, cost_fraction 0.0
```

The task base at the same step had `forward_return=328.9`, `final_x=16.4`,
`cost_fraction=0.185`.  A per-step penalty of 10 makes an early wall-hitting
policy worse than the +1/step survive reward for standing, so the policy
collapses to the safe standstill.  These two 10M runs were stopped at ~1M and
are not the mainline.

A second sweep with `cost_penalty=1.0` also stood still at 0.5M:
`cost_only forward_return≈0.9`, `h_only≈-3.1`.  That is why the current
safety-penalty pilot uses `cost_penalty=0.1`
(`v6_ant_cost_p01.yaml`, `v6_ant_h_p01.yaml`).  The aim is to preserve forward
learning first, then see whether the h features reduce wall/fall cost relative
to cost-only at a matched seed and step.  Do not select a penalty by
`safe_episode` alone.

### Matched 1M p0.1 pilot (20 eval episodes, seed 20000)

| arm | mode | forward_return | mean_x | final_x | safe_ep | wall_ep | cost_fraction |
|---|---|---:|---:|---:|---:|---:|---:|
| task_only | deterministic | 328.9 | 12.4 | 16.4 | 0.0 | 1.00 | 0.185 |
| cost_only | deterministic | 222.9 | 8.1 | 11.1 | 0.1 | 0.90 | 0.566 |
| h_only | deterministic | 299.1 | 7.9 | 15.0 | 1.0 | 0.00 | 0.000 |
| task_only | stochastic | 288.8 | 7.9 | 14.4 | 0.0 | 0.95 | 0.366 |
| cost_only | stochastic | 168.0 | 5.1 | 8.4 | 0.0 | 0.85 | 0.209 |
| h_only | stochastic | 185.5 | 5.3 | 9.3 | 0.0 | 0.75 | 0.269 |

The deterministic h-only policy is the clear winner at this checkpoint: it
keeps most of the task progress while eliminating wall cost.  However, the
same h-only checkpoint is **not** robustly safe under stochastic actions
(`wall_episode=0.75`, `cost_fraction=0.269`), so this is preliminary evidence,
not a final h benefit.  All three arms are being extended to 10M with matched
seed/step/eval protocol; the stochastic result must be reported alongside the
deterministic one.

A 100-episode deterministic check at the same 1M checkpoints confirms the
signal:

| arm | forward_return | mean_x | final_x | safe_ep | wall_ep | cost_fraction |
|---|---:|---:|---:|---:|---:|---:|
| task_only | 283.3 | 11.0 | 14.2 | 0.00 | 1.00 | 0.177 |
| cost_only | 233.5 | 8.6 | 11.7 | 0.03 | 0.97 | 0.630 |
| h_only | 280.7 | 7.7 | 14.0 | 0.91 | 0.09 | 0.015 |

At this checkpoint h_only retains essentially the task-only forward progress
while cutting deterministic wall cost from 17.7% to 1.5% and keeping
`safe_episode=0.91`; cost-only is both slower and much less safe.  This is a
real incremental h signal, but 100-episode stochastic evaluation and the 10M
runs are still needed before declaring the h benefit robust.

The 10M h/cost runs at their 1M intermediate checkpoints show the same
direction: `h_only forward_return=397.5, final_x=19.9, cost_fraction=0.042,
safe_episode=0.70` versus `cost_only 237.8, 11.9, 0.600, 0.00`.  This is still
a single seed and 20 evaluation episodes, so it is a gate signal, not a final
result.

The same 1M checkpoints under stochastic actions (100 episodes) are much less
favourable to h:

| arm | forward_return | mean_x | final_x | safe_ep | wall_ep | cost_fraction | survived |
|---|---:|---:|---:|---:|---:|---:|---:|
| task_only | 255.0 | 7.2 | 12.8 | 0.00 | 0.87 | 0.288 | 0.43 |
| cost_only | 158.0 | 4.9 | 7.9 | 0.05 | 0.77 | 0.226 | 0.53 |
| h_only | 191.2 | 5.5 | 9.6 | 0.03 | 0.86 | 0.253 | 0.46 |

h_only is better than cost_only on forward progress at similar stochastic
cost, but it is not robustly safe and is below task_only on forward.  The
deterministic safety result must not be reported alone.  This is the main open
question for the 10M runs.

## Final status

1. Corrected pure-task baseline: complete at 10M (table above).  It matches the
   V5 pure-task forward level and is the valid control.
2. h branch: terminated.  The 100-episode matched tests at 2M and all
   intermediate checks show no robust h benefit over the cost-only schedule;
   details below.
3. Cost-only scheduled control: complete at 10M.  It retains 90% of pure-task
   forward (`4055.4` vs `4496.8`) with `cost_fraction=0.0222` vs `0.1323`;
   `safe_episode=0.10` vs 0.00 because any wall still counts as unsafe.  This
   is a useful safe/task control, not an h result.
4. The dynamic-h target (`kappa=0.5`, `kappa_z=0.2`) plus a fixed h barrier
   (`2.0 * max(0, 0.1-h)`) was tested after the static-margin audit.  It also
   failed: dynamic h alone was less safe than cost-only, and the barrier
   collapsed to standstill.  The details are in `DYNAMIC_H_GATE.md` and `EFFECT_ANALYSIS.md`.  Do not
   scale this mechanism to 3 seeds, 10M, or the navigation task joint run.

## Final corrected pure-task baseline (10M, 100 episodes)

`mappo_task_base/seed0/policy_final.pth`, deterministic mean policy:

| metric | value |
|---|---:|
| forward_return | 4496.85 |
| mean_x | 111.69 |
| final_x | 224.85 |
| episode_return | 5041.77 |
| wall_episode | 0.99 |
| topple_episode | 0.05 |
| cost_fraction | 0.1323 |
| survived | 0.95 |
| action_abs | 0.2708 |

Stochastic 100-episode: `forward_return=3620.85`, `mean_x=89.88`,
`final_x=181.05`, `cost_fraction=0.1533`, `wall_episode=1.0`,
`survived=0.75`.  This matches the V5 pure-task forward level (V5:
`forward_return≈4477`, `final_x≈203`) and is the correct control.  The wall
episode rate is higher than the V5 reference, so the safety comparison remains
important.

### Cost-only scheduled control at 10M (100 episodes)

| mode | forward_return | mean_x | final_x | safe_episode | wall_episode | cost_fraction | survived |
|---|---:|---:|---:|---:|---:|---:|---:|
| deterministic | 4055.37 | 95.78 | 202.77 | 0.10 | 0.89 | 0.0222 | 0.98 |
| stochastic | 3067.39 | 72.97 | 153.38 | 0.04 | 0.91 | 0.0508 | 0.81 |

This is the best task/safety trade found in the overnight run.  It is a
cost-only control (no h features); it shows that a pre-scheduled fixed cost
penalty can reduce per-step cost without collapsing forward.  It does not
rescue the h hypothesis.

## Decisive matched 2M comparison (100 deterministic episodes)

All checkpoints are seed 0, eval seed 20000.  `cost-only` is the pre-scheduled
cost coefficient (0.1 -> 1.0 by 2M); the h arms use scale 1 and the same
schedule.

| arm | forward_return | mean_x | final_x | safe_episode | wall_episode | cost_fraction | survived |
|---|---:|---:|---:|---:|---:|---:|---:|
| task_only | 1003.66 | 31.99 | 50.19 | 0.00 | 1.00 | 0.2646 | 0.88 |
| cost_only ramp1.0 | 840.11 | 23.41 | 42.01 | 0.90 | 0.10 | 0.0044 | 0.99 |
| h(scale1) ramp0.5 | 814.45 | 27.23 | 40.73 | 0.00 | 1.00 | 0.5651 | 0.87 |
| h(scale1) ramp1.0 | 957.99 | 28.26 | 47.91 | 0.38 | 0.62 | 0.0256 | 0.97 |

The h ramp1.0 arm has 14% more forward than cost-only but 6x the per-step cost
and much lower `safe_episode` (0.38 vs 0.90).  The h ramp0.5 arm is both slower
and less safe.  Under stochastic actions the same ordering holds:
cost-only ramp1.0 `safe=0.14, cost_fraction=0.0721`; h ramp1.0
`safe=0.04, cost_fraction=0.1035`; h ramp0.5 `safe=0.02, cost_fraction=0.2554`.
No h configuration is an incremental safety improvement.

## H-branch attempts (all terminated)

| variant | step | forward_return | safe_episode | cost_fraction | verdict |
|---|---:|---:|---:|---:|---|
| h scale1, p0.1 | 1M | 280.7 | 0.91 | 0.015 | good deterministic but transient |
| h scale1, p0.1 | 2.5M | 1113.7 | 0.00 | 0.4565 | safety lost |
| h scale3, p0.1 | 0.5M | 67.3 | 0.85 | 0.009 | good early |
| h scale3, p0.1 | 1M | 99.9 | 0.10 | 0.7217 | safety lost |
| h scale3, ramp0.5 | 2M | 638.6 | 0.15 | 0.4222 | worse than cost |
| h scale3, ramp1.0 | 2M | 531.4 | 0.20 | 0.1796 | worse than cost |
| h scale1, ramp0.5 | 2M | 814.5 | 0.00 | 0.5651 | no benefit |
| h scale1, ramp1.0 | 2M | 958.0 | 0.38 | 0.0256 | forward benefit, safety loss |

The offline h is still good (validation BA 0.97+), but closed-loop h/grad
features do not yield a robust safety advantage in this MAPPO actor.  This is
consistent with the earlier navigation observation that high h classification
quality alone is insufficient.  The correct conclusion is negative for the
current h integration, not a reason to scale seeds.

### Optimization: h feature scale

At 0.5M, stronger cost penalties again produced standstill
(`cost_p0.5 forward≈9`, `h_p0.5≈7`; `cost_p0.2≈7`, `h_p0.2≈10`), so those
runs were stopped.  The useful variant is **h/grad feature scale 3.0** at
`cost_penalty=0.1`: at 0.5M its deterministic evaluation is
`forward_return=67.3`, `mean_x=2.55`, `final_x=3.37`,
`safe_episode=0.85`, `cost_fraction=0.0091`, while the scale-1 h arm is nearly
stationary (`forward≈7`) and the cost-only p0.1 arm is at `forward≈33`.
That scale-3 run reaches 1M with `forward_return=99.9` but
`cost_fraction=0.722` and `safe_episode=0.1`; the early safety advantage is not
stable.  Scale-1 h also loses safety after 1M (`cost_fraction` rises from
~0.04 at 1M to ~0.36–0.46 at 1.5–2.5M).  The h features are therefore not
being used strongly enough once the actor learns to move.

The next optimization is a **pre-scheduled cost coefficient** (not a
Lagrangian/dual update): `0.1` for the first 0.5M steps, then linearly to
`0.5` or `1.0` by 2M and held fixed.  The h arm uses feature scale 3; matched
cost-only controls use the same schedule.  This keeps the task reward/cost
unchanged and does not add QP, action projection, or h-based action correction.
If neither schedule yields a robust h benefit, the h branch is terminated.

At the 1M schedule checkpoints (20 deterministic episodes), the **cost-only**
schedule is much stronger than the scaled-h schedule:

| 1M schedule | forward_return | final_x | safe_episode | cost_fraction |
|---|---:|---:|---:|---:|
| cost ramp to 0.5 | 222.7 | 11.1 | 0.25 | 0.316 |
| h(scale3) ramp to 0.5 | 98.2 | 4.9 | 0.50 | 0.403 |
| cost ramp to 1.0 | 335.9 | 16.8 | 0.50 | 0.060 |
| h(scale3) ramp to 1.0 | 160.2 | 8.0 | 0.15 | 0.623 |

This is evidence that the scale-3 h channel is actively harmful, not merely
ignored.  The final optimization is the same schedule with **scale 1**, to
separate feature-scale damage from schedule damage.  If h(scale1)+schedule
does not beat the matched cost-only schedule on both forward and cost, the h
branch will be terminated rather than scaled further.


## Dynamic-h / h-barrier follow-up (final; negative)

The static h audit found zero velocity gradient and ignored future-unsafe
labels.  The final follow-up recomputed the same V6 h geometry with a
dynamic margin (`kappa=0.5`, `kappa_z=0.2`) and added a fixed soft barrier
`2.0 * relu(0.1-h_next)`.  The dynamic h is offline good (test BA 0.9612,
unsafe recall 0.9666, false alarm 0.0275, gradient cosine 0.870) and its
velocity gradient is nonzero; increasing approach speed by 2 reduced h by
0.415.  Closed-loop, however, it failed the matched 1M gate (100 episodes,
eval seed 30000):

| arm | mode | forward | final_x | safe | wall | cost_fraction |
|---|---|---:|---:|---:|---:|---:|
| cost-only | det | 344.73 | 17.24 | 0.29 | 0.70 | 0.0784 |
| static h | det | 243.06 | 12.16 | 0.77 | 0.23 | 0.0782 |
| dynamic h | det | 509.06 | 25.46 | 0.09 | 0.91 | 0.2666 |
| dynamic h + barrier | det | -2.05 | -0.10 | 1.00 | 0.00 | 0.0000 |
| cost-only | sto | 151.78 | 7.59 | 0.04 | 0.76 | 0.2279 |
| dynamic h | sto | 231.46 | 11.58 | 0.00 | 0.81 | 0.2357 |
| dynamic h + barrier | sto | 54.72 | 2.74 | 0.17 | 0.54 | 0.1691 |

At 1.5M (20 deterministic episodes), dynamic h was `forward=844.52`,
`safe=0.00`, `cost_fraction=0.3514`; the barrier arm was `forward=13.61`,
`safe=1.00` (standstill); the matched cost-only control was `forward=476.07`,
`safe=0.40`, `cost_fraction=0.0464`.  The runs were stopped at 1.5M/2.0M.
No 3-seed, 10M, or navigation task-joint run is justified.  See
`DYNAMIC_H_GATE.md`.
