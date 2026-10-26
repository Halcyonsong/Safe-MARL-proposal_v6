# V6 navigation C1a: actor2 from scratch with h2 (seed-0 pilot)

Status: **FAILED (seed 0, 100ep final).  All five pre-registered checks
failed.  The h2/actor2 from-scratch branch is stopped; C1b is not authorized.**

Canonical status: [`../../STATUS.md`](../../STATUS.md).  C0a result and
post-hoc h-swap: [`STAGE_C0_H2_TEMPORAL.md`](STAGE_C0_H2_TEMPORAL.md),
[`../nav_c0_h2_actor1_swap/README.md`](../nav_c0_h2_actor1_swap/README.md).

## 1. Why this is a new pilot, not the original C1

The original Stage C1 was authorized only if the full C0 passed.  C0a failed
its pre-registered offline rule: h2 worsened the 17-32-step risk-region MAE
(`0.826` vs `0.709`) even though global MAE improved.

A post-hoc, read-only seed-0 h-swap of h2 into actor1 was then run.  It was
directionally better than h1 (safe episode `0.90` vs `0.84`, official safety
`0.985` vs `0.974`, success `0.439` vs `0.428`, return `61.40` vs `60.91`),
but the differences are one-seed and not statistically separated, and the
swap is off-policy because actor1 was trained with h1.

The user explicitly approved a **C1a seed-0 pilot** to test whether a freshly
trained actor2 with h2 improves closed-loop safety without task collapse.
This is an exploratory pilot outside the original C0/C1 gate; it is not a
retroactive C0 pass and not a 3-seed claim.

## 2. Scope

In scope:

- train actor2 from scratch with the frozen C0 h2 seed-0 checkpoint;
- same environment, network, cost, optimizer, budget, and evaluation protocol
  as the repaired Stage-B actor1;
- one seed (0), 16,384 steps, 100 deterministic episodes at eval seed
  10000;
- compare against the existing actor1 seed-0 result and the post-hoc
  h2+actor1 h-swap.

Not in scope:

- no warm start from actor1 or any other policy;
- no h2 seeds 1/2 (that would be the original C0b, not authorized);
- no gate sweep, gate-scope change, cost-schedule change, task channel,
  QP, action projection/filter, or action conditioning;
- no 3-seed claim;
- no formal/CBF-certified safety claim.

## 3. Fixed protocol

- actor2 seed: `0`
- actor budget: `16,384` steps, `rollout_steps=128`, `epochs=4`,
  `minibatch_size=256`, `lr=3e-4`, `cost_coef=1.0`,
  `cost_actor_coef=0.5`, `unsafe_penalty=10.0`, `reward_scale=20.0`;
- policy: same `QpFreePolicy`, hidden 256, correction radius 0.3, from
  scratch (`init_checkpoint=null`);
- h: `outputs/v6/independent/h_c0/seed0/v6_h_c0.pkl`, frozen;
- h offset: from the h2 checkpoint (`+0.0189005`), not overridden;
- gate: `low=0.0`, `high=0.6620830297470093` (the h2 q10-safe calibration),
  `positive_only=true`, `scope=progress_only`; no sweep;
- actor features: h/grad_h enabled, `feature_scale=1.0`;
- no task scalar, no task gradient, no h co-training, no CBF loss;
- evaluation: 100 deterministic episodes, seeds `10000..10099`, gate
  disabled, fixed final checkpoint;
- training-time monitor eval: 20 episodes at 8,192 steps, seed 10000; it does
  not select or alter the final checkpoint.

Config: `nav/configs/v6/v6_nav_c1a_h2_actor2.yaml`.

## 4. Baselines

| arm | safe_ep | official_safety | success | reach | return |
|---|---:|---:|---:|---:|---:|
| actor1 seed0 + h1 (formal) | 0.840 | 0.97375 | 0.4275 | 0.4350 | 60.91 |
| actor1 seed0 + h2 (post-hoc swap) | 0.900 | 0.98500 | 0.43875 | 0.44625 | 61.40 |

The actor1+h2 row is a diagnostic, not a training result; it is shown to
separate "h2 with the old actor" from "h2 with a newly trained actor."

## 5. Pre-registered pilot rule

The pilot passes only if **all** of the following hold on the 100-episode
deterministic evaluation:

1. `official_safety >= 0.978` (at least +0.00425 over actor1 seed0);
2. `safe_episode >= 0.84` (not worse than actor1 seed0);
3. `official_success >= 0.40` (no more than 0.0275 below actor1 seed0);
4. `official_reach >= 0.40` (no more than 0.035 below actor1 seed0);
5. `episode_return >= 58.0` (no more than 2.91 below actor1 seed0).

These are one-seed pilot guardrails: a directional safety signal plus no task
collapse.  They are not a final claim and do not replace the 3-seed Stage-B
protocol.

If the pilot passes, C1b (seeds 1/2) may be proposed, but it requires a
separate decision because matched h2 seeds 1/2 do not exist.  The options are
(a) train h2 seeds 1/2 (the original C0b), or (b) pre-register a fixed-h2
protocol using h2 seed 0 for all actor seeds.  Neither is automatic.

If the pilot fails, stop the h2/actor2 branch and move to the next controlled
arm (safety coupling or action conditioning), with explicit approval.

## 6. Result (C1a seed 0): FAILED

Final 100-episode deterministic evaluation (eval seed 10000, gate off):

| metric | actor1 + h1 | C1a actor2 + h2 | delta |
|---|---:|---:|---:|
| safe_episode | 0.840 | **0.510** | -0.330 |
| collision_episode | 0.160 | **0.490** | +0.330 |
| warning_episode | 0.210 | **0.510** | +0.300 |
| official_safety | 0.97375 | **0.89375** | -0.08000 |
| official_success | 0.42750 | **0.33125** | -0.09625 |
| official_reach | 0.43500 | **0.36250** | -0.07250 |
| episode_return | 60.91 | **52.22** | -8.68 |
| h_mean | 0.836 | 0.898 | +0.062 |
| h_min | 0.648 | 0.664 | +0.015 |
| cbf_violation | 0.0446 | 0.0768 | +0.0322 |

All five checks failed:

- `official_safety 0.89375 < 0.978`;
- `safe_episode 0.510 < 0.840`;
- `official_success 0.33125 < 0.400`;
- `official_reach 0.36250 < 0.400`;
- `episode_return 52.22 < 58.0`.

The safe-episode drop is about `z=-5.3` under a binomial approximation, so it
is not one-seed noise.  The 20-episode training-time monitor at 16k was even
worse (`safe 0.40`, `official_safety 0.881`); the 100-episode result is the
pre-registered one.

### Interpretation

- The from-scratch actor2 did not merely lose task performance: it collided
  more (`collision_episode 0.49` vs `0.16`) and had lower official safety.
- The gate is not the main cause.  Training logs show
  `task_gate_mean ≈ 0.99` and `task_gate_closed ≈ 0.2%`, so dense progress
  reward was almost always available.
- The h2 tail miscalibration documented in C0a (17-32-step, 9-16-step, and
  impact MAE worse than h1) is consistent with the actor treating risky
  states as safer.  With one seed, this is a plausible mechanism, not a
  proven causal explanation.
- The post-hoc actor1+h2 h-swap was slightly better than h1, so h2 alone does
  not force a worse closed-loop result.  The failure is specific to training
  actor2 from scratch under h2; possible causes include the h2 input
  distribution, the actor's learned use of h/grad_h, and one-seed
  optimization variance.
- **Decision: stop the h2/actor2 from-scratch branch.  Do not run C1b.**
  C1w (4k warm-start of actor1 with h2) is a separate, explicitly approved
  diagnostic adaptation arm; it is not a C1a pass and cannot revive C1b.

## 7. Artifacts

- actor2 checkpoint: `outputs/v6/independent/nav_c1a_h2_actor2/seed0/
  policy_final.pth`
- 100ep evaluation: `outputs/v6/independent/nav_c1a_h2_actor2/seed0/
  eval_100ep_final.json`
- pilot decision: `outputs/v6/independent/nav_c1a_h2_actor2/seed0/
  pilot_decision.json` (passed=false)
- training log: `outputs/v6/independent/nav_c1a_h2_actor2/seed0/train.log`
- evaluation log: `outputs/v6/independent/nav_c1a_h2_actor2/seed0/
  eval_100ep.log`
- config snapshot: `outputs/v6/independent/nav_c1a_h2_actor2/seed0/
  config.yaml`

## 8. Command

Full pilot (requires explicit confirmation):

```bash
CONFIRM=1 bash nav/scripts/run_v6_nav_c1a_h2_actor2.sh
```

Smoke (pipeline check only, no claim):

```bash
SMOKE=1 bash nav/scripts/run_v6_nav_c1a_h2_actor2.sh
```

The runner refuses the full pilot without `CONFIRM=1`.  It never loads a
warm-start checkpoint and never trains h.

## 9. Status

- [x] Protocol pre-registered.
- [x] User approval for the seed-0 pilot.
- [x] Smoke test (256 steps + 2ep eval): pipeline completes; metrics are not
      evidence.
- [x] Full actor2 seed-0 training: complete.
- [x] 100ep evaluation and pilot decision: **FAILED all five checks**.
      `pilot_decision.json` has `passed=false`.
- [ ] C1b: **not authorized**; the from-scratch h2/actor2 branch is stopped.
