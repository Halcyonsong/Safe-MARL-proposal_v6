# V6 navigation C1w: actor1 + h2 4k warm-start (diagnostic)

Status: **complete; failed the C1a guardrails, so `promising_adaptation =
false`.**  The 100ep task collapses (success `0.2025` vs actor1+h1 `0.4275`,
return `28.85` vs `60.91`) while safety only matches actor1+h1 and stays below
the no-training h-swap.  Per the pre-registered staged rule, the h1-continued
control is **not run** and the h2 warm-start branch is stopped.  C1a has also
failed.  C1w is a separate diagnostic adaptation arm; it is not a C1a pass.

Canonical status: [`../../STATUS.md`](../../STATUS.md).  C1a protocol:
[`STAGE_C1A_H2_ACTOR2_PILOT.md`](STAGE_C1A_H2_ACTOR2_PILOT.md).  C0a and the
post-hoc h-swap: [`STAGE_C0_H2_TEMPORAL.md`](STAGE_C0_H2_TEMPORAL.md),
[`../nav_c0_h2_actor1_swap/README.md`](../nav_c0_h2_actor1_swap/README.md).

## 1. Why this arm exists

C1a trained actor2 from scratch with h2 seed 0.  Its 20-episode training-time
monitor already showed a large failure:

| metric | actor2 @16k | actor1 @16k |
|---|---:|---:|
| safe_episode | 0.400 | 0.900 |
| collision_episode | 0.600 | 0.100 |
| official_safety | 0.881 | 0.981 |
| official_success | 0.313 | 0.488 |
| episode_return | 48.06 | 64.74 |

The final 100-episode C1a evaluation confirmed the failure: safe `0.510`,
official safety `0.89375`, success `0.33125`, reach `0.36250`, return `52.22`,
collision `0.49`; all five pre-registered checks failed.  The from-scratch
h2/actor2 branch is stopped.

The post-hoc no-training h-swap of h2 into actor1 seed 0 was slightly better
than h1:

| metric | actor1 + h1 | actor1 + h2 (no training) |
|---|---:|---:|
| safe_episode | 0.840 | 0.900 |
| official_safety | 0.97375 | 0.98500 |
| official_success | 0.42750 | 0.43875 |
| official_reach | 0.43500 | 0.44625 |
| episode_return | 60.91 | 61.40 |

This is one seed and not significant, but it suggests actor1 tolerates h2
without the from-scratch collapse.  C1w asks whether a short continuation of
actor1 with h2 makes the pair better matched than the no-training swap.

The user explicitly chose the **h2 arm only, with no h1-continued control**.
Therefore C1w is an adaptation diagnostic, not a causal comparison of h2
against h1.  Any change could come from the extra 4,096 training steps.

## 2. Scope

In scope:

- initialize from the fixed V6 actor1 seed-0 final checkpoint;
- freeze h2 seed 0;
- continue for exactly 4,096 additional steps (16384 -> 20480);
- same cost, network, optimizer, seed, gate calibration, and evaluation as
  C1a;
- 100 deterministic episodes at eval seed 10000, gate disabled;
- compare with actor1+h1 and the no-training actor1+h2 swap.

Not in scope:

- no h1-continued control;
- no from-scratch claim;
- no h2 seeds 1/2 (C0b), no multi-seed claim;
- no gate sweep, gate-scope change, cost change, task channel, QP, action
  projection/filter, action conditioning, or h feature-scale change;
- no formal/CBF-certified safety claim.

## 3. Fixed protocol

- resume checkpoint:
  `outputs/v6/independent/nav_stageb_repair_16k/seed0/policy_final.pth`
  (steps = 16,384)
- h: `outputs/v6/independent/h_c0/seed0/v6_h_c0.pkl`, frozen
- gate: `low=0.0`, `high=0.6620830297470093`, `positive_only=true`,
  `scope=progress_only`
- additional steps: `4,096`; total steps `20,480`
- `rollout_steps=128`, `epochs=4`, `minibatch_size=256`, `lr=3e-4`,
  `cost_coef=1.0`, `cost_actor_coef=0.5`, `unsafe_penalty=10.0`,
  `reward_scale=20.0`
- policy: same `QpFreePolicy`, hidden 256, correction radius 0.3
- h/grad_h features enabled, `feature_scale=1.0`
- monitor eval: 20 episodes at steps 18,432 and 20,480, seed 10000; it does
  not select or alter the final checkpoint
- final evaluation: 100 deterministic episodes, seeds `10000..10099`, gate
  off, final checkpoint

Config: `nav/configs/v6/v6_nav_c1w_h2_actor1_4k.yaml`.
Runner: `nav/scripts/run_v6_nav_c1w_h2_actor1_4k.sh`.
Summarizer: `nav/scripts/summarize_v6_nav_c1w.py`.

## 4. Diagnostic rule

C1w uses the same five one-seed guardrails as C1a, relative to actor1+h1:

1. `official_safety >= 0.978`;
2. `safe_episode >= 0.84`;
3. `official_success >= 0.40`;
4. `official_reach >= 0.40`;
5. `episode_return >= 58.0`.

In addition, the summarizer reports whether C1w improves on the no-training
h-swap:

- `beats_swap_safety`: official safety and safe episode are at least the
  swap values;
- `task_not_worse_than_swap`: success/reach/return are within the same C1a
  task margins of the swap.

`promising_adaptation` is true only if all three hold: C1a guardrails, swap
safety improvement, and no task regression versus the swap.  Even then it is
a one-seed diagnostic, not a formal pass and not evidence that h2 is better
than h1.

### Staged ablation rule (pre-registered before seeing C1w)

C1w has no h1-continued control, so a gain could come from h2 or from the
extra 4,096 training steps.  The control is run only if C1w first shows a
signal:

- **C1w `promising_adaptation = true`:** run the matched h1-continued control
  (same actor1 seed-0 init, same 4,096 steps, same cost/network/seed, h1 and
  h1 gate high `0.267270`, 100ep eval seed 10000).  Only then can the h2
  effect be separated from extra training.
- **C1w meets guardrails but does not beat the no-training swap:** do not run
  the control; the 4k continuation did not add value beyond simply swapping
  h, so the cheaper swap remains the diagnostic.
- **C1w fails the C1a guardrails:** stop the h2 continuation branch; do not
  run the control.

This rule is fixed before the C1w result.  The control is an ablation, not a
new promotion candidate; it remains one seed and is still not a from-scratch
V6 result.

### Result (100ep, final checkpoint 20,480, gate off)

| metric | actor1+h1 | actor1+h2 swap | C1w actor1+h2 4k | delta vs h1 | delta vs swap |
|---|---:|---:|---:|---:|---:|
| safe_episode | 0.840 | 0.900 | 0.840 | 0.000 | -0.060 |
| collision_episode | 0.160 | 0.100 | 0.160 | 0.000 | +0.060 |
| official_safety | 0.97375 | 0.98500 | 0.97625 | +0.00250 | -0.00875 |
| official_success | 0.42750 | 0.43875 | 0.20250 | -0.22500 | -0.23625 |
| official_reach | 0.43500 | 0.44625 | 0.20625 | -0.22875 | -0.24000 |
| episode_return | 60.91 | 61.40 | 28.85 | -32.06 | -32.54 |
| action_abs | 0.2769 | 0.2749 | 0.2181 | - | - |

Guardrail checks: `official_safety` **fail** (`0.97625 < 0.978`),
`safe_episode` pass (`0.84 >= 0.84`), `official_success` **fail**,
`official_reach` **fail**, `episode_return` **fail**.  `beats_swap_safety =
false`, `task_not_worse_than_swap = false`, `promising_adaptation = false`.

The failure is a **task collapse with mean-action shrink** (`action_abs`
`0.277 -> 0.218`), not a safety gain: safe episodes and collisions are exactly
the actor1+h1 baseline, and official safety is below the no-training swap.
This matches the 20ep monitor and the earlier cause analysis.  Because the
guardrails fail, the staged rule says: **do not run the h1-continued control;
stop the h2 warm-start branch.**  The extra 4k steps are confounded with h2,
so this result cannot be attributed to h2 causally.

## 5. Expected outcomes

- **Promising adaptation:** the short continuation recovers/improves safety
  beyond the swap without task drift.  It would justify a controlled
  h1-continued arm and/or a multi-seed version, not a formal claim.
- **Meets guardrails but does not beat the swap:** the h2 continuation did
  not make the pair better matched than simply swapping h.  The no-training
  swap remains the cheaper diagnostic; no further warm-start work.
- **Fails guardrails or task margins:** the actor drifted or learned to trust
  h2's optimistic tail.  Stop the h2 warm-start branch.

## 6. Artifacts

- checkpoint: `outputs/v6/independent/nav_c1w_h2_actor1_4k/seed0/
  policy_final.pth`
- 100ep evaluation: `outputs/v6/independent/nav_c1w_h2_actor1_4k/seed0/
  eval_100ep_final.json`
- diagnostic decision: `outputs/v6/independent/nav_c1w_h2_actor1_4k/seed0/
  diagnostic_decision.json`
- logs: `train.log`, `eval_100ep.log`, `diagnostic_decision.log`
- config snapshot: `outputs/v6/independent/nav_c1w_h2_actor1_4k/seed0/
  config.yaml`

## 7. Command

Full run (requires explicit confirmation):

```bash
CONFIRM=1 bash nav/scripts/run_v6_nav_c1w_h2_actor1_4k.sh
```

Smoke (pipeline only, 128 additional steps + 2ep eval; no claim):

```bash
SMOKE=1 bash nav/scripts/run_v6_nav_c1w_h2_actor1_4k.sh
```

The runner refuses the full run without `CONFIRM=1`, refuses a missing resume
checkpoint or h2 checkpoint, and refuses to overwrite an existing result.

## 8. Status

- [x] Protocol pre-registered.
- [x] User approval for the 4k h2-only warm-start (no control).
- [x] C1a 100ep final result recorded: failed all five checks.
- [x] C1w smoke test (128 additional steps + 2ep eval): pipeline completes;
      metrics are not evidence.
- [x] C1w 4k continuation: complete (16,384 -> 20,480).
- [x] C1w 100ep evaluation and diagnostic summary: **failed**.  safe `0.840`,
      official safety `0.97625`, success `0.20250`, reach `0.20625`, return
      `28.85`; `promising_adaptation = false`.
- [x] Matched h1-continued control: **not run**, because the pre-registered
      trigger (`promising_adaptation = true`) did not fire.
- [x] h2 warm-start branch stopped.
- [ ] Decision on any multi-seed follow-up (separate approval; not proposed
      for this stopped branch).
