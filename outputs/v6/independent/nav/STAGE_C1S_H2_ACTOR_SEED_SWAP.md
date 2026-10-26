# V6 navigation C1s: h2 seed-0 + actor1 seed replication (no-training swap)

Status: **complete (2026-10-26): `consistent_positive`, but only exactly at
the pre-registered `mean_delta_safe >= +0.03` boundary.**  This is an
evaluation-only replication of the post-hoc actor1+h2 h-swap.  It does not
train an actor, train an h, or change the gate.  It is not a C0/C1 pass and
does not support a causal h2 claim.

Canonical status: [`../../STATUS.md`](../../STATUS.md).  Seed-0 swap:
[`../nav_c0_h2_actor1_swap/README.md`](../nav_c0_h2_actor1_swap/README.md).
C0a/C1a/C1w context:
[`STAGE_C0_H2_TEMPORAL.md`](STAGE_C0_H2_TEMPORAL.md),
[`STAGE_C1A_H2_ACTOR2_PILOT.md`](STAGE_C1A_H2_ACTOR2_PILOT.md),
[`STAGE_C1W_H2_ACTOR1_4K.md`](STAGE_C1W_H2_ACTOR1_4K.md).

## 1. Question

The only positive h2 closed-loop signal so far is the seed-0 no-training
h-swap: actor1 seed 0 + h2 seed 0 was slightly better than actor1 + h1
(safe `0.90` vs `0.84`, official safety `0.985` vs `0.974`, success `0.439`
vs `0.428`, return `61.40` vs `60.91`).  It is one seed and not significant.

C1s asks a narrow question:

> Does the h2 seed-0 swap signal survive on actor1 seeds 1 and 2, using the
> same h2 checkpoint and the same 100-episode evaluation?

It does **not** ask whether h2 is better for a newly trained actor, whether
h2 seed 0 generalizes to other h2 seeds, or whether the h2/actor iteration
works.  Those questions are already closed or out of scope.

## 2. Scope

In scope:

- fixed final Stage-B actor1 seeds 1 and 2;
- frozen C0 h2 seed 0 instead of h1;
- 100 deterministic episodes, eval seeds `10000..10099`, gate off;
- compare each swap with that actor seed's existing h1 baseline;
- report per-seed and two-seed mean deltas;
- a pre-registered consistency classification.

Not in scope:

- no actor training, no h training, no warm start, no continuation;
- no C0b h2 seeds 1/2;
- no from-scratch actor2 / C1b;
- no gate sweep, task channel, QP, action projection/filter, or action
  conditioning;
- no formal/CBF-certified safety claim.

## 3. Fixed protocol

Actor checkpoints:

- seed 1: `outputs/v6/independent/nav_stageb_repair_16k/seed1/policy_final.pth`
- seed 2: `outputs/v6/independent/nav_stageb_repair_16k/seed2/policy_final.pth`

Both are the fixed final Stage-B actor1 checkpoints (steps = 16,384).

h checkpoint:

- `outputs/v6/independent/h_c0/seed0/v6_h_c0.pkl` (frozen h2 seed 0;
  gate calibration `high=0.6620830297470093`; the gate is not active during
  evaluation)

Evaluation:

- `nav/scripts/evaluate_v6.py`
- deterministic actions (no `--stochastic`)
- 100 episodes, eval seed `10000` (seeds `10000..10099`)
- device `cuda`
- same environment/config as the checkpoint's formal Stage-B evaluation
- outputs:
  - `outputs/v6/independent/nav_c1s_h2_actor_seed_swap/seed1/`
  - `outputs/v6/independent/nav_c1s_h2_actor_seed_swap/seed2/`

Baselines (already complete, same protocol):

| actor seed | h1 safe | h1 official safety | h1 success | h1 reach | h1 return |
|---|---:|---:|---:|---:|---:|
| 1 | 0.51 | 0.90000 | 0.4575 | 0.49625 | 70.91862 |
| 2 | 0.73 | 0.95125 | 0.3450 | 0.35750 | 68.73629 |

Seed-0 reference (not part of the primary replication test):

| arm | safe | official safety | success | reach | return |
|---|---:|---:|---:|---:|---:|
| actor1 seed 0 + h1 | 0.84 | 0.97375 | 0.42750 | 0.43500 | 60.90853 |
| actor1 seed 0 + h2 seed 0 | 0.90 | 0.98500 | 0.43875 | 0.44625 | 61.39588 |

## 4. Pre-registered consistency rule

Primary replication set: actor seeds **1 and 2**.  Seed 0 is a reference and
is not allowed to override the primary classification.

For each seed `s in {1,2}`, define:

- `delta_safe(s) = safe_episode_swap - safe_episode_h1`
- `delta_official_safety(s) = official_safety_swap - official_safety_h1`
- `delta_success(s)`, `delta_reach(s)`, `delta_return(s)` analogously.

`consistent_positive` is true only if all of the following hold:

1. `delta_safe(s) >= 0` for both seeds;
2. `delta_official_safety(s) >= 0` for both seeds;
3. mean `delta_safe >= +0.03`;
4. mean `delta_official_safety >= +0.005`;
5. mean `delta_success >= -0.0275`;
6. mean `delta_reach >= -0.035`;
7. mean `delta_return >= -2.91`;
8. no seed has `delta_success < -0.05` or `delta_return < -5.0`.

Classification:

- **consistent_positive:** all eight checks hold.  The seed-0 swap signal
  survives actor seeds 1/2.  This is a robustness signal for h2 seed 0 as a
  frozen input, not a formal pass, not evidence that h2 is better in general,
  and not a from-scratch V6 result.
- **mixed:** not `consistent_positive`, but at least one seed has
  `delta_safe > 0` and `delta_official_safety > 0`, and no seed violates the
  catastrophic-task guard (check 8).
- **negative:** everything else, including a catastrophic task regression or
  no positive safety signal on either seed.

The two-seed mean is the primary summary.  A three-seed mean including seed 0
is reported as secondary only.

### Result (100ep, fixed final actor1, frozen h2 seed 0, gate off)

| metric | seed 1 h1 | seed 1 + h2 | Δ1 | seed 2 h1 | seed 2 + h2 | Δ2 | mean Δ (primary) |
|---|---:|---:|---:|---:|---:|---:|---:|
| safe_episode | 0.51 | 0.56 | +0.05 | 0.73 | 0.74 | +0.01 | **+0.03** |
| collision_episode | 0.49 | 0.44 | -0.05 | 0.27 | 0.26 | -0.01 | -0.03 |
| official_safety | 0.90000 | 0.91250 | +0.01250 | 0.95125 | 0.95625 | +0.00500 | **+0.00875** |
| official_success | 0.45750 | 0.49125 | +0.03375 | 0.34500 | 0.33875 | -0.00625 | +0.01375 |
| official_reach | 0.49625 | 0.52750 | +0.03125 | 0.35750 | 0.34250 | -0.01500 | +0.00813 |
| episode_return | 70.92 | 74.59 | +3.67 | 68.74 | 64.97 | -3.77 | -0.05 |
| warning_episode | 0.59 | 0.45 | -0.14 | 0.28 | 0.26 | -0.02 | -0.08 |

All eight pre-registered checks hold, so the classification is
**`consistent_positive`**:

- both seeds non-negative on `safe_episode` and `official_safety`;
- mean `Δsafe = +0.030` (met the `>= +0.03` threshold exactly, not
  comfortably);
- mean `Δofficial_safety = +0.00875 >= +0.005`;
- mean task deltas pass (`Δsuccess +0.01375`, `Δreach +0.00813`,
  `Δreturn -0.05`), and no seed has a catastrophic regression
  (seed 2 return `-3.77`, within the `-5.0` guard).

Secondary three-seed mean including seed 0: `Δsafe +0.040`,
`Δofficial_safety +0.00958`, `Δsuccess +0.01292`, `Δreach +0.00917`,
`Δreturn +0.129`.

The signal is not uniform: seed 1 provides most of the safety and task gain
(+0.05 safe, +0.0125 official safety, +3.67 return); seed 2 is marginal
(+0.01 safe, +0.005 official safety, -3.77 return).  Both are non-negative on
safety, but the two-seed result is a thin, exactly-at-threshold pass rather
than a strong replication.

## 5. Interpretation

- **consistent_positive:** h2 seed 0 is not obviously seed-0-actor-specific
  in this narrow off-policy swap.  It would justify discussing a matched
  h2-seed study (C0b), not a main claim.  It would not revive C1b or the
  from-scratch actor2 branch.
- **mixed / negative:** the seed-0 +6 pp safe result is not robust enough to
  pursue.  Stop the h2 pairing line and move to safety coupling or action
  conditioning, with explicit approval.

## 6. Caveats

- The swap is **off-policy**: actor1 seeds 1/2 were trained with h1, not h2.
- h2 seed 0 was trained on actor1 seed 0 data plus fixed random coverage.
  Actor1 seeds 1/2 are distribution-shifted for h2, so a negative result is
  ambiguous (it could reflect the OOD h input rather than h2 being bad).
- Including seed 0, `n=3`; the seed-0 difference is small (`+0.06` safe,
  `+0.01125` official safety) and not statistically significant.
- The two-seed `consistent_positive` classification is exactly on the
  pre-registered `mean_delta_safe >= +0.03` boundary.  It should not be
  described as a comfortable or strong pass.
- The result uses one h2 checkpoint (seed 0) and does not test h2 seed
  reproducibility or a from-scratch actor2.
- The gate is off at evaluation; C1s does not test gate calibration.
- C1s is a diagnostic.  It is not a C0/C1 pass and cannot be used as a main
  V6 navigation claim.

## 7. Artifacts

- pre-registration: this file
- runner: `nav/scripts/run_v6_nav_c1s_h2_actor_seed_swap.sh`
- summarizer: `nav/scripts/summarize_v6_nav_c1s.py`
- seed1: `outputs/v6/independent/nav_c1s_h2_actor_seed_swap/seed1/
  eval_100ep_final.json`, `eval_100ep.log`
- seed2: `outputs/v6/independent/nav_c1s_h2_actor_seed_swap/seed2/
  eval_100ep_final.json`, `eval_100ep.log`
- decision: `outputs/v6/independent/nav_c1s_h2_actor_seed_swap/decision.json`

## 8. Status

- [x] Protocol pre-registered.
- [x] Seed 1 100ep evaluation: safe `0.56`, official safety `0.91250`,
      success `0.49125`, reach `0.52750`, return `74.59`.
- [x] Seed 2 100ep evaluation: safe `0.74`, official safety `0.95625`,
      success `0.33875`, reach `0.34250`, return `64.97`.
- [x] Consistency summary and decision: `consistent_positive` (all eight
      checks hold; mean `Δsafe` exactly `+0.03`).
- [x] Documentation update.
- [ ] Matched h2-seed study (C0b): only a discussion item; requires a new
      pre-registration and explicit approval.  Not launched.
