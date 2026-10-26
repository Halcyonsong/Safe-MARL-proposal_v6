# V6 navigation h2/actor2 diagnostic (read-only)

Status: complete, read-only.  No training, no checkpoint selection, no gate or
cost change, no task channel.

Canonical status: [`../../STATUS.md`](../../STATUS.md).  Protocol that produced
the audited checkpoints:
[`STAGE_B_REPAIR_MULTISEED.md`](STAGE_B_REPAIR_MULTISEED.md).

## 1. Question and pre-registered rule

The repaired Stage-B gate is promoted, but collisions remain.  This audit asks
whether a collision onset happens in a state the frozen h1 still calls safe
(an **h-limited** failure), or in a state h1 already flags and the gate
already closes (an **actor/action-limited** failure).

The rule was fixed before the audit:

- **h-limited**: at least 50% of collision onsets have `h_cur >= gate_high`
  or `gate_next` open, **and** calibration on actor1 states degrades (pooled
  target MAE `>= 0.20`, or safe-row target correlation drop `>= 0.10` versus
  the offline repaired test split).
- **h-safe-labeled**: the same high/gate-open collision pattern, but
  calibration does not degrade.  h2 is plausible, but the policy/target
  mismatch may need action conditioning.
- **actor/action-limited**: at least 50% of collision onsets have
  `h_cur <= 0` or `gate_next` closed, without calibration degradation.
- otherwise **mixed**.

Target semantics match h training: wall-free `tanh(future_min_margin / 0.2)`,
horizon 32, horizon-unsafe rows forced to `<= -0.1`.

## 2. Protocol

- Fixed final actor1 checkpoint of each repaired Stage-B seed: `seed{0,1,2}/
  policy_final.pth`.  No selection on the formal test set.
- Frozen repaired h: `h_stageb_repair/v6_h_stageb_repair.pkl`.
- 100 deterministic episodes per seed, eval seeds `10000..10099`, as in the
  formal evaluation.
- Gate is disabled at evaluation, exactly as in the promoted protocol.  The
  gate values below are therefore **diagnostic** values computed from
  `h_next`; they are not applied to the evaluated actions.
- 614,400 agent-step rows saved to `.npz`; collision onsets are the first step
  of each contiguous collision run for an agent (persistent contact counts
  once).

## 3. Reproduction of the formal actor1 result

The audit replays the same deterministic trajectories and reproduces the
formal 100-episode evaluation exactly for `safe_episode`,
`official_safety`, `official_success`, `official_reach`, and `episode_return`.
The first pass had a diagnostic-only return-accounting bug (the reach bonus
was paid once instead of per step); the final pass uses the same per-agent
accumulation as `evaluate_v6`, was checked against a 1-episode `evaluate_v6`
run, and then matched all three formal 100-episode references to
`< 1e-9`.  Safety/h/gate metrics were unaffected by that bug.

| seed | safe_ep | official_safety | success | reach | return | collision ep | onsets |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.840 | 0.9738 | 0.4275 | 0.4350 | 60.91 | 16 | 21 |
| 1 | 0.510 | 0.9000 | 0.4575 | 0.4963 | 70.92 | 49 | 90 |
| 2 | 0.730 | 0.9513 | 0.3450 | 0.3575 | 68.74 | 27 | 44 |
| **mean** | **0.693** | **0.9417** | **0.4100** | **0.4296** | **66.85** | **30.7** | **51.7** |

## 4. Collision-state evidence

`h_cur` is the provider-space h at the pre-action state; `gate_next` is the
gate at the next state.  `gate_high = 0.26726993918418884`.

| seed | onsets | h_cur median | h_cur >= high | h_cur > 0 | gate open | gate closed | h_next median |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 21 | -0.1833 | 0.000 | 0.000 | 0.000 | 1.000 | -0.1942 |
| 1 | 90 | -0.1960 | 0.000 | 0.000 | 0.000 | 1.000 | -0.2093 |
| 2 | 44 | -0.2053 | 0.000 | 0.000 | 0.000 | 1.000 | -0.2248 |

Pooled: **155/155 onsets (100%) occur with `h_cur < 0` and
`gate_next = 0`.**  None occur with `h_cur >= gate_high`; none with the gate
open.  The range of `h_cur` at collision is `[-0.273, -0.127]`.

This is the direct signature of an **actor/action-limited** failure: h1 is not
blind at the moment of impact, and the diagnostic gate is already fully
closed.  The actor collides anyway.

## 5. Warning lead

h is already negative at impact by construction of the target, so the more
informative quantity is how early h drops.  Pre-collision windows over the
155 onsets:

| lag (steps) | mean h | fraction h > 0 | fraction h >= high |
|---:|---:|---:|---:|
| 0 | -0.198 | 0.000 | 0.000 |
| 1 | -0.172 | 0.000 | 0.000 |
| 2 | -0.136 | 0.013 | 0.000 |
| 3 | -0.090 | 0.090 | 0.000 |
| 4 | -0.036 | 0.290 | 0.000 |
| 5 | +0.018 | 0.581 | 0.019 |

At five steps (0.15 s) before collision, h is approximately zero and only
1.9% of onsets are still above `gate_high`.  A cluster-level reconstruction of
the saved rows puts the last `h >= gate_high` at a median of **7 steps** before
collision (IQR 6-11) and the last `h > 0` at a median of **2 steps** (IQR 1-3);
the exact lag table above is the primary evidence.  All 155 onsets have at
least five preceding steps, so none is a reset-time contact.

## 6. Calibration on actor1 states

`target MAE/RMSE` are the primary calibration check; correlation is attenuated
by the narrower target range on actor1 states (std 0.178 versus 0.378 on the
offline test split).

| source | target corr | safe corr | MAE | RMSE | target std | BA | AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| offline repaired test | 0.7940 | 0.7364 | 0.1510 | 0.2456 | 0.3783 | 0.9926 | 0.9997 |
| actor1 states, seed 0 | 0.6961 | 0.7016 | 0.0925 | 0.1454 | 0.1454 | — | — |
| actor1 states, seed 1 | 0.7067 | 0.6853 | 0.1035 | 0.1698 | 0.2094 | — | — |
| actor1 states, seed 2 | 0.7145 | 0.7086 | 0.0969 | 0.1557 | 0.1714 | — | — |
| **actor1 pooled** | **0.7040** | **0.6968** | **0.0976** | **0.1573** | **0.1779** | **0.9978** | **0.9999** |

Pooled target MAE is **better** than the offline repaired test MAE
(0.0976 versus 0.1510).  The safe-row correlation drop is 0.040, below the
pre-registered 0.10 threshold.  By the pre-registered pooled calibration
criterion, calibration does **not** degrade.  That criterion is dominated by
far-safe rows; the risk-region analysis in sections 7-8 is the more relevant
test for the 32-step warning.

Future-collision AUROC (h as a ranker of "collision within 32 steps") is
0.908 pooled; the current-unsafe/warning-label AUROC is effectively 1.0.

## 7. Risk-region calibration and the temporal caveat

The global MAE is dominated by easy safe rows.  In the regions that matter:

| region | rows | h mean | target mean | MAE | corr |
|---|---:|---:|---:|---:|---:|
| target < 0.5 | 25,636 | +0.324 | +0.186 | 0.309 | 0.286 |
| h < gate high | 18,720 | +0.085 | +0.285 | 0.262 | 0.765 |
| future collision (32 steps) | 6,285 | +0.158 | -0.215 | 0.394 | 0.094 |
| current unsafe label | 2,669 | -0.214 | -0.209 | 0.053 | 0.411 |

`current unsafe label` is the h-provider warning/unsafe mask used by h
training, not the collision flag; `future collision` is the actual collision
flag.  The h target itself is the wall-free future-min margin, so the
future-collision row is the relevant risk-region check.

There is a genuine **temporal** limitation: on actor1 states, h1 does not
reliably predict the 32-step target at long lead.  Approximate time-to-next-
collision profile from the saved 32-step collision flags (collisions within
32 steps merge into one cluster, so the 143 at-distance-0 rows are fewer than
the 155 raw onsets):

| steps to collision | rows | h mean | h < 0 | h < gate high | gate closed |
|---|---:|---:|---:|---:|---:|
| 0 | 143 | -0.203 | 1.000 | 1.000 | 0.979 |
| 1 | 143 | -0.224 | 1.000 | 1.000 | 1.000 |
| 2-4 | 429 | -0.225 | 1.000 | 1.000 | 1.000 |
| 5-8 | 572 | -0.184 | 0.927 | 0.993 | 0.960 |
| 9-16 | 1,144 | -0.050 | 0.689 | 0.878 | 0.731 |
| 17-32 | 2,165 | +0.242 | 0.319 | 0.545 | 0.343 |
| > 32 / none | 609,804 | +0.832 | 0.003 | 0.025 | 0.003 |

So h1 is a short-horizon signal on this distribution: it is usually negative
inside 8-16 steps, but positive and only partially gated at 17-32 steps even
though the target is negative.  This is consistent with the future-collision
MAE of 0.394.  The likely cause is the training-distribution mismatch discussed
in section 8: h1 was trained on h-only/Stage-A rollouts, not final actor1
states.  This is a real h2 motivation, even though at impact h1 and the
diagnostic gate have already flagged the state.

## 8. Data-source mismatch: why the impact-state label is not sufficient

The impact-state rule above asks only whether h1 is blind **at the collision
state**.  It does not ask whether h1 warned early enough.  The h1 training
data makes that distinction important:

- `data_stageb/summary.json` records the behavior policies as
  `formal_3seed/h_only/seed0/policy_final.pth` and
  `nav_gate_progress_16k/seed0/policy_final.pth`, both evaluated with the
  original `h/v6_h.pkl`.  These are **not** the final repaired Stage-B actor1.
- The repaired h training mix is 50% rebuilt Stage-B data and 50% random
  coverage (`data_stageb_repair/summary.json`).  No final actor1 rollouts were
  used.
- The final actor1 is therefore off-policy relative to h1.  It can visit
  states and use h1 in ways the h1 training distribution did not cover.

The audit sees exactly this signature.  Global MAE is dominated by far-safe
rows and looks better than offline test (0.0976 versus 0.1510), but the
risk-region error is large: future-collision MAE 0.394, and in the 17-32 steps
before collision h1 is positive (`+0.242`) while the 32-step target is
negative (`-0.222`).  h1 has learned the imminent part of the target on actor1
states, but not the long-horizon part.

This does not contradict the impact-state result; it changes its scope.  The
correct overall reading is **mixed**:

- **temporal h-limited before impact** (the h1 training distribution does not
  match actor1, so the warning is late);
- **actor/action-limited at impact** (the actor still fails after h1 turns
  negative and the diagnostic gate closes).

The user-facing implication is important: iterating h on actor1 data is a
legitimate way to close this gap, and the pre-registered impact-state label
should not be used to rule it out.

## 9. Decision and staged recommendation

**Pre-registered impact-state label: actor/action-limited (h2 likely
marginal).**  Every recorded collision onset is in an h1-flagged, gate-closed
state, and the pre-registered pooled calibration check does not degrade.

**Overall diagnosis: mixed, with a clear temporal h limitation.**  The
data-source mismatch is real, and the risk-region evidence supports the
hypothesis that h1's information is late relative to the actor1 policy.

**Recommendation: run the iteration in two stages, not as one confounded
jump.**

1. **Stage C0 — h2-only temporal-calibration screen (no actor2, no closed-loop
   training).**
   - For each seed, collect actor1 stochastic rollouts plus fixed random
     coverage; rebuild the same wall-free 32-step target.
   - Train h2 from scratch with the same architecture, loss, and calibration
     rule; no gate-scope, cost, or task-channel change.
   - Evaluate on held-out actor1 episodes (new seeds, not the formal 100ep
     test episodes): 17-32-step risk-region MAE, warning lead (last
     `h >= gate_high` before collision), future-collision AUROC, and global
     MAE/BA.  The formal 100ep audit is only a final read-only confirmation,
     never a C0 selection set.
   - Pre-register a pass rule, e.g. at least 20% lower 17-32-step risk-region
     MAE and a longer warning lead, without degrading global MAE/BA.
   - This directly tests the user's hypothesis: does actor1 data close the
     long-horizon h gap?
2. **Stage C1 — actor2 from scratch with h2, only if C0 passes.**
   - Same cost, 16,384 steps, 3 seeds, 100ep eval seed 10000.
   - No warm start, no gate sweep, no task channel, no action filter/QP.
   - If h2 improves offline calibration but actor2 still collides after
     warning, the residual failure is actor/action-limited and the next arm is
     safety coupling / action conditioning (explicit approval required).

A single joint h2+actor2 run is also possible, but it confounds "better h"
with "different actor" and makes the result harder to attribute.  C0 is
cheaper and answers the user's question directly.

**C0a update (2026-10-26): the iteration hypothesis was tested and failed.**
The seed-0 pilot trained h2 from scratch on 100 actor1 stochastic episodes plus
the fixed random coverage.  h2 improved global actor1 MAE (`0.084` vs h1
`0.092`) but worsened the 17-32-step risk-region MAE (`0.826` vs `0.709`,
+16.5%), 9-16-step MAE (`0.567` vs `0.442`), and future-collision MAE
(`0.478` vs `0.375`).  The nominal +11-step gate lead is a threshold artifact:
h2's calibrated `gate_high` rose to `0.662`, while threshold-free last
`h > 0` stayed at 8 steps.  Per the pre-registered staging, C0b was not run
and the original C1 was not authorized.  A separately approved **C1a seed-0
actor2 pilot** was then run and **failed all five checks**: 100ep safe
`0.510` vs `0.840`, official safety `0.89375` vs `0.97375`, success
`0.33125` vs `0.42750`, reach `0.36250` vs `0.43500`, return `52.22` vs
`60.91`; collisions `0.16 -> 0.49`.  The training gate was almost always
open (`task_gate_mean≈0.99`), so this is not gate suppression.  C0a result:
`STAGE_C0_H2_TEMPORAL.md`; C1a result:
`STAGE_C1A_H2_ACTOR2_PILOT.md`.

A subsequent post-hoc seed-0 h-swap put h2 into the fixed actor1 policy
(100 deterministic episodes, eval seed 10000).  It was slightly better than
h1: safe episode `0.90` vs `0.84`, official safety `0.985` vs `0.974`,
success `0.439` vs `0.428`, return `61.40` vs `60.91`.  The differences are
one-seed and not significant, and the swap is off-policy because actor1 was
trained with h1.  The result does **not** reproduce the offline risk-region
degradation, but it does not overturn the pre-registered C0a failure or
authorize C1; it is recorded as a diagnostic only.  Record:
`../nav_c0_h2_actor1_swap/README.md`.

What would falsify the iteration hypothesis: if h2 trained on actor1 data does
not improve the 17-32-step risk-region MAE or warning lead, then the
distribution mismatch is not the main bottleneck and action conditioning is
the better next step.  If h2 improves calibration but actor2 still fails, the
impact-state actor/action limitation dominates.

C0a h2 training was run and failed.  The separately approved C1a actor2
from-scratch pilot was run and failed all five checks; the from-scratch
h2/actor2 branch is stopped.  C0b and C1b remain unauthorized.  A **C1w 4k
warm-start of actor1 with h2** (no h1 control) is separately pre-registered as
a diagnostic adaptation arm; it does not revive C1b.  The next controlled arm
(safety coupling or action conditioning) requires explicit approval.  Do not
combine it with a gate-scope, cost-schedule, or task-channel change.  Do not
warm-start actor2 from h2; C1w starts from actor1 and is diagnostic only.  Do
not sweep `task_gate_high`.

## 10. Artifacts and reproduction

Artifacts:

- `outputs/v6/independent/nav_h2_audit/seed{0,1,2}_100ep.json` — audit record,
  verification, collision windows, calibration.
- `outputs/v6/independent/nav_h2_audit/seed{0,1,2}_100ep.npz` — per-row
  arrays (`h`, `h_next`, `gate_next`, `target`, `target_raw`, `margin`,
  `current_unsafe`, `horizon_safe`, `future_collision`, `future_unsafe`,
  `episode`, `step`, `agent`).
- `outputs/v6/independent/nav_h2_audit/summary.{json,md}` — pooled summary and
  pre-registered classification.
- `outputs/v6/independent/nav_h2_audit/manifest.json` — protocol,
  checkpoint/script/output SHA-256 hashes, result.
- `outputs/v6/independent/nav_h2_audit/logs/seed{0,1,2}_100ep.log` — logs.
- `nav/scripts/diagnose_v6_nav_h2.py` — read-only audit.
- `nav/scripts/summarize_v6_nav_h2_audit.py` — aggregation/classification.

Reproduce:

```bash
python nav/scripts/diagnose_v6_nav_h2.py \
  --checkpoint outputs/v6/independent/nav_stageb_repair_16k/seed0/policy_final.pth \
  --h-checkpoint outputs/v6/independent/h_stageb_repair/v6_h_stageb_repair.pkl \
  --num-episodes 100 --seed 10000 --device cuda \
  --reference-eval outputs/v6/independent/nav_stageb_repair_16k/seed0/eval_100ep_final.json \
  --out outputs/v6/independent/nav_h2_audit/seed0_100ep.json --tag seed0_100ep

python nav/scripts/summarize_v6_nav_h2_audit.py \
  --dir outputs/v6/independent/nav_h2_audit
```

C0a h2 training was launched and failed its pre-registered pilot rule.  The
separately approved C1a actor2-from-scratch pilot was launched and failed all
five checks.  The promoted Stage-B result and the formal claim boundary are
unchanged; C1w was run as a diagnostic-only warm-start and also failed the
guardrails (task collapse, no safety gain), so the h2 warm-start branch is
stopped.
