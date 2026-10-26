# V6 navigation baseline comparison

Canonical project status: [`../../STATUS.md`](../../STATUS.md).

Status: 3-seed comparison, 2026-10-25.  The main V6 row is the repaired
Stage-B actor (seeds 0/1/2).  Seed-0 matched rows are retained for ablation
comparison.  External V5/CBF rows are archived references, not reruns.

## Protocol

Matched V6 rows use:

- 8 agents, 4x4 area, 8 obstacles, 256 steps;
- 16,384 training steps, seeds 0/1/2;
- separate 100-episode deterministic evaluation, eval seed 10000;
- fixed final checkpoint (not the monitor-best checkpoint);
- frozen V6-owned repaired h; no official h, QP, expert action, or V5 artifact.

`official_success` is the per-agent any-time success rate (4 agents x 100
episodes = 400 trajectories per seed); `safe_episode` is the fraction of the
100 episodes with no collision.  `official_safety` is the environment safety
metric, higher is better.

## V6 repaired Stage B: 3 seeds x 100 episodes

| seed | safe_ep | collision_ep | official_safety | success | reach | return |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.840 | 0.160 | 0.9738 | 0.4275 | 0.4350 | 60.91 |
| 1 | 0.510 | 0.490 | 0.9000 | 0.4575 | 0.4963 | 70.92 |
| 2 | 0.730 | 0.270 | 0.9513 | 0.3450 | 0.3575 | 68.74 |
| **mean** | **0.693** | **0.307** | **0.9417** | **0.4100** | **0.4296** | **66.85** |
| **std** | 0.168 | 0.168 | 0.0378 | 0.0583 | 0.0695 | 5.26 |

Source: `outputs/v6/independent/nav_stageb_repair_16k/aggregate_100ep.json`.

Matched h-only reference (same training seeds and eval scenes):

| h-only 3-seed | safe_ep | collision_ep | official_safety | success | reach | return |
|---|---:|---:|---:|---:|---:|---:|
| mean | 0.450 | 0.550 | 0.8783 | 0.4067 | 0.4450 | 61.73 |
| std | 0.1931 | 0.1931 | 0.0659 | 0.1316 | 0.1332 | 10.64 |

Delta (repaired - h-only): safe `+24.3 pp`, official safety `+6.3 pp`,
success `+0.3 pp`, reach `-1.5 pp`, return `+8.3%`.

## Matched 16k / 100ep / seed-0 arms

| arm | safe_ep | collision_ep | official_safety | success | reach | return | success retention |
|---|---:|---:|---:|---:|---:|---:|---:|
| h-only (matched seed 0) | 0.620 | 0.380 | 0.9287 | **0.5325** | **0.5763** | **69.78** | 100.0% |
| broad positive-reward gate | 0.700 | 0.300 | 0.9400 | 0.3725 | 0.3913 | 64.96 | 69.9% |
| Stage A progress-only gate | 0.530 | 0.470 | 0.9113 | 0.5075 | 0.5475 | 76.85 | 95.3% |
| old Stage B (bad future-min h) | 0.000 | 1.000 | 0.4400 | 0.0375 | 0.0700 | -23.60 | 7.0% |
| old Stage B, gate off (diagnostic) | 0.030 | 0.970 | 0.4850 | 0.0763 | 0.1350 | 7.34 | 14.3% |
| **repaired Stage B seed 0** | **0.840** | **0.160** | **0.9738** | 0.4275 | 0.4350 | 60.91 | **80.3%** |

These are seed-0 ablations; only the repaired Stage-B arm has a formal
3-seed result.  Do not compare their single-seed retention percentages to the
3-seed mean as if they were equally sampled.

## Other references (not matched)

| arm | budget / episodes | safe_ep | official_safety | success | reach | return |
|---|---|---:|---:|---:|---:|---:|
| h-off pilot | 8k / 20ep | 0.050 | 0.5188 | 0.0188 | 0.0500 | -31.73 |
| barrier fine-tune | 24.5k / 100ep | 0.730 | 0.9500 | 0.31875 | 0.32625 | 65.77 |
| barrier screen | 16k / 20ep | 1.000 | 1.0000 | 0.3375 | 0.3375 | 68.70 |
| CBF-loss screen | 16k / 20ep | 0.650 | 0.9438 | 0.4688 | 0.4813 | 63.75 |
| barrier + both screen | 24.5k / 20ep | 0.850 | 0.9688 | 0.1063 | 0.1063 | 23.12 |

The h-off pilot is not a 16k baseline: the formal h-off 16k 3-seed run is
still absent.  The barrier fine-tune uses a 24,576-step budget, so it is a
reference, not a matched baseline.

## External CBF-family and Safe-MARL reference

These are archived V5 results, read-only.  They use the same official
DoubleIntegrator, the same 100 fixed scenes (`10000..10099`), deterministic
actors, and the same metric definitions, but they are **not rerun** and are
**not V6 main-table baselines**: V5 uses the frozen official h and a safety
fine-tune from 8k; GCBF+ is a pretrained actor; CBF-QP is a model-based
controller with an online QP.  V6 is from scratch with V6-owned h and no QP.
Compare as a reference, not as a controlled experiment.

| method | family | budget / seeds | QP? | official safety | safe_ep | success | reach | return |
|---|---|---|---:|---:|---:|---:|---:|---:|
| GCBF+ official | CBF / learned | pretrained step 1000 | no (actor-only) | 0.9988 | 0.990 | 0.8613 | 0.8625 | 480.05 |
| centralized CBF-QP | CBF-QP | model-based | yes (OSQP) | 0.9788 | 0.900 | 0.7813 | 0.7888 | 406.82 |
| V5 full | CBF-informed Safe MARL | 16k, 3 seeds, from 8k | no | 0.9500 ± 0.0308 | 0.7467 ± 0.1159 | 0.5121 ± 0.1352 | 0.5363 ± 0.1490 | 81.13 ± 6.00 |
| V5 h-input | ablation | 16k, 3 seeds | no | 0.9533 ± 0.0402 | 0.7633 ± 0.1872 | 0.2779 ± 0.0567 | 0.2875 ± 0.0579 | 47.35 ± 9.61 |
| V5 no-h | ablation | 16k, 3 seeds | no | 0.4179 ± 0.0657 | 0.0133 ± 0.0058 | 0.0475 ± 0.0142 | 0.1125 ± 0.0344 | 9.98 ± 14.09 |
| pure MAPPO 16k | task-only MARL | 16k, 3 seeds | no | 0.3079 ± 0.0995 | 0.0033 ± 0.0058 | 0.0571 ± 0.0281 | 0.1713 ± 0.0435 | 32.17 ± 4.80 |
| MAPPO-Lag 100k | constrained MARL | 100k, 3 seeds | no | 0.4325 ± 0.0379 | 0.0033 ± 0.0058 | 0.1363 ± 0.0508 | 0.3088 ± 0.0928 | 50.36 ± 17.85 |
| **V6 repaired Stage B** | predictive h + gate | 16k, 3 seeds | no | 0.9417 ± 0.0378 | 0.6933 ± 0.1680 | 0.4100 ± 0.0583 | 0.4296 ± 0.0695 | 66.85 ± 5.26 |

`return` is the task return `20 * progress + 5 * reached`, not the native
GCBF+ control reward.

Coverage note: GCBF v0, decentralized CBF-QP, a distinct HandCBF
implementation, and MACPO are not available in this workspace.  Do not invent
rows for them.  The centralized CBF-QP row is the hand-crafted CBF-QP
reference; it uses the official formulation with OSQP instead of the
unavailable `jaxproxqp`.

Matched seed-0 reference for the two V5 learned-policy arms:

| method | official safety | safe_ep | success | reach | return |
|---|---:|---:|---:|---:|---:|
| V5 full seed 0 | 0.9788 | 0.870 | 0.5525 | 0.5650 | 85.84 |
| V5 h-input seed 0 | 0.9825 | 0.900 | 0.3113 | 0.3125 | 55.44 |
| V6 repaired seed 0 | 0.9738 | 0.840 | 0.4275 | 0.4350 | 60.91 |

Positioning:

- GCBF+ is the strongest reference on both axes, but it is an official
  pretrained teacher with actor-only inference; V6 cannot copy its weights or
  h.  CBF-QP is also far ahead on task, with an online QP and full-state
  information.
- V6 3-seed safety `0.942` is close to CBF-QP `0.979` and V5 full 3-seed
  `0.950`, but V6 success `0.410` is below V5 full `0.512` and far below
  CBF-QP `0.781` / GCBF+ `0.861`.  V6 is not a replacement for the full
  Safe-MARL or CBF-QP baselines on task.
- V6 beats V5 h-input on task and return with slightly lower safety: success
  `0.410` vs `0.278` (`+13.2 pp`), reach `0.430` vs `0.288`, return `66.85`
  vs `47.35` (`+41%`), while safe episodes are `0.693` vs `0.763` and
  official safety `0.942` vs `0.953`.
- V6 dominates the unsafe MARL baselines: vs V5 no-h, pure MAPPO, and
  MAPPO-Lag it is higher on safety, safe episodes, success, reach, and return.
- V5 full is the strongest learned Safe-MARL reference: `0.950/0.747/0.512/
  81.13` over 3 seeds.  V6 loses `5.3 pp` safe episodes, `10.2 pp` success,
  and `17.6%` return while being within `0.8 pp` official safety.  V5 full
  uses official h and a warm start from 8k, so this is a reference gap, not an
  equal-information comparison.

## Reading

1. **Safety is the clearest 3-seed result.**  Mean `safe_episode` rises from
   `0.450` to `0.693` (`+24.3 pp`) and official safety from `0.878` to `0.942`
   (`+6.3 pp`).  All three seed-level safety deltas are non-negative
   (`+0.22/+0.02/+0.49`), but seed 1 is nearly neutral and the gain is carried
   by seeds 0/2.  With `n=3`, the paired seed-level difference is not
   statistically separated from zero (`p=0.216`); the pre-registered
   aggregate rule passes, but the claim is not "policy-level proven".
2. **Task does not collapse on average.**  Mean success is `0.410` vs h-only
   `0.407` (`+0.3 pp`) and return is `+8.3%`; per-seed success shifts are
   `-10.5/+4.0/+7.5 pp`, so retention is heterogeneous rather than uniform.
   The 3-seed success standard deviation also falls from `0.132` to `0.058`.
3. **It is stronger than the earlier gate designs at seed 0.**  Repaired
   seed 0 has `safe_episode=0.840` vs broad gate `0.700` and Stage A `0.530`,
   with success `0.4275` vs broad `0.3725`; Stage A retains more task at seed 0
   (`0.5075`) but loses safety.  Old Stage B and its gate-off control remain
   catastrophic and confirm that the old h features were harmful.
4. **It compares well with the barrier-loss alternative.**  At 16k seed 0 the
   repaired arm has higher safety and success than the 24.5k barrier
   fine-tune, with lower return.  The 20ep barrier screen reaches
   `safe_episode=1.0` but at `success=0.3375`; the repaired 16k 20ep monitor
   was `safe_episode=0.90`, `success=0.4875`, return `64.74` (optimistic vs
   the final 100ep).
5. **Recorded V5 safety target.**  Seed 0 meets the archived target of
   environment safety `>=0.95` and safe episode `>=0.70` (`0.9738/0.840`).
   The 3-seed mean falls just short (`0.9417/0.693`).  The target does not
   specify a numeric success floor, so this is a safety-target check, not an
   official-h reproduction.

## Claim boundary

- The repaired Stage-B arm passes the pre-registered 3-seed guardrails; this
  is a 3-seed navigation gate result.
- `n=3` only.  Seed-level paired differences are not statistically separated
  from zero; do not claim policy-level superiority over h-only.
- Do not claim superiority over V5 full, CBF-QP, or GCBF+ on task; V6 is
  still `10.2 pp` below V5 full and `37.1 pp` below CBF-QP success.
- Do not sweep `task_gate_high` to close the remaining gap.
- The formal 16k h-off 3-seed baseline is still absent; h-off is not a matched
  comparison.
- No action-conditioned filter/QP was used or evaluated here.
