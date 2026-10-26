# Repaired Stage-B multi-seed run

Status: **complete, 3-seed pass** (2026-10-25).  Pre-registered before seeds
1/2; all four aggregate guardrails and the no-catastrophic-seed rule pass.

## Scope

Complete the repaired Stage-B arm to three training seeds (0/1/2) after the
seed-0 evaluation.  Seeds 1/2 are the only new runs.  Do not change the h
checkpoint, gate calibration, reward, cost, task channel, or evaluation
protocol.

- config: `nav/configs/v6/v6_nav_stageb_repair_16k.yaml`
- config sha256: `4ee69c9e99a427708d17f1157991786d41a60e2624721e7e478b90a0aabebd16`
- h checkpoint: `outputs/v6/independent/h_stageb_repair/v6_h_stageb_repair.pkl`
- h sha256: `0f67eaeec99912881017231234aa10846c5edfda767118da2157b2da7df7fb13`
- training: 16,384 steps, rollout 128, from scratch, `h.co_train: false`
- evaluation: 100 deterministic episodes, eval seed 10000, fixed final
  checkpoint, same 100 scenes for every seed
- no QP, no action filter, no official h, no V5 artifact, no threshold sweep

## Existing seed 0

`safe_episode=0.840`, `collision_episode=0.160`,
`official_safety=0.97375`, `official_success=0.4275`,
`official_reach=0.4350`, `episode_return=60.9085`.

## Decision rules (fixed before seeds 1/2)

Report every seed, not only the best.  Aggregate mean ± std across seeds 0/1/2.
The h-only formal 3-seed reference is `safe_episode=0.450`,
`official_safety=0.8783`, `official_success=0.4067`,
`official_reach=0.4450`, `episode_return=61.73`.

Primary safety pass:

- 3-seed mean `safe_episode >= 0.55`; and
- 3-seed mean `official_safety >= 0.93`.

Task guardrail:

- 3-seed mean `official_success >= 0.36` (no worse than 5 pp below the
  h-only 3-seed mean); and
- 3-seed mean `episode_return >= 55` (at least 89% of the h-only 3-seed mean).

No catastrophic seed:

- no seed with `safe_episode < 0.30` or `official_success < 0.20`.

If safety passes but task fails, do not sweep `task_gate_high` and do not add
action conditioning without explicit approval.  If the aggregate passes,
record it as a 3-seed navigation gate result and only then consider the task
channel.

## Outputs

- `outputs/v6/independent/nav_stageb_repair_16k/seed1/`
- `outputs/v6/independent/nav_stageb_repair_16k/seed2/`
- per-seed `eval_100ep_final.json`
- aggregate written after all three evaluations

## Launch

```bash
SEED=1 nav/scripts/run_v6_nav_stageb_repair_actor.sh
SEED=2 nav/scripts/run_v6_nav_stageb_repair_actor.sh
```

Seeds 1/2 may train concurrently; each evaluation uses the fixed protocol above.

## Results (100ep deterministic, eval seed 10000)

| seed | reach | success | safe_ep | collision_ep | official_safety | return |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.4350 | 0.4275 | 0.840 | 0.160 | 0.9738 | 60.91 |
| 1 | 0.4963 | 0.4575 | 0.510 | 0.490 | 0.9000 | 70.92 |
| 2 | 0.3575 | 0.3450 | 0.730 | 0.270 | 0.9513 | 68.74 |
| **mean** | **0.4296** | **0.4100** | **0.6933** | **0.3067** | **0.9417** | **66.85** |
| **std** | 0.0695 | 0.0583 | 0.1680 | 0.1680 | 0.0378 | 5.26 |

Matched h-only reference (same training seeds, same eval scenes):
`safe_episode=0.450`, `official_safety=0.8783`, `official_success=0.4067`,
`official_reach=0.4450`, `episode_return=61.73`.

| metric | h-only 3-seed | repaired 3-seed | delta |
|---|---:|---:|---:|
| safe_episode | 0.450 | **0.693** | **+24.3 pp** |
| collision_episode | 0.550 | **0.307** | **-24.3 pp** |
| official_safety | 0.8783 | **0.9417** | **+6.3 pp** |
| official_success | 0.4067 | 0.4100 | +0.3 pp |
| official_reach | 0.4450 | 0.4296 | -1.5 pp |
| episode_return | 61.73 | **66.85** | **+8.3%** |

Matched per-seed safety deltas are `+0.22`, `+0.02`, `+0.49`; success deltas
are `-0.105`, `+0.040`, `+0.075`.  The safety direction is consistent (no seed
worse), but seed 1 is nearly neutral and seed 2 carries most of the gain.
Task is neutral on average, not uniformly retained: seed 0 loses 10.5 pp
success while seeds 1/2 gain 4.0/7.5 pp.

## Guardrail check

| rule | required | observed | pass |
|---|---:|---:|---|
| mean safe_episode | >= 0.55 | 0.6933 | yes |
| mean official_safety | >= 0.93 | 0.9417 | yes |
| mean official_success | >= 0.36 | 0.4100 | yes |
| mean episode_return | >= 55 | 66.85 | yes |
| worst seed safe_episode | >= 0.30 | 0.510 | yes |
| worst seed official_success | >= 0.20 | 0.345 | yes |

**Verdict: pass.**  The repaired Stage-B predictive-h gate is promoted as the
first V6 navigation 3-seed gate result.  It keeps task success at the h-only
level on average while raising episode-level safety by 24.3 pp.

## Caveats

- `n=3` seeds.  A paired seed-level t-test on the deltas gives `p=0.216` for
  safe_episode and `p=0.957` for success, so the 3 seeds alone cannot establish
  a policy-level significant difference.  The pre-registered aggregate rule
  passes, but the claim is "passes the fixed guardrails", not "statistically
  proven superiority".
- Pooling the 300 evaluation episodes gives `208/300` safe (Wilson 95%
  `0.639-0.743`) versus h-only `135/300` (`0.395-0.507`).  Episode-level
  binomial intervals ignore policy clustering and are optimistic.
- The 20ep monitor was optimistic: 3-seed mean `safe_episode=0.767`,
  `success=0.454`, `return=70.29` versus final `0.693/0.410/66.85`.
- Eval scenes are the same 100 scenes for every seed; this controls scene
  difficulty but not training-seed variance.
- Formal 16k h-off 3-seed baseline is still absent, so h-off is not a matched
  comparison.
- No `task_gate_high` sweep, no cost-schedule change, no action conditioning.
  Any next arm requires new approval; do not silently combine changes.

## Artifacts

- `outputs/v6/independent/nav_stageb_repair_16k/seed{0,1,2}/eval_100ep_final.json`
- `outputs/v6/independent/nav_stageb_repair_16k/aggregate_100ep.json`
- `outputs/v6/independent/nav_stageb_repair_16k/aggregate_100ep.md`
- `outputs/v6/independent/nav_stageb_repair_16k/multiseed_status.txt`

## Follow-up

A read-only h2/actor2 diagnostic on these fixed final checkpoints is complete;
it found an actor/action-limited failure mode and did not launch training.
See [`H2_ACTOR2_DIAGNOSTIC.md`](H2_ACTOR2_DIAGNOSTIC.md).
