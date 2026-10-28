# P1: repaired-h 2x2 factorial (pre-registration + results)

Status: **completed 2026-10-27**.  The protocol below was frozen before any P1
(B/C/D) training started; the results section was appended after all nine runs
and all nine 100-episode evaluations finished.  P0 evaluation was already
running when the protocol was frozen (it is a separate h-quality row).

Canonical status: [`../../STATUS.md`](../../STATUS.md).

## Motivation

The promoted repaired Stage-B row changes two things simultaneously relative
to the earlier "matched h-only" reference:

1. the h checkpoint is retrained (`v6_h_stageb_repair.pkl` instead of the
   original `v6_h.pkl`);
2. the V6 progress gate is active at training time.

The previously reported `+24.3 pp` safe-episode delta therefore conflates
h-checkpoint quality with gate activation.  P0 and P1 separate these factors
with a minimal clean attribution experiment.  No new model code is required:
all cells are the existing `train_v6.py` with CLI overrides.

## Design

### P0: original-h + gate row (h-quality control)

Complete the existing original-h broad-gate seeds 1/2 and evaluate all three
seeds with the formal 100-episode protocol.  This is the row that the repaired
Stage-B result was previously compared against without a full 3-seed eval.

- h: `outputs/v6/independent/h/v6_h.pkl` (original)
- gate: on (`low=0.0`, `high=0.13`)
- actor h input: on
- steps/seeds/eval: 16,384; 0/1/2; 100 deterministic episodes, eval seed 10000

### P1: repaired-h 2x2

All cells use the same frozen repaired h
(`outputs/v6/independent/h_stageb_repair/v6_h_stageb_repair.pkl`), the same
16,384-step budget, seeds 0/1/2, fixed unsafe penalty 10, actor trained from
scratch, final checkpoint, and 100 deterministic episodes on scenes
`10000..10099`.

| cell | actor h/grad input | training-time gate | source |
|---|---|:---:|---|
| **A** | on | on | existing `nav_stageb_repair_16k`, not retrained |
| **B** | on | off | new: `--task-gate-high 0.0` |
| **C** | off (zeroed) | on | new: `--no-use-h-features` |
| **D** | off (zeroed) | off | new: `--no-use-h-features --task-gate-high 0.0` |

The gate, when on, uses the frozen repaired h's `h_next`.  Thus C is
"gate-only, no h actor input", not "no h network at all".  `--no-use-h-features`
zeros the actor's h/grad_h channels; it does not change the h checkpoint or the
gate computation.

The gate is disabled at evaluation in every cell; gate on/off is a
training-time reward-shaping difference.  No threshold sweep is performed:
A/C use the pre-calibrated repaired `high=0.26726993918418884`, and B/D set
`high=0.0`, which disables the gate because `high <= low`.

## Estimands

Descriptive 3-seed means (no significance claim with `n=3`):

- **h-input effect, gate off:** B − D.
- **h-input effect, gate on:** A − C.
- **gate effect, h input on:** A − B.
- **gate effect, h input off:** C − D.
- **interaction:** (A − B) − (C − D), equivalently (A − C) − (B − D).
- **h-checkpoint effect (separate, not a 2x2 cell):** A − P0.  P0 differs in
  both h weights and gate threshold (`0.13` vs `0.26727`), so it is an
  h-quality/row comparison, not a single-factor estimate.

## Protocol guards

- No official h, `cbf.pkl`, QP, `u_ref`, expert labels/actions, or V5/V4
  weights/data.
- No checkpoint selection on evaluation episodes; the final checkpoint is
  always used.
- No seed shopping and no metric-dependent reruns.  A crashed or interrupted
  run may be rerun with the identical config and seed; this must be recorded.
- No changes to `task_gate_high` beyond on/off, no PPO hyperparameter tuning,
  no extra training steps.
- All seeds are reported, including failures.

## Results (appended 2026-10-27 after all runs finished)

All nine B/C/D runs completed training at 14:47 CST; all nine formal 100-episode
evaluations finished at 17:08 CST.  Aggregation:
`outputs/v6/independent/nav_repair_factorial/aggregate_100ep.{json,md}`.

| cell | actor h/grad input | training gate | safe_ep | official_safety | success | reach | return |
|---|---|:---:|---:|---:|---:|---:|---:|
| A repaired h + gate | on | on | 0.6933 | 0.9417 | 0.4100 | 0.4296 | 66.85 |
| B repaired h, gate off | on | off | 0.6333 | 0.9267 | 0.3229 | 0.3517 | 55.67 |
| C no actor h input, gate on | off | on | 0.0200 | 0.4987 | 0.0367 | 0.1008 | -6.63 |
| D no actor h input, gate off | off | off | 0.0333 | 0.5479 | 0.0367 | 0.0708 | 7.21 |

Pre-registered estimands (descriptive 3-seed means):

| contrast | safe_ep | official_safety | success | reach | return |
|---|---:|---:|---:|---:|---:|
| B - D (h input, gate off) | +60.0 pp | +37.9 pp | +28.6 pp | +28.1 pp | +48.5 |
| A - C (h input, gate on) | +67.3 pp | +44.3 pp | +37.3 pp | +32.9 pp | +73.5 |
| A - B (gate, h input on) | +6.0 pp | +1.5 pp | +8.7 pp | +7.8 pp | +11.2 |
| C - D (gate, h input off) | -1.3 pp | -4.9 pp | 0.0 pp | +3.0 pp | -13.8 |
| (A-B)-(C-D) interaction | +7.3 pp | +6.4 pp | +8.7 pp | +4.8 pp | +25.0 |

Reading:

- **h input is load-bearing at 16k from scratch.**  C and D collapse to
  `safe 2.0%/3.3%` and `success 3.7%` across all six seeds; the collapse is
  visible in training reward and unsafe rate, not only at evaluation.  D is
  consistent with the archived no-h fixed-penalty baseline (safe `0.033`,
  official safety `0.548` vs `0.519`), which corroborates the result.
- **The gate cannot substitute for h input.**  C - D is flat on safety and
  success; gate-only training does not recover task performance.
- **With h input, the gate improves every endpoint directionally** (A - B:
  `+6.0 pp` safe, `+8.7 pp` success, `+11.2` return), but per-seed safe deltas
  are `+20/+12/-14 pp` and paired `p=0.618`; this is a direction, not a
  demonstrated effect.
- **Repaired h alone is a safety/task trade.**  B - H0 is `+18.3 pp` safe but
  `-8.4 pp` success and `-6.1` return, concentrated in B seed 2.  A's combined
  safety + task profile is not reproduced by either factor alone.
- **No policy-level superiority claim.**  `n=3`, no multiplicity correction.
  The only `p<0.05` single-factor contrast is the gate-on h-input effect
  (A - C safe `p=0.015`), driven by the collapsed C cell; the gate-off
  counterpart (B - D) is `p=0.064`.  The interaction return difference
  (`+25.0`) has paired `p=0.034`, but it is not interpreted because both C
  and D are collapsed and return is heavy-tailed.  The promoted claim remains
  the pre-registered guardrail pass.

`--no-use-h-features` zeros the policy h/grad channels while the C gate still
reads repaired `h_next`; `h_barrier_coef=0`, so no dense barrier cost is
removed.  The V6 policy shares a trunk between actor and value heads, so this
ablation affects both; "actor h input" in the tables is shorthand for the
policy-level h/grad input.  `evaluate_v6.py` reads the config embedded in each
checkpoint, so C/D keep `use_features=false` at evaluation; there is no
train/eval h-input mismatch.  The gate is off at evaluation everywhere.

## Artifacts

```text
outputs/v6/independent/nav_h_gate_16k/seed{0,1,2}/eval_100ep_final.json   # P0
outputs/v6/independent/nav_stageb_repair_16k/seed{0,1,2}/                # A
outputs/v6/independent/nav_repair_factorial/
  B_h_on_gate_off/seed{0,1,2}/
  C_h_off_gate_on/seed{0,1,2}/
  D_h_off_gate_off/seed{0,1,2}/
  aggregate_100ep.{json,md}
  manifest.json
```

Runners:

- `nav/scripts/run_v6_nav_repair_factorial_arm.sh`
- `nav/scripts/run_v6_nav_repair_factorial.sh`
- `nav/scripts/eval_v6_nav_100ep_chunked.sh`
- `nav/scripts/aggregate_v6_nav_factorial.py`

## Status log

- 2026-10-27 11:07 CST: protocol frozen; P0 seed 1/2 evaluation launched
  first.  No P1 cell was trained before this file was written.
- 2026-10-27 11:26 CST: P0 finished (`0.6433/0.9325/0.4263/70.33`).  P1
  training (B/C/D) started at 11:26:33, i.e. after the P0 row was unblinded.
  The P1 cells, endpoints, seeds, and analysis were **not changed** in
  response; P0 is a separate h-quality row and does not enter the 2x2.
- 2026-10-27 14:47 CST: all nine B/C/D training runs finished; all nine
  `policy_final.pth` present.
- 2026-10-27 ~15:08 CST: an orchestrator process was killed by a server
  restart during B seed 2 evaluation.  Completed training and B seed 0/1
  evaluations were unaffected.  B seed 2 had chunk logs but no chunk JSON, so
  it was re-evaluated from scratch with the identical checkpoint/config/seed.
- 2026-10-27 17:08 CST: all nine formal evaluations finished (100 episodes
  each, eval seed 10000).  Aggregation produced
  `nav_repair_factorial/aggregate_100ep.{json,md}`.  Results are reported
  above; no run was selected, rerun, or excluded on the basis of its result.
