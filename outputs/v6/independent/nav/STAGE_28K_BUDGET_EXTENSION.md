# Navigation 28k budget extension: A cell only

Status: **pre-registered and launched 2026-10-27 19:29 CST**.  This is a
budget-sensitivity experiment, not a replacement for the promoted 16k row.

Canonical status: [`../../STATUS.md`](../../STATUS.md).
Planning matrix: [`PENDING_28K_EXPERIMENTS.md`](PENDING_28K_EXPERIMENTS.md).

## Scope (frozen)

- Cell **A only**: repaired `h1` + progress-only gate, actor h input on.
- Seeds `0/1/2`, trained **from scratch** for **28,672 steps** each.
- No H0@28k, no B@28k, no C/D@28k, no external 28k rows, no H0G, no P0.
  H0/B external rows are not approved; without H0@28k there is **no
  budget-matched A-vs-h-only claim**.  The only comparison is against the
  promoted 16k A row and the same-run 16k checkpoint.
- No gate-threshold sweep, no `policy_best` selection, no per-seed extension,
  no resume from a 16k checkpoint.
- V6 independence unchanged: no official h, `cbf.pkl`, QP, `u_ref`, or
  V5/V4 artifact.

## Frozen artifacts

| item | value |
|---|---|
| config | `nav/configs/v6/v6_nav_stageb_repair_28k.yaml` |
| config sha256 | `153e705227a70c7a386bb308bb4d5ac0c01ad8dac163dd977490f92972d8e116` |
| runner | `nav/scripts/run_v6_nav_stageb_repair_28k.sh` |
| runner sha256 | `7189190a8063c1a3048527c2fb51b68edd87394a38a9ad0f8fed4ec12d618697` |
| h checkpoint | `outputs/v6/independent/h_stageb_repair/v6_h_stageb_repair.pkl` |
| h sha256 | `0f67eaeec99912881017231234aa10846c5edfda767118da2157b2da7df7fb13` |
| output root | `outputs/v6/independent/nav_stageb_repair_28k/` |
| training seed | `0/1/2` |
| env | official GCBF+ DoubleIntegrator, 8 agents, area 4.0, 8 obstacles, 256 steps |

The 28k config differs from the promoted 16k config **only** in
`training.total_steps` (`16384 -> 28672`) and comments.  There is no LR
schedule or other step-count-dependent term, so the first 16,384 steps of a
28k run follow the same trajectory as the corresponding 16k run for that
seed.

## Endpoints (frozen before launch)

- **Primary:** `policy_final.pth` at 28,672 steps, 100 deterministic episodes,
  eval seeds `10000..10099`, fixed final checkpoint.
- **Paired budget control:** `policy_step_16384.pth` from the **same** 28k
  run, same 100 scenes, evaluated once.
- **Curve:** the training-time monitor eval (`eval_episodes=20`,
  deterministic) at 8,192 / 16,384 / 24,576 steps, plus 8k/12k/20k/24k
  checkpoints retained.  Do not switch the primary endpoint if the monitor
  curve peaks earlier.
- Saved checkpoints: `policy_step_{4096,...,28672}.pth` and
  `policy_final.pth`; `policy_best.pth` exists but is **not** used.
- No evaluation of the old 16k promoted run is repeated; the same-run 16k
  checkpoint is the paired control.  The promoted 16k row remains the
  external reference.

## Variance and promotion rule (copied from section 7.1)

Baseline 16k A: `safe_episode 0.6933 ± 0.1680`, per-seed `0.84/0.51/0.73`,
range `0.33`, worst seed `0.51`; task `success 0.4100`, `return 66.85`.

Apply in order to the 28k final:

1. **Mean not degraded:** mean `safe_episode >= 0.6933` and mean
   `official_safety >= 0.9417`.
2. **Task guardrails:** mean `official_success >= 0.36`, mean
   `episode_return >= 55`, worst seed `official_success >= 0.20`,
   worst seed `safe_episode >= 0.30`.
3. **Observed spread narrowed:** between-seed range `max - min <= 0.20`
   and worst seed `safe_episode >= 0.55`.
4. **Paired direction:** report 28k vs same-run 16k per seed (all three
   deltas), not only the mean.

Outcomes:

- **1-3 pass:** report "28k maintained/improved mean safety and narrowed the
  observed between-seed spread."  Descriptive only; `n=3` is not proof of
  variance reduction.
- **1-2 pass, 3 fails:** report "higher/equal mean, similar or high
  variance."  Only then consider adding seeds to 5 for **both** A and H0 if
  H0@28k is approved; do not add seeds to A alone.  Use the same 28k runs'
  16k checkpoints as the 5-seed 16k control.
- **1 or 2 fails:** report a negative or task-costly budget result; do not
  add seeds to rescue it.  Keep 16k as the promoted row.

**Interpretation before the 3-seed table is complete (2026-10-27):** the 28k
endpoint is a budget-sensitivity test, not a commitment to promote 28k.  If
the final checkpoint trades task for safety (lower `official_success`/return
with equal or slightly higher safety), report that as "more budget trades
task for safety" and keep 16k as the promoted row.  The 24k 100ep point is a
secondary curve location only; promoting 24k, or a 24k+task warm-start arm,
requires a new pre-registration and 3-seed confirmation.  Do not switch the
primary endpoint post hoc on the test scenes.

Statistical caveat: `n=3` std/range is unstable.  A small observed range is
consistent with, not proof of, reduced variance; a large range is stronger
evidence of heterogeneity.  A formal variance claim needs at least 5 seeds.

## Launch and verification

- 2026-10-27 19:29:16: first invocation failed **before any training step**
  because the runner passed `nav/configs/...` while `train_v6.py` resolves
  config paths under `nav/` (the resolved path was `nav/nav/configs/...`).
  No checkpoints were written and no protocol setting differed; this is not a
  partial run.
- 2026-10-27 19:29:34: relaunched with the corrected
  `configs/v6/v6_nav_stageb_repair_28k.yaml` path, three training processes in
  parallel, GPU free after the Ant 10M screen.
- Training logs: `outputs/v6/independent/nav_stageb_repair_28k/logs/`.
- **OOM incident 2026-10-27 21:42 CST.**  The kernel OOM killer terminated
  seed 2 at 27,776/28,672 steps while seeds 0/1 were running their final
  20-episode evals; three trainers at ~4.7 GB each exceed the 15 GB machine.
  Seed 0 and seed 1 reached 28,672 and are valid.  The partial seed-2
  directory is archived at `seed2_oom_partial_27776/` for provenance and was
  **not** used for the formal endpoint.  Seed 2 was restarted from scratch at
  21:54 with the identical config and seed, in a single-process run (no
  memory contention).  The queued MAPPO watcher was stopped to prevent it
  from starting during recovery; it is relaunched only after the 28k formal
  evaluations finish.
- **Training-log check (seed 0/1, last 4k, 2026-10-27):** no late-training
  collapse.  Reward mean `0.21`, cost `0.004-0.006`, task-gate mean
  `0.965-0.978`, `h<0` rate `0.008-0.015`, entropy `~0.82`; the value-loss
  spikes are ordinary PPO noise, not divergence.  The 24k/28k task dip is a
  policy trade-off, not a training failure; the 100ep formal is the decider.
- Runtime estimate: training is mixed with 20-episode monitor evaluations at
  8k/16k/24k; the monitor evals dominate wall time.  Expect ~2-3 h for
  training plus the six 100-episode formal evaluations.  The seed-2 rerun
  adds roughly its own training time; the revised ETA is reported in
  `STATUS.md` once the first checkpoint lands.
- Post-run checks: all three `policy_final.pth` and
  `policy_step_16384.pth` exist; `train.jsonl` ends at 28,672; final and
  paired-16k eval JSONs exist; config/runner/h hashes above match.

## Result (2026-10-28)

Formal 100ep means:

| budget | safe | official | success | return |
|---|---:|---:|---:|---:|
| 16k | 0.6933 | 0.9417 | 0.4100 | 66.85 |
| 24k | 0.7000 | 0.9479 | 0.2704 | 59.97 |
| 28k | 0.7800 | 0.9604 | 0.2767 | 63.45 |

28k per seed: safe `0.86/0.70/0.78`, official `0.9750/0.9450/0.9613`,
success `0.19875/0.40375/0.22750`, return `55.52/84.53/50.29`.

Rule 7.1 fails on **mean success** (`0.2767 < 0.36`) and **worst-seed
success** (`0.19875 < 0.20`).  Safety, return, safe range (`0.16`) and
worst-seed safe (`0.70`) pass.  The task loss is already present at 24k
(`0.2704`), while 28k recovers the 24k safety dip (`0.7000 -> 0.7800`).
This is a **negative task result / task-costly budget extension**, not a
promotion.  Keep 16k as the promoted row; do not switch the primary endpoint
post hoc, do not add 28k seeds, and do not promote 24k.

Full table: `independent/nav/nav_3seed_16k_24k_28k_table.json`.

The separate W16 task warm-start recovers task strongly at 28k
(`success 0.6175` vs `0.2767`) but fails the 7.1 mean-safety and spread
gates (`safe 0.6633`, range `0.27`); it is a task-first trade-off, not a
replacement for the 16k row.  Record:
`independent/nav/STAGE_W16_TASK_WARMSTART.md`.

