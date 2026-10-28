# outputs/v6 index

Canonical status and promoted results: [`STATUS.md`](STATUS.md).

## Start here

| need | file |
|---|---|
| current status, claims, pending items | `STATUS.md` |
| consolidated experiment record | `V6_COMPLETE_RECORD.md` |
| pending TODO | `TODO_V6.md` |
| navigation design/result index | `independent/nav/README.md` |
| navigation baseline comparison | `independent/nav/BASELINE_COMPARISON.md` |
| navigation P0/P1 attribution (completed) | `independent/nav/P1_REPAIRED_H_FACTORIAL.md` |
| h2/actor2 diagnostic (read-only) | `independent/nav/H2_ACTOR2_DIAGNOSTIC.md` |
| Stage C0 h2-only screen (C0a failed) | `independent/nav/STAGE_C0_H2_TEMPORAL.md` |
| Post-hoc seed-0 h2+actor1 h-swap (diagnostic) | `independent/nav_c0_h2_actor1_swap/README.md` |
| C1a actor2 seed-0 pilot with h2 (failed all five checks) | `independent/nav/STAGE_C1A_H2_ACTOR2_PILOT.md` |
| C1w actor1+h2 4k warm-start (diagnostic, failed) | `independent/nav/STAGE_C1W_H2_ACTOR1_4K.md` |
| C1s h2 seed-0 + actor1 seed replication (evaluation only, `consistent_positive`) | `independent/nav/STAGE_C1S_H2_ACTOR_SEED_SWAP.md` |
| T0w actor1 + frozen finite-horizon V_T 4k warm-start (diagnostic; strict safety-first rule missed by 1 episode, task-heavy trade-off) | `independent/nav/STAGE_T0W_VT_WARMSTART.md` |
| Ant records index | `independent/ant/README.md` |
| Ant summary | `independent/ant/FINAL_ANT_REPORT.md` |
| Ant 10M retention | `independent/ant/RETENTION_10M.md` |
| Ant V_T transfer diagnostic (single seed, closed null) | `independent/ant/STAGE_ANT_VT_TRANSFER.md` |

## Directory classification

### `independent/` — V6 main evidence

All V6 from-scratch data, h checkpoints, actors, evaluations, and experiment
records are here.  Promoted artifacts are listed in `STATUS.md`.

Useful entry points:

- `independent/data/` — navigation h data.
- `independent/h/` — original navigation h.
- `independent/h_stageb_repair/` — repaired predictive h for the promoted
  Stage-B gate.
- `independent/nav_stageb_repair_16k/` — promoted 3-seed actor and aggregate.
- `independent/nav_repair_factorial/` — P1 repaired-h 2x2 (B/C/D), 3-seed
  aggregate and manifest.
- `independent/formal_3seed/h_only/` — matched h-only 3-seed baseline.
- `independent/ant/h_predictive/` — Ant predictive h.
- `independent/ant/mappo_h_pred_gate_10m/` — Ant 10M gate seeds 0/1/2.
- `independent/ant/mappo_cost_ramp10_10m/` — Ant matched cost-only seeds 0/1/2.

Historical/failed runs remain under `independent/` for the record
(`nav_h_gate_16k`, `nav_gate_progress_16k`, `nav_stageb_16k`,
`nav_stageb_gateoff_16k`, `safety_opt`, `single_seed_16k`, `runs`).  Smoke and
debug artifacts are archived under `independent/_archive_smoke_20261026/`.
They are not promoted.

### Top-level directories outside `independent/` — legacy diagnostics

`co_train_*`, `frozen_v6_*`, `h_data*`, `h_mech`, `h_swap`, `official_h*`,
`safety_h`, `gnn`, `ablation`, `protocol`, `smoke_co_train` are early coupled
experiments.  Many explicitly configure `arch: official_cbf` and load
`outputs/v6/official_h/official_h.pkl`.  They must not be used as V6 main
evidence, and must not be loaded by the independent protocol.

### Logs and one-off status files

Top-level `independent/*.log` and `overnight_status.json` are historical run
logs, not current status.  The older failed-path/pre-fix logs were moved to
`independent/_archive_smoke_20261026/nav/`.  The canonical status is
`STATUS.md`.

## Integrity rule

The V6 main path uses only the official simulator/reward/cost plus V6-collected
data and V6-trained checkpoints.  It does not use official h, `cbf.pkl`, QP /
`u_ref`, expert labels/actions, V5/V4 weights/data/checkpoints, `data_v5h0`,
`h_mech`, or official pretrained weights.
