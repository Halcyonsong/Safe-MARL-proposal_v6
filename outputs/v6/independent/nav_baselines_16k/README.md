# Matched 16k collision-aware navigation baselines

Status: **complete 2026-10-27 10:51 CST**, 3 seeds per family, 100
deterministic episodes each.  Canonical status:
[`../../STATUS.md`](../../STATUS.md).

## Why these runs exist

The existing `BASELINE_COMPARISON.md` rows have a gap:

- `pure MAPPO 16k` uses `collision_penalty=0` (task-only reward);
- `MAPPO-Lag 100k` uses the official collision cost but a 100k budget.

Neither is a matched 16k collision-aware baseline.  These runs fill that gap.
They are **baselines only**; the promoted V6 main claim is unchanged and still
comes from the frozen-h Stage-B actor.

## Protocol (pre-registered)

| arm | mechanism | reward / cost | steps | seeds | eval |
|---|---|---|---:|---|---|
| `mappo_same_reward` | MAPPO, fixed penalty | task reward + `collision_penalty=10` | 16,384 | 0/1/2 | 100 det episodes, eval seeds 10000..10099 |
| `mappo_lag` | MAPPO-Lagrangian | task reward + official collision cost, `cost_limit=0.01` | 16,384 | 0/1/2 | 100 det episodes, eval seeds 10000..10099 |

Common protocol: official GCBF+ `DoubleIntegrator`, 8 agents, 4x4 area,
8 obstacles, 256 steps, final checkpoint (`policy_final.pth`, step 16,384),
same 100 fixed scenes as the V6 rows.  No official h, `cbf.pkl`, QP, `u_ref`,
or V5/V4 weights are used.

The two arms are deliberately labelled as **different mechanisms**:

- `mappo_same_reward` is MAPPO with a fixed collision penalty in the reward;
- `mappo_lag` is MAPPO with a learned Lagrangian multiplier on the official
  collision cost.

## Artifacts

```text
outputs/v6/independent/nav_baselines_16k/
  mappo_same_reward/seed{0,1,2}/   # train + 100ep eval + aggregate
  mappo_lag/seed{0,1,2}/          # train + 100ep eval + aggregate
  logs/                           # chain and family logs
  chain.log                       # start/finish timestamps
```

## Results (100 deterministic episodes, eval seeds 10000..10099)

### MAPPO + fixed collision penalty 10

| seed | safe_ep | collision_ep | official_safety | success | reach | return |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.000 | 1.000 | 0.3513 | 0.0187 | 0.0737 | -2.72 |
| 1 | 0.040 | 0.960 | 0.5837 | 0.0138 | 0.0238 | -43.05 |
| 2 | 0.060 | 0.940 | 0.6225 | 0.0100 | 0.0325 | -45.28 |
| **mean** | **0.0333** | **0.9667** | **0.5192** | **0.0142** | **0.0433** | **-30.35** |
| **std** | 0.0306 | 0.0306 | 0.1467 | 0.0044 | 0.0267 | 23.95 |

### MAPPO-Lag + official collision cost

| seed | safe_ep | collision_ep | official_safety | success | reach | return |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.000 | 1.000 | 0.3075 | 0.0788 | 0.2587 | 43.24 |
| 1 | 0.000 | 1.000 | 0.4000 | 0.0550 | 0.1362 | 21.59 |
| 2 | 0.000 | 1.000 | 0.2575 | 0.1250 | 0.4238 | 53.83 |
| **mean** | **0.0000** | **1.0000** | **0.3217** | **0.0863** | **0.2729** | **39.55** |
| **std** | 0.0000 | 0.0000 | 0.0723 | 0.0356 | 0.1443 | 16.43 |

### Comparison to the promoted V6 row

| arm | safe_ep | official_safety | success | reach | return |
|---|---:|---:|---:|---:|---:|
| MAPPO + fixed penalty 10, 16k | 0.0333 | 0.5192 | 0.0142 | 0.0433 | -30.35 |
| MAPPO-Lag, 16k | 0.0000 | 0.3217 | 0.0863 | 0.2729 | 39.55 |
| **V6 repaired Stage B, 16k** | **0.6933** | **0.9417** | **0.4100** | **0.4296** | **66.85** |

Interpretation:

- The fixed penalty suppresses task at 16k (`reach=0.043`); the negative
  return includes `-10` per collision step.  It is also not safe
  (`safe_ep=0.033`).
- MAPPO-Lag moves more than the fixed-penalty arm but has no collision-free
  episode at 16k.  Its learned multiplier is around `0.20-0.21`.
- Neither matched baseline approaches V6 on safety or task.  The older
  100k MAPPO-Lag row remains a generous external reference, not a matched row.

## Evaluation note

One MAPPO-Lag seed-2 chunk (eval scenes `10000..10019`) hit a transient
JAX `SystemError: unknown opcode` while ten chunk processes were running
concurrently.  The chunk was rerun alone with the identical checkpoint,
episode count, and scene range; the rerun succeeded.  No episode was dropped
or replaced.  Final counts are `100` episodes for every seed.

## Decision rule

Rows are reported as 3-seed mean ± std over `safe_episode`,
`official_safety`, `official_success`, `reach`, and `episode_return`.
No safety ranking is read from the 10-episode in-training monitor; only the
100-episode evaluation counts.
