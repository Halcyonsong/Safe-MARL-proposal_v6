# V6 Ant 10M multi-seed run (seeds 1 and 2)

Status: **complete** (2026-10-25).  The first launch was 2026-10-25 00:21:52,
but the machine restarted around 05:10 and the background runner was
relaunched; the four runs that produced the results below started at 05:10 and
all training plus the final 100-episode deterministic/stochastic evaluations
finished at 02:34 UTC.  All four arms reached 10,000,000 steps.

## Purpose

Complete the matched 3-seed 10M comparison for the promoted Ant mechanism:

- predictive V6 Ant h (minimum future geometry margin over 32 steps)
- safety-gated forward task reward
- fixed scheduled official-cost penalty (no Lagrangian, no QP)

Seed 0 already passed the 10M retention check.  Seeds 1/2 are the missing
multi-seed evidence.

## Arms

| arm | config | h | output |
|---|---|---|---|
| predictive gate | `nav/configs/v6/v6_ant_h_pred_gate_2m.yaml` + `--total-steps 10000000` | `outputs/v6/independent/ant/h_predictive/ant_h.pkl` | `outputs/v6/independent/ant/mappo_h_pred_gate_10m/seed{1,2}` |
| matched cost-only | `nav/configs/v6/v6_ant_cost_ramp10_10m.yaml` | none | `outputs/v6/independent/ant/mappo_cost_ramp10_10m/seed{1,2}` |

Both arms use the same MAPPO base, 10M steps, 8 envs, 500k eval interval,
20 training-eval episodes, and the scheduled cost penalty
`0.1 -> 1.0` (ramp start 500k, length 1.5M).  The predictive arm additionally
gates only the forward component with `low=0.0`, `high=0.05`.

## Seeds and evaluation

| training seed | training eval seed | formal eval episodes |
|---:|---:|---:|
| 1 | 20100 | 100 deterministic + 100 stochastic |
| 2 | 20200 | 100 deterministic + 100 stochastic |

Seed 0 used eval seed 20000.  Evaluation uses the official task reward and
official cost, not the training gate.

## Results

### Deterministic, 100 episodes

| seed | arm | forward | final_x | safe | wall | cost_fraction | survived |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | predictive gate | 3723.09 | 186.16 | 0.97 | 0.03 | 0.0018 | 1.00 |
| 0 | cost-only | 4055.37 | 202.77 | 0.10 | 0.89 | 0.0222 | 0.98 |
| 1 | predictive gate | 3700.54 | 185.03 | 0.95 | 0.03 | 0.0018 | 0.98 |
| 1 | cost-only | 4111.44 | 205.57 | 0.08 | 0.89 | 0.0264 | 0.92 |
| 2 | predictive gate | 3890.90 | 194.55 | 0.99 | 0.00 | 0.0000 | 0.99 |
| 2 | cost-only | 4192.30 | 209.62 | 0.01 | 0.99 | 0.0401 | 0.96 |
| mean | predictive gate | **3771.51** | 188.58 | **0.97** | **0.02** | **0.0012** | 0.99 |
| mean | cost-only | 4119.70 | 205.99 | 0.063 | 0.923 | 0.0296 | 0.953 |

### Stochastic, 100 episodes

| seed | arm | forward | final_x | safe | wall | cost_fraction | survived |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | predictive gate | 2923.73 | 146.19 | 0.78 | 0.05 | 0.0042 | 0.83 |
| 0 | cost-only | 3067.39 | 153.38 | 0.04 | 0.91 | 0.0508 | 0.81 |
| 1 | predictive gate | 2936.00 | 146.80 | 0.85 | 0.01 | 0.0011 | 0.86 |
| 1 | cost-only | 3094.38 | 154.72 | 0.02 | 0.94 | 0.0548 | 0.69 |
| 2 | predictive gate | 3172.47 | 158.63 | 0.89 | 0.01 | 0.0007 | 0.90 |
| 2 | cost-only | 3457.07 | 172.86 | 0.00 | 1.00 | 0.0798 | 0.81 |
| mean | predictive gate | **3010.73** | 150.54 | **0.84** | **0.023** | **0.0020** | 0.863 |
| mean | cost-only | 3206.28 | 160.32 | 0.020 | 0.950 | 0.0618 | 0.770 |

### Verdict

- Deterministic: the gate retains **91.6%** of cost-only forward progress
  (`3771.51 / 4119.70`), cuts `cost_fraction` by **96.0%**
  (`0.0012 vs 0.0296`), and raises mean `safe_episode` from `0.063` to
  `0.97`.  Every seed is safe (`0.97/0.95/0.99`) while every cost-only seed
  is unsafe (`0.10/0.08/0.01`).
- Stochastic: the gate retains **93.9%** of forward progress, cuts cost by
  **96.8%** (`0.0020 vs 0.0618`), and raises mean `safe_episode` from `0.02`
  to `0.84`.
- No seed shows a late safety collapse at 10M.  The 3-seed 10M retention
  claim now holds for Ant.
- The gate is training-only; evaluation uses the official task reward and
  official cost.

## Runner

`nav/scripts/run_v6_ant_multiseed_overnight.sh` launches the four runs
concurrently on CPU (`--device cpu`) with `nice 5`, so the Stage-B navigation
GPU work is not blocked.  Each run writes its final `policy_final.pth`,
`eval_100ep_deterministic.json`, and `eval_100ep_stochastic.json` under its
seed directory.  A `FAILED` marker is written on failure.

Launch record: `outputs/v6/independent/ant/multiseed_10m_launch.log`.

## Analysis after completion

- Compare per-seed and mean `safe_episode`, `collision`/`wall`/`topple`,
  `forward_return`, `final_x`, and `cost_fraction`.
- Check that the gate retains task progress and does not show a late safety
  collapse at 10M.
- Update `RETENTION_10M.md`, `FINAL_ANT_REPORT.md`, and
  `outputs/v6/V6_COMPLETE_RECORD.md`; do not overwrite seed 0.
