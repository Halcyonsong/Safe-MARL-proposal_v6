# V6 Ant records

Canonical project status: [`../../STATUS.md`](../../STATUS.md).

## Promoted result

Predictive-h + safety-gated forward, 3 seeds x 10M, 100 deterministic and 100
stochastic episodes:

| mode | arm | forward | safe_ep | cost_fraction | retention |
|---|---|---:|---:|---:|---:|
| deterministic | cost-only | 4119.70 | 0.063 | 0.0296 | 100% |
| deterministic | predictive gate | **3771.51** | **0.97** | **0.0012** | **91.6%** |
| stochastic | cost-only | 3206.28 | 0.020 | 0.0618 | 100% |
| stochastic | predictive gate | **3010.73** | **0.84** | **0.0020** | **93.9%** |

Every gate seed is safe (`0.97/0.95/0.99` deterministic); every cost-only seed
is unsafe (`0.10/0.08/0.01`).

Raw artifacts:

- `h_predictive/` — predictive h checkpoint and offline metrics.
- `mappo_h_pred_gate_10m/seed{0,1,2}/` — promoted gate actors.
- `mappo_cost_ramp10_10m/seed{0,1,2}/` — matched cost-only controls.
- `eval_10m_pred_gate/` — formal evaluation outputs.

## Document index

| document | status / role |
|---|---|
| [`FINAL_ANT_REPORT.md`](FINAL_ANT_REPORT.md) | current consolidated Ant report (updated with 3-seed 10M) |
| [`RETENTION_10M.md`](RETENTION_10M.md) | **promoted**: 2M/5M/10M seed-0 retention + 3-seed 10M sweep |
| [`MULTISEED_10M_PLAN.md`](MULTISEED_10M_PLAN.md) | **promoted**: pre-registered 3-seed 10M protocol and results |
| [`PREDICTIVE_H_GATE_2M.md`](PREDICTIVE_H_GATE_2M.md) | **promoted**: predictive h + forward gate at 2M, 3 seeds |
| [`GATE_2M_RESULTS.md`](GATE_2M_RESULTS.md) | archived: earlier safety-gated forward gate; did not pass its robust rule, superseded by predictive h |
| [`DYNAMIC_H_GATE.md`](DYNAMIC_H_GATE.md) | archived negative: dynamic-h + barrier, terminated |
| [`STAGE_ANT_VT_TRANSFER.md`](STAGE_ANT_VT_TRANSFER.md) | diagnostic, **closed null**: single-seed 1-D `A_T` warm-start did not beat the matched no-V_T 1M control |
| [`EFFECT_ANALYSIS.md`](EFFECT_ANALYSIS.md) | analysis of the dynamic-h failure; not a promoted result |
| [`PILOT_DIAGNOSIS.md`](PILOT_DIAGNOSIS.md) | historical 1M pilot diagnosis; its "h branch terminated" refers to static/dynamic h, not predictive h |

## Status labels

- **Promoted:** predictive-h + safety-gated forward at 2M and 10M.
- **Matched control:** scheduled cost-only at 10M.
- **Archived/terminated:** static h actor features, dynamic-h barrier, earlier
  safety-gated forward gate.
- **Open:** optional 2M/5M multi-checkpoint retention evaluation; navigation
  task-joint run not started.
- **Closed null diagnostic:** the single-seed Ant V_T transfer screen
  ([`STAGE_ANT_VT_TRANSFER.md`](STAGE_ANT_VT_TRANSFER.md)) passed its V_T
  quality gate and had a valid control, but the 1-D `A_T` warm-start did not
  beat the matched no-V_T 1M continuation on deterministic forward
  (`3960.83` vs `4014.03`) or return (`4574.53` vs `4630.37`); safety was
  equal (`0.96` vs `0.96`).  `vt_transfer_supported = false`; the line is
  stopped.  It reuses the promoted seed-0 10M h-only base and does not change
  the promoted 3-seed claim.
