# Predictive V6 Ant h + safety-gated forward: 2M three-seed gate

Date: 2026-10-24.  This is the follow-up to `GATE_2M_RESULTS.md`.  It fixes the
two gaps of the previous gate:

1. the h collector now retains the old `0` rows (current-safe but
   future-unsafe in the 32-step window);
2. h is trained on a future-window risk target, not the current margin.

The gate still acts only on the forward reward.  It is not an action filter,
QP, Lagrangian, or actor input correction.  Evaluation uses the official task
reward and official cost with the gate disabled.

## 1. Data and h

Command behavior policy: the matched V6 cost-only 2M checkpoint, stochastic
actions.  Only the official simulator/reward/cost is used; no expert action,
V5 artifact, official h, or task return is used by h.

- 120 episodes, 1000-step limit, 84,523 rollout rows;
- 30,000 threshold-focused augmentation rows;
- 114,523 total rows; 29,791 future/current-unsafe labels, 84,732 far-safe;
- horizon = 32, `h_scale=0.5`, `kappa=0.5`, `kappa_z=0.2`;
- target = `tanh(min_{k<=32} future_margin_{t+k} / 0.5)`;
- old `0` labels become `-1` (near-future unsafe); every row is retained;
- target and gradient supervision are applied to all rows.

Offline h (test split, `h_predictive/ant_h.pkl`):

| metric | value |
|---|---:|
| balanced accuracy | 0.9145 |
| unsafe recall | 0.8793 |
| safe recall | 0.9497 |
| false alarm | 0.0364 |
| AUROC | 0.9511 |
| gradient cosine | 0.9225 |
| h mean / std | -0.0909 / 0.3393 |

The previous dynamic h had no future labels and a current-margin target; this
predictive h is the temporal version that was missing.

## 2. Gate

Training only:

```text
gate = clip((h_next - 0.0) / (0.05 - 0.0), 0, 1)

reward_train =
    ctrl + contact + survive
    + gate * forward
    - cost_penalty * official_cost
```

`low=0.0`, `high=0.05`.  The threshold is not swept: safe states in the
cost-only behavior data have median h≈0.066, while future-unsafe states are
negative.  A lower threshold would re-enable forward in risk states; a higher
one would suppress ordinary forward movement.  Training gate activation was
0.76-0.80 on average and 0.81-0.92 over the last ten updates.

## 3. Three-seed 2M gate

100 evaluation episodes, eval seed 20000, identical seeds and checkpoints for
the cost-only control and predictive-h gate.

### Deterministic

| seed | arm | forward | final_x | safe | wall | topple | cost_fraction | survived |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | cost-only | 840.11 | 42.01 | 0.90 | 0.10 | 0.00 | 0.0044 | 0.99 |
| 0 | predictive gate | 572.13 | 28.61 | 0.95 | 0.05 | 0.00 | 0.0069 | 1.00 |
| 1 | cost-only | 888.47 | 44.43 | 0.24 | 0.74 | 0.02 | 0.0430 | 0.98 |
| 1 | predictive gate | 779.30 | 39.07 | 0.98 | 0.02 | 0.00 | 0.00007 | 1.00 |
| 2 | cost-only | 988.67 | 49.44 | 0.58 | 0.42 | 0.00 | 0.0124 | 1.00 |
| 2 | predictive gate | 918.37 | 45.98 | 1.00 | 0.00 | 0.00 | 0.0000 | 1.00 |
| **mean** | **cost-only** | **905.75** | **45.29** | **0.573** | **0.420** | **0.010** | **0.0199** | **0.990** |
| **mean** | **predictive gate** | **756.60** | **37.84** | **0.977** | **0.023** | **0.000** | **0.00232** | **1.000** |

### Stochastic

| seed | arm | forward | safe | wall | topple | cost_fraction | survived |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | cost-only | 496.91 | 0.14 | 0.57 | 0.55 | 0.0721 | 0.47 |
| 0 | predictive gate | 402.24 | 0.27 | 0.27 | 0.53 | 0.0306 | 0.47 |
| 1 | cost-only | 513.46 | 0.02 | 0.63 | 0.61 | 0.1165 | 0.39 |
| 1 | predictive gate | 499.30 | 0.30 | 0.29 | 0.48 | 0.0288 | 0.52 |
| 2 | cost-only | 565.77 | 0.04 | 0.61 | 0.54 | 0.1075 | 0.47 |
| 2 | predictive gate | 515.11 | 0.38 | 0.24 | 0.40 | 0.0198 | 0.60 |
| **mean** | **cost-only** | **525.38** | **0.067** | **0.603** | **0.563** | **0.0987** | **0.443** |
| **mean** | **predictive gate** | **472.22** | **0.317** | **0.267** | **0.470** | **0.0264** | **0.530** |

## 4. Verdict

- Deterministic mean cost falls by **88%** (`0.0199 -> 0.00232`); safe
  episodes rise `0.573 -> 0.977`, wall episodes `0.420 -> 0.023`, topple
  episodes `0.010 -> 0.000`.
- Forward retains **83.5%** of the cost-only mean (`756.6 / 905.75`), not a
  standstill or forward collapse.
- Stochastic mean cost falls by **73%** (`0.0987 -> 0.0264`); safe episodes
  rise `0.067 -> 0.317`; forward retains **90%**.
- All three stochastic seeds improve cost and safe episodes.  Deterministic
  seeds 1 and 2 improve both safety and cost.  Seed 0 is the caveat: its
  deterministic cost is slightly higher than its cost-only control
  (`0.0069 vs 0.0044`) while safe/wall improve (`0.95/0.05 vs 0.90/0.10`) and
  forward falls 32%.  The mean is nevertheless a clear robust safety win.
- Compared with the previous current-h gate (`GATE_2M_RESULTS.md`), predictive
  h is much safer and removes the seed-1 deterministic failure:
  det mean cost `0.0223 -> 0.00232`, safe `0.693 -> 0.977`, at 5% lower mean
  forward; stochastic cost `0.0432 -> 0.0264`, safe `0.180 -> 0.317`.

The pre-agreed 2M robust gate is therefore **passed**.  The seed-0 10M
retention follow-up also passes (see below); multi-seed 10M and navigation
transfer remain open.

## 5. 10M follow-up and remaining items

The 10M retention check passes for all three seeds.  Seed 0 deterministic:
`forward=3723.09`, `safe=0.97`, `cost=0.00176` versus cost-only
`4055.37 / 0.10 / 0.0222`; stochastic `2923.73 / 0.78 / 0.00419` versus
`3067.39 / 0.04 / 0.0508`.  The completed 3-seed sweep gives deterministic
gate mean `forward=3771.51` (`91.6%`), `safe=0.97`, cost_fraction `0.0012`
versus cost-only `4119.70 / 0.063 / 0.0296`; stochastic `3010.73` (`93.9%`),
`safe=0.84` versus `3206.28 / 0.020`.  See `RETENTION_10M.md` and
`MULTISEED_10M_PLAN.md`.

Still open:

- Navigation task-joint run.
- No separate cost critic; the safety-gated forward was sufficient for both
  the 2M and 3-seed 10M gates.  A cost advantage/cost critic remains a
  fallback only if the navigation transfer fails.
- No explicit actor `hdot` feature; the future-window h target supplies the
  temporal signal.  `hdot` remains a possible diagnostic/fallback.
