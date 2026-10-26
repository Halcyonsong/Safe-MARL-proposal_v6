# V6 navigation Stage C0: h2-only temporal-calibration screen

Status: **C0a (seed 0) failed the pre-registered pilot rule.  C0b was not
run; the original C1 was not authorized.**  At the C0 decision no actor2 was
trained.  A separately approved **C1a actor2-from-scratch pilot** was later
run and failed all five checks (`safe 0.510`, `official_safety 0.89375`,
`success 0.33125`, `return 52.22`); the from-scratch h2/actor2 branch is
stopped.  The h2-only data/target retraining did not close the long-horizon
gap — it improved global/safe calibration but worsened every risk-region
metric.  A post-hoc seed-0 h2 + actor1 closed-loop swap was slightly better
than h1, but it is one seed and not significant; it does not change the C0a
decision.  See sections 6c and 6d; C1a record:
[`STAGE_C1A_H2_ACTOR2_PILOT.md`](STAGE_C1A_H2_ACTOR2_PILOT.md).

Canonical status: [`../../STATUS.md`](../../STATUS.md).  Diagnostic that
motivated C0:
[`H2_ACTOR2_DIAGNOSTIC.md`](H2_ACTOR2_DIAGNOSTIC.md).

## 1. Motivation and hypothesis

h1 was trained on h-only seed0 and Stage-A gate-progress seed0 rollouts plus
fixed random coverage, not on the final repaired Stage-B actor1.  The
read-only audit shows the resulting off-policy gap:

- 155/155 collision onsets occur with `h_cur < 0` and `gate_next = 0`, so h1
  is not blind at impact;
- but on actor1 states h1 is optimistic at long lead: 17-32 steps before
  collision, h mean `+0.242` versus target `-0.222`;
- the future-collision-region MAE is `0.394`, and pooled MAE `0.0976` is
  dominated by far-safe rows.

**Hypothesis:** retraining h on the final actor1 distribution, with the same
32-step wall-free target and the same architecture/loss/calibration rule, will
reduce the 17-32-step risk-region error and lengthen the warning lead.

This is the first half of the iterative h/actor loop.  C0 tests the h side in
isolation; C1 (actor2 from scratch) is only considered if C0 passes.

## 2. Scope and non-goals

In scope:

- collect actor1 stochastic rollouts per seed;
- merge them with the same fixed random coverage used by h1;
- train h2 from scratch;
- calibrate the gate on h2 validation with the same rule;
- evaluate h1 vs h2 offline on held-out actor1 states.

Not in scope:

- no actor2 training;
- no closed-loop policy evaluation in C0;
- no gate-scope, cost-schedule, or task-channel change;
- no `task_gate_high` sweep;
- no warm start from h1;
- no official h, `cbf.pkl`, QP/`u_ref`, expert labels/actions, or V5/V4
  artifacts.

## 3. Data collection

For each seed `s in {0,1,2}`:

- behavior policy: `outputs/v6/independent/nav_stageb_repair_16k/seed{s}/
  policy_final.pth` (fixed final actor1);
- behavior h: `outputs/v6/independent/h_stageb_repair/v6_h_stageb_repair.pkl`
  (frozen h1);
- **training data:** 100 stochastic episodes, 256 steps, environment seeds
  `30000 + 1000*s .. 30099 + 1000*s`; split by episode 80 train / 10
  validation / 10 test;
- **held-out evaluation data:** 101 deterministic episodes (mean actions,
  matching the formal evaluation), 256 steps, environment seeds
  `31000 + 1000*s .. 31100 + 1000*s`; the collector is run with
  `val_ratio=0.01, test_ratio=1.0` so the held-out test split contains 100
  episodes and 25,600 timesteps.  This is the C0 pass/fail set and is disjoint
  from both the training seeds and the formal evaluation seeds `10000..10099`;
- target: wall-free `tanh(future_min_margin / 0.2)`, horizon 32,
  horizon-unsafe rows forced to `<= -0.1`.

The fixed random coverage is
`outputs/v6/independent/data_random_rebuilt/` (80/10/10 episodes, already
rebuilt with the same wall-free target and target gradients).  The training
merge is 50/50 by timesteps, matching the h1 training mix:

| split | actor1 rows | random rows |
|---|---:|---:|
| train | 20,480 | 20,480 |
| val | 2,560 | 2,560 |
| test | 2,560 | 2,560 |

The merged test split is a training diagnostic only.  C0 pass/fail uses the
separate 100-episode held-out actor1 evaluation set above, not the formal
100-episode audit.

## 4. h2 training

Same objective and hyperparameters as the repaired h1:

- V6 full-graph GNN, hidden 128, 2 layers, `gnn_layers=1`;
- 40 epochs, batch 256, AdamW, lr `1e-3`, weight decay `1e-4`;
- classification hinge with `safe_margin=0.02`, `unsafe_weight=1.0`;
- continuous target `graph_margin_targets`, coefficient `0.5`,
  SmoothL1(`delta=1.0`), on all target rows;
- diagonal state-gradient alignment, coefficient `0.1`, on all labeled rows;
- `hdot_coef=0.0`;
- global gradient-norm clip `1.0`;
- central-difference gradient step `1e-3`;
- trained from scratch, one h2 per seed.

## 5. Gate calibration

Same rule as h1, on the merged h2 validation split:

```
low  = 0.0
high = max(0.02, q10_safe(h_next + h_offset))
```

No sweep.  The resulting `gate_calibration.json` is used only for the C0
evaluation; C0 does not train an actor or apply the gate.

## 6. Evaluation and pre-registered pass rule

For each seed, evaluate h1 and h2 on exactly the same rows of the 100-episode
held-out actor1 evaluation set (`seed{s}_eval/test_data.npz`).  Both are
shifted by their own checkpoint `h_offset`; each uses its own calibrated gate
high for lead/region diagnostics.

Primary metric:

- **17-32-step risk-region MAE**: rows whose next geometric collision
  (`graph_margin_values <= 0`) is 17-32 steps away; MAE between h and the
  32-step target.

Secondary metrics:

- warning lead: last step with `h >= gate_high` before a collision onset
  (geometric margin crosses zero), and last step with `h > 0`;
- 9-16-step risk-region MAE;
- future-collision AUROC (`graph_future_margin <= 0`);
- global target MAE/RMSE/correlation, safe-row correlation, BA/AUROC.

**Pass requires all of:**

1. mean 17-32-step MAE reduction versus h1 `>= 20%` across the three seeds;
2. at least 2/3 seeds individually show `>= 15%` reduction;
3. mean warning-lead median increases by `>= 2` steps, with no seed decreasing
   by more than 1 step;
4. mean global target MAE is no worse than h1 by more than `0.01`;
5. mean safe-row correlation is no worse than h1 by more than `0.02`;
6. mean future-collision AUROC is no worse than h1 by more than `0.02`;
7. mean balanced accuracy is no worse than h1 by more than `0.01`.

If the rule passes, C1 may be proposed: actor2 from scratch with h2, same
cost, 16,384 steps, 3 seeds, 100ep eval seed 10000, no warm start, no gate
sweep, no task channel.  If it fails, do not train actor2; the next arm is
safety coupling / action conditioning, with explicit approval.

The formal 100-episode actor1 audit is **not** a C0 selection set.  After C0,
it may be rerun read-only with h2 as a final confirmation.

## 6b. Staged execution

The full h2 training is CPU-heavy (h1 took several hours per seed under CPU
JAX).  C0 is therefore staged:

- **C0a (seed 0 pilot):** run the full protocol above for seed 0 only.  Pilot
  pass is a relaxed version of the final rule: 17-32-step MAE reduction
  `>= 15%`, warning-lead median increase `>= 1` step, global MAE no worse by
  more than `0.02`, and safe correlation no worse by more than `0.03`.
- **C0b (seeds 1/2):** only if C0a passes.  The final C0 decision uses the
  full 3-seed rule in section 6; C0a is not a final claim.
- If C0a fails, do not run C0b or C1; the next arm is safety coupling /
  action conditioning, with explicit approval.

This staging is fixed before C0a results are seen.

## 6c. Result (C0a, seed 0): FAILED

Run: 2026-10-25T15:01Z to 2026-10-25T23:30Z.  h2 trained 40 epochs from
scratch; best epoch 33 by the standard target-correlation selection score.

Evaluation set: 100 deterministic actor1 episodes, seeds
`31000..31100`, 25,600 rows.  After the time-to-collision fix (see below),
the 17-32-step bin contains 361 rows, the 9-16-step bin 243 rows, current
collisions 181 rows, and 41 collision onsets for the lead statistic.  The
six 0-32-step bins sum to 1,090 rows, exactly the number of
`graph_future_margin <= 0` rows, which is the consistency check for the fix.

| metric | h1 | h2 | h2 - h1 |
|---|---:|---:|---:|
| **17-32-step MAE** (primary) | 0.7087 | 0.8258 | **+0.1170 (+16.5% worse)** |
| 9-16-step MAE | 0.4419 | 0.5666 | +0.1247 (+28.2% worse) |
| future-collision MAE | 0.3753 | 0.4784 | +0.1031 |
| current-collision MAE | 0.0442 | 0.1280 | +0.0838 |
| global target MAE | 0.0921 | 0.0840 | -0.0081 (better) |
| safe-row target corr | 0.6862 | 0.7347 | +0.0484 (better) |
| balanced accuracy | 0.99955 | 0.99531 | -0.0042 |
| future-collision AUROC | 0.99999 | 0.99998 | ~0 |
| warning lead, last `h >= gate_high`, median | 17 | 28 | +11 |
| warning lead, last `h > 0`, median | 8 | 8 | 0 |

Pilot checks:

| check | result |
|---|---|
| 17-32 MAE reduction `>= 15%` | **false** (16.5% worse) |
| warning-lead median increase `>= 1` step | true (nominal only) |
| global MAE no worse by `> 0.02` | true |
| safe corr no worse by `> 0.03` | true |

**Decision: C0a fails.**  The primary risk-region metric moved in the wrong
direction, and every other risk-region metric (9-16 MAE, future-collision
MAE, current-collision MAE) also degraded.  C0b is not run and the original
C1 is not authorized.  A separately approved C1a actor2 pilot was later run
and failed all five checks; see `STAGE_C1A_H2_ACTOR2_PILOT.md`.

Interpretation:

- h2 did learn the *average* target better: offline validation/test target
  correlation rose from h1's `0.8004/0.7940` to `0.8314/0.8345`, and global
  actor1 MAE fell.  But the improvement is concentrated in far-safe rows.
- In the 9-32-step pre-collision region h2 is **more optimistic** than h1:
  h mean `+0.623` versus target `-0.203` in the 17-32 bin, with only 12.5% of
  rows negative (same fraction as h1).  The 17-32 MAE therefore increased.
- The apparent +11-step warning lead is a **gate-threshold artifact**, not an
  earlier warning.  h2's own q10-safe calibration moved `gate_high` from
  `0.2673` to `0.6621`; under that higher threshold, far-safe h values stay
  above the gate longer.  The threshold-free lead, last `h > 0`, is unchanged
  at 8 steps, and h2 is *less* negative at impact (`-0.068` versus `-0.206`).
  h2's gate lead is available for only 39/41 onsets (two never cross the
  0.662 threshold), and h2's gate would close on 14,589 evaluation rows
  versus h1's 4,554.
- This is consistent with the diagnostic's mixed label: h1's early warning is
  imperfect, but retraining h only on actor1 data does not fix it and can
  trade risk-region calibration for global fit.  The impact-time failure
  remains actor/action-limited.
- Data-coverage check (read-only): in the actor1 training half, only
  `299/163,840` rows (0.18%) fall in the 17-32-step pre-collision bin and
  `944` (0.58%) have a future collision; the random half supplies `5,575`
  (3.40%) and `17,834` (10.9%).  The 50/50 merged set has 1.79% / 5.73%,
  close to h1's merged 1.96% / 6.18%.  The actor1 distribution itself carries
  very little long-horizon risk signal, so C0 did not add the temporal
  examples the hypothesis needed.  This is a one-seed pilot, so it is a
  decisive failure of the pre-registered rule, not a variance estimate.

Evaluator note: the first C0a evaluation run had a time-to-collision
recurrence bug (`dist[t] = dist[t] + 1` instead of `dist[t+1] + 1`), which
silently emptied every 1-32-step bin and made the primary metric `null`.
The evaluator was corrected to the pre-registered definition; the pass rule
and data were not changed.  The corrected numbers above are the final C0a
result.  The conclusion does not depend on the fix: h2 was already worse on
future-collision and 9-16-step MAE before it.

## 6d. Post-hoc seed-0 closed-loop h2 + actor1 diagnostic

After C0a failed, a read-only h-swap was run at the user's request: the same
actor1 seed-0 checkpoint, h2 seed-0 instead of h1, 100 deterministic
episodes, eval seed 10000.  This is **not** a C0/C1 result and does not change
the pre-registered decision.  Full record:
[`../nav_c0_h2_actor1_swap/README.md`](../nav_c0_h2_actor1_swap/README.md).

| metric | actor1 + h1 | actor1 + h2 | delta |
|---|---:|---:|---:|
| safe_episode | 0.840 | 0.900 | +0.060 |
| collision_episode | 0.160 | 0.100 | -0.060 |
| official_safety | 0.97375 | 0.98500 | +0.01125 |
| official_success | 0.42750 | 0.43875 | +0.01125 |
| official_reach | 0.43500 | 0.44625 | +0.01125 |
| episode_return | 60.91 | 61.40 | +0.49 |
| warning_episode | 0.210 | 0.110 | -0.100 |
| h_mean | 0.836 | 0.930 | +0.093 |
| h_min | 0.648 | 0.762 | +0.114 |
| cbf_violation | 0.0446 | 0.0381 | -0.0065 |

Interpretation:

- The closed-loop seed-0 diagnostic does **not** reproduce the offline
  risk-region degradation.  It shows a small improvement instead, but the
  differences are one seed and small: safe episode +6 pp (approximately
  `z=1.27`), official safety/success/reach +1.1 pp, return +0.49.  They are
  not statistically separated.
- The swap is off-policy: actor1 was trained with h1, and h2 shifts the h
  input upward (`h_mean +0.093`, `h_min +0.114`).  The actor's behavior
  changes because of the input shift; a higher h can make actor1 more
  conservative.  This is not evidence that h2 is better calibrated for a
  newly trained actor2.
- Offline 17-32 MAE weights states uniformly, while closed-loop safety
  depends on the actor's response.  The two can diverge.  This is a
  limitation of using offline calibration alone as the C0 gate.
- **The pre-registered C0a rule remains failed.  C0b and C1 remain
  unauthorized.**  Do not use this diagnostic to promote h2 or to authorize
  actor2.  Pursuing h2 further would require a new pre-registration with a
  multi-seed closed-loop criterion and explicit approval.

## 7. Artifacts

- actor1 training data: `outputs/v6/independent/data_c0_h2/seed{s}_actor1/`
- held-out actor1 evaluation data:
  `outputs/v6/independent/data_c0_h2/seed{s}_eval/`
- merged training data: `outputs/v6/independent/data_c0_h2/seed{s}_merged/`
- h2 checkpoint/summary: `outputs/v6/independent/h_c0/seed{s}/`
- gate calibration: `outputs/v6/independent/h_c0/seed{s}/
  gate_calibration.json`
- h1-vs-h2 evaluation: `outputs/v6/independent/nav_c0_eval/seed{s}.json`
- logs: `outputs/v6/independent/nav_c0/logs/seed{s}.log`
- corrected seed-0 evaluation log:
  `outputs/v6/independent/nav_c0/logs/seed0_eval_recheck.log`
- post-hoc closed-loop h2 + actor1 seed-0 diagnostic:
  `outputs/v6/independent/nav_c0_h2_actor1_swap/`
- config: `nav/configs/v6/v6_nav_c0_h2.yaml`
- collector/merge/train/eval scripts: see below.

## 8. Commands

Smoke (pipeline check only, no claim):

```bash
SMOKE=1 SEEDS=0 bash nav/scripts/run_v6_nav_c0_h2.sh
```

Full C0 (requires explicit confirmation):

```bash
CONFIRM=1 SEEDS="0 1 2" bash nav/scripts/run_v6_nav_c0_h2.sh
```

The runner refuses full C0 without `CONFIRM=1`.  It never trains an actor.

## 9. Status

- [x] Pre-registration written.
- [x] Collector/merge/train/eval/runner scripts written.
- [x] Smoke test (5 train + 5 eval episodes, 1 epoch): pipeline completes
      end-to-end; smoke metrics are not evidence.
- [x] C0a seed-0 pilot (full 40 epochs, 100ep held-out evaluation):
      **failed**; 17-32-step MAE 0.826 versus h1 0.709 (+16.5%).
- [x] C0b seeds 1/2: **not run**, because C0a failed.
- [x] C0 decision: **failed; stop h-only iteration.**
- [x] Original C1 actor2: **not authorized / not trained.**
- [x] Separately approved C1a actor2-from-scratch pilot: **failed all five
      checks**; from-scratch h2/actor2 branch stopped.  Record:
      `STAGE_C1A_H2_ACTOR2_PILOT.md`.
- [x] Post-hoc seed-0 h2 + actor1 closed-loop diagnostic: slightly better
      than h1 but one seed and not significant; does not change C0a.
- [ ] Next controlled arm, if approved: safety coupling or action
      conditioning (one axis only).
