# V6 TODO

Updated: 2026-10-26.

Canonical status and artifact map: [`STATUS.md`](STATUS.md).

Current state: Ant 10M 3-seed pass; navigation repaired Stage-B 3-seed pass.
The unchecked items below are follow-ups and explicit non-goals, not blockers
for the promoted claims.

## Highest priority

- [x] Ant 10M seeds 1 and 2, same predictive-h gate config.  Completed
      2026-10-25; det safe `0.95/0.99`, forward `3700.5/3890.9`.
- [x] Matched cost-only 10M seeds 1 and 2.  Det safe `0.08/0.01`, forward
      `4111.4/4192.3`.
- [x] Final 100-episode deterministic and stochastic evaluation, all three
      seeds, eval seed 20000 + 100*seed.
      3-seed det gate mean: forward 3771.51 (91.6% retention), safe 0.97,
      cost_fraction 0.0012.  3-seed stoch: forward 3010.73 (93.9%),
      safe 0.84, cost_fraction 0.0020.
- [x] Updated `outputs/v6/independent/ant/RETENTION_10M.md`,
      `MULTISEED_10M_PLAN.md`, and `outputs/v6/V6_COMPLETE_RECORD.md`.
- [ ] Optional: Ant 10M multi-checkpoint evaluation at 2M/5M (retention
      already established at 10M; not needed for the main claim).
- [x] **Ant V_T transfer diagnostic (single seed, closed null):** 1-D `A_T`
      warm-start from the existing seed-0 10M h-only base, 1M extra steps,
      matched no-V_T 1M control; frozen h/cost/reward, V_T trained offline
      on Ant task return only.  V_T gate passed (val EV `0.4838`, test
      Pearson `0.6597`), control valid (safe `0.96`, cost_fraction
      `0.000187`), but warm-start forward `3960.83` vs control `4014.03`
      (`-1.33%`) and return `4574.53` vs `4630.37` (`-1.21%`); safety equal
      (`0.96`).  `vt_transfer_supported = false`; line stopped.  Diagnostic
      only; the promoted Ant 3-seed claim is unchanged.  Record:
      `outputs/v6/independent/ant/STAGE_ANT_VT_TRANSFER.md`.

## Navigation transfer

- [x] Add per-agent positive-task gate to `nav/v6/training.py`:
      `gate_i = clip((h_next_i-low)/(high-low),0,1)`;
      gate progress/reach only, never collision cost.
- [x] Calibrate `low=0.0`, `high=0.13` on V6 nav h (healthy plateau ~0.138).
- [x] Add `nav/configs/v6/v6_nav_h_gate.yaml` and gate unit test.
- [x] Run 1-seed 16k h+gate and 100ep eval vs existing h-only.
      Seed 0: safe 0.62 -> 0.70, success 0.53 -> 0.37.  Current gate is a
      safety/task tradeoff, not a promoted result.
- [ ] Broad-gate seeds 1/2 remain unevaluated; that gate design is archived
      (safety/task tradeoff), not promoted.
- [ ] Add formal 16k h-off 3-seed baseline if needed.
- [ ] Add monitor-only safety eval and intermediate checkpoints for the next
      run; do not select the formal checkpoint from training eval.
- [ ] Add 100ep stochastic evaluation of the existing h-only/gate checkpoints.
- [x] Stage A code/config (isolate safety feedback): keep frozen V6 h,
      retrain actor from scratch with Ant-style reward decomposition: gate
      dense progress only, leave reach/terminal components un-gated; keep
      official cost penalty unchanged.  Config:
      `nav/configs/v6/v6_nav_gate_progress_16k.yaml`.
- [x] Stage A seed-0 16k training.  Final 20ep monitor: safe 0.700,
      success 0.5813, return 80.19 (h-only 16k monitor: 0.500 / 0.5125 /
      68.68).  The 8k monitor was bad (safe 0.100), so the recovery is
      late-training, not monotonic.
- [x] Stage A seed-0 100ep deterministic eval: safe_episode 0.530 vs h-only
      0.620, success 0.5075 vs 0.5325, return 76.85 vs 69.78.  Stage A fails
      the safety guardrail; do not run seeds 1/2.
- [x] Stage B (isolate h quality): 2026-10-25 complete.  100ep on-policy
      data; h trained and calibrated (val BA 0.8714, AUROC 0.9406; test BA
      0.9029, AUROC 0.9673); gate low/high=0.0/0.02 from the pre-registered
      rule.  Actor seed0 100ep: `safe_episode=0.000`, `collision=1.000`,
      `success=0.0375`, `return=-23.60`.  **Stage B fails both guardrails;
      seeds 1/2 not run.**  Diagnosis: new h flags 40% of early from-scratch
      actor states negative and the gate closes 40%, suppressing progress.
      Record: `STAGE_B_H_DATA_TARGET.md`.
- [x] Decide next controlled nav experiment.  Audit shows the Stage-B h did
      not learn the future-min target (corr 0.079 vs Ant 0.811; nav data only
      0.6% unsafe vs Ant 26.1%; no gradient-alignment loss; hdot term
      dominates).  The repair is an Ant-faithful h retrain: wall-free targets,
      SmoothL1(beta=1), 40 epochs/lr 1e-3, gradient alignment, no hdot, and
      risk-enriched data.  Do not sweep `task_gate_high`.
- [x] Gate-off control (same Stage-B h, `task_gate_high=0.0`) 100ep eval.
      Result: `safe_episode=0.03`, `official_reach=0.135`,
      `official_success=0.0763`, `episode_return=7.34`, versus h-only
      `0.620/0.576/0.5325/69.78`.  The Stage-B h features are harmful even
      without the gate.
- [x] Fix the repaired-h NaN.  Two causes: second-order `jacfwd` gradient
      alignment (replaced by central differences) and a `0/0`
      `jnp.linalg.norm` derivative at exactly-zero gradients in the cosine
      (replaced by a differentiable `sqrt(pred_sq*target_sq + 1e-12)`).
      Added global gradient clipping `max_grad_norm=1.0`.
- [x] Full-data 1-epoch stability check after the fix: val target corr
      `0.685`, safe-row corr `0.684`, gradient cosine `0.724`, BA `0.914`,
      AUROC `0.980`.
- [x] Repaired h training (seed 0, 40 epochs) completed from scratch on
      `data_stageb_repair` (best epoch 26).  Val: target corr `0.8004`,
      safe-row corr `0.7683`, gradient cosine `0.9111`, BA `0.9936`, AUROC
      `0.9997`.  Test: target corr `0.7940`, safe-row corr `0.7364`,
      gradient cosine `0.9108`, BA `0.9926`, AUROC `0.9997`.  All offline
      guardrails pass.
- [x] Gate calibrated from the repaired h validation split:
      `threshold_raw=0.061149`, `low=0.0`, `high=0.267270`
      (`max(0.02, q10_safe)`).  On validation the gate is fully closed on
      `100%` of unsafe rows, mean gate `0.95` on safe rows, and only `4.6%`
      of safe rows are below 0.5.  Record:
      `outputs/v6/independent/h_stageb_repair/gate_calibration.json`.
- [x] Stage-B repair actor seed 0 trained from scratch with the repaired h
      and calibrated gate, then formally evaluated (100ep deterministic,
      eval seed 10000).  Result: `safe_episode=0.840`,
      `official_safety=0.9738`, `official_success=0.4275`,
      `official_reach=0.4350`, `episode_return=60.91`.  Safety passes
      clearly vs h-only seed 0 (`0.620/0.9287`); task retention is partial
      (success 80.3%, return 87.3%; success just below the 0.45
      matched-seed floor but above the formal h-only 3-seed mean 0.407).
      This is a substantial recovery from the old Stage B
      (`0.000/0.4250/0.0375/-23.60`) and the first nav gate arm without a
      task collapse.  Record: `STAGE_B_H_DATA_TARGET.md`.
- [x] Multi-seed repaired Stage B (seeds 1/2), pre-registered protocol:
      16,384 steps, same frozen h/gate/config, 100ep deterministic eval seed
      10000.  **3-seed pass** (2026-10-25): `safe_episode=0.693 +/- 0.168`,
      `official_safety=0.9417 +/- 0.0378`, `official_success=0.4100 +/-
      0.0583`, `official_reach=0.4296 +/- 0.0695`, `episode_return=66.85 +/-
      5.26`.  All four aggregate guardrails and the no-catastrophic-seed
      rule pass.  Record: `STAGE_B_REPAIR_MULTISEED.md`.
      Caveat: `n=3`; seed-level paired differences are not statistically
      separated from zero (safe `p=0.216`, success `p=0.957`).  Safety gain
      is seed-0/2 driven; seed 1 is nearly neutral.
- [x] Do not sweep `task_gate_high`; it was not swept.
- [x] Read-only h2/actor2 diagnostic on the fixed final actor1 checkpoints,
      3 seeds x 100 episodes, eval seed 10000.  **Complete 2026-10-25.**
      Result: 155/155 new collision onsets have `h_cur < 0` and
      `gate_next = 0`; no collision starts in an h1-safe/gate-open state.
      Impact-state label: **actor/action-limited**.
      However, h1 was trained on h-only seed0 and Stage-A seed0 rollouts, not
      final actor1.  The risk-region evidence shows the resulting temporal
      gap: future-collision MAE `0.394`; 17-32-step h mean `+0.242` vs target
      `-0.222`; pooled MAE `0.0976` is dominated by far-safe rows and is not
      sufficient.  Overall diagnosis: **mixed** (temporal h limitation before
      impact + actor/action limitation at impact).  Record:
      `outputs/v6/independent/nav/H2_ACTOR2_DIAGNOSTIC.md`.
- [x] Stage C0a (h2-only temporal-calibration screen, seed 0): **FAILED**.
      h2 17-32-step risk-region MAE `0.826` vs h1 `0.709` (+16.5% worse);
      9-16-step MAE `0.567` vs `0.442`; future-collision MAE `0.478` vs
      `0.375`.  Global MAE improved (`0.084` vs `0.092`) but the risk region
      degraded.  The nominal +11-step lead is a gate-threshold artifact;
      threshold-free last `h > 0` is unchanged at 8 steps.  Record:
      `outputs/v6/independent/nav/STAGE_C0_H2_TEMPORAL.md`.
- [x] Stage C0b seeds 1/2: **not run** (C0a failed).  No h-only retraining
      variant on the same axis.
- [x] Original Stage C1 actor2: **not authorized / not trained**.
- [x] Post-hoc seed-0 h2+actor1 closed-loop h-swap: slightly better than h1
      (safe 0.90 vs 0.84, official safety 0.985 vs 0.974, success 0.439 vs
      0.428, return 61.40 vs 60.91), but one seed and not significant.  It
      does not change C0a's failure or authorize the original C1.  Record:
      `outputs/v6/independent/nav_c0_h2_actor1_swap/README.md`.
- [x] **C1a seed-0 actor2 pilot** (new, explicitly approved): **FAILED all
      five checks**.  100ep: safe `0.510` vs `0.840`, official safety
      `0.89375` vs `0.97375`, success `0.33125` vs `0.42750`, reach
      `0.36250` vs `0.43500`, return `52.22` vs `60.91`.  Collisions rose
      `0.16 -> 0.49`.  The gate was almost always open
      (`task_gate_mean≈0.99`), so this is not gate suppression.  The
      from-scratch h2/actor2 branch is stopped; C1b is not authorized.
      Record:
      `outputs/v6/independent/nav/STAGE_C1A_H2_ACTOR2_PILOT.md`.
- [x] **C1w actor1+h2 4k warm-start** (diagnostic, no h1 control): **failed**
      the C1a guardrails.  100ep: safe `0.840`, official safety `0.97625`,
      success `0.20250`, reach `0.20625`, return `28.85`; it matches actor1 on
      safe episodes, stays below the no-training h2 swap on safety, and
      collapses the task.  `promising_adaptation = false`.  The
      pre-registered staged rule was applied: the matched h1-continued control
      was **not run**, and the h2 warm-start branch is stopped.  Record:
      `outputs/v6/independent/nav/STAGE_C1W_H2_ACTOR1_4K.md`.
- [x] **C1s h2 seed-0 + actor1 seed replication** (evaluation only):
      **`consistent_positive`**.  Seed 1: safe `0.56` vs `0.51`, official
      safety `0.91250` vs `0.90000`, return `74.59` vs `70.92`.  Seed 2:
      safe `0.74` vs `0.73`, official safety `0.95625` vs `0.95125`, return
      `64.97` vs `68.74`.  Mean `Δsafe +0.030` (exactly the pre-registered
      `+0.03` boundary), `Δofficial_safety +0.00875`, `Δsuccess +0.01375`,
      `Δreach +0.00813`, `Δreturn -0.05`; all eight checks hold.  Thin,
      seed-1-driven, off-policy, one h2 checkpoint; not a main claim and not
      a from-scratch result.  Record:
      `outputs/v6/independent/nav/STAGE_C1S_H2_ACTOR_SEED_SWAP.md`.
- [ ] **C0b matched h2-seed study: NOT AUTHORIZED / stopped.**  C1s only
      tests one h2 checkpoint, but its signal is exactly at the `+0.03`
      boundary and seed-1-driven, so it is not a robust enough benefit to
      scale.  User chose to stop after C1s (2026-10-26); do not launch C0b.
- [ ] **T0w task-channel line: strict safety-first rule missed; task-heavy
      trade-off documented.**
      Warm-start 4k (V_T quality gate passed: val EV `0.2270`, test Pearson
      `0.4991`) reached 100ep safe `0.890`, official safety `0.98250`,
      success `0.67875`, reach `0.685`, return `161.88`.  Matched no-V_T 4k
      control: safe `0.930`, official safety `0.99000`, success `0.380`,
      return `69.67`.  `control_valid = true`, `task_gain = true`
      (success `+0.29875`, return `+92.20`), but `safety_not_worse = false`
      (safe `0.890 < 0.900`, one episode) -> `vt_channel_supported = false`
      under the pre-registered safety-first rule.  Task-first reading: a
      task-heavy Pareto point with a large task gain and a small,
      statistically unresolved safety discount; not a safety-non-inferiority
      pass.  One seed, diagnostic only.  Record:
      `outputs/v6/independent/nav/STAGE_T0W_VT_WARMSTART.md`.
- [ ] Optional (task-first): 3-seed from-scratch `h1 + gate + V_T`
      replication.  Not authorized yet; requires a new pre-registration.
      The strict safety-first line remains stopped.
- [ ] Safety coupling / action conditioning only if work resumes and is
      approved; action conditioning/filtering requires explicit approval.
      The impact-time failure is actor/action-limited.
- [ ] Do not combine Stage A, Stage B, Stage C, cost-schedule, team-risk, and
      task channel in one run.

## Do not do

- Do not copy Ant h weights or 0.05 threshold into navigation.
- Do not add QP/action projection/task leakage.
- Do not extend nav claims beyond the recorded 3-seed pass (`n=3`, no
  policy-level significance, task below V5 full/CBF-QP/GCBF+).  The Ant 10M
  three-seed claim is supported by formal 100ep results.
