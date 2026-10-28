# V6 TODO

Updated: 2026-10-27.

Canonical status and artifact map: [`STATUS.md`](STATUS.md).

Current state: Ant 10M 3-seed pass; navigation repaired Stage-B 3-seed pass;
navigation P0/P1 attribution complete (actor h input is load-bearing at 16k,
gate ineffective without it).  The unchecked items below are follow-ups and
explicit non-goals, not blockers for the promoted claims.

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
- [x] **Ant 10M from-scratch MAPPO-Lag external baseline — COMPLETE,
      2026-10-27 19:17.**  Fresh experiment
      `v6_lag_10m_seed0_restart_20261027`, seed 0, exactly 10M steps,
      unchanged upstream recipe, no checkpoint selection.  Frozen evaluation
      at 100 deterministic + 100 stochastic episodes (eval seeds
      `20000..20099`): deterministic `safe=0.97`, `forward=3011.02`,
      `return=3603.84`; stochastic `safe=0.66`, `forward=2305.82`,
      `return=2624.36`.  Matched V6 seed-0: det `safe=0.97`,
      `forward=3723.09`; stoch `safe=0.78`, `forward=2923.73`.  This is a
      single-seed external screen, not a main-table row; no 3-seed expansion
      is launched.  Record:
      `independent/ant/STAGE_ANT_MAPPO_LAG_10M.md`.
- [x] **Navigation 28k budget extension — A-only COMPLETED, negative task
      result (2026-10-28).**  V6 repaired-h + progress-gate, 3 seeds, from
      scratch, 28,672 steps.  Mean 16k/24k/28k success
      `0.4100/0.2704/0.2767`; mean safe `0.6933/0.7000/0.7800`; mean
      official `0.9417/0.9479/0.9604`; mean return `66.85/59.97/63.45`.
      28k fails mean success (`<0.36`) and worst-seed success
      (`0.19875<0.20`), while safety/return/spread pass.  **Not promoted;
      16k remains the promoted row.**  No extra 28k seeds, no endpoint
      switch, no 24k promotion.  Pre-registration and result:
      `independent/nav/STAGE_28K_BUDGET_EXTENSION.md`; table:
      `independent/nav/nav_3seed_16k_24k_28k_table.json`.
- [x] **MAPPO fixed-penalty 16k sweep — completed 2026-10-28.**
      `penalty=1` 3-seed mean `official 0.4096`, `safe 0.0100`, `success
      0.1325`, `return 59.36`; `penalty=100` mean `0.6737/0.0733/0.0158/
      -59.48`.  Neither is close to V6; penalty 1 remains unsafe, penalty
      100 trades task away.  `penalty=0/10` already existed.  Fixed penalty
      only; MAPPO-Lag `cost_limit` sweep remains deferred.  Record:
      `independent/nav/PENDING_28K_EXPERIMENTS.md` and `STATUS.md`.
- [x] **V6 density stress test — first block completed 2026-10-28.**
      4 conditions (8/8, 4 obstacles, 16 obstacles, 16 agents) x 3 policies
      (A28, A16, H0), seed 0, 20 episodes, scenes `30000..30019`.  A16 keeps
      substantially more task than A28 in every condition (`success
      0.34-0.39` vs `0.19-0.22`) with comparable/better safety in 8/8,
      4 obstacles and 16 agents; A28 is safer only at 16 obstacles
      (`0.60` vs `0.50`).  H0 has A16-like task but much lower safety at 16
      obstacles/16 agents (`safe 0.25/0.05`).  **No A28 density-robustness
      benefit; report the task boundary.**  These are descriptive 20-episode
      screens; no 100ep x 3-seed confirmatory runs.  12-agent/12-obstacle
      rows deferred.  Record: `independent/nav/DENSITY_STRESS_TEST.md`,
      `independent/nav_density/seed0/summary.json`.
- [x] **P0: original-h + broad-gate 3-seed completion (2026-10-27).**
      Evaluated `nav_h_gate_16k` seeds 1/2 at 100 deterministic episodes,
      eval seed 10000.  3-seed mean: `safe=0.6433`, `official_safety=0.9325`,
      `success=0.4263`, `reach=0.4525`, `return=70.33`.  This isolated the
      broad-gate effect on the original h: P0 - h-only = `+19.3 pp`; A - P0 =
      `+5.0 pp` with a task cost and non-uniform per-seed deltas
      (`+14/+2/-1 pp`).  The later P1 factorial revised the repaired-h
      attribution (B - H0 `+18.3 pp` h retrain, A - B `+6.0 pp` gate).  Docs
      updated: `independent/nav/BASELINE_COMPARISON.md`, `STATUS.md`.
- [x] **P1: repaired-h 2x2 factorial (completed 2026-10-27).**  B = h input
      on + gate off; C = h input off + gate on; D = h input off + gate off.
      16,384 steps, seeds 0/1/2, fixed penalty 10, repaired h, 100
      deterministic episodes, eval seed 10000.  All nine runs trained by
      14:47 and all nine formal evaluations finished by 17:08.  Means:
      A `0.6933/0.9417/0.4100/66.85`; B `0.6333/0.9267/0.3229/55.67`;
      C `0.0200/0.4987/0.0367/-6.63`; D `0.0333/0.5479/0.0367/7.21`
      (safe / official safety / success / return).  Clean contrasts:
      B - H0 `+18.3 pp` safe but `-8.4 pp` success; A - B `+6.0 pp` safe /
      `+8.7 pp` success; A - C / B - D `+67.3/+60.0 pp` safe; C - D
      `-1.3 pp` safe.  Reading: actor h input is load-bearing at 16k from
      scratch (C/D collapse; D is consistent with the no-h fixed-penalty baseline), the
      gate is ineffective without it, and within the repaired-h 2x2 only A
      combines safety with retained task.  `n=3`, descriptive only.  Artifacts:
      `independent/nav_repair_factorial/aggregate_100ep.{json,md}`;
      record: `independent/nav/P1_REPAIRED_H_FACTORIAL.md`.  Docs updated:
      `BASELINE_COMPARISON.md`, `STATUS.md`, `V6_COMPLETE_RECORD.md`.
      One server restart interrupted B seed 2 evaluation; it was rerun with
      the identical checkpoint/config/seed and no completed result changed.
- [x] **Matched 16k collision-aware navigation baselines (completed
      2026-10-27):** MAPPO + fixed collision penalty 10 and MAPPO-Lag +
      official collision cost, seeds 0/1/2, 100 deterministic episodes at
      eval seed 10000.  Results: fixed penalty `safe=0.033,
      official_safety=0.519, success=0.014, return=-30.35`; Lag 16k
      `safe=0.000, official_safety=0.322, success=0.086, return=39.55`;
      V6 Stage-B `0.693/0.9417/0.410/66.85`.  Artifacts:
      `independent/nav_baselines_16k/`; updated
      `independent/nav/BASELINE_COMPARISON.md`.
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
- [x] Broad-gate seeds 1/2 completed (P0, 2026-10-27): 3-seed
      `0.6433/0.9325/0.4263/70.33`; that gate design remains archived
      (safety/task tradeoff), not promoted.
- [x] Formal 16k h-off 3-seed baseline (P1 D, 2026-10-27): `safe 0.0333`,
      `official_safety 0.5479`, `success 0.0367`, `return 7.21`; confirms the
      no-h collapse at the matched budget.
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
- [x] **T0w-28k task warm-start screen (W16; completed 2026-10-28,
      task-first trade-off).**
      Primary arm: from the same 28k run's 16k checkpoint,
      continue 12,288 steps with the frozen V_T task channel to 28,672,
      saving/evaluating both the **24k and 28k** checkpoints.  Compare each
      to the from-scratch 24k/28k checkpoints (same seed, same 16k prefix,
      same total budget); no separate C16/C24 control is required for this
      procedure-level comparison.  The existing T0w 16k+4k no-V_T control
      remains the channel-attribution evidence; add a C24 (4k) only if a
      strict "V_T alone causes the gain" claim is needed.  Optional short
      arm: 24k -> 28k (4,096 steps) with a V_T retrained on 24k rollouts,
      only if the 16k -> 28k task gain fades at 28k.  V_T: train on 16k base
      rollouts for the primary arm; 24k rollouts for the optional short arm.
      Seed-0 screen first, 3 seeds only if the task gain holds with safety
      not worse.  100ep eval `10000..10099`; V_T data `30000..30099`.
      **Queue after the 3-seed 16/24/28 formal table and before the MAPPO
      sweep / density block.**  This is a separate warm-start arm, not the
      from-scratch 28k budget claim.  Seed-0 100ep passed: 24k
      `0.93/0.98875/0.57/152.42`; 28k `0.83/0.9725/0.62125/237.36`
      (safe/official/success/return), versus control 28k
      `0.86/0.975/0.19875/55.52`.  Seeds 1/2 used per-seed V_T; the 3-seed
      result is a task-first trade-off: W16 28k mean `safe 0.6633`,
      `official 0.9329`, `success 0.6175`, `return 198.57`; the 24k
      diagnostic mean is `0.7233/0.9504/0.5958/143.73`.  Rule 7.1 fails on
      28k mean safe/official and on the safe range (0.27; 24k range 0.38),
      while task/return/worst-seed bounds pass.  **Keep 16k promoted; do not
      switch to 28k/24k post hoc.**  A safety-regularized task channel or a
      5-seed confirmatory pre-registration is a separate proposal.  MAPPO
      was preempted for this block and restarted after.
- [x] **Ant V_T failure diagnosis (completed 2026-10-28).**  The 1-D
      `A_T` warm-start did not improve task (forward `3960.83` vs control
      `4014.03`, `-1.33%`).  The V_T predictor passed quality (EV `0.4838`,
      Pearson `0.6597`), so the interface was the primary suspect.  The
      fixed `[A_T, unit_norm(grad_s V_T)]` 250k pilot is also null on task:
      deterministic forward `3816.50` vs control `3811.20` (`+0.14%`),
      return `4458.09` vs `4449.88` (`+0.18%`); safety improved (`0.99` vs
      `0.89`, cost `0.00094` vs `0.00661`), stochastic forward `2968.53` vs
      `2856.21`, safe `0.78` vs `0.62`.  `vt_grad_supported=false`.
      **Do not expand this arm to 3 seeds and do not claim an Ant task
      channel.**  Remaining candidates (longer warm-start / V_T refresh,
      V_T-vs-critic residual, actor capacity, task-reward scale) require a
      new pre-registration; one cause at a time; no test-set tuning.  The
      nav W16 gate is unaffected and remains the task-feasibility test.
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
