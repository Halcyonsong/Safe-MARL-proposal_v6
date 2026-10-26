# V6 navigation h2/actor2 audit artifacts

Read-only audit of the fixed final repaired Stage-B actor1 checkpoints.  No
training, no checkpoint selection, no gate/cost/task-channel change.  The audit
reproduces the formal 100-episode `safe_episode`, `official_safety`,
`official_success`, `official_reach`, and `episode_return` for all three seeds
to `< 1e-9`.

Interpretation and decision:
[`../nav/H2_ACTOR2_DIAGNOSTIC.md`](../nav/H2_ACTOR2_DIAGNOSTIC.md).

## Files

- `seed{0,1,2}_100ep.json` — per-seed audit record: formal-eval verification,
  collision onsets, pre-collision h windows, calibration, episode rows.
- `seed{0,1,2}_100ep.npz` — per-row arrays: `h`, `h_next`, `gate_next`,
  `target`, `target_raw`, `margin`, `current_unsafe`, `horizon_safe`,
  `future_collision`, `future_unsafe`, `episode`, `step`, `agent`.
- `summary.json`, `summary.md` — pooled calibration/risk regions and the
  pre-registered **impact-state** classification.  The overall diagnosis is
  mixed: see the diagnostic document for the temporal/data-source
  reinterpretation.
- `manifest.json` — protocol, checkpoint/script/output SHA-256 hashes, result.
- `logs/seed{0,1,2}_100ep.log` — stdout/stderr from the audit runs.
- `smoke/` — early 2-episode smoke check, not evidence.

`current_unsafe` is the h-provider warning/unsafe label used by h training,
not the collision flag.  `future_collision` is the actual collision flag.

## Reproduce

```bash
python nav/scripts/diagnose_v6_nav_h2.py \
  --checkpoint outputs/v6/independent/nav_stageb_repair_16k/seed0/policy_final.pth \
  --h-checkpoint outputs/v6/independent/h_stageb_repair/v6_h_stageb_repair.pkl \
  --num-episodes 100 --seed 10000 --device cuda \
  --reference-eval outputs/v6/independent/nav_stageb_repair_16k/seed0/eval_100ep_final.json \
  --out outputs/v6/independent/nav_h2_audit/seed0_100ep.json --tag seed0_100ep
# repeat for seeds 1/2, then:
python nav/scripts/summarize_v6_nav_h2_audit.py \
  --dir outputs/v6/independent/nav_h2_audit
```
