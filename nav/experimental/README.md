# Experimental / fixed-h components

These are **not** the official learned-h V5 chain. Keep them for controlled
baselines and historical comparisons.

## Fixed-h Ant

- `algos/safe_policy/ant_h.py` — analytic official-threshold Ant margin;
- `scripts/train_ant_qp_free.py` without `--safety-h-checkpoint`;
- `scripts/evaluate_ant_qp_free.py` without a learned checkpoint;
- `outputs/v5/ant/` — fixed-h and early learned-h interface runs.

## Fixed-h navigation

- `algos/safe_policy/fixed_h.py` — frozen official GCBF+ checkpoint;
- `scripts/train_qp_free.py` / `scripts/evaluate_qp_free.py`;
- `outputs/v5/seeds_*`, `outputs/v5/h_ablation/` — fixed-h results.

## Official path

See `nav/v5/README.md` for the archived V5 pipeline.

Rule of thumb: if a run does not pass `--safety-h-checkpoint` (or the
navigation learned graph equivalent), it is a baseline, not official V5.
