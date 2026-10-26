# Navigation / Ant legacy code workspace

Status: **V5 is retained as a legacy code/design reference only.  It is not a
V6 runtime dependency.**  The standalone V6 protocol is documented in
[`README_V6.md`](README_V6.md); V6 uses no V5 weights, data, checkpoints, or
official h/QP artifacts.  Commands below describe historical V5 code paths and
must not be used for the independent V6 main path.

Official chain:

```text
nav/v5/safety/      learned cost + survival safety heads
nav/scripts/train_learned_h.py
        |
        v
frozen learned-h provider
        |
        v
nav/scripts/train_ant_qp_free.py --safety-h-checkpoint ...
```

- `nav/v5/README.md` — archived V5 package/file map;
- `nav/configs/v5/learned_h_ant.yaml` — Ant learned-h stage-1 config.

## Active, reusable components

- `v5/safety/`: learned h heads, losses, dataset, checkpoint/provider;
- `algos/safe_policy/`: QP-free policy and training utilities;
- `algos/stage2/`: official MAPPO task-policy adapter (warm start, not expert
  `u_ref`);
- `envs/gcbfplus/`: official-aligned DoubleIntegrator adapter;
- `envs/mamujoco/`: official Ant adapter; accepts a learned-h provider;
- `vendor/onpolicy/`: vendored official MAPPO implementation.

## Baseline / experimental

- `algos/safe_policy/fixed_h.py`: frozen official GCBF+ navigation h;
- `algos/safe_policy/ant_h.py`: analytic Ant h;
- `scripts/train_qp_free.py` / `scripts/evaluate_qp_free.py`: fixed-h
  navigation baseline;
- `scripts/train_ant_qp_free.py` without `--safety-h-checkpoint`: fixed-h Ant
  baseline;
- `outputs/v5/ant/`, `outputs/v5/seeds_*`, `outputs/v5/h_ablation/`: fixed-h
  results.

See `experimental/README.md` for the baseline boundary.

## V5 method boundary

- Official h: learned `h_cost` + `h_surv`; frozen before policy training.
- h loss: cost/survival/barrier/invariance only; no return.
- No QP solver, QP teacher, QP labels, expert `u_ref`, or BC/DAgger in official
  V5.
- Fixed h is an ablation, not the official chain.
- h is scenario-specific; no navigation-to-Ant h transfer.

## Archived, not part of V5

The old V4 learned-h/QP code, entry points, and checkpoints are not included
in this repository. Only the V5 legacy reference code under `nav/` is kept.

See `v5/README.md` for the archived V5 pipeline.
