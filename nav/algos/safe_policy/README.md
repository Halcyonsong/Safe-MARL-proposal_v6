# Safe policy (V5)

The QP-free policy is reusable by the official learned-h chain and by fixed-h
baselines/ablations.

## Official learned-h chain

- `nav/v5/safety/`: learned `h_cost` + `h_surv` heads, losses, dataset,
  checkpoint/provider;
- `nav/scripts/train_learned_h.py`: offline h training;
- `nav/scripts/train_ant_qp_free.py --safety-h-checkpoint ...`: stage-2 policy
  with frozen learned h.

## Fixed-h baseline / ablation

- `fixed_h.py`: read-only official GCBF+ navigation `cbf.pkl`; returns h and
  local ∇h;
- `ant_h.py`: analytic official-threshold Ant margin;
- `qp_free.py`: task head, bounded correction head, feasibility/value/cost
  heads, h/∇h inputs, forward-invariance / gradient-direction / barrier losses;
- `training.py`: official-env rollout, GAE, PPO update, deterministic eval,
  checkpoint I/O;
- `nav/scripts/train_qp_free.py`: fixed-h navigation baseline;
- `nav/scripts/train_ant_qp_free.py` without a learned checkpoint: fixed-h Ant
  baseline.

No QP solver, QP teacher, QP labels, expert `u_ref`, or BC target is used in the
official method. The official `actor.pkl` is not loaded.

See `nav/v5/README.md` for the archived V5 pipeline.
