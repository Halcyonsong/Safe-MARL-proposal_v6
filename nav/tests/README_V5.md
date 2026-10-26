# V5 tests

Implemented:

- `test_gcbfplus_env.py`: official environment adapter reset/step and masks;
- `test_fixed_h.py`: official h checkpoint load + h/∇h finite;
- `test_qp_free.py`: task/correction/feasibility heads, action bounds,
  stochastic interface, safety losses;
- `test_v5_integration.py`: official env + fixed h + QP-free policy + losses,
  plus GAE terminal bootstrap;
- `test_v5_boundaries.py`: main path contains no QP import or expert `u_ref`.

Still to add:

1. fixed-seed trajectory parity against the official JAX rollout;
2. full 3-seed/100-episode evaluation regression;
3. h/∇h and correction/barrier ablations.
