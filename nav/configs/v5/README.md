# V5 configs

## Official learned-h

- `learned_h_ant.yaml` — Ant cost+survival safety-head training; checkpoint then
  consumed by `train_ant_qp_free.py --safety-h-checkpoint`.

## Fixed-h baselines / ablations

- `qp_free_double_integrator.yaml` — navigation full V5 with frozen official
  GCBF+ h;
- `qp_free_double_integrator_h_input.yaml` — h input only;
- `qp_free_double_integrator_no_h.yaml` — no h features/losses;
- `qp_free_double_integrator_task_only.yaml` — task-only control;
- `qp_free_double_integrator_safety.yaml` — safety-focused fixed-h run;
- `navigation_gcbfplus_double_integrator.reference.yaml` — official reference.

Official learned-h runs must use `nav/v5/`; navigation learned graph h is not
implemented yet. See `nav/v5/README.md`.
