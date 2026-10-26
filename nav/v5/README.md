# V5 official chain: learned h

**Status: the Ant learned-h checkpoint is trained and frozen; the official
stage-2 policy has been run for 3 seeds and evaluated on 100 fixed scenes.**
The current result is **not** independent V5: it still uses the MAPPO-Lag 2M
task prior.

The official V5 chain is now:

```text
1. collect safety data (official cost + terminal/survival labels)
2. train learned safety heads (shared trunk)
      h_cost: wall / floor / ceil / upright
      h_surv: terminal / survival risk
3. freeze the learned h checkpoint
4. train the QP-free task policy + cost/survival safety corrections
5. evaluate cost, survival, return on fixed scenes
```

The fixed/analytic h implementations are **baselines and ablations**, not the
official chain.

## Official vs baseline

| component | official V5 | baseline / ablation |
|---|---|---|
| Ant h | learned `h_cost` + `h_surv` | analytic `ant_h.py` |
| Navigation h | learned graph h (pending) | frozen official GCBF+ h |
| h training | cost/survival labels, barrier/invariance; no QP | none (fixed) |
| policy | QP-free task + safety corrections | same policy with fixed h |
| results | none yet | existing fixed-h Ant/navigation results |

## File map

```text
nav/v5/
├── README.md                  # this file
├── factory.py                 # build learned/fixed safety provider
└── safety/
    ├── types.py               # SafetyOutput / SafetyPrediction
    ├── learned.py             # shared trunk + cost/survival heads
    ├── fixed.py               # analytic Ant / official navigation baselines
    ├── losses.py              # cost margin, survival BCE, barrier/invariance
    ├── dataset.py             # offline safety dataset
    ├── train.py               # offline h training/evaluation
    └── checkpoint.py          # frozen provider + checkpoint I/O

nav/scripts/train_learned_h.py          # official stage-1 entry point
nav/scripts/run_v5_official_ant.sh      # stage-1 + stage-2 Ant entry point
nav/configs/v5/learned_h_ant.yaml       # official Ant learned-h config
nav/scripts/train_ant_qp_free.py        # stage-2 policy; pass --safety-h-checkpoint
```

The stage-2 Ant entry point accepts:

```bash
python scripts/train_ant_qp_free.py \
  --safety-h-checkpoint outputs/v5/official/ant/learned_h/learned_h.pt \
  ...
```

If `--safety-h-checkpoint` is omitted, the old analytic Ant h is used and the
run must be reported as a fixed-h baseline, not official V5.

## Learned safety output

`SafetyOutput` contains:

- `h_cost`: `[B, C]`, signed margins; negative means that cost component is
  unsafe;
- `survival_logit`: `[B]`, terminal-risk logit;
- `survival_risk = sigmoid(survival_logit)`;
- `h_cost_scalar = min(h_cost)`;
- `active_cost`.

`h_cost` is CBF-like. `survival_risk` is a **trajectory-level risk head**, not
a CBF value. Do not apply the same forward-invariance claim to it.

## Loss boundary

The h loss contains only:

```text
cost safe/unsafe margin
survival BCE / calibration
barrier
forward invariance (cost margin)
```

There is **no task return** and **no QP target/expert u_ref** in the h loss.
Return is handled by the task branch in stage 2.

## Data format

`SafetyDataset` consumes `.npz`:

```text
states       [N, state_dim]      (Ant: 9-D geometry)
cost_labels  [N, C]              (wall/floor/ceil/upright)
terminal     [N]                 (1 = episode terminated early)
next_states  [N, state_dim]      optional, for invariance/direction
```

No return field is needed or used.

## Current gaps

- collect a multi-policy Ant safety dataset;
- train and validate the learned h checkpoint (false alarm / miss / survival
  calibration);
- add a navigation learned graph-h adapter (the current navigation path still
  uses the frozen official GCBF+ checkpoint);
- stage 2 currently consumes the four cost margins through the Ant policy;
  `survival_risk` is exposed by the provider/env and its separate correction
  head is the next policy-side step;
- wire the frozen learned h into the official stage-2 trainer and run 3 seeds.
