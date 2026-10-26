# V6 Ant h acceptance (before h-only)

Checkpoint: `outputs/v6/independent/ant/h/ant_h.pkl`
Data: `outputs/v6/independent/ant/h_data`
Validation: `outputs/v6/independent/ant/h_validation/validation.json`,
`outputs/v6/independent/ant/h_validation_seed31000/validation.json`

## Held-out training split (forward-window labels)

| split | rows | unsafe rate | balanced acc. | unsafe recall | safe recall | false alarm | AUROC | grad cosine |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| val | 6189 | 0.336 | 0.9776 | 0.9745 | 0.9808 | 0.0128 | 0.9973 | 0.9301 |
| test | 5410 | 0.377 | 0.9733 | 0.9715 | 0.9751 | 0.0155 | 0.9963 | 0.9255 |

Rollout-only test rows (source 0) have unsafe recall 1.000 and false alarm
0.0018; the augmented test rows have unsafe recall 0.9746 and false alarm
0.0196.

## Independent fresh rollouts

Frozen h, no fitting on these rollouts. Each regime uses a different episode
seed; 50 episodes/regime for seed 30000, 30 for seed 31000.

| validation seed | rows | unsafe rows | balanced acc. | unsafe recall | false alarm | forward BA | grad cosine | reset alarm |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30000 | 105347 | 215 | 0.9974 | 0.9953 | 0.00046 | 0.9977 | 0.9717 | 0.000 |
| 31000 | 63692 | 214 | 0.9881 | 0.9766 | 0.00049 | 0.9883 | 0.9708 | 0.000 |

Action regimes: zero, random std 0.1, random std 0.7 (training-like), random
std 1.0. In the two unsafe regimes, the worst per-regime metrics were
BA 0.986, unsafe recall 0.979, false alarm 0.010 (seed 30000) and BA 0.980,
unsafe recall 0.960, false alarm 0.000 (seed 31000). The zero and small-random
regimes had no false alarms over 100k held-out steps at seed 30000.

## Wall-specific augmented test

On held-out augmented wall geometry (source != 0):

* wall unsafe recall = 0.981;
* wall safe false alarm = 0.111 overall, but all false alarms are within
  0.067 m of the wall threshold;
* false alarm is 0.018 for safe wall states above 0.05 m and 0 above 0.10 m;
* floor/ceil/upright recalls are 0.962/0.958/0.970.

## Gate

`validation.json` gates all pass:

* pooled instantaneous BA >= 0.95, unsafe recall >= 0.90, false alarm <= 0.05;
* forward BA/recall >= 0.90;
* gradient cosine >= 0.80;
* reset false alarm <= 0.05;
* every regime with >=10 unsafe states has BA/recall >= 0.85 and false alarm
  <= 0.10.

Conclusion: the frozen V6 Ant h is accurate enough in classification and
state-gradient direction to enter the 1M h-only pilot. This is a safety-h
validation only; it does not establish that the actor will learn the task.
