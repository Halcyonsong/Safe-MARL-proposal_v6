# V6 navigation h2/actor2 diagnostic audit

Read-only audit of the fixed final actor1 checkpoints.  No training, no checkpoint selection, no gate change.

The classification below is the pre-registered **impact-state** rule.  It does not test whether h1 warned early enough.  See H2_ACTOR2_DIAGNOSTIC.md for the temporal/data-source reinterpretation (overall diagnosis: mixed).

## Actor1 reproduction

| seed | safe_ep | official_safety | success | reach | return | collision ep | events |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.840 | 0.9738 | 0.4275 | 0.4350 | 60.91 | 16 | 21 |
| 1 | 0.510 | 0.9000 | 0.4575 | 0.4963 | 70.92 | 49 | 90 |
| 2 | 0.730 | 0.9513 | 0.3450 | 0.3575 | 68.74 | 27 | 44 |

## Collision states (new collision onsets)

| seed | events | h_cur med | h_cur >= high | h_cur > 0 | gate next open | gate next closed | h_next med |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 21 | -0.1833 | 0.000 | 0.000 | 0.000 | 1.000 | -0.1942 |
| 1 | 90 | -0.1960 | 0.000 | 0.000 | 0.000 | 1.000 | -0.2093 |
| 2 | 44 | -0.2053 | 0.000 | 0.000 | 0.000 | 1.000 | -0.2248 |

## h1 before a collision onset (state h, lag in steps)

| seed | lag 0 | lag 1 | lag 2 | lag 3 | lag 4 | lag 5 | lag5 >= high | lag5 > 0 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | -0.1827 | -0.1632 | -0.1365 | -0.0982 | -0.0523 | 0.0021 | 0.000 | 0.524 |
| 1 | -0.1976 | -0.1710 | -0.1330 | -0.0837 | -0.0255 | 0.0304 | 0.033 | 0.611 |
| 2 | -0.2053 | -0.1792 | -0.1430 | -0.1002 | -0.0515 | -0.0003 | 0.000 | 0.545 |

## h1 calibration on actor1 states

| source | target corr | safe target corr | target MAE | target RMSE | BA | AUROC | future-collision AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| offline repaired test | 0.7940 | 0.7364 | 0.1510 | 0.2456 | 0.9926 | 0.9997 | n/a |
| offline repaired val | 0.8004 | 0.7683 | 0.1333 | 0.2174 | 0.9936 | 0.9997 | n/a |
| actor1 states (pooled) | 0.7040 | 0.6968 | 0.0976 | 0.1573 | 0.9978 | 0.9999 | 0.9079 |

Target std: offline test 0.3783, actor1 pooled 0.1779.  A narrow actor1 target range attenuates correlation; target MAE/RMSE are the primary calibration comparison.

## Calibration in risk regions (pooled)

| region | rows | h mean | target mean | MAE | corr |
|---|---:|---:|---:|---:|---:|
| target < 0.5 | 25636 | 0.3236 | 0.1861 | 0.3089 | 0.2864 |
| h < gate high | 18720 | 0.0851 | 0.2850 | 0.2619 | 0.7646 |
| future collision | 6285 | 0.1582 | -0.2150 | 0.3944 | 0.0941 |
| current unsafe label | 2669 | -0.2136 | -0.2091 | 0.0527 | 0.4106 |

## Decision evidence

- classification: **actor/action-limited: h2 likely marginal**
- new collision onsets: 155
- fraction h_cur >= high: 0.000
- fraction gate_next open: 0.000
- fraction h_cur <= 0: 1.000
- fraction gate_next closed: 1.000
- target-corr drop vs offline test: 0.090
- safe-corr drop vs offline test: 0.040
- target MAE on actor1 states: 0.0976 (0.65x offline test MAE)
- range restriction (target std ratio): 0.47
- future-collision rows: 6285, h mean 0.1582, target mean -0.2150, MAE 0.3944
- pre-collision lag5 h >= high: 0.019
- pre-collision lag5 h > 0: 0.581

Rule: h-limited if >=50% of new collision onsets have h_cur>=high or gate_next open, and actor1-state calibration degrades (pooled target MAE >=0.20 vs offline test 0.151, or safe corr drop >=0.10); h-safe-labeled if the same high/gate-open collision pattern holds but calibration does not degrade (policy/target mismatch, h2 possible but action conditioning may be needed); actor/action-limited if >=50% have h_cur<=0 or gate_next closed without calibration degradation; else mixed.

