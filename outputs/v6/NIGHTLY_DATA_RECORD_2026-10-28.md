# V6 夜间实验数据记录（2026-10-28）

## 0.0 Metadata

| field | value |
|---|---|
| date | 2026-10-28 |
| nav python | /home/xiang/miniconda3/envs/safe_marl/bin/python |
| ant python | /home/xiang/miniconda3/envs/mappo_lag/bin/python |
| nav train/eval device | cuda |
| ant train/eval device | cpu |
| nav eval protocol | 100 deterministic episodes, eval seeds 10000..10099 |
| ant eval protocol | 100 deterministic + 100 stochastic episodes, eval seeds 20000..20099 |
| density eval protocol | 20 deterministic episodes, eval seeds 30000..30019 |

## 0. 运行清单

| run | steps | seeds | episodes | eval seeds | output |
|---|---:|---|---:|---|---|
| Nav A-only 16k | 16,384 | 0/1/2 | 100 | 10000..10099 | outputs/v6/independent/nav_stageb_repair_16k |
| Nav A-only 24k | 24,576 | 0/1/2 | 100 | 10000..10099 | outputs/v6/independent/nav_stageb_repair_28k |
| Nav A-only 28k | 28,672 | 0/1/2 | 100 | 10000..10099 | outputs/v6/independent/nav_stageb_repair_28k |
| Nav W16 task warm-start | 16,384 -> 28,672 | 0/1/2 | 100 | 10000..10099 | outputs/v6/independent/nav_w16_task_12k |
| Ant V_T gradient | 10,000,000 -> 10,250,000 | 0 | 100 det + 100 stoch | 20000..20099 | outputs/v6/independent/ant_vt_grad_250k |
| Ant no-V_T control | 10,000,000 -> 10,250,000 | 0 | 100 det + 100 stoch | 20000..20099 | outputs/v6/independent/ant_vt_grad_control_250k |
| MAPPO fixed penalty 1 | 16,384 | 0/1/2 | 100 | 10000..10099 | outputs/v6/independent/nav_baselines_16k/mappo_penalty1 |
| MAPPO fixed penalty 10 | 16,384 | 0/1/2 | 100 | 10000..10099 | outputs/v6/independent/nav_baselines_16k/mappo_same_reward |
| MAPPO fixed penalty 100 | 16,384 | 0/1/2 | 100 | 10000..10099 | outputs/v6/independent/nav_baselines_16k/mappo_penalty100 |
| Density screen | 20 episodes | 0 | 20 | 30000..30019 | outputs/v6/independent/nav_density/seed0 |

## 1. Nav A-only 16k/24k/28k

### 1.1 Per-seed

| budget | seed | safe_ep | collision_ep | official_safety | success | reach | reach_final | return | unsafe | cbf_violation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16k | 0 | 0.84000 | 0.16000 | 0.97375 | 0.42750 | 0.43500 | 0.03625 | 60.91 | 0.001279 | 0.044642 |
| 16k | 1 | 0.51000 | 0.49000 | 0.90000 | 0.45750 | 0.49625 | 0.05500 | 70.92 | 0.005737 | 0.082686 |
| 16k | 2 | 0.73000 | 0.27000 | 0.95125 | 0.34500 | 0.35750 | 0.05875 | 68.74 | 0.003613 | 0.047962 |
| 24k | 0 | 0.65000 | 0.35000 | 0.93250 | 0.36875 | 0.38000 | 0.05250 | 73.17 | 0.003618 | 0.060296 |
| 24k | 1 | 0.73000 | 0.27000 | 0.95500 | 0.25625 | 0.26375 | 0.04000 | 62.50 | 0.002939 | 0.044291 |
| 24k | 2 | 0.72000 | 0.28000 | 0.95625 | 0.18625 | 0.19500 | 0.01375 | 44.24 | 0.002881 | 0.060490 |
| 28k | 0 | 0.86000 | 0.14000 | 0.97500 | 0.19875 | 0.20625 | 0.03250 | 55.52 | 0.001357 | 0.034147 |
| 28k | 1 | 0.70000 | 0.30000 | 0.94500 | 0.40375 | 0.42000 | 0.07125 | 84.53 | 0.003418 | 0.059723 |
| 28k | 2 | 0.78000 | 0.22000 | 0.96125 | 0.22750 | 0.23750 | 0.04125 | 50.29 | 0.002739 | 0.038012 |

### 1.2 Aggregate

| budget | metric | mean | std | min | max | range |
|---|---|---:|---:|---:|---:|---:|
| 16k | safe_episode | 0.69333 | 0.16803 | 0.51000 | 0.84000 | 0.33000 |
| 16k | collision_episode | 0.30667 | 0.16803 | 0.16000 | 0.49000 | 0.33000 |
| 16k | official_safety | 0.94167 | 0.03780 | 0.90000 | 0.97375 | 0.07375 |
| 16k | official_success | 0.41000 | 0.05826 | 0.34500 | 0.45750 | 0.11250 |
| 16k | official_reach | 0.42958 | 0.06953 | 0.35750 | 0.49625 | 0.13875 |
| 16k | reach_final | 0.05000 | 0.01205 | 0.03625 | 0.05875 | 0.02250 |
| 16k | episode_return | 66.85 | 5.26 | 60.91 | 70.92 | 10.01 |
| 16k | unsafe | 0.003543 | 0.002230 | 0.001279 | 0.005737 | 0.004458 |
| 16k | cbf_violation | 0.058430 | 0.021072 | 0.044642 | 0.082686 | 0.038043 |
| 24k | safe_episode | 0.70000 | 0.04359 | 0.65000 | 0.73000 | 0.08000 |
| 24k | collision_episode | 0.30000 | 0.04359 | 0.27000 | 0.35000 | 0.08000 |
| 24k | official_safety | 0.94792 | 0.01337 | 0.93250 | 0.95625 | 0.02375 |
| 24k | official_success | 0.27042 | 0.09207 | 0.18625 | 0.36875 | 0.18250 |
| 24k | official_reach | 0.27958 | 0.09351 | 0.19500 | 0.38000 | 0.18500 |
| 24k | reach_final | 0.03542 | 0.01978 | 0.01375 | 0.05250 | 0.03875 |
| 24k | episode_return | 59.97 | 14.63 | 44.24 | 73.17 | 28.92 |
| 24k | unsafe | 0.003146 | 0.000410 | 0.002881 | 0.003618 | 0.000737 |
| 24k | cbf_violation | 0.055026 | 0.009297 | 0.044291 | 0.060490 | 0.016199 |
| 28k | safe_episode | 0.78000 | 0.08000 | 0.70000 | 0.86000 | 0.16000 |
| 28k | collision_episode | 0.22000 | 0.08000 | 0.14000 | 0.30000 | 0.16000 |
| 28k | official_safety | 0.96042 | 0.01502 | 0.94500 | 0.97500 | 0.03000 |
| 28k | official_success | 0.27667 | 0.11099 | 0.19875 | 0.40375 | 0.20500 |
| 28k | official_reach | 0.28792 | 0.11545 | 0.20625 | 0.42000 | 0.21375 |
| 28k | reach_final | 0.04833 | 0.02032 | 0.03250 | 0.07125 | 0.03875 |
| 28k | episode_return | 63.45 | 18.45 | 50.29 | 84.53 | 34.24 |
| 28k | unsafe | 0.002505 | 0.001050 | 0.001357 | 0.003418 | 0.002061 |
| 28k | cbf_violation | 0.043961 | 0.013787 | 0.034147 | 0.059723 | 0.025577 |

### 1.3 Paired deltas

| seed | contrast | d_safe | d_official | d_success | d_reach | d_return |
|---:|---|---:|---:|---:|---:|---:|
| 0 | 24k-16k | -0.19000 | -0.04125 | -0.05875 | -0.05500 | 12.26 |
| 0 | 28k-16k | 0.02000 | 0.00125 | -0.22875 | -0.22875 | -5.39 |
| 0 | 28k-24k | 0.21000 | 0.04250 | -0.17000 | -0.17375 | -17.65 |
| 1 | 24k-16k | 0.22000 | 0.05500 | -0.20125 | -0.23250 | -8.42 |
| 1 | 28k-16k | 0.19000 | 0.04500 | -0.05375 | -0.07625 | 13.61 |
| 1 | 28k-24k | -0.03000 | -0.01000 | 0.14750 | 0.15625 | 22.04 |
| 2 | 24k-16k | -0.01000 | 0.00500 | -0.15875 | -0.16250 | -24.49 |
| 2 | 28k-16k | 0.05000 | 0.01000 | -0.11750 | -0.12000 | -18.45 |
| 2 | 28k-24k | 0.06000 | 0.00500 | 0.04125 | 0.04250 | 6.05 |

### 1.4 20ep checkpoint monitors

| budget | seed | safe_ep | official_safety | success | return |
|---|---:|---:|---:|---:|---:|
| 16k | 0 | 0.90000 | 0.98125 | 0.48750 | 64.74 |
| 24k | 0 | 0.75000 | 0.95625 | 0.40625 | 84.96 |
| 28k | 0 | 0.90000 | 0.98750 | 0.25000 | 58.61 |
| 16k | 1 | 0.60000 | 0.94375 | 0.51250 | 78.72 |
| 24k | 1 | 0.70000 | 0.95000 | 0.27500 | 62.09 |
| 28k | 1 | 0.75000 | 0.96250 | 0.48125 | 95.01 |
| 16k | 2 | 0.80000 | 0.96250 | 0.36250 | 67.41 |
| 24k | 2 | 0.80000 | 0.96250 | 0.20625 | 45.76 |
| 28k | 2 | 0.75000 | 0.95000 | 0.23750 | 50.59 |

### 1.5 Last 4,096 training steps

| seed | step | reward_mean | unsafe_mean | task_gate_mean | h_neg_rate | entropy |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 28672 | 0.20690 | 0.004150 | 0.97809 | 0.007812 | 0.81992 |
| 1 | 28672 | 0.21482 | 0.005981 | 0.96467 | 0.014740 | 0.82376 |
| 2 | 28672 | 0.13280 | 0.010101 | 0.96365 | 0.017578 | 0.82070 |

### 1.6 Rule 7.1

| check | value | threshold | pass |
|---|---:|---:|---|
| mean_safe | 0.78000 | 0.69330 | true |
| mean_official | 0.96042 | 0.94170 | true |
| mean_success | 0.27667 | 0.36000 | false |
| mean_return | 63.44775 | 55.00000 | true |
| worst_success | 0.19875 | 0.20000 | false |
| worst_safe | 0.70000 | 0.30000 | true |
| safe_range | 0.16000 | 0.20000 | true |
| worst_safe_spread | 0.70000 | 0.55000 | true |
| full_pass |  |  | false |

## 2. Nav W16 task warm-start

### 2.0 V_T quality

| seed | val_EV | val_pearson | val_RMSE | test_EV | test_pearson | test_RMSE |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.227009 | 0.486044 | 14.927160 | 0.244591 | 0.499138 | 15.566322 |
| 1 | 0.161035 | 0.423226 | 14.963327 | 0.215367 | 0.467664 | 15.237938 |
| 2 | 0.293987 | 0.552379 | 17.703530 | 0.179406 | 0.429041 | 18.180140 |

### 2.1 Per-seed

| arm | seed | safe_ep | collision_ep | official_safety | success | reach | reach_final | return | unsafe | cbf_violation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| W16-24k | 0 | 0.93000 | 0.07000 | 0.98875 | 0.57000 | 0.57250 | 0.21000 | 152.42 | 0.000806 | 0.034139 |
| W16-24k | 1 | 0.55000 | 0.45000 | 0.91250 | 0.63250 | 0.68250 | 0.19000 | 156.31 | 0.005669 | 0.075176 |
| W16-24k | 2 | 0.69000 | 0.31000 | 0.95000 | 0.58500 | 0.60750 | 0.13125 | 122.47 | 0.003247 | 0.065553 |
| W16-28k | 0 | 0.83000 | 0.17000 | 0.97250 | 0.62125 | 0.63625 | 0.33875 | 237.36 | 0.002187 | 0.049309 |
| W16-28k | 1 | 0.56000 | 0.44000 | 0.90750 | 0.67375 | 0.72750 | 0.30125 | 217.40 | 0.004741 | 0.068623 |
| W16-28k | 2 | 0.60000 | 0.40000 | 0.91875 | 0.55750 | 0.59375 | 0.16750 | 140.95 | 0.004360 | 0.065562 |
| scratch-24k | 0 | 0.65000 | 0.35000 | 0.93250 | 0.36875 | 0.38000 | 0.05250 | 73.17 | 0.003618 | 0.060296 |
| scratch-24k | 1 | 0.73000 | 0.27000 | 0.95500 | 0.25625 | 0.26375 | 0.04000 | 62.50 | 0.002939 | 0.044291 |
| scratch-24k | 2 | 0.72000 | 0.28000 | 0.95625 | 0.18625 | 0.19500 | 0.01375 | 44.24 | 0.002881 | 0.060490 |
| scratch-28k | 0 | 0.86000 | 0.14000 | 0.97500 | 0.19875 | 0.20625 | 0.03250 | 55.52 | 0.001357 | 0.034147 |
| scratch-28k | 1 | 0.70000 | 0.30000 | 0.94500 | 0.40375 | 0.42000 | 0.07125 | 84.53 | 0.003418 | 0.059723 |
| scratch-28k | 2 | 0.78000 | 0.22000 | 0.96125 | 0.22750 | 0.23750 | 0.04125 | 50.29 | 0.002739 | 0.038012 |

### 2.2 Aggregate

| arm | metric | mean | std | min | max | range |
|---|---|---:|---:|---:|---:|---:|
| W16-24k | safe_episode | 0.72333 | 0.19218 | 0.55000 | 0.93000 | 0.38000 |
| W16-24k | collision_episode | 0.27667 | 0.19218 | 0.07000 | 0.45000 | 0.38000 |
| W16-24k | official_safety | 0.95042 | 0.03813 | 0.91250 | 0.98875 | 0.07625 |
| W16-24k | official_success | 0.59583 | 0.03263 | 0.57000 | 0.63250 | 0.06250 |
| W16-24k | official_reach | 0.62083 | 0.05620 | 0.57250 | 0.68250 | 0.11000 |
| W16-24k | reach_final | 0.17708 | 0.04093 | 0.13125 | 0.21000 | 0.07875 |
| W16-24k | episode_return | 143.73 | 18.52 | 122.47 | 156.31 | 33.84 |
| W16-24k | unsafe | 0.003241 | 0.002432 | 0.000806 | 0.005669 | 0.004863 |
| W16-24k | cbf_violation | 0.058290 | 0.021461 | 0.034139 | 0.075176 | 0.041037 |
| W16-28k | safe_episode | 0.66333 | 0.14572 | 0.56000 | 0.83000 | 0.27000 |
| W16-28k | collision_episode | 0.33667 | 0.14572 | 0.17000 | 0.44000 | 0.27000 |
| W16-28k | official_safety | 0.93292 | 0.03474 | 0.90750 | 0.97250 | 0.06500 |
| W16-28k | official_success | 0.61750 | 0.05822 | 0.55750 | 0.67375 | 0.11625 |
| W16-28k | official_reach | 0.65250 | 0.06834 | 0.59375 | 0.72750 | 0.13375 |
| W16-28k | reach_final | 0.26917 | 0.09002 | 0.16750 | 0.33875 | 0.17125 |
| W16-28k | episode_return | 198.57 | 50.89 | 140.95 | 237.36 | 96.41 |
| W16-28k | unsafe | 0.003763 | 0.001378 | 0.002187 | 0.004741 | 0.002554 |
| W16-28k | cbf_violation | 0.061165 | 0.010381 | 0.049309 | 0.068623 | 0.019314 |
| scratch-24k | safe_episode | 0.70000 | 0.04359 | 0.65000 | 0.73000 | 0.08000 |
| scratch-24k | collision_episode | 0.30000 | 0.04359 | 0.27000 | 0.35000 | 0.08000 |
| scratch-24k | official_safety | 0.94792 | 0.01337 | 0.93250 | 0.95625 | 0.02375 |
| scratch-24k | official_success | 0.27042 | 0.09207 | 0.18625 | 0.36875 | 0.18250 |
| scratch-24k | official_reach | 0.27958 | 0.09351 | 0.19500 | 0.38000 | 0.18500 |
| scratch-24k | reach_final | 0.03542 | 0.01978 | 0.01375 | 0.05250 | 0.03875 |
| scratch-24k | episode_return | 59.97 | 14.63 | 44.24 | 73.17 | 28.92 |
| scratch-24k | unsafe | 0.003146 | 0.000410 | 0.002881 | 0.003618 | 0.000737 |
| scratch-24k | cbf_violation | 0.055026 | 0.009297 | 0.044291 | 0.060490 | 0.016199 |
| scratch-28k | safe_episode | 0.78000 | 0.08000 | 0.70000 | 0.86000 | 0.16000 |
| scratch-28k | collision_episode | 0.22000 | 0.08000 | 0.14000 | 0.30000 | 0.16000 |
| scratch-28k | official_safety | 0.96042 | 0.01502 | 0.94500 | 0.97500 | 0.03000 |
| scratch-28k | official_success | 0.27667 | 0.11099 | 0.19875 | 0.40375 | 0.20500 |
| scratch-28k | official_reach | 0.28792 | 0.11545 | 0.20625 | 0.42000 | 0.21375 |
| scratch-28k | reach_final | 0.04833 | 0.02032 | 0.03250 | 0.07125 | 0.03875 |
| scratch-28k | episode_return | 63.45 | 18.45 | 50.29 | 84.53 | 34.24 |
| scratch-28k | unsafe | 0.002505 | 0.001050 | 0.001357 | 0.003418 | 0.002061 |
| scratch-28k | cbf_violation | 0.043961 | 0.013787 | 0.034147 | 0.059723 | 0.025577 |

### 2.3 Paired W16 minus scratch

| checkpoint | seed | d_safe | d_official | d_success | d_reach | d_return |
|---|---:|---:|---:|---:|---:|---:|
| 24k | 0 | 0.28000 | 0.05625 | 0.20125 | 0.19250 | 79.26 |
| 24k | 1 | -0.18000 | -0.04250 | 0.37625 | 0.41875 | 93.81 |
| 24k | 2 | -0.03000 | -0.00625 | 0.39875 | 0.41250 | 78.23 |
| 28k | 0 | -0.03000 | -0.00250 | 0.42250 | 0.43000 | 181.84 |
| 28k | 1 | -0.14000 | -0.03750 | 0.27000 | 0.30750 | 132.87 |
| 28k | 2 | -0.18000 | -0.04250 | 0.33000 | 0.35625 | 90.66 |

### 2.4 Seed-0 20ep monitors

| step | safe_ep | official_safety | success | reach | return |
|---:|---:|---:|---:|---:|---:|
| 20480 | 0.80000 | 0.96875 | 0.73750 | 0.73750 | 179.69 |
| 24576 | 0.90000 | 0.98125 | 0.62500 | 0.63125 | 182.27 |
| 28672 | 0.75000 | 0.96250 | 0.63125 | 0.65625 | 261.36 |

### 2.5 Last 4,096 training steps

| seed | step | reward_mean | unsafe_mean | task_gate_mean | h_neg_rate | entropy |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 28672 | 0.39458 | 0.000519 | 0.98862 | 0.001923 | 0.79681 |
| 1 | 28672 | 0.29000 | 0.014557 | 0.94680 | 0.030334 | 0.78622 |
| 2 | 28672 | 0.29368 | 0.008331 | 0.95693 | 0.020355 | 0.84760 |

### 2.6 Rule 7.1

| check | value | threshold | pass |
|---|---:|---:|---|
| mean_safe | 0.66333 | 0.69330 | false |
| mean_official | 0.93292 | 0.94170 | false |
| mean_success | 0.61750 | 0.36000 | true |
| mean_return | 198.56688 | 55.00000 | true |
| worst_success | 0.55750 | 0.20000 | true |
| worst_safe | 0.56000 | 0.30000 | true |
| safe_range | 0.27000 | 0.20000 | false |
| worst_safe_spread | 0.56000 | 0.55000 | true |
| full_pass |  |  | false |

### 2.7 Seed screen flags

| seed | task_benefit | safety_ok | cost_ok | passed |
|---:|---|---|---|---|
| 0 | true | true | true | true |
| 1 | true | false | true | false |
| 2 | true | false | true | false |

## 3. Ant V_T gradient 250k

### 3.0 V_T quality

| split | EV | pearson | RMSE | target_std |
|---|---:|---:|---:|---:|
| val | 0.483832 | 0.697111 | 24.103290 | 33.549084 |
| test | 0.404695 | 0.659731 | 26.051007 | 33.764072 |

| arm | mode | safe_ep | forward_return | episode_return | cost_fraction | episode_length | final_x |
|---|---|---:|---:|---:|---:|---:|---:|
| gradient warm | det | 0.99000 | 3816.50 | 4458.09 | 0.000940 | 1000.00 | 190.83 |
| no-V_T control | det | 0.89000 | 3811.20 | 4449.88 | 0.006610 | 993.46 | 190.57 |
| gradient warm | stoch | 0.78000 | 2968.53 | 3394.34 | 0.003424 | 906.15 | 148.43 |
| no-V_T control | stoch | 0.62000 | 2856.21 | 3267.17 | 0.002977 | 879.44 | 142.82 |

| metric | det_abs | det_rel | stoch_abs | stoch_rel |
|---|---:|---:|---:|---:|
| safe_episode | 0.100000 | 0.112360 | 0.160000 | 0.258065 |
| forward_return | 5.298747 | 0.001390 | 112.316194 | 0.039324 |
| episode_return | 8.214611 | 0.001846 | 127.165037 | 0.038922 |
| cost_fraction | -0.005670 | -0.857788 | 0.000446 | 0.149924 |

| flag | value |
|---|---|
| task_gain | false |
| safety_not_worse | true |
| vt_grad_supported | false |

## 4. MAPPO fixed-penalty 16k

| variant | seed | safe_ep | collision_ep | official_safety | success | reach | return |
|---|---:|---:|---:|---:|---:|---:|---:|
| penalty1 | 0 | 0.00000 | 1.00000 | 0.45875 | 0.19375 | 0.38125 | 83.10 |
| penalty1 | 1 | 0.00000 | 1.00000 | 0.26625 | 0.08000 | 0.30375 | 46.97 |
| penalty1 | 2 | 0.03000 | 0.97000 | 0.50375 | 0.12375 | 0.21875 | 48.03 |
| penalty10 | 0 | 0.00000 | 1.00000 | 0.35125 | 0.01875 | 0.07375 | -2.72 |
| penalty10 | 1 | 0.04000 | 0.96000 | 0.58375 | 0.01375 | 0.02375 | -43.05 |
| penalty10 | 2 | 0.06000 | 0.94000 | 0.62250 | 0.01000 | 0.03250 | -45.28 |
| penalty100 | 0 | 0.03000 | 0.97000 | 0.62875 | 0.02000 | 0.02625 | -48.93 |
| penalty100 | 1 | 0.10000 | 0.90000 | 0.66875 | 0.01625 | 0.03250 | -58.33 |
| penalty100 | 2 | 0.09000 | 0.91000 | 0.72375 | 0.01125 | 0.01750 | -71.17 |

| variant | metric | mean | std | min | max | range |
|---|---|---:|---:|---:|---:|---:|
| penalty1 | safe_episode | 0.01000 | 0.01732 | 0.00000 | 0.03000 | 0.03000 |
| penalty1 | official_safety | 0.40958 | 0.12615 | 0.26625 | 0.50375 | 0.23750 |
| penalty1 | official_success | 0.13250 | 0.05738 | 0.08000 | 0.19375 | 0.11375 |
| penalty1 | official_reach | 0.30125 | 0.08128 | 0.21875 | 0.38125 | 0.16250 |
| penalty1 | episode_return | 59.36 | 20.56 | 46.97 | 83.10 | 36.13 |
| penalty10 | safe_episode | 0.03333 | 0.03055 | 0.00000 | 0.06000 | 0.06000 |
| penalty10 | official_safety | 0.51917 | 0.14671 | 0.35125 | 0.62250 | 0.27125 |
| penalty10 | official_success | 0.01417 | 0.00439 | 0.01000 | 0.01875 | 0.00875 |
| penalty10 | official_reach | 0.04333 | 0.02670 | 0.02375 | 0.07375 | 0.05000 |
| penalty10 | episode_return | -30.35 | 23.95 | -45.28 | -2.72 | 42.56 |
| penalty100 | safe_episode | 0.07333 | 0.03786 | 0.03000 | 0.10000 | 0.07000 |
| penalty100 | official_safety | 0.67375 | 0.04770 | 0.62875 | 0.72375 | 0.09500 |
| penalty100 | official_success | 0.01583 | 0.00439 | 0.01125 | 0.02000 | 0.00875 |
| penalty100 | official_reach | 0.02542 | 0.00753 | 0.01750 | 0.03250 | 0.01500 |
| penalty100 | episode_return | -59.48 | 11.16 | -71.17 | -48.93 | 22.23 |

## 5. Density screen

| condition | policy | safe_ep | official_safety | success | reach | return |
|---|---|---:|---:|---:|---:|---:|
| 8x8 | A28 | 0.75000 | 0.95000 | 0.21250 | 0.21875 | 60.11 |
| 8x8 | A16 | 0.75000 | 0.95625 | 0.35625 | 0.36250 | 55.45 |
| 8x8 | H0 | 0.65000 | 0.93125 | 0.35625 | 0.38750 | 70.00 |
| 4obs | A28 | 0.70000 | 0.95000 | 0.18750 | 0.20000 | 55.06 |
| 4obs | A16 | 0.80000 | 0.96875 | 0.38750 | 0.39375 | 54.50 |
| 4obs | H0 | 0.60000 | 0.91875 | 0.41250 | 0.44375 | 65.65 |
| 16obs | A28 | 0.60000 | 0.90000 | 0.18750 | 0.19375 | 51.48 |
| 16obs | A16 | 0.50000 | 0.90625 | 0.34375 | 0.35625 | 56.29 |
| 16obs | H0 | 0.25000 | 0.84375 | 0.34375 | 0.37500 | 64.85 |
| 16agents | A28 | 0.30000 | 0.91875 | 0.21875 | 0.23438 | 61.03 |
| 16agents | A16 | 0.45000 | 0.92188 | 0.35938 | 0.37812 | 61.52 |
| 16agents | H0 | 0.05000 | 0.85313 | 0.30938 | 0.35313 | 65.07 |

## 6. Record files

| path | sha256 |
|---|---|
| outputs/v6/independent/nav/nav_3seed_16k_24k_28k_table.json | 3446594f79f2a2099c4a7994c89376567cb1fa8d707ae9392fb70769cef8cdce |
| outputs/v6/independent/nav_w16_task_12k/table_3seed.json | 0a9b4c4d47d1be04980fd33d5cb4879209dd9541facb8dd9ba6ba2abd712a7ec |
| outputs/v6/independent/ant_vt_grad_250k/decision.json | b9528ea15f609256e8902eb2205d918940810c1af32ccfff19667226085f13f5 |
| outputs/v6/independent/nav_density/seed0/summary.json | 179d6d90f71577673d56bdea490fcf8d563d66b3e5ac47006d44ce86717eb16c |

## 7. Config and checkpoint hashes

| item | path | bytes | sha256 |
|---|---|---:|---|
| nav 28k config | outputs/v6/independent/nav_stageb_repair_28k/seed0/config.yaml | 2461 | 6fba009c2e7bb5a862ed4e0de1502aa7e9c335e4b466e97323e2805917920bd2 |
| nav repaired h | outputs/v6/independent/h_stageb_repair/v6_h_stageb_repair.pkl | 1467835 | 0f67eaeec99912881017231234aa10846c5edfda767118da2157b2da7df7fb13 |
| W16 config | nav/configs/v6/v6_nav_w16_task_12k.yaml | 2501 | 9261e3c6a672dc0291681a9a00c1401865e9ac65f79b1292dda1684aeac23744 |
| T0w V_T seed0 | outputs/v6/independent/t0w_vt/v6_task_value.pkl | 2418369 | 9038e503571d2ba19dd4bc537d4643097a435bcdfaf451f7c3ff5e8b13fb88ce |
| W16 V_T seed1 | outputs/v6/independent/nav_w16_task_12k/vt_data/seed1/v6_task_value.pkl | 2418433 | ba026d1b1263f6eb5cbc29e7aa77e43f4ba0e172b2802dc35d2f846433999d6f |
| W16 V_T seed2 | outputs/v6/independent/nav_w16_task_12k/vt_data/seed2/v6_task_value.pkl | 2418433 | 9bb360d9fef62560ddb783f21d231bd769c805417b2f96a35ab894bb3d9aaa17 |
| Ant V_T | outputs/v6/independent/ant_vt/ant_task_value.pkl | 93245 | 289b2bbe49be895d4cb3c96d41b6a550bdffe510516a2b62ae2437494d4465f0 |
| Ant grad config | nav/configs/v6/v6_ant_vt_grad_250k.yaml | 1479 | fa1cc13c34910ea10693eec123014246744a65882d0cb953e431256bfd9deb11 |
| Ant grad control config | nav/configs/v6/v6_ant_vt_grad_control_250k.yaml | 1287 | 555665bce6dee566b9590c78eb73b091ab057c956f66ec1afcd5e08a0d64161e |
| MAPPO penalty1 config | nav/configs/baselines/mappo_double_integrator_penalty1.yaml | 964 | 6424e094b6ad5d9f055450b79f7ce017b1d15e328cd0513b41d40ed6a9d5189c |
| MAPPO penalty100 config | nav/configs/baselines/mappo_double_integrator_penalty100.yaml | 968 | 1d98e292815ca8c8034000e3bbb8a27059336a8cfd32c433aecaa8be80f96074 |

