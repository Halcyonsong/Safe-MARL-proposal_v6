# CBF-guided MARL → V6：predictive-h safety gate

> 核心动机：CBF 提供较强的安全基线，但依赖 `u_ref` 和 QP；V6 改为学习 predictive `h(s)`，把 `h` / `grad_h` 作为 actor 特征，并用 `h` 做训练期 reward gate，全流程 QP-free。
>
> V6 的 `h` / actor 均从零训练，不复用 V5 或 official h 权重。

# nav 任务

- 8 agents，二维 double integrator；状态 `[x,y,vx,vy]`，质量 `0.1`，半径 `0.05`，速度每轴限制 `±0.5`；动作是二维力 `[-1,1]^2`。
- 4×4 只是采样范围，无外墙；每 episode 随机 8 个矩形障碍（边长 `0.1–0.5`）。32 线 LiDAR，感知/通信半径 `0.5`；actor 取最近 5 个邻居，h 用半径内全部邻居。
- 256 步，`dt=0.03`；碰撞或到达都不提前终止，超时才结束。碰撞：agent 中心距 `<0.1`，或进入按 `0.05` 膨胀的障碍；到达：距自身目标 `<0.1`。

## h

### 输入

- padded 官方 graph：273 节点 = 8 agent + 8 goal + 8×32 LiDAR + 1 dummy；384 边槽位 = agent-agent + agent-goal + agent-LiDAR。
- 节点特征 `[B,273,3]`（one-hot 类型），节点状态 `[B,273,4]`，边特征 `[B,384,4]`（相对状态）。
- GNN 只用节点类型和相对边，不用绝对坐标；节点状态用于重构图和 `grad_h`。

### 网络

- 1 层 GNN message passing，hidden `128`；MLP head `128 → 256 → 256 → 1`，最后 `tanh`；约 366k 参数。

### 输出

- `h_raw: [B,8]`；`h_shifted = h_raw - 0.061149`；`grad_h: [B,8,4]`，局部对角 `∂h/∂s`，作为 actor 特征。
- `h_offset` 只在 h validation split 的 `label=±1` 行上、从 1%–99% 分位点中选 balanced accuracy 最大点；只移动分类边界，不改变排序、AUROC 或 `grad_h`。
- h 是 state-only、冻结的 learned safety function，不是 CBF certificate；`grad_h` 是 state-space 局部斜率，不是 action-space 安全方向。

### 训练目标

- 当前几何 margin：`m = min(邻居距离 - 0.1, LiDAR 距离 - 0.05)`，clamp 到 `[-0.5,0.5]`；不含外墙。
- future-min target：`y = tanh(min_{k=0..32} m_{t+k} / 0.2)`；当前或窗口内 official unsafe 时 `y ≤ -0.1`。
- 二值 forward-window 标签（当前 unsafe 优先）：

| 当前 | 未来 32 步 | label | 用途 |
|---|---|---|---|
| unsafe | 任意 | `-1` | unsafe hinge |
| safe | 至少一次 unsafe | `0` | ignored；不进 hinge |
| safe | 全 safe | `+1` | safe hinge |

- `0` 只是 mask，不是输出目标：h0 完全忽略；h1 进入 future-min regression，但不进 safe/unsafe hinge。
- 32 步二值标签语义沿用官方 GCBF+ `GCBFPlus.safe_mask`（`horizon=32`；当前 safe 但窗口内 unsafe 的行在 safe/unsafe hinge 中都不参与）。h1 的 continuous future-min regression 和 gradient alignment 是 V6 扩展；GCBF+ 本身用 hinge + hdot，不做 future-min regression。
- 标签里的 `unsafe` 用官方 `unsafe_mask`（collision + heading-dependent warning）；评估 `safe_episode` 只用 collision。
- 训练数据：最终 h 用 50% 策略 rollout + 50% 随机策略；随机数据补状态覆盖，不是任务分布，存在偏移。
- 边界：32 步是训练窗口，不是预测保证。长时程校准弱：actor1 状态上 17–32 步 h 均值 `+0.242`，target `-0.222`，仅 `31.9%` 的 h<0；future-collision AUROC `0.908` 只说明排序。h 不条件于动作，不能区分同一状态下的不同未来。

### 损失

- `L = L_safe + L_unsafe + 0.5 L_target + 0.1 L_grad`；hdot 关闭（`0`）。
- safe hinge：`max(0, 0.02 - h)`；unsafe hinge：`max(0, h + 0.02)`；target：SmoothL1 拟合 future-min target，包含 ignored 行；grad alignment：对齐当前几何 margin 的局部梯度，仅覆盖 `±1` 行，`0` 行无梯度约束。
- `±1` 只是 hinge 的侧别，不是回归目标；hinge 是单边约束，满足 margin 后无梯度。连续数值监督来自 future-min target，所以输出不会退化成两个值；远安全区会被 `tanh` 饱和到接近 `+1`，不能把数值大小当成安全概率。

### 数据与迭代

| 阶段 | 数据 / 目标 | 结果 |
|---|---|---|
| h0 | 随机策略 100×256，25,600 行 + 8,000 增广；二值 horizon 标签；hinge + hdot | actor0 的输入 |
| actor0 | 冻结 h0，16k，无 gate，即 h-only | seed0 `safe_episode 0.620`、`success 0.5325` |
| h1 | 50% 策略 rollout（h-only seed0 + Stage A seed0，各 50 episodes）+ 50% 随机覆盖（100 episodes），40,960 行；future-min target 全行 SmoothL1 + hinge + grad；hdot `0` | actor1 的输入 |
| actor1 | 冻结 h1，16k，gate `0/0.267270`，3 seeds | `safe_episode 0.693±0.168`、`success 0.410±0.058` |
| h2/actor2 | final actor1 100 条 stochastic + 100 条随机覆盖；C0a 17–32 步 MAE `0.826` vs h1 `0.709`；actor2 `safe 0.510` vs `0.840` | 预注册检查全失败，分支停止 |

- `h2+actor1 = 0.90` 是单 seed、off-policy 诊断，不是 h2/actor2 结果，也不进主结果；无 h3/actor3。

## actor

### 输入

- `102 = 4` own state + `4` goal-relative + `20` 最近 5 个 agent 的相对位置/速度 + `64` LiDAR + `1` `h_shifted` + `4` `grad_h` + `5` 零填充 task feature。
- 8 agents 共享 per-agent MLP；graph 只用于冻结 h。当前 h provider 用 full padded graph 批量计算，是集中式 provider，不是严格 decentralized。

### 网络与输出

- 共享 trunk `102 → 256 → 256`，然后分头输出：task action `tanh(Wz)`、correction `0.3 tanh(Wz)`、action mean `μ=clip(task+correction,-1,1)`，以及 per-agent `V(s)`、`C(s)`。
- 训练时 `a ~ N(μ, diag(σ²))`；评估用确定性 `μ`。correction 只是有界残差，不是 QP 投影或安全滤波器。

### 训练

- 冻结 h，actor 从零用 PPO 训练；cost 信号来自共享 trunk 的 cost head。
- gate：`g = clip(h_next / 0.267270, 0, 1)`，只乘正 progress；`h_next≤0` 时 `g=0`，`h_next≥0.267270` 时 `g=1`，中间线性。它用下一状态 h 动态折扣前进奖励：低 h 区域的正 progress 变便宜，远安全区基本不打折；reach、terminal、collision 都不 gate。gate 是训练期 reward 乘子，不进入 actor 输入、不反传，只改 advantage；不是 runtime action filter/shield，评估关闭不改变动作。
- 奖励：`r = 20·[Δd>0 ? Δd·g : Δd] + 5·1[reached] + 20·1[terminal ∧ reached] - 10·1[collision]`，其中 `Δd = ||p_t-g|| - ||p_{t+1}-g||`。碰撞不终止 episode，只产生 cost。
- 多头 AC：共享 encoder `102→256→256`，分头输出 per-agent `V(s)`、`C(s)`；8 个 agent 的 V/C 是同一 head 的 batch 输出，不是 8 个 critic。V 回归 shaped return（task reward − `10·cost`），C 回归 collision cost return；不是多个独立 critic。
- 反传：一次总 loss、一次 backward；V/C head 参数各收自己的回归梯度，共享 encoder 收 policy/value/cost 梯度之和。`A/A_c` 是 rollout 常量，当前 V/C 输出不进入 policy loss；`cost_actor` 只经 policy ratio 反传，不通过 C。系数固定 `value 0.5`、`cost 1.0`、`cost_actor 0.5`，不是 Lagrangian；`h_barrier_coef=0`，没有 h barrier/CBF loss。
- 预算：16,384 steps/seed，rollout `128`，PPO `4` epochs / minibatch `256`，`γ/λ=0.99/0.95`，3 seeds；正式评估 100 条 deterministic episodes，gate 关闭；目前无 100ep stochastic。

## 测评

### 协议

- 环境：官方 GCBF+ `DoubleIntegrator`，8 agents、4×4、8 obstacles、256 steps；固定 100 个场景（seed `10000..10099`），所有方法共用。
- V6：确定性均值动作，gate 关闭，无 QP / action filter / runtime shield；每个 seed 用 final checkpoint，报告 seeds 0/1/2 的 mean ± std。
- 预算：V6 16,384 steps/seed，from scratch；GCBF+ 预训练 step 1000，actor-only；HandCBF/CBF-QP 无训练 + 在线 QP；Safe-MARL 16k（从 8k warm start）；MAPPO 16k；MAPPO-Lag 100k；Safe-MARL/MAPPO/MAPPO-Lag 均为 3 seeds。
- 训练中每 8192 步的 20ep 监控只记录指标，不反传；正式结果只用 final checkpoint。

### 指标

| 指标 | 定义 |
|---|---|
| `safe_episode` | 一个 episode 内 8 个 agent 均无 collision；**主安全指标** |
| `official_safety` | 8 个 agent 中“整个 episode 从未 collision”的比例；per-agent any-time |
| `official_success` | per-agent：至少到达一次自身目标，且整个 episode 从未 collision |
| `official_reach` | per-agent：至少到达一次自身目标，不考虑 collision |
| `episode_return` | V6 shaped return；V5 `return` 定义相近，可作参考；不同 baseline 的累计未完全对齐，主表不横比 |

`official_*` 是历史字段名，指环境 collision 语义，和 official h / CBF certificate 无关；这些指标直接来自环境 `info['collision']` / `reached`，不经过 cost head，也不依赖 h 的符号。`safe_episode` 比 per-agent 的 `official_safety` 更严格；跨方法对照只用同名列。

### 基线

| 方法 | 类型 | 预算 / 来源 | QP? | `safe_episode` ↑ | `official_safety` ↑ | `official_success` ↑ | `official_reach` ↑ |
|---|---|---|---:|---:|---:|---:|---:|
| **V6 final**（predictive h + gate） | learned, QP-free | 16k, 3 seeds, from scratch | 否 | 0.6933 ± 0.1680 | 0.9417 ± 0.0378 | 0.4100 ± 0.0583 | 0.4296 ± 0.0695 |
| GCBF+ official | CBF / learned | pretrained step 1000，actor-only | 否 | 0.9900 | 0.9988 | 0.8613 | 0.8625 |
| HandCBF / centralized CBF-QP | CBF-QP / model-based | 无训练；在线 QP（OSQP） | 是 | 0.9000 | 0.9788 | 0.7813 | 0.7888 |
| Safe-MARL（V5 full, ref） | CBF-informed Safe MARL | 16k, 3 seeds, from 8k | 否 | 0.7467 ± 0.1159 | 0.9500 ± 0.0308 | 0.5121 ± 0.1352 | 0.5363 ± 0.1490 |
| MAPPO（V5 ref, 16k） | task-only MARL | 16k, 3 seeds | 否 | 0.0033 ± 0.0058 | 0.3079 ± 0.0995 | 0.0571 ± 0.0281 | 0.1713 ± 0.0435 |
| MAPPO-Lag（V5 ref, 100k） | constrained MARL | 100k, 3 seeds | 否 | 0.0033 ± 0.0058 | 0.4325 ± 0.0379 | 0.1363 ± 0.0508 | 0.3088 ± 0.0928 |

- 这些外部行都是 V5 归档参考，不是 matched V6 rerun。Safe-MARL（V5 full）用 official h + 8k warm start；nav MAPPO 是 task-only（`collision_penalty=0`，无 cost critic）；nav MAPPO-Lag 是 100k constrained run（task reward 不含 collision penalty；碰撞只进入 cost critic + Lagrange，`cost_limit=0.01/step`），归档 final `cost_mean≈0.04`，约束未满足。Ant 的 V5 MAPPO-Lag 2M fine-tune 是另一项参考，不要和 nav 行混用。
- `safe_episode` 要求 8 agents × 256 steps 全零碰撞；归档 eval 的 per-step collision rate（`unsafe`）MAPPO/MAPPO-Lag 约 `6.6%/4.5%`，所以 `safe_episode 0.0033` 与 `official_safety 0.31/0.43` 不矛盾：后者是 per-agent any-time，前者是 all-agent all-time。
- 归档里另有 **same-reward MAPPO seed0 pilot**（`collision_penalty=10`，100k，100ep）：`official_safety 0.4775`、`safe_episode 0.0`、`reach 0.0813`、`return 2.47`；只有 1 个 seed，未进主表。它说明 reward 里加碰撞惩罚能改善 per-agent safety，但会把任务压塌，且仍不能消除 all-agent 碰撞。
- V6 `official_safety 0.942` 接近 Safe-MARL `0.950`、CBF-QP `0.979`；但 `safe_episode 0.693` 低于 Safe-MARL `0.747`、HandCBF `0.900`、GCBF+ `0.990`。任务 `success 0.410` 也明显低于 Safe-MARL `0.512`、HandCBF `0.781`、GCBF+ `0.861`。安全有潜力，任务不是 SOTA。
- V6 3-seed per-seed `safe_episode`：`0.840 / 0.510 / 0.730`（mean `0.693±0.168`）；h-only（h0，无 gate）：`0.620 / 0.490 / 0.240`；per-seed deltas `+0.22/+0.02/+0.49`，paired `p=0.216`（n=3）。seed1 基本持平，收益不均匀；任务 mean success `0.410` vs h-only `0.407`，但 seed0 `0.4275` vs `0.5325`（低 10.5 pp）。当前结果支持“有潜力但方差大”，不支持“稳定达到 0.84/0.90”。

# Ant 任务

Ant 是第二个验证场景，用来检验 V6 机制能否用于不同安全几何和长时程连续控制；h、数据和 actor 都独立于 nav。

- 2 个 agent 协作控制**同一只**官方 `Ant-v2` 的 8 个执行器（`2x4` 配置，4 个 hip + 4 个 ankle）：每个 agent 输出 4 维关节力矩，按 actuator 顺序拼成 8 维；agent 0 驱动前 4 个 actuator，agent 1 驱动后 4 个（XML 顺序，代码不做按腿重排）。
- 观测 31 维：29 维官方 Ant 状态 + 2 维 agent one-hot，逐样本归一化；两个 agent 的原始 29 维本体状态相同，仅 agent ID 不同（归一化后所有分量会有轻微差异）。
- 1000 个控制步，控制 `dt=0.05 s`（5 个物理子步），约 50 s；无目标点，任务是沿 +x 方向尽量前进。
- 初始状态在默认站立姿态附近加噪声：qpos `±0.1` 均匀，qvel `0.1×N(0,1)`。
- 走廊：三段 30° 折线墙，以 x=0、40、80 为中心；两条平行墙 y 向间距 10 m，中心线在 x=20、60 处折返；官方 wall 条件 `|y_walldist| < 1.8`。

## h

### 输入

- 9 维躯干几何 `[d_left, d_right, v_n_left, v_n_right, z, z_rot, vx, vy, vz]`：`d_*` 为到两侧墙的绝对距离，`v_n_*` 为法向接近速度，`z` 为躯干高度，`z_rot` 为直立度，`vx/vy/vz` 为躯干速度；Ant 的安全状态是单个体几何，不需要 graph/LiDAR。
- 四个解析 margin：wall `d_wall - 1.8 - 0.5 max(v_n,0)`；floor `z - 0.2 - 0.2 max(-vz,0)`；ceil `1 - z`；upright `0.2(z_rot + 0.7)`；当前 margin 取四者最小值。

### 网络

- `9 → 128 → 128 → 1` MLP，LayerNorm + SiLU，最后 `tanh`，约 18k 参数。
- 不用 GNN：Ant 的安全状态没有 nav 的图结构，9 维几何已包含 wall / height / upright 信息。

### 输出

- `h_raw`：标量安全值；`h_shifted = h_raw - 0.0002995`，`h_shifted < 0` 为 unsafe。
- `grad_h`：9 维 `∂h/∂geom`，作为 actor 特征；`h_offset` 只移动分类边界，不改变排序、AUROC 或 `grad_h`。
- 4 个解析 margin 是 4 维诊断量（`h_vec`），不进入 actor；actor 只接收 1 维 `h_shifted` 和 9 维 `grad_h`。

### 训练目标

- rollout 行 target：`y = tanh(min_{k=0..32} m_{t+k} / 0.5)`；阈值覆盖增广行用当前 margin `tanh(m/0.5)`。
- 标签：当前或未来 32 步内 unsafe 记 `-1`，否则 `+1`；未来 unsafe 行不再置 `0`，没有 ignored 行；target/grad 覆盖全部行。
- 损失：`L = L_safe + L_unsafe + 0.5 L_target + 0.1 L_grad`；hinge margin `0.05`；无 hdot 项。
- 数据：120 个 episode 的 cost-only 2M 策略 rollout（stochastic）+ 30k 阈值覆盖增广，共 114,523 行，约 26% unsafe。离线 test：balanced accuracy `0.9145`、unsafe recall `0.8793`、false alarm `0.0364`、AUROC `0.9511`、grad cosine `0.9225`。

## actor

### 输入

- 每个 agent 41 维：31 维观测 + 1 维 `h_shifted` + 9 维 `grad_h`。
- 31 维 = 29 维 Ant 状态 + 2 维 agent one-hot；29 维 = `z`(1) + 四元数(4) + 8 个关节角 + 躯干速度(6) + 8 个关节速度 + `x/5`(1) + `y_off`(1)。
- `h` 是 1 维标量；`grad_h` 是 `∂h/∂geom` 的 9 维向量，顺序与 h 输入的 9 维几何一致。
- 两个 agent 的原始 29 维本体状态相同，仅 2 维 agent one-hot 不同（归一化后所有分量会有轻微差异）；h/grad 相同，但各自有独立 actor，不是 nav 的参数共享 actor。

### 网络

- actor：每个 agent 一个 `41 → 64 → 64 → 4` MLP（ReLU、feature norm + LayerNorm）。
- critic：集中式 `31 → 64 → 64 → 1`，带 ValueNorm；actor/critic 使用独立 optimizer。

### 输出

- 4 维高斯策略的均值；`log_std` 初始 `-1.0`（std≈`0.368`），训练采样、评估取均值。
- 环境 actuator 的 ctrlrange 为 `[-1,1]`；策略侧不做 tanh squash。

### 机制

每个 agent 以 41 维输入生成 4 维高斯动作；两个 agent 的动作拼成 8 维，截断到 `[-1,1]` 后经 `motor gear=150` 产生关节力矩；没有 PD/IK/步态生成器。`h` 网络冻结，critic 只看 31 维状态；`h` 是躯干级安全分数，`grad_h` 是 9 维状态空间斜率，不是动作空间安全方向。训练时 `h` 只 gate forward reward，配合固定 cost penalty（`0.1→1.0`）塑形；没有 QP、action projection 或 runtime shield；评估关闭 gate，动作仍是 actor 原始输出。

### 训练

- 冻结 h，actor/critic 从零训练 10M steps/seed × 3 seeds；8 envs × 2048 rollout = 16,384 transitions/update。
- PPO：5 epochs、minibatch 400、actor lr `9e-5`、critic lr `5e-3`、`γ/λ=0.99/0.95`、clip `0.2`、entropy `0.01`。
- 奖励：`r_train = (r_env - forward) + g·forward - c·cost`，`g=clip(h_next/0.05,0,1)`；只 gate forward，cost 不 gate。gate 只改训练 reward/advantage，不反传；评估关闭不改变动作。
- `c` 从 `0.1` 线性升到 `1.0`（0.5M–2M steps）；`h_barrier_coef=0`。
- 没有 Lagrangian 乘子或 cost head；安全信号是固定的官方 cost penalty；critic 是单头 centralized V，不是 nav 的共享 V/C 多头。

## 测评

### 协议

- 场景与预算：官方 `Ant-v2 2x4`，1000 步；V6 为 10M steps/seed × 3 seeds，from scratch；matched cost-only 同 10M、同 seeds、同评估。
- checkpoint：每个 seed 的 final 10M checkpoint，不用训练中 monitor-best。
- 评估：每个 seed 100 deterministic + 100 stochastic episodes；eval seed `20000 + 100*seed`；gate 关闭，只用官方 reward/cost。
- 训练中每 500k 步的 20ep 监控只记录，不反传。

### 指标

| 指标 | 定义 |
|---|---|
| `safe_episode` | 整个 episode 无 wall/fall cost；**主安全指标** |
| `cost_fraction` | 有 cost 的步占比；per-step cost 频率 |
| `wall_episode` / `topple_episode` | 至少一次 wall / topple |
| `forward_return` | 累计 `(x_t-x_{t-1})/dt` |
| `final_x` / `survived` | episode 结束时 torso x / 跑满 1000 步 |
| `episode_return` | 官方总 return；V5 参考对照用 |

`safe_episode` 只要有一次 wall/fall 就记 0；`cost_fraction` 低不代表 episode-level 安全，二者要一起看。

### 基线

| 方法 | 训练 / 求解 | 模式 | `episode_return` | `cost_fraction` | `safe_episode` | `survived` | `final_x` |
|---|---|---:|---:|---:|---:|---:|---:|
| **V6 predictive gate** | 10M, 3 seeds, from scratch | det | 4426.96 ± 105.61 | 0.0012 ± 0.0010 | 0.970 ± 0.020 | 0.990 ± 0.010 | 188.58 ± 5.20 |
| **V6 predictive gate** | 10M, 3 seeds, from scratch | stoch | 3477.87 ± 168.21 | 0.0020 ± 0.0019 | 0.840 ± 0.056 | 0.863 ± 0.035 | 150.54 ± 7.01 |
| V6 cost-only matched | 10M, 3 seeds, from scratch | det | 4705.51 ± 79.04 | 0.0296 ± 0.0094 | 0.063 ± 0.047 | 0.953 ± 0.031 | 205.99 ± 3.44 |
| V6 cost-only matched | 10M, 3 seeds, from scratch | stoch | 3587.33 ± 237.59 | 0.0618 ± 0.0157 | 0.020 ± 0.020 | 0.770 ± 0.069 | 160.32 ± 10.88 |
| MAPPO（V5 reference） | 10M, λ=0 | det | 4477.2 | 0.3157 | 0.00 | 0.89 | 203.4 |
| MAPPO-Lag（V5 reference） | 2M constrained fine-tune | det | 4471.2 | 0.0218 | 0.00 | 0.95 | 203.2 |

- 外部 MAPPO/MAPPO-Lag 的正式 eval 是 deterministic（上游 `deterministic=True`，V5 复现同）；V6 的 stoch 行只用于 V6 内部 gate vs cost-only 的噪声鲁棒性检查，不能和 det 行或外部 det 基线直接比较；若要把 stoch 作为正式对比，需同协议重跑基线。
- V6 cost-only matched 保留相同 cost penalty、只是不用 h，不是纯 MAPPO；纯 MAPPO 见 V5 MAPPO 行；V6 内部 pure-task MAPPO 只有 seed 0，仅作方向参考，不放进主表。
- V5 Ant reference 是单组归档值（eval seed `10000..10099`，100 个固定场景，无 3-seed 方差）；V6 用 `20000+100*seed`，场景不同，不是 paired 对照；V5 MAPPO-Lag 是 2M constrained fine-tune，不是 10M from scratch。
- 官方 MAPPO-Lag 论文（Gu et al., 2023）给的是多 seed cost/reward 曲线和 Ant safety bound `0.2`（10M steps、32 eval episodes），没有 Ant 2x4 的逐项最终表；V5 `MAPPO-Lag` 行不能当作官方 claim 的复刻。
- V6 det `forward_return 3771.51` vs cost-only `4119.70`；stoch `3010.73` vs `3206.28`。V5 不报告 `forward_return`，不能和 V5 `return` 直接比。
- det：`safe_episode +90.7 pp`、`cost_fraction -96.0%`、forward 保留 `91.6%`；stoch：`+82.0 pp`、`-96.8%`、`93.9%`。det 残余风险约 `2.0% wall + 1.0% topple`；stoch 主要是 topple `13.7%`、wall `2.3%`。
- 对 MAPPO/MAPPO-Lag：V6 det `episode_return 4426.96` 比 `4477.2/4471.2` 低约 `1%`，`final_x 188.58` 比 `203.4/203.2` 低约 `7%`；`cost_fraction 0.0012` 远低于 `0.3157/0.0218`，`safe_episode 0.97` 高于 `0.00`。安全/cost 明显更好，任务不是全面领先。

### 指标差异

- `return` / `episode_return`：V5 `return` 是官方 total return；V6 `episode_return` 同定义，可作参考；V6 主任务指标 `forward_return` 只累计 forward 分量，不能和 V5 `return` 直接比。
- `cost rate` / `cost_fraction`：两者定义相同，都是二值 cost 的逐 step 频率，可直接对照。V6 另外报告 `wall_fraction` / `topple_fraction`，与 V5 的 per-step `wall rate` / `topple rate` 同口径。
- `safe_episode`：V5 和 V6 都是“整个 episode cost=0”，定义相同；V5 MAPPO/MAPPO-Lag 的 `wall_episode` 都是 `1.00`，`topple_episode` 分别为 `0.11/0.05`，所以 `safe_episode=0` 是真失败，不是指标更严。V6 gate 的 `wall_episode` 为 det `0.020` / stoch `0.023`，`topple_episode` 为 det `0.010` / stoch `0.137`。
- `wall rate` / `topple rate` vs `wall_episode` / `topple_episode`：前者是 per-step 频率，后者是 episode-level“至少一次”；episode-level 更严格，但只在 V6 内部报告，不替换 V5 的 per-step 数值。
- 评估协议：V6 是 10M × 3 seeds ×（100 deterministic + 100 stochastic）final checkpoint；V5 MAPPO/MAPPO-Lag 是各一个归档 checkpoint、100 deterministic episodes。**V6 的协议更严格**（多 seed、加 stochastic、固定 final checkpoint），但主表 det 对 det 仍是同口径；stochastic 是 V6 额外的鲁棒性检查。

# nav 任务：task 实验（附加）

> 定位：主安全结论不变。这里只检验把任务价值 `V_T` 作为 actor 特征，能否在不破坏 h/gate 安全收益的前提下改善任务；nav 已有 T0w 4k 诊断，Ant 见下一节，暂占位。

## 增加的输入

- nav 原 actor 输入 102 维 = 92 obs + 1 `h` + 4 `grad_h` + 5 维 task 零占位；task 实验打开这 5 维，网络宽度不变。
- task 特征：`[A_T, ∂V_T/∂x, ∂V_T/∂y, ∂V_T/∂vx, ∂V_T/∂vy]`。
- 拼接顺序：`[obs(92), h(1), grad_h(4), task(5)]`。
- `A_T` 是相对任务价值，不是 PPO advantage；`grad_s V_T` 是状态空间斜率，不是动作空间安全方向；两者都不进入 h、gate 或 cost。

## V_T 设计

- 独立任务价值网络，与 h、PPO critic 不共享参数；只拟合 task return，不看 cost，不看 h gate。它是离线有限时程预测器，不参与 GAE，也不替代 PPO value。
- 目标：`G_t^(K) = Σ_{k=0}^{K-1} γ^k r_task(t+k)`，`K=64`，`γ=0.99`；到 episode 边界停止，不跨 terminal bootstrap。`r_task` 用未 gate 的任务奖励，不含安全惩罚。
- 网络：92 维 base obs → `LocalGraphNet`（hidden 128，2 层，线性输出）→ per-agent scalar。
- 数据：冻结 actor1 seed0 的 stochastic rollout，100 episodes × 256 steps × 8 agents，种子 `30000..30099`；按 episode 80/10/10 分 train/val/test，正式评估 seeds 不参与训练或选模。
- 归一化：`A_T = clip((V_T - μ_EMA)/sqrt(σ²_EMA + 1e-6), -5, 5)`，EMA 在 V_T 训练后冻结；`grad_s V_T` 先按行单位化，再乘 `0.5`。
- 质量门：validation EV ≥ `0.20` 且 test Pearson ≥ `0.40`；不过门只报告 V_T 质量问题，不把 actor 结果解释为 task channel 效果。
- 训练隔离：V_T 数据不用于 h，h/gate/cost 不用于 V_T；两者只在 actor 输入处拼接。

## 测评

### 协议与判据

- 从 actor1 seed0 出发 warm-start 4,096 steps（16,384 → 20,480）；task block 开 vs 恒零，h1/gate/cost/reward/optimizer/seed 全部相同。from-scratch 版本需另行预注册。
- task gain：`official_success` 或 `episode_return` 超过 control 的预设阈值；safety non-inferior：`safe_episode` / `official_safety` 不低于 control 的预设 margin。
- 正式评估 100 deterministic episodes，eval seeds `10000..10099`，gate 关闭；没有 QP、action filter 或 runtime shield。

### 已有结果（T0w 4k，100ep）

V_T 质量门：

| 指标 | 阈值 | validation | test |
|---|---:|---:|---:|
| explained variance | ≥ 0.20 | 0.227 | 0.245 |
| Pearson | ≥ 0.40 | 0.486 | 0.499 |

| 指标 | actor1 init | matched no-V_T 4k | T0w 4k | T0w − control |
|---|---:|---:|---:|---:|
| `safe_episode` | 0.840 | 0.930 | 0.890 | -0.040 |
| `official_safety` | 0.97375 | 0.99000 | 0.98250 | -0.00750 |
| `official_success` | 0.42750 | 0.38000 | 0.67875 | +0.29875 |
| `official_reach` | 0.43500 | 0.38125 | 0.68500 | +0.30375 |
| `episode_return` | 60.91 | 69.67 | 161.88 | +92.20 |

- `control_valid=true`；`task_gain=true`（success +29.9 pp，return +92.2）；`safety_not_worse=false`：`safe_episode 0.890 < 0.930 - 0.03 = 0.900`，差 1 个 episode；`official_safety -0.0075` 仍在预设 `-0.01` 内，失败点是 scene-level `safe_episode`。
- 结论：`vt_channel_supported=false`（预注册 safety-first 规则），但 task-first 视角下 T0w 是一个 **task-heavy Pareto 点**：task gain 明显，scene-level safety 只有小的、统计上未分辨的折扣（差 1 个 episode，Wilson 区间重叠）。它能支持“task gain 明显、safety 有小折扣”的 trade-off 表述，不能写成 safety 非劣通过，也不能写成“V_T 不安全”。严格 safety-first 线停止；3-seed from-scratch gate+V_T 复现是可选后续。

# Ant 任务：task 实验（附加）

> 占位：Ant 的 task 输入、V_T 设计、训练协议和结果待补；本轮不写未验证数字，也不把 Ant task 实验写成主结果。

## 增加的输入

（待补）

## V_T 设计

（待补）

## 测评

### 协议与判据

（待补）

### 结果

（待补）

# 结论边界

- h 是 CBF-inspired learned safety function，不是 certificate；`grad_h` 只是 state-space 特征；没有 QP、shield 或 runtime filter。
- gate 只在训练期乘 positive progress，评估关闭；它不是 runtime safety mechanism。
- nav：3-seed 支持“有潜力但方差大”的 preliminary，不支持“稳定达到 0.84/0.90”。paper/final 还需更多 seeds、h1 gate-off 对照、h scalar vs h+grad 消融、seed1 诊断、100ep stochastic；当前 h-only 是 h0 no-gate，gate 与 h 版本仍混淆。deployment 不支持。
- Ant：3-seed 比 nav 稳，但仍只是 benchmark safety，不是 formal guarantee；任务不全面领先。
- Task 实验：附加诊断，不是主结果；`V_T/A_T` 只改任务特征，不改 h/gate/cost。nav 的 T0w 4k 在预注册 safety-first 规则下未过非劣界，但可表述为 task-heavy Pareto 点；不能写成 safety 非劣或 task channel 已成功，Ant 暂占位。
