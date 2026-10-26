# Safe-MARL V6 详细设计

## 独立学习 \(h(s)\) 的安全 MAPPO

**状态：历史设计稿（2026-10-22）。**  
本文保留独立 V6 的设计契约和实验方案，不再是当前进度记录。主实验已完成
h-only、预测式 h gate 和 3-seed 验证；当前结果、artifact 分类和待办见
[`outputs/v6/STATUS.md`](../outputs/v6/STATUS.md) 与
[`outputs/v6/V6_COMPLETE_RECORD.md`](../outputs/v6/V6_COMPLETE_RECORD.md)。

---

## 0. 设计目标与独立性契约

### 0.1 核心目标

用 V6 自己采集的数据、从零训练的 \(h(s)\)，替代 official h / 专家
\(u_{\text{ref}}\) / QP 设计，向 MAPPO actor 提供安全信息：

```text
official h / QP / u_ref
        ↓  替换为
V6 learned h(s) + ∇s h(s)
        ↓
from-scratch MAPPO/PPO actor
```

主实验只验证：

> **V6 自己训练的 h(s) 能否让从零训练的 MAPPO actor 明显比 no-h 更安全，并接近 V5 + official h 的安全水平。**

task channel 是后续优化，不是本轮的前置条件。

### 0.2 允许与禁止

**允许**

- 官方多智能体导航场景：agents、障碍物、目标、动力学、观测、collision/unsafe 标签；
- 通用设计/代码复用：MAPPO/PPO、GAE、actor-critic、图网络/MLP、reward、训练配方；
- V6 自己采集的数据、自己训练的 h/actor、自己产生的 checkpoint；
- V5 的**经验**：\(h,\nabla_s h\) 作为安全特征可以提升安全；但这只是动机，不是 V6 的证据。

**禁止**

- official h checkpoint、`cbf.pkl`、预训练 h 权重；
- official h 的值、梯度、teacher label、蒸馏目标；
- QP teacher/label/action、\(u_{\text{ref}}\)、expert action、projection、CBF filter；
- V5/V4 的权重、数据、checkpoint、warm-start；
- 当前 `data_v5h0` 数据和 `h_mech` h checkpoint；
- 任何按 official h 的 h-swap 调出来的 margin/offset/scale；
- 把 V5 结果作为 V6 主表 baseline 或主证据。

### 0.3 通用设计的判定

一个组件只要同时满足以下条件，就可以保留或重实现为 V6 自有模块：

1. 不需要 official h 的权重、值、梯度、标签或校准；
2. 不需要 V5/V4 的权重或数据；
3. 不依赖 QP、expert action、\(u_{\text{ref}}\) 或 projection；
4. 可以只用 V6 自采状态 + 环境安全标签 + 通用优化训练；
5. 可以独立测试和消融。

**代码可以复用，产物必须独立。**  
例如：MAPPO/PPO 代码、图网络结构、cost critic、correction head 可以复用；
但 actor 权重、h 权重、训练数据、checkpoint 必须是 V6 从零产生的。

主路径的 policy/loss 已复制为 V6-owned 的 `nav/v6/policy.py`，
`nav/v6/training.py` 不再 runtime import `algos.safe_policy`；V5 代码只作为
历史参考，不是 V6 运行时依赖。

### 0.4 证据规则

- \(h(s)\) 的价值必须用 **MAPPO/PPO 训练结果** 证明：h-on vs h-off，matched seeds/episodes；
- h-swap 只作为诊断，不作为主结果；
- h-swap 只能比较 V6 h 与 h=0，不能拿 official h 作对照或调参目标；
- 单 seed 只做 go/no-go，不做 superiority 结论；
- 不声称硬 CBF 保证，只报告经验安全。

---

## 1. 问题与符号

多智能体导航场景包含 \(N\) 个 agents、静态障碍物和目标点。每个 agent 观测局部图，执行二维加速度动作。环境返回：

- 任务 reward / reach；
- collision / unsafe mask；
- safety 指标；
- episode 终止信号。

记 agent \(i\) 的状态为 \(s_i\in\mathbb{R}^4\)，联合状态为 \(s\)，动作为 \(u_i\in\mathbb{R}^2\)。V6 学习一个安全函数

\[
h_\phi(s): \mathbb{R}^{N\times 4}\rightarrow\mathbb{R}^N
\]

以及 actor

\[
\pi_\theta(u\mid o,h_\phi(s),\nabla_s h_\phi(s)).
\]

\(h_\phi\) 只接收状态/图，不接收 task reward、\(V_T\)、QP 标签或 official h。

**安全语义：**

- \(h>0\)：V6 认为当前状态安全；
- \(h<0\)：V6 认为当前状态不安全；
- 阈值/offset 在 V6 validation 上校准，不复制 official h 的工作点。

---

## 2. V6 h(s) 的数据

### 2.1 数据来源

主 h 数据由 V6 自己采集：

```text
V6 random policy rollouts
        +
near-boundary augmentation
        ↓
V6 h training set / val / test
```

默认使用随机行为策略，原因是：

- 不依赖 V5 权重；
- 不依赖 V5 h=0 数据；
- 不需要 official h / QP / expert；
- 与“V6 能独立从零完成”的要求一致。

后续可选：用 V6 自己从零训练的 behavior policy 补充覆盖，但必须带 manifest，且不能是 V5/V4 checkpoint。

### 2.2 数据目录

```text
outputs/v6/independent/data/
  train_data.npz
  val_data.npz
  test_data.npz
  train_aug.npz
  train_merged.npz
  summary.json
  merge_summary.json
```

禁止使用：

```text
outputs/v6/h_mech/data_v5h0/
outputs/v6/h_data/            # 旧默认目录，可能混有旧行为
```

### 2.3 标签

主标签为 forward horizon-32：

- 当前 unsafe / collision：负；
- 未来 32 步内没有 unsafe：正；
- 32 步内会 unsafe 但当前不 unsafe：忽略；
- terminal unsafe 事件用于关闭前向窗口。

另计算：

- instantaneous unsafe 标签，仅作消融；
- \(h_{\text{dot}}\) mask：只对 V6 自己的有效 transition 启用，terminal transition 排除。

标签只来自环境 collision/unsafe 语义，不使用：

- official h 的值/梯度；
- QP 的 \(u_{\text{ref}}\)；
- expert action；
- task return。

### 2.4 数据划分与可复现性

- 按 episode 划分 train/val/test，避免同轨迹泄漏；
- 固定采集 seed、episode 数、steps；
- 每份数据保存 `summary.json` 和 manifest：行为策略、seed、标签定义、代码版本、配置 hash；
- 所有数据由 V6 自己产生，禁止外部 `.npz`。

---

## 3. V6 h(s) 的网络与训练

### 3.1 网络

使用 V6 自有的 full-graph 安全网络：

- 输入：环境 `GraphsTuple`（agent / goal / LiDAR 节点与边）；
- 编码：GNN + MLP head；
- 输出：每个 agent 一个标量，`tanh` 到 \([-1,1]\)；
- 参数从随机初始化开始训练。

实现文件当前为 `nav/v6/official_h.py`，模块名是历史兼容名称；它**不加载** official h，也不使用 official h 的权重/校准。后续可重命名为 `v6/safety_h.py`。

### 3.2 损失

只用安全标签和 \(h_{\text{dot}}\)：

\[
\mathcal{L}
=
\mathcal{L}_{\text{safe}}
+
\lambda_{\text{unsafe}}\mathcal{L}_{\text{unsafe}}
+
\lambda_{\text{dot}}\mathcal{L}_{\text{dot}}.
\]

- safe：\(\max(0, m-h)\)；
- unsafe：\(\max(0, h+m)\)；
- \(h_{\text{dot}}\)：沿 V6 transition 的 barrier consistency；
- 不含 task reward、QP、\(u_{\text{ref}}\)、official h 蒸馏。

\(h_{\text{dot}}\) 的有限差分要 detach 当前 \(h_t\)，否则 \(1/\Delta t\) 会把 barrier 尺度压塌。行为策略不安全时只对 horizon-safe 状态启用 \(h_{\text{dot}}\)；独立主配置使用 `hdot_safe_only: true`，因为默认行为是随机策略。

### 3.3 校准

- 阈值只在 V6 validation 上选；
- 目标是 V6 自己的 safe/unsafe 分类指标（balanced accuracy / unsafe recall）；
- 不按 official h 的 h-swap 结果选 margin/offset；
- 当前 `safe_margin=0.5` 一类按 official h 调过的设置，不作为 V6 默认；
- 默认使用通用 margin（如 0.02），可在 V6 validation 上做小范围消融。

### 3.4 主实验中的 h

- 主实验：h 冻结，actor 从零训练；
- 这样可以干净归因“h 特征是否有用”；
- h/actor 交替 co-train 作为后续增强，不与主实验混合；
- h 的保存/加载必须带 V6 manifest，禁止 official h。

---

## 4. Actor 与 MAPPO

### 4.1 主配方

V6 主实验沿用 V5 的 MAPPO/PPO 口径（当前实现是共享参数 multi-agent PPO + per-agent GAE）：

```text
obs + h + ∇s h  ->  actor
```

- actor 从随机初始化开始；
- 不使用 V5 checkpoint；
- 不使用 official h；
- 不使用 QP / projection / expert action；
- cost critic、correction head 作为通用安全组件保留；
- action-space CBF loss 默认 0，只作为后续消融；
- 不接 task channel。

### 4.2 奖励

主实验使用任务 reward + 安全惩罚：

\[
r = r_{\text{task}} - \lambda_{\text{unsafe}}\cdot c.
\]

其中 \(c\) 是环境 collision/unsafe 代价。该 reward 是通用 PPO 设计，不依赖 official h 或 QP。

### 4.3 从零训练与 warm-start

- **主实验：actor 从零训练**，与 V5 主配方一致；
- 不使用 V5 warm-start；
- V6 self-warm-start 只作为后续可选实验：
  - Stage 1：从零训练 h-only actor，保存 V6 checkpoint W；
  - Stage 2：从 W 继续训练（task channel 或 co-train）；
  - 必须报告 Stage 1 是从零的，不能把 V6 self-warm-start 说成 from-scratch。

### 4.4 单 seed pilot

先做：

```text
seed0, 8192 steps, 20 episodes
h-on vs h-off
same eval seeds
```

判断：

- collision episode / safe episode 是否有明确方向；
- success/reach 是否崩塌；
- 若 h-on 无信号，先检查 h 数据、标签、校准和特征尺度；
- 单 seed 只做 go/no-go，不写 superiority。

方向明确后：

```text
3 seeds x 100 episodes
```

---

## 5. 训练协议

### Stage A：V6 h 数据

```bash
python nav/scripts/collect_v6_h_data.py \
  --config configs/v6/v6_nav_independent.yaml \
  --out outputs/v6/independent/data \
  --episodes 100 --steps 256 --seed 2026

python nav/scripts/augment_v6_h_data.py \
  --config configs/v6/v6_nav_independent.yaml \
  --data-dir outputs/v6/independent/data \
  --out outputs/v6/independent/data/train_aug.npz \
  --scenes 8000 --seed 777

python nav/scripts/merge_v6_h_data.py \
  --data-dir outputs/v6/independent/data \
  --out outputs/v6/independent/data/train_merged.npz
```

### Stage B：V6 h 从零训练

```bash
python nav/scripts/train_v6_official_h.py \
  --config configs/v6/v6_nav_independent.yaml \
  --train-data outputs/v6/independent/data/train_merged.npz \
  --val-data outputs/v6/independent/data/val_data.npz \
  --test-data outputs/v6/independent/data/test_data.npz \
  --out outputs/v6/independent/h --tag v6_h
```

产物：`outputs/v6/independent/h/v6_h.pkl`。

### Stage C：from-scratch actor

h-on：

```bash
python nav/scripts/train_v6.py \
  --config configs/v6/v6_nav_independent.yaml \
  --h-checkpoint outputs/v6/independent/h/v6_h.pkl \
  --seed 0 --total-steps 8192 --eval-episodes 20 \
  --out outputs/v6/independent/runs/seed0_h
```

h-off 对照：

```bash
python nav/scripts/train_v6.py \
  --config configs/v6/v6_nav_independent.yaml \
  --h-checkpoint outputs/v6/independent/h/v6_h.pkl \
  --no-use-h-features \
  --seed 0 --total-steps 8192 --eval-episodes 20 \
  --out outputs/v6/independent/runs/seed0_no_h
```

### Stage D：V6 self-warm-start（可选，后置）

只在 Stage C 出现稳定性问题时做：

- 保存 V6 Stage 1 checkpoint；
- 从 V6 checkpoint 继续；
- 不与主实验混合；
- 不放宽“主实验 from scratch”的 claim。

### Stage E：task channel（后置）

h-only 达到目标后再接 \(V_T/A_T/\nabla_s V_T\)。

---

## 6. 实验协议

### 6.1 主对照

| Arm | h 输入 | actor 初始化 | task channel |
|---|---|---|---|
| `scratch_h` | V6 h + ∇h | 随机 | 无 |
| `scratch_no_h` | 零 | 随机 | 无 |
| `selfwarm_h`（后置） | V6 h + ∇h | V6 Stage 1 checkpoint | 可选 |
| `selfwarm_no_h`（后置） | 零 | V6 Stage 1 checkpoint | 可选 |

主 claim 先看 `scratch_h` vs `scratch_no_h`。

### 6.2 指标

主指标：

- collision episode rate；
- safe episode rate。

次指标：

- success；
- reach；
- episode return；
- h 的离线分类指标（balanced accuracy / unsafe recall / AUROC），仅作诊断。

### 6.3 目标

安全优先，参考已归档的 V5 16k MAPPO + official h 记录：

- environment safety（历史 JSON 字段 `official_safety`，与环境碰撞指标对应，**不是** official h）约 0.95 量级；
- safe episode 约 0.70 量级；
- success 不崩塌。

V5 8k 的 success 可能更高，但 V6 以安全优先，采用 V5 16k safety 记录作参考。
V5 数字只作为记录目标，不重新运行 official h；V6 不把 V5 作为主表 baseline。

### 6.4 Go / No-Go

**单 seed pilot：**

- h-on 的 collision/safe episode 方向明显好于 h-off；
- success/reach 不崩；
- 若方向不对，先诊断 h 质量和数据，不直接上 3 seeds。

**正式实验：**

- 3 seeds × 100 episodes；
- formal actor training 16384 steps（pilot 8192）；
- 报告 mean ± std；
- h-on 相对 h-off 有稳定安全收益；
- 接近 V5 16k safety 目标；
- 没有任何单 seed 结论。

### 6.5 禁止的比较

- 不运行 official h / V5 baseline；
- 不用 official h 做 h-swap 调参；
- 不把 V5 warm-start 结果作为 V6 主结果；
- 不用 h-swap 代替 PPO 训练证据。

### 6.6 可复现协议（2026-10-23）

正式 h-only 复现采用：

```text
训练 seeds: 0, 1, 2
actor: from scratch
h: 冻结 V6 h
task channel: 关闭
训练步数: 16,384
评估 episodes: 100
评估 seeds: 10000..10099（所有训练 seed 共用）
报告 checkpoint: policy_final.pth（预先指定，不按 20ep 结果挑 best）
```

`nav/scripts/run_v6_formal_3seed.sh` 负责：

- 固定训练/评估 seed；
- 保存 `config.yaml`、训练日志和 100ep 评估 JSON；
- 生成 `manifest.json`，记录 h/config/data 的 SHA-256、Python/Torch/JAX 版本和运行参数；
- 用 `summarize_v6_formal.py` 汇总 mean ± std；
- 支持断点续跑；seed0 可显式复用已有单 seed 结果，复用来源写入 manifest。

主指标为 `safe_episode` / `collision_episode`。JSON 中的
`official_safety` / `official_success` / `official_reach` 是 per-agent
any-time 的历史字段名，不是 episode-level 安全率。

### 6.7 安全性优化 screen

在不动 h、不接 task、不做 QP/projection 的前提下，先做 20ep 安全性
screen，再只把有希望的变体提升到 100ep：

| variant | 机制 | 关键参数 |
|---|---|---|
| `cost` | 更强碰撞代价 | `unsafe_penalty=40`, `cost_actor_coef=1.0` |
| `cost20` | 中等碰撞代价（若 40 不稳） | `unsafe_penalty=20`, `cost_actor_coef=1.0` |
| `barrier` | V6 h 的稠密安全代价 | `h_barrier_coef=1.0`, margin `0.05` |
| `cbf` | 动作空间软 CBF 正则 | `cbf_coef=0.2` |
| `hscale` | 放大 actor 的 h/∇h 输入 | `h.feature_scale=3.0`，h 语义不变 |
| `ft_cost` / `ft_barrier` | 从 h-only final 做安全 fine-tune | 8,192 步 |

所有变体都：

- 使用同一个冻结 V6 h；
- 不更新 h；
- 不接收 task channel；
- 不使用 official h / QP / expert action；
- 与 h-only 使用相同训练 seed、评估 seed 和 episode 数。

提升标准：`safe_episode` 提升且 `official_success` 不明显崩塌；否则不进入
100ep，也不进入 3-seed 正式实验。

---

## 7. Task channel（后置）

task channel 的目标是后续优化任务完成能力，不是本轮基础目标。

要求：

- \(V_T\) 只学 task return，不接收 safety cost；
- \(V_T\) 由 V6 自采 task return 训练；
- \(A_T\) 只作为相对任务标量，不进入安全 gate；
- \(\nabla_s V_T\) 与 h/∇h 分离；
- task trunk 与 safety/correction trunk 隔离；
- 不共享 h 参数；
- 不使用 V5 的 task/value 结果；
- h-only 未成立前不接 task。

---

## 8. Ant 场景（已完成 3-seed 10M 验证）

核心思想是通用的：

> 用 V6 自己采集的安全数据训练 \(h(s)\)，向 MAPPO actor 提供安全特征，替代 expert \(u_{\text{ref}}\)/QP。

但导航和 Ant 不要求相同 h 结构或相同 MAPPO 细节：

- Ant 的状态/图/安全标签单独定义；
- Ant h 单独采集、单独训练；
- Ant actor 单独从零训练；
- 不把导航 h 迁移到 Ant；
- 不把导航结果当作 Ant 结果。

---

## 9. 实现文件

```text
nav/configs/v6/v6_nav_independent.yaml   # 独立主配置
nav/scripts/collect_v6_h_data.py         # V6 自采 h 数据
nav/scripts/augment_v6_h_data.py         # near-boundary 增广
nav/scripts/merge_v6_h_data.py           # 合并数据
nav/scripts/train_v6_official_h.py       # V6 h 从零训练
nav/scripts/train_v6.py                  # from-scratch actor
nav/scripts/evaluate_v6.py               # V6 h/actor 评估
nav/scripts/run_v6_protocol.sh           # 单 seed / 3 seed 协议
nav/scripts/check_v6_independence.py     # 独立性静态检查
nav/v6/official_h.py                     # V6 graph safety net（历史文件名）
nav/v6/policy.py                         # V6-owned QP-free policy/loss copy
nav/v6/training.py                       # rollout / PPO / eval
nav/v6/provenance.py                     # 禁止路径守卫
```

`nav/configs/v5/`、`nav/algos/safe_policy/` 等旧代码只作为通用设计参考；
主路径不加载 V5 产物。

---

## 10. 风险与开放问题

### 10.1 风险

- **h 数据覆盖不足**：随机策略可能缺少 near-boundary 状态；用增广和后续 V6 behavior 数据补。
- **h 校准耦合**：任何按 official h 调的 margin/offset 都不能用；必须 V6-only 校准。
- **from-scratch actor 不稳定**：旧 V5-warm-start/task pilot 出现过安全与任务相互干扰，不能直接外推到 h-only from-scratch；需要单独验证。
- **单 seed 噪声**：只做 go/no-go，不能下结论。
- **计算成本**：3 seeds × 100 episodes 需要预算；先单 seed。
- **无硬保证**：h 是学习的安全分数，不声称 CBF 硬约束。

### 10.2 开放问题

- 随机 + 增广是否足以训练出接近 V5 official h 的 V6 h？
- h 的阈值/margin 在 V6-only 校准下是否足够稳定？
- h-on 的收益来自 h 特征、∇h 特征，还是两者？
- 是否需要 V6 self-warm-start 才能稳定 from-scratch actor？
- task channel 在 h-only 成立后是否还有增益？
- Ant 场景是否需要不同的 h 结构和安全标签？

---

## 11. Legacy 结果处理

以下内容全部视为 legacy diagnostic，不进入 V6 主表：

- V5 warm-start 的 V6-0 / V6-3 结果；
- `data_v5h0` 训练的 h checkpoint；
- `h_mech` 下的 h-swap 和 actor 结果；
- 任何使用 official h 或 V5 checkpoint 的 eval。

处理方式：

- 归档到 `archive/v5_coupled/`；
- 文档只保留“历史诊断”说明，不作为证据；
- 主路径只读 `outputs/v6/independent/`。

---

## 12. 总结

V6 的独立主线是：

\[
\boxed{
\text{V6 自采数据}
\rightarrow
h_\phi \text{ 从零训练}
\rightarrow
\text{from-scratch MAPPO}
(h_\phi,\nabla_s h_\phi)
}
\]

- h 只学安全；
- actor 只从任务 reward 学；
- 不接 official h / QP / \(u_{\text{ref}}\)；
- 不接 V5 权重/数据；
- 主实验从零 actor；
- 已完成单 seed、navigation 3 seeds 和 Ant 3 seeds × 10M；
- task channel 仍未开启，Ant self-warm-start 未使用。

**一句话：V6 不复制 official h，而是用 V6 自己的数据从零学习一个可替代 official h 的安全 \(h(s)\)，并证明它能让从零训练的 MAPPO 更安全。**
