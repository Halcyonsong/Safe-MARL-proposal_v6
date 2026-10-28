# Safe-MARL V6 提案

## 用独立学习的 \(h(s)\) 替代 expert \(u_{\text{ref}}\) / QP 的安全 MAPPO

**状态：历史提案稿（2026-10-22）。** 本文保留最初的研究设计和实验计划，
不再是当前进度记录。当前结果、promoted/archived artifact 和待办见
[`outputs/v6/STATUS.md`](../outputs/v6/STATUS.md) 与
[`outputs/v6/V6_COMPLETE_RECORD.md`](../outputs/v6/V6_COMPLETE_RECORD.md)。

---

## 0. 一句话

> V6 用自己采集的数据、从零训练的安全函数 \(h(s)\)，替代 official h / 专家
> \(u_{\text{ref}}\) / QP 设计，把 \(h,\nabla_s h\) 作为安全特征交给从零训练的
> MAPPO actor；先证明 h-only 能明显提升安全，再接 task channel。

---

## 1. 背景与动机

### 1.1 问题

多智能体导航和 Ant 等安全控制任务通常依赖专家设计的
\(u_{\text{ref}}\)、QP 投影、CBF filter 或 official h。 这些设计在以下情况可能不可用：

- 难以获得可靠的专家动作或 QP 标签；
- 场景动力学/约束变化后，专家设计需要重做；
- 部署时不允许求解 QP 或访问 official h；
- 希望用强化学习直接学习安全机制，而不是依赖手工控制器。

### 1.2 核心思想

用一个可学习的安全函数 \(h(s)\) 替代 expert safety mechanism：

\[
h_\phi(s): \text{state/graph}\rightarrow\mathbb{R}
\]

- \(h>0\) 表示安全，\(h<0\) 表示不安全；
- actor 接收 \(h,\nabla_s h\) 作为安全特征；
- h 只从安全标签和 V6 自己的 transition 学习；
- actor 仍由任务 reward 训练；
- 默认不做 QP/projection；
- 先只接 h(s)，task channel 后置。

### 1.3 V5 经验与边界

V5 的经验说明：\(h,\nabla_s h\) 作为 actor 输入可以显著影响安全。 但 V5 使用的是 official h，因此：

- V5 可以作为**经验/设计参考**；
- V5 的权重、数据、checkpoint、warm-start 不能作为 V6 产物；
- V6 必须自己采集数据、从零训练 h 和 actor；
- V5 的 official-h 结果只能作为已归档目标，不能作为 V6 主表 baseline，也不能重新运行 official h。

---

## 2. 独立性契约

### 2.1 允许

- 官方多智能体导航场景：动力学、观测、障碍物、目标、collision/unsafe 标签；
- 通用代码/设计：MAPPO/PPO、GAE、actor-critic、GNN/MLP、reward、训练配方；
- V6 自己采集的数据、自己训练的 h/actor、自己产生的 self-warm-start；
- V5 的**经验性结论**（h/∇h 可能提升安全），仅作动机。

### 2.2 禁止

- official h / `cbf.pkl` / 预训练 h 权重；
- official h 的值、梯度、teacher label、蒸馏目标；
- QP 标签/动作、\(u_{\text{ref}}\)、expert action、projection、CBF filter；
- V5/V4 权重、数据、checkpoint、warm-start；
- 当前 `data_v5h0` 和 `h_mech` checkpoint；
- 按 official h 的 h-swap 校准的 margin/offset/scale；
- 把 V5 结果作为 V6 主证据。

### 2.3 判定标准

一个组件可以保留，当且仅当：

1. 只用 V6 自采状态 + 环境安全标签 + 通用优化即可训练；
2. 不需要 official h 的权重/值/梯度/标签/校准；
3. 不需要 QP、\(u_{\text{ref}}\)、expert action；
4. 不需要 V5/V4 权重或数据；
5. 可以独立测试和消融。

**代码可以复用，产物必须独立。**

---

## 3. 研究问题与假设

### RQ1：V6 能否用自采数据从零训练出可用的 h(s)？

H1：只用环境 collision/unsafe 标签 + \(h_{\text{dot}}\)，V6 h 可以达到可用的离线安全分类指标，并能在 actor 训练中提供有效安全特征。

### RQ2：h(s) 能否让从零训练的 MAPPO actor 明显更安全？

H2：在 matched seeds/episodes 下，`h-on` 的 collision episode / safe episode 明显优于 `h-off`；success/reach 不崩塌。

### RQ3：V6 h 能否接近 V5 + official h 的安全水平？

H3：h-only from-scratch 的安全指标接近已归档的 V5 16k safety 目标（environment safety 约 0.95 量级、safe episode 约 0.70 量级；历史字段名 `official_safety` 指环境安全指标，不是 official h），同时不要求完全复制 V5 的 success。

### RQ4：V6 self-warm-start 是否有帮助？

H4：如果 from-scratch actor 不稳定，V6 Stage 1 自产 checkpoint 的 warm-start 可以改善稳定性；但主结果仍以 from-scratch 为主，warm-start 单独报告。

### RQ5：task channel 是否能在 h-only 成立后进一步提升任务？

H5：只有在 h-only 安全和任务都不崩的前提下，独立的 \(V_T/A_T/\nabla_s V_T\) 才可能带来任务收益；否则不接 task。

---

## 4. 方法

### 4.1 总览

```text
V6 random rollouts + near-boundary augmentation
        ↓
V6 h(s) from scratch (safety-only)
        ↓
freeze h
        ↓
from-scratch MAPPO actor
  input: obs + h + ∇s h
        ↓
h-on vs h-off evaluation
```

### 4.2 h(s) 数据

- 行为策略：默认随机；后续可选 V6 自训 behavior policy；
- 标签：forward horizon-32 safe/unsafe + V6 transition 的 \(h_{\text{dot}}\)；
- 不使用 official h、QP、\(u_{\text{ref}}\)、expert action、task reward；
- 按 episode 划分 train/val/test；
- 所有数据带 manifest，禁止 V5 数据。

### 4.3 h(s) 网络与训练

- full-graph GNN + MLP head + tanh 标量输出；
- 参数从零训练；
- 损失：safe/unsafe margin + \(h_{\text{dot}}\)；
- 阈值/offset 只在 V6 validation 上校准；
- 不使用 official h 的 work point、margin、offset、scale；
- 主实验冻结 h；co-train 后置。

### 4.4 Actor / MAPPO

- 复用 V5 的 MAPPO/PPO 口径（共享参数 multi-agent PPO + per-agent GAE）；
- policy/loss 已复制为 V6-owned 的 `nav/v6/policy.py`，主路径不 runtime import
  V5 `algos.safe_policy`；
- actor 从零训练，不加载 V5 checkpoint；
- 输入 \(obs,h,\nabla_s h\)；
- cost critic / correction head 作为通用安全组件保留；
- 不做 QP/projection；
- action-space CBF loss 默认 0，只作后续消融；
- 主实验不接 task channel。

### 4.5 Warm-start

- V5 warm-start 禁止；
- V6 self-warm-start 后置、可选：
  - Stage 1：from-scratch h-only actor → V6 checkpoint W；
  - Stage 2：从 W 继续；
  - 必须明确“Stage 1 从零”，不能把 self-warm-start 称为 from-scratch。

---

## 5. 实验计划

### 5.1 单 seed 快速验证

```text
seed0, 8192 steps, 20 episodes
scratch_h vs scratch_no_h
same eval seeds
```

判断：

- collision episode / safe episode 是否有明确方向；
- success/reach 是否崩塌；
- 若 h-on 无信号，先诊断 h 数据和校准；
- 单 seed 只做 go/no-go。

### 5.2 正式实验

方向明确后：

```text
3 seeds × 100 episodes; formal actor training 16384 steps
```

主指标：

- collision episode rate；
- safe episode rate。

次指标：

- success；
- reach；
- episode return；
- h 离线指标（仅诊断）。

目标：

- h-on 明显优于 h-off；
- 接近 V5 16k safety 目标（V5 8k 的 success 可能更高，但 V6 安全优先）；
- success 不崩塌；
- 不运行 official h / V5 baseline，只使用已归档数字作为记录目标。

### 5.3 变体

| 变体 | h 输入 | actor 初始化 | task |
|---|---|---|---|
| `scratch_h` | V6 h + ∇h | 随机 | 无 |
| `scratch_no_h` | 零 | 随机 | 无 |
| `selfwarm_h`（后置） | V6 h + ∇h | V6 Stage 1 | 可选 |
| `selfwarm_no_h`（后置） | 零 | V6 Stage 1 | 可选 |
| task variants（后置） | V6 h + ∇h | 主 arm | V_T/A_T |

### 5.4 Go / No-Go

- 单 seed：方向对 + 不崩 → 进入 3 seeds；
- 单 seed：无信号/崩 → 先诊断 h 数据、标签、校准、特征尺度，不直接扩大实验；
- 3 seeds：安全收益稳定 + 任务不崩 → h-only 成立；
- 否则：报告 h-only 失败，继续在 V6 内迭代，不回到 V5/official h。

---

## 6. 贡献

1. **独立 h(s)**：不依赖 official h/QP/expert，只用 V6 自采数据从零训练安全函数。
2. **安全特征替代 expert 设计**：把 \(h,\nabla_s h\) 作为 actor 特征，验证 RL 学到的安全信息能否替代专家安全机制。
3. **严格独立性协议**：明确允许/禁止清单、产物 manifest、静态检查，避免 V5/official h 二次加工。
4. **安全优先的 h-only 基线**：先证明 h-only，再接 task channel，避免任务信号掩盖安全归因。
5. **跨场景思想**：导航和 Ant 共享“自采数据 + 从零 h + MAPPO”的思想，但不强求相同结构。

---

## 7. Ant 场景

Ant 是第二个验证场景，后置。

- 复用通用思想：自采安全数据、从零 h、MAPPO actor；
- 不复用导航 h 权重；
- 不要求导航和 Ant 的 h/MAPPO 完全相同；
- Ant 的 h 数据、安全标签、actor 全部独立；
- Ant 结果不能由导航结果外推。

---

## 8. 风险与缓解

| 风险 | 缓解 |
|---|---|
| 随机 h 数据覆盖不足 | near-boundary 增广；后续 V6 behavior policy |
| h 校准带 official h 耦合 | V6 validation 重新校准；禁止 official h-swap 调参 |
| from-scratch actor 不稳定 | 先单 seed；必要时 V6 self-warm-start，单独报告 |
| 单 seed 噪声 | 只做 go/no-go；正式 3 seeds × 100 |
| 计算成本 | 先 8192/20 episodes；方向明确再扩大 |
| 无硬安全保证 | 只报告经验安全，不声称 CBF 保证 |
| 误用 legacy 产物 | 静态检查 + manifest + 主路径只读 `outputs/v6/independent` |

---

## 9. 里程碑

状态已按 2026-10-25 的实际进展更新；结果与 artifact 见
[`outputs/v6/STATUS.md`](../outputs/v6/STATUS.md)。

| 阶段 | 内容 | 状态 |
|---|---|---|
| M0 | 独立性契约、文档、配置、清理方案 | 完成 |
| M1 | V6 自采 h 数据 + 从零 h | 完成 |
| M2 | 单 seed `scratch_h` vs `scratch_no_h` | 完成：8k h-off pilot + 正式 16k h-off 3-seed（P1 D: safe `0.033` / official `0.548` / success `0.037`） |
| M3 | 方向明确后 3 seeds × 100 | 完成：h-only 3-seed + repaired Stage-B gate 3-seed + P1 repaired-h 2×2（B/C/D） |
| M4 | V6 self-warm-start（可选） | 未使用 |
| M5 | task channel | 后置，尚未开启 |
| M6 | Ant 独立验证 | 完成：predictive-h gate 3 seeds × 10M |

---

## 10. 当前不做什么（历史约束）

- 本文档本身不启动训练；实际执行状态见 `outputs/v6/STATUS.md`；
- 不使用 V5 warm-start；
- 不加载 official h / `cbf.pkl`；
- 不使用 QP/\(u_{\text{ref}}\)/expert；
- 不接 task channel；
- 不把 legacy pilot 当主结果；
- 不在单 seed 上声称 superiority。

---

## 11. 总结

> V6 的核心不是复现 official h，而是用 V6 自己采集的数据从零学习一个安全
> \(h(s)\)，并用 from-scratch MAPPO 证明它能让策略更安全。  
> 当前已完成 h-only 3-seed、repaired Stage-B gate 3-seed 和 Ant 10M
> 3-seed；task channel 仍未开启。  
> 代码可以复用，经验可以借鉴，但产物和证据必须独立。
