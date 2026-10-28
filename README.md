# Safe-MARL v6

**独立 V6**：用 V6 自己采集数据、从零训练的 \(h(s)\) 替代 official h / 专家
\(u_{\text{ref}}\) / QP 设计，再把 \(h,\nabla_s h\) 作为安全特征交给从零训练的
MAPPO/PPO actor。当前主结果包括 h-only 基线、预测式 h 任务 gate，以及 Ant
和 navigation 两个场景的 3-seed 验证。

## 文档入口

- [当前状态与结论](outputs/v6/STATUS.md)
- [输出目录索引与 legacy 分类](outputs/v6/README.md)
- [完整实验记录](outputs/v6/V6_COMPLETE_RECORD.md)
- [待办与边界](outputs/v6/TODO_V6.md)
- [导航运行手册](nav/README_V6.md)
- [导航实验记录索引](outputs/v6/independent/nav/README.md)
- [导航基线对比](outputs/v6/independent/nav/BASELINE_COMPARISON.md)
- [技术总结与实验设计](intro.md)（中文长文）
- [提案大纲](docs/PROJECT_PROPOSAL_V6.md)（历史设计稿）
- [详细设计](docs/DETAILED_DESIGN_V6.md)（历史设计稿）

## 仓库结构

| 路径 | 内容 |
|---|---|
| `nav/v6/` | V6 主实现：安全 h、策略、训练与环境适配 |
| `nav/scripts/` | 数据采集、h/actor 训练、评估与汇总入口 |
| `nav/configs/v6/` | V6 navigation / Ant 实验配置 |
| `nav/tests/` | 单元测试与独立性边界测试 |
| `outputs/v6/` | 实验记录、聚合 JSON 与 artifact 索引 |
| `docs/` | V6 历史提案与详细设计 |
| `third_party/gcbfplus/`、`nav/vendor/onpolicy/` | vendored 官方/基线代码（保留原 LICENSE） |

## 环境与测试

V6 navigation 主路径只需要核心依赖：

```bash
python -m pip install -r nav/requirements-core.txt
pytest
```

`pytest.ini` 已把 `nav/` 加入测试导入路径；在仓库根目录直接运行 `pytest`
即可。静态独立性检查可单独运行：

```bash
python nav/scripts/check_v6_independence.py
```

官方基线依赖见 `nav/requirements-v5.txt` 与
`nav/requirements-baselines.txt`。Ant 分支还需要单独的
`mappo_lag` Python 环境和 MuJoCo 2.1.0；用 `SAFE_MAMUJOCO_ROOT` 指向该
上游仓库根目录（默认尝试同级 `Safe-MARL-proposal_v2`）。详细命令见
`nav/README_V6_ANT.md`。

## 当前状态

### Ant

预测式 h + safety-gated forward 通过 3 seeds × 10M retention：

- deterministic：forward `3771.51`（91.6% retention），safe `0.97`，
  cost fraction `0.0012`；matched cost-only safe `0.063`。
- stochastic：forward `3010.73`（93.9% retention），safe `0.84`，
  cost fraction `0.0020`；matched cost-only safe `0.020`。
- 每个 gate seed 都安全（det `0.97/0.95/0.99`），每个 cost-only seed 都不安全
  （det `0.10/0.08/0.01`）。

记录：`outputs/v6/independent/ant/RETENTION_10M.md`。

补充诊断（已关闭，null）：在现有 seed-0 10M h-only base 上做 1M 的 1-D
\(A_T\) warm-start，并与 matched no-V_T 1M control 对比。V_T gate 通过
（val EV `0.4838`、test Pearson `0.6597`），control 有效（safe `0.96`），
但 warm-start 未在任务上超过 control（det forward `3960.83` vs `4014.03`，
return `4574.53` vs `4630.37`；safety 均为 `0.96`），
`vt_transfer_supported=false`，该线停止。单 seed，仅诊断；promoted 3-seed
结论不变。记录：`outputs/v6/independent/ant/STAGE_ANT_VT_TRANSFER.md`。

### Navigation

Repaired Stage-B 预测式 h gate 通过预注册 3-seed guardrails（16,384 steps/seed，
100ep deterministic，eval seed 10000）：

| metric | h-only 3-seed | repaired 3-seed |
|---|---:|---:|
| safe_episode | 0.450 | **0.693 ± 0.168** |
| official_safety | 0.8783 | **0.9417 ± 0.0378** |
| official_success | 0.4067 | **0.4100 ± 0.0583** |
| official_reach | 0.4450 | **0.4296 ± 0.0695** |
| episode_return | 61.73 | **66.85 ± 5.26** |

安全性接近 CBF-QP（`0.979`）和 V5 full（`0.950`），但任务成功率仍低于
V5 full（`0.512`）、CBF-QP（`0.781`）和 GCBF+（`0.861`）。`n=3`，seed-level
差异尚未达到统计显著；安全增益主要由 seeds 0/2 贡献。

**P0 修正（2026-10-27）**：补齐的 original-h + broad-gate 3-seed 行为
`safe 0.6433 / official_safety 0.9325 / success 0.4263 / reach 0.4525 /
return 70.33`；P0 − h-only 为 `+19.3pp` safe，A − P0 仅 `+5.0pp` safe
（per-seed `+14/+2/−1pp`），且 success `−1.6pp`、return `−4.9%`，A 并未在
任务上支配 P0。P0 与 A 的 gate scope/threshold 也不同，所以“主要靠 gate”
只适用于 original h + broad gate 这一行，不是 repaired-h 的通用归因。

**P1 repaired-h 2×2（2026-10-27 完成）**：

| cell | h 输入 | gate | safe | official safety | success | return |
|---|:---:|:---:|---:|---:|---:|---:|
| A repaired h + gate | on | on | 0.6933 | 0.9417 | 0.4100 | 66.85 |
| B repaired h, gate off | on | off | 0.6333 | 0.9267 | 0.3229 | 55.67 |
| C no h input, gate on | off | on | 0.0200 | 0.4987 | 0.0367 | −6.63 |
| D no h input, gate off | off | off | 0.0333 | 0.5479 | 0.0367 | 7.21 |

清晰对比：B − H0 `+18.3pp` safe 但 success `−8.4pp`；A − B `+6.0pp` safe /
`+8.7pp` success；A − C、B − D 分别 `+67.3/+60.0pp` safe；C − D `−1.3pp`
safe、success 持平。结论：在 16k from-scratch 预算下 actor h 输入是必需的
（C/D 在全部 6 个 seed 上崩溃，D 与无 h 固定惩罚基线一致），gate 离开 h
输入无效；在 repaired-h 2×2 内只有 repaired h + gate（A）同时保住安全与任务
（P0 是独立的 original-h broad-gate 行）。`+24.3pp` 可分解为
repaired h（gate off）`+18.3pp` + progress gate `+6.0pp`；`n=3`，仅作描述性
归因，不主张策略级显著优越。

记录：`outputs/v6/independent/nav/STAGE_B_REPAIR_MULTISEED.md` 和
`outputs/v6/independent/nav/BASELINE_COMPARISON.md`。

2026-10-27 补齐了两条 matched 16k 碰撞感知基线：固定
`collision_penalty=10` 的 MAPPO（safe `0.033`，official safety `0.519`，
success `0.014`，return `-30.35`）和官方 collision cost 的 MAPPO-Lag 16k
（safe `0.000`，official safety `0.322`，success `0.086`，return `39.55`）。
两者都不使用 h/QP，V6 Stage-B 在同预算下仍为 `0.693 / 0.9417 / 0.410 /
66.85`。旧的 MAPPO-Lag 100k 只作为宽松参考，不再当作 matched 行。
记录：`outputs/v6/independent/nav_baselines_16k/README.md`。

### 尚未完成

- Navigation task channel / \(V_T\)：T0w 4k warm-start 的严格
  safety-first 非劣规则未通过（差 1 episode）；task-first 视角下是
  task-heavy Pareto 点（success `+29.9pp`，return `+92.2`，safe `-4pp`），
  但仍是单 seed / 4k warm-start，暂不作为主 claim；3-seed from-scratch
  复现待定。
- 可选的 navigation 100ep stochastic evaluation。
- 可选的 Ant 10M 2M/5M checkpoint retention 评估。

## 独立性边界

允许：

- 官方多智能体导航 / Ant 场景、动力学、观测、collision/unsafe 标签和官方
  task reward/cost；
- 通用网络、PPO/MAPPO、GAE、reward 和训练代码设计；
- V6 自己采集的数据、自己训练的 h/actor、自己产生的 warm-start。

禁止：

- official h / `cbf.pkl` / 预训练 h 权重、值、梯度、teacher label；
- QP 标签/动作、专家 \(u_{\text{ref}}\)、projection；
- V5/V4 权重、数据、checkpoint、warm-start；
- 当前 `h_mech` h 和 `data_v5h0`；
- 按 official h 的 h-swap 校准的 margin/offset。

`outputs/v6/` 根目录下的 `co_train_*`、`frozen_v6_*`、`h_mech`、
`official_h*` 等是早期 coupled/legacy 诊断，不是 V6 主证据。主证据在
`outputs/v6/independent/`。

## 复现

```bash
# 0) 静态独立性检查
python nav/scripts/check_v6_independence.py

# 1) 重新聚合已有 navigation 3-seed 正式评估（不重训）
python nav/scripts/summarize_v6_formal.py \
  --dir outputs/v6/independent/nav_stageb_repair_16k \
  --out outputs/v6/independent/nav_stageb_repair_16k/aggregate_100ep.json

# 2) 重新评估一个 promoted checkpoint（示例：seed 0）
python nav/scripts/evaluate_v6.py \
  --checkpoint outputs/v6/independent/nav_stageb_repair_16k/seed0/policy_final.pth \
  --h-checkpoint outputs/v6/independent/h_stageb_repair/v6_h_stageb_repair.pkl \
  --num-episodes 100 --seed 10000 --device cuda \
  --out outputs/v6/independent/nav_stageb_repair_16k/seed0/eval_100ep_rerun.json
```

完整 artifact map、hash 和 claim boundary 见
`outputs/v6/STATUS.md`。

## 数据与检查点

Git 仓库只提交代码、配置、Markdown/JSON 记录。`outputs/` 下的 `*.npz`
数据、`*.pth`/`*.pkl` 权重和 `*.log` 日志约 3.3 GB，已由 `.gitignore`
排除，但本地文件不会被删除或改写。重跑时使用 `nav/README_V6.md` 的脚本；
promoted checkpoint 的 SHA-256 仍保留在 `outputs/v6/STATUS.md`。

## 第三方代码

`nav/vendor/onpolicy/` 和 `third_party/gcbfplus/` 是 vendored 的 MIT
许可代码，各自目录保留原始 `LICENSE`；复用或再分发时请保留版权声明。
