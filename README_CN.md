# 双AI生成-审核架构：更安全的共情心理健康支持

[English](README.md) | 中文

本仓库包含论文"双AI Maker-Checker架构在心理健康共情支持中的安全性研究"的代码、数据与评估流水线。核心研究问题：**角色分离**（共情生成 vs 安全审查）能否在不显著损失温度感的前提下提升AI心理支持的安全性？

---

## 研究问题

单Agent LLM设计让一个模型同时承担多个互相冲突的目标：温暖共情、安全检测、风险识别、边界维护、不确定性沟通。我们发现，这种**角色冲突**导致30%的高风险场景中出现升级遗漏。我们的**Maker-Checker架构**通过结构性分工解决这一问题。

## 三个实验条件

| 条件 | 描述 |
|------|------|
| **A：单Agent** | 一次性生成共情回复，无安全审查 |
| **B：隐式审核** | Maker生成 → Checker审核 → 用户只看到最终回复 |
| **C：显式审核** | 同B，但用户可看到安全审核标识和可展开说明 |

## 基准测试

90个心理支持场景，按风险等级分层（低30 / 中30 / 高30），话题涵盖工作压力、孤独感、自伤、自杀意念、物质滥用等。每个场景 × 3条件 = 270个评估输出。

## 评估方案

- **6个维度**：情感识别、验证与温暖、实用性、安全性、边界遵守、升级适当性（各1-5分）
- **2个复合指标**：共情复合分（情感+验证）、安全复合分（安全+边界+升级）
- **LLM-as-a-Judge** 结构化评估（DeepSeek-Chat）
- **鲁棒性验证**：严格第二裁判 + 3人格多裁判交叉验证

## 核心发现

| 复合指标 | A：单Agent | B：隐式审核 | C：显式审核 |
|----------|:-:|:-:|:-:|
| 共情 | **5.00** | 4.76 | 4.76 |
| 安全 | 4.71 | 4.83 | **4.87** |
| 实用性 | 3.83 | **4.03** | **4.03** |

- **A在共情上优于B、C**（p < .005，Holm校正）——单Agent更温暖
- **C在安全上优于A**——显式审核最安全，尤其在高风险场景
- **Checker = 风险敏感安全网**：低风险100%通过，高风险63-70%升级，误报率仅1.7%
- **权衡在5种裁判变体中稳健一致**（原始、严格、严格/中等/宽松人格）

## 计划中的用户研究

基于情景的实验（N约36），测量感知共情、温暖、安全、信任、透明度与校准依赖度。核心假设：显式审核促进**适当依赖**（适度信任 + 高求助意愿），而非盲目信任。详见 `docs/user_study_design.md`。

---

## 仓库结构

```
generation/                      # 回复生成（3个条件）
  run_single.py                  # 条件A：单Agent
  run_double_hidden.py           # 条件B：隐式审核
  run_double_visible.py          # 条件C：显式审核
prompts/
  maker/                         # Maker系统提示词
  checker/                       # Checker系统提示词
  visible_checker/               # 显式审核指示模板
checker/                         # Checker策略层
  policy_rules.py                # 通过/修改/弃权/升级逻辑
  checker_schema.py              # 结构化输出schema
data/
  scenarios/benchmark.jsonl      # 90场景基准测试
results/
  offline_eval_v2_final/         # 冻结的评估结果
    scenarios.csv                # 基准场景
    outputs_A.jsonl              # 单Agent输出
    outputs_B_hidden.jsonl       # 隐式审核输出
    outputs_C_visible.jsonl      # 显式审核输出
    judge_scores_main.csv        # 主裁判评分（270行）
    statistics.json              # 完整统计分析
    composite_stats.json         # 复合指标数值
    figures/                     # 论文图（PDF + PNG）
    tables/                      # LaTeX表格
    metadata.yaml                # 冻结元数据
  generate_paper_assets.py       # 重新生成图表
  reproduce_all.py               # 一键复现脚本
docs/
  paper_results.md               # 完整论文初稿
  user_study_design.md           # 用户研究方案
  survey_instrument.md           # 问卷工具
  irb_consent.md                 # IRB知情同意书
  appendix_qualitative.md        # 6个定性案例
  appendix_materials.md          # 提示词、量表、schema
evaluation/
  offline_metrics.py             # 指标计算
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 设置API密钥
export DEEPSEEK_API_KEY=sk-...

# 3. 生成回复（3条件 x 90场景）
python generation/run_single.py
python generation/run_double_hidden.py
python generation/run_double_visible.py

# 4. 运行LLM裁判
python results/run_statistics.py

# 5. 生成论文图表
python results/generate_paper_assets.py
```

## 可复现性

所有结果可从冻结目录 `results/offline_eval_v2_final/` 复现。`metadata.yaml` 记录了模型版本、提示词、commit hash和生成参数。完整文件清单见 `results/offline_eval_v2_final/README.md`。

## 历史组件

本仓库还包含早期的共情对话生成工作（GPT-2微调、Chain-of-Empathy），作为当前Maker-Checker研究的基线开发。这些组件位于 `src/models/`、`Model_Baseline.py`、`Train_Baseline.py` 等文件中，不属于当前论文但保留供参考。
