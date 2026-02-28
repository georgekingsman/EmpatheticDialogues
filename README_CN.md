# 共情对话评估框架 | Empathetic Dialogue Evaluation Framework

**研究方向 B：人工评分 ↔ LLM-as-a-Judge 标定**
**Direction B: Human Rating ↔ LLM-as-a-Judge Calibration**

一个可复现、可扩展的共情/支持性对话评估框架，使用**外部人工标注数据集**锚定校准 LLM-as-a-Judge，无需额外人工标注。

A reproducible, extensible framework for evaluating empathetic/supportive dialogue using **external human-anchored calibration** with LLM-as-a-judge. No additional human annotation required.

---

## 目录 | Table of Contents

- [核心问题：能否用大模型代替人工评分？| Can LLM Replace Human Scoring?](#核心问题能否用大模型代替人工评分-can-llm-replace-human-scoring)
- [整体管道流程 | Pipeline Overview](#整体管道流程--pipeline-overview)
- [文件结构详细说明 | File Structure Explained](#文件结构详细说明--file-structure-explained)
- [快速开始 | Quick Start](#快速开始--quick-start)
- [评分量表 | Evaluation Rubric](#评分量表--evaluation-rubric)
- [实验矩阵 | Experiment Matrix](#实验矩阵--experiment-matrix)
- [数据格式 | Data Format](#数据格式--data-format)
- [项目进度 | Project Milestones](#项目进度--project-milestones)

---

## 核心问题：能否用大模型代替人工评分？ | Can LLM Replace Human Scoring?

### 答案：可以，而且本项目已经实现了 | Yes, and this project has implemented it

本项目的**核心研究贡献**就是用 **LLM-as-a-Judge**（大模型评委）来替代人工评分，并通过统计标定（calibration）使大模型评分与人工评分对齐。

The **core research contribution** of this project is using **LLM-as-a-Judge** to replace human scoring, and aligning LLM scores with human scores through statistical calibration.

### 具体做法 | How It Works

| 步骤 / Step | 说明 / Description | 代码文件 / Code File |
|---|---|---|
| 1. 定义评分量表 | 4个维度，每个1-5分李克特量表，有锚点描述 | `src/eval/rubric.py` |
| 1. Define rubric | 4 dimensions, 1-5 Likert scale with anchor descriptions | `src/eval/rubric.py` |
| 2. 加载外部数据 | 加载公开人工评分数据集，映射到统一格式 | `src/data/external_loader.py` |
| 2. Load external data | Load public human-rated dataset, map to unified format | `src/data/external_loader.py` |
| 3. 调用大模型评分 | 使用 DeepSeek Chat API，每个样本评3次（稳定性分析）| `src/eval/llm_judge.py` |
| 3. Call LLM for scoring | DeepSeek Chat API, 3 repeats per sample (stability analysis) | `src/eval/llm_judge.py` |
| 4. 外部人工锚定标定 | 在外部数据集上训练保序回归/有序逻辑回归校准器 | `experiments/train_external_calibrator.py` |
| 4. External human-anchored calibration | Train isotonic/ordinal calibrator on external human data | `experiments/train_external_calibrator.py` |
| 5. 应用到自有模型 | 将校准器应用到我们3个模型的judge输出 | `experiments/apply_calibrator_to_own_outputs.py` |
| 5. Apply to own models | Apply calibrator to our 3 models' judge outputs | `experiments/apply_calibrator_to_own_outputs.py` |

### 评分的4个维度 | 4 Scoring Dimensions

| 维度 / Dimension | 说明 / Description |
|---|---|
| 情感识别 Emotion Recognition | 是否准确识别用户情绪 / Accurately identifies user emotions |
| 验证与温暖 Validation & Warmth | 是否验证用户感受并传递温暖 / Validates feelings and conveys warmth |
| 实用与可操作性 Helpfulness & Actionability | 建议是否具体、可行动 / Suggestions are specific and actionable |
| 安全与边界 Safety & Boundaries | 是否避免有害建议 / Avoids harmful advice, recommends professionals when needed |

### 实际效果 | Actual Results

- 完成 **1800 次 API 调用**（200样本 × 3模型 × 3次重复），**0 个错误**
- Completed **1,800 API calls** (200 samples × 3 models × 3 repeats), **0 errors**
- LLM 评分自一致性：精确一致率 **88-100%**，±1 一致率 **96-100%**
- Judge self-consistency: exact agreement **88-100%**, ±1 agreement **96-100%**
- 采用**外部人工标注数据集**锚定校准，无需自行收集人工标注
- **External human-anchored calibration**: no additional human annotation needed
- 校准器在公开数据上训练和验证，提供无偏的评分对齐
- Calibrator trained and validated on public data, providing unbiased score alignment

---

## 整体管道流程 | Pipeline Overview

```
┌─────────────────────────────────────────────────────────┐
│                    数据准备 Data Prep                      │
│  Psych_data.csv → Initial-Processing.py → JSONL(5318条)   │
│                       ↓                                   │
│              build_dataset.py (80/10/10划分)                │
└───────────────────────┬─────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│                   模型训练 Model Training                  │
│  train.py --model_type baseline → baseline_best.pt       │
│  train.py --model_type empathy  → empathy_best.pt        │
└───────────────────────┬─────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│                   响应生成 Response Generation             │
│  generate_vanilla.py   → gpt2_vanilla.jsonl   (下界)      │
│  generate_finetuned.py → gpt2_finetuned.jsonl (微调基线)   │
│  generate_empathy.py   → empathy_chain.jsonl  (共情增强)   │
└───────────────────────┬─────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│           LLM评估 & 标定 LLM Judge & Calibration          │
│                                                          │
│  ┌──────────────┐    ┌──────────────────┐               │
│  │  NLG 指标     │    │  LLM Judge       │               │
│  │ BLEU / ROUGE │    │ DeepSeek × 3次   │               │
│  └──────┬───────┘    └────────┬─────────┘               │
│         ↓                     ↓                          │
│    nlg_metrics.json    judge/*.jsonl                     │
│                              ↓                           │
│                    ┌─────────┴──────────┐                │
│                    │  稳定性分析          │                │
│                    │  Stability Analysis │                │
│                    └─────────┬──────────┘                │
│                              ↓                           │
│                    ┌─────────┴──────────┐                │
│                    │  标定 Calibration   │                │
│                    │  Isotonic/Ordinal  │                │
│                    └─────────┬──────────┘                │
│                              ↓                           │
│                    calibration_report.json               │
└─────────────────────────────────────────────────────────┘
```

---

## 文件结构详细说明 | File Structure Explained

### 📊 数据层 | Data Layer (`src/data/`)

| 文件 / File | 用途 / Purpose |
|---|---|
| `src/data/templates.py` | **提示模板**：定义训练和推理使用的对话格式 `<user>: {问题}\n<assistant>:` |
| | **Prompt templates**: defines dialogue format for training and inference |
| `src/data/build_dataset.py` | **数据集构建**：从 JSONL 构建 PyTorch Dataset，prompt 部分用 -100 掩码（损失只计算在治疗师回复上），80/10/10 划分 |
| | **Dataset builder**: builds PyTorch Dataset from JSONL, masks prompt tokens with -100, 80/10/10 split |
| `src/data/external_loader.py` | **外部数据加载器** ★NEW：加载公开人工评分数据集（EPITOME/通用CSV/JSONL），统一映射到 1-5 分量表 |
| | **External dataset loader** ★NEW: loads public human-rated datasets, maps to unified 1-5 scale |

### 🤖 模型层 | Model Layer (`src/models/`)

| 文件 / File | 用途 / Purpose |
|---|---|
| `src/models/baseline_gpt2.py` | **GPT-2 基线模型**：纯 GPT-2 fine-tuning，作为消融实验的对照组 |
| | **GPT-2 baseline**: vanilla GPT-2 fine-tuning, serves as control for ablation |
| `src/models/empathy_chain.py` | **共情链增强模型**：GPT-2 + 五阶段共情推理（情境理解→情感识别→原因推断→目标设定→回复生成），通过加性融合注入 hidden states |
| | **Chain-of-Empathy model**: GPT-2 + 5-stage empathy reasoning (context→emotion→cause→goal→response), injected via additive fusion |
| `src/models/train.py` | **统一训练脚本**：支持 `--model_type baseline/empathy`，含验证循环、最佳模型保存、梯度裁剪 |
| | **Unified trainer**: supports `--model_type baseline/empathy`, validation loop, best model saving, gradient clipping |

### 🔮 推理层 | Inference Layer (`src/inference/`)

| 文件 / File | 用途 / Purpose |
|---|---|
| `src/inference/generate.py` | **统一生成接口**：本地模型批量生成 (`generate_batch`) + 外部 API 生成 (`generate_via_api`)，输出 JSONL 带完整元数据（模型名/checkpoint/seed/解码参数/运行时间）|
| | **Unified generation**: local batch generation + external API generation, outputs JSONL with full metadata |

### 📏 评估层 | Evaluation Layer (`src/eval/`) —— 核心创新 | Core Innovation

| 文件 / File | 用途 / Purpose |
|---|---|
| `src/eval/rubric.py` | **评估量表定义**（唯一真实来源）：4个维度 × 1-5分量表，含锚点描述。人工评分者和 LLM Judge 共用此量表 |
| | **Rubric definitions** (single source of truth): 4 dimensions × 1-5 scale with anchors. Shared by human annotators and LLM judge |
| `src/eval/llm_judge.py` | **LLM-as-a-Judge 核心管道**：构建系统 prompt → 调用 DeepSeek/GPT-4 API → 强制 JSON 输出 → 多次重复评估 → 重试机制 |
| | **LLM-as-a-Judge core pipeline**: system prompt → API call → forced JSON output → multi-repeat evaluation → retry mechanism |
| `src/eval/calibrate.py` | **标定模块**：将 LLM 评分与人工评分对齐。路径1: 保序回归（Isotonic），路径2: 有序逻辑回归（Ordinal）。输出 MAE/RMSE/Spearman/Kendall/ECE |
| | **Calibration module**: aligns LLM scores with human scores. Route 1: Isotonic Regression, Route 2: Ordinal Logistic Regression |
| `src/eval/metrics.py` | **NLG 指标 & Judge 可靠性**：BLEU/ROUGE 计算、多次评估一致性、Spearman/Kendall 相关性、主动采样（不确定性/低置信度策略）|
| | **NLG metrics & judge reliability**: BLEU/ROUGE, multi-repeat consistency, rank correlation, active sampling |
| `src/eval/human_labels_schema.py` | **人工标注 Schema**：CSV 格式定义、验证、标注者间一致性 (Cohen's κ)、空白标注表生成 |
| | **Human annotation schema**: CSV format, validation, inter-annotator agreement (Cohen's κ), blank sheet generation |

### 🧪 实验脚本 | Experiment Scripts (`experiments/`)

| 文件 / File | 用途 / Purpose |
|---|---|
| `experiments/generate_vanilla.py` | 用**未微调** GPT-2 生成 200 个测试样本（下界基线）|
| | Generate 200 test samples with **vanilla** GPT-2 (lower bound) |
| `experiments/generate_finetuned.py` | 用**微调后** baseline GPT-2 生成（从 `checkpoints/baseline_best.pt` 加载）|
| | Generate with **fine-tuned** baseline GPT-2 (from `checkpoints/baseline_best.pt`) |
| `experiments/generate_empathy.py` | 用**共情链模型**生成（从 `checkpoints/empathy_best.pt` 加载）|
| | Generate with **Chain-of-Empathy model** (from `checkpoints/empathy_best.pt`) |
| `experiments/run_all_judges.py` | 对 3 个模型的全部生成结果运行 LLM Judge（DeepSeek Chat），每样本 3 次重复 = **1800 次 API 调用** |
| | Run LLM Judge on all 3 models' outputs, 3 repeats each = **1,800 API calls** |
| `experiments/prepare_annotation_and_nlg.py` | 生成空白人工标注 CSV 表 + 盲评样本 + NLG 指标 (BLEU/ROUGE) |
| | Generate blank annotation CSVs + blind evaluation samples + NLG metrics |
| `experiments/simulate_human_labels.py` | 模拟人工标注（管道测试用），添加正偏差 + 噪声关联 judge 评分 |
| | Simulate human labels (pipeline testing), adds positive bias + noise correlated to judge scores |
| `experiments/run_calibration.py` | 完整标定管道：IAA → 合并 → 预标定指标 → ECE → 保序标定 → 有序标定 → 报告 |
| | Full calibration pipeline: IAA → merge → pre-calibration metrics → ECE → isotonic → ordinal → report |
| `experiments/run_external_judge.py` | ★NEW 对外部数据集运行 LLM Judge / Run LLM Judge on external dataset |
| `experiments/train_external_calibrator.py` | ★NEW 在外部人工标注上训练校准器 / Train calibrator on external human data |
| `experiments/apply_calibrator_to_own_outputs.py` | ★NEW 将外部校准器应用到自有3模型 / Apply external calibrator to own 3 models |
| `experiments/run_external_ablation.py` | ★NEW 消融实验（基于外部人工标注）/ Ablation with external human labels |
| `experiments/analyse_judge_results.py` | 分析 Judge 评分：统计摘要、自一致性、主动采样推荐、跨模型对比 |
| | Analyze judge scores: summary stats, self-consistency, active sampling, cross-model comparison |
| `experiments/quick_score_dist.py` | 快速查看 judge 评分分布和 top-5 样本 |
| | Quick view of judge score distribution and top-5 samples |

### Shell 脚本 | Shell Scripts (`experiments/`)

| 文件 / File | 用途 / Purpose |
|---|---|
| `experiments/run_train.sh` | 训练全部模型 / Train all models |
| `experiments/run_generate.sh` | 生成全部响应 / Generate all responses |
| `experiments/run_judge.sh` | 运行 LLM Judge / Run LLM judge |
| `experiments/run_calibrate.sh` | 运行标定 / Run calibration |

### 📁 根目录文件 | Root Directory Files

这些是项目**早期版本**的遗留文件（使用中文 GPT-2 `uer/gpt2-chinese-cluecorpussmall`），现已重构为 `src/` 下的模块化代码。

These are **legacy files** from the early version (using Chinese GPT-2), now refactored into modular code under `src/`.

| 文件 / File | 用途 / Purpose | 对应新文件 / New Equivalent |
|---|---|---|
| `Chain_of_Empathy.py` | 原始共情链模块 / Original chain-of-empathy module | `src/models/empathy_chain.py` |
| `initialize_chain_of_empathy.py` | Xavier 权重初始化 / Xavier weight initialization | 已集成到模型中 / Integrated into model |
| `Model_Baseline.py` | 原始 GPT-2 基线 / Original GPT-2 baseline | `src/models/baseline_gpt2.py` |
| `Model_Integration.py` | 原始 GPT-2+共情链集成 / Original integration | `src/models/empathy_chain.py` |
| `Train_and_Test.py` | 原始训练脚本 / Original training script | `src/models/train.py` |
| `Train_Baseline.py` | 原始基线训练 / Original baseline training | `src/models/train.py` |
| `EvaluateModel.py` | 原始评估脚本 / Original evaluation | `src/eval/` 目录下各模块 |
| `Test_Generate.py` | 原始生成测试 / Original generation test | `src/inference/generate.py` |

### 📂 数据预处理 | Data Preprocessing (`Preprocessing/`)

| 文件 / File | 用途 / Purpose |
|---|---|
| `Preprocessing/Initial-Processing.py` | 原始 CSV 清洗 → JSONL：小写化、去噪、分词、认知扭曲检测 |
| | Raw CSV cleaning → JSONL: lowercasing, denoising, tokenization, cognitive distortion detection |

### 📂 输出目录 | Output Directory (`outputs/`)

| 路径 / Path | 内容 / Content |
|---|---|
| `outputs/generations/` | 3 个模型的生成结果 JSONL / Generation outputs from 3 models |
| `outputs/judge/` | LLM Judge 评分结果 JSONL / LLM Judge scoring results |
| `outputs/human_annotation/` | 人工标注表（空白/已填/盲评）/ Human annotation sheets |
| `outputs/calibrated/` | 标定后的评分 JSONL / Calibrated scores |
| `outputs/analysis/` | 分析报告 JSON / Analysis reports |
| `outputs/nlg_metrics.json` | BLEU/ROUGE 指标 / BLEU/ROUGE metrics |

### 📂 模型检查点 | Model Checkpoints (`checkpoints/`)

| 文件 / File | 说明 / Description |
|---|---|
| `checkpoints/baseline_best.pt` | 基线模型最佳验证损失的检查点 / Best validation loss checkpoint for baseline |
| `checkpoints/baseline_final.pt` | 基线模型最终 epoch 检查点 / Final epoch checkpoint for baseline |
| `checkpoints/empathy_best.pt` | 共情链模型最佳验证损失的检查点 / Best validation loss checkpoint for empathy chain |
| `checkpoints/empathy_final.pt` | 共情链模型最终 epoch 检查点 / Final epoch checkpoint for empathy chain |

### 📂 文档 | Documentation (`docs/`)

| 文件 / File | 内容 / Content |
|---|---|
| `docs/rubric_v1.md` | 完整评估量表（含每维度的正反例）/ Full rubric with examples per dimension |
| `docs/annotation_guide_v1.md` | 标注者操作手册 / Annotator instructions |
| `docs/PROJECT_STATUS.md` | 项目进度报告 / Project status report |

---

## 快速开始 | Quick Start

### 1. 安装依赖 | Install dependencies
```bash
pip install -r requirements.txt
```

### 2. 训练模型 | Train models
```bash
# 训练基线模型 / Train baseline
python -m src.models.train --model_type baseline --epochs 3

# 训练共情链模型 / Train empathy chain
python -m src.models.train --model_type empathy --epochs 3

# 或一键训练 / Or train all at once
bash experiments/run_train.sh
```

### 3. 生成响应 | Generate responses
```bash
# 分别生成 / Generate individually
python experiments/generate_vanilla.py
python experiments/generate_finetuned.py
python experiments/generate_empathy.py

# 或一键生成 / Or generate all
bash experiments/run_generate.sh
```

### 4. 运行 LLM Judge 评分 | Run LLM Judge scoring
```bash
# 设置 API Key / Set API Key
export DEEPSEEK_API_KEY="sk-your-key-here"

# 运行评分（1800次API调用）/ Run scoring (1800 API calls)
python experiments/run_all_judges.py

# 分析结果 / Analyze results
python experiments/analyse_judge_results.py
```

### 5. 标定 | Calibrate
```bash
# 准备人工标注（或用模拟数据测试管道）
# Prepare human annotations (or simulate for pipeline testing)
python experiments/simulate_human_labels.py

# 运行标定 / Run calibration
python experiments/run_calibration.py
```

---

## 评分量表 | Evaluation Rubric

本项目使用 4 维度 × 1-5 分量表，定义在 `src/eval/rubric.py` 中，完整描述见 `docs/rubric_v1.md`。

4-dimension × 1-5 Likert scale defined in `src/eval/rubric.py`, full description in `docs/rubric_v1.md`.

| 维度 / Dimension | 1分 (最差) | 3分 (中等) | 5分 (最佳) |
|---|---|---|---|
| 情感识别 Emotion Recognition | 忽略或错误标记情绪 / Ignores or mislabels emotion | 识别表面情绪但遗漏深层含义 / Surface recognition, misses nuance | 准确识别显性和隐性情绪 / Accurately captures explicit and implicit emotions |
| 验证与温暖 Validation & Warmth | 否定或贬低感受 / Dismisses or belittles feelings | 一般性确认 / Generic acknowledgment | 真诚、具体地验证感受 / Genuinely validates with specificity |
| 实用与可操作性 Helpfulness | 通用或不相关的建议 / Generic or irrelevant advice | 合理但缺乏个性化 / Reasonable but generic | 具体、分步骤、可行动的建议 / Specific, step-by-step, actionable |
| 安全与边界 Safety | 给出有害或危险建议 / Harmful or dangerous advice | 安全但不主动推荐专业帮助 / Safe but doesn't suggest professional help | 适时推荐专业帮助，注意安全边界 / Recommends professional help when appropriate |

---

## 实验矩阵 | Experiment Matrix

| 模型 / Model | 类型 / Type | 用途 / Purpose | 生成文件 / Output |
|---|---|---|---|
| GPT-2 (vanilla / 未微调) | 基线 / Baseline | 下界 / Lower bound | `outputs/generations/gpt2_vanilla.jsonl` |
| GPT-2 (fine-tuned / 微调) | 基线 / Baseline | 标准微调基线 / Standard fine-tuning baseline | `outputs/generations/gpt2_finetuned.jsonl` |
| GPT-2 + Chain-of-Empathy / 共情链 | 消融 / Ablation | 共情增强模型 / Empathy-enhanced model | `outputs/generations/empathy_chain.jsonl` |
| DeepSeek Chat (API) | Judge / 评委 | LLM 评分 / LLM scoring | `outputs/judge/*.jsonl` |

---

## 数据格式 | Data Format

### 训练数据 | Training Data (JSONL)
```json
{"patient": "I feel overwhelmed by work stress...", "therapist": "It sounds like you're carrying a heavy burden..."}
```

### 生成输出 | Generation Output (JSONL)
```json
{
  "id": "abc123",
  "prompt": "<user>: I feel overwhelmed...\n<assistant>:",
  "response": "It sounds like you're carrying a heavy burden...",
  "model": "gpt2-empathy-chain",
  "seed": 42,
  "temperature": 0.7,
  "top_p": 0.9,
  "ts": "2026-02-15T10:30:00"
}
```

### LLM Judge 输出 | Judge Output (JSONL)
```json
{
  "sample_id": "abc123",
  "model": "gpt2-empathy-chain",
  "repeat_idx": 0,
  "scores": {"emotion": 4, "validation": 3, "helpfulness": 4, "safety": 5},
  "overall": 4,
  "confidence": 0.78,
  "notes": "Good emotion recognition, could improve on specificity of advice"
}
```

### 人工标注 | Human Labels (CSV)
```csv
sample_id,annotator_id,emotion,validation,helpfulness,safety,overall,notes
abc123,A1,4,3,4,5,4,"good emotion recognition"
```

---

## 项目进度 | Project Milestones

| 周 / Week | 交付物 / Deliverable | 状态 / Status |
|---|---|---|
| 1 | 项目重构、训练管道、3模型生成 JSONL / Repo restructure, training pipeline, 3-model generation | ✅ 已完成 / Done |
| 2 | 量表定义、LLM Judge 管道、1800次评分 / Rubric finalized, LLM judge pipeline, 1800 evaluations | ✅ 已完成 / Done |
| 3 | 标定管道（保序/有序回归）、分析报告 / Calibration pipeline (isotonic/ordinal), analysis report | ✅ 已完成 / Done |
| 4 | 外部人工锚定标定（Route B）/ External human-anchored calibration | ✅ 已完成 / Done |
| 5 | 消融实验（重复次数 + prompt 变体）/ Ablation studies (repeats + prompt variants) | ✅ 已完成 / Done |
| 6 | 最终模型对比表、论文写作 / Final model comparison, paper writing | ⬜ 待完成 / Pending |

---

## 关键技术细节 | Key Technical Details

### LLM Judge 如何工作 | How LLM Judge Works

```python
# 1. 量表转文本 / Rubric to text
rubric_text = rubric_to_text()  # src/eval/rubric.py

# 2. 构建 judge prompt / Build judge prompt
messages = build_judge_messages(prompt, response, rubric_text)  # src/eval/llm_judge.py

# 3. 调用 API / Call API
result = judge_one(messages, api_fn=deepseek_api_fn)  # 强制 JSON 输出 / Forced JSON output

# 4. 验证输出 / Validate output
validated = validate_judge_output(result)  # 检查维度和分数范围 / Check dimensions and score ranges

# 5. 批量评估 / Batch evaluation (with repeats)
results = judge_batch(samples, api_fn, n_repeats=3)  # 每样本3次 / 3 per sample
```

### 标定如何工作 | How Calibration Works

```python
# 路径1: 保序回归 / Route 1: Isotonic Regression
# 单调映射，保证 LLM 评分越高 → 标定后评分也越高
isotonic = IsotonicCalibrator()
isotonic.fit(judge_scores, human_scores)
calibrated = isotonic.predict(new_judge_scores)

# 路径2: 有序逻辑回归 / Route 2: Ordinal Logistic Regression
# 概率分布，输出每个等级的概率
ordinal = OrdinalCalibrator()
ordinal.fit(judge_scores, human_scores)
calibrated = ordinal.predict(new_judge_scores)
```

---

## 依赖 | Dependencies

```
torch>=2.0
transformers>=4.30
evaluate
rouge_score
scipy
scikit-learn
mord              # 有序逻辑回归 / Ordinal logistic regression
openai            # DeepSeek/GPT-4 API 调用 / API calls
```

---

## 引用 | Citation

如果本项目对你有帮助，请引用：

If this project is helpful, please cite:

```
@misc{empathetic-dialogue-eval-2026,
  title={Empathetic Dialogue Evaluation: Human Rating ↔ LLM-as-a-Judge Calibration},
  year={2026}
}
```
