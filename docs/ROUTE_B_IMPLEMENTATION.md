# Route B 实施记录：外部人工锚定校准 | Route B Implementation: External Human-Anchored Calibration

**日期 Date**: 2026-02-28
**版本 Version**: v2.0 — Route B

---

## 目录 | Table of Contents

- [概述 | Overview](#概述--overview)
- [架构变更总览 | Architecture Changes Overview](#架构变更总览--architecture-changes-overview)
- [Step 1：外部数据加载器 | External Dataset Loader](#step-1外部数据加载器--external-dataset-loader)
- [Step 2：外部数据 Judge | External Judge](#step-2外部数据-judge--external-judge)
- [Step 3：训练外部校准器 | Train External Calibrator](#step-3训练外部校准器--train-external-calibrator)
- [Step 4：应用校准器到自有模型 | Apply Calibrator to Own Models](#step-4应用校准器到自有模型--apply-calibrator-to-own-models)
- [Step 5：消融实验 | Ablation Studies](#step-5消融实验--ablation-studies)
- [Step 6：README 更新 | README Updates](#step-6readme-更新--readme-updates)
- [新增文件清单 | New Files List](#新增文件清单--new-files-list)
- [修改文件清单 | Modified Files List](#修改文件清单--modified-files-list)
- [完整运行指南 | Full Run Guide](#完整运行指南--full-run-guide)
- [数据格式规范 | Data Format Specification](#数据格式规范--data-format-specification)
- [设计决策说明 | Design Decisions](#设计决策说明--design-decisions)

---

## 概述 | Overview

### 问题 | Problem

原方案要求我们自己收集人工标注来校准 LLM Judge，但这增加了成本和时间。

The original plan required collecting our own human annotations to calibrate the LLM Judge, which adds cost and time.

### 解决方案：Route B | Solution: Route B

使用**公开可用的人工评分数据集**（如 EPITOME、EmpatheticDialogues-EVAL 等）作为 human anchor，在该数据集上：
1. 用同一个 LLM Judge 打分
2. 与公开人工评分对齐训练校准器
3. 将校准器应用到我们自己的 3 个模型输出

Use **publicly available human-rated datasets** (e.g., EPITOME, EmpatheticDialogues-EVAL) as a human anchor:
1. Score the external data with the same LLM Judge
2. Train a calibrator aligned with public human ratings
3. Apply the calibrator to our own 3 model outputs

### 核心优势 | Core Advantages

| 优势 / Advantage | 说明 / Description |
|---|---|
| 零人工标注成本 | 不需要自己标注，使用公开数据 |
| Zero annotation cost | No self-labeling needed, uses public data |
| 无偏锚点 | 外部数据集的人工评分是独立的 |
| Unbiased anchor | External human ratings are independent |
| 可复现 | 任何人可以下载同样的公开数据重现结果 |
| Reproducible | Anyone can download the same public data |
| 改动最小 | 复用现有 judge/calibrate 模块 |
| Minimal changes | Reuses existing judge/calibrate modules |

---

## 架构变更总览 | Architecture Changes Overview

```
┌────────────────────────────────────────────────────────────────┐
│  NEW: 外部数据管道 / External Data Pipeline                      │
│                                                                │
│  external CSV/JSONL                                            │
│       ↓                                                        │
│  src/data/external_loader.py  →  unified.jsonl                 │
│       ↓                                                        │
│  experiments/run_external_judge.py                             │
│       ↓                                                        │
│  outputs/judge_external/<dataset>_<judge>.jsonl                │
│       ↓                                                        │
│  experiments/train_external_calibrator.py                      │
│       ↓                                                        │
│  checkpoints/calibrators/<dataset>_<judge>_isotonic.pkl        │
│  outputs/analysis/external_calibration_report.json             │
│                                                                │
│  ┌─────────────────────────────────────────────────┐           │
│  │ EXISTING: 自有模型管道 / Own Model Pipeline      │           │
│  │                                                  │           │
│  │  outputs/judge/*_judge.jsonl  (已有 / existing)  │           │
│  │       ↓                                          │           │
│  │  experiments/apply_calibrator_to_own_outputs.py  │  ★ NEW   │
│  │       ↓                                          │           │
│  │  outputs/calibrated/<judge>_human_calibrated.jsonl          │
│  └─────────────────────────────────────────────────┘           │
│                                                                │
│  experiments/run_external_ablation.py  ★ NEW                   │
│       ↓                                                        │
│  outputs/analysis/external_ablation_repeats.json/md            │
└────────────────────────────────────────────────────────────────┘
```

---

## Step 1：外部数据加载器 | External Dataset Loader

### 新增文件 | New File

**`src/data/external_loader.py`**

### 功能说明 | Functionality

| 功能 / Feature | 说明 / Description |
|---|---|
| 多格式支持 | CSV 和 JSONL 自动检测 |
| Multi-format support | CSV and JSONL auto-detection |
| 命名数据集加载器 | EPITOME、EmpatheticDialogues-EVAL 专用解析 |
| Named dataset loaders | Specialized parsing for EPITOME, EmpatheticDialogues-EVAL |
| 通用加载器 | 自定义列名 + 分数范围映射 |
| Generic loader | Custom column names + score range mapping |
| 分数归一化 | 任意量表 → 1-5 线性映射 |
| Score normalization | Any scale → 1-5 linear mapping |
| 格式转换 | 转为 judge_batch 和 merge_human_and_judge 的输入格式 |
| Format conversion | Convert to judge_batch and merge_human_and_judge input format |

### 统一输出格式 | Unified Output Format

```json
{
    "item_id": "a1b2c3d4e5f6",
    "prompt": "I feel overwhelmed by work stress and don't know what to do...",
    "response": "It sounds like you're carrying a heavy burden...",
    "human_overall": 3.67,
    "source": "epitome"
}
```

### 关键函数 | Key Functions

| 函数 / Function | 签名 / Signature | 用途 / Purpose |
|---|---|---|
| `rescale_to_1_5` | `(value, src_min, src_max) → float` | 线性映射到 1-5 |
| `load_generic_csv` | `(path, *, prompt_col, response_col, score_col, ...) → list[dict]` | 通用 CSV 加载 |
| `load_generic_jsonl` | `(path, *, ...) → list[dict]` | 通用 JSONL 加载 |
| `load_epitome` | `(path) → list[dict]` | EPITOME 数据集 (0-2 × 3 维度 → 1-5) |
| `load_empatheticdialogues_eval` | `(path) → list[dict]` | ED-EVAL 数据集 |
| `load_external` | `(path, *, dataset, ...) → list[dict]` | **统一入口**：自动选择加载器 |
| `save_unified_jsonl` | `(records, path) → None` | 保存统一 JSONL |
| `convert_to_generation_format` | `(records) → list[dict]` | 转为 judge_batch 输入格式 |
| `convert_to_human_labels` | `(records, annotator_id) → list[dict]` | 转为 merge_human_and_judge 输入格式 |

### CLI 用法 | CLI Usage

```bash
# 通用 CSV (自定义列名和分数范围)
python -m src.data.external_loader \
    --input data/external/my_dataset.csv \
    --output data/external/unified.jsonl \
    --prompt_col context \
    --response_col reply \
    --score_col quality \
    --score_min 1 --score_max 7

# EPITOME 数据集
python -m src.data.external_loader \
    --input data/external/epitome.csv \
    --output data/external/epitome_unified.jsonl \
    --dataset epitome
```

---

## Step 2：外部数据 Judge | External Judge

### 新增文件 | New File

**`experiments/run_external_judge.py`**

### 功能说明 | Functionality

| 功能 / Feature | 说明 / Description |
|---|---|
| 复用现有 judge | 直接调用 `judge_batch`、`save_judge_results` |
| Reuses existing judge | Directly calls `judge_batch`, `save_judge_results` |
| 断点续传 | `--resume` 参数跳过已评分样本 |
| Resume support | `--resume` flag skips already-scored samples |
| 采样限制 | `--max_samples` 限制评分数量（测试用）|
| Sample limiting | `--max_samples` limits scoring count (for testing) |
| 灵活输入 | 支持预转换 JSONL 或原始文件（自动转换）|
| Flexible input | Supports pre-converted JSONL or raw files (auto-convert) |

### 输出 | Output

```
outputs/judge_external/<dataset>_<judge_model>.jsonl
```

每条记录格式与现有 judge JSONL 完全一致：

Each record follows the exact same format as existing judge JSONL:

```json
{
    "sample_id": "a1b2c3d4e5f6",
    "model": "external_epitome",
    "repeat_idx": 0,
    "judge_model": "deepseek-chat",
    "judge_temp": 0.3,
    "ts": "2026-02-28T10:30:00+00:00",
    "scores": {"emotion": 3, "validation": 2, "helpfulness": 3, "safety": 5},
    "overall": 3,
    "confidence": 0.75,
    "notes": "..."
}
```

### CLI 用法 | CLI Usage

```bash
export DEEPSEEK_API_KEY="sk-..."

python experiments/run_external_judge.py \
    --input data/external/unified.jsonl \
    --dataset my_dataset \
    --judge_model deepseek-chat \
    --judge_backend deepseek \
    --n_repeats 3 \
    --delay 0.5

# 测试模式（只评 10 条）
python experiments/run_external_judge.py \
    --input data/external/unified.jsonl \
    --dataset test \
    --max_samples 10

# 断点续传
python experiments/run_external_judge.py \
    --input data/external/unified.jsonl \
    --dataset my_dataset \
    --resume
```

---

## Step 3：训练外部校准器 | Train External Calibrator

### 新增文件 | New File

**`experiments/train_external_calibrator.py`**

### 功能说明 | Functionality

这是 `run_calibration_paper.py` 的外部数据版本。核心逻辑完全复用现有 `src/eval/calibrate.py` 模块。

This is the external-data version of `run_calibration_paper.py`. Core logic fully reuses the existing `src/eval/calibrate.py` module.

| 功能 / Feature | 说明 / Description |
|---|---|
| 数据合并 | external human labels + external judge scores → merged |
| Data merge | external human labels + external judge scores → merged |
| 60/20/20 分割 | 训练/开发/测试，按 sample_id 确定性划分 |
| 60/20/20 split | train/dev/test, deterministic by sample_id |
| Isotonic 标定 | 主线路径，保序回归 |
| Isotonic calibration | Primary route, isotonic regression |
| Ordinal 标定 | 对比路径，有序逻辑回归 |
| Ordinal calibration | Comparison route, ordinal logistic regression |
| Bootstrap 95% CI | 1000 次重采样计算置信区间 |
| Bootstrap 95% CI | 1000 resamples for confidence intervals |
| 保存校准器 | pickle 序列化，供 Step 4 使用 |
| Save calibrator | pickle serialization, for Step 4 use |

### 输出 | Outputs

| 文件 / File | 用途 / Purpose |
|---|---|
| `checkpoints/calibrators/<dataset>_<judge>_isotonic.pkl` | Isotonic 校准器模型 |
| `checkpoints/calibrators/<dataset>_<judge>_ordinal.pkl` | Ordinal 校准器模型 |
| `outputs/analysis/external_calibration_report.json` | 完整指标 JSON（MAE/RMSE/Spearman + Bootstrap CI）|
| `outputs/analysis/external_calibration_report.md` | Markdown 报告 |
| `outputs/analysis/external_split.json` | 数据分割记录（可复现）|
| `outputs/calibrated/external_<dataset>_isotonic_test.jsonl` | 测试集校准结果 |

### CLI 用法 | CLI Usage

```bash
python experiments/train_external_calibrator.py \
    --external_data data/external/unified.jsonl \
    --judge_results outputs/judge_external/my_dataset_deepseek_chat.jsonl \
    --dataset my_dataset \
    --judge_model deepseek_chat \
    --n_bootstrap 1000
```

### 与 run_calibration_paper.py 的区别 | Differences from run_calibration_paper.py

| 方面 / Aspect | run_calibration_paper.py | train_external_calibrator.py |
|---|---|---|
| 数据来源 | 自有 simulated/human labels | 外部公开数据集 |
| Data source | Own simulated/human labels | External public dataset |
| 目的 | 验证管道可行性 | 训练生产级校准器 |
| Purpose | Validate pipeline feasibility | Train production calibrator |
| 输出 | 仅报告 | 报告 + **持久化校准器 .pkl** |
| Output | Report only | Report + **persisted calibrator .pkl** |
| 后续 | 独立实验 | 供 Step 4 apply_calibrator 使用 |
| Next step | Standalone experiment | For Step 4 apply_calibrator |

---

## Step 4：应用校准器到自有模型 | Apply Calibrator to Own Models

### 新增文件 | New File

**`experiments/apply_calibrator_to_own_outputs.py`**

### 功能说明 | Functionality

加载 Step 3 产出的校准器 `.pkl`，对现有 3 个模型的 judge 输出进行校准转换。

Loads the calibrator `.pkl` from Step 3 and applies calibration transform to the existing 3 models' judge outputs.

| 功能 / Feature | 说明 / Description |
|---|---|
| 自动聚合 | 每个 sample 的多次 repeat 自动求均值 |
| Auto-aggregation | Automatically averages multiple repeats per sample |
| 逐维校准 | 4 个维度分别校准 |
| Per-dimension calibration | Calibrates each of 4 dimensions separately |
| 模型对比表 | 输出 vanilla vs finetuned vs empathy_chain 对比 |
| Model comparison table | Outputs vanilla vs finetuned vs empathy_chain comparison |
| 双格式报告 | JSON + Markdown |
| Dual format report | JSON + Markdown |

### 输出 | Outputs

| 文件 / File | 用途 / Purpose |
|---|---|
| `outputs/calibrated/<judge>_human_calibrated.jsonl` | 所有模型的校准评分 |
| `outputs/analysis/human_calibrated_comparison.json` | 模型对比 JSON |
| `outputs/analysis/human_calibrated_comparison.md` | 模型对比 Markdown |

### 输出记录格式 | Output Record Format

```json
{
    "sample_id": "abc123",
    "model": "empathy_chain",
    "judge_raw": {"emotion": 3.33, "validation": 2.67, "helpfulness": 3.0, "safety": 5.0},
    "judge_std": {"emotion": 0.47, "validation": 0.47, "helpfulness": 0.0, "safety": 0.0},
    "judge_overall_raw": 3.67,
    "judge_confidence": 0.82,
    "calibrated": {"emotion": 2.89, "validation": 2.15, "helpfulness": 2.67, "safety": 4.5},
    "calibrated_overall": 3.053,
    "n_repeats": 3
}
```

### CLI 用法 | CLI Usage

```bash
python experiments/apply_calibrator_to_own_outputs.py \
    --calibrator checkpoints/calibrators/my_dataset_deepseek_chat_isotonic.pkl \
    --method isotonic

# 使用 ordinal 校准器
python experiments/apply_calibrator_to_own_outputs.py \
    --calibrator checkpoints/calibrators/my_dataset_deepseek_chat_ordinal.pkl \
    --method ordinal
```

### 最终主表示例 | Final Main Table Example

| Model | Raw Overall | Calibrated Overall | N |
|-------|:---:|:---:|:---:|
| gpt2_vanilla | 1.500 ± 0.300 | 1.234 ± 0.250 | 200 |
| gpt2_finetuned | 2.100 ± 0.500 | 1.876 ± 0.400 | 200 |
| empathy_chain | 2.800 ± 0.600 | 2.432 ± 0.500 | 200 |

---

## Step 5：消融实验 | Ablation Studies

### 新增文件 | New File

**`experiments/run_external_ablation.py`**

### Ablation A：重复次数敏感性 | Repeats Sensitivity

| 分析 / Analysis | 说明 / Description |
|---|---|
| k=1 vs k=2 vs k=3 | 从现有 3 次重复中子采样，分析边际收益 |
| k=1 vs k=2 vs k=3 | Subsample from existing 3 repeats, analyze marginal gains |
| 稳定性 | 不同 k 值下的评分标准差 |
| Stability | Score std at different k values |
| 对齐度 | 不同 k 值下与外部人工评分的 MAE/Spearman |
| Alignment | MAE/Spearman against external human at different k |
| 校准效果 | 不同 k 值下 isotonic 校准的 MAE 变化 |
| Calibration quality | Isotonic calibration MAE change at different k |

**零额外 API 成本**：完全复用已有 3 次重复数据。

**Zero extra API cost**: fully reuses existing 3-repeat data.

### Ablation B：Prompt 变体（可选） | Prompt Variant (Optional)

使用现有 `experiments/run_ablation_prompt.py`，只需将 `--human` 参数替换为外部人工标注：

Use existing `experiments/run_ablation_prompt.py`, just replace `--human` parameter with external human labels:

```bash
# 先将外部数据转为 human labels CSV
# Then use existing ablation script with external labels
python experiments/run_ablation_prompt.py \
    --human outputs/human_annotation/external_labels.csv \
    --n_samples 200
```

### 输出 | Outputs

| 文件 / File | 用途 / Purpose |
|---|---|
| `outputs/analysis/external_ablation_repeats.json` | 重复次数消融 JSON |
| `outputs/analysis/external_ablation_repeats.md` | 重复次数消融 Markdown |

---

## Step 6：README 更新 | README Updates

### 修改说明 | Changes

两个 README 文件均已更新：

Both README files have been updated:

| 修改内容 / Change | 说明 / Description |
|---|---|
| 项目描述 | "human annotations" → "external human-anchored calibration" |
| Project description | "human annotations" → "external human-anchored calibration" |
| 研究贡献 | 增加 external calibration 和 ablation studies |
| Research contributions | Added external calibration and ablation studies |
| 项目结构 | 添加新增文件和目录 |
| Project structure | Added new files and directories |
| Quick Start | 添加 Route B 完整步骤 |
| Quick Start | Added complete Route B steps |
| 里程碑 | 更新进度和 Week 4-6 计划 |
| Milestones | Updated progress and Week 4-6 plan |
| 实际效果 | 移除"标定后 MAE 下降"→ 改为"外部人工锚定校准" |
| Results | Removed "post-calibration MAE" → changed to "external human-anchored calibration" |
| 管道步骤 | 增加外部数据加载和格式转换步骤 |
| Pipeline steps | Added external data loading and format conversion steps |

---

## 新增文件清单 | New Files List

| # | 文件路径 / File Path | 类型 / Type | 代码行数 / LOC | 说明 / Description |
|---|---|---|---|---|
| 1 | `src/data/external_loader.py` | 核心模块 | ~310 | 外部数据集加载器，统一格式转换 |
| | | Core module | | External dataset loader, unified format conversion |
| 2 | `experiments/run_external_judge.py` | 实验脚本 | ~170 | 对外部数据运行 LLM Judge |
| | | Experiment script | | Run LLM Judge on external data |
| 3 | `experiments/train_external_calibrator.py` | 实验脚本 | ~340 | 训练外部人工锚定校准器 |
| | | Experiment script | | Train external human-anchored calibrator |
| 4 | `experiments/apply_calibrator_to_own_outputs.py` | 实验脚本 | ~250 | 将校准器应用到自有模型 |
| | | Experiment script | | Apply calibrator to own models |
| 5 | `experiments/run_external_ablation.py` | 实验脚本 | ~280 | 消融实验 (repeats k=1/2/3) |
| | | Experiment script | | Ablation experiments (repeats k=1/2/3) |
| 6 | `docs/ROUTE_B_IMPLEMENTATION.md` | 文档 | 本文件 | 详细实施记录（中英对照）|
| | | Documentation | This file | Detailed implementation record (bilingual) |

---

## 修改文件清单 | Modified Files List

| # | 文件路径 / File Path | 修改类型 / Change Type | 说明 / Description |
|---|---|---|---|
| 1 | `README.md` | 内容更新 | 添加 Route B 说明、新文件结构、Quick Start 步骤 |
| | | Content update | Added Route B description, new file structure, Quick Start steps |
| 2 | `README_CN.md` | 内容更新 | 添加外部校准说明、新文件表、更新里程碑 |
| | | Content update | Added external calibration info, new file table, updated milestones |

---

## 完整运行指南 | Full Run Guide

### 前置条件 | Prerequisites

```bash
# 1. 激活虚拟环境
source .venv/bin/activate

# 2. 确保依赖已安装
pip install -r requirements.txt

# 3. 设置 API Key (用于 LLM Judge)
export DEEPSEEK_API_KEY="sk-your-key-here"

# 4. 准备外部数据集（放到 data/external/ 目录）
mkdir -p data/external
# 将下载的公开数据集 CSV 放到 data/external/
```

### 完整管道（按顺序执行）| Full Pipeline (Execute in Order)

```bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 1: 加载外部数据集
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
python -m src.data.external_loader \
    --input data/external/my_dataset.csv \
    --output data/external/unified.jsonl \
    --dataset generic \
    --prompt_col context \
    --response_col response \
    --score_col empathy_score \
    --score_min 1 --score_max 5

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 2: 对外部数据运行 LLM Judge
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
python experiments/run_external_judge.py \
    --input data/external/unified.jsonl \
    --dataset my_dataset \
    --judge_model deepseek-chat \
    --judge_backend deepseek \
    --n_repeats 3

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 3: 训练校准器
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
python experiments/train_external_calibrator.py \
    --external_data data/external/unified.jsonl \
    --judge_results outputs/judge_external/my_dataset_deepseek_chat.jsonl \
    --dataset my_dataset \
    --judge_model deepseek_chat \
    --n_bootstrap 1000

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 4: 应用校准器到自有 3 模型
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
python experiments/apply_calibrator_to_own_outputs.py \
    --calibrator checkpoints/calibrators/my_dataset_deepseek_chat_isotonic.pkl \
    --method isotonic

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 5: 消融实验
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
python experiments/run_external_ablation.py \
    --external_data data/external/unified.jsonl \
    --judge_results outputs/judge_external/my_dataset_deepseek_chat.jsonl
```

### 批量执行脚本 | Batch Script

可选创建一个 shell 脚本一键执行：

Optionally create a one-click shell script:

```bash
#!/bin/bash
# experiments/run_route_b.sh
set -e
DATASET="my_dataset"
JUDGE="deepseek_chat"
INPUT="data/external/unified.jsonl"
JUDGE_RESULTS="outputs/judge_external/${DATASET}_${JUDGE}.jsonl"
CALIBRATOR="checkpoints/calibrators/${DATASET}_${JUDGE}_isotonic.pkl"

echo "=== Step 1: Load external data ==="
python -m src.data.external_loader --input data/external/my_dataset.csv --output "$INPUT"

echo "=== Step 2: Judge external data ==="
python experiments/run_external_judge.py --input "$INPUT" --dataset "$DATASET"

echo "=== Step 3: Train calibrator ==="
python experiments/train_external_calibrator.py \
    --external_data "$INPUT" --judge_results "$JUDGE_RESULTS" --dataset "$DATASET"

echo "=== Step 4: Apply to own models ==="
python experiments/apply_calibrator_to_own_outputs.py --calibrator "$CALIBRATOR"

echo "=== Step 5: Ablation ==="
python experiments/run_external_ablation.py \
    --external_data "$INPUT" --judge_results "$JUDGE_RESULTS"

echo "✅ Route B pipeline complete!"
```

---

## 数据格式规范 | Data Format Specification

### 外部数据统一格式 | External Data Unified Format

```json
{
    "item_id": "string (12-char hex hash or original ID)",
    "prompt": "string (user statement / context)",
    "response": "string (assistant / therapist response)",
    "human_overall": "float (1.0 - 5.0, rescaled from source)",
    "human_scores": {"dim1": score, ...},
    "source": "string (dataset name)"
}
```

### 分数映射规则 | Score Mapping Rules

| 源数据 / Source | 范围 / Range | 映射方式 / Mapping |
|---|---|---|
| 已经 1-5 | 1-5 | 直接使用 / Direct use |
| EPITOME (0-2 × 3) | 0-6 | `rescale_to_1_5(sum, 0, 6)` |
| 自定义 | 任意 | `--score_min` / `--score_max` 参数线性映射 |
| Custom | Any | `--score_min` / `--score_max` linear mapping |

### 校准器格式 | Calibrator Format

校准器通过 Python `pickle` 序列化为 `.pkl` 文件。包含：

Calibrators are serialized via Python `pickle` to `.pkl` files. Contains:

- `IsotonicCalibrator.models`: `dict[dim → sklearn.isotonic.IsotonicRegression]`
- `OrdinalCalibrator.models`: `dict[dim → mord.LogisticAT]`

---

## 设计决策说明 | Design Decisions

### 为什么不修改现有文件？| Why Not Modify Existing Files?

| 决策 / Decision | 原因 / Reason |
|---|---|
| `src/eval/calibrate.py` 不修改 | 现有 merge/calibrate/metrics 接口已完美适用 |
| `calibrate.py` unchanged | Existing merge/calibrate/metrics interfaces already fit perfectly |
| `src/eval/llm_judge.py` 不修改 | judge_batch 已支持任意 generation list |
| `llm_judge.py` unchanged | judge_batch already supports any generation list |
| 新增 external_loader 而非修改 build_dataset | 外部数据格式和内部训练数据格式不同 |
| New external_loader vs modifying build_dataset | External data format differs from internal training data |

### 为什么用 pickle 保存校准器？| Why Pickle for Calibrators?

- sklearn 的 IsotonicRegression 和 mord 的 LogisticAT 都支持 pickle
- 简单、标准、无额外依赖
- 文件很小（通常 < 100KB）

### 为什么 overall 用维度均值？| Why Overall = Mean of Dimensions?

外部数据通常只有一个 overall 分数，我们将其复制到 4 个维度作为代理。校准后的 overall 取维度均值，而非直接校准 overall 字段，因为：

1. per-dimension 校准更精细
2. 可以看到每个维度的校准效果
3. 与现有管道的 per-dimension 分析保持一致

External data usually only has one overall score, which we replicate to 4 dimensions as a proxy. Calibrated overall uses dimension means rather than calibrating overall directly because:

1. Per-dimension calibration is more fine-grained
2. Can see calibration effect per dimension
3. Consistent with existing pipeline's per-dimension analysis

### ID 对齐机制 | ID Alignment Mechanism

```
external_loader → item_id (md5 hash of prompt||response)
       ↓
convert_to_generation_format → id = item_id
       ↓
judge_batch → sample_id = id
       ↓
convert_to_human_labels → sample_id = item_id
       ↓
merge_human_and_judge → matches on sample_id
```

所有 ID 通过 `item_id` 贯穿始终，确保 human labels 和 judge scores 正确对齐。

All IDs flow through `item_id`, ensuring human labels and judge scores are correctly aligned.

---

## 附录：代码复用统计 | Appendix: Code Reuse Statistics

| 模块 / Module | 复用方式 / Reuse | 新增行数 / New LOC |
|---|---|---|
| `src/eval/llm_judge.py` | 原样调用 judge_batch, load/save | 0 |
| `src/eval/calibrate.py` | 原样调用 IsotonicCalibrator, OrdinalCalibrator, merge_human_and_judge | 0 |
| `src/eval/metrics.py` | 可选复用 | 0 |
| `src/eval/human_labels_schema.py` | 格式参考 | 0 |
| `src/eval/rubric.py` | import DIMENSION_KEYS | 0 |
| `src/data/external_loader.py` | **新增** | ~310 |
| `experiments/run_external_judge.py` | **新增** | ~170 |
| `experiments/train_external_calibrator.py` | **新增** | ~340 |
| `experiments/apply_calibrator_to_own_outputs.py` | **新增** | ~250 |
| `experiments/run_external_ablation.py` | **新增** | ~280 |
| **总计 / Total** | | **~1350 新增行** |

现有代码修改：**仅 README.md 和 README_CN.md**（内容更新，无代码逻辑变更）。

Existing code changes: **Only README.md and README_CN.md** (content updates, no code logic changes).

---

*本文档由 GitHub Copilot 自动生成 | Auto-generated by GitHub Copilot*
*最后更新 Last updated: 2026-02-28*
