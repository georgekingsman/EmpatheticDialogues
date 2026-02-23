# Research Pipeline 使用指南

> 本文档描述了为将 EmpatheticDialogues 项目从「工程验证完成」推进到「论文级研究证据链」而构建的完整实验管线。

## 1. 整体架构

整个管线分为 **5 个阶段**，以 Phase 1 的 GO/NO-GO 决策门为核心分支点：

```
Phase 1: Pilot 标注 (150 样本, 2 标注员)
    │
    ├── GO  (κ_w ≥ 0.4) ──→ Phase 2-5: 全量标注 + 分析
    │
    └── HOLD (κ_w < 0.4) ──→ 对齐会 → 修订 Rubric → Mini-pilot (50) → 重测
                                 │
                                 └── 最多 2 轮迭代后强制 re-evaluate
```

## 2. 新增 / 修改的文件清单

### 实验脚本 (`experiments/`)

| 脚本 | 阶段 | 用途 |
|------|------|------|
| `generate_pilot_annotation.py` | Phase 1 | 生成 150+10(隐藏重复) 样本的 pilot 标注包 |
| `run_pilot_iaa.py` | Phase 1 | 计算 IAA (weighted κ, Krippendorff α, Spearman 等)，输出 GO/NO-GO 决策 |
| `nogo_recovery.py` | Phase 1 | NO-GO 恢复流程：提取分歧案例、生成对齐会材料、rubric 修订模板、mini-pilot |
| `generate_full_annotation.py` | Phase 1→2 | GO 后生成 600+30(隐藏重复) 样本的全量标注包 |
| `judge_vs_human_analysis.py` | Phase 2 | Judge↔Human 对齐分析：per-dim 相关性、error 分解、top-20 error cases |
| `run_calibration_paper.py` | Phase 3 | 论文级校准：60/20/20 split + isotonic 回归 + 1000× bootstrap CI |
| `run_ablation_prompt.py` | Phase 4A | Prompt 变体消融实验 (default/strict/minimal 三种 prompt) |
| `run_ablation_repeats.py` | Phase 4B | 重复次数灵敏度分析 (k=1/2/3，无需额外 API 调用) |
| `_simulate_pilot_for_testing.py` | 测试 | 生成模拟标注数据，用于端到端验证管线 |
| `run_paper_pipeline.sh` | 全部 | 一键执行脚本，自动判断 GO/NO-GO 分支 |

### 评估模块 (`src/eval/`)

| 文件 | 修改内容 |
|------|---------|
| `human_labels_schema.py` | 新增 `compute_iaa_extended()` (完整 metric 套件)、`_krippendorff_alpha_ordinal()`、`iaa_go_nogo()` (自动 GO/HOLD 判断)、`compute_self_consistency()` (隐藏重复 QC) |

### 文档 (`docs/`)

| 文件 | 内容 |
|------|------|
| `rubric_v2.md` | 完整 5 档锚点 (v1 只有 1/3/5)，boundary 指引，boundary example |
| `annotation_guide_v2.md` | 每维度决策树、校准练习要求、扩展边界案例表、常见错误清单 |
| `PROJECT_STATUS.md` | 更新了完整执行指南和输出目录结构 |

## 3. 使用方法

### 3.1 环境准备

```bash
pip install scikit-learn scipy numpy openai
pip install mord  # 可选，ordinal 校准需要
```

### 3.2 一键执行（推荐）

```bash
# 完整流程（会在需要人工介入时停下来）
bash experiments/run_paper_pipeline.sh full

# 仅 pilot 阶段
bash experiments/run_paper_pipeline.sh pilot

# 仅分析阶段（需要已有全量人工标注）
bash experiments/run_paper_pipeline.sh analysis
```

### 3.3 分步执行

#### Phase 1: Pilot 标注

**Step 1 — 生成 pilot 标注包：**

```bash
python experiments/generate_pilot_annotation.py
```

输出目录 `outputs/human_annotation/pilot/`：
- `pilot_samples.csv` — 160 条样本 (150 正式 + 10 隐藏重复)，含 context 和 response
- `pilot_annotation_R1.csv` — 标注员 R1 的空白表
- `pilot_annotation_R2.csv` — 标注员 R2 的空白表
- `_pilot_mapping.json` — eval_id → sample_id 映射（**不要给标注员看**）
- `_duplicate_pairs.json` — 隐藏重复对列表

**Step 2 — 分发给标注员：**

将 `pilot_samples.csv` + 对应的 `pilot_annotation_Rx.csv` 发给两名标注员。标注员需要：
1. 阅读 `docs/annotation_guide_v2.md`  
2. 完成 `docs/rubric_v2.md` 中描述的校准练习  
3. 为每条样本在 4 个维度 (emotion, validation, helpfulness, safety) + overall 打 1-5 分

**Step 3 — 收回并分析 IAA：**

```bash
python experiments/run_pilot_iaa.py \
  --r1 outputs/human_annotation/pilot/pilot_annotation_R1.csv \
  --r2 outputs/human_annotation/pilot/pilot_annotation_R2.csv \
  --mapping outputs/human_annotation/pilot/_pilot_mapping.json \
  --duplicates outputs/human_annotation/pilot/_duplicate_pairs.json \
  --output_dir outputs/analysis
```

输出：
- `outputs/analysis/pilot_iaa_report.json` — 完整 metric JSON
- `outputs/analysis/pilot_iaa_report.md` — 可读报告，含 GO/NO-GO 判定

**GO/NO-GO 判定标准：**

| weighted κ | 判定 | 行动 |
|:---:|:---:|------|
| ≥ 0.40 | **GO** | 该维度通过 |
| 0.25 – 0.39 | REVISE | 补充 rubric 示例/边界规则，做 50 样本 mini-pilot |
| < 0.25 | REWRITE | 重写该维度的 rubric，做 50 样本 mini-pilot |

**总体判定**：所有 4 个维度 GO → 进入 Phase 2；任何维度 REVISE/REWRITE → 进入恢复流程。

#### Phase 1 (NO-GO 分支): 恢复流程

```bash
python experiments/nogo_recovery.py \
  --iaa_report outputs/analysis/pilot_iaa_report.json \
  --r1 outputs/human_annotation/pilot/pilot_annotation_R1.csv \
  --r2 outputs/human_annotation/pilot/pilot_annotation_R2.csv
```

自动生成 `outputs/nogo_recovery/`：

| 文件 | 用途 |
|------|------|
| `alignment_meeting_cases.md` | 标注对齐会讨论材料（per-dim top 15 分歧案例） |
| `disagreement_cases.csv` | 结构化分歧数据（含空白 consensus_score 和 resolution_note 列） |
| `rubric_revision_template.md` | Rubric 修订模板（自动填入常见混淆边界） |
| `mini_pilot/` | 50 样本 mini-pilot 包（含 5 个 trap 案例） |

**恢复步骤：**

1. **标注对齐会** (~30 min) — 用 `alignment_meeting_cases.md` 讨论分歧，填写共识分数
2. **修订 Rubric** — 用 `rubric_revision_template.md` 为模板，更新 `docs/rubric_v2.md`
3. **Mini-pilot 重测** — 分发 `mini_pilot/mini_annotation_R1.csv` & `R2.csv`，收回后再跑 `run_pilot_iaa.py`
4. 通过 → 进 Phase 2；再次不通过 → 最多再迭代 1 轮

#### Phase 1 (GO 分支): 生成全量标注

```bash
python experiments/generate_full_annotation.py \
  --n_per_model 200 \
  --n_duplicates 30 \
  --output_dir outputs/human_annotation/full
```

输出 `outputs/human_annotation/full/`：
- 630 条样本 (600 正式 + 30 隐藏重复)
- 采样策略：每模型 200 条，其中 top 30% 按 judge 方差优先选取

#### Phase 2: Judge↔Human 对齐分析

```bash
python experiments/judge_vs_human_analysis.py \
  --human outputs/labels/human/ratings_r1.csv,outputs/labels/human/ratings_r2.csv
```

输出：
- `outputs/analysis/judge_vs_human_raw.json` + `.md` — per-dim Spearman/Kendall/MAE/RMSE/bias
- `outputs/analysis/error_cases.json` — 按 response 长度、safety flag、模型分组的 error 分解 + top-20 极端 error cases

#### Phase 3: 论文级校准

```bash
python experiments/run_calibration_paper.py \
  --human outputs/labels/human/ratings_r1.csv,outputs/labels/human/ratings_r2.csv \
  --n_bootstrap 1000
```

输出 `outputs/analysis/`：
- `calibration_split.json` — 固定的 60/20/20 train/dev/test split
- `calibration_report_paper.json` + `.md` — 含 Isotonic 校准结果 + 95% bootstrap CI

#### Phase 4: 消融实验

**4A — Prompt 变体 (需 API)：**

```bash
# 先预览 prompt（不花钱）
python experiments/run_ablation_prompt.py --dry_run

# 实际运行
python experiments/run_ablation_prompt.py \
  --human outputs/labels/human/ratings_r1.csv,outputs/labels/human/ratings_r2.csv
```

**4B — 重复次数灵敏度 (无需 API)：**

```bash
python experiments/run_ablation_repeats.py \
  --human outputs/labels/human/ratings_r1.csv,outputs/labels/human/ratings_r2.csv
```

## 4. 输出目录结构

执行完整管线后，目录结构如下：

```
outputs/
├── analysis/
│   ├── pilot_iaa_report.json          # Phase 1: IAA 分析
│   ├── pilot_iaa_report.md
│   ├── judge_vs_human_raw.json        # Phase 2: 对齐分析
│   ├── judge_vs_human_raw.md
│   ├── error_cases.json
│   ├── calibration_split.json         # Phase 3: 校准
│   ├── calibration_report_paper.json
│   ├── calibration_report_paper.md
│   ├── ablation_repeats.json          # Phase 4B
│   └── ablation_repeats.md
├── human_annotation/
│   ├── pilot/                         # 150 样本 pilot 包
│   └── full/                          # 600 样本全量包 (GO 后生成)
├── nogo_recovery/                     # NO-GO 恢复材料
│   ├── alignment_meeting_cases.md
│   ├── disagreement_cases.csv
│   ├── rubric_revision_template.md
│   └── mini_pilot/                    # 50 样本 mini-pilot
├── generations/                       # 模型生成结果 (已有)
├── judge/                             # LLM Judge 评分 (已有)
└── calibrated/                        # 校准后结果 (已有)
```

## 5. 核心指标说明

| 指标 | 含义 | 论文报告用途 |
|------|------|-------------|
| weighted κ (linear) | 有序评分的加权 Cohen's kappa | IAA 主指标，决定 GO/NO-GO |
| Krippendorff α | 任意标注员数量的一致性度量 | IAA 辅助指标，鲁棒性检验 |
| Spearman ρ | 等级相关系数 | Judge↔Human 相关性 |
| MAE | 平均绝对误差 | 校准效果 |
| ECE | 期望校准误差 | 校准精度 |
| Bootstrap 95% CI | 1000× 重采样置信区间 | 所有指标的显著性检验 |

## 6. 常见问题

**Q: pilot 数据已经有了，为什么生成了 160 条而不是 150？**  
A: 多出的 10 条是隐藏重复样本 (hidden duplicates)，用于检测同一标注员的自我一致性。标注员不知道哪些是重复的。

**Q: NO-GO 了怎么办？**  
A: 这是正常的——pilot 的意义就是在小规模上发现问题。运行 `nogo_recovery.py` 后按指引修订 rubric，做一轮 mini-pilot 验证修订效果即可。

**Q: `_simulate_pilot_for_testing.py` 是什么？**  
A: 这是内部测试工具，用来生成模拟的标注数据以验证管线正确性。**不参与正式实验**，实际使用时不需要运行它。

**Q: Ablation A (prompt 变体) 需要多少 API 调用？**  
A: 需要额外 1200 次调用 (200 样本 × 3 模型 × 2 新 prompt)。已有的 default prompt 结果会复用，不会重复调用。先用 `--dry_run` 预览 prompt 再决定是否执行。

**Q: 全部跑完需要多少预算？**  
A: Judge 调用 1800 (已完成) + Ablation A 1200 (可选) = 最多 3000 次 DeepSeek API 调用。人工标注：2 人 × 600 样本。
