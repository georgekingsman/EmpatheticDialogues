# 面试项目展示指南 | Interview Project Presentation Guide
## Empathetic Dialogue Evaluation Framework

> 本文档为美国大学面试准备（PhD / RA / MS 面试），中英对照，帮助你全方位展示项目。  
> This guide is for US university interviews (PhD / RA / MS). Bilingual Chinese-English, covering all aspects of the project.
>
> 参考回答优先给出 **英文版**（面试语言），附中文注释帮助理解。  
> Suggested answers are given in **English first** (interview language), with Chinese notes for reference.

---

## 目录 | Table of Contents

- [一、30 秒电梯演讲 | 30-Second Elevator Pitch](#一30-秒电梯演讲--30-second-elevator-pitch)
- [二、3 分钟项目概述 | 3-Minute Project Overview](#二3-分钟项目概述--3-minute-project-overview)
- [三、技术深潜 | Technical Deep Dive](#三技术深潜--technical-deep-dive)
  - [3.1 数据管线 | Data Pipeline](#31-数据管线--data-pipeline)
  - [3.2 模型架构 | Model Architecture: Chain-of-Empathy](#32-模型架构--model-architecture-chain-of-empathy)
  - [3.3 LLM-as-a-Judge 评估管道 | LLM Judge Pipeline](#33-llm-as-a-judge-评估管道--llm-judge-pipeline)
  - [3.4 统计校准 | Statistical Calibration](#34-统计校准--statistical-calibration)
  - [3.5 消融实验 | Ablation Studies](#35-消融实验--ablation-studies)
- [四、工程能力展示 | Engineering Skills to Highlight](#四工程能力展示--engineering-skills-to-highlight)
- [五、研究思维展示 | Research Thinking to Demonstrate](#五研究思维展示--research-thinking-to-demonstrate)
- [六、面试高频问题 | Frequently Asked Interview Questions](#六面试高频问题--frequently-asked-interview-questions)
- [七、演示 Demo 流程 | Demo Walkthrough](#七演示-demo-流程--demo-walkthrough)
- [八、按面试场景调整重心 | Adjusting Focus by Interview Type](#八按面试场景调整重心--adjusting-focus-by-interview-type)
- [九、简历 Bullet Points | Resume Bullet Points](#九简历-bullet-points--resume-bullet-points)
- [附录：关键数字速查表 | Appendix: Key Numbers Cheat Sheet](#附录关键数字速查表--appendix-key-numbers-cheat-sheet)

---

## 一、30 秒电梯演讲 | 30-Second Elevator Pitch

> **背下来英文版，随时能讲。**  
> **Memorize the English version. Be ready to deliver it anytime.**

### English Version (面试时说这个)

"I built an evaluation framework for empathetic dialogue systems. The core question is: **how do you automatically and reliably measure whether an AI response is empathetic?** Traditional metrics like BLEU and ROUGE only measure lexical overlap — they can't capture empathy at all. And large-scale human annotation is expensive, slow, and hard to reproduce.

My approach is to design a **4-dimension scoring rubric**, use **LLM-as-a-Judge** for automated scoring, and then apply **statistical calibration** — specifically isotonic regression trained on public human-rated datasets — to align LLM scores with human judgments. The result: **MAE reduced by 31% to 63%**, while preserving rank correlation. The entire pipeline is fully reproducible at zero human annotation cost."

### 中文版（帮助理解逻辑）

"我做了一个共情对话评估框架。核心问题：怎么自动、可靠地评估 AI 回复的共情质量？BLEU/ROUGE 只衡量词匹配，评不了共情；人工标注贵、慢、不可复现。我的方案：四维评分量表 + LLM-as-a-Judge 自动打分 + 公开人工数据集做 isotonic regression 校准。MAE 降了 31%-63%，排序一致性不变，零人工标注成本。"

---

## 二、3 分钟项目概述 | 3-Minute Project Overview

> 按 **Problem → Approach → Implementation → Results → Contributions** 的故事线讲。  
> Follow the **P-A-I-R-C** narrative arc.

### 1) Problem 问题（30s）

**English:**
"Evaluating empathy in dialogue is an open challenge. Reference-based metrics like BLEU and ROUGE measure surface-level word overlap, but empathy is a semantic property — two sentences can be equally empathetic with completely different wording. Human annotation is the gold standard, but it's costly, time-consuming, and suffers from low inter-annotator agreement. In my project, the Krippendorff's alpha between human annotators was actually negative, showing how difficult it is for humans to reliably rate empathy."

**中文要点：**
- BLEU/ROUGE 是词匹配，衡量不了共情
- 人工标注贵/慢/一致性差（Krippendorff α 为负值）
- 这是一个开放性研究问题

### 2) Approach 方案（30s）

**English:**
"My solution has three components:
1. A **4-dimension rubric** — emotion recognition, validation & warmth, helpfulness, and safety — on a 1-5 Likert scale with detailed anchor descriptions.
2. An **LLM-as-a-Judge pipeline** — I use DeepSeek Chat to score each response 3 times with structured JSON output, enabling multi-repeat stability analysis.
3. **External human-anchored calibration** — rather than collecting our own expensive annotations, I train a calibrator on publicly available human-rated datasets and transfer it to align our LLM judge scores with human judgments."

**中文要点：**
- 四维量表（情感识别 / 验证温暖 / 实用性 / 安全边界）× 1-5 Likert
- LLM Judge（DeepSeek Chat，每样本 3 次，结构化 JSON）
- 外部人工锚定校准（公开数据集训练校准器 → 迁移到自有模型）

### 3) Implementation 实现（60s）

**English:**
"I trained 3 contrastive models on 5,318 mental health counseling dialogues from PsychCentral:
- **Vanilla GPT-2** as a lower bound — no fine-tuning at all.
- **Fine-tuned GPT-2** as a standard baseline.
- **GPT-2 + Chain-of-Empathy** — my custom architecture that adds a 5-stage empathy reasoning chain inspired by CBT cognitive processes, fused into GPT-2 via additive injection.

For evaluation, I executed **1,800 LLM API calls** — 200 test samples × 3 models × 3 repeats — with zero parsing failures. I then implemented two calibration routes: isotonic regression and ordinal logistic regression. I also ran ablation studies on repeat sensitivity and prompt variant robustness."

**中文要点：**
- 5,318 条心理咨询对话，3 个对照模型
- Chain-of-Empathy：5 阶段推理链 + 加性融合
- 1,800 次 API 调用，0 错误
- 双路校准 + 消融实验

### 4) Results 结果（30s）

**English:**

| Metric | Value |
|--------|-------|
| Judge self-consistency | Exact: 88-100%, ±1: 96-100% |
| Calibration MAE reduction | 31.3% – 62.8% (isotonic) |
| Post-calibration MAE range | 0.20 – 0.29 (on a 1-5 scale) |
| Spearman rank correlation | Preserved after calibration (0.32 – 0.79) |
| Optimal repeat count | k=1 sufficient (additional repeats: marginal gain < 0.01) |

"The key finding is that isotonic regression reduces MAE by over 50% on average, **while preserving rank correlation** — meaning the calibration corrects scale bias without hurting the judge's discriminative ability."

**中文要点：**
- 校准后 MAE 降 31-63%，排序不变
- Judge 极度自洽（≥96% near agreement）
- k=1 够用 → 可省 66% API 成本

### 5) Contributions 贡献（30s）

**English:**
"Three contributions:
1. **Methodological**: I propose external human-anchored calibration — using public datasets as an unbiased anchor instead of collecting your own annotations.

公开数据集（已有人工评分）
    ↓
同一个 Judge 给它打分
    ↓
对比：Judge 分 vs 人工分 → 训练校准器
    ↓
把校准器应用到你自己的模型 → 得到校准后的分数

2. **Engineering**: A complete end-to-end pipeline from data processing through model training, generation, evaluation, to calibration — fully reproducible with shell scripts.
3. **Practical**: The ablation shows k=1 repeat suffices, cutting API costs by 66%. The calibrated scores have MAE under 0.3 on a 5-point scale, making them practically usable."

**中文要点：**
- 方法论创新：外部锚定校准
- 工程完整性：端到端可复现管线
- 实用价值：k=1 够用，成本优化

---

## 三、技术深潜 | Technical Deep Dive

> 面试时根据教授兴趣选择深入某个模块。  
> During the interview, dive deeper into whichever module the professor is interested in.

### 3.1 数据管线 | Data Pipeline

**Key Files:** `src/data/build_dataset.py`, `src/data/external_loader.py`

**What to say / 讲什么：**

| Topic 话题 | English | 中文 |
|---|---|---|
| Data source 数据来源 | 5,318 patient-therapist Q&A records from PsychCentral | 5,318 条 PsychCentral 心理咨询对话 |
| Preprocessing 预处理 | Lowercasing, denoising, tokenization, cognitive distortion detection | 小写化、去噪、分词、认知扭曲检测 |
| Label masking 标签掩码 | Prompt tokens set to `-100`; loss computed only on therapist response | Prompt tokens 设为 -100，loss 只算在治疗师回复上 |
| Data split 数据划分 | 80/10/10 (train/val/test) | 80/10/10 |
| External loader 外部加载 | Supports EPITOME, generic CSV/JSONL; auto-normalizes to 1-5 scale | 支持 EPITOME/通用格式，自动归一化到 1-5 |

**Design decisions to mention / 可提的设计决策：**

- **English:** "Why label masking? To prevent the model from simply learning to copy user input. The loss is only computed on the therapist's response tokens."
- **中文：** 为什么用 label masking → 防止模型学复制 user 输入

- **English:** "Why external data for calibration anchoring? Using your own annotations to calibrate your own judge creates circularity bias. External public datasets provide an independent, unbiased anchor."
- **中文：** 为什么用外部数据 → 消除"自己标注自己评"的偏差

---

### 3.2 模型架构 | Model Architecture: Chain-of-Empathy

**Key File:** `src/models/empathy_chain.py`

**Architecture (可以在白板上画 / draw on whiteboard):**

```
Input Hidden States
    ↓ mean pooling → sentence-level representation (B, H)
    ↓
┌──────────────────────────────────────────────┐
│  Stage 1: Scenario Understanding             │  Linear + ReLU
│  Stage 2: Emotion Recognition                │  Linear + ReLU
│  Stage 3: Cause Inference                    │  Linear + ReLU
│  Stage 4: Goal Setting                       │  Linear + ReLU
│  Stage 5: Response Planning                  │  Linear (no activation)
└──────────────────────────────────────────────┘
    ↓
Emotion-Scenario Fusion Layer:
    concat(emotion_rep, scenario_rep) → Linear(2H, H) + ReLU
    ↓
Residual Connection: response_rep + fused_rep
    ↓
Broadcast-add to GPT-2 hidden states → lm_head → logits
```

**Key design points / 关键设计点：**

| Question 面试官可能问 | Answer 回答 |
|---|---|
| Why 5 stages? 为什么5阶段？ | "Inspired by CBT (Cognitive Behavioral Therapy) — mirrors how humans develop empathetic responses: understand context → recognize emotion → infer cause → set goal → formulate response." 模拟 CBT 认知行为疗法中人类产生共情的认知过程 |
| Why additive fusion, not cross-attention? 为什么加性融合？ | "At GPT-2 scale (124M params), additive fusion is parameter-efficient and converges faster. Cross-attention would add significant overhead for minimal gain at this scale." 在 124M 规模下，加性融合参数少、收敛快 |
| Why no activation in Stage 5? 第5阶段为什么没激活函数？ | "The final stage needs to preserve both positive and negative activations for maximum representational capacity before fusion." 保留正负值信息 |
| Initialization? 初始化？ | "Xavier Uniform for all custom layers — ensures stable gradients in the chain." Xavier Uniform，保证梯度稳定 |

**Bonus point / 面试加分点：**

> **English:** "I want to be upfront: this model's absolute performance is limited — GPT-2 at 124M parameters scores only 1-2 out of 5 on empathy dimensions, which is expected for such a small model on complex therapeutic dialogue. But its value is as an **ablation control**: same data, same training loop, the only variable is the presence of the empathy reasoning chain. This lets us validate that the Judge can discriminate between models."

> **中文：** 模型本身分数很低（GPT-2 太小），但它的价值在于作为消融对照，验证 Judge 的判别能力。

---

### 3.3 LLM-as-a-Judge 评估管道 | LLM Judge Pipeline

**Key Files:** `src/eval/llm_judge.py`, `src/eval/rubric.py`

**Pipeline (可以画流程图 / draw this as a flowchart):**

```
rubric.py → rubric_to_text() → Render rubric as Markdown
    ↓
build_judge_messages() → System prompt + User message
    ↓
judge_one() → API call (DeepSeek / GPT-4)
    ↓
extract_json() → Robust JSON extraction (strip markdown fences, regex match)
    ↓
validate_judge_output() → Validate dimensions + score range + fallback
    ↓
judge_batch() → Batch processing + n_repeats=3 (stability analysis)
    ↓
save_judge_results() → JSONL output with full metadata
```

**Key design points / 关键设计点：**

| Pattern 设计模式 | English | 中文 |
|---|---|---|
| Single Source of Truth | Rubric defined once in `rubric.py`, shared by human annotators and LLM Judge | 评分标准唯一定义，人工和 LLM 共用 |
| Strategy Pattern | `api_fn` parameter decouples API backends — swap DeepSeek / OpenAI / local without code changes | `api_fn` 解耦 API 后端 |
| Robust JSON Parsing | `extract_json()` handles markdown code fences, extraneous text, format anomalies | 鲁棒 JSON 提取，处理各种 LLM 输出格式 |
| Retry with backoff | Exponential backoff + max retries for transient API failures | 指数退避重试 |
| Full metadata | Each result includes timestamp, judge model, repeat index, confidence | 完整元数据追踪 |

**Key numbers / 关键数字：**
- 1,800 API calls, **0 failures** | 1,800 次调用，0 失败
- Exact self-consistency: **88-100%** | 精确一致率 88-100%
- Near agreement (±1): **96-100%** | ±1 一致率 96-100%

---

### 3.4 统计校准 | Statistical Calibration

**Key Files:** `src/eval/calibrate.py`, `experiments/train_external_calibrator.py`

**Core idea (画这个图 / draw this diagram):**

```
                   Ideal: y = x
                    ╱
 Calibrated Score  ╱
                  ╱
                 ╱  ← Isotonic Regression
                ╱     (monotone staircase mapping)
               ╱
              ╱
             ╱
            ╱ • • •  ← Raw judge scores (systematic bias)
           ╱
          ╱_______________
                Human Score
```

**Two calibration routes compared / 双路校准对比：**

| Aspect 维度 | Isotonic Regression 等张回归 | Ordinal Logistic Regression 有序逻辑回归 |
|---|---|---|
| Method 方法 | Non-parametric monotone mapping 非参数单调映射 | Ordered classification probability distribution 有序分类概率 |
| Pros 优点 | Assumption-free, preserves rank order 无假设，保序 | Outputs probability distribution 输出概率分布 |
| Cons 缺点 | Needs sufficient samples 需要足够样本 | Needs more features & samples 需要更多特征和样本 |
| Result in this project 本项目结果 | **MAE reduced 31-63%** | Failed — training set too small (n=60) 过拟合 |

**Calibration results / 校准结果：**

| Dimension 维度 | Raw MAE → Cal MAE | Reduction 降幅 | Spearman (before → after) |
|---|:---:|:---:|:---:|
| Emotion Recognition 情感识别 | 0.547 → 0.205 | **62.8%** | 0.61 → 0.58 |
| Validation & Warmth 验证温暖 | 0.544 → 0.249 | **54.1%** | 0.32 → 0.33 |
| Helpfulness 实用性 | 0.506 → 0.219 | **56.8%** | 0.76 → 0.76 |
| Safety 安全边界 | 0.425 → 0.285 | **31.3%** | 0.79 → 0.78 |

**Highlight phrasing for interview / 面试亮点说法：**

> **English:** "Isotonic regression cut MAE by more than half on average, and **Spearman rank correlation remained essentially unchanged** — meaning calibration only corrects scale bias without hurting the judge's discriminative ability. This tells us the LLM judge's **ranking is reliable**, it just has an absolute value offset."

> **中文：** 校准只调了尺度偏差，排序不变 → Judge 的判别能力本身是可靠的。

---

### 3.5 消融实验 | Ablation Studies

#### Ablation A: Repeat Sensitivity (k=1 vs k=2 vs k=3) | 重复次数敏感性

| k | Emotion Spearman | Safety Spearman | Emotion MAE |
|---|:---:|:---:|:---:|
| 1 | 0.658 | 0.855 | 0.201 |
| 2 | 0.661 | 0.871 | 0.206 |
| 3 | 0.651 | 0.875 | 0.204 |

> **English:** "k=1 is sufficient — going from 1 to 3 repeats yields marginal improvement of less than 0.01 on all metrics. This means we can **cut API costs by 66%** without sacrificing calibration quality."

> **中文：** k=1 够用，额外重复边际收益 < 0.01 → 省 66% API 成本。

#### Ablation B: Prompt Variants (default / strict / minimal) | Prompt 变体

- Tests how different system prompt styles affect score distributions
- Validates that the rubric is **robust across prompt formulations**
- 测试不同 prompt 风格对评分分布的影响，验证 rubric 的鲁棒性

---

## 四、工程能力展示 | Engineering Skills to Highlight

> 对 PhD 面试，工程能力说明你能独立落地研究 idea。  
> For PhD interviews, engineering shows you can independently implement research ideas.

### 4.1 项目架构 | Project Architecture

```
src/
├── data/          ← Data layer (templates, dataset builder, external loader)
│                    数据层（模板、数据集构建、外部加载器）
├── models/        ← Model layer (baseline GPT-2, Chain-of-Empathy, unified trainer)
│                    模型层（基线 GPT-2、共情链、统一训练器）
├── inference/     ← Inference layer (unified generation interface, local + API)
│                    推理层（统一生成接口，本地 + API）
└── eval/          ← Evaluation layer (rubric, llm_judge, calibrate, metrics)
                     评估层（量表、LLM 评委、校准、指标）
                     ↑ Core innovation lives here / 核心创新在这一层

experiments/       ← Experiment scripts (reproducible shell + Python)
                     实验脚本（可复现的 shell + Python）
outputs/           ← Output layer (structured JSONL + analysis reports)
                     输出层（结构化 JSONL + 分析报告）
```

**Points to make / 要点：**

| English | 中文 |
|---|---|
| Modular layered design (data → model → inference → eval) | 模块化分层 |
| Single source of truth (`rubric.py` is the global reference) | 单一事实来源 |
| Strategy pattern (`api_fn` decouples API backends) | 策略模式解耦 |
| One-command reproducibility via shell scripts | 一键复现 |
| Complete metadata tracking (model name, seed, temperature, timestamp per output) | 完整元数据 |

### 4.2 代码质量亮点 | Code Quality Highlights

| Feature 特性 | English | 中文 |
|---|---|---|
| Robust JSON parsing | `extract_json()` handles markdown fences, extraneous text, format edge cases. 1,800 calls, 0 parse failures. | 鲁棒 JSON 解析，1800 次 0 失败 |
| Resume-from-checkpoint | `--resume` flag skips already-processed samples — critical for large-scale API work | 断点续传，大规模 API 不怕中断 |
| Label masking | Prompt tokens masked with `-100` — standard NLP fine-tuning practice | 标准 NLP 训练实践 |
| Xavier initialization | Custom modules use proper weight init for gradient stability | 合理权重初始化 |
| Bootstrap CI | Calibration results come with 95% confidence intervals (1,000 bootstrap iterations) | 校准结果带 95% CI |

### 4.3 重构经验 | Refactoring Experience

> **English:** "The project evolved from a prototype to production-quality code. Legacy files in the root directory used Chinese GPT-2 and monolithic scripts. I refactored everything into a modular `src/` package with unified interfaces, so the full pipeline can be reproduced with a single command."

> **中文：** 从原型重构到生产级代码：模块化、统一接口、一键复现。

| Legacy 旧文件 | New 新文件 | Improvement 改进 |
|---|---|---|
| `Chain_of_Empathy.py` | `src/models/empathy_chain.py` | Integrated into full model class 集成到完整模型 |
| `Model_Baseline.py` | `src/models/baseline_gpt2.py` | Unified interface 统一接口 |
| `Train_and_Test.py` | `src/models/train.py` | Multi-model support 多模型支持 |
| Scattered scripts 零散脚本 | `experiments/*.py` | Structured experiment pipeline 结构化管线 |

---

## 五、研究思维展示 | Research Thinking to Demonstrate

### 5.1 正确定位研究贡献 | Positioning the Contribution

> **不要说 "I built a better empathy chatbot"。**  
> **Do NOT say "I built a better empathy chatbot."**

**Instead, say / 应该说：**

> **English:** "This is an **evaluation framework** project, not a model project. The core contribution is: how to obtain reliable empathy scores using LLM-as-a-Judge plus statistical calibration, at **zero human annotation cost**. The three models are ablation subjects to validate the evaluation framework — they are not the point."

> **中文：** 这是**评估框架**项目，不是模型项目。模型只是验证评估框架的消融对象。

### 5.2 方法论亮点 | Methodological Highlights

| Insight 洞察 | English | 中文 |
|---|---|---|
| External vs self-anchoring 外部锚定 vs 自我锚定 | "Annotating your own data and then calibrating your own judge creates circularity bias. External public datasets provide independent, reproducible anchors." | 自己标注自己评有循环偏差，外部公开数据是独立锚点 |
| Calibration, not replacement 校准而非替代 | "I'm not claiming LLM judge equals human. I'm saying the judge's **ranking is reliable** — it just needs scale correction via calibration." | LLM Judge 排序可靠，只需调尺度 |
| Ablation mindset 消融思维 | "The 3 models aren't about finding the best model. They're about validating that the judge can discriminate quality differences. The repeat ablation proves k=1 suffices, saving 66% cost." | 3 个模型是为了验证 Judge，k=1 够用省 66% 成本 |

### 5.3 主动面对局限 | Proactively Address Limitations

> 面试中**主动提出局限**比被问到更加分。  
> **Volunteering** limitations is more impressive than being caught off guard.

| Limitation 局限 | Honest Assessment 诚实评估 | Future Work 未来方向 |
|---|---|---|
| Small model scale 模型规模小 | "GPT-2 at 124M only scores 1-2/5 on empathy — expected for this scale." GPT-2 124M 分数低是预期内的 | Research goal is the evaluation framework, not the model 研究目标是评估框架 |
| Small calibration set 校准样本少 | "n=60 caused ordinal regression to overfit completely." n=60 让 Ordinal 过拟合 | Need larger external datasets 需要更大外部数据集 |
| Validation dimension 验证维度难 | "Spearman only 0.32 — lowest among dimensions." | May need finer-grained rubric anchors 可能需要更精细的 rubric |
| Single judge model 单一 Judge | "Only used DeepSeek Chat." 只用了 DeepSeek | Extend to cross-judge validation (GPT-4, Claude) |
| Domain specificity 领域特定 | "Mental health counseling — generalizability untested." 只测了心理咨询 | Extend to customer service, education 可扩展到客服/教育 |

---

## 六、面试高频问题 | Frequently Asked Interview Questions

> 每个问题给出 **英文口语版回答**（面试时直接用）+ 中文理解注释。  
> Each question provides an **English spoken answer** (use directly) + Chinese notes.

---

### Q1: "Why not just use BLEU/ROUGE?"

> **English:** "BLEU and ROUGE measure lexical overlap with reference responses, but empathy is a semantic property. For example, 'I understand how you feel' and 'Your feelings are completely valid to me' express the same empathy, but BLEU would score near zero. My own data confirms this: the fine-tuned model has a BLEU of only 0.016, but the LLM judge rates its empathy significantly higher than vanilla GPT-2. This dimension simply isn't captured by n-gram overlap metrics."

> **中文注释：** BLEU 衡量词匹配，但共情是语义属性。数据证实 Finetuned BLEU 只有 0.016，但 Judge 评分显著更高。

---

### Q2: "Is LLM-as-a-Judge reliable? Doesn't it hallucinate?"

> **English:** "Great question — that's exactly why I built the multi-repeat and calibration pipeline. Empirically, the judge shows near-agreement rates of 96-100% across 3 independent repeats, which is actually higher than typical human inter-rater agreement on subjective tasks. But there is a systematic scale bias, which is why I apply isotonic regression calibration. After calibration, MAE drops by 31-63%, but rank correlation stays essentially unchanged — meaning the judge's **discriminative ability is reliable**, it just needs scale correction."

> **中文注释：** Judge 自一致率 96-100%，比人类标注者都高。但有系统偏差 → 校准修正尺度，排序不变 → 判别能力可靠。

---

### Q3: "Why isotonic regression? Why not something more sophisticated?"

> **English:** "Three reasons:
> 1. **Non-parametric** — it makes no assumption about the functional form between judge and human scores.
> 2. **Order-preserving** — it guarantees that the calibrated ranking is identical to the original ranking, which is critical for evaluation.
> 3. **Sample-efficient** — I only had 60 training samples. I actually tried ordinal logistic regression as well, and it completely overfitted. Isotonic regression is robust to small samples.
> 
> It's a principled engineering decision: choose the simplest method that works given the constraints."

> **中文注释：** 三个理由：无假设、保序、对小样本鲁棒。Ordinal 在 n=60 上过拟合了。工程原则：约束下选最简方案。

---

### Q4: "Chain-of-Empathy performs worse than the baseline?"

> **English:** "On the surface, yes — Empathy Chain scores 1.28 overall versus 1.33 for fine-tuned baseline. But both scores are extremely low — 1-2 out of 5 — because GPT-2 at 124M parameters simply doesn't have enough capacity for complex therapeutic dialogue. The difference is within the confidence interval.

> The correct interpretation is three-fold: First, GPT-2 scale is insufficient for this task. Second, the Chain-of-Empathy architecture needs to be validated on larger models — 7B+ — to see its true benefit. Third, and most importantly for this project, **the model comparison validates that the judge can discriminate** — vanilla scores 1.0, fine-tuned scores 1.33. That's the purpose of the models in this framework."

> **中文注释：** 两者绝对分数都很低（GPT-2 太小），差异在 CI 内。重要的是 Judge 能区分 vanilla (1.0) vs finetuned (1.33)。

---

### Q5: "What about domain shift between external calibration data and your own data?"

> **English:** "Excellent question. This is a real risk. My mitigation strategy has three layers:
> 1. The rubric uses **domain-general empathy dimensions** — emotion recognition and validation are universal, not domain-specific.
> 2. The calibrator learns the **judge's systematic bias pattern**, not domain features. If the judge consistently overrates emotion recognition, that bias transfers across domains.
> 3. For future work, a small amount of in-domain human annotation could enable domain adaptation — even 50-100 samples would significantly reduce domain shift. Or we can select public datasets closer to our domain."

> **中文注释：** 三层应对：通用维度、学的是 Judge 偏差模式、未来可少量 in-domain 标注做 adaptation。

---

### Q6: "What would you do with more time/resources?"

> **English:** "Three things, in priority order:
> 1. **Cross-judge validation** — run the same rubric with DeepSeek, GPT-4, and Claude, then analyze whether different judges have different bias patterns. This would tell us whether the calibration is judge-specific or generalizable.
> 2. **Scale up the model** — validate Chain-of-Empathy on LLaMA-7B or 13B to see if the architecture truly helps at sufficient scale.
> 3. **Interactive user study** — build a web interface where real users chat with different models, use the judge pipeline for real-time quality monitoring, and correlate automated scores with user satisfaction. This would close the loop between automated evaluation and actual user experience."

> **中文注释：** 三件事：Cross-Judge 验证、更大模型验证、真人交互用户研究。

---

### Q7: "What's your publication plan?"

> **English:** "The contribution supports a workshop paper or short paper. Target venues include the EMNLP/ACL Workshop on NLP for Mental Health, or AIES — AI Ethics and Society. The paper structure would be: problem definition → rubric design → judge reliability analysis → external-anchored calibration → ablation studies → practical guidelines for using LLM judges in evaluation."

> **中文注释：** 可以投 EMNLP/ACL mental health workshop 或 AIES。

---

### Q8: "What was the biggest challenge?"

> **English:** "Two challenges stand out:
> 1. **Robust LLM output parsing** — different API versions return different formats: sometimes markdown code fences, sometimes extra explanatory text. I wrote an `extract_json()` function with regex fallback that handles all edge cases. 1,800 calls with zero parse failures.
> 2. **Calibration with limited data** — with only 60 training samples, ordinal logistic regression completely overfitted. This forced a principled engineering decision: sometimes the simplest method — isotonic regression — is the right choice when you're data-constrained. That's a lesson I internalized through this project."

> **中文注释：** 两个挑战：LLM 输出解析（1800 次 0 失败）+ 小样本校准（最简方案反而最好）。

---

### Q9: "How does this relate to Professor X's research?" (量身定制 Customize this)

> **准备策略 / Preparation Strategy：**
> 1. Read professor's 2-3 most recent papers 提前读教授最近 2-3 篇论文
> 2. Find intersection points 找交叉点, for example:
>    - If professor does **NLG evaluation**: "My calibration approach could extend to their evaluation framework..."
>    - If **AI safety**: "The safety dimension in my rubric directly connects to..."
>    - If **human-AI interaction**: "My judge pipeline could serve as a real-time quality monitor in..."
>    - If **computational social science**: "The empathy measurement framework could be applied to..."
>    - If **mental health NLP**: "My evaluation framework directly addresses the core challenge of measuring therapeutic quality..."

---

### Q10: "Can you walk me through the code?"

> **English:** "Sure. The project has a clean layered architecture:
> - `src/data/` handles data loading and preprocessing — 5,318 dialogues, with label masking so loss is only computed on therapist responses.
> - `src/models/` has two model classes: a thin GPT-2 wrapper and the Chain-of-Empathy model with 5-stage reasoning.
> - `src/eval/` is where the core contribution lives — rubric definitions, the LLM judge pipeline with structured JSON output, and the calibration module supporting isotonic and ordinal methods.
> - `experiments/` contains reproducible experiment scripts — each one is independently runnable and idempotent.
> 
> Would you like me to dive into any specific module?"

> **中文注释：** 简洁描述分层，然后让教授选择深入哪一层。主动引导对话。

---

## 七、演示 Demo 流程 | Demo Walkthrough

### 有电脑 / 屏幕共享 | With Computer / Screen Share

#### Step 1: Show Project Structure 展示项目结构 (1min)
```bash
tree src/ -L 2   # Show clean module layout
```

#### Step 2: Show Data Samples 展示数据 (1min)
```bash
# Show one training sample 展示一条训练数据
head -1 data/formatted_Psych_data.jsonl | python -m json.tool

# Show one generation output 展示一条生成结果
head -1 outputs/generations/empathy_chain.jsonl | python -m json.tool
```

#### Step 3: Show Judge Results 展示评分结果 (2min)
```bash
# Score distribution 分数分布
python experiments/quick_score_dist.py

# Or full analysis 完整分析
python experiments/analyse_judge_results.py 2>/dev/null | head -60
```

#### Step 4: Show Calibration 展示校准 (2min)
```bash
cat outputs/analysis/calibration_report_paper.md
```

#### Step 5: Show Ablation 展示消融 (1min)
```bash
cat outputs/analysis/ablation_repeats.md
```

### 无设备 / 幻灯片 | Without Computer / Slides Only

建议 **5-7 页** slides / Recommend **5-7 slides**:

| Slide | English Content | 中文内容 |
|---|---|---|
| 1 | Title + one-line summary: "Automated Evaluation Framework for Empathetic Dialogue" | 标题 + 一句话 |
| 2 | Problem: BLEU fails → Human costly → Need LLM Judge | 问题：BLEU 不够 → 人工太贵 → LLM Judge |
| 3 | System architecture diagram: Data → Models → Generation → Judge → Calibration | 架构图 |
| 4 | Chain-of-Empathy: 5-stage reasoning chain + fusion | 5 阶段推理链 + 融合机制 |
| 5 | Calibration results table: MAE reduced 31-63%, Spearman preserved | 校准结果 |
| 6 | Ablation: k=1 suffices → 66% cost reduction | 消融结论 |
| 7 | Contributions + 3 future directions | 贡献 + 未来方向 |

---

## 八、按面试场景调整重心 | Adjusting Focus by Interview Type

### 🎓 PhD Interview / PhD 面试

**Emphasize / 重点展示：**
- Research motivation and problem formulation 研究动机与问题定义
- Methodological novelty (external anchored calibration, ablation design) 方法论创新
- Statistical rigor (bootstrap CI, IAA analysis, GO/NO-GO gates) 统计严谨性
- Paper-writing ability (rubric design, annotation protocol) 论文写作能力
- Future research directions and how they connect to advisor's work 未来方向与导师方向的交集
- Intellectual honesty about limitations 诚实面对局限性

**De-emphasize / 少讲：**
- Engineering details (code structure, refactoring history) 工程细节

### 🔬 RA (Research Assistant) Interview / RA 面试

**Emphasize / 重点展示：**
- Independent research execution ability 独立落地能力
- Full-stack skills: model training + API integration + statistical analysis 全栈能力
- Reproducibility and documentation quality 可复现性与文档质量
- Ability to handle real-world constraints (small data, API costs) 实际约束处理
- Self-directed problem solving 自主解决问题

**De-emphasize / 少讲：**
- Deep theoretical motivation 过深的理论动机

### 💻 Industry ML / NLP Engineer Interview / 工业界面试

**Emphasize / 重点展示：**
- End-to-end system design 端到端系统
- Code quality (modular, resume-from-checkpoint, robust parsing, metadata) 代码质量
- Cost consciousness (ablation proves k=1 → 66% cost reduction) 成本意识
- Prototype-to-production refactoring experience 重构经验
- API integration experience (DeepSeek, OpenAI) API 集成

**De-emphasize / 少讲：**
- Mathematical details of statistical methods 统计方法的数学细节

### 🤖 AI Safety / AI Ethics 岗位

**Emphasize / 重点展示：**
- Safety dimension design philosophy (boundary awareness, professional referral) Safety 维度
- LLM Judge reliability and bias analysis Judge 偏差分析
- Fairness implications of calibration (unbiased anchoring) 校准的公平性
- HAI evaluation methodology HAI 评估方法论

---

## 九、简历 Bullet Points | Resume Bullet Points

> 根据目标岗位选一组。  
> Pick the set that matches your target position.

### General / 通用版
- Designed and implemented an end-to-end evaluation framework for empathetic dialogue systems, using LLM-as-a-Judge with statistical calibration (isotonic regression), reducing score MAE by 31-63% against human ratings with zero manual annotation cost
- Built a 5-stage Chain-of-Empathy neural module for GPT-2, modeled after CBT cognitive processes, with additive fusion into transformer hidden states
- Executed 1,800 LLM API evaluations with 0 errors, achieving 96-100% self-consistency rate across 3 independent repeats

### Research-Oriented / 研究版
- Proposed external human-anchored calibration for LLM-as-a-Judge, training isotonic/ordinal calibrators on public datasets (EPITOME) and transferring to own model outputs; achieved MAE reduction of 31-63% with preserved rank correlation (Spearman 0.32-0.79)
- Designed 4-dimension empathy evaluation rubric (emotion recognition, validation & warmth, helpfulness, safety) as single source of truth for both human annotators and LLM judge, with inter-annotator agreement protocol (weighted κ, Krippendorff α)
- Conducted ablation studies on judge repeat sensitivity (k=1/2/3) and prompt variants, demonstrating k=1 sufficiency for 66% API cost reduction without calibration quality degradation

### Engineering-Oriented / 工程版
- Architected modular NLP evaluation pipeline (Python, PyTorch, HuggingFace): data processing → model training → batch inference → LLM judge → statistical calibration, with shell scripts for full reproducibility
- Implemented robust LLM output parsing with regex-based JSON extraction, exponential backoff retry, and resume-from-checkpoint, handling 1,800 API calls with zero failures
- Refactored legacy monolithic codebase into layered architecture (data/models/inference/eval), unified training interface supporting multiple model types, and standardized JSONL output with complete metadata tracking

---

## 附录：关键数字速查表 | Appendix: Key Numbers Cheat Sheet

> 面试前过一遍，确保英文能脱口而出。  
> Review before interview — make sure you can say these numbers fluently in English.

| Item 项目 | Value 数值 | How to say it 英文口语 |
|---|---|---|
| Training data 训练数据 | 5,318 dialogues | "About fifty-three hundred mental health counseling dialogues" |
| Model size 模型参数 | GPT-2 124M | "GPT-2 with 124 million parameters" |
| Test samples 测试样本 | 200 per model | "Two hundred test samples per model" |
| Number of models 模型数 | 3 | "Three contrastive models: vanilla, fine-tuned, and empathy chain" |
| Total API calls API 总调用 | 1,800 | "Eighteen hundred API calls — two hundred times three models times three repeats" |
| API errors 错误数 | 0 | "Zero failures" |
| Scoring dimensions 评分维度 | 4 | "Four dimensions: emotion, validation, helpfulness, and safety" |
| Likert scale 量表 | 1-5 | "One-to-five Likert scale" |
| Judge exact agreement 精确一致 | 88-100% | "Eighty-eight to one hundred percent exact agreement" |
| Judge near agreement (±1) | 96-100% | "Ninety-six to one hundred percent near agreement" |
| Isotonic MAE reduction 降幅 | 31-63% | "Thirty-one to sixty-three percent MAE reduction" |
| Post-calibration MAE 校准后 | 0.20-0.29 | "Point two to point two nine on a five-point scale" |
| Optimal k (repeats) 最优重复数 | k=1 | "k equals one is sufficient" |
| Bootstrap iterations | 1,000 | "One thousand bootstrap iterations" |
| Calibration train set 校准训练集 | 60 | "Sixty training samples" |
| Calibration test set 校准测试集 | 20 | "Twenty test samples" |
| Best BLEU (finetuned) | 0.016 | "Point zero one six" |
| Best ROUGE-1 (finetuned) | 0.297 | "Point two nine seven" |
| Vanilla judge overall | 1.00 | "One point zero" |
| Finetuned judge overall | 1.33 | "One point three three" |
| Empathy Chain judge overall | 1.28 | "One point two eight" |

---

## 面试前 Checklist ✅ | Pre-Interview Checklist

- [ ] 能流利说出 30s Elevator Pitch (English) | Can deliver 30s pitch fluently in English
- [ ] 能画 Chain-of-Empathy 架构图 | Can draw Chain-of-Empathy architecture on whiteboard
- [ ] 能画 Calibration 核心概念图 | Can draw calibration concept diagram
- [ ] 能脱口而出 5 个关键数字 (English) | Can cite 5 key numbers from memory in English
- [ ] 能回答 "why not BLEU" in English | Can answer "why not BLEU" fluently
- [ ] 能回答 "is LLM judge reliable" in English | Can answer "is LLM judge reliable" fluently
- [ ] 能主动说出 3 个局限性 in English | Can proactively state 3 limitations in English
- [ ] 能说出 3 个未来方向 in English | Can state 3 future directions in English
- [ ] 已阅读目标教授最近 2-3 篇论文 | Have read target professor's 2-3 recent papers
- [ ] 准备了项目与教授方向的交叉点 | Prepared intersection points with advisor's research
- [ ] 测试过 Demo 命令能否运行 | Tested that demo commands run successfully
- [ ] 准备了 5-7 页 backup slides (English) | Prepared 5-7 backup slides in English
