# 演示指南 — Empathetic Dialogue Evaluation Framework

> **演示时长 Duration**：15–20 分钟（可根据需要调整）  
> **演示目标 Goal**：展示项目与导师方向的 Fit、核心创新、实验结果与工程能力  
> **PPT**：`docs/project_slides.html`（共 11 页，Slides 0–10）

---

## 一、演示前准备（提前 5 分钟）

### 1.1 打开终端，激活环境

```bash
cd ~/Desktop/EmpatheticDialogues
source .venv/bin/activate
```

### 1.2 提前准备好以下窗口

| 窗口 | 内容 | 用途 |
|------|------|------|
| **浏览器** | 打开 `docs/project_slides.html` | PPT 演示 |
| **终端 1** | 项目根目录 | 运行 demo 脚本 |
| **VS Code** | 打开项目 | 展示代码结构 |

### 1.3 测试 demo 脚本能跑通

```bash
python demo_live.py --quick   # 先用快速模式测试
```

---

## 二、演示流程（10 个环节，对应 Slides 0–9）

### 环节 0: Research Fit + 开场（Slides 0–1）— 2 分钟

**操作**：浏览器展示 Slide 0 → Slide 1

**要说的话（中英对照）**：

> **中**："首先介绍一下我和这个研究方向的契合度。老师的方向是 Human–AI Interaction，关注信任、接受度和交互系统。我这个项目正好是做 calibrated measurement——校准后的自动化评估，可以直接集成到 HAI 用户实验平台中。"
>
> **EN**: "Let me start with why I'm a good fit. Your focus is Human–AI Interaction — trust, acceptance, and interactive systems. My project is about calibrated measurement — automated evaluation anchored to human judgment, ready to integrate into HAI experiment platforms."

**翻到 Slide 1**：

> **中**："项目的全名是 Automated Evaluation Framework for Empathetic Dialogue——用 LLM-as-a-Judge 加统计校准来自动评估心理咨询对话的共情质量。零人工标注成本，MAE 降低 31–63%。"
>
> **EN**: "The project title: Automated Evaluation Framework for Empathetic Dialogue — LLM-as-a-Judge plus statistical calibration. Zero human annotation cost, MAE reduced by 31–63%."

---

### 环节 1: 问题定义 + 数据展示（Slide 2 + Demo Section 1）— 2 分钟

**操作**：翻到 Slide 2，然后切终端运行：

```bash
python demo_live.py --section 1
```

**要说的话（中英对照）**：

Slide 2:

> **中**："核心问题：怎么衡量共情？BLEU 和 ROUGE 只测词汇重叠，完全不能衡量共情——'I understand how you feel' 和 'Your feelings are completely valid' 共情程度相同，但 BLEU 接近 0。人工标注又昂贵、慢、不可复现，而且共情本身主观性很高，标注者之间的一致性非常低。"
>
> **EN**: "The core problem: how to measure empathy? BLEU and ROUGE only measure lexical overlap — 'I understand how you feel' and 'Your feelings are completely valid' have similar empathy but BLEU ≈ 0. Human annotation is expensive, slow, and unreproducible — inter-annotator agreement on empathy is notoriously low."

切到终端展示数据：

> **中**："我们的训练数据是 PsychCentral 的 5,318 条心理咨询对话，每条包含用户陈述和治疗师回复。"
>
> **EN**: "Our training data is 5,318 mental health dialogues from PsychCentral, each containing a user statement and therapist response."

---

### 环节 2: 系统架构 + 工程亮点（Slide 3）— 2 分钟

**操作**：翻到 Slide 3

**要说的话（中英对照）**：

> **中**："这是我们的端到端管线：5,318 对话 → 3 个 GPT-2 模型（消融对照）→ 600 条生成（200/模型）→ DeepSeek LLM Judge 1,800 次调用 → Isotonic Regression 校准。"
>
> **EN**: "Here's our end-to-end pipeline: 5,318 dialogues → 3 GPT-2 model variants for contrastive ablation → 600 generated responses → DeepSeek LLM Judge with 1,800 API calls → isotonic regression calibration."

指着 4 维度量表卡片：

> **中**："评估基于 4 个维度：情感识别、验证与温暖、实用性、安全与边界。量表是 Judge 和人类标注共用的——single source of truth，定义在 `src/eval/rubric.py` 里。"
>
> **EN**: "Evaluation uses a 4-dimension rubric: emotion recognition, validation & warmth, helpfulness, safety & boundaries. The rubric is shared by LLM judge and human annotators — a single source of truth defined in `src/eval/rubric.py`."

指着工程亮点卡片：

> **中**："工程方面有 5 个亮点：分层模块架构、single source of truth 的量表、1,800 次调用零解析失败的 robust JSON parsing、Strategy Pattern 可换后端、断点续传。"
>
> **EN**: "Five engineering highlights: modular layered architecture, single source of truth rubric, robust JSON parsing with 0 failures across 1,800 calls, strategy pattern for swappable backends, and resume-from-checkpoint for API interruptions."

---

### 环节 3: BeFuller 全栈工程能力（Slide 4）— 1 分钟

**操作**：翻到 Slide 4

**要说的话（中英对照）**：

> **中**："除了 NLP 管线，我还有全栈工程经验。BeFuller.com 是一个生产级 IoT 监控平台，部署在香港大学校园里，实时监控温度、CO₂、湿度和空气质量。我负责前后端和运维——React/Vue 前端、Python API 后端、MongoDB 时序数据存储。"
>
> **EN**: "Beyond NLP pipelines, I also have full-stack engineering experience. BeFuller.com is a production IoT monitoring platform deployed for a Hong Kong university — real-time monitoring of temperature, CO₂, humidity, and air quality. I was responsible for front-end, back-end, and operations — React/Vue, Python API, MongoDB."

> **中**："这和 HAI 研究的关联是：同样的技术栈可以直接用来搭建实验 UI——聊天界面、问卷表单、知情同意流程、结构化的数据收集和日志记录。"
>
> **EN**: "The relevance to HAI research: these same skills — React/Vue for experiment UIs, Python API for ML backends, MongoDB for structured data collection — are directly applicable to building interactive research platforms."

---

### 环节 4: Chain-of-Empathy + 生成（Slide 5 + Demo Section 2/3）— 3 分钟

**操作**：翻到 Slide 5，然后切终端：

```bash
python demo_live.py --section 2    # 模型架构 + checkpoint
python demo_live.py --section 3    # 已有生成结果对比
```

**要说的话（中英对照）**：

Slide 5:

> **中**："Chain-of-Empathy 的灵感来自 CBT 认知行为疗法，模拟人类产生共情回复的过程，分成 5 个阶段：场景理解、情绪识别、原因推理、目标设定、回复规划。"
>
> **EN**: "Chain-of-Empathy is inspired by Cognitive Behavioral Therapy — it mirrors how humans develop empathetic responses through 5 stages: scenario understanding, emotion recognition, cause inference, goal setting, and response planning."

> **中**："融合机制是 additive injection：把共情推理的向量加到 GPT-2 的隐状态上。选择加法而不是 cross-attention，因为在 124M 参数规模下更高效。Stage 5 没有激活函数，保留正负信号；初始化用 Xavier Uniform 保证梯度稳定。"
>
> **EN**: "The fusion mechanism uses additive injection — empathy reasoning vectors are added to GPT-2 hidden states. We chose additive over cross-attention because it's more parameter-efficient at 124M scale. No activation in Stage 5 to preserve positive and negative signals; Xavier Uniform initialization for stable gradients."

切到终端展示生成结果：

> **中**："每个模型生成了 200 条回复，共 600 条。所有生成参数——seed、temperature、top_p——都记录在 JSONL 里，完全可复现。这里可以看到 vanilla、fine-tuned、empathy chain 三个模型的输出对比。"
>
> **EN**: "Each model generated 200 responses, 600 total. All generation parameters — seed, temperature, top_p — are logged in JSONL for full reproducibility. Here you can see the comparison across vanilla, fine-tuned, and empathy chain outputs."

> **中**："GPT-2 只有 124M 参数，分数低是完全预期的。模型是消融对象，不是主要贡献——重点是评估框架。"
>
> **EN**: "GPT-2 at 124M parameters scores low — that's expected. The models serve as ablation subjects to validate the evaluation framework — they are not the primary contribution."

---

### 环节 5: LLM Judge + NLG 对比（Slide 2 回顾 + Demo Section 4/5）— 2 分钟

**操作**：

```bash
python demo_live.py --section 4    # LLM Judge 评分结果
python demo_live.py --section 5    # NLG 指标对比
```

**要说的话（中英对照）**：

Judge 评分：

> **中**："我们用 DeepSeek 作为 LLM Judge，对 600 条回复做 3 次重复评分，总共 1,800 次 API 调用，零解析失败。这得益于我们设计的三层 robust JSON parsing——先去掉 markdown 围栏，再直接解析，最后正则兜底。"
>
> **EN**: "We used DeepSeek as LLM Judge — 3 repeated evaluations on 600 responses = 1,800 API calls, zero parse failures. This is thanks to our 3-layer robust JSON parsing: strip markdown fences → direct parse → regex fallback."

指着对比表：

> **中**："可以清楚看到 Judge 能区分三个模型：vanilla 分最低，fine-tuned 分最高。"
>
> **EN**: "You can clearly see the judge discriminates between models — vanilla scores lowest, fine-tuned scores highest."

NLG 对比：

> **中**："再看传统 NLG 指标——微调模型的 BLEU 只有 0.016，但 LLM Judge 给的共情评分明显更高。这恰恰证明 BLEU/ROUGE 不适合评估共情，验证了我们选择 LLM-as-Judge 方案的正确性。"
>
> **EN**: "Look at the traditional NLG metrics — fine-tuned model's BLEU is only 0.016, yet the LLM judge rates empathy significantly higher. This confirms that BLEU/ROUGE are inadequate for empathy evaluation, validating our LLM-as-Judge approach."

---

### 环节 6: 校准结果（Slide 6 + Demo Section 6）— 3 分钟 ⭐ 核心

**操作**：翻到 Slide 6，然后切终端：

```bash
python demo_live.py --section 6
```

**要说的话（中英对照）**：

> **中**："这是我们最核心的贡献：External Human-Anchored Calibration。我们用公开的人类评分数据集（EPITOME 和 EmpatheticDialogues-EVAL）来训练 isotonic regression 校准器，不需要自己做任何人工标注——避免了循环偏差，成本为零。"
>
> **EN**: "This is our core contribution: External Human-Anchored Calibration. We train an isotonic regression calibrator on public human-rated datasets — EPITOME and EmpatheticDialogues-EVAL — eliminating the need for any manual annotation. Zero cost, zero circularity bias."

指着 4 个 MAE 数字：

> **中**："MAE 降幅：Emotion 62.8%（0.547→0.205），Validation 54.1%，Helpfulness 56.8%，Safety 31.3%。四个维度全部显著下降。"
>
> **EN**: "MAE reduction: Emotion 62.8% (0.547→0.205), Validation 54.1%, Helpfulness 56.8%, Safety 31.3%. Significant improvement across all four dimensions."

> **中**："关键 insight：Spearman rank correlation 保持不变（0.32–0.79）。这说明校准修正的是绝对分数偏移，而不会损害 Judge 的判别能力。Judge 的排序是可靠的——只是有个系统性偏移，校准正好修正它。"
>
> **EN**: "Key insight: Spearman rank correlation is preserved (0.32–0.79). This means calibration corrects scale bias without hurting discriminative ability. The judge's ranking is reliable — it just has an absolute offset that calibration fixes."

> **中**："Judge 的自我一致性：精确一致 88–100%，近似一致（±1）96–100%。校准后 MAE < 0.3（5 分制），已经达到实用水平。"
>
> **EN**: "Judge self-consistency: exact agreement 88–100%, near agreement (±1) 96–100%. Post-calibration MAE < 0.3 on a 5-point scale — practically usable."

**补充**：在 VS Code 中打开 `outputs/analysis/calibration_report_paper.md` 展示完整报告。

---

### 环节 7: 消融实验（Slide 7 + Demo Section 7）— 2 分钟

**操作**：翻到 Slide 7，然后切终端：

```bash
python demo_live.py --section 7
```

**要说的话（中英对照）**：

> **中**："消融实验验证了框架的鲁棒性。首先是重复次数敏感性：k=1、2、3 的 Spearman ρ 几乎没有变化（差异 < 0.01）。k=1 就够了——这意味着可以节省 66% 的 API 成本，对实际部署意义重大。"
>
> **EN**: "Ablation studies validate framework robustness. First, repeat sensitivity: Spearman ρ at k=1, 2, 3 shows marginal difference (< 0.01). k=1 is sufficient — that's a 66% API cost reduction, significant for practical deployment."

> **中**："跨模型判别力：Judge 给 vanilla 平均 1.0 分，fine-tuned 1.33 分——清楚区分了不同质量的模型，证明 rubric 是有效的。我们也做了 prompt 变体测试（default / strict / minimal），Score 分布保持稳定，验证了 rubric 对 prompt 措辞的鲁棒性。"
>
> **EN**: "Cross-model discrimination: vanilla averages 1.0, fine-tuned 1.33 — the judge clearly distinguishes model quality, confirming rubric validity. We also tested 3 prompt variants (default/strict/minimal) — score distributions remain stable, validating robustness across prompt formulations."

> **中**："诚实评估：GPT-2（124M）只得 1–2 / 5——这在这个规模是预期的。模型是消融对象，不是贡献本身。"
>
> **EN**: "Honest assessment: GPT-2 (124M) scores only 1–2 out of 5 — expected at this scale. The models are ablation subjects, not the contribution itself."

---

### 环节 8: 可复现性（Slide 3 回顾 + Demo Section 8）— 2 分钟

**操作**：

```bash
python demo_live.py --section 8
```

然后展示脚本内容：

```bash
cat experiments/run_train.sh
cat experiments/run_generate.sh
cat experiments/run_paper_pipeline.sh | head -30
```

**要说的话（中英对照）**：

> **中**："整个管线是端到端可复现的。每一步——训练、生成、评分、校准——都是 bash/python 一键执行。所有中间结果保存为 JSONL/JSON/CSV，方便下游分析。"
>
> **EN**: "The entire pipeline is end-to-end reproducible. Every step — training, generation, scoring, calibration — is executable with a single bash/python command. All intermediate results are saved as JSONL/JSON/CSV for downstream analysis."

> **中**："代码采用分层架构：data → models → inference → eval。校准器采用 Strategy Pattern 设计，可以无代码改动换算法（isotonic、ordinal）。还有 resume-from-checkpoint 机制——如果 API 调用中途中断，可以从断点继续。"
>
> **EN**: "Code follows a layered architecture: data → models → inference → eval. The calibrator uses the Strategy Pattern — swap algorithms (isotonic, ordinal) with no code changes. Plus resume-from-checkpoint — if API calls are interrupted, pick up from where you left off."

---

### 环节 9: 总结 + HAI 愿景（Slides 8–9）— 2 分钟

**操作**：翻到 Slide 8，然后 Slide 9

**要说的话（中英对照）**：

Slide 8 — 三个核心贡献：

> **中**：
> 1. "**方法创新**：外部人类锚定校准——用公开数据集作为无偏锚点，零标注成本，消除循环偏差。"
> 2. "**工程完整性**：端到端可复现管线，shell 脚本一键执行。"
> 3. "**实用价值**：k=1 省 66% 成本，校准 MAE < 0.3（5 分制），达到实用水平。"
>
> **EN**:
> 1. "**Methodological innovation**: External human-anchored calibration — public datasets as unbiased anchors, zero annotation cost, eliminates circularity bias."
> 2. "**Engineering completeness**: End-to-end reproducible pipeline, one-command execution via shell scripts."
> 3. "**Practical value**: k=1 saves 66% API cost; calibrated MAE < 0.3 on a 5-point scale — practically usable."

翻到 Slide 9 — HAI 平台愿景：

> **中**："未来方向：把这个评估框架变成一个交互式 HAI 实验平台。用户实验流程包括知情同意、聊天任务（A/B condition）、实时质量评分、问卷调查和数据导出。可以研究解释风格（有无评分展示）对信任的影响、人工审核 vs 全自动对接受度的影响、隐私控制对披露意愿的影响。"
>
> **EN**: "Future direction: transform this evaluation framework into an interactive HAI experiment platform. User study flow includes consent, chat tasks with randomized conditions, real-time quality scoring, post-task surveys, and data export. We can study how explanation style affects trust, how human-in-the-loop affects acceptance, and how privacy controls affect disclosure willingness."

> **中**："我的全栈经验——React/Vue 做实验 UI、Python API 连接 ML 后端、MongoDB 做结构化数据收集——可以直接用来搭建这个平台。"
>
> **EN**: "My full-stack experience — React/Vue for experiment UIs, Python API for ML backends, MongoDB for structured data collection — can directly support building this platform."

---

## 三、备选演示命令（如果某环节出问题）

### 直接看数据

```bash
# 数据样本（JSONL 需逐行解析）
head -3 data/formatted_Psych_data.jsonl | while read line; do echo "$line" | python -m json.tool; done

# 生成结果
head -1 outputs/generations/empathy_chain.jsonl | python -m json.tool

# Judge 评分
head -1 outputs/judge/empathy_chain_judge.jsonl | python -m json.tool

# 校准结果
head -1 outputs/calibrated/isotonic_calibrated.jsonl | python -m json.tool

# NLG 指标
python -m json.tool outputs/nlg_metrics.json

# 分析报告
cat outputs/analysis/calibration_report_paper.md
cat outputs/analysis/ablation_repeats.md
```

### 展示代码结构

```bash
# 目录树
find src -name '*.py' | head -20
find experiments -name '*.py' -o -name '*.sh' | head -15

# 核心模块行数
wc -l src/eval/*.py Chain_of_Empathy.py Model_*.py
```

### 不依赖 demo 脚本的模型生成

```bash
python -c "
from Model_Baseline import GPT2BaselineModel
import torch
model = GPT2BaselineModel('gpt2')
model.load_state_dict(torch.load('checkpoints/baseline_best.pt', map_location='cpu'))
model.eval()
print(model.generate_response('I have been feeling really stressed lately and cannot sleep.', max_length=60))
"
```

---

## 四、Q&A 准备（中英对照）

| 可能的提问 | 回答要点 Answer |
|-----------|---------|
| 为什么用 GPT-2 而不是更大的模型？ | GPT-2 is an ablation subject to validate the framework; the framework is the contribution, not the model. / GPT-2 作为消融对象验证框架；框架是主要贡献，不是模型 |
| 为什么不自己做人工标注？ | External anchoring avoids circularity bias, costs zero, and is reproducible. / 外部锚定避免循环偏差，成本为零，且可复现 |
| DeepSeek 作 Judge 可靠吗？ | Self-consistency 88–100%; Spearman ρ 0.3–0.8; future work: compare GPT-4/Claude. / 自我一致性 88-100%；未来可对比 GPT-4/Claude |
| Ordinal 校准为什么没效果？ | Only 60 training samples — ordinal regression needs more data; isotonic is non-parametric, works better with small data. / 训练数据仅 60 条，ordinal 需要更多数据 |
| k=1 真的够吗？ | ρ difference < 0.01 across k=1/2/3; calibration compensates single-measurement noise. / 三次重复的 ρ 差异 < 0.01；校准可以补偿单次测量的噪声 |
| 评分量表有效性怎么保证？ | Rubric is grounded in clinical psychology (CBT); validated via pilot annotation with GO/NO-GO gating. / 量表基于临床心理学文献；经过 pilot 标注验证（含 GO/NO-GO 门控） |
| 这跟 HAI 有什么关系？ | Calibrated measurement enables scalable user studies; platform vision on Slide 9. / 校准测量支撑可扩展的用户研究；平台愿景见 Slide 9 |

---

## 五、时间分配建议

| 时间 | 环节 | Slide | 优先级 |
|------|------|-------|--------|
| 0:00–2:00 | 环节 0: Research Fit + 开场 | Slides 0–1 | 必须 |
| 2:00–4:00 | 环节 1: 问题定义 + 数据 | Slide 2 + Demo 1 | 必须 |
| 4:00–6:00 | 环节 2: 系统架构 + 工程 | Slide 3 | 必须 |
| 6:00–7:00 | 环节 3: BeFuller 全栈 | Slide 4 | 可选 |
| 7:00–10:00 | 环节 4: CoE + 生成 | Slide 5 + Demo 2/3 | 必须 |
| 10:00–12:00 | 环节 5: Judge + NLG | Demo 4/5 | 必须 |
| 12:00–15:00 | **环节 6: 校准结果** ⭐ | Slide 6 + Demo 6 | **核心** |
| 15:00–17:00 | 环节 7: 消融实验 | Slide 7 + Demo 7 | 必须 |
| 17:00–19:00 | 环节 8: 可复现性 | Demo 8 | 可选 |
| 19:00–21:00 | 环节 9: 总结 + HAI 愿景 | Slides 8–9 | 必须 |
| 21:00–25:00 | Q&A | — | 灵活 |

> **Slide 10（Appendix: External Anchor Datasets）** 不主动展示，仅在 Q&A 中被问到校准数据来源时翻到。





# React / Vue 面试高频问题 & 回答话术（中英对照）

---

## Q1: "你在项目中用了 React 哪些核心概念？"

**EN:** "I heavily used React Hooks throughout the project — `useState` for managing sensor data and UI states, `useEffect` for data fetching with polling intervals and cleanup, and `useRef` for DOM references. I also used React Router for client-side routing with conditional route guards based on user roles — admin, active, or inactive."

**中文：** "我在项目中大量使用了 React Hooks — `useState` 管理传感器数据和 UI 状态，`useEffect` 实现数据轮询与清理，`useRef` 用于 DOM 引用。同时使用 React Router 实现客户端路由，并根据用户角色（管理员/已激活/未激活）做条件路由守卫。"

**可以直接指向的代码：**

```jsx
// App.js — 条件路由守卫（基于角色的权限控制）
<Routes>
  {(userData?.is_active) && (<Route path="/sensors" element={<Sensors />} />)}
  {(userData?.is_admin) && (<Route path="/edit" element={<EditSensor />} />)}
  {(!userData?.email) && (<Route path="/login" element={<Login />} />)}
  {(!userData?.is_active && userData?.email) && (
    <Route path="/inactivate" element={<InActive />} />
  )}
</Routes>
```

---

## Q2: "你怎么处理实时数据更新的？"

**EN:** "I implemented a polling mechanism using `setInterval` inside `useEffect`. The dashboard fetches sensor data every 5 seconds from the backend API. I also track each sensor's last updated timestamp so the UI can visually indicate stale data — if a sensor hasn't reported in over 10 minutes, the name turns grey and the status shows red. I made sure to return a cleanup function to clear the interval when the component unmounts."

**中文：** "我在 `useEffect` 里用 `setInterval` 实现轮询机制，仪表盘每 5 秒从后端 API 拉取传感器数据。同时追踪每个传感器的最后更新时间戳——如果超过 10 分钟没有上报，名称变灰、状态变红。组件卸载时通过返回的清理函数清除定时器，防止内存泄漏。"

**典型代码：**

```jsx
// Sensors.js — 实时轮询 + 清理
useEffect(() => {
  fetchLastRecord();
  const interval = setInterval(fetchLastRecord, 5000);
  return () => clearInterval(interval);  // 防止内存泄漏
}, []);

// SensorCard — 实时计时器显示"X秒前更新"
useEffect(() => {
  const interval = setInterval(() => {
    setTimeSinceUpdate(Math.floor((Date.now() - lastUpdated) / 1000));
  }, 1000);
  return () => clearInterval(interval);
}, [lastUpdated]);
```

---

## Q3: "你用了哪些性能优化手段？"

**EN:** "The sensor list can have 50+ devices, so I used `react-window`'s `FixedSizeList` for virtualized rendering — only visible rows are rendered in the DOM, which dramatically improves scroll performance on mobile. For the history chart, I implemented data downsampling — dividing timestamps into intervals based on the time period (1H/3H/24H) and averaging the values within each interval, rather than plotting every single data point."

**中文：** "传感器列表可能有 50 多个设备，所以我用了 `react-window` 的 `FixedSizeList` 做虚拟化渲染——只渲染可视区域内的行，大幅提升移动端滚动性能。对于历史图表，我实现了数据降采样——根据时间段（1H/3H/24H）将时间戳分组并取平均值，而不是绘制每一个数据点。"

**典型代码：**

```jsx
// Sensors.js — 虚拟化列表
<FixedSizeList
  height={375}
  itemSize={75}
  itemCount={filteredRecords.length}
  width="100%"
>
  {({ index, style }) => (
    <Row style={style} data={filteredRecords[index]}
         onClick={() => setSelectedSensorId(filteredRecords[index][0])} />
  )}
</FixedSizeList>
```

```jsx
// HistoryChart.js — 数据降采样（时间窗口内取平均值）
const interval = (dataPeriod === 1) ? hour1_interval 
               : (dataPeriod === 3) ? hour3_interval 
               : hour24_interval;
const filteredTimestamps = timestamps.filter((_, i) => i % interval === 0);
const values = filteredTimestamps.map((timestamp, _index) => {
  const _templist = timestamps.slice(previousIndex, originalIndex);
  const _values = _templist.map(ts => data[...][ts][...] / divisor);
  return (_values.reduce((a, b) => a + b, 0) / _templist.length).toFixed(2);
});
```

---

## Q4: "你怎么实现告警系统的？"

**EN:** "I defined threshold values for each sensor type directly in the component — for example, CO₂ above 1000 ppm, temperature outside 15-30°C, humidity outside 30-70%. I also calculated the humidity ratio using the IAPWS IF-97 thermodynamic formula to detect condensation risk. When any threshold is exceeded, the card border turns red, and individual values are highlighted. This gives operators an at-a-glance view of which sensors need attention."

**中文：** "我在组件中直接定义每种传感器的阈值——比如 CO₂ 超过 1000 ppm、温度在 15-30°C 范围外、湿度在 30-70% 范围外。还用 IAPWS IF-97 热力学公式计算了湿度比来检测凝结风险。当任何阈值被超过，卡片边框变红，异常值高亮显示，让操作人员一眼就能看到哪些传感器需要关注。"

**典型代码：**

```jsx
// Sensors.js — 多级告警判断
const scd_co2_alarm = (data.scd?.["3"] > 1000);
const scd_temp_alarm = (data.scd?.["4"]/100 > 30) || (data.scd?.["4"]/100 < 15);
const scd_humdity_alarm = (data.scd?.["5"]/100 > 70) || (data.scd?.["5"]/100 < 30);
const scd_alarm = (scd_co2_alarm || scd_temp_alarm || scd_humdity_alarm);

// 视觉反馈 — 动态边框颜色
<Card style={{ borderColor: scd_alarm ? '#990000' : '#009999' }}>
  <Typography style={{ color: scd_co2_alarm ? '#BB0000' : 'inherit' }}>
    {data.scd["3"]} ppm
  </Typography>
</Card>
```

---

## Q5: "为什么用 React 而不是 Vue？或者说你怎么看这两个框架的区别？"

**EN:** "I actually used both in this project ecosystem. React was our primary choice because of JSX flexibility — when building complex sensor dashboards with conditional rendering and dynamic data transformations, JSX gives you the full power of JavaScript inline. Vue was used in some earlier components for its simpler template syntax, which is faster for building standard CRUD pages. In practice, the two are very similar in capability — React gives more flexibility with its 'everything is JavaScript' philosophy, while Vue provides more structure out of the box with its Options API and built-in directives like `v-for` and `v-if`."

**中文：** "我在这个项目生态中实际上两个都用了。React 是主要选择，因为 JSX 的灵活性——在构建复杂的传感器仪表盘时需要大量条件渲染和动态数据转换，JSX 让你可以直接内联使用 JavaScript 的全部能力。Vue 用在了一些早期组件中，因为它的模板语法更简洁，构建标准 CRUD 页面更快。实际上两者能力非常相似——React 的 'everything is JavaScript' 哲学更灵活，Vue 则通过 Options API 和内置指令（如 `v-for`、`v-if`）提供更多开箱即用的结构。"

---

## Q6: "你状态管理用的什么方案？为什么没用 Redux？"

**EN:** "I used React's built-in `useState` and `useEffect` hooks for state management, without Redux or Context API. The reason is that our data flow is relatively straightforward — sensor data flows from one API endpoint down to child components via props. There's no complex cross-component state sharing that would justify the Redux boilerplate. The user authentication state is fetched once in `App.js` and passed down as props. For a larger team or more complex state interactions, I would consider Redux Toolkit or Zustand."

**中文：** "我使用 React 内置的 `useState` 和 `useEffect` 来管理状态，没有用 Redux 或 Context API。原因是我们的数据流相对直接——传感器数据从一个 API 端点流向子组件，通过 props 传递。没有复杂的跨组件状态共享来证明引入 Redux 样板代码的必要性。用户认证状态在 `App.js` 中获取一次，通过 props 向下传递。如果团队更大或状态交互更复杂，我会考虑 Redux Toolkit 或 Zustand。"

---

## Q7: "你项目的数据可视化是怎么做的？"

**EN:** "I used Chart.js via `react-chartjs-2` for time-series visualization. The history chart component supports switching between 1-hour, 3-hour, and 24-hour windows with a date picker. A key technical detail is the data downsampling — for a 24-hour view, I average values within time intervals to keep the chart responsive, rather than plotting thousands of raw data points. For the PM sensor and weather station pages, I used bar charts with similar aggregation logic."

**中文：** "我使用 `react-chartjs-2`（Chart.js 的 React 封装）做时序数据可视化。历史图表组件支持 1 小时、3 小时、24 小时时间窗口切换和日期选择。一个关键技术细节是数据降采样——对于 24 小时视图，我在时间段内取平均值来保持图表流畅，而不是绘制数千个原始数据点。"

---

## 面试加分建议

| 策略 | 说明 |
|------|------|
| **讲决策而非语法** | 老师不会问你 `useState` 怎么写，会问你**为什么这么设计** |
| **准备一个"最难的 bug"故事** | 比如：轮询导致的内存泄漏、传感器离线检测的时间阈值调优 |
| **强调工程公式嵌入** | IAPWS IF-97 湿度比计算嵌入 React 组件非常独特，说明你不只是前端开发，还理解领域知识 |
| **如果被问到 Vue** | 强调你理解两者的核心差异（JSX vs Template、Composition API vs Hooks），并能根据场景选择合适的工具 |
| **连接到 HAI 研究** | "These same patterns — real-time data polling, conditional UI rendering, threshold-based alerts — are directly applicable to building experiment UIs that react to user behavior in real time." |
