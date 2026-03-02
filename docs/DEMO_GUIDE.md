# 演示指南 — Empathetic Dialogue Evaluation Framework

> **演示时长**：15-20 分钟（可根据需要调整）  
> **演示目标**：展示项目的可复现性、核心创新和实验结果

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

## 二、演示流程（8 个环节）

### 环节 0: 开场（Slides 1-2）— 2分钟

**操作**：浏览器中展示 `project_slides.html`

**要点**：
- "我们的项目是：用 LLM-as-a-Judge + 统计校准来自动评估心理咨询对话的共情质量"
- "核心问题：BLEU/ROUGE 无法衡量共情，人工标注又太贵"
- "我们的方案：4 维度量表 + DeepSeek 自动评分 + 外部人类数据校准"

---

### 环节 1: 展示数据（终端）— 1分钟

**操作**：

```bash
python demo_live.py --section 1
```

**要说的话**：
- "我们的训练数据是 PsychCentral 的 5,318 条心理咨询对话"
- "每条包含用户陈述、治疗师回复、和认知扭曲标签"
- 指着屏幕上的 3 条随机样本，展示数据的多样性

**补充（如有时间）**：在 VS Code 中打开 `data/formatted_Psych_data.jsonl` 快速浏览数据格式。

---

### 环节 2: 模型架构 & 生成（终端 + Slides 3-4）— 3分钟

**操作**：

```bash
# 展示模型架构图和 checkpoint
python demo_live.py --section 2
```

**要说的话**：
- "我们训练了 3 个模型作为消融对照：原始 GPT-2、微调 GPT-2、GPT-2 + 共情链"
- "共情链灵感来自 CBT 认知行为疗法的五步推理"  
- "关键设计选择：additive fusion（非 cross-attention），在 124M 参数规模更高效"
- "两个模型的 checkpoint 都保存在 checkpoints/ 下"

**现场生成**（如果来得及加载模型）：
- 选择一个预设输入，或者让观众提供一句话
- 展示基线 vs. 共情链的回复对比
- 强调："GPT-2 只有 124M 参数，分数低是预期的——模型是消融对象，不是主要贡献"

**如果模型加载太慢**：直接展示已有的生成结果：

```bash
python demo_live.py --section 3
```

---

### 环节 3: 展示已有生成结果（终端）— 1分钟

**操作**：

```bash
python demo_live.py --section 3
```

或者直接在终端展示 JSONL 文件：

```bash
# 展示 3 个模型的生成样本
head -3 outputs/generations/gpt2_vanilla.jsonl | python -m json.tool
head -3 outputs/generations/empathy_chain.jsonl | python -m json.tool
```

**要说的话**：
- "每个模型生成了 200 条回复，共 600 条"
- "所有生成参数（seed, temperature, top_p）都记录在 JSONL 中，完全可复现"

---

### 环节 4: LLM Judge 评分（终端 + Slide 3）— 2分钟

**操作**：

```bash
python demo_live.py --section 4
```

**要说的话**：
- "我们用 DeepSeek 作为 LLM Judge，对 600 条回复进行 3 次重复评分 = 1,800 次 API 调用"
- "零解析失败——因为我们设计了 robust JSON parsing"
- 指着对比表："可以清楚看到 Judge 能区分 vanilla（最低）和 fine-tuned（最高）"
- "4 个评估维度就是我们的 rubric（量表），rubric 是 Judge 和人类标注共用的——single source of truth"

**补充**：在 VS Code 中打开 `src/eval/rubric.py` 展示量表定义。

---

### 环节 5: NLG 指标 vs. LLM Judge（终端 + Slide 2）— 1分钟

**操作**：

```bash
python demo_live.py --section 5
```

**要说的话**：
- "传统 BLEU/ROUGE 分数很低——但这恰恰说明它们不适合评估共情"
- "微调模型的 BLEU = 0.016，但 LLM Judge 给的共情评分明显更高"
- "这验证了我们选择 LLM-as-Judge 方案的正确性"

---

### 环节 6: 校准结果（终端 + Slide 5）— 3分钟 ⭐ 核心

**操作**：

```bash
python demo_live.py --section 6
```

**切换到 Slide 5** 展示校准结果图表。

**要说的话**：
- "这是我们最核心的贡献：External Human-Anchored Calibration"
- "我们用公开的人类评分数据集训练 isotonic regression 校准器"
- "不需要自己做任何人工标注——避免了循环偏差"
- 指着表格："MAE 从 ~0.5 降到 ~0.2，降幅 31-63%"
- "关键是：Spearman ρ 保持不变——说明校准修正的是分数偏移，而不是排序"
- "校准后 MAE < 0.3（5 分制），已经达到实用水平"

**补充**：在 VS Code 中打开 `outputs/analysis/calibration_report_paper.md` 展示完整报告。

---

### 环节 7: 消融实验（终端 + Slide 6）— 2分钟

**操作**：

```bash
python demo_live.py --section 7
```

**切换到 Slide 6** 展示消融实验。

**要说的话**：
- "k=1 已经足够 —— 三次重复的 Spearman ρ 几乎没有变化"
- "这意味着可以节省 66% 的 API 成本"
- "Judge 自我一致性 88-100%，非常稳定"
- "我们还做了 prompt 变体测试，验证 rubric 对 prompt 措辞的鲁棒性"

---

### 环节 8: 可复现性演示（终端 + Slide 3）— 2分钟

**操作**：

```bash
python demo_live.py --section 8
```

然后实际展示 shell 脚本：

```bash
# 展示训练脚本的内容
cat experiments/run_train.sh

# 展示生成脚本
cat experiments/run_generate.sh

# 展示完整管线脚本
cat experiments/run_paper_pipeline.sh | head -30
```

**要说的话**：
- "整个管线是端到端可复现的，每一步都是 bash/python 一键执行"
- "所有中间结果都保存为 JSONL/JSON/CSV，方便下游分析"
- "代码采用分层架构：data → models → inference → eval"
- "校准器采用 Strategy Pattern 设计，可以无代码改动换算法"

---

### 环节 9: 总结（Slide 7）— 1分钟

**切换到 Slide 7**（Contributions & Future Directions）

**要说的话**：
- "三个核心贡献：" 
  1. "方法创新：外部人类锚定校准，零标注成本"
  2. "工程完整性：端到端可复现管线"
  3. "实用价值：k=1 省 66% 成本，校准 MAE < 0.3"
- "未来方向：跨 Judge 对比（GPT-4 / Claude）、更大模型验证（LLaMA）"

---

## 三、备选演示命令（如果某环节出问题）

### 直接看数据

```bash
# 数据样本
head -3 data/formatted_Psych_data.jsonl | python -m json.tool

# 生成结果
head -1 outputs/generations/empathy_chain.jsonl | python -m json.tool

# Judge 评分
head -1 outputs/judge/empathy_chain_judge.jsonl | python -m json.tool

# 校准结果
head -1 outputs/calibrated/isotonic_calibrated.jsonl | python -m json.tool

# NLG 指标
cat outputs/nlg_metrics.json | python -m json.tool

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
model = GPT2BaselineModel('uer/gpt2-chinese-cluecorpussmall')
model.load_state_dict(torch.load('checkpoints/baseline_best.pt', map_location='cpu'))
model.eval()
print(model.generate_response('我最近压力很大，总是睡不着觉。', max_length=60))
"
```

---

## 四、Q&A 准备

| 可能的提问 | 回答要点 |
|-----------|---------|
| 为什么用 GPT-2 而不是更大的模型？ | GPT-2 作为消融对象验证框架；框架是主要贡献，不是模型本身 |
| 为什么不自己做人工标注？ | 外部锚定避免循环偏差，成本为零，且可复现 |
| DeepSeek 作 Judge 可靠吗？ | 自我一致性 88-100%；Spearman ρ 0.3-0.8；未来可对比 GPT-4/Claude |
| Ordinal 校准为什么没效果？ | 训练数据仅 60 条，ordinal regression 需要更多数据 |
| k=1 真的够吗？ | 三次重复的 ρ 差异 < 0.01；校准可以补偿单次测量的噪声 |
| 如何保证评分量表的有效性？ | 量表基于临床心理学文献；经过 pilot 标注验证（含 GO/NO-GO 门控） |

---

## 五、时间分配建议

| 时间 | 环节 | 优先级 |
|------|------|--------|
| 0:00-2:00 | 开场 + 问题定义 (Slides 1-2) | 必须 |
| 2:00-3:00 | 数据展示 (Demo 1) | 必须 |
| 3:00-6:00 | 模型 + 生成 (Demo 2-3) | 必须 |
| 6:00-8:00 | LLM Judge (Demo 4) | 必须 |
| 8:00-9:00 | NLG vs Judge (Demo 5) | 可选 |
| 9:00-12:00 | **校准结果 (Demo 6)** ⭐ | 核心 |
| 12:00-14:00 | 消融 + 可复现 (Demo 7-8) | 必须 |
| 14:00-15:00 | 总结 (Slide 7) | 必须 |
| 15:00-20:00 | Q&A | 灵活 |
