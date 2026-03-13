# 最简发起清单 — Maker-Checker User Study
## 照着一项项勾，完成即发起正式收数

> **原则**: anonymous · minimal-risk · vignette-based evaluation  
> **设计选择**: 见下方 ⚠️ 说明，选定后不可更改  
> **目标**: 2–3 周内完成门槛步骤，进入正式收数

---

## ⚠️ 先做这个设计决策（只需一次，决定后不要改）

### Between-subjects（最省事版）vs Within-subjects（原版）

| | **Between-subjects** | **Within-subjects（原设计）** |
|---|---|---|
| Qualtrics | 3 个固定链接，各发一个条件 | 12-cell 反平衡，URL 参数路由 |
| 每人看几条 | 12 vignette，全是同一条件 | 12 vignette，4A+4B+4C 混合 |
| 每组 N | 16（总 N=48） | 36（每个 cell 3 人） |
| 统计功效 | 低（d=0.5 时约 35%） | 高（d=0.5 时约 80%） |
| 可否写进论文 | 可以，但需改分析计划 | 是当前预注册的方案 |
| 复杂度 | ★☆☆ 极简 | ★★★ 需 Qualtrics 配置 |

**推荐**：如果你的时间/预算极度紧张 → 选 Between-subjects，但要同步更新预注册分析计划。  
**当前预注册方案** = Within-subjects，不改则保持现状，走 12-cell 路线。

**你的选择**（填写后此文档固定）：`[ ] Between-subjects` &nbsp;&nbsp;&nbsp; `[ ] Within-subjects`

---

## 阶段 0：研究材料准备（在提交 IRB 前）

### 0-A 问卷定稿

- [ ] 自己从头到尾完整跑一次 Qualtrics 问卷
- [ ] 确认总时长 15–25 分钟
- [ ] 确认 Condition C 显示 "✓ This response has been reviewed for safety" 字样
- [ ] 确认 Condition A / B 不显示任何 safety indicator
- [ ] 确认 attention check 题目位于第 6 条 vignette 后，要求选 5
- [ ] 确认开放题全部设为 **optional**（不强制作答）
- [ ] 确认问卷页首显示："**This study is not clinical advice. If you are in crisis, please call 988.**"

### 0-B 量表精简（可选但推荐）

当前 8 个 per-vignette items；如要进一步简化，可删减至：

| Primary（必留） | Secondary（可留） | Exploratory（可砍） |
|---|---|---|
| Q2 Warmth | Q1 Empathy | — |
| Q3 Safety | Q4 Boundary Clarity | — |
| Q5 Transparency | Q6 Trust | Q8 Seek Real Help |
| Q7 Willingness to Rely | — | — |

> 砍掉 Q8 会让 RCI 分析受影响；建议最少保留以上 Primary 4 项。

- [ ] 决定最终保留哪些量表项，并在 Qualtrics 中删除不要的（删前备份）

---

## 阶段 1：IRB / 伦理审批

**目标：把研究框成 "anonymous minimal-risk vignette evaluation of AI UX"，不要写成 clinical study。**

### 1-A 材料准备（1–2 天）

- [ ] 向导师 / 系里 admin 要一份**近期过审的 minimal-risk survey 研究模板**（比从零写快很多）
- [ ] 在模板中替换以下内容（四件事，不需要写更多）：
  - [ ] 研究标题 → "Evaluating AI-Generated Supportive Responses: A Vignette Study"
  - [ ] 研究目的 → 评估不同 AI 回复呈现方式对 perceived warmth/safety/transparency 的影响
  - [ ] 样本来源 → 18+ 成人，Prolific 或学校被试池
  - [ ] 风险说明 → minimal，vignette-only，无临床干预，不收 PHI
  - [ ] compensation → ≈ £3 / $3.75 for ~20 min（≥ 国际标准时薪）
  - [ ] 数据存储 → Qualtrics encrypted + PI 机构存储，5 年后删除

### 1-B 附件清单（最小集）

只需附以下四件，不要多放：

| # | 文件 | 来源 |
|---|------|------|
| 1 | Protocol / submission form | 从模板改 |
| 2 | Consent / information sheet | `docs/irb_consent.md` |
| 3 | Recruitment text (~50 words) | 见下方 §附录 A |
| 4 | Survey question list | `docs/survey_instrument.md` 节选 |

`docs/irb_submission_package.md` 是完整版参考，用于对照，不需要全文提交。

### 1-C 提交

- [ ] 使用机构 IRB 在线门户提交（not email）
- [ ] 记录提交日期：`_________`
- [ ] 记录协议号（收到后填写）：`_________`

---

## 阶段 2：OSF 预注册（与 IRB 并行，IRB 批准前完成）

**目标：只注册核心内容，5–10 分钟填完，OSF 时间戳锁定即可。**

### 2-A 注册哪些内容（最简版，照填即可）

在 OSF → "Pre-register a study" → AsPredicted 模板，填：

1. **Research questions**:  
   Does a double-AI maker-checker architecture improve perceived safety and calibrated reliance, and what trade-off does it create with perceived warmth?

2. **Hypotheses**:  
   - H1: Conditions B and C score higher than A on perceived safety (Q3) and boundary clarity (Q4)  
   - H2: Condition C scores higher than A and B on perceived transparency (Q5)  
   - H3: Condition A scores higher than B and C on perceived warmth (Q2) — expected small-to-medium effect  
   - H4: Condition C shows the highest appropriate reliance (Q7) especially in high-risk scenarios

3. **Primary outcomes** (only 4):  
   Q2 Warmth, Q3 Safety, Q5 Transparency, Q7 Willingness to Rely

4. **Analysis plan**:  
   Linear mixed-effects models (within-subjects) or one-way ANOVA (between-subjects), pairwise contrasts A–B, A–C, B–C with Holm correction; α = .05

5. **Sample**:  
   Target N = 48 enrolled; minimum 36 analyzable after exclusions

6. **Exclusions**:  
   (a) attention check fail, (b) < 5 min completion, (c) zero-variance responses

7. **Existing data**:  
   Offline LLM-judge evaluation exists and informed hypothesis direction; no human participant data collected yet

- [ ] 在 OSF 填写并提交（不要过度润色，越简洁越好）
- [ ] 记录 OSF 预注册 URL：`_________________________`
- [ ] 把 URL 写入 `docs/preregistration.md` 的顶部注释行

---

## 阶段 3：Qualtrics 配置（与 IRB 并行）

### If Between-subjects：3 个固定链接方案

- [ ] 创建 Survey A（Single Agent）链接，记录 URL
- [ ] 创建 Survey B（Hidden Checker）链接，记录 URL
- [ ] 创建 Survey C（Visible Checker）链接，记录 URL
- [ ] 制作一张分配追踪表（见 §附录 B）
- [ ] 确认三份问卷的量表、指南、结束页完全一致，只有刺激材料不同

### If Within-subjects：URL 参数路由方案

- [ ] 在 Qualtrics 中设置 Embedded Data 字段 `cell_id`（从 URL 参数读取）
- [ ] 创建一个主问卷链接，参数格式：`?cell_id=1` 到 `?cell_id=12`
- [ ] 测试所有 12 个 cell 链接能正常加载并分配正确条件
- [ ] 制作一张 cell 分配追踪表（见 §附录 B）

### 通用步骤（两种方案都需要）

- [ ] 按 `docs/qualtrics_qa_checklist.md` 过一遍——至少通过 Part 1–5
- [ ] 自己从头到尾完整跑一次，确认时长和显示正确
- [ ] 确认导出 CSV 的列名与 `results/analyse_user_study.py` 的 schema 完全一致

---

## 阶段 4：IRB 批准后 → Dry Run（1–2 天）

**目标：找 bug，不收正式数据。**

- [ ] 找 2–3 个人完整跑一次问卷（每人跑不同 condition / cell）
- [ ] 他们跑完后，口头问：
  - 哪一页最慢？
  - 哪个题目不清楚？
  - visible checker 的文案是否自然？
  - 感觉总时长合理吗？
- [ ] 根据反馈只改**措辞**，不改题目结构或分析计划
- [ ] 记录 dry run 参与者 ID，这些数据**不进正式分析**

---

## 阶段 5：Small Pilot（N = 6–8）

**目标：确认流程没有系统性问题，不做统计分析。**

- [ ] 发起招募（见 §附录 A 的招募文字）
- [ ] 收满 6–8 人
- [ ] 检查（只看这四件事）：
  - [ ] 平均完成时间在 15–25 分钟
  - [ ] dropout 率 < 30%
  - [ ] attention check 通过率 > 70%
  - [ ] 每个条件 / cell 都有观测

- [ ] 决定是否需要微调措辞（改了就记录 change log）
- [ ] **不根据 pilot 方向修改假设**
- [ ] Pilot 日期阈值（之前的数据不进正式分析）：`_________`

---

## 阶段 6：冻结 → 正式招募

- [ ] 在 Qualtrics 中设置问卷为**只读 / 禁止编辑**（或截图存档当前版本）
- [ ] 更新 OSF 预注册，注明 pilot 已完成，正式收数开始日期
- [ ] 开始招募，目标 N = 48 enrolled
- [ ] **在达到 36 analyzable 或 72 enrolled 前，不得打开数据看结果**

### 追踪表（每天更新）

| 日期 | 今日完成数 | 累计完成数 | Dropout | Attn Fail | 备注 |
|------|-----------|-----------|---------|-----------|------|
| | | | | | |

---

## 阶段 7：数据分析

收够数后才做这步。

- [ ] 从 Qualtrics 导出 CSV
- [ ] 重命名列，使其与分析脚本 schema 匹配
- [ ] 运行：`.venv/bin/python results/analyse_user_study.py`
- [ ] 检查 `results/user_study_results/` 下的 JSON 和图表
- [ ] 对照 H1–H4 写结论

---

## 附录 A：50 词招募文字（直接用）

**Prolific / 邮件通用版**：

> We are recruiting adults (18+) for a 20-minute online study about AI-generated responses in emotional support contexts. You will read short fictional scenarios and rate AI responses on qualities like empathy, safety, and trust. No clinical or personal information is required. Compensation: £3.00 (≈ £9/hr). **Not eligible if you are currently in a mental health crisis.**

---

## 附录 B：分配追踪表模板

### Between-subjects 版

| participant_id | 分配条件 | 完成 | Attn | 纳入 |
|---|---|---|---|---|
| P001 | A | ✅ | ✅ | ✅ |

目标：每个条件 16 人，按顺序轮换（A → B → C → A → …）。

### Within-subjects 版

| participant_id | Cell | 完成 | Attn | 纳入 |
|---|---|---|---|---|
| P001 | 3 | ✅ | ✅ | ✅ |

目标：每个 Cell (1–12) 各 4 人，共 48 enrolled。
