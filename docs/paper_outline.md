# Paper Outline v1

## Title (preferred)

**A Double-AI Maker-Checker Architecture for Safer Empathetic Mental-Health Support**

## Central Thesis

We study whether separating empathetic response generation from safety checking
can improve appropriate reliance, boundary clarity, and perceived safety in
mental-health support, and what trade-off this creates with perceived warmth and
empathy.

---

## Research Questions

### RQ1 — Safety improvement
Compared with a single-agent empathetic assistant, can a role-separated
double-AI architecture improve response safety, boundary adherence, and
escalation appropriateness?

### RQ2 — Visibility effect
Does making the checking process visible change users' perceived warmth, trust,
transparency, and willingness to rely on the system?

### RQ3 — Warmth–safety trade-off
What trade-off emerges between emotional support quality and safety-oriented
checking in mental-health support interactions?

---

## Contributions

1. **Reframing**: We reframe hallucination in empathetic mental-health support
   as a human-centered problem of *unsafe reliance* rather than a pure
   generation error.
2. **Architecture**: We propose a role-separated maker-checker architecture
   with hidden and visible checking conditions.
3. **Evaluation protocol**: We introduce a structured evaluation protocol
   covering empathy, safety, boundaries, uncertainty, and escalation.
4. **Empirical characterization**: We empirically characterize the
   warmth–safety trade-off through both offline evaluation and a user study.

---

## 1  Introduction

### Para 1 — Opportunity and concern
LLMs are increasingly used for emotionally sensitive support interactions,
including stress, loneliness, anxiety, and mental-health-related help-seeking.
Recent reviews show strong interest in this area, but also point out persistent
concerns about reliability, safety, evaluation quality, and clinical
applicability.

### Para 2 — Unsafe reliance
The core challenge is not only whether a response is fluent or empathetic, but
whether users may form unsafe reliance on responses that sound caring while
missing risk signals, overstepping boundaries, or failing to escalate
appropriately.

### Para 3 — Role conflict in single-agent design
Existing systems often overload one model with conflicting responsibilities: be
warm, be safe, detect risk, maintain boundaries, and explain uncertainty. We
argue that this role conflict can be addressed through a maker-checker
architecture that separates supportive response generation from safety-oriented
review.

### Para 4 — Visibility as HCI variable
Checking is not only a backend intervention but also an HCI design variable.
Making the checking process visible may improve transparency and trust
calibration, but could also reduce perceived warmth. Recent work on visible
thinking in chatbots supports the idea that displaying internal deliberation
changes user perceptions of empathy, competence, and engagement.

### Para 5 — Study overview
We study a three-condition design — single agent, hidden checker, visible
checker — and evaluate both output quality and user perceptions.

---

## 2  Related Work

### 2.1 Empathetic dialogue systems
- Neural empathetic generation (EMPATHETICDIALOGUES, EPITOME, etc.)
- Chain-of-thought / chain-of-empathy approaches

### 2.2 Safety in mental-health AI
- Risk taxonomy: distress detection, boundary adherence, crisis escalation,
  emotional reliance
- OpenAI / industry safety evaluations for mental-health use-cases

### 2.3 LLM-as-a-judge and calibration
- Using LLMs to evaluate open-ended generation
- Human-anchored calibration (isotonic, ordinal)
- Connection to our existing evaluation backbone

### 2.4 Trust, reliance, and transparency in AI
- Appropriate reliance vs. over-reliance
- Visible vs. hidden AI deliberation (visible thinking literature)
- Maker-checker / dual-process designs in safety-critical systems

---

## 3  Method

### 3.1 Three experimental conditions

| Condition | Architecture | User sees |
|-----------|-------------|-----------|
| A — Single Agent | One model generates supportive reply | Final reply only |
| B — Double AI, Hidden Checker | Maker generates → Checker reviews silently | Final (possibly revised) reply |
| C — Double AI, Visible Checker | Maker generates → Checker reviews | Final reply + safety-review indicator / explanation |

### 3.2 Maker agent
- System prompt: compassionate, professional peer-supporter
- Generates initial empathetic response

### 3.3 Checker agent — four modules

#### 3.3.1 Emotional Appropriateness
Checks whether the reply correctly recognises and matches the user's emotional
state without being dismissive or exaggerated.

#### 3.3.2 Safety and Boundaries
Checks for diagnostic language, false reassurance, dangerous advice, or
therapist/clinician role-playing.

#### 3.3.3 Uncertainty and Grounding
Checks whether the system is overconfident given limited information; whether it
should ask a clarifying question instead of advising.

#### 3.3.4 Escalation Need
Checks for self-harm signals, crisis indicators, or severe deterioration that
require handoff / referral / crisis response.

### 3.4 Policy layer
Decision logic after checker review:
- **approve** — pass Maker response through unchanged
- **revise** — Maker regenerates with checker feedback
- **abstain** — system declines to answer, explains limitation
- **escalate** — switch to crisis-oriented response with resources

### 3.5 Visible checker UX (Condition C)
- Subtle indicator: "✓ Safety reviewed"
- Expandable panel with per-module checker reasoning

---

## 4  Offline Evaluation (Study 1)

### 4.1 Scenario benchmark
- 90 scenarios: 30 low-risk, 30 medium-risk, 30 high-risk/escalation
- Each annotated with: risk_level, topic, recommended_strategy,
  escalation_expected

### 4.2 Metrics
| Metric | What it measures |
|--------|-----------------|
| Emotional appropriateness | Emotion match quality |
| Perceived helpfulness | Practical value of advice |
| Safety | Absence of harmful content |
| Boundary adherence | No diagnostic / clinician overreach |
| Escalation appropriateness | Correct handoff when needed |
| Abstention / revision rate | How often checker intervenes |

### 4.3 Evaluation method
- LLM-as-a-judge with 4-dimension rubric (reused from existing pipeline)
- External human-anchored calibration (existing calibration backbone)
- 3× repeats for stability

### 4.4 Expected analyses
- Per-condition score comparison (A vs B vs C)
- Per-risk-level breakdown
- Error case taxonomy
- Checker intervention rate and accuracy

---

## 5  User Study (Study 2)

### 5.1 Design
- Between-subjects, 3 conditions (A / B / C)
- Participants interact with system on 3–5 pre-defined scenarios
- Mixed: Likert questionnaires + open-ended qualitative feedback

### 5.2 Dependent variables

| Variable | Measurement |
|----------|------------|
| Perceived empathy | Likert scale |
| Perceived warmth | Likert scale |
| Perceived safety | Likert scale |
| Trust | Likert scale |
| Appropriate reliance | Likert scale |
| Transparency | Likert scale |
| Boundary clarity | Likert scale |
| Mental workload | NASA-TLX (adapted) |
| Adoption intention | Likert scale |

### 5.3 Procedure
1. Informed consent + demographics
2. Brief system introduction (condition-specific)
3. Interaction phase (3–5 scenarios)
4. Post-interaction questionnaire
5. Semi-structured debrief (optional, for qualitative data)

### 5.4 Analysis plan
- ANOVA / Kruskal-Wallis across 3 conditions
- Pairwise comparisons (A↔B, A↔C, B↔C)
- Effect sizes (Cohen's d or η²)
- Qualitative coding of open-ended responses

---

## 6  Results

### 6.1 Offline results
- Table: per-condition means across all metrics
- Figure: radar chart (empathy, helpfulness, safety, boundaries, escalation)
- Checker intervention breakdown (approve / revise / abstain / escalate)

### 6.2 User study results
- Table: per-condition means for all DV scales
- Figure: trust × warmth scatter by condition
- Qualitative themes

---

## 7  Discussion

### 7.1 The warmth–safety trade-off
- Quantify how much warmth is lost when safety checking is added
- Discuss whether visibility mitigates or amplifies this cost

### 7.2 Calibrated trust, not blind trust
- "The ideal outcome is not blind trust; it is calibrated trust."
- How the visible checker condition shifts user mental models

### 7.3 Implications for deployment
- When to use hidden vs. visible checking
- Integration with clinical workflows
- Ethical considerations

### 7.4 Limitations
- Scenario-based (not longitudinal)
- English-only
- LLM-based checker may itself have failure modes
- Sample size / power

---

## 8  Conclusion

Restate central finding: role separation improves safety metrics at a
measurable warmth cost; visibility of checking reshapes user trust dynamics.
Call to action: future mental-health AI should separate generation from safety
review, and the HCI design of that separation matters.

---

## Appendices

- A: Full rubric (4 + 4 checker dimensions)
- B: Scenario benchmark (90 items)
- C: Prompt templates (Maker, Checker, Visible Checker)
- D: Survey instrument
- E: Qualitative codebook

---

## Working Abstract (v0)

Large language models are increasingly used for emotionally sensitive support
interactions, yet their failures in mental-health-related settings are not only
matters of factual error, but also of unsafe reliance, unclear boundaries, and
missed escalation. In this paper, we propose a role-separated double-AI
maker-checker architecture for safer empathetic mental-health support. A Maker
agent generates an initial supportive response, while an independent Checker
agent reviews emotional appropriateness, safety and boundaries, uncertainty, and
escalation need before a policy layer determines whether to approve, revise,
abstain, or escalate. We further treat checking visibility as a human-computer
interaction variable, comparing three conditions: a single-agent baseline, a
double-AI system with hidden checking, and a double-AI system with visible
checking explanations. We evaluate the proposed design through both offline
scenario-based benchmarking and a user study measuring perceived empathy,
warmth, safety, transparency, trust, appropriate reliance, and boundary clarity.
Our goal is not blind trust, but calibrated trust: users should feel supported
while also recognising uncertainty and the need for real-world help when
appropriate. This work reframes empathetic mental-health support as a
maker-checker problem and offers a reproducible platform for studying the
trade-off between warmth and safety in supportive AI systems.
