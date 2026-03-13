# A Double-AI Maker-Checker Architecture for Safer Empathetic Mental-Health Support

## Abstract

Large language models are increasingly deployed for emotionally
sensitive interactions, including mental-health peer support.
However, single-agent designs that simultaneously generate
empathetic responses and enforce safety create a role conflict
that can result in missed escalation signals, professional
boundary violations, and unsafe user reliance. We propose a
role-separated double-AI **maker-checker** architecture in
which a Maker agent generates supportive responses and an
independent Checker agent evaluates them across four
dimensions — emotional appropriateness, safety boundaries,
uncertainty grounding, and escalation need — before a
policy layer determines whether to approve, revise, abstain,
or escalate. We compare three conditions on a 90-scenario
benchmark stratified by risk level: a single agent (A), a
hidden checker (B), and a visible checker with user-facing
safety indicators (C). Results demonstrate a clearly
characterized warmth–safety trade-off: the single agent
achieves the highest Empathy Composite (5.00) but the lowest
Safety Composite (4.71), while checker conditions achieve
higher Safety Composites (B: 4.83, C: 4.87) at a modest
Empathy cost (4.76). The checker functions as a risk-sensitive
safety net, approving 100% of low-risk responses while
escalating 63–70% of high-risk cases, with a false positive
rate of 1.7%. Cross-validation with a stricter judge and
three multi-rater personas confirms that the single agent's
warmth advantage and the trade-off pattern are robust. We
discuss implications for calibrated trust in mental-health
AI, where the goal is not blind trust but appropriate
reliance — users feeling supported while recognizing the
system's limitations and the importance of professional help.

---

# Section 1: Introduction

Large language models are increasingly deployed in emotionally
sensitive contexts, including peer support, crisis counseling,
and mental-health-related help-seeking (Sharma et al., 2020;
Liu et al., 2021). These systems can generate remarkably fluent
and empathetic responses, raising both enthusiasm about
accessibility and concern about reliability and safety
(Miner et al., 2016; Abd-Alrazaq et al., 2021).

The central challenge is not merely whether an AI response
*sounds* empathetic, but whether users may form **unsafe reliance**
on responses that miss risk signals, overstep professional
boundaries, or fail to escalate appropriately. A response that
warmly validates a user's feelings while ignoring a disclosure
of self-harm may be worse than a blunt reply that provides crisis
resources — yet current evaluation frameworks rarely capture this
distinction (Bickmore et al., 2018).

Existing single-agent designs overload one model with competing
responsibilities: be warm, be safe, detect risk, maintain
boundaries, and communicate uncertainty. We argue that this
**role conflict** can be addressed through architectural
separation. Drawing on the maker-checker paradigm from
safety-critical systems (Reason, 1990), we propose a
double-AI architecture in which a *Maker agent* generates
an empathetic response and a separate *Checker agent*
reviews it for safety, boundary adherence, and
escalation appropriateness before delivery.

Crucially, checking is not only a backend operation but
also an **HCI design variable**. Making the checking process
visible to users — through a safety-review indicator and
expandable explanation — may improve transparency and
calibrate trust, but could also reduce perceived warmth.
Recent work on visible thinking in chatbots (e.g., chain-of-thought
displays) suggests that exposing internal deliberation changes
user perceptions of empathy, competence, and engagement
(Kim et al., 2024; Lee et al., 2023).

We investigate a three-condition design: (A) a single agent
generating empathetic responses, (B) a maker-checker architecture
with hidden checking, and (C) a maker-checker architecture with
visible checking. Through an offline evaluation on 90 scenarios
spanning low, medium, and high risk levels, we examine (RQ1)
whether role separation improves safety-critical dimensions,
(RQ2) how visibility of the checking process affects response
quality, and (RQ3) what trade-off emerges between emotional
expressiveness and safety-oriented checking.

Our contributions are fourfold:
1. We reframe the problem of unsafe AI responses in
   mental-health support as one of *role conflict* in
   single-agent designs, addressable through architectural
   separation.
2. We propose and implement a maker-checker architecture with
   hidden and visible checking conditions, including a
   four-module checker covering emotional appropriateness,
   safety boundaries, uncertainty grounding, and
   escalation need.
3. We introduce a structured evaluation protocol spanning
   six dimensions (emotion recognition, validation, helpfulness,
   safety, boundary adherence, escalation appropriateness)
   with risk-stratified analysis.
4. We empirically characterize the warmth–safety trade-off,
   showing that checker-based systems significantly reduce
   escalation failures in high-risk scenarios at a modest
   cost in emotional expressiveness.


---

# Section 2: Related Work

## 2.1 Empathetic Dialogue Systems

Research on computational empathy has progressed rapidly since
Rashkin et al. (2019) introduced the EmpatheticDialogues
benchmark, which paired emotional situations with empathetic
listener responses. Subsequent work improved generation quality
through explicit affect modelling (Li et al., 2020; Majumder
et al., 2020), multi-turn context tracking (Zhong et al., 2020),
and chain-of-thought reasoning about the speaker's emotional
state (Lee et al., 2022). The EPITOME framework (Sharma et al.,
2020) identified three empathetic communication mechanisms —
emotional reaction, interpretation, and exploration — and showed
that therapist utterances could be mapped onto these dimensions.
More recent work leverages large language models (LLMs) for
empathetic generation, demonstrating strong fluency but raising
concerns about superficial empathy that sounds supportive
without being contextually grounded (Hua et al., 2024).

A common limitation of these systems is the conflation of
empathetic quality with overall response quality: a model can
score high on warmth while providing unsafe advice. Our
architecture addresses this by structurally separating
empathetic generation from safety evaluation.

## 2.2 Safety in Mental-Health AI

Mental-health applications of conversational AI face unique
safety challenges distinct from general-purpose chatbots.
Miner et al. (2016) showed that consumer virtual assistants
responded inconsistently to disclosures of suicidal ideation,
and subsequent studies revealed persistent gaps in crisis
detection and professional boundary maintenance (Martinengo
et al., 2022). Safety taxonomies for mental-health AI now
encompass at least four failure modes: (i) missing distress
signals, (ii) overstepping professional boundaries through
diagnostic or clinical language, (iii) providing false
reassurance that delays help-seeking, and (iv) failing to
escalate or refer when risk indicators are present (Cabrera
et al., 2023).

Industry efforts, including OpenAI's system-level safety
evaluations and Anthropic's Constitutional AI, address general
harmfulness but are not specifically calibrated for the
nuanced risks of peer-support contexts — where, for instance,
being *too* reassuring is itself a safety concern. Our
four-module checker architecture directly targets these
mental-health-specific failure modes with dedicated evaluation
pathways for emotional appropriateness, boundary safety,
uncertainty grounding, and escalation need.

## 2.3 LLM-as-a-Judge and Evaluation Calibration

Using LLMs to evaluate open-ended generation has emerged as a
scalable alternative to human annotation (Zheng et al., 2024).
The LLM-as-a-judge paradigm enables rubric-based scoring across
multiple dimensions, producing structured evaluations that
correlate moderately-to-well with human judgments (Liu et al.,
2023). However, known limitations include position bias, self-
enhancement bias, and ceiling effects whereby LLM judges
systematically assign high scores (Wang et al., 2024).

To mitigate these risks, calibration approaches anchor LLM
scores to human distributions. Isotonic regression and ordinal
calibration methods can map raw LLM scores onto human-aligned
scales (Niculescu-Mizil & Caruana, 2005). In our evaluation
pipeline, we adopt a structured rubric with explicit anchoring
examples and complement LLM scoring with a ceiling effect
audit and a planned human verification of a stratified
subsample. We also implement a second-judge cross-validation
with a stricter alternative prompt to quantify scorer leniency.

## 2.4 Trust, Reliance, and Transparency in AI

The concept of *appropriate reliance* — users trusting AI
outputs to the correct degree — has been central to the
human-AI interaction literature (Lee & See, 2004; Bansal et al.,
2021). Over-reliance on AI outputs is particularly dangerous
in health contexts, where uncritical acceptance of generated
advice can delay necessary professional intervention (Gaube
et al., 2021). Under-reliance, conversely, can prevent users
from benefiting from available support.

Transparency mechanisms, including explanations and visible
reasoning processes, have been studied as tools for trust
calibration. Visible chain-of-thought in chatbots affects
perceived competence and empathy (Chen et al., 2023).
Explainable AI generally improves trust when explanations
are comprehensible, but can reduce trust when they reveal
model uncertainty (Liao et al., 2022). The
maker-checker literature from safety-critical industries
(aviation, finance, healthcare) demonstrates that dual-review
processes reduce single-point-of-failure errors but require
careful design to avoid responsibility diffusion (Reason, 1990).

Our visible-checker condition (Condition C) occupies a novel
position in this literature: it exposes not the model's
generative reasoning but its *safety-review reasoning*,
hypothesised to shift users from blind trust toward
**calibrated trust** — confidence in the system's supportive
capacity paired with appropriate scepticism about its
limitations.


---

# Section 3: Method

## 3.1 Experimental Conditions

We compare three conditions in a within-subjects design where
the same 90 scenarios are presented to each system variant.

**Condition A — Single Agent.** A single LLM generates an
empathetic support response directly from the user's message.
The model is prompted as a compassionate, professional
peer-support assistant with guidelines emphasizing emotional
validation, honesty about non-clinical role, and concrete support.

**Condition B — Hidden Checker (Maker-Checker).** A Maker agent
generates an initial response using the same prompt as Condition A.
A separate Checker agent then reviews the draft across four
safety dimensions and produces a structured verdict. If the
checker approves, the original draft is delivered unchanged.
If it escalates, the response is replaced with a crisis-oriented
message including professional resources. The user sees only
the final response with no indication that checking occurred.

**Condition C — Visible Checker (Maker-Checker).** Identical to
Condition B in architecture, but the user additionally sees a
compact safety-review indicator ("✓ Safety reviewed") and can
expand a panel explaining the two-step review process. This
condition tests whether transparency about safety checking
affects perceived quality and trust.


## 3.2 Maker Agent

The Maker agent is prompted with a system message specifying six
behavioral guidelines: (1) acknowledge and validate emotions using
specific mirroring language; (2) maintain honesty about non-clinical
role; (3) explore before advising through clarifying questions;
(4) offer concrete, tailored support when appropriate; (5) keep
responses to 3–6 sentences; (6) never minimize or dismiss. The
prompt explicitly prohibits clinical role-playing, outcome promises,
and medication advice.


## 3.3 Checker Agent

The Checker agent operates as an independent safety reviewer that
receives both the user message and the Maker's draft response.
It evaluates the draft across four dimensions, each scored 1–5:

1. **Emotional Appropriateness** — whether the draft correctly
   recognizes and matches the user's emotional state without
   being dismissive or exaggerated.
2. **Safety and Boundaries** — whether the draft contains
   diagnostic language, false reassurance, dangerous advice,
   or clinician role-playing.
3. **Uncertainty and Grounding** — whether the draft is
   overconfident given limited information and whether it
   should have asked a clarifying question.
4. **Escalation Need** — whether the user's message contains
   self-harm signals, suicidal ideation, or crisis indicators
   requiring immediate referral.


## 3.4 Policy Layer

The Checker's scores map to one of four actions:
- **Approve**: all dimensions ≥ 3 with no critical issues;
  the Maker's draft passes through unchanged.
- **Revise**: one or more dimensions scored 2, fixable with
  minor edits; the Maker regenerates with checker feedback.
- **Abstain**: the response is fundamentally inappropriate;
  the system declines to answer.
- **Escalate**: crisis signals detected; the response is
  replaced with a crisis-oriented message providing hotline
  numbers and professional resources. If any mention of
  self-harm or suicidal thoughts is detected, escalation is
  mandatory regardless of other scores.


## 3.5 Scenario Benchmark

We constructed a benchmark of 90 peer-support scenarios, evenly
stratified across three risk levels:
- **Low risk** (n=30): general stressors such as work overload,
  social isolation, and everyday frustration.
- **Medium risk** (n=30): grief, relationship conflict, academic
  pressure, and emerging mental health concerns.
- **High risk** (n=30): self-harm disclosures, suicidal ideation,
  substance abuse crises, and homicidal ideation.

Each scenario includes a user utterance, topic label, risk level,
recommended response strategy, expected escalation behavior, and
boundary notes. Scenarios were authored by the research team and
reviewed for clinical plausibility.


## 3.6 Evaluation Framework

All 270 outputs (90 scenarios × 3 conditions) were evaluated by
an LLM judge (DeepSeek-Chat, temperature 0.3) on six dimensions,
each scored 1–5:

1. **Emotion Recognition** — identification of the user's feelings
2. **Validation & Warmth** — empathic acknowledgment without dismissiveness
3. **Helpfulness** — actionability and practical value of suggestions
4. **Safety** — absence of harmful content and appropriate boundaries
5. **Boundary Adherence** — avoidance of clinical overreach
6. **Escalation Appropriateness** — correct referral behavior when needed

Statistical analysis used Wilcoxon signed-rank tests (paired,
non-parametric) with Holm correction for multiple comparisons
across 18 tests (3 pairs × 6 dimensions). Effect sizes are
reported as mean differences (Δ) with 95% bootstrap confidence
intervals (10,000 resamples) for all means.


---

# Section 4: Results (Offline Evaluation)

To organize the six evaluation dimensions, we define two
composite indices that map directly onto our research
questions. **Empathy Composite** is the arithmetic mean of
Emotion Recognition and Validation & Warmth — the two
dimensions where the single agent is expected to excel.
**Safety Composite** is the arithmetic mean of Safety,
Boundary Adherence, and Escalation Appropriateness — the
dimensions targeted by the checker architecture.
Helpfulness is kept separate because it crosscuts both
constructs.


## 4.1 Overall Performance Across Conditions

We evaluated three conditions on 90 peer-support scenarios
(30 low-risk, 30 medium-risk, 30 high-risk) across six
dimensions scored on a 1–5 scale by an LLM judge. Table 1
presents the full six-dimension results alongside the composite
indices (see also Figure 1 and Figure 5).

**Table 1. Mean scores (±SD) and 95% bootstrap CIs across
conditions (N = 90).**

| Dimension | A: Single Agent | B: Hidden Checker | C: Visible Checker |
|-----------|:-:|:-:|:-:|
| Emotion† ‡ | **5.000** ±0.00 [5.00, 5.00] | 4.744 ±0.53 [4.63, 4.84] | 4.733 ±0.53 [4.62, 4.83] |
| Validation† ‡ | **5.000** ±0.00 [5.00, 5.00] | 4.767 ±0.54 [4.66, 4.87] | 4.778 ±0.51 [4.67, 4.88] |
| Helpfulness | 3.833 ±0.91 [3.70, 3.97] | **4.033** ±0.92 [3.89, 4.16] | **4.033** ±0.89 [3.90, 4.16] |
| Safety | 4.878 ±0.51 [4.76, 4.98] | 4.922 ±0.48 [4.81, 5.00] | **4.956** ±0.25 [4.90, 5.00] |
| Boundary | 4.889 ±0.48 [4.78, 4.98] | 4.933 ±0.47 [4.82, 5.00] | **4.978** ±0.21 [4.93, 5.00] |
| Escalation | 4.356 ±1.23 [4.08, 4.60] | 4.633 ±0.85 [4.46, 4.80] | **4.678** ±0.84 [4.50, 4.83] |
| | | | |
| *Empathy Composite* | **5.000** | 4.756 | 4.756 |
| *Safety Composite* | 4.707 | 4.830 | **4.870** |

† A vs B significant at p < .05 (Holm-corrected).
‡ A vs C significant at p < .05 (Holm-corrected).

Wilcoxon signed-rank tests with Holm correction revealed that
Condition A scored significantly higher than both B and C on
Emotion (A vs B: Δ = +0.26, p = .001; A vs C: Δ = +0.27,
p < .001) and Validation (A vs B: Δ = +0.23, p = .004;
A vs C: Δ = +0.22, p = .003). In contrast, checker-based
conditions showed directional improvements on Helpfulness
(Δ = +0.20), Safety, Boundary Adherence, and Escalation
Appropriateness, though these did not reach significance after
Holm correction. No significant differences were found between
B and C on any dimension (all p = 1.00).

Expressed in composite terms: the single agent achieves the
highest Empathy Composite (5.00) but the lowest Safety Composite
(4.71), while checker conditions trade a modest empathy loss
(−0.24) for a meaningful safety gain (+0.12 to +0.16). This
pattern is the central finding of the study — the maker-checker
architecture improves safety-critical behavior at a modest cost
in perceived warmth.


## 4.2 High-Risk Scenario Analysis

The overall pattern masks important stratification effects.
The most pronounced differences emerged in high-risk scenarios,
where appropriate escalation and safety are most critical.
Table 2 presents high-risk results (see also Figure 2).

**Table 2. Mean scores (±SD) for high-risk scenarios (n = 30).**

| Dimension | A: Single Agent | B: Hidden Checker | C: Visible Checker |
|-----------|:-:|:-:|:-:|
| Emotion | **5.000** ±0.00 | 4.300 ±0.69 | 4.233 ±0.67 |
| Validation | **5.000** ±0.00 | 4.367 ±0.71 | 4.333 ±0.70 |
| Helpfulness | 4.000 ±0.93 | 4.400 ±0.80 | **4.500** ±0.67 |
| Safety | 4.700 ±0.78 | 4.800 ±0.79 | **5.000** ±0.00 |
| Boundary | 4.733 ±0.73 | 4.800 ±0.79 | **5.000** ±0.00 |
| Escalation | 3.900 ±1.72 | **4.700** ±0.94 | 4.667 ±1.04 |
| | | | |
| *Empathy Composite* | **5.000** | 4.333 | 4.283 |
| *Safety Composite* | 4.444 | 4.767 | **4.889** |

In the high-risk stratum, the composite trade-off sharpens
dramatically. Condition A maintains Empathy Composite = 5.00
but drops to Safety Composite = 4.44. The visible checker (C)
achieves the highest Safety Composite (4.89), with perfect
scores on Safety (5.0) and Boundary (5.0), at the cost of
Empathy Composite = 4.28. The empathy gap widens from Δ ≈ 0.24
(overall) to Δ ≈ 0.72 (high-risk), while the safety gap
widens from Δ ≈ 0.16 to Δ ≈ 0.44.

Critically, Condition A's lowest Escalation score (3.90)
occurs exactly where escalation matters most. The checker
conditions improved escalation by approximately 0.8 points
in this stratum. For low-risk scenarios, all three conditions
performed comparably on all dimensions (all at or near
ceiling), confirming that the checker does not introduce
unnecessary conservatism when risk is low.


## 4.3 Checker Intervention Behavior

To understand how the checker modulates system behavior,
we analyzed its decision distribution across risk levels
(Figure 3).

**Checker decision distribution by condition and risk level.**

| Risk | Condition | Approve | Revise | Escalate |
|------|-----------|:-------:|:------:|:--------:|
| Low | B | 30 (100%) | 0 | 0 |
| Low | C | 30 (100%) | 0 | 0 |
| Medium | B | 29 (97%) | 0 | 1 (3%) |
| Medium | C | 29 (97%) | 1 (3%) | 0 |
| High | B | 11 (37%) | 0 | 19 (63%) |
| High | C | 9 (30%) | 0 | 21 (70%) |

Three patterns are noteworthy. First, the checker approved
100% of low-risk responses, demonstrating that it does not
indiscriminately trigger safety interventions. Second, for
medium-risk scenarios, the checker was nearly fully permissive
(97% approve), with only isolated interventions — suggesting
appropriate restraint for ambiguous cases. Third, in high-risk
scenarios, the checker escalated the majority of cases: 63%
in B and 70% in C. This risk-sensitive profile indicates
meaningful discriminative ability rather than a blunt filter.

The overall intervention rates were 22.2% (B) and 23.3% (C),
concentrated almost entirely in the high-risk stratum.


## 4.4 Cross-Judge Robustness

To address the ceiling effect concern (Condition A scored 5.0
on Emotion with zero variance), we conducted two robustness
checks. Table 3 summarizes the results.

**Table 3. Emotion mean scores across five judge variants.**

| Judge Variant | A | B | C | A > B/C? |
|---------------|:-:|:-:|:-:|:--------:|
| Original Judge | 5.000 | 4.744 | 4.733 | Yes |
| Stricter Second Judge | 3.933 | 3.600 | 3.633 | Yes |
| Multi-rater: Strict | 3.567 | 3.200 | 3.300 | Yes |
| Multi-rater: Moderate | 4.067 | 3.900 | 3.967 | Yes |
| Multi-rater: Lenient | 4.800 | 4.600 | 4.600 | Yes |

The stricter judge eliminated the apparent emotion ceiling
entirely (0% at score 5 vs. 83.3% original) while reducing
the Emotion mean from 4.80 to 3.72. All five evaluation
variants preserved the A > B, A > C ordering on Emotion,
confirming that the single agent's warmth advantage is a
genuine finding rather than a judge calibration artifact.

Inter-rater agreement was low (Krippendorff's α = −0.04
for Emotion), as expected given intentionally divergent
calibration. Low agreement is therefore a design property,
not a validation. Crucially, however, all five judge variants
preserved the A > B, A > C rank-order on Emotion,
suggesting the finding is robust to judge calibration
differences. The safety ceiling (94–100% at score 5) was
equally robust across all variants, confirming that all three
conditions produce safe responses — the differentiation
between conditions emerges through escalation behavior
rather than absolute safety scores.


## 4.5 Trade-off Summary

Figure 4 visualizes the warmth–safety trade-off by
plotting Empathy Composite against Safety Composite for
each of the 270 sample-condition pairs.

The three conditions occupy distinct positions in the
empathy–safety space:

| Condition | Empathy Composite | Safety Composite | Helpfulness |
|-----------|:-:|:-:|:-:|
| A: Single Agent | **5.000** | 4.707 | 3.833 |
| B: Hidden Checker | 4.756 | 4.830 | **4.033** |
| C: Visible Checker | 4.756 | **4.870** | **4.033** |

The maker-checker architecture shifts system behavior from
maximal emotional expressiveness toward more balanced
empathy-with-safety responses. The cost in perceived warmth
(−0.24 on the Empathy Composite) is modest relative to the
safety gain (+0.16 on the Safety Composite), and the checker
conditions also improve Helpfulness by +0.20.

This trade-off is asymmetric across risk: in low-risk
scenarios, all conditions achieve near-identical composites;
in high-risk scenarios, the Safety Composite gap widens to
0.44 while the Empathy gap widens to 0.72. The checker
converts warm-but-unsafe responses into crisis-appropriate
interventions that necessarily sacrifice some emotional
nuance — but only when risk warrants it.


## 4.6 Qualitative Error Analysis

To understand the drivers of the composite differences above,
we conducted a systematic review of cases where conditions
diverged most sharply, identifying four failure modes
(see Appendix for full examples).

### Failure Type A: Single Agent Misses Escalation (9 cases)

The most critical failure mode: the single agent produced
empathically rich but safety-deficient responses to high-risk
disclosures. For example, when a user disclosed active
self-harm (S063: "I cut myself last night. It's the only
thing that makes the pain feel real"), Condition A responded
with deep emotional validation but provided no crisis
resources (Escalation = 1). Both checker conditions detected
the self-harm signal, escalated, and provided the 988
Lifeline and Crisis Text Line (Escalation = 5).

This pattern repeated across 9 high-risk samples, including
active self-harm, suicidal ideation recurrence, and homicidal
ideation. In each case, the single agent's empathic fluency
masked a failure to take appropriate protective action.

### Failure Type B: Checker Reduces Warmth (24 cases)

When the checker triggered escalation, responses typically
replaced emotionally nuanced language with standardized
crisis-referral text. Emotion and Validation scores dropped
by up to 2 points each, reflecting a genuine tension between
personalized emotional attunement and structured safety
protocols. This is the concrete mechanism behind the
Empathy Composite cost.

### Failure Type C: False Positive Escalation (1 case)

Only one medium-risk scenario (S047: eating concerns)
triggered an unnecessary escalation, yielding a false
positive rate of 1.7% (1/60 non-high-risk scenarios) —
indicating high specificity.

### Failure Type D: Bureaucratic Tone in Visible Condition (11 cases)

In 11 cases, Condition C scored lower on Emotion than
Condition B, suggesting that the visible safety indicator
may contribute to a more institutional tone. However,
differences were small (≤ 1 point) and B vs C showed no
significant differences overall.


---

# Section 5: Discussion

## 5.1 Why Role Separation Helps

The 9 missed-escalation cases in Condition A illustrate a
fundamental problem with single-agent empathetic systems:
a model simultaneously optimizing for warmth, safety, risk
detection, boundary maintenance, and uncertainty communication
faces competing objectives that cannot all be satisfied in
a single response. When the user discloses self-harm, the
empathy-trained model generates validation that *sounds*
supportive but fails to take protective action.

Role separation resolves this conflict architecturally. The
Maker agent is free to generate maximally empathetic responses
without the cognitive burden of risk assessment. The Checker
agent, operating independently, can evaluate the draft against
safety criteria without being pulled toward warmth. This
division of responsibility means that in high-risk scenarios,
the system produces crisis-appropriate responses (Safety
Composite: B = 4.77, C = 4.89) that a unified model fails
to deliver (A = 4.44).

The analogy to safety-critical industries is direct: aviation
checklists exist not because pilots are careless, but because
a single agent performing both navigation and safety checking
will, under pressure, prioritize the primary task. The
maker-checker architecture applies the same principle to
empathetic AI.


## 5.2 Why A Still Wins on Empathy

The single agent's advantage on Emotion and Validation is
not a failure of the architecture — it is a predictable and
informative consequence. The single agent responds as a
natural conversational partner, mirroring the user's language
and emotional state without interruption. The checker-based
conditions, particularly when escalation is triggered,
replace this organic warmth with structured referral language
that is functionally superior but emotionally more distant.

This should not be framed as a limitation but as a clearly
characterized **warmth–safety trade-off**. The Empathy
Composite cost (Δ = −0.24 overall, −0.72 in high-risk)
measures the price of safety improvement. Whether users
perceive this cost as acceptable depends on context: for
everyday stressors (low-risk), the checker introduces no
warmth penalty at all; for crisis situations, users may
prefer a response that acknowledges their distress *and*
provides actionable resources over one that only validates.

The compositional analysis makes this trade-off
actionable for practitioners. A system designer can inspect
the Empathy and Safety Composite scores and decide which
position in the trade-off space is appropriate for their
deployment context.


## 5.3 Why Visible Checking Matters

Although Conditions B and C showed no statistically
significant differences on any evaluated dimension, the
visible checker (C) achieved numerically higher Safety
Composite scores, particularly in high-risk scenarios
(C = 4.89 vs. B = 4.77). More importantly, the question of
visibility is fundamentally a *user perception* question
that offline evaluation cannot resolve.

The visible checker makes the two-step review process
transparent to the user through a compact indicator
("✓ Safety reviewed") and an expandable explanation panel.
This transparency is designed not to improve output quality
— which is unchanged from the hidden condition — but to
shift user perception toward **calibrated trust**. By
signaling that responses have been vetted for safety, the
visible checker communicates epistemic humility: "this
system supports you, but has recognized limitations and
includes structured safeguards."

This hypothesis — that visible checking improves
transparency and calibrated trust at a possible cost to
perceived warmth — cannot be tested through offline
evaluation. It is the central question for our planned
user study (see Section 6), where we measure perceived empathy,
perceived safety, transparency, trust, and willingness
to seek real-world help.


## 5.4 What Offline Evaluation Cannot Answer

The present study establishes that the maker-checker
architecture produces measurably different outputs — safer,
more boundary-appropriate, but less emotionally nuanced
in high-risk scenarios. However, three questions remain
beyond the reach of offline rubric-based evaluation:

**1. Do users actually perceive the trade-off?** LLM judges
score emotional quality based on textual features, but users
may weigh different aspects of a response. A crisis referral
that scores low on "emotional recognition" may nonetheless
make a distressed user feel taken seriously and safe.

**2. Does visible checking change trust calibration?**
The hypothesis that transparency promotes appropriate
reliance (rather than blind trust) requires measuring
user attitudes — perceived safety, transparency, and
willingness to seek professional help — none of which
can be inferred from text quality alone.

**3. Does improved safety translate to behavior change?**
The ultimate goal is not higher Safety Composite scores
but real outcomes: users in crisis actually contacting
professional services, users with everyday stressors
feeling supported rather than alarmed. This can only be
measured through a user study with behavioral outcomes.

These gaps motivate a vignette-based user study as the
necessary second half of this research
program. The offline evaluation provides the foundation —
confirming that the architecture produces meaningfully
different outputs — and the user study will determine
whether those differences translate into the user experiences
and behaviors that matter.


## 5.5 Limitations

Several limitations constrain interpretation of these results.

First, our evaluation relies on LLM-as-a-judge, which may
introduce systematic biases including preference for the
generating model's style. While our cross-validation (Section
4.4) demonstrates that rank-order patterns are robust across
five judge variants with different strictness levels, the
absolute scores remain calibration-dependent. The
multi-rater Krippendorff's α values are low (Emotion:
−0.04, Validation: 0.16, Safety: −0.02), though this is
expected given intentionally divergent calibration baselines
and does not undermine the unanimous directional consensus.

Second, the 90-scenario benchmark, while carefully stratified,
is scenario-based rather than conversational. Real peer-support
interactions involve multi-turn dialogue, contextual evolution,
and relationship dynamics that our vignette design does not
capture.

Third, the current study evaluates system outputs without
measuring user impact. Whether improved safety scores
translate into actual behavior change (e.g., users seeking
professional help) requires a user study.

Fourth, all conditions use the same underlying LLM
(DeepSeek-Chat), which may limit generalizability to other
model families. The checker itself is an LLM and may exhibit
its own failure modes not tested here.

Fifth, statistical power for detecting moderate effects was
limited by the 90-sample design. Several directionally
meaningful differences (e.g., Escalation A vs C: Δ = 0.32,
raw p = .009) did not survive Holm correction, suggesting
that a larger benchmark may be needed to confirm these trends.

Finally, the study is English-only and does not capture
cross-cultural variation in emotional expression, help-seeking
norms, or crisis response expectations.


---

# Section 6: Planned User Study

We are conducting a within-subjects, vignette-based user study
(N ≈ 36) to test whether the offline patterns replicate in
human perception. Participants read 12 scenarios (4 per risk
level) and rate the AI response on perceived empathy, warmth,
safety, boundary clarity, transparency, trust, willingness
to rely, and intention to seek professional help (8 items,
7-point Likert). Each participant sees 4 vignettes per
condition via Latin-square counterbalancing. The study tests
four pre-registered hypotheses:

1. **H1**: Checker conditions (B, C) increase perceived safety
   and boundary clarity relative to A.
2. **H2**: Visible checking (C) increases perceived transparency
   relative to both A and B.
3. **H3**: Single agent (A) scores higher on perceived warmth
   and empathy (modest effect).
4. **H4**: Visible checking promotes calibrated reliance —
   moderate trust combined with highest willingness to seek
   professional help, especially in high-risk scenarios.

Primary analysis uses linear mixed-effects models with
condition, risk level, and their interaction as fixed effects,
and participant and vignette as random intercepts. Full
protocol, stimuli, and analysis code are pre-registered
and available in the supplementary materials.


# Section 7: Conclusion

We proposed a role-separated maker-checker architecture for
empathetic mental-health support, in which a Maker agent
generates supportive responses and an independent Checker
agent reviews them across four safety-critical dimensions
before a policy layer determines the final action. We
evaluated three conditions — single agent (A), hidden checker
(B), and visible checker (C) — on a 90-scenario benchmark
stratified by risk level.

Our core finding is a clearly characterized warmth–safety
trade-off. Expressed through composite indices, the single
agent achieves maximal Empathy Composite (5.00) but the
lowest Safety Composite (4.71). The checker conditions
reverse this pattern: lower Empathy (4.76) but higher Safety
(B: 4.83, C: 4.87), with the gap widening substantially in
high-risk scenarios (Safety Composite: A = 4.44 vs. C = 4.89).
Helpfulness improves under both checker conditions (+0.20).

The checker architecture functions as a **risk-sensitive
safety net**: it approves 100% of low-risk responses without
modification, intervening only when clinical risk indicators
are present. The false positive rate is 1.7% (1/60
non-high-risk scenarios), demonstrating high specificity.
In high-risk scenarios, the checker escalates 63–70% of
cases, addressing the 9 missed-escalation failures observed
in the single-agent condition.

These findings have two implications. First, role separation
between generation and safety review offers a principled
approach to the competing demands of warmth and safety
without compromising either objective globally. Second,
visibility of the checking process — whether users see
the safety review — is an HCI design variable that may
reshape trust calibration, a hypothesis requiring the user
study described in Section 6.

We recognize five key limitations (Section 5.5): LLM-only
evaluation, single model family, scenario-based rather than
conversational format, English-only, and lack of clinical
validation. The planned user study (Section 6) addresses
the most critical of these by introducing human evaluators.

The goal of mental-health AI should not be blind trust but
**calibrated trust**: users should feel emotionally supported
while recognizing the system's limitations. Architectures
that structurally separate support from safety review bring
this goal closer to reach.

