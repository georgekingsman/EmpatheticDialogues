# Pre-Registration: Visible Safety Checking and Calibrated Trust in AI Mental-Health Support

> **Template**: AsPredicted / OSF Pre-Registration  
> **Status**: DRAFT — submit before data collection  
> **Related files**: `docs/user_study_design.md`, `docs/survey_instrument.md`

---

## 1. Authors

[Author names and affiliations — complete before submission]

## 2. Title

Does Visible Safety Checking Promote Calibrated Trust? A Vignette-Based User Study of Maker-Checker AI for Mental-Health Support

## 3. Research Questions

1. **RQ1**: Does a maker-checker architecture improve users' perceived safety and boundary clarity compared to a single-agent baseline?
2. **RQ2**: Does making the checker process visible increase perceived transparency without reducing warmth?
3. **RQ3**: Does visible checking promote calibrated trust (appropriate reliance modulated by risk level) rather than blind trust?

## 4. Hypotheses

| ID | Hypothesis | Direction | Key Contrast |
|----|-----------|-----------|-------------|
| H1 | Checker conditions (B, C) will score higher on perceived safety and boundary clarity than Single Agent (A) | B, C > A | Pairwise A–B, A–C on Q3 (Safety), Q4 (Boundary) |
| H2 | Visible Checker (C) will score higher on perceived transparency than both A and B | C > A, B | Pairwise A–C, B–C on Q5 (Transparency) |
| H3 | Single Agent (A) will score higher on perceived warmth and empathy than B and C (modest effect) | A > B, C | Pairwise A–B, A–C on Q1 (Empathy), Q2 (Warmth) |
| H4 | Visible Checker (C) will show the best reliance calibration: moderate trust combined with highest willingness to seek professional help, especially in high-risk scenarios | Condition × Risk interaction on Q6 × Q8 | Interaction term in LMM |

## 5. Dependent Variables

### Per-Vignette (7-point Likert, 1 = Strongly Disagree, 7 = Strongly Agree)

| Label | Item |
|-------|------|
| Q1: Perceived Empathy | "This response shows understanding of what the person is feeling." |
| Q2: Perceived Warmth | "This response feels warm and caring." |
| Q3: Perceived Safety | "This response avoids giving harmful or dangerous advice." |
| Q4: Boundary Clarity | "This response clearly communicates what the system can and cannot help with." |
| Q5: Transparency | "I understand how this response was generated and reviewed." |
| Q6: Trust | "I trust this system to give me appropriate support." |
| Q7: Willingness to Rely | "I would be comfortable relying on this system's advice for this situation." |
| Q8: Seek Real Help | "After reading this response, I would be more likely to seek professional help if needed." |

### Derived Composites

- **Empathy Composite** = mean(Q1, Q2)
- **Safety Composite** = mean(Q3, Q4)
- **Trust Composite** = mean(Q6, Q7)
- **Reliance Calibration Index** = Trust Composite × Q8, analyzed as function of risk level

### Post-Study

- Overall satisfaction (1–7)
- Crisis comfort (1–7)
- Mental workload — NASA-TLX single item (1–7)
- Referral accuracy (checklist; proportion correct)

## 6. Conditions (Independent Variable)

Within-subjects, 3 levels:

| Condition | Label | Description |
|-----------|-------|-------------|
| A | Single Agent | One AI generates empathetic response; no safety review |
| B | Hidden Checker | Second AI reviews for safety; user sees clean response only |
| C | Visible Checker | Second AI reviews; user sees "✓ Safety reviewed" indicator |

## 7. Design

- **Design**: Within-subjects, vignette-based
- **Stimuli**: 12 vignettes (4 low-risk, 4 medium-risk, 4 high-risk mental health scenarios)
- **Counterbalancing**: 12 Latin-square cells; each participant sees 4A + 4B + 4C; each vignette × condition pairing appears equally across cells
- **Presentation order**: Randomized within participant
- **Practice trial**: 1 neutral scenario (excluded from analysis)

## 8. Participants

- **Target N**: 36 (3 per counterbalancing cell × 12 cells)
- **Recruitment**: Prolific or university participant pool
- **Eligibility**: 18+, fluent in English, not currently in acute mental health crisis (self-reported)
- **Compensation**: [£X / $X] via platform
- **Exclusion criteria (post-hoc)**:
  - Failed attention check
  - Completed study in < 5 minutes (insufficient engagement)
  - Uniform responses across all items (zero variance)

## 9. Analysis Plan

### 9.1 Primary Analyses

For each of the 8 per-vignette measures (Q1–Q8):

**Linear Mixed-Effects Model (LMM):**
```
DV ~ Condition + RiskLevel + Condition:RiskLevel + (1|Participant) + (1|Vignette)
```
- **Fixed effects**: Condition (A, B, C; reference = A), Risk Level (low, medium, high; reference = low), Condition × Risk Level interaction
- **Random effects**: Participant (random intercept), Vignette (random intercept)
- **Pairwise contrasts**: A–B, A–C, B–C with Holm correction for 3 comparisons per DV
- **Effect sizes**: Cohen's d from LMM estimated marginal means and pooled SD
- **Significance threshold**: α = .05 (two-tailed)

**Hypothesis-specific tests:**
- H1: One-sided contrast B > A and C > A on Q3, Q4
- H2: One-sided contrast C > A and C > B on Q5
- H3: One-sided contrast A > B and A > C on Q1, Q2
- H4: Condition × Risk interaction on Reliance Calibration Index

### 9.2 Secondary Analyses

1. **Reliance Calibration Index**:
   - Compute RCI = mean(Q6, Q7) × Q8 for each observation
   - LMM: RCI ~ Condition × RiskLevel + (1|Participant) + (1|Vignette)
   - Key test: Condition C should show steepest positive slope of RCI across risk levels (i.e., trust moderated by risk)

2. **Overall Satisfaction**: One-way repeated-measures ANOVA or Friedman test

3. **Referral Accuracy**: Proportion of correctly identified referral-worthy scenarios; compared descriptively across conditions

4. **Qualitative Analysis**: Open-ended responses coded by two independent raters using thematic analysis; inter-rater reliability assessed with Cohen's κ

### 9.3 Robustness Checks

1. **Ordinal logistic mixed model** (CLMM) as sensitivity analysis for Likert data
2. **Bayesian LMM** with weakly informative priors (if frequentist tests are inconclusive)
3. **Exclude fastest 10%** of respondents and re-run primary analyses
4. **Item-level analysis**: Check individual items (not just composites) for consistent direction

### 9.4 Multiple Comparisons

- Within each hypothesis: Holm correction across the 3 pairwise comparisons
- Across hypotheses: No additional correction (each hypothesis addresses a distinct construct)
- Total pre-registered tests: 4 hypotheses × 3 pairwise = 12 primary contrasts

## 10. Sample Size Justification

- **Effect size**: Medium (d = 0.5), based on offline evaluation showing Empathy Composite difference of Δ = 0.24 on a 1–5 scale (d ≈ 0.46 using pooled SD)
- **Design**: Within-subjects, 12 observations per participant, assumed correlation r = 0.5 between conditions
- **Power analysis**: G*Power, repeated-measures ANOVA, 3 groups, α = .05, power = .80 → N ≈ 28
- **Target**: N = 36 (12 cells × 3 each) to account for ~20% attrition and exclusions

## 11. Existing Data

- **Offline evaluation data exists** (90 scenarios × 3 conditions, LLM-judged) and informed hypothesis direction
- **No human participant data has been collected**
- Hypotheses are derived from offline LLM-judge scores; the user study tests whether these patterns replicate in human perception

## 12. Data Exclusion Criteria

Participants will be excluded if:
1. They fail the embedded attention check item
2. They complete the study in under 5 minutes
3. They provide identical responses (zero variance) across all 12 vignettes

Excluded participants will be replaced to maintain target N per cell.

## 13. Exploratory Analyses (Not Pre-Registered)

The following analyses may be conducted but are explicitly exploratory:
- Moderation by prior chatbot experience or mental health service use
- Response time analysis (does visible checker increase reading time?)
- Mental workload differences across conditions
- Correlation between crisis comfort and reliance calibration index
- Qualitative theme differences across conditions

## 14. Timeline

| Phase | Target |
|-------|--------|
| Pre-registration submission | Before IRB approval |
| IRB submission | Week 1 |
| Pilot study (N = 3–5) | Week 2 |
| Main data collection | Weeks 3–4 |
| Analysis and write-up | Week 5 |

## 15. Materials Availability

All study materials are publicly available in the study repository:
- Benchmark scenarios: `data/scenarios/benchmark.jsonl`
- Frozen stimuli: `results/offline_eval_v2_final/user_study_stimuli.json`
- Counterbalancing matrix: `results/offline_eval_v2_final/counterbalancing_matrix.json`
- Survey instrument: `docs/survey_instrument.md`
- Consent form: `docs/irb_consent.md`
- Analysis script: `results/analyse_user_study.py`
