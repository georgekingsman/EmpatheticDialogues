# Pre-Registration: Visible Safety Checking and Calibrated Trust in AI Mental-Health Support

> **Template**: AsPredicted / OSF Pre-Registration  
> **Status**: FINAL — submit before IRB approval and before any data collection  
> **Related files**: `docs/user_study_design.md`, `docs/survey_instrument.md`, `docs/irb_submission_package.md`

---

## 1. Authors

[PI NAME], [INSTITUTION]  
Supervised by: [SUPERVISOR NAME], [INSTITUTION]  
*(Complete author names before uploading to OSF)*

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

### Primary Outcomes (4 items — these are the only pre-registered confirmatory tests)

| Label | Item | Hypothesis |
|-------|------|-----------|
| Q2: Perceived Warmth | "This response feels warm and caring." | H3 |
| Q3: Perceived Safety | "This response avoids giving harmful or dangerous advice." | H1 |
| Q5: Transparency | "I understand how this response was generated and reviewed." | H2 |
| Q7: Willingness to Rely | "I would be comfortable relying on this system's advice for this situation." | H4 |

### Secondary Outcomes (collected but not primary confirmatory tests)

| Label | Item |
|-------|------|
| Q1: Perceived Empathy | "This response shows understanding of what the person is feeling." |
| Q4: Boundary Clarity | "This response clearly communicates what the system can and cannot help with." |
| Q6: Trust | "I trust this system to give me appropriate support." |
| Q8: Seek Real Help | "After reading this response, I would be more likely to seek professional help if needed." |

### Derived Composites (exploratory)

- **Empathy Composite** = mean(Q1, Q2)
- **Safety Composite** = mean(Q3, Q4)
- **Trust Composite** = mean(Q6, Q7)
- **Reliance Calibration Index** = Trust Composite × Q8, analyzed as function of risk level

### Post-Study (descriptive only)

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

- **Minimum N**: 36 (3 per counterbalancing cell × 12 cells)
- **Target N**: 48 (4 per cell), to allow for ~25% exclusion rate while retaining ≥ 36 analyzable participants
- **Upper cap**: 72 (6 per cell); recruitment stops when cap is reached regardless of exclusion rate
- **Stopping rule**: Recruitment halts when the *analyzable* sample (post-exclusion) reaches 36 **or** when 72 enrolled participants is reached, whichever comes first
- **Recruitment**: Prolific Academic or equivalent university participant pool
- **Eligibility**: 18+, fluent in English, not currently in acute mental health crisis (self-reported)
- **Compensation**: ≥ £9.00/hr pro-rated for ~20 min (≈ £3.00; adjust to platform minimum)
- **Pilot data**: The first 6–12 participants (pilot run) will **not** be included in the main analysis; they are excluded by date threshold set before pilot begins
- **Exclusion criteria (post-hoc, applied before opening the data)**:
  1. Failed the embedded attention-check item
  2. Completed the full study in under 5 minutes (insufficient engagement)
  3. Provided identical responses (zero variance) across all 12 vignette rating items
  4. Withdrew before completing 50% of vignettes

## 9. Analysis Plan

### 9.1 Primary Confirmatory Analyses

**Four pre-registered primary tests** (one per hypothesis, each using one primary DV):

| Hypothesis | DV | Key Contrast | Direction |
|------------|-----|-------------|-----------|
| H1 | Q3 Safety | A–B, A–C | B, C > A |
| H2 | Q5 Transparency | A–C, B–C | C > A, C > B |
| H3 | Q2 Warmth | A–B, A–C | A > B, A > C |
| H4 | Q7 Willingness to Rely | Condition × Risk interaction | C steepest positive slope |

For each primary DV, the **Linear Mixed-Effects Model** is:

```
DV ~ C(condition, Treatment('A')) * C(risk_level, Treatment('low'))
   + (1|Participant) + (1|Vignette)
```

- Pairwise contrasts with **Holm correction** (3 comparisons per DV)
- **Significance threshold**: α = .05 (two-tailed for H1/H2/H3; interaction term for H4)
- **Effect sizes**: Cohen's d from estimated marginal means

> **If Between-subjects design is used**: replace LMM with one-way ANOVA + Tukey HSD; remove Vignette random effect.

### 9.2 Secondary Analyses

All 8 items (Q1–Q8) and all derived composites will be reported descriptively. Additional mixed-model tests on secondary items (Q1, Q4, Q6, Q8) and composite scores are **not pre-registered** and will be clearly labelled as exploratory in the paper.

### 9.3 Robustness Checks (exploratory)

1. **Ordinal logistic mixed model** (CLMM) as sensitivity for Likert data
2. **Exclude fastest 10%** of respondents and re-run primary analyses

### 9.4 Multiple Comparisons

- Within each hypothesis: Holm correction across 3 pairwise comparisons per DV
- **Total pre-registered tests**: 4 hypotheses × up to 3 pairwise = at most 12 contrasts
- No additional cross-hypothesis correction (each addresses a distinct construct)

## 10. Sample Size Justification

- **Effect size target**: Medium (d = 0.5). Offline LLM-judge evaluation found Empathy Composite differences of Δ ≈ 0.24 on a 1–5 scale (d ≈ 0.46). Human ratings are expected to be noisier; d = 0.5 is conservative.
- **Design**: Within-subjects, each participant contributes 4 observations per condition (4 vignettes × 3 conditions = 12 total per participant)
- **Power analysis** (G*Power 3.1, repeated-measures ANOVA, 3 groups, within-factors correlation r = 0.5, α = .05, power = .80): required N ≈ 28
- **Enrolled target**: N = 48 (12 cells × 4) maps to ~36 analyzable after ~25% expected exclusion rate — well above the 28 minimum
- **Upper cap**: N = 72 provides power > .95 even if exclusion rate is lower than expected

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

The following will be clearly labelled as exploratory in any publication:
- All secondary DVs (Q1, Q4, Q6, Q8) and derived composites
- Reliance Calibration Index (Trust × Seek-Help)
- Post-study satisfaction and mental workload
- Referral accuracy checklist
- Moderation by prior chatbot experience or mental health service use
- Response time analysis and mental workload differences

## 14. Timeline

| Phase | Target | Notes |
|-------|--------|-------|
| Pre-registration submission | Before IRB approval | This document; freeze before uploading to OSF |
| IRB submission | Day 1–2 | Use `docs/irb_submission_package.md` |
| IRB approval | ~Day 7–14 | Estimated; institution-dependent |
| Qualtrics QA (full-chain audit) | Day 3–5 | Use `docs/qualtrics_qa_checklist.md` |
| Pilot study (N = 6–12) | Day 1–2 after IRB approval | Flow-check only; data NOT in main analysis |
| Pilot debrief + minor revisions | Day 3 after pilot | Freeze survey; no changes after this point |
| Main data collection (target N = 48) | Days 4–14 after freeze | Do NOT inspect results mid-collection |
| Exclusions applied + analysis | Day 15 | Run `results/analyse_user_study.py` |
| Paper user-study sections | Days 16–30 | |

## 15. Materials Availability

All study materials are publicly available in the study repository:
- Benchmark scenarios: `data/scenarios/benchmark.jsonl`
- Frozen stimuli: `results/offline_eval_v2_final/user_study_stimuli.json`
- Counterbalancing matrix: `results/offline_eval_v2_final/counterbalancing_matrix.json`
- Survey instrument: `docs/survey_instrument.md`
- Consent form: `docs/irb_consent.md`
- Analysis script: `results/analyse_user_study.py`
