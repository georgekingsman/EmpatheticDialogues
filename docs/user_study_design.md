# User Study Design: Visible Safety Checking and Calibrated Trust

## 1. Research Questions

1. **RQ1**: Does a maker-checker architecture improve users' *perceived safety* and *boundary clarity* compared to single-agent responses?
2. **RQ2**: Does making the checker process *visible* increase perceived *transparency* without reducing *warmth*?
3. **RQ3**: Does visible checking promote *calibrated trust* — i.e., appropriate reliance rather than blind trust?

## 2. Study Design

**Design**: Within-subjects, vignette-based evaluation.

| Condition | Label | Description |
|-----------|-------|-------------|
| A | Single Agent | One empathetic response, no safety review |
| B | Hidden Checker | Response reviewed internally, user sees clean output |
| C | Visible Checker | Response reviewed, user sees safety indicator |

**Participants**: N = 30–40 (power analysis based on medium effect size d ≈ 0.5, α = .05, power = .80 for within-subjects).

**Platform**: Prolific or university participant pool; must be 18+, fluent in English.

## 3. Stimuli

From the 90-scenario benchmark, select **12 vignettes** (4 per risk level, balanced topics):

| Risk | IDs (example) | Topics |
|------|---------------|--------|
| Low (4) | S002, S005, S008, S012 | loneliness, work stress, identity |
| Medium (4) | S035, S040, S045, S050 | grief, relationship conflict, academic pressure |
| High (4) | S065, S070, S080, S088 | suicidal ideation, self-harm, substance abuse crisis |

Each participant sees all 12 vignettes. For each vignette, they read **one** response (assigned by Latin-square counterbalancing so each participant sees 4 × A, 4 × B, 4 × C across risk levels).

## 4. Measures

### 4.1 Per-Vignette Items (7-point Likert)

| Construct | Items | Source |
|-----------|-------|--------|
| **Perceived Empathy** | "This response shows understanding of what the person is feeling." | Barrett-Lennard (adapted) |
| **Perceived Warmth** | "This response feels warm and caring." | Fiske et al. warmth scale |
| **Perceived Safety** | "This response avoids giving harmful or dangerous advice." | Custom |
| **Boundary Clarity** | "This response clearly communicates what it can and cannot help with." | Custom |
| **Transparency** | "I understand how this response was generated and reviewed." | Shin (2021) adapted |
| **Trust** | "I trust this system to give me appropriate support." | Mayer et al. (adapted) |
| **Willingness to Rely** | "I would be comfortable relying on this system's advice for this situation." | Lee & See (2004) adapted |
| **Seek Real Help** | "After reading this response, I would be more likely to seek professional help if needed." | Custom |

### 4.2 Post-Study Measures (once, after all vignettes)

- **Overall preference ranking** (A vs B vs C, forced choice)
- **Appropriate reliance** composite (calibration questions):
  - "For which types of situations should a system like this recommend professional help?"
  - "If you were in a crisis, how would you feel about receiving this type of response?"
- **Mental workload** (NASA-TLX single-item): "How mentally demanding was it to evaluate these responses?" (1–7)
- **Open-ended feedback**: "What stood out to you about how the system handled sensitive topics?"

### 4.3 Demographics

- Age, gender, mental health service use (Y/N), prior experience with chatbots

## 5. Hypotheses

| # | Hypothesis | Expected Direction | Rationale (from offline results) |
|---|------------|-------------------|--------------------------------|
| H1 | B and C will increase perceived safety and boundary clarity relative to A | B, C higher | Offline Safety Composite: A = 4.71, B = 4.83, C = 4.87 |
| H2 | C will increase perceived transparency relative to both A and B | C highest | Visible safety indicator + expandable explanation unique to C |
| H3 | A will score higher on perceived warmth and empathy than B and C | A highest (modest) | Offline Empathy Composite: A = 5.00, B/C = 4.76 |
| H4 | The ideal outcome is not maximal trust but **calibrated reliance**: C should show moderate trust combined with highest willingness to seek professional help | C: moderate trust + high seek-help | Visible checking signals system limitations; encourages appropriate skepticism |

### Distinguishing Trust from Appropriate Reliance

A critical conceptual distinction: we do **not** hypothesize that
visible checking will maximize trust. Instead, we predict it will
improve *calibrated trust* — users should feel confident the system
provides reasonable support while recognizing when professional help
is needed. Operationally, appropriate reliance is indexed as:

- **Trust × Risk interaction**: Trust should be higher for low-risk
  scenarios and appropriately moderated for high-risk scenarios.
- **Seek-help intention**: Higher willingness to seek professional help
  when the scenario involves serious risk.
- **Reliance calibration index**: The product of (trust) × (seek-help
  intent) as a function of scenario risk level.

A system with blindly high trust across all risk levels is *worse*
than one where trust is modulated by risk — because the former
encourages over-reliance on AI for crises.

## 6. Analysis Plan

### Primary Analyses
- **Linear mixed-effects model** (LMM) for each Likert measure:
  - Fixed effects: Condition (A/B/C), Risk Level (low/medium/high), Condition × Risk
  - Random effects: Participant (intercept), Vignette (intercept)
- **Pairwise contrasts** with Holm correction (A–B, A–C, B–C)
- **Effect sizes**: Cohen's d from LMM estimates

### Secondary Analyses
- **Appropriate reliance index**: Composite of (trust × seek-help) analyzed as function of risk level
- **Preference ranking**: Friedman test
- **Qualitative coding** of open-ended responses (thematic analysis, two coders, Cohen's κ)

### Sensitivity / Robustness
- Ordinal logistic mixed model as robustness check for Likert data
- Bayesian LMM for evidential value (if standard tests are underpowered)

## 7. Procedure

1. **Informed consent** (approved by IRB/ethics board)
2. **Instructions**: "You will read scenarios where someone is seeking emotional support, along with a response generated by an AI system. Please rate each response."
3. **Practice trial** with a neutral scenario (not analyzed)
4. **12 vignettes** (randomized order within participant)
   - Read scenario → Read AI response → Rate on 8 items
5. **Post-study questionnaire** (demographics + overall preference + open-ended)
6. **Debriefing**: Explanation of study conditions, mental health resources provided

**Estimated duration**: ~20 minutes

## 8. Ethical Considerations

- Participants will read fictional descriptions of distressing scenarios (including suicidal ideation)
- **Exclusion criteria**: Anyone currently in acute mental health crisis (self-reported screening)
- **Trigger warning** at study start
- **Debrief** with crisis helpline numbers (988, Crisis Text Line)
- IRB approval required before data collection
- No deception regarding AI-generated content

## 9. Materials Needed

- [x] Finalize 12 vignette stimuli from benchmark → `results/offline_eval_v2_final/user_study_stimuli.json`
- [x] Generate/freeze A/B/C responses for selected vignettes → included in stimuli JSON
- [x] Build survey instrument (Qualtrics / Gorilla) → `docs/survey_instrument.md`
- [x] Latin-square counterbalancing matrix → `results/offline_eval_v2_final/counterbalancing_matrix.json`
- [x] IRB application → `docs/irb_consent.md`
- [ ] Pilot with 3–5 participants (cognitive interviews)
- [ ] Pre-registration on OSF / AsPredicted

## 10. Power Justification

For a within-subjects design with 12 observations per participant across 3 conditions, assuming correlation r = 0.5 between conditions and medium effect size (d = 0.5):

- Using G*Power: ANOVA (repeated measures), 3 groups, α = .05, β = .20 → N ≈ 28
- We target **N = 36** (12 per counterbalancing cell) to account for attrition (~15%)

## 11. Timeline

| Week | Task |
|------|------|
| 1 | Finalize stimuli, build survey |
| 2 | IRB submission, pilot testing |
| 3–4 | Data collection |
| 5 | Analysis and write-up |


## 12. Pilot Study Plan

Before the main study, conduct a small pilot (N = 3–5 participants)
with cognitive interviews. The pilot has four specific goals:

### Pilot Goal 1: Questionnaire Clarity
- Can participants distinguish between "perceived safety" and
  "boundary clarity" items?
- Are any items ambiguous or redundant?
- Does the 7-point scale produce sufficient variance?

### Pilot Goal 2: Visible Checker Wording
- Does the "✓ Safety reviewed" indicator feel informative or intrusive?
- Is the expandable explanation panel understandable?
- Does it affect reading flow or create confusion?

### Pilot Goal 3: Condition Discriminability
- Can participants perceive meaningful differences between the
  three conditions, or do responses seem interchangeable?
- If conditions are indistinguishable for low-risk scenarios,
  consider using only medium/high-risk items in the main study.

### Pilot Goal 4: Ceiling/Floor Effects
- Check for ceiling effects on trust/safety measures (especially
  for low-risk scenarios where all conditions may seem fine).
- Check for floor effects on warmth for checker conditions.
- If detected, adjust scale anchoring or add reverse-coded items.

### Pilot Procedure
- Semi-structured cognitive interview after the survey.
- Ask participants to think aloud while rating.
- Record item-level completion times to identify confusing items.
- Revise survey based on pilot feedback before main data collection.

