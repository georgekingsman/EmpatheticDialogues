# User Study Protocol

## Overview

| Parameter | Value |
|-----------|-------|
| Design | Between-subjects, 3 conditions |
| Conditions | A (Single Agent), B (Hidden Checker), C (Visible Checker) |
| Participants per condition | Target: 20–30 (total 60–90) |
| Duration per participant | ~25 minutes |
| Scenarios per participant | 5 (1 low, 2 medium, 2 high risk) |
| Ethics approval | Required before data collection |

---

## Condition Assignments

### Condition A — Single Agent
- Participant sees: AI response only
- No mention of safety review

### Condition B — Double AI, Hidden Checker
- Participant sees: Final (possibly revised) AI response
- No mention of safety review (checker operates silently)

### Condition C — Double AI, Visible Checker
- Participant sees: Final AI response + safety review indicator
- Expandable safety review explanation available

---

## Procedure

### Step 1: Screening & Consent (3 min)
1. Participant reads consent form (`consent.md`)
2. Confirms eligibility (age ≥ 18, English-fluent)
3. Provides digital consent
4. Completes demographic questions

### Step 2: System Introduction (2 min)
- **Condition A**: "You will interact with an AI support assistant."
- **Condition B**: "You will interact with an AI support assistant."
  (Same wording — checker is hidden from participant.)
- **Condition C**: "You will interact with an AI support assistant that
  includes a built-in safety review process. After the system drafts a
  response, a separate safety module reviews it. You will see a brief
  indicator of this review."

### Step 3: Interaction Phase (10 min)
1. Present 5 scenarios in fixed order (counterbalanced across participants):
   - Scenario set: 1 low-risk, 2 medium-risk, 2 high-risk
   - Select from `data/scenarios/benchmark.jsonl`
2. For each scenario:
   a. Show user utterance: "Imagine someone said this to the AI system:"
   b. Show system response (condition-specific output)
   c. For Condition C: also show safety review indicator
   d. Participant reads and moves to next scenario
3. No free-form interaction (controlled stimuli for reproducibility)

### Step 4: Post-Interaction Questionnaire (8 min)
- Administer all scales from `survey_items.md`
- Condition A skips Transparency (TP) items
- All conditions complete all other scales

### Step 5: Open-Ended Questions (3 min, optional)
- 4 questions for A/B, 5 questions for C
- Text input boxes, no minimum length

### Step 6: Debrief (2 min)
- Reveal study design and all three conditions
- Explain maker-checker architecture
- Check for distress, provide real crisis resources
- Thank participant

---

## Scenario Selection for User Study

From the 90 benchmark scenarios, select 15 for the user study (3 sets of 5,
counterbalanced):

| Set | Low | Medium | High |
|-----|-----|--------|------|
| Set 1 | S005 | S031, S035 | S061, S065 |
| Set 2 | S016 | S040, S043 | S066, S073 |
| Set 3 | S025 | S048, S054 | S076, S083 |

Each participant sees one set. Sets are balanced for topic diversity and
risk distribution.

---

## Analysis Plan

### Quantitative
1. Compute scale means and Cronbach's α for each scale
2. One-way ANOVA (or Kruskal-Wallis if non-normal) across 3 conditions
3. Post-hoc pairwise comparisons with Bonferroni correction
4. Effect sizes: Cohen's d (pairwise) and η² (overall)
5. Key contrasts:
   - A vs B: Does hidden checking change perceived quality?
   - A vs C: Does visible checking affect trust/warmth?
   - B vs C: Is visibility itself the active ingredient?

### Qualitative
1. Open coding of free-text responses
2. Thematic analysis (Braun & Clarke)
3. Compare emergent themes across conditions

### Reporting
- Table: condition means (SD) for all scales
- Figure: radar chart of all DV means by condition
- Figure: trust × warmth scatter by condition
- Supplementary: full item-level descriptive statistics

---

## Ethical Considerations

- **No real mental-health interaction**: participants evaluate pre-generated
  responses to hypothetical scenarios
- **Sensitive content warning**: consent form discloses emotional distress
  topics
- **Withdrawal**: participant can exit at any time
- **Debrief**: real crisis resources provided after study
- **Data minimisation**: only anonymous demographics collected
- **IRB/Ethics approval**: must be obtained before any data collection
