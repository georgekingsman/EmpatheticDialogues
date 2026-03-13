# IRB Submission Package
## Evaluating AI-Generated Supportive Responses for Mental Health Scenarios

> **Document version**: 1.0 (ready to submit)  
> **Date**: 2026-03-13  
> **Instruction**: Fill in every `[INSTITUTION]`, `[PI NAME]`, `[PI EMAIL]`, `[PI PHONE]`, `[SUPERVISOR NAME]` and `[COMPENSATION AMOUNT]` placeholder before uploading to your institution's IRB portal.

---

## Section 1: Study Overview

| Field | Content |
|-------|---------|
| **Protocol Title** | Evaluating AI-Generated Supportive Responses for Mental Health Scenarios |
| **Short Title** | Maker-Checker AI User Study |
| **Principal Investigator** | [PI NAME], [INSTITUTION] |
| **Supervisor / Faculty Sponsor** | [SUPERVISOR NAME], [INSTITUTION] |
| **PI Contact** | [PI EMAIL] · [PI PHONE] |
| **Expected Start Date** | [DATE] |
| **Expected End Date** | [DATE + 6 WEEKS] |
| **Funding Source** | None / Self-funded (delete as appropriate) |
| **Study Site** | Online (Prolific Academic or university participant pool) |

---

## Section 2: Study Purpose and Background

### 2.1 Scientific Background

AI-powered conversational agents are increasingly deployed to provide emotional support and mental health information. While these systems can reduce access barriers, they carry risks: they may fail to recognize crisis-level disclosures, validate harmful coping strategies, or neglect to recommend professional care. Recent proposals for "maker-checker" dual-agent architectures — in which a second AI reviews and optionally annotates first-AI responses — suggest a route toward safer deployment. However, no empirical study has examined how users perceive such architectures in terms of trust, warmth, safety, and calibrated reliance.

### 2.2 Study Purpose

This study asks:

1. Does a maker-checker architecture improve *perceived* safety and boundary clarity compared to a single-agent system?
2. Does making the checking process *visible* to users increase perceived transparency without reducing warmth?
3. Does visible safety checking promote *calibrated* reliance — appropriate trust that rises with risk severity — rather than unconditional trust or distrust?

### 2.3 Significance

Findings will directly inform the design of safer AI mental-health support tools and contribute a vignette-based evaluation methodology reusable by the research community. All materials will be publicly released.

---

## Section 3: Participant Population

| Field | Content |
|-------|---------|
| **Intended N** | Minimum 36; target 45–60; upper cap 72 |
| **Minimum age** | 18 years |
| **Language requirement** | Fluent in English (self-reported) |
| **Location** | No restriction (online study) |
| **Recruitment platform** | Prolific Academic or equivalent university participant pool |

### 3.1 Inclusion Criteria

- Age 18 or older
- Self-reported fluency in English
- Access to a computer or mobile device with internet

### 3.2 Exclusion Criteria (Screening, Pre-Participation)

- Currently experiencing a mental health crisis (self-reported at screening)
- Age under 18

### 3.3 Exclusion Criteria (Post-Hoc, Analysis)

The following participants will be excluded from analysis after data collection (and replaced to maintain target N per counterbalancing cell):

1. Failed the embedded attention-check item
2. Completed the full study in under 5 minutes (insufficient engagement)
3. Provided identical responses (zero variance) across all 12 vignette rating items
4. Withdrew before completing 50% of vignettes

---

## Section 4: Study Procedures

### 4.1 Overview

This is a **single-session, online, survey-based study**. All data are collected through [Qualtrics / Gorilla]. Estimated duration: **20 minutes**.

### 4.2 Step-by-Step Protocol

| Step | Action | Duration |
|------|--------|----------|
| 1 | Participant clicks study link (Prolific or email invite) | — |
| 2 | Eligibility screening (3 questions) | ~1 min |
| 3 | Trigger warning and informed consent | ~2 min |
| 4 | Study instructions + practice trial (1 neutral scenario) | ~2 min |
| 5 | 12 experimental vignettes (counterbalanced condition assignment) | ~13 min |
| 6 | Post-study questionnaire (demographics + open-ended) | ~2 min |
| 7 | Debrief and crisis resources | — |
| 8 | Completion code / redirect to Prolific | — |

### 4.3 Counterbalancing

Participants are randomly assigned to one of 12 Latin-square counterbalancing cells. Each cell specifies, for each of the 12 vignettes, which of the three conditions (A, B, C) the participant will see. Every participant sees exactly 4 vignettes per condition; condition assignment is balanced across risk levels (low, medium, high).

### 4.4 Condition Details

| Condition | Label | What the Participant Sees |
|-----------|-------|--------------------------|
| A | Single Agent | Empathetic AI response only |
| B | Hidden Checker | Same empathetic response (safety-reviewed internally); no indicator shown |
| C | Visible Checker | Same empathetic response, plus: "✓ This response has been reviewed for safety by a second AI system." |

Participants are **not informed** of the condition structure during the study. The debrief form reveals all three conditions after completion.

---

## Section 5: Stimuli

- **12 fictional vignettes** depicting people seeking emotional support
- Risk levels: 4 Low (work stress, loneliness, academic pressure, relationship), 4 Medium (eating disorder, substance use, depression, anxiety), 4 High (self-harm, suicidal ideation, trauma, domestic violence)
- All vignettes are **fictional**; no real person's disclosure is used
- High-risk vignettes are appropriate for a vignette study of this type and are necessary to test whether safety checking affects user perception specifically at elevated risk levels
- A **trigger warning** is shown before consent; participants may opt out at any time

---

## Section 6: Measures

### 6.1 Per-Vignette Ratings (8 items, 7-point Likert: 1 = Strongly Disagree, 7 = Strongly Agree)

| Item | Wording |
|------|---------|
| Q1: Perceived Empathy | "This response shows understanding of what the person is feeling." |
| Q2: Perceived Warmth | "This response feels warm and caring." |
| Q3: Perceived Safety | "This response avoids giving harmful or dangerous advice." |
| Q4: Boundary Clarity | "This response clearly communicates what the system can and cannot help with." |
| Q5: Transparency | "I understand how this response was generated and reviewed." |
| Q6: Trust | "I trust this system to give me appropriate support." |
| Q7: Willingness to Rely | "I would be comfortable relying on this system's advice for this situation." |
| Q8: Seek Real Help | "After reading this response, I would be more likely to seek professional help if needed." |

### 6.2 Embedded Attention Check (1 item, placed after Vignette 6)

"This is an attention check. Please select 'Moderately Agree' (5) for this item."  
→ Correct response = 5; participants selecting any other value are flagged for exclusion.

### 6.3 Post-Study Questionnaire

- Overall satisfaction with the AI responses encountered (1–7)
- Crisis comfort: "How comfortable would you be using a system like this if you were personally in crisis?" (1–7)
- Mental workload (NASA-TLX adapted, single item, 1–7)
- Referral accuracy checklist: "For which of the following situations should an AI system recommend professional help?" (tick-box, 6 items)
- Demographics: age group, gender (optional), prior experience with chatbots (yes/no), prior use of mental health services (yes/no/prefer not to say)
- Open-ended: "What stood out to you about how the AI handled sensitive topics?" (free text, optional)

---

## Section 7: Risk Assessment

### 7.1 Study Classification

This study qualifies as **minimal-risk** under standard IRB definitions:

- No medical procedures, physiological measurements, or deception about physical risk
- No collection of clinical mental health data or personally identifiable information
- Vignette-based exposure to distressing content is comparable to reading a newspaper article about crisis situations

### 7.2 Risk: Exposure to Distressing Content

**Description**: Participants will read fictional scenarios depicting suicidal ideation, self-harm, and domestic violence.  
**Magnitude**: Minimal to mild; comparable to other published vignette studies using similar stimuli (e.g., Althoff et al., 2016; Laranjo et al., 2018).  
**Mitigation**:
1. Eligibility screening excludes participants currently in acute crisis
2. Explicit trigger warning before consent
3. Right to withdraw at any time without penalty
4. Comprehensive crisis resources provided at end of study and in debrief

### 7.3 Risk: Disclosure of Sensitive Information

**Description**: Open-ended items could elicit personal disclosures about mental health history.  
**Mitigation**: Items are phrased to elicit *impressions of the AI system*, not personal disclosures. Responses are anonymous. No therapist-participant relationship is created.

### 7.4 Risk: Privacy / Data Security

**Description**: Participant Prolific IDs are temporarily linked to responses for compensation.  
**Mitigation**: IDs will be permanently deleted from research records within 7 days of payment confirmation. Published data will contain no individual identifiers.

---

## Section 8: Benefits

- No direct personal benefit to participants
- Indirect benefit: contributing to research on safer AI mental health support
- Societal benefit: findings will inform design of AI systems used in mental health contexts, potentially reducing risk of harm in deployed systems

---

## Section 9: Compensation

| Field | Value |
|-------|-------|
| **Amount** | [COMPENSATION AMOUNT] (e.g., £3.00 / $3.75 for ~20 min = ≥ £9/hr) |
| **Distribution method** | Prolific payment system / university participant payment system |
| **Timing** | Within 48 hours of approved submission |
| **Partial completion** | Participants who withdraw before completing 50% of vignettes will not be compensated (this is stated in consent) |

---

## Section 10: Confidentiality and Data Management

| Field | Content |
|-------|---------|
| **Identified data collected** | Prolific ID only (for payment); deleted after payment |
| **Anonymous data collected** | Per-vignette ratings, post-study questionnaire, open-ended responses |
| **Linkage** | Cell assignment (1–12) recorded; no name, email, or IP collected |
| **Storage location** | Qualtrics (encrypted); exported to PI's secure institutional storage |
| **Access** | PI and supervising faculty only |
| **Retention period** | 5 years post-publication, then deleted |
| **Publication** | De-identified, aggregated data only; open-access dataset will be deposited after publication |

---

## Section 11: Informed Consent Process

- Consent is obtained electronically at the start of the Qualtrics survey
- Participants must click "I agree to participate" to proceed (forced-choice item)
- Participants who select "I do not agree" are redirected to a thank-you page and not enrolled
- Full consent text is available to participants throughout the study via a link at the bottom of each page

---

## Section 12: Voluntary Participation and Withdrawal

- Participation is entirely voluntary
- Participants may withdraw at any time by closing the browser window
- Partial data (from participants who complete < 50% of vignettes) will not be included in analysis
- Participants may request deletion of their submitted data within 14 days of submission by emailing [PI EMAIL] with their Prolific ID

---

## Section 13: Deception

**No deception is used in this study.**

Participants are not told which of the three conditions each vignette represents during the study (as revealing the conditions would affect ratings). However, this constitutes **withholding information** rather than active deception, and is standard practice for within-subjects vignette studies. Full information about all three conditions is provided in the debrief.

---

## Section 14: Vulnerable Populations

This study does **not** intentionally recruit individuals from vulnerable populations. Participants who are currently in mental health crisis are explicitly screened out at the eligibility stage. No clinical population is targeted.

---

## Section 15: Pilot Study

Before full data collection, a **pilot study** will be conducted with N = 6–12 participants (using the same protocol). Pilot data will:
- Not be included in the main statistical analysis
- Be used only to evaluate survey flow, comprehension, completion time, and item variance
- Results will be reviewed with the supervising faculty before proceeding to full collection

---

## Section 16: Pre-Registration

The study hypotheses, primary outcomes, exclusion criteria, and analysis plan are pre-registered on the Open Science Framework (OSF) prior to IRB approval and before any data collection. Pre-registration document: `docs/preregistration.md`.

---

## Section 17: Attached Documents

The following documents are attached to this submission:

| Document | File |
|----------|------|
| Informed consent form | `docs/irb_consent.md` |
| Debriefing form | (last section of `docs/irb_consent.md`) |
| Survey instrument | `docs/survey_instrument.md` |
| Pre-registration | `docs/preregistration.md` |
| Vignette stimuli list | `results/offline_eval_v2_final/user_study_stimuli.json` |
| Counterbalancing matrix | `results/offline_eval_v2_final/counterbalancing_matrix.json` |

---

*End of IRB Submission Package*
