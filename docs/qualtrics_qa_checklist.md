# Qualtrics Full-Chain QA Checklist
## Before Launching the Maker-Checker User Study

> **Purpose**: Run through every item before the pilot. Repeat after any survey edit.  
> **How to use**: Self-test first (you complete the full survey), then ask 2–3 colleagues to test. Mark each item ✅ / ❌ / N/A.

---

## Part 1: Survey Setup (Admin Panel)

| # | Check | Pass? | Notes |
|---|-------|-------|-------|
| 1.1 | Survey is set to **Anonymous** (no IP, name, or email collected) | | |
| 1.2 | Response limit per survey link is **off** or set to upper cap (72) | | |
| 1.3 | Survey **language** is set to English | | |
| 1.4 | **Prevent ballot-box stuffing** (one response per session) is enabled | | |
| 1.5 | **Save & Return Later** is **disabled** (prevent partial revisits distorting counterbalance) | | |
| 1.6 | Survey URL parameter `cell_id` is correctly configured in Embedded Data | | |
| 1.7 | Prolific completion URL / redirect is set | | |
| 1.8 | Thank-you / end-of-survey page displays crisis resources | | |

---

## Part 2: Counterbalancing Logic

| # | Check | Pass? | Notes |
|---|-------|-------|-------|
| 2.1 | All 12 cell IDs (1–12) can be reached via URL parameter `?cell_id=X` | | |
| 2.2 | For cell 1: each of the 12 vignettes shows the correct condition (A/B/C) per the matrix | | |
| 2.3 | For cell 2: same check — a different distribution of A/B/C | | |
| 2.4 | Each cell shows **exactly 4 A, 4 B, 4 C** vignettes | | |
| 2.5 | Condition C vignettes display the safety indicator ("✓ This response has been reviewed…") | | |
| 2.6 | Condition B vignettes do **not** display any safety indicator | | |
| 2.7 | Condition A vignettes do **not** display any safety indicator | | |
| 2.8 | Vignette order is **randomised** within the 12-item block (not fixed order) | | |
| 2.9 | Practice trial (Vignette 0 / neutral scenario) appears **before** the 12 experimental vignettes | | |
| 2.10 | Practice trial is **not** assigned a condition label / indicator | | |

---

## Part 3: Stimulus Display

| # | Check | Pass? | Notes |
|---|-------|-------|-------|
| 3.1 | Scenario text (person seeking support) displays correctly for all 12 vignettes | | |
| 3.2 | AI response text displays correctly for all 12 vignettes × 3 conditions = 36 combinations | | |
| 3.3 | High-risk vignettes (S063, S076, S081, S065) display without truncation on desktop | | |
| 3.4 | High-risk vignettes display without truncation on **mobile** | | |
| 3.5 | Safety indicator for Condition C is visually distinct (e.g., ✓ icon + text) | | |
| 3.6 | No HTML encoding errors (e.g., `&quot;`, `&amp;` appearing as raw text) | | |
| 3.7 | Response text does not reveal condition identity (i.e., conditions B and C use the same text body) | | |

---

## Part 4: Rating Scales

| # | Check | Pass? | Notes |
|---|-------|-------|-------|
| 4.1 | All 8 Likert items (Q1–Q8) appear after every vignette | | |
| 4.2 | Scale anchors are shown: **1 = Strongly Disagree, 7 = Strongly Agree** | | |
| 4.3 | All scale items are **forced response** (participant cannot proceed without answering) | | |
| 4.4 | Scale labels display correctly on mobile (not cut off) | | |
| 4.5 | No items appear as blank / missing on first load | | |
| 4.6 | Validate a complete response exports 8 columns per vignette × 12 vignettes = 96 rating columns | | |

---

## Part 5: Attention Check

| # | Check | Pass? | Notes |
|---|-------|-------|-------|
| 5.1 | Attention check item is embedded at **Vignette 6** (midpoint), not at start/end | | |
| 5.2 | Item wording: "This is an attention check. Please select **Moderately Agree (5)** for this item." | | |
| 5.3 | Correct response is option **5** | | |
| 5.4 | Attention check response is **exported** as a named column (`attention_check`) | | |
| 5.5 | **No branching or rejection** based on attention check during survey (flag post-hoc; do not interrupt flow) | | |

---

## Part 6: Eligibility Screening and Consent

| # | Check | Pass? | Notes |
|---|-------|-------|-------|
| 6.1 | Age < 18 → immediate screen-out page (not just skip) | | |
| 6.2 | Not fluent in English → screen-out page | | |
| 6.3 | Currently in crisis → screen-out page **with crisis resources** prominently shown | | |
| 6.4 | Trigger warning page appears before consent (separate page, requires explicit click to continue) | | |
| 6.5 | Consent "I do not agree" → graceful exit page (not error) | | |
| 6.6 | Consent "I agree" → proceeds to instructions | | |

---

## Part 7: Post-Study Questionnaire

| # | Check | Pass? | Notes |
|---|-------|-------|-------|
| 7.1 | Overall satisfaction item (1–7) appears after all 12 vignettes | | |
| 7.2 | Crisis comfort item (1–7) appears | | |
| 7.3 | Mental workload item (NASA-TLX adapted, 1–7) appears | | |
| 7.4 | Referral accuracy checklist (6 tick-box items) appears | | |
| 7.5 | Demographics block (age group, gender, chatbot experience, MH service use) appears | | |
| 7.6 | Open-ended feedback field is **optional** (not forced response) | | |
| 7.7 | Demographic items (especially gender) include **prefer not to say** option | | |

---

## Part 8: Data Export Verification

Run a test response through the full survey, then export the raw CSV and verify:

| # | Check | Pass? | Notes |
|---|-------|-------|-------|
| 8.1 | Export contains `participant_id` (Prolific ID or equivalent) | | |
| 8.2 | Export contains `cell_id` (1–12) | | |
| 8.3 | Export contains `vignette_id` for each of the 12 vignettes | | |
| 8.4 | Export contains `condition` (A/B/C) for each vignette | | |
| 8.5 | Export contains `risk_level` (low/medium/high) for each vignette | | |
| 8.6 | Export contains Q1–Q8 for each vignette | | |
| 8.7 | Export contains `attention_check_passed` (1/0) | | |
| 8.8 | Export contains `completion_time_s` (or equivalent duration in seconds) | | |
| 8.9 | Column names match exactly the schema expected by `results/analyse_user_study.py` | | |
| 8.10 | No identifying information (name, email, IP) appears in export | | |

### Expected column schema (verify against analysis script)

```
participant_id, cell_id, vignette_id, condition, risk_level,
Q1_empathy, Q2_warmth, Q3_safety, Q4_boundary, Q5_transparency,
Q6_trust, Q7_rely, Q8_seekhelp,
attention_check_passed, completion_time_s
```

Post-study file columns:
```
participant_id, overall_satisfaction, crisis_comfort,
mental_workload, referral_correct_count,
age_group, gender, mh_service_use, chatbot_experience,
open_feedback_safety, open_feedback_improve
```

---

## Part 9: Timing and Completion

| # | Check | Pass? | Notes |
|---|-------|-------|-------|
| 9.1 | Your own completion time is **15–25 minutes** (outside that range → investigate) | | |
| 9.2 | Colleagues' average is within the same range | | |
| 9.3 | No page takes > 3 minutes for a typical reader (if so, that page may need simplification) | | |
| 9.4 | Compensation amount is proportionate to actual completion time (≥ £9/hr) | | |

---

## Part 10: Mobile & Cross-Browser

| # | Check | Pass? | Notes |
|---|-------|-------|-------|
| 10.1 | Survey renders correctly on **Chrome** (desktop) | | |
| 10.2 | Survey renders correctly on **Safari** (desktop) | | |
| 10.3 | Survey renders correctly on **Chrome** (Android mobile) | | |
| 10.4 | Survey renders correctly on **Safari** (iPhone) | | |
| 10.5 | Likert scale radio buttons are tappable on a phone screen (not too small) | | |
| 10.6 | Long scenario text is scrollable on mobile (not cut off behind fixed UI elements) | | |

---

## Part 11: End-to-End Pilot Debrief

After at least 3 test participants complete the survey, review:

| # | Check | Pass? | Notes |
|---|-------|-------|-------|
| 11.1 | All 12 counterbalancing cells have been activated at least once | | |
| 11.2 | No single condition (A/B/C) was always assigned to the same vignette in all test runs | | |
| 11.3 | At least one Condition C response showed the safety indicator | | |
| 11.4 | Testers could distinguish Condition A from C (ask verbally after their session) | | |
| 11.5 | Open-ended responses contain no sign of confusion about task instructions | | |
| 11.6 | No tester reported a broken page, missing text, or error message | | |
| 11.7 | Post-pilot: decide whether any **wording** changes are needed (acceptable) | | |
| 11.8 | Post-pilot: confirm **no hypothesis or measure changes** will be made | | |
| 11.9 | Survey is **frozen** (locked in Qualtrics) before main collection begins | | |

---

## Sign-off

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Principal Investigator | | | |
| Supervising Faculty | | | |

*Once both sign-offs are complete and all critical items above are marked ✅, proceed to main data collection.*
