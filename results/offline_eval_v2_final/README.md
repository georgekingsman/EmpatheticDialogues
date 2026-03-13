# Offline Evaluation v2 Final — Frozen Results

> **Frozen: 2026-03-13 | Commit: ec33ce0**
> This directory contains all artifacts for the offline evaluation reported in the paper.
> **Do not modify these files.** Any re-analysis should read from this directory without writing back.

## Directory Contents

| File | Description | Rows |
|------|-------------|------|
| `scenarios.csv` | 90 benchmark scenarios (30 low / 30 medium / 30 high risk) | 90 |
| `outputs_A.jsonl` | Condition A (Single Agent) responses | 90 |
| `outputs_B_hidden.jsonl` | Condition B (Hidden Checker) responses | 90 |
| `outputs_C_visible.jsonl` | Condition C (Visible Checker) responses | 90 |
| `judge_scores_main.csv` | Primary LLM judge scores (6 dimensions) | 270 |
| `judge_scores_second.csv` | Stricter second-judge scores (emotion, validation, safety) | 90 |
| `multi_rater_scores.csv` | Three-persona rater scores (strict/moderate/lenient) | 270 |
| `multi_rater_report.json` | Aggregated multi-rater analysis + Krippendorff's α | 1 |
| `checker_actions.csv` | Checker decisions and dimension scores for B/C | 180 |
| `statistics.json` | Full statistical analysis (Wilcoxon, effect sizes, bootstrap CIs) | 1 |
| `ceiling_audit.json` | Ceiling effect rates per dimension per condition | 1 |
| `error_analysis.json` | Qualitative error taxonomy (4 types, 45 instances) | 1 |
| `metadata.yaml` | Frozen experimental parameters, model versions, prompts | 1 |
| `figures/` | 4 paper figures (PDF + PNG) | 8 |

## Key Results Summary

### Main Finding
The maker-checker architecture improves safety-critical behavior at a modest cost in emotional expressiveness.

### Condition Means (Overall, N=90)

| Dimension | A: Single Agent | B: Hidden Checker | C: Visible Checker |
|-----------|:-:|:-:|:-:|
| Emotion | **5.00** | 4.74 | 4.73 |
| Validation | **5.00** | 4.77 | 4.78 |
| Helpfulness | 3.83 | **4.03** | **4.03** |
| Safety | 4.88 | 4.92 | **4.96** |
| Boundary | 4.89 | 4.93 | **4.98** |
| Escalation | 4.36 | 4.63 | **4.68** |

### Composite Indices

| Composite | A | B | C |
|-----------|:-:|:-:|:-:|
| Empathy (emotion + validation) | **5.00** | 4.76 | 4.76 |
| Safety (safety + boundary + escalation) | 4.71 | 4.83 | **4.87** |

### Significance (Holm-corrected)
- A > B on Emotion (p=.001) and Validation (p=.004)
- A > C on Emotion (p<.001) and Validation (p=.003)
- B ≈ C on all dimensions (all p=1.00)

### Cross-Validation
- Second-judge eliminated emotion ceiling (0% at 5 vs 83.3% original)
- All 3 multi-rater personas confirmed A > B, A > C on emotion
- Safety ceiling robust across all evaluation variants

## Figures

1. **fig1_overall_6dim** — 6-dimension grouped bar chart (3 conditions)
2. **fig2_highrisk_focus** — High-risk subset: Safety/Boundary/Escalation/Helpfulness
3. **fig3_checker_decisions** — Checker decision distribution by risk level
4. **fig4_tradeoff** — Empathy composite vs Safety composite scatter
