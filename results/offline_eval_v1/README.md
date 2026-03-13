# Offline Evaluation v1

**Frozen:** 2026-03-12  
**Commit:** `ec33ce0`

## Overview

This directory contains the frozen results of the first offline evaluation
comparing three conditions on 90 peer-support scenarios (30 low / 30 medium / 30 high risk):

| Condition | Code | Description |
|-----------|------|-------------|
| A — Single Agent | `single_agent` | One-shot empathetic response |
| B — Double Hidden | `double_hidden` | Maker-Checker, user sees final response only |
| C — Double Visible | `double_visible` | Maker-Checker, user sees safety review indicator |

## Files

| File | Description |
|------|-------------|
| `config.yaml` | Generation / judge / checker parameters and commit hash |
| `scenarios.csv` | 90 benchmark scenarios with risk level and metadata |
| `outputs_A.jsonl` | Single-agent generation outputs |
| `outputs_B.jsonl` | Double-hidden generation outputs (includes checker fields) |
| `outputs_C.jsonl` | Double-visible generation outputs (includes visible indicator) |
| `judge_scores.csv` | Per-sample 6-dimension judge scores for all 270 samples |
| `checker_actions.csv` | Checker decisions (approve/revise/escalate) for B and C |

## Judge Dimensions (1–5 scale)

1. **Emotion** — Emotion Recognition
2. **Validation** — Validation & Warmth
3. **Helpfulness** — Helpfulness & Actionability
4. **Safety** — Safety & Boundaries
5. **Boundary Adherence** — Adherence to response guidelines
6. **Escalation** — Appropriateness of escalation decisions

## Reproduction

```bash
# Generate outputs
python generation/run_single.py
python generation/run_double_hidden.py
python generation/run_double_visible.py

# Judge
python evaluation/offline_metrics.py \
  --single outputs/generations/single_agent.jsonl \
  --hidden outputs/generations/double_hidden.jsonl \
  --visible outputs/generations/double_visible.jsonl
```
