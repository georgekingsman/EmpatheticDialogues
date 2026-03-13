# A Double-AI Maker-Checker Architecture for Safer Empathetic Mental-Health Support

This repository contains the code, data, and evaluation pipeline for a study
investigating whether **role separation** between empathetic response generation
and safety checking can improve mental-health peer-support AI — and what
trade-off this creates with perceived warmth and empathy.

---

## Problem Statement

Single-agent LLM designs overload one model with competing objectives: be warm,
be safe, detect risk, maintain boundaries, and communicate uncertainty. We show
that this **role conflict** leads to missed escalation in 30% of high-risk
scenarios. Our **maker-checker architecture** separates empathetic generation
(Maker) from safety review (Checker), addressing this conflict structurally.

## Three Conditions

| Condition | Description |
|-----------|-------------|
| **A: Single Agent** | One-shot empathetic response, no checker |
| **B: Hidden Checker** | Maker drafts → Checker reviews → user sees only final response |
| **C: Visible Checker** | Same as B, but user sees a safety-review indicator + expandable explanation |

## Benchmark

90 peer-support scenarios stratified by risk level (30 low / 30 medium / 30 high),
covering topics from work stress and loneliness to self-harm, suicidal ideation,
and substance abuse. Each scenario × 3 conditions = 270 evaluated outputs.

## Evaluation Protocol

- **6 dimensions**: Emotion Recognition, Validation & Warmth, Helpfulness, Safety, Boundary Adherence, Escalation Appropriateness (each 1–5)
- **2 composite indices**: Empathy Composite (Emotion + Validation), Safety Composite (Safety + Boundary + Escalation)
- **LLM-as-a-judge** with structured rubric (DeepSeek-Chat)
- **Robustness**: Stricter second-judge + 3-persona multi-rater cross-validation

## Key Findings

| Composite | A: Single Agent | B: Hidden Checker | C: Visible Checker |
|-----------|:-:|:-:|:-:|
| Empathy | **5.00** | 4.76 | 4.76 |
| Safety | 4.71 | 4.83 | **4.87** |
| Helpfulness | 3.83 | **4.03** | **4.03** |

- **A > B, C on Empathy** (p < .005, Holm-corrected) — single agent is warmer
- **C > A on Safety** — visible checker is safest, especially in high-risk
- **Checker = risk-sensitive safety net**: 100% approve on low-risk, 63–70% escalate on high-risk, 1.7% false positive rate
- **Trade-off robust across 5 judge variants** (original, strict, strict/moderate/lenient personas)

## Planned User Study

A vignette-based study (N ≈ 36) measuring perceived empathy, warmth, safety,
trust, transparency, and calibrated reliance. Central hypothesis: visible
checking promotes **appropriate reliance** (moderate trust + high seek-help
intention) rather than blind trust. See `docs/user_study_design.md`.

---

## Repository Structure

```
├── generation/                  # Response generation (3 conditions)
│   ├── run_single.py            # Condition A: Single Agent
│   ├── run_double_hidden.py     # Condition B: Hidden Checker
│   └── run_double_visible.py    # Condition C: Visible Checker
├── prompts/
│   ├── maker/                   # Maker agent system prompt
│   ├── checker/                 # Checker agent system prompt
│   └── visible_checker/         # Visible indicator templates
├── checker/                     # Checker policy layer
│   ├── policy_rules.py          # approve/revise/abstain/escalate logic
│   └── checker_schema.py        # Structured output schema
├── data/
│   └── scenarios/benchmark.jsonl  # 90-scenario benchmark
├── results/
│   ├── offline_eval_v2_final/   # ★ Frozen evaluation results
│   │   ├── scenarios.csv        # Benchmark scenarios
│   │   ├── outputs_A.jsonl      # Single agent outputs
│   │   ├── outputs_B_hidden.jsonl  # Hidden checker outputs
│   │   ├── outputs_C_visible.jsonl # Visible checker outputs
│   │   ├── judge_scores_main.csv   # Primary judge (270 rows)
│   │   ├── judge_scores_second.csv # Stricter second-judge (90 rows)
│   │   ├── multi_rater_scores.csv  # 3-persona rater (270 rows)
│   │   ├── checker_actions.csv     # Checker decisions (180 rows)
│   │   ├── statistics.json         # Full statistical analysis
│   │   ├── composite_stats.json    # Composite index values
│   │   ├── figures/                # Paper figures (PDF + PNG)
│   │   ├── tables/                 # LaTeX tables
│   │   └── metadata.yaml          # Frozen metadata
│   ├── generate_paper_assets.py    # Regenerate figures + tables
│   ├── run_statistics.py           # Statistical tests
│   ├── run_second_judge.py         # Second-judge cross-validation
│   └── run_multi_rater.py          # Multi-rater simulation
├── docs/
│   ├── paper_results.md            # Full paper draft (Sections 1–6 + Abstract)
│   ├── paper_outline.md            # Status tracker
│   ├── user_study_design.md        # User study protocol
│   ├── appendix_qualitative.md     # 6 qualitative case examples
│   └── appendix_materials.md       # Prompts, rubrics, schema
├── src/                            # Shared utilities
│   ├── eval/llm_judge.py          # LLM judge pipeline
│   └── data/build_dataset.py      # Data loading
└── evaluation/
    └── offline_metrics.py          # Metric computation
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set API key
export DEEPSEEK_API_KEY=sk-...

# 3. Generate responses (3 conditions × 90 scenarios)
python generation/run_single.py
python generation/run_double_hidden.py
python generation/run_double_visible.py

# 4. Run LLM judge
python results/run_statistics.py

# 5. Generate paper figures and tables
python results/generate_paper_assets.py
```

## Reproducibility

All results can be reproduced from the frozen `results/offline_eval_v2_final/`
directory. The `metadata.yaml` file records model versions, prompts, commit hash,
and generation parameters. See `results/offline_eval_v2_final/README.md` for
a complete file manifest.

## Legacy Components

This repository also contains earlier work on empathetic dialogue generation
(GPT-2 fine-tuning, Chain-of-Empathy) which served as *baseline development*
for the current maker-checker study. These components are in `src/models/`,
`Model_Baseline.py`, `Train_Baseline.py`, etc. — they are not part of the
current paper but remain available for reference.
