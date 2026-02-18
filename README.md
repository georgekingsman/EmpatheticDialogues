# Empathetic Dialogue Evaluation Framework

**Direction B: Human Rating ↔ LLM-as-a-Judge Calibration**

A reproducible, extensible framework for evaluating empathetic/supportive dialogue using human annotations and LLM-as-a-judge, with statistical calibration between the two.

---

## Research Contributions

1. **Reproducible evaluation rubric** — 4-dimension rubric (Emotion Recognition, Validation & Warmth, Helpfulness, Safety & Boundaries) with anchor descriptions and annotation protocol
2. **LLM-as-a-judge pipeline** — structured JSON output, multi-sample stability analysis, supports any OpenAI-compatible API
3. **Calibration framework** — Isotonic regression (Route 1), ordinal logistic regression (Route 2), with diagnostic metrics (MAE, RMSE, rank correlation, ECE)
4. *(Optional)* **Active sampling** — uncertainty-driven annotation selection to minimize labeling cost

---

## Project Structure

```
EmpatheticDialogues/
  src/
    data/
      build_dataset.py        # Dataset loading, splitting, label masking
      templates.py             # Prompt/response templates
    models/
      baseline_gpt2.py         # GPT-2 baseline (control)
      empathy_chain.py         # Chain-of-Empathy model (ablation)
      train.py                 # Unified training script
    inference/
      generate.py              # Unified generation interface (JSONL output)
    eval/
      rubric.py                # Rubric definitions (single source of truth)
      human_labels_schema.py   # Annotation schema, validation, IAA
      llm_judge.py             # LLM-as-a-judge pipeline
      calibrate.py             # Calibration (isotonic / ordinal / IRT)
      metrics.py               # NLG metrics, judge reliability, active sampling
  experiments/
    run_train.sh               # Train all models
    run_generate.sh            # Generate responses
    run_judge.sh               # Run LLM judge
    run_calibrate.sh           # Calibrate and report
  outputs/
    generations/*.jsonl        # Model outputs
    labels/human/*.csv         # Human annotations
    judge/*.jsonl              # Judge scores
    calibrated/*.jsonl         # Calibrated scores
  docs/
    rubric_v1.md               # Full rubric with examples
    annotation_guide_v1.md     # Annotator instructions
  data/
    formatted_Psych_data.jsonl # Training data (5319 samples)
    Psych_data.csv             # Raw data
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train models
```bash
bash experiments/run_train.sh
```

### 3. Generate responses
```bash
bash experiments/run_generate.sh
```

### 4. Run LLM judge
```bash
OPENAI_API_KEY=sk-... bash experiments/run_judge.sh
# or
DEEPSEEK_API_KEY=sk-... bash experiments/run_judge.sh --backend deepseek --judge_model deepseek-chat
```

### 5. Calibrate
```bash
bash experiments/run_calibrate.sh
```

---

## Data Format

### Generation output (JSONL)
```json
{"id": "abc123", "prompt": "<user>: ...\n<assistant>:", "response": "...", "model": "gpt2", "seed": 42, "temperature": 0.7, "top_p": 0.9, "ts": "..."}
```

### Judge output (JSONL)
```json
{"sample_id": "abc123", "model": "gpt2", "repeat_idx": 0, "scores": {"emotion": 4, "validation": 3, "helpfulness": 4, "safety": 5}, "overall": 4, "confidence": 0.78, "notes": "..."}
```

### Human labels (CSV)
```
sample_id,annotator_id,emotion,validation,helpfulness,safety,overall,notes
abc123,A1,4,3,4,5,4,"good emotion recognition"
```

---

## Experiment Matrix

| Model | Type | Purpose |
|-------|------|---------|
| GPT-2 (vanilla) | Baseline | Lower bound |
| GPT-2 (fine-tuned) | Baseline | Standard fine-tuning baseline |
| GPT-2 + Chain-of-Empathy | Ablation | Empathy-enhanced model |
| GPT-4 / DeepSeek (API) | Strong baseline | Upper bound / judge reference |

---

## Milestones

| Week | Deliverable |
|------|-------------|
| 1 | Repo restructure, training pipeline, 3-model generation JSONL |
| 2 | Rubric finalized, 200-sample human annotation |
| 3–4 | LLM judge pipeline, judge reliability report |
| 5–6 | Calibration (isotonic → ordinal), diagnostic figures |
| 7–8 | Active sampling experiments (optional) |
