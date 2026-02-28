# Empathetic Dialogue Evaluation Framework

**Direction B: Human Rating ↔ LLM-as-a-Judge Calibration**

A reproducible, extensible framework for evaluating empathetic/supportive dialogue using **external human-anchored calibration** and LLM-as-a-judge, with statistical calibration between the two.

We do **not** perform our own human annotation. Instead, we validate and calibrate our LLM judge against **publicly available human-rated datasets**, providing an unbiased, reproducible anchor for score alignment.

---

## Research Contributions

1. **Reproducible evaluation rubric** — 4-dimension rubric (Emotion Recognition, Validation & Warmth, Helpfulness, Safety & Boundaries) with anchor descriptions and annotation protocol
2. **LLM-as-a-judge pipeline** — structured JSON output, multi-sample stability analysis, supports any OpenAI-compatible API
3. **External human-anchored calibration** — Calibrator trained on public human-rated datasets (not our own labels); isotonic regression (Route 1) + ordinal logistic regression (Route 2), with bootstrap 95% CI
4. **Ablation studies** — Repeats sensitivity (k=1/2/3) and prompt variant comparison, validated against external human labels

---

## Project Structure

```
EmpatheticDialogues/
  src/
    data/
      build_dataset.py        # Dataset loading, splitting, label masking
      templates.py             # Prompt/response templates
      external_loader.py       # External dataset loader (Route B) ★ NEW
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
    run_external_judge.py      # Judge external dataset ★ NEW
    train_external_calibrator.py  # Train calibrator on external human data ★ NEW
    apply_calibrator_to_own_outputs.py  # Apply to our 3 models ★ NEW
    run_external_ablation.py   # Ablation with external labels ★ NEW
  outputs/
    generations/*.jsonl        # Model outputs
    judge/*.jsonl              # Judge scores (our models)
    judge_external/*.jsonl     # Judge scores (external dataset) ★ NEW
    calibrated/*.jsonl         # Calibrated scores
  checkpoints/
    calibrators/*.pkl          # Trained calibrator models ★ NEW
  docs/
    rubric_v1.md               # Full rubric with examples
    annotation_guide_v1.md     # Annotator instructions
  data/
    formatted_Psych_data.jsonl # Training data (5319 samples)
    Psych_data.csv             # Raw data
    external/                  # External human-rated datasets ★ NEW
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

### 5. External Human-Anchored Calibration (Route B)
```bash
# Step 1: Load external dataset
python -m src.data.external_loader --input data/external/my_dataset.csv --output data/external/unified.jsonl

# Step 2: Judge external data
python experiments/run_external_judge.py --input data/external/unified.jsonl --dataset my_dataset

# Step 3: Train calibrator on external human labels
python experiments/train_external_calibrator.py \
    --external_data data/external/unified.jsonl \
    --judge_results outputs/judge_external/my_dataset_deepseek_chat.jsonl \
    --dataset my_dataset

# Step 4: Apply calibrator to our 3 models
python experiments/apply_calibrator_to_own_outputs.py \
    --calibrator checkpoints/calibrators/my_dataset_deepseek_chat_isotonic.pkl

# Step 5: Ablation
python experiments/run_external_ablation.py \
    --external_data data/external/unified.jsonl \
    --judge_results outputs/judge_external/my_dataset_deepseek_chat.jsonl
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
| 2 | Rubric finalized, LLM judge pipeline, 1800 API evaluations |
| 3 | Calibration pipeline (isotonic + ordinal), analysis report |
| 4 | External human-anchored calibration (Route B) |
| 5 | Ablation studies (repeats + prompt variants) |
| 6 | Final model comparison table, paper writing |
