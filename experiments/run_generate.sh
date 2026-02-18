#!/usr/bin/env bash
# ============================================================
# run_generate.sh — Generate responses from multiple models
#
# Usage:
#   bash experiments/run_generate.sh
#
# Prerequisites:
#   - Trained checkpoints at ./checkpoints/ (or modify paths)
#   - pip install transformers torch
# ============================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

SEED=42
N_SAMPLES=200
MAX_NEW_TOKENS=128
TEMPERATURE=0.7
TOP_P=0.9
DATA_PATH="data/formatted_Psych_data.jsonl"
OUT_DIR="outputs/generations"

mkdir -p "$OUT_DIR"

echo "=== Generating responses (seed=$SEED, n=$N_SAMPLES) ==="

# --- 1. GPT-2 Baseline (no fine-tuning) ---
echo "[1/3] GPT-2 vanilla baseline..."
python -m src.inference.generate \
    --data_path "$DATA_PATH" \
    --model_type baseline \
    --model_name gpt2 \
    --output "$OUT_DIR/gpt2_vanilla.jsonl" \
    --n_samples "$N_SAMPLES" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --seed "$SEED"

# --- 2. Fine-tuned GPT-2 Baseline ---
echo "[2/3] Fine-tuned GPT-2 baseline..."
if [ -f "checkpoints/gpt2_baseline.pt" ]; then
    python -m src.inference.generate \
        --data_path "$DATA_PATH" \
        --model_type baseline \
        --model_name gpt2 \
        --checkpoint "checkpoints/gpt2_baseline.pt" \
        --output "$OUT_DIR/gpt2_finetuned.jsonl" \
        --n_samples "$N_SAMPLES" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --temperature "$TEMPERATURE" \
        --top_p "$TOP_P" \
        --seed "$SEED"
else
    echo "  ⚠ checkpoints/gpt2_baseline.pt not found, skipping."
fi

# --- 3. Empathy Chain model ---
echo "[3/3] CBT Empathy Chain model..."
if [ -f "checkpoints/cbt_gpt2_model.pt" ]; then
    python -m src.inference.generate \
        --data_path "$DATA_PATH" \
        --model_type empathy \
        --model_name gpt2 \
        --checkpoint "checkpoints/cbt_gpt2_model.pt" \
        --output "$OUT_DIR/empathy_chain.jsonl" \
        --n_samples "$N_SAMPLES" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --temperature "$TEMPERATURE" \
        --top_p "$TOP_P" \
        --seed "$SEED"
else
    echo "  ⚠ checkpoints/cbt_gpt2_model.pt not found, skipping."
fi

echo ""
echo "=== Generation complete. Outputs in $OUT_DIR/ ==="
ls -lh "$OUT_DIR/"
