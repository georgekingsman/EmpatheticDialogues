#!/usr/bin/env bash
# ============================================================
# run_train.sh — Train baseline and empathy models
#
# Usage:
#   bash experiments/run_train.sh
# ============================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

SEED=42
EPOCHS=3
BATCH_SIZE=4
LR=5e-5
MAX_LENGTH=512
DATA_PATH="data/formatted_Psych_data.jsonl"
OUTPUT_DIR="checkpoints"

mkdir -p "$OUTPUT_DIR"

echo "=== Training models (seed=$SEED, epochs=$EPOCHS) ==="

# --- 1. GPT-2 Baseline ---
echo "[1/2] Training GPT-2 baseline..."
python -m src.models.train \
    --model_type baseline \
    --model_name gpt2 \
    --data_path "$DATA_PATH" \
    --max_length "$MAX_LENGTH" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --seed "$SEED" \
    --output_dir "$OUTPUT_DIR"

# --- 2. Empathy Chain model ---
echo "[2/2] Training CBT Empathy Chain..."
python -m src.models.train \
    --model_type empathy \
    --model_name gpt2 \
    --data_path "$DATA_PATH" \
    --max_length "$MAX_LENGTH" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --seed "$SEED" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "=== Training complete. Checkpoints: ==="
ls -lh "$OUTPUT_DIR/"
