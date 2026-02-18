#!/usr/bin/env bash
# ============================================================
# run_judge.sh — Run LLM-as-a-judge over all generated outputs
#
# Usage:
#   OPENAI_API_KEY=sk-... bash experiments/run_judge.sh
#   DEEPSEEK_API_KEY=sk-... bash experiments/run_judge.sh --backend deepseek
#
# Prerequisites:
#   - pip install openai
#   - Generated JSONL files in outputs/generations/
# ============================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# Defaults
JUDGE_MODEL="gpt-4"
BACKEND="openai"
N_REPEATS=3
TEMPERATURE=0.3
DELAY=0.5
GEN_DIR="outputs/generations"
OUT_DIR="outputs/judge"

# Parse flags
while [[ $# -gt 0 ]]; do
    case $1 in
        --judge_model) JUDGE_MODEL="$2"; shift 2;;
        --backend) BACKEND="$2"; shift 2;;
        --n_repeats) N_REPEATS="$2"; shift 2;;
        --delay) DELAY="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

mkdir -p "$OUT_DIR"

echo "=== LLM-as-a-Judge (model=$JUDGE_MODEL, backend=$BACKEND, repeats=$N_REPEATS) ==="

for GEN_FILE in "$GEN_DIR"/*.jsonl; do
    BASENAME=$(basename "$GEN_FILE" .jsonl)
    OUT_FILE="$OUT_DIR/${BASENAME}_judge.jsonl"

    echo "Judging: $GEN_FILE → $OUT_FILE"
    python -m src.eval.llm_judge \
        --generations "$GEN_FILE" \
        --output "$OUT_FILE" \
        --judge_model "$JUDGE_MODEL" \
        --judge_backend "$BACKEND" \
        --temperature "$TEMPERATURE" \
        --n_repeats "$N_REPEATS" \
        --delay "$DELAY"
done

echo ""
echo "=== Judge outputs in $OUT_DIR/ ==="
ls -lh "$OUT_DIR/"
