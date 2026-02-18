#!/usr/bin/env bash
# ============================================================
# run_calibrate.sh — Calibrate judge scores against human ratings
#
# Usage:
#   bash experiments/run_calibrate.sh
#
# Prerequisites:
#   - Human annotations in outputs/labels/human/
#   - Judge outputs in outputs/judge/
#   - pip install scikit-learn scipy mord (for ordinal)
# ============================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

HUMAN_DIR="outputs/labels/human"
JUDGE_DIR="outputs/judge"
OUT_DIR="outputs/calibrated"
METHOD="isotonic"  # or "ordinal"

mkdir -p "$OUT_DIR"

echo "=== Calibration (method=$METHOD) ==="

# Find human label files
for HUMAN_FILE in "$HUMAN_DIR"/*.csv; do
    if [ ! -f "$HUMAN_FILE" ]; then
        echo "No human label files found in $HUMAN_DIR/"
        exit 1
    fi

    BASENAME=$(basename "$HUMAN_FILE" .csv)

    # Look for matching judge file
    JUDGE_FILE="$JUDGE_DIR/${BASENAME}_judge.jsonl"
    if [ ! -f "$JUDGE_FILE" ]; then
        # Try to find any judge file
        JUDGE_FILE=$(ls "$JUDGE_DIR"/*.jsonl 2>/dev/null | head -1)
        if [ -z "$JUDGE_FILE" ]; then
            echo "  ⚠ No judge file found for $BASENAME, skipping."
            continue
        fi
    fi

    echo "Calibrating: human=$HUMAN_FILE, judge=$JUDGE_FILE"
    python -m src.eval.calibrate \
        --human_labels "$HUMAN_FILE" \
        --judge_results "$JUDGE_FILE" \
        --output "$OUT_DIR/${BASENAME}_calibrated.jsonl" \
        --method "$METHOD" \
        --report "$OUT_DIR/${BASENAME}_report.json"
done

echo ""
echo "=== Calibration outputs in $OUT_DIR/ ==="
ls -lh "$OUT_DIR/" 2>/dev/null || echo "(empty)"
