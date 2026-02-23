#!/bin/bash
# Full research pipeline execution script.
#
# Prerequisites:
#   - Human annotations collected and placed at:
#     outputs/labels/human/ratings_r1.csv
#     outputs/labels/human/ratings_r2.csv
#   - Python environment with: scikit-learn, scipy, mord (optional), openai
#
# Usage:
#   bash experiments/run_paper_pipeline.sh          # full pipeline
#   bash experiments/run_paper_pipeline.sh pilot     # pilot only
#   bash experiments/run_paper_pipeline.sh analysis   # analysis only (skip pilot)

set -euo pipefail
cd "$(dirname "$0")/.."

HUMAN_R1="outputs/labels/human/ratings_r1.csv"
HUMAN_R2="outputs/labels/human/ratings_r2.csv"
HUMAN_LABELS="$HUMAN_R1,$HUMAN_R2"

PILOT_R1="outputs/human_annotation/pilot/pilot_annotation_R1.csv"
PILOT_R2="outputs/human_annotation/pilot/pilot_annotation_R2.csv"

MODE="${1:-full}"

echo "=========================================="
echo "  EmpatheticDialogues Research Pipeline"
echo "  Mode: $MODE"
echo "=========================================="

# ---- PHASE 1: PILOT ----
if [[ "$MODE" == "full" || "$MODE" == "pilot" ]]; then
    echo ""
    echo ">>> Phase 1: Pilot annotation generation"
    python experiments/generate_pilot_annotation.py

    echo ""
    echo ">>> WAITING: Annotators need to fill pilot sheets"
    echo "    R1: $PILOT_R1"
    echo "    R2: $PILOT_R2"
    echo ""

    if [[ -f "$PILOT_R1" && -f "$PILOT_R2" ]]; then
        # Check if files have data (more than just header)
        r1_lines=$(wc -l < "$PILOT_R1" | tr -d ' ')
        if [[ "$r1_lines" -gt 1 ]]; then
            echo ">>> Phase 1b: Running pilot IAA"
            python experiments/run_pilot_iaa.py \
                --r1 "$PILOT_R1" \
                --r2 "$PILOT_R2" \
                --mapping outputs/human_annotation/pilot/_pilot_mapping.json \
                --duplicates outputs/human_annotation/pilot/_duplicate_pairs.json \
                --output_dir outputs/analysis
            echo ">>> Check outputs/analysis/pilot_iaa_report.md for GO/NO-GO decision"

            # Auto-check GO/NO-GO
            if python -c "
import json
with open('outputs/analysis/pilot_iaa_report.json') as f:
    r = json.load(f)
exit(0 if r['go_nogo_decisions']['_overall']['all_dimensions_pass'] else 1)
" 2>/dev/null; then
                echo ""
                echo ">>> GO! Generating full 600-sample annotation batch..."
                python experiments/generate_full_annotation.py
            else
                echo ""
                echo ">>> HOLD — Running NO-GO recovery workflow..."
                python experiments/nogo_recovery.py \
                    --iaa_report outputs/analysis/pilot_iaa_report.json \
                    --r1 "$PILOT_R1" \
                    --r2 "$PILOT_R2"
                echo ">>> Fix rubric, run alignment meeting, then do mini-pilot."
                echo ">>> See outputs/nogo_recovery/ for materials."
                exit 0
            fi
        else
            echo ">>> Pilot sheets exist but appear empty. Fill them and re-run."
        fi
    else
        echo ">>> Pilot sheets not yet filled. Generate done. Stopping."
        echo "    After annotators complete, run:"
        echo "    python experiments/run_pilot_iaa.py"
        exit 0
    fi
fi

# ---- PHASE 2-5: ANALYSIS (requires full human labels) ----
if [[ "$MODE" == "full" || "$MODE" == "analysis" ]]; then

    # Check for human labels
    if [[ ! -f "$HUMAN_R1" || ! -f "$HUMAN_R2" ]]; then
        echo ""
        echo ">>> ERROR: Full human labels not found at:"
        echo "    $HUMAN_R1"
        echo "    $HUMAN_R2"
        echo ">>> Complete annotation first, then re-run with: bash $0 analysis"
        exit 1
    fi

    echo ""
    echo ">>> Phase 3: Judge ↔ Human alignment analysis"
    python experiments/judge_vs_human_analysis.py \
        --human "$HUMAN_LABELS"

    echo ""
    echo ">>> Phase 4: Paper-grade calibration (with bootstrap CI)"
    python experiments/run_calibration_paper.py \
        --human "$HUMAN_LABELS" \
        --n_bootstrap 1000

    echo ""
    echo ">>> Phase 5a: Ablation B — repeats sensitivity (no API needed)"
    python experiments/run_ablation_repeats.py \
        --human "$HUMAN_LABELS"

    echo ""
    echo ">>> Phase 5b: Ablation A — prompt variant (dry run preview)"
    python experiments/run_ablation_prompt.py \
        --human "$HUMAN_LABELS" \
        --dry_run

    echo ""
    echo "=========================================="
    echo "  Pipeline complete! Check:"
    echo "    outputs/analysis/judge_vs_human_raw.md"
    echo "    outputs/analysis/calibration_report_paper.md"
    echo "    outputs/analysis/ablation_repeats.md"
    echo "=========================================="
fi
