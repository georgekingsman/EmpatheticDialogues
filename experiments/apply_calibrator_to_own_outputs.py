"""
Step 4: Apply external-human-anchored calibrator to own model outputs.

Loads the calibrator trained on external human data (Step 3) and applies it
to the existing judge scores for our 3 models (vanilla / finetuned / empathy_chain),
producing human-calibrated empathy scores.

Usage:
    python experiments/apply_calibrator_to_own_outputs.py \\
        --calibrator checkpoints/calibrators/my_dataset_deepseek_chat_isotonic.pkl \\
        --method isotonic

Outputs:
    outputs/calibrated/<judge>_human_calibrated.jsonl     (all models combined)
    outputs/analysis/human_calibrated_comparison.json      (model comparison table)
    outputs/analysis/human_calibrated_comparison.md
"""

import sys, os, json, argparse, pickle
sys.path.insert(0, ".")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from collections import defaultdict
from pathlib import Path

from src.eval.rubric import DIMENSION_KEYS
from src.eval.llm_judge import load_judge_results


JUDGE_FILES = {
    "gpt2_vanilla": "outputs/judge/gpt2_vanilla_judge.jsonl",
    "gpt2_finetuned": "outputs/judge/gpt2_finetuned_judge.jsonl",
    "empathy_chain": "outputs/judge/empathy_chain_judge.jsonl",
}


def aggregate_judge_by_sample(judge_results: list[dict]) -> dict[str, dict]:
    """Aggregate multiple judge repeats per sample into mean scores."""
    by_sample: dict[str, list[dict]] = defaultdict(list)
    for r in judge_results:
        if "scores" in r:
            by_sample[r["sample_id"]].append(r)

    aggregated = {}
    for sid, repeats in by_sample.items():
        scores_mean = {}
        scores_std = {}
        for dim in DIMENSION_KEYS:
            vals = [r["scores"][dim] for r in repeats]
            scores_mean[dim] = float(np.mean(vals))
            scores_std[dim] = float(np.std(vals))

        overalls = [r["overall"] for r in repeats]
        confs = [r.get("confidence", 0.5) for r in repeats]

        aggregated[sid] = {
            "judge_mean": scores_mean,
            "judge_std": scores_std,
            "overall_mean": float(np.mean(overalls)),
            "overall_std": float(np.std(overalls)),
            "confidence_mean": float(np.mean(confs)),
            "model": repeats[0].get("model", "unknown"),
            "n_repeats": len(repeats),
        }
    return aggregated


def main():
    parser = argparse.ArgumentParser(
        description="Apply external calibrator to own model judge outputs"
    )
    parser.add_argument("--calibrator", type=str, required=True,
                        help="Path to calibrator .pkl (from train_external_calibrator)")
    parser.add_argument("--method", choices=["isotonic", "ordinal"], default="isotonic",
                        help="Calibration method (must match pkl)")
    parser.add_argument("--output_dir", default="outputs/calibrated",
                        help="Output directory for calibrated scores")
    parser.add_argument("--analysis_dir", default="outputs/analysis",
                        help="Output directory for comparison report")
    parser.add_argument("--judge_tag", default="deepseek_chat",
                        help="Judge model tag for output naming")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.analysis_dir, exist_ok=True)

    # ─── Load calibrator ───
    print("=" * 60)
    print("Loading calibrator")
    print("=" * 60)

    with open(args.calibrator, "rb") as f:
        calibrator = pickle.load(f)
    print(f"  Loaded calibrator from {args.calibrator}")
    print(f"  Method: {args.method}")
    print(f"  Dimensions with fitted models: "
          f"{[d for d in DIMENSION_KEYS if d in calibrator.models]}")

    # ─── Load and calibrate each model's judge outputs ───
    all_records = []
    model_summaries = {}

    for model_tag, judge_path in JUDGE_FILES.items():
        print(f"\n{'=' * 60}")
        print(f"Model: {model_tag}")
        print("=" * 60)

        if not os.path.exists(judge_path):
            print(f"  SKIP: {judge_path} not found")
            continue

        judge_results = load_judge_results(judge_path)
        print(f"  Judge results: {len(judge_results)}")

        aggregated = aggregate_judge_by_sample(judge_results)
        print(f"  Unique samples: {len(aggregated)}")

        # Apply calibrator
        calibrated_records = []
        for sid, data in aggregated.items():
            if args.method == "isotonic":
                cal_scores = calibrator.transform(data["judge_mean"])
            elif args.method == "ordinal":
                cal_scores = calibrator.transform(
                    data["judge_mean"],
                    judge_std=data["judge_std"],
                    confidence=data["confidence_mean"],
                )
            else:
                cal_scores = data["judge_mean"]

            # Compute calibrated overall as mean of calibrated dimensions
            cal_overall = float(np.mean([cal_scores[d] for d in DIMENSION_KEYS]))

            record = {
                "sample_id": sid,
                "model": model_tag,
                "judge_raw": data["judge_mean"],
                "judge_std": data["judge_std"],
                "judge_overall_raw": data["overall_mean"],
                "judge_confidence": data["confidence_mean"],
                "calibrated": cal_scores,
                "calibrated_overall": round(cal_overall, 3),
                "n_repeats": data["n_repeats"],
            }
            calibrated_records.append(record)
            all_records.append(record)

        # Model-level summary
        raw_overalls = [r["judge_overall_raw"] for r in calibrated_records]
        cal_overalls = [r["calibrated_overall"] for r in calibrated_records]

        per_dim_raw = {dim: [] for dim in DIMENSION_KEYS}
        per_dim_cal = {dim: [] for dim in DIMENSION_KEYS}
        for r in calibrated_records:
            for dim in DIMENSION_KEYS:
                per_dim_raw[dim].append(r["judge_raw"][dim])
                per_dim_cal[dim].append(r["calibrated"][dim])

        summary = {
            "n_samples": len(calibrated_records),
            "raw_overall": {
                "mean": round(float(np.mean(raw_overalls)), 3),
                "std": round(float(np.std(raw_overalls)), 3),
            },
            "calibrated_overall": {
                "mean": round(float(np.mean(cal_overalls)), 3),
                "std": round(float(np.std(cal_overalls)), 3),
            },
            "per_dimension": {},
        }
        for dim in DIMENSION_KEYS:
            summary["per_dimension"][dim] = {
                "raw_mean": round(float(np.mean(per_dim_raw[dim])), 3),
                "calibrated_mean": round(float(np.mean(per_dim_cal[dim])), 3),
            }

        model_summaries[model_tag] = summary

        print(f"  Raw overall:        {summary['raw_overall']['mean']:.3f} ± {summary['raw_overall']['std']:.3f}")
        print(f"  Calibrated overall: {summary['calibrated_overall']['mean']:.3f} ± {summary['calibrated_overall']['std']:.3f}")
        for dim in DIMENSION_KEYS:
            d = summary["per_dimension"][dim]
            print(f"    {dim}: {d['raw_mean']:.3f} → {d['calibrated_mean']:.3f}")

    # ─── Save all calibrated records ───
    output_path = os.path.join(
        args.output_dir, f"{args.judge_tag}_human_calibrated.jsonl"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\n  Calibrated scores → {output_path}")
    print(f"  Total records: {len(all_records)}")

    # ─── Generate comparison report ───
    print(f"\n{'=' * 60}")
    print("Model Comparison (Human-Calibrated)")
    print("=" * 60)

    report = {
        "calibrator": args.calibrator,
        "method": args.method,
        "models": model_summaries,
    }
    json_path = os.path.join(args.analysis_dir, "human_calibrated_comparison.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Markdown table
    md_lines = [
        "# Human-Calibrated Model Comparison",
        "",
        f"**Calibrator**: {args.calibrator}",
        f"**Method**: {args.method}",
        "",
        "## Overall Empathy Score (Human-Calibrated)",
        "",
        "| Model | Raw Overall | Calibrated Overall | N |",
        "|-------|:---:|:---:|:---:|",
    ]
    for model_tag in ["gpt2_vanilla", "gpt2_finetuned", "empathy_chain"]:
        s = model_summaries.get(model_tag)
        if not s:
            continue
        md_lines.append(
            f"| {model_tag} | "
            f"{s['raw_overall']['mean']:.3f} ± {s['raw_overall']['std']:.3f} | "
            f"{s['calibrated_overall']['mean']:.3f} ± {s['calibrated_overall']['std']:.3f} | "
            f"{s['n_samples']} |"
        )

    md_lines.extend([
        "",
        "## Per-Dimension Comparison",
        "",
        "| Model | Dimension | Raw Mean | Calibrated Mean |",
        "|-------|-----------|:---:|:---:|",
    ])
    for model_tag in ["gpt2_vanilla", "gpt2_finetuned", "empathy_chain"]:
        s = model_summaries.get(model_tag)
        if not s:
            continue
        for dim in DIMENSION_KEYS:
            d = s["per_dimension"][dim]
            md_lines.append(
                f"| {model_tag} | {dim} | {d['raw_mean']:.3f} | {d['calibrated_mean']:.3f} |"
            )

    md_lines.extend([
        "",
        "## Interpretation",
        "",
        "These scores are **human-anchored**: the calibrator was trained on external",
        "human annotations from a public dataset, so the calibrated scores approximate",
        "what human raters would assign on a 1–5 empathy scale.",
        "",
        "- **Higher is better** for all dimensions",
        "- The calibration corrects for systematic judge bias (typically optimistic)",
        "- The ranking across models should be more reliable than raw judge scores",
    ])

    md = "\n".join(md_lines)
    md_path = os.path.join(args.analysis_dir, "human_calibrated_comparison.md")
    with open(md_path, "w") as f:
        f.write(md)
    print(f"\nJSON  → {json_path}")
    print(f"MD    → {md_path}")

    # Print final comparison table
    print(f"\n{'=' * 60}")
    print("FINAL COMPARISON TABLE")
    print("=" * 60)
    print(f"{'Model':<20} {'Raw Overall':>14} {'Calibrated':>14} {'N':>5}")
    print("-" * 55)
    for model_tag in ["gpt2_vanilla", "gpt2_finetuned", "empathy_chain"]:
        s = model_summaries.get(model_tag)
        if not s:
            continue
        print(f"{model_tag:<20} "
              f"{s['raw_overall']['mean']:>6.3f} ± {s['raw_overall']['std']:.3f} "
              f"{s['calibrated_overall']['mean']:>6.3f} ± {s['calibrated_overall']['std']:.3f} "
              f"{s['n_samples']:>5}")

    print("\n✅ Calibrator application complete.")


if __name__ == "__main__":
    main()
