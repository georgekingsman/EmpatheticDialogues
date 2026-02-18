"""
Analyse LLM-as-Judge results after evaluation completes.

Produces:
  - Per-model score summary (mean/std per dimension)
  - Judge self-consistency / stability analysis
  - Active sampling recommendations (which samples to annotate first)
  - Cross-model comparison table
  - outputs/analysis/judge_analysis_report.json
"""
import sys, os
sys.path.insert(0, ".")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import numpy as np
from collections import defaultdict
from src.eval.llm_judge import load_judge_results
from src.eval.metrics import compute_judge_stability, select_for_annotation
from src.eval.rubric import DIMENSION_KEYS

JUDGE_FILES = {
    "gpt2_vanilla": "outputs/judge/gpt2_vanilla_judge.jsonl",
    "gpt2_finetuned": "outputs/judge/gpt2_finetuned_judge.jsonl",
    "empathy_chain": "outputs/judge/empathy_chain_judge.jsonl",
}


def analyse_model_scores(results: list[dict]) -> dict:
    """Compute per-dimension statistics for one model's judge results."""
    # Group by sample_id, average across repeats first
    by_sample = defaultdict(list)
    for r in results:
        if "scores" in r:
            by_sample[r["sample_id"]].append(r)

    dim_scores = {dim: [] for dim in DIMENSION_KEYS}
    overall_scores = []
    confidence_vals = []

    for sid, repeats in by_sample.items():
        for dim in DIMENSION_KEYS:
            vals = [r["scores"][dim] for r in repeats]
            dim_scores[dim].append(float(np.mean(vals)))
        overalls = [r["overall"] for r in repeats]
        overall_scores.append(float(np.mean(overalls)))
        confs = [r.get("confidence", 0.5) for r in repeats]
        confidence_vals.append(float(np.mean(confs)))

    stats = {}
    for dim in DIMENSION_KEYS:
        vals = dim_scores[dim]
        stats[dim] = {
            "mean": round(float(np.mean(vals)), 3),
            "std": round(float(np.std(vals)), 3),
            "median": round(float(np.median(vals)), 3),
            "min": round(float(np.min(vals)), 1),
            "max": round(float(np.max(vals)), 1),
        }
    stats["overall"] = {
        "mean": round(float(np.mean(overall_scores)), 3),
        "std": round(float(np.std(overall_scores)), 3),
    }
    stats["confidence"] = {
        "mean": round(float(np.mean(confidence_vals)), 3),
        "std": round(float(np.std(confidence_vals)), 3),
    }
    stats["n_samples"] = len(by_sample)
    stats["n_scored"] = sum(1 for r in results if "scores" in r)
    stats["n_errors"] = sum(1 for r in results if "error" in r)

    return stats


def print_comparison_table(all_stats: dict):
    """Print a formatted comparison table."""
    models = list(all_stats.keys())
    dims = DIMENSION_KEYS + ["overall"]

    print("\n" + "=" * 80)
    print("CROSS-MODEL COMPARISON (mean ± std, averaged over repeats)")
    print("=" * 80)

    header = f"{'Dimension':<15}" + "".join(f"{m:>22}" for m in models)
    print(header)
    print("-" * len(header))

    for dim in dims:
        row = f"{dim:<15}"
        for m in models:
            s = all_stats[m].get(dim, {})
            mean = s.get("mean", 0)
            std = s.get("std", 0)
            row += f"{mean:>7.2f} ± {std:<5.2f}      "
        print(row)

    # Also print confidence and error rates
    print("-" * len(header))
    row = f"{'confidence':<15}"
    for m in models:
        s = all_stats[m].get("confidence", {})
        row += f"{s.get('mean', 0):>7.2f} ± {s.get('std', 0):<5.2f}      "
    print(row)

    row = f"{'errors':<15}"
    for m in models:
        n_err = all_stats[m].get("n_errors", 0)
        n_total = all_stats[m].get("n_scored", 0) + n_err
        row += f"{n_err:>7d} / {n_total:<5d}      "
    print(row)
    print("=" * 80)


def main():
    os.makedirs("outputs/analysis", exist_ok=True)

    all_results = {}
    all_stats = {}
    all_stability = {}
    all_active = {}
    combined_judge = []

    for tag, path in JUDGE_FILES.items():
        if not os.path.exists(path):
            print(f"  SKIP {tag}: {path} not found")
            continue

        results = load_judge_results(path)
        all_results[tag] = results
        combined_judge.extend(results)

        print(f"\n--- {tag} ({len(results)} records) ---")

        # Per-model score stats
        stats = analyse_model_scores(results)
        all_stats[tag] = stats
        print(f"  Samples: {stats['n_samples']}, Scored: {stats['n_scored']}, Errors: {stats['n_errors']}")
        for dim in DIMENSION_KEYS:
            s = stats[dim]
            print(f"  {dim:15s}: {s['mean']:.2f} ± {s['std']:.2f} (median={s['median']:.1f})")
        print(f"  {'overall':15s}: {stats['overall']['mean']:.2f} ± {stats['overall']['std']:.2f}")

        # Stability analysis
        stability = compute_judge_stability(results)
        all_stability[tag] = stability
        print(f"\n  Judge stability ({stability['n_groups']} groups, {stability['total_pairs']} pairs):")
        for dim in DIMENSION_KEYS:
            ds = stability["per_dimension"][dim]
            print(f"    {dim:15s}: mean_std={ds['mean_std']}, exact_agree={ds['exact_agreement_rate']}, near_agree={ds['near_agreement_rate']}")

        # Active sampling
        active = select_for_annotation(results, n=20, strategy="uncertainty")
        all_active[tag] = active
        print(f"\n  Top-20 uncertain samples: {active[:5]}...")

    # Cross-model comparison
    if len(all_stats) > 1:
        print_comparison_table(all_stats)

    # Combined active sampling across all models
    if combined_judge:
        combined_active = select_for_annotation(combined_judge, n=50, strategy="uncertainty")
        all_active["combined_top50"] = combined_active
        print(f"\nTop-50 most uncertain samples (combined): {combined_active[:10]}...")

    # Save full report
    report = {
        "model_scores": all_stats,
        "stability": all_stability,
        "active_sampling": all_active,
    }
    report_path = "outputs/analysis/judge_analysis_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved → {report_path}")


if __name__ == "__main__":
    main()
