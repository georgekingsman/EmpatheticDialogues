"""
Run the full calibration pipeline with simulated human labels.

Tests both isotonic and ordinal calibration, computes all metrics,
and produces the calibration report.
"""
import sys, os, json
sys.path.insert(0, ".")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from src.eval.human_labels_schema import load_human_labels_csv, compute_iaa
from src.eval.llm_judge import load_judge_results
from src.eval.calibrate import (
    merge_human_and_judge, calibrate_all,
    compute_calibration_metrics, compute_ece,
    save_calibrated,
)
from src.eval.metrics import compute_judge_stability, compute_judge_human_correlation
from src.eval.rubric import DIMENSION_KEYS


def main():
    os.makedirs("outputs/calibrated", exist_ok=True)
    os.makedirs("outputs/analysis", exist_ok=True)

    # ---------------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------------
    print("=" * 60)
    print("Loading data")
    print("=" * 60)

    human = load_human_labels_csv("outputs/human_annotation/simulated_human_labels.csv")
    print(f"  Human labels: {len(human)} rows")

    # Combine all judge results
    all_judge = []
    for tag in ["gpt2_vanilla", "gpt2_finetuned", "empathy_chain"]:
        path = f"outputs/judge/{tag}_judge.jsonl"
        results = load_judge_results(path)
        all_judge.extend(results)
        print(f"  Judge {tag}: {len(results)} records")
    print(f"  Total judge records: {len(all_judge)}")

    # ---------------------------------------------------------------
    # 2. Inter-annotator agreement
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Inter-Annotator Agreement (Cohen's Kappa)")
    print("=" * 60)
    iaa = compute_iaa(human)
    for dim, kappa in iaa.items():
        print(f"  {dim:15s}: κ = {kappa:.4f}")

    # ---------------------------------------------------------------
    # 3. Merge human + judge
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Merging human labels and judge scores")
    print("=" * 60)
    merged = merge_human_and_judge(human, all_judge)
    print(f"  Merged {len(merged)} common samples")

    # ---------------------------------------------------------------
    # 4. Pre-calibration metrics
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Pre-Calibration Metrics (raw judge vs human)")
    print("=" * 60)
    pre_metrics = compute_calibration_metrics(merged)
    for dim in DIMENSION_KEYS:
        m = pre_metrics[dim]
        if "note" in m:
            print(f"  {dim}: {m['note']}")
            continue
        print(f"  {dim:15s}: MAE={m['raw_mae']:.3f}  RMSE={m['raw_rmse']:.3f}  "
              f"Spearman={m['raw_spearman']:.3f} (p={m['raw_spearman_p']:.4f})  "
              f"Kendall={m['raw_kendall']:.3f}")

    # ---------------------------------------------------------------
    # 5. ECE
    # ---------------------------------------------------------------
    ece = compute_ece(merged)
    print("\n  Expected Calibration Error (ECE):")
    for dim, val in ece.items():
        print(f"    {dim:15s}: {val:.4f}")

    # ---------------------------------------------------------------
    # 6. Judge ↔ Human correlation
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Judge ↔ Human Correlation")
    print("=" * 60)
    corr = compute_judge_human_correlation(human, all_judge)
    for dim, c in corr.items():
        if "note" in c:
            print(f"  {dim}: {c['note']}")
        else:
            print(f"  {dim:15s}: Spearman={c['spearman']:.3f} (p={c['spearman_p']:.4f})  "
                  f"Kendall={c['kendall']:.3f} (p={c['kendall_p']:.4f})  n={c['n']}")

    # ---------------------------------------------------------------
    # 7. Isotonic calibration
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Route 1: Isotonic Calibration")
    print("=" * 60)
    iso_scores, iso_cal = calibrate_all(merged, method="isotonic")
    iso_metrics = compute_calibration_metrics(merged, iso_scores)

    for dim in DIMENSION_KEYS:
        m = iso_metrics[dim]
        if "note" in m:
            print(f"  {dim}: {m['note']}")
            continue
        pre = pre_metrics[dim]
        print(f"  {dim:15s}: MAE {pre['raw_mae']:.3f} → {m['calibrated_mae']:.3f}  "
              f"RMSE {pre['raw_rmse']:.3f} → {m['calibrated_rmse']:.3f}  "
              f"Spearman {pre['raw_spearman']:.3f} → {m['calibrated_spearman']:.3f}")

    save_calibrated(merged, iso_scores, "outputs/calibrated/isotonic_calibrated.jsonl")

    # ---------------------------------------------------------------
    # 8. Ordinal calibration
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Route 2: Ordinal Calibration")
    print("=" * 60)
    try:
        ord_scores, ord_cal = calibrate_all(merged, method="ordinal")
        ord_metrics = compute_calibration_metrics(merged, ord_scores)

        for dim in DIMENSION_KEYS:
            m = ord_metrics[dim]
            if "note" in m:
                print(f"  {dim}: {m['note']}")
                continue
            pre = pre_metrics[dim]
            print(f"  {dim:15s}: MAE {pre['raw_mae']:.3f} → {m['calibrated_mae']:.3f}  "
                  f"RMSE {pre['raw_rmse']:.3f} → {m['calibrated_rmse']:.3f}  "
                  f"Spearman {pre['raw_spearman']:.3f} → {m['calibrated_spearman']:.3f}")

        save_calibrated(merged, ord_scores, "outputs/calibrated/ordinal_calibrated.jsonl")
    except Exception as e:
        print(f"  Ordinal calibration failed: {e}")
        ord_metrics = None

    # ---------------------------------------------------------------
    # 9. Save comprehensive report
    # ---------------------------------------------------------------
    report = {
        "n_human_labels": len(human),
        "n_judge_results": len(all_judge),
        "n_merged_samples": len(merged),
        "inter_annotator_agreement": iaa,
        "pre_calibration": pre_metrics,
        "ece": ece,
        "judge_human_correlation": corr,
        "isotonic_calibration": iso_metrics,
    }
    if ord_metrics:
        report["ordinal_calibration"] = ord_metrics

    report_path = "outputs/analysis/calibration_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n\nFull calibration report → {report_path}")
    print("Pipeline test complete!")


if __name__ == "__main__":
    main()
