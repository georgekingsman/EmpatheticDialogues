"""
Paper-grade calibration pipeline with proper train/dev/test split and bootstrap CI.

Key differences from run_calibration.py (which is for pipeline testing):
  1. Fixed sample_id-based split: train=60%, dev=20%, test=20%
  2. Calibrator fitted ONLY on train, hyperparams tuned on dev, final result on test
  3. Bootstrap 1000× for 95% confidence intervals on test metrics
  4. Both isotonic (Route 1, primary) and ordinal (Route 2, comparison)
  5. Split saved to outputs/analysis/split.json for reproducibility

Outputs:
  - outputs/analysis/split.json
  - outputs/calibrated/calib_isotonic_test.jsonl
  - outputs/calibrated/calib_ordinal_test.jsonl
  - outputs/analysis/calibration_report_paper.json  (with CIs)
  - outputs/analysis/calibration_report_paper.md

Usage:
    python experiments/run_calibration_paper.py \
        --human outputs/labels/human/ratings_r1.csv,outputs/labels/human/ratings_r2.csv \
        --n_bootstrap 1000
"""
import sys, os, json, argparse
sys.path.insert(0, ".")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from collections import defaultdict
from pathlib import Path
from scipy import stats as scipy_stats

from src.eval.rubric import DIMENSION_KEYS
from src.eval.human_labels_schema import load_human_labels_csv, compute_iaa_extended
from src.eval.llm_judge import load_judge_results
from src.eval.calibrate import (
    merge_human_and_judge,
    IsotonicCalibrator,
    OrdinalCalibrator,
    compute_ece,
    save_calibrated,
)

JUDGE_FILES = [
    "outputs/judge/gpt2_vanilla_judge.jsonl",
    "outputs/judge/gpt2_finetuned_judge.jsonl",
    "outputs/judge/empathy_chain_judge.jsonl",
]


# ===================================================================
# Train / Dev / Test split (fixed by sample_id hash)
# ===================================================================

def create_split(
    sample_ids: list[str],
    train_frac: float = 0.6,
    dev_frac: float = 0.2,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Create a deterministic split based on sample_id sorting + seed."""
    rng = np.random.RandomState(seed)
    ids_shuffled = sorted(sample_ids)  # sort first for reproducibility
    rng.shuffle(ids_shuffled)

    n = len(ids_shuffled)
    n_train = int(n * train_frac)
    n_dev = int(n * (train_frac + dev_frac))

    return {
        "train": ids_shuffled[:n_train],
        "dev": ids_shuffled[n_train:n_dev],
        "test": ids_shuffled[n_dev:],
    }


def split_merged(
    merged: dict[str, dict],
    split: dict[str, list[str]],
) -> dict[str, dict[str, dict]]:
    """Partition merged data into train/dev/test."""
    result = {"train": {}, "dev": {}, "test": {}}
    id_to_split = {}
    for s, ids in split.items():
        for sid in ids:
            id_to_split[sid] = s
    for sid, data in merged.items():
        s = id_to_split.get(sid, "test")
        result[s][sid] = data
    return result


# ===================================================================
# Metrics computation
# ===================================================================

def compute_metrics(
    merged_subset: dict[str, dict],
    calibrated_scores: dict[str, dict[str, float]] | None = None,
) -> dict:
    """Compute per-dimension metrics (raw vs calibrated) on a data subset."""
    results = {}
    for dim in DIMENSION_KEYS:
        h_vals, j_raw, j_cal = [], [], []
        for sid, data in merged_subset.items():
            hv = data["human"].get(dim)
            jv = data["judge_mean"].get(dim)
            if hv is None or jv is None or np.isnan(hv):
                continue
            h_vals.append(hv)
            j_raw.append(jv)
            if calibrated_scores and sid in calibrated_scores:
                j_cal.append(calibrated_scores[sid].get(dim, jv))
            else:
                j_cal.append(jv)

        if len(h_vals) < 5:
            results[dim] = {"n": len(h_vals), "note": "insufficient"}
            continue

        h, jr, jc = np.array(h_vals), np.array(j_raw), np.array(j_cal)

        def _m(y_true, y_pred):
            errors = y_pred - y_true
            sp, sp_p = scipy_stats.spearmanr(y_true, y_pred)
            kt, kt_p = scipy_stats.kendalltau(y_true, y_pred)
            return {
                "mae": round(float(np.mean(np.abs(errors))), 4),
                "rmse": round(float(np.sqrt(np.mean(errors ** 2))), 4),
                "bias": round(float(np.mean(errors)), 4),
                "spearman": round(float(sp), 4),
                "kendall": round(float(kt), 4),
            }

        results[dim] = {
            "n": len(h_vals),
            "raw": _m(h, jr),
            "calibrated": _m(h, jc),
        }

    return results


# ===================================================================
# Bootstrap confidence intervals
# ===================================================================

def bootstrap_ci(
    merged_subset: dict[str, dict],
    calibrator,
    method: str,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict:
    """Bootstrap 95% CI for test-set calibration metrics.

    Resamples test samples, applies calibrator, computes metrics each time.
    """
    rng = np.random.RandomState(seed)
    sids = list(merged_subset.keys())
    n = len(sids)

    boot_results: dict[str, dict[str, list]] = {
        dim: {
            "raw_mae": [], "raw_spearman": [],
            "cal_mae": [], "cal_spearman": [],
            "mae_reduction_pct": [],
        }
        for dim in DIMENSION_KEYS
    }

    for b in range(n_bootstrap):
        # Resample with replacement
        idx = rng.choice(n, size=n, replace=True)
        boot_sids = [sids[i] for i in idx]
        boot_merged = {sid: merged_subset[sid] for sid in boot_sids}

        # Calibrate (using already-fitted calibrator)
        boot_cal = {}
        for sid in boot_sids:
            data = merged_subset[sid]
            if method == "isotonic":
                boot_cal[sid] = calibrator.transform(data["judge_mean"])
            elif method == "ordinal":
                boot_cal[sid] = calibrator.transform(
                    data["judge_mean"],
                    judge_std=data["judge_std"],
                    confidence=data["judge_confidence_mean"],
                )

        for dim in DIMENSION_KEYS:
            h_vals, j_raw, j_cal = [], [], []
            for sid in boot_sids:
                data = merged_subset[sid]
                hv = data["human"].get(dim)
                jv = data["judge_mean"].get(dim)
                if hv is None or jv is None or np.isnan(hv):
                    continue
                h_vals.append(hv)
                j_raw.append(jv)
                j_cal.append(boot_cal.get(sid, {}).get(dim, jv))

            if len(h_vals) < 5:
                continue

            h, jr, jc = np.array(h_vals), np.array(j_raw), np.array(j_cal)
            raw_mae = float(np.mean(np.abs(jr - h)))
            cal_mae = float(np.mean(np.abs(jc - h)))

            sp_raw, _ = scipy_stats.spearmanr(h, jr) if len(h) > 2 else (0, 1)
            sp_cal, _ = scipy_stats.spearmanr(h, jc) if len(h) > 2 else (0, 1)

            boot_results[dim]["raw_mae"].append(raw_mae)
            boot_results[dim]["cal_mae"].append(cal_mae)
            boot_results[dim]["raw_spearman"].append(float(sp_raw))
            boot_results[dim]["cal_spearman"].append(float(sp_cal))
            if raw_mae > 0:
                boot_results[dim]["mae_reduction_pct"].append(
                    (raw_mae - cal_mae) / raw_mae * 100
                )

    # Compute CIs
    alpha = 1 - ci_level
    ci = {}
    for dim in DIMENSION_KEYS:
        br = boot_results[dim]
        dim_ci = {}
        for metric_key in ["raw_mae", "cal_mae", "raw_spearman", "cal_spearman", "mae_reduction_pct"]:
            vals = br.get(metric_key, [])
            if len(vals) < 50:
                dim_ci[metric_key] = {"mean": None, "ci_lower": None, "ci_upper": None}
                continue
            vals = np.array(vals)
            dim_ci[metric_key] = {
                "mean": round(float(np.mean(vals)), 4),
                "ci_lower": round(float(np.percentile(vals, alpha / 2 * 100)), 4),
                "ci_upper": round(float(np.percentile(vals, (1 - alpha / 2) * 100)), 4),
            }
        ci[dim] = dim_ci

    return ci


# ===================================================================
# Markdown report
# ===================================================================

def generate_markdown(
    split_sizes: dict,
    train_metrics: dict,
    test_metrics_iso: dict,
    test_metrics_ord: dict | None,
    ci_iso: dict,
    ci_ord: dict | None,
    n_bootstrap: int,
) -> str:
    lines = [
        "# Calibration Report (Paper-Grade)",
        "",
        "## Data Split",
        f"- Train: {split_sizes['train']} samples (60%)",
        f"- Dev: {split_sizes['dev']} samples (20%)",
        f"- Test: {split_sizes['test']} samples (20%)",
        "",
        "## Isotonic Calibration (Route 1 — Primary Result)",
        "",
        "### Test-Set Metrics",
        "",
        "| Dimension | Raw MAE | Cal MAE | MAE Reduction | Raw ρ | Cal ρ |",
        "|-----------|:---:|:---:|:---:|:---:|:---:|",
    ]
    for dim in DIMENSION_KEYS:
        t = test_metrics_iso.get(dim, {})
        if "note" in t:
            lines.append(f"| {dim} | — | — | — | — | — |")
            continue
        raw = t.get("raw", {})
        cal = t.get("calibrated", {})
        raw_mae = raw.get("mae", 0)
        cal_mae = cal.get("mae", 0)
        reduction = (raw_mae - cal_mae) / raw_mae * 100 if raw_mae > 0 else 0
        lines.append(
            f"| {dim} | {raw_mae:.3f} | {cal_mae:.3f} | "
            f"{reduction:.1f}% | {raw.get('spearman', 0):.3f} | {cal.get('spearman', 0):.3f} |"
        )

    lines.extend([
        "",
        f"### Bootstrap 95% CI ({n_bootstrap} iterations)",
        "",
        "| Dimension | MAE reduction % [95% CI] | Cal MAE [95% CI] | Cal ρ [95% CI] |",
        "|-----------|:---:|:---:|:---:|",
    ])
    for dim in DIMENSION_KEYS:
        c = ci_iso.get(dim, {})
        mr = c.get("mae_reduction_pct", {})
        cm = c.get("cal_mae", {})
        cs = c.get("cal_spearman", {})
        lines.append(
            f"| {dim} | "
            f"{mr.get('mean', 'N/A')} [{mr.get('ci_lower', '?')}, {mr.get('ci_upper', '?')}] | "
            f"{cm.get('mean', 'N/A')} [{cm.get('ci_lower', '?')}, {cm.get('ci_upper', '?')}] | "
            f"{cs.get('mean', 'N/A')} [{cs.get('ci_lower', '?')}, {cs.get('ci_upper', '?')}] |"
        )

    if test_metrics_ord:
        lines.extend([
            "",
            "## Ordinal Calibration (Route 2 — Comparison)",
            "",
            "| Dimension | Raw MAE | Cal MAE | MAE Reduction | Raw ρ | Cal ρ |",
            "|-----------|:---:|:---:|:---:|:---:|:---:|",
        ])
        for dim in DIMENSION_KEYS:
            t = test_metrics_ord.get(dim, {})
            if "note" in t:
                lines.append(f"| {dim} | — | — | — | — | — |")
                continue
            raw = t.get("raw", {})
            cal = t.get("calibrated", {})
            raw_mae = raw.get("mae", 0)
            cal_mae = cal.get("mae", 0)
            reduction = (raw_mae - cal_mae) / raw_mae * 100 if raw_mae > 0 else 0
            lines.append(
                f"| {dim} | {raw_mae:.3f} | {cal_mae:.3f} | "
                f"{reduction:.1f}% | {raw.get('spearman', 0):.3f} | {cal.get('spearman', 0):.3f} |"
            )

    return "\n".join(lines)


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Paper-grade calibration")
    parser.add_argument(
        "--human",
        default="outputs/human_annotation/simulated_human_labels.csv",
        help="Comma-separated paths to human label CSVs",
    )
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="outputs/analysis")
    parser.add_argument("--split_file", default=None,
                        help="Path to existing split JSON (reuse for reproducibility)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("outputs/calibrated", exist_ok=True)

    # --- Load human labels ---
    print("=" * 60)
    print("Loading data")
    print("=" * 60)
    all_human = []
    for p in args.human.split(","):
        p = p.strip()
        if os.path.exists(p):
            h = load_human_labels_csv(p)
            all_human.extend(h)
            print(f"  Human: {len(h)} rows from {p}")
    print(f"  Total human: {len(all_human)}")

    # --- Load judge ---
    all_judge = []
    for path in JUDGE_FILES:
        if os.path.exists(path):
            results = load_judge_results(path)
            all_judge.extend(results)
    print(f"  Total judge: {len(all_judge)}")

    # --- IAA (informational) ---
    print("\n  Inter-Annotator Agreement:")
    iaa = compute_iaa_extended(all_human)
    for dim in DIMENSION_KEYS:
        d = iaa.get(dim, {})
        wk = d.get("weighted_kappa_linear", "N/A")
        print(f"    {dim}: κ_w = {wk}")

    # --- Merge ---
    merged = merge_human_and_judge(all_human, all_judge)
    print(f"\n  Merged: {len(merged)} samples")

    # --- Create or load split ---
    split_path = args.split_file or os.path.join(args.output_dir, "split.json")
    if os.path.exists(split_path):
        print(f"\n  Loading existing split from {split_path}")
        with open(split_path) as f:
            split = json.load(f)
        # Filter to only IDs present in merged
        for s in ["train", "dev", "test"]:
            split[s] = [sid for sid in split[s] if sid in merged]
    else:
        split = create_split(list(merged.keys()), seed=args.seed)
        with open(split_path, "w") as f:
            json.dump(split, f, indent=2)
        print(f"\n  Created split → {split_path}")

    split_sizes = {s: len(ids) for s, ids in split.items()}
    print(f"  Split: train={split_sizes['train']}, dev={split_sizes['dev']}, test={split_sizes['test']}")

    # Partition data
    parts = split_merged(merged, split)

    # Verify no leakage
    train_ids = set(split["train"])
    test_ids = set(split["test"])
    assert len(train_ids & test_ids) == 0, "DATA LEAKAGE: overlap between train and test!"

    # === ISOTONIC CALIBRATION ===
    print("\n" + "=" * 60)
    print("Route 1: Isotonic Calibration")
    print("=" * 60)

    # Fit on train only
    iso_cal = IsotonicCalibrator()
    iso_cal.fit(parts["train"])
    print("  Fitted on train set")

    # Evaluate on dev (informational)
    iso_dev_scores = {sid: iso_cal.transform(d["judge_mean"]) for sid, d in parts["dev"].items()}
    dev_metrics = compute_metrics(parts["dev"], iso_dev_scores)
    print("  Dev metrics:")
    for dim in DIMENSION_KEYS:
        d = dev_metrics.get(dim, {})
        if "note" in d:
            continue
        print(f"    {dim}: MAE {d['raw']['mae']:.3f} → {d['calibrated']['mae']:.3f}")

    # Evaluate on test (FINAL)
    iso_test_scores = {sid: iso_cal.transform(d["judge_mean"]) for sid, d in parts["test"].items()}
    test_metrics_iso = compute_metrics(parts["test"], iso_test_scores)
    print("\n  TEST metrics (FINAL):")
    for dim in DIMENSION_KEYS:
        d = test_metrics_iso.get(dim, {})
        if "note" in d:
            continue
        raw_mae = d["raw"]["mae"]
        cal_mae = d["calibrated"]["mae"]
        reduction = (raw_mae - cal_mae) / raw_mae * 100 if raw_mae > 0 else 0
        print(f"    {dim}: MAE {raw_mae:.3f} → {cal_mae:.3f} ({reduction:.1f}% ↓)  "
              f"ρ {d['raw']['spearman']:.3f} → {d['calibrated']['spearman']:.3f}")

    # Bootstrap CI
    print(f"\n  Running {args.n_bootstrap}× bootstrap for 95% CI...")
    ci_iso = bootstrap_ci(parts["test"], iso_cal, "isotonic",
                          n_bootstrap=args.n_bootstrap, seed=args.seed)
    for dim in DIMENSION_KEYS:
        mr = ci_iso[dim].get("mae_reduction_pct", {})
        print(f"    {dim}: MAE reduction = {mr.get('mean', '?')}% "
              f"[{mr.get('ci_lower', '?')}%, {mr.get('ci_upper', '?')}%]")

    # Save isotonic test output
    save_calibrated(parts["test"], iso_test_scores, "outputs/calibrated/calib_isotonic_test.jsonl")

    # === ORDINAL CALIBRATION ===
    print("\n" + "=" * 60)
    print("Route 2: Ordinal Calibration")
    print("=" * 60)
    test_metrics_ord = None
    ci_ord = None
    try:
        ord_cal = OrdinalCalibrator()
        ord_cal.fit(parts["train"])
        print("  Fitted on train set")

        ord_test_scores = {}
        for sid, data in parts["test"].items():
            ord_test_scores[sid] = ord_cal.transform(
                data["judge_mean"],
                judge_std=data["judge_std"],
                confidence=data["judge_confidence_mean"],
            )

        test_metrics_ord = compute_metrics(parts["test"], ord_test_scores)
        print("  TEST metrics:")
        for dim in DIMENSION_KEYS:
            d = test_metrics_ord.get(dim, {})
            if "note" in d:
                continue
            print(f"    {dim}: MAE {d['raw']['mae']:.3f} → {d['calibrated']['mae']:.3f}")

        ci_ord = bootstrap_ci(parts["test"], ord_cal, "ordinal",
                              n_bootstrap=args.n_bootstrap, seed=args.seed)

        save_calibrated(parts["test"], ord_test_scores, "outputs/calibrated/calib_ordinal_test.jsonl")
    except Exception as e:
        print(f"  Ordinal calibration failed: {e}")

    # === SAVE REPORT ===
    report = {
        "split": {s: len(ids) for s, ids in split.items()},
        "split_file": split_path,
        "iaa": {dim: iaa.get(dim, {}).get("weighted_kappa_linear", None) for dim in DIMENSION_KEYS},
        "isotonic": {
            "test_metrics": test_metrics_iso,
            "bootstrap_ci_95": ci_iso,
            "n_bootstrap": args.n_bootstrap,
        },
    }
    if test_metrics_ord:
        report["ordinal"] = {
            "test_metrics": test_metrics_ord,
            "bootstrap_ci_95": ci_ord,
            "n_bootstrap": args.n_bootstrap,
        }

    json_path = os.path.join(args.output_dir, "calibration_report_paper.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nJSON report → {json_path}")

    # Markdown
    md = generate_markdown(
        split_sizes, {}, test_metrics_iso, test_metrics_ord,
        ci_iso, ci_ord, args.n_bootstrap,
    )
    md_path = os.path.join(args.output_dir, "calibration_report_paper.md")
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Markdown → {md_path}")

    print("\n✅ Paper-grade calibration complete.")


if __name__ == "__main__":
    main()
