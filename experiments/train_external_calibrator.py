"""
Step 3: Train external-human-anchored calibrator.

Merges external human ratings with LLM judge scores on the external dataset,
trains isotonic (primary) + ordinal (comparison) calibrators, and evaluates
with bootstrap 95% CI.

This replaces the "simulated human" labels used in run_calibration_paper.py
with real external human labels from a public dataset.

Usage:
    python experiments/train_external_calibrator.py \\
        --external_data data/external/unified.jsonl \\
        --judge_results outputs/judge_external/my_dataset_deepseek_chat.jsonl \\
        --dataset my_dataset \\
        --n_bootstrap 1000

Outputs:
    checkpoints/calibrators/<dataset>_<judge>_isotonic.pkl
    checkpoints/calibrators/<dataset>_<judge>_ordinal.pkl
    outputs/analysis/external_calibration_report.json
    outputs/analysis/external_calibration_report.md
    outputs/analysis/external_split.json
"""

import sys, os, json, argparse, pickle
sys.path.insert(0, ".")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from collections import defaultdict
from pathlib import Path
from scipy import stats as scipy_stats

from src.eval.rubric import DIMENSION_KEYS
from src.eval.llm_judge import load_judge_results
from src.eval.calibrate import (
    merge_human_and_judge,
    IsotonicCalibrator,
    OrdinalCalibrator,
    compute_ece,
    save_calibrated,
)
from src.data.external_loader import load_external, convert_to_human_labels


# ===================================================================
# Split utility (same logic as run_calibration_paper.py)
# ===================================================================

def create_split(
    sample_ids: list[str],
    train_frac: float = 0.6,
    dev_frac: float = 0.2,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Deterministic split by sample_id."""
    rng = np.random.RandomState(seed)
    ids_shuffled = sorted(sample_ids)
    rng.shuffle(ids_shuffled)

    n = len(ids_shuffled)
    n_train = int(n * train_frac)
    n_dev = int(n * (train_frac + dev_frac))

    return {
        "train": ids_shuffled[:n_train],
        "dev": ids_shuffled[n_train:n_dev],
        "test": ids_shuffled[n_dev:],
    }


def split_merged(merged, split):
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
# Metrics
# ===================================================================

def compute_metrics(merged_subset, calibrated_scores=None):
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
            sp_res = scipy_stats.spearmanr(y_true, y_pred)
            kt_res = scipy_stats.kendalltau(y_true, y_pred)
            return {
                "mae": round(float(np.mean(np.abs(errors))), 4),
                "rmse": round(float(np.sqrt(np.mean(errors ** 2))), 4),
                "bias": round(float(np.mean(errors)), 4),
                "spearman": round(float(sp_res[0]), 4),
                "kendall": round(float(kt_res[0]), 4),
            }

        results[dim] = {
            "n": len(h_vals),
            "raw": _m(h, jr),
            "calibrated": _m(h, jc),
        }
    return results


# ===================================================================
# Bootstrap CI
# ===================================================================

def bootstrap_ci(merged_subset, calibrator, method, n_bootstrap=1000, ci_level=0.95, seed=42):
    rng = np.random.RandomState(seed)
    sids = list(merged_subset.keys())
    n = len(sids)

    boot_results = {
        dim: {"raw_mae": [], "cal_mae": [], "raw_spearman": [], "cal_spearman": [], "mae_reduction_pct": []}
        for dim in DIMENSION_KEYS
    }

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_sids = [sids[i] for i in idx]

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

            sp_raw = scipy_stats.spearmanr(h, jr) if len(h) > 2 else None
            sp_cal = scipy_stats.spearmanr(h, jc) if len(h) > 2 else None

            boot_results[dim]["raw_mae"].append(raw_mae)
            boot_results[dim]["cal_mae"].append(cal_mae)
            boot_results[dim]["raw_spearman"].append(float(sp_raw[0]) if sp_raw else 0.0)
            boot_results[dim]["cal_spearman"].append(float(sp_cal[0]) if sp_cal else 0.0)
            if raw_mae > 0:
                boot_results[dim]["mae_reduction_pct"].append(
                    (raw_mae - cal_mae) / raw_mae * 100
                )

    alpha = 1 - ci_level
    ci = {}
    for dim in DIMENSION_KEYS:
        br = boot_results[dim]
        dim_ci = {}
        for key in ["raw_mae", "cal_mae", "raw_spearman", "cal_spearman", "mae_reduction_pct"]:
            vals = br.get(key, [])
            if len(vals) < 50:
                dim_ci[key] = {"mean": None, "ci_lower": None, "ci_upper": None}
                continue
            vals = np.array(vals)
            dim_ci[key] = {
                "mean": round(float(np.mean(vals)), 4),
                "ci_lower": round(float(np.percentile(vals, alpha / 2 * 100)), 4),
                "ci_upper": round(float(np.percentile(vals, (1 - alpha / 2) * 100)), 4),
            }
        ci[dim] = dim_ci
    return ci


# ===================================================================
# Markdown report
# ===================================================================

def generate_markdown(dataset, judge_model, split_sizes, test_metrics_iso,
                      test_metrics_ord, ci_iso, ci_ord, n_bootstrap):
    lines = [
        "# External Human-Anchored Calibration Report",
        "",
        f"**Dataset**: {dataset}",
        f"**Judge model**: {judge_model}",
        "",
        "## Data Split",
        f"- Train: {split_sizes['train']} samples (60%)",
        f"- Dev: {split_sizes['dev']} samples (20%)",
        f"- Test: {split_sizes['test']} samples (20%)",
        "",
        "## Isotonic Calibration (Route 1 — Primary)",
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

    lines.extend([
        "",
        "## Methodology",
        "",
        "This calibration uses **external human annotations** from a public dataset,",
        "rather than our own human labeling. The judge is evaluated and calibrated",
        "against these independent human ratings, providing an unbiased anchor for",
        "score alignment.",
        "",
        "The calibrated judge can then be applied to our own model outputs to produce",
        "human-anchored quality scores.",
    ])

    return "\n".join(lines)


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Train external-human-anchored calibrator")
    parser.add_argument("--external_data", type=str, required=True,
                        help="Path to unified external JSONL (from external_loader)")
    parser.add_argument("--judge_results", type=str, required=True,
                        help="Path to external judge JSONL (from run_external_judge)")
    parser.add_argument("--dataset", type=str, default="external",
                        help="Dataset name tag")
    parser.add_argument("--judge_model", type=str, default="deepseek_chat",
                        help="Judge model tag for filenames")
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="outputs/analysis")
    parser.add_argument("--calibrator_dir", default="checkpoints/calibrators")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.calibrator_dir, exist_ok=True)
    os.makedirs("outputs/calibrated", exist_ok=True)

    # ─── Load external data ───
    print("=" * 60)
    print("Loading data")
    print("=" * 60)

    # Load unified external records
    ext_records = []
    with open(args.external_data, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ext_records.append(json.loads(line))
    print(f"  External records: {len(ext_records)}")

    # Convert to human-labels format
    ext_human_labels = convert_to_human_labels(ext_records)
    print(f"  External human labels: {len(ext_human_labels)}")

    # Load judge results
    judge_results = load_judge_results(args.judge_results)
    print(f"  Judge results: {len(judge_results)}")

    # ─── Merge ───
    merged = merge_human_and_judge(ext_human_labels, judge_results)
    print(f"  Merged: {len(merged)} samples")

    if len(merged) < 30:
        print("ERROR: Too few merged samples. Check that item_id / sample_id match.")
        return

    # ─── Create split ───
    split_path = os.path.join(args.output_dir, "external_split.json")
    split = create_split(list(merged.keys()), seed=args.seed)
    with open(split_path, "w") as f:
        json.dump(split, f, indent=2)
    print(f"\n  Split → {split_path}")

    split_sizes = {s: len(ids) for s, ids in split.items()}
    print(f"  train={split_sizes['train']}, dev={split_sizes['dev']}, test={split_sizes['test']}")

    parts = split_merged(merged, split)

    # Verify no leakage
    assert len(set(split["train"]) & set(split["test"])) == 0, "DATA LEAKAGE!"

    # ═══ ISOTONIC CALIBRATION ═══
    print(f"\n{'=' * 60}")
    print("Route 1: Isotonic Calibration")
    print("=" * 60)

    iso_cal = IsotonicCalibrator()
    iso_cal.fit(parts["train"])
    print("  Fitted on train set")

    # Dev (informational)
    iso_dev_scores = {sid: iso_cal.transform(d["judge_mean"]) for sid, d in parts["dev"].items()}
    dev_metrics = compute_metrics(parts["dev"], iso_dev_scores)
    print("  Dev metrics:")
    for dim in DIMENSION_KEYS:
        d = dev_metrics.get(dim, {})
        if "note" in d:
            continue
        print(f"    {dim}: MAE {d['raw']['mae']:.3f} → {d['calibrated']['mae']:.3f}")

    # Test (FINAL)
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

    # Save isotonic calibrator
    iso_pkl_path = os.path.join(
        args.calibrator_dir, f"{args.dataset}_{args.judge_model}_isotonic.pkl"
    )
    with open(iso_pkl_path, "wb") as f:
        pickle.dump(iso_cal, f)
    print(f"\n  Calibrator → {iso_pkl_path}")

    # Save calibrated test outputs
    save_calibrated(
        parts["test"], iso_test_scores,
        f"outputs/calibrated/external_{args.dataset}_isotonic_test.jsonl"
    )

    # ═══ ORDINAL CALIBRATION ═══
    print(f"\n{'=' * 60}")
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

        # Save ordinal calibrator
        ord_pkl_path = os.path.join(
            args.calibrator_dir, f"{args.dataset}_{args.judge_model}_ordinal.pkl"
        )
        with open(ord_pkl_path, "wb") as f:
            pickle.dump(ord_cal, f)
        print(f"  Calibrator → {ord_pkl_path}")

        save_calibrated(
            parts["test"], ord_test_scores,
            f"outputs/calibrated/external_{args.dataset}_ordinal_test.jsonl"
        )
    except Exception as e:
        print(f"  Ordinal calibration failed: {e}")

    # ═══ SAVE REPORT ═══
    report = {
        "dataset": args.dataset,
        "judge_model": args.judge_model,
        "n_external_records": len(ext_records),
        "n_merged": len(merged),
        "split": split_sizes,
        "split_file": split_path,
        "isotonic": {
            "calibrator_path": iso_pkl_path,
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

    json_path = os.path.join(args.output_dir, "external_calibration_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nJSON report → {json_path}")

    md = generate_markdown(
        args.dataset, args.judge_model, split_sizes,
        test_metrics_iso, test_metrics_ord, ci_iso, ci_ord, args.n_bootstrap,
    )
    md_path = os.path.join(args.output_dir, "external_calibration_report.md")
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Markdown → {md_path}")

    print("\n✅ External calibrator training complete.")


if __name__ == "__main__":
    main()
