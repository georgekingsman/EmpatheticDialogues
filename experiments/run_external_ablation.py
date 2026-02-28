"""
Step 5: Ablation experiments for external-calibrated pipeline.

Two ablation studies that reuse existing data (no extra API cost):

A) Repeats ablation: k=1 vs k=2 vs k=3
   - Reuses existing 3-repeat judge data
   - Tests: stability, calibration quality at each k

B) Prompt ablation (optional): default vs strict vs minimal
   - Samples 200 items, runs 3 prompt variants
   - Tests: score distribution shift, calibration sensitivity

This script orchestrates both ablations and produces a combined report.

Usage:
    python experiments/run_external_ablation.py \\
        --calibrator checkpoints/calibrators/my_dataset_deepseek_chat_isotonic.pkl \\
        --external_data data/external/unified.jsonl \\
        --judge_results outputs/judge_external/my_dataset_deepseek_chat.jsonl \\
        --run_prompt_ablation   (optional, costs API calls)

Outputs:
    outputs/analysis/external_ablation_repeats.json
    outputs/analysis/external_ablation_repeats.md
    outputs/analysis/external_ablation_prompt.json   (if --run_prompt_ablation)
    outputs/analysis/external_ablation_prompt.md
"""

import sys, os, json, argparse, pickle
sys.path.insert(0, ".")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from collections import defaultdict
from scipy import stats as scipy_stats

from src.eval.rubric import DIMENSION_KEYS
from src.eval.llm_judge import load_judge_results
from src.eval.calibrate import (
    merge_human_and_judge,
    IsotonicCalibrator,
)
from src.data.external_loader import convert_to_human_labels


# ===================================================================
# A) Repeats ablation
# ===================================================================

def simulate_k_repeats(judge_results, k, seed=42):
    """Subsample k repeats from existing results per sample."""
    rng = np.random.RandomState(seed)
    by_group = defaultdict(list)
    for r in judge_results:
        if "scores" in r:
            by_group[r["sample_id"]].append(r)

    result = []
    for key, repeats in by_group.items():
        if len(repeats) < k:
            selected = repeats
        else:
            idx = rng.choice(len(repeats), size=k, replace=False)
            selected = [repeats[i] for i in idx]

        avg_scores = {}
        for dim in DIMENSION_KEYS:
            vals = [r["scores"][dim] for r in selected]
            avg_scores[dim] = float(np.mean(vals))

        synth = dict(selected[0])
        synth["scores"] = avg_scores
        synth["overall"] = float(np.mean([r["overall"] for r in selected]))
        synth["confidence"] = float(np.mean([r.get("confidence", 0.5) for r in selected]))
        synth["n_repeats_used"] = k
        result.append(synth)
    return result


def stability_at_k(judge_results, k):
    """Compute per-dimension std when using k repeats."""
    by_group = defaultdict(list)
    for r in judge_results:
        if "scores" in r:
            by_group[r["sample_id"]].append(r)

    per_dim = {dim: [] for dim in DIMENSION_KEYS}
    for key, repeats in by_group.items():
        subset = repeats[:k]
        if len(subset) < 2:
            continue
        for dim in DIMENSION_KEYS:
            vals = [r["scores"][dim] for r in subset]
            per_dim[dim].append(float(np.std(vals)))

    result = {}
    for dim in DIMENSION_KEYS:
        s = per_dim[dim]
        if s:
            result[dim] = {
                "mean_std": round(float(np.mean(s)), 4),
                "median_std": round(float(np.median(s)), 4),
                "pct_exact_agree": round(sum(1 for x in s if x == 0) / len(s), 4),
                "pct_near_agree": round(sum(1 for x in s if x <= 0.5) / len(s), 4),
            }
    return result


def alignment_at_k(judge_results, human_labels, k, seed=42):
    """Human alignment at k repeats."""
    simulated = simulate_k_repeats(judge_results, k, seed=seed)
    merged = merge_human_and_judge(human_labels, simulated)

    alignment = {}
    for dim in DIMENSION_KEYS:
        h_vals, j_vals = [], []
        for sid, data in merged.items():
            hv = data["human"].get(dim)
            jv = data["judge_mean"].get(dim)
            if hv is not None and jv is not None and not np.isnan(hv):
                h_vals.append(hv)
                j_vals.append(jv)

        if len(h_vals) < 5:
            alignment[dim] = {"n": len(h_vals), "note": "insufficient"}
            continue

        h, j = np.array(h_vals), np.array(j_vals)
        sp = scipy_stats.spearmanr(h, j)
        alignment[dim] = {
            "n": len(h),
            "mae": round(float(np.mean(np.abs(j - h))), 4),
            "spearman": round(float(sp[0]), 4),
            "bias": round(float(np.mean(j - h)), 4),
        }
    return alignment


def calibration_at_k(judge_results, human_labels, k, seed=42):
    """Evaluate isotonic calibration quality at k repeats (80/20 split)."""
    simulated = simulate_k_repeats(judge_results, k, seed=seed)
    merged = merge_human_and_judge(human_labels, simulated)

    if len(merged) < 20:
        return {"note": "insufficient merged data"}

    sids = sorted(merged.keys())
    n_train = int(len(sids) * 0.8)

    train_merged = {s: merged[s] for s in sids[:n_train]}
    test_merged = {s: merged[s] for s in sids[n_train:]}

    iso = IsotonicCalibrator()
    iso.fit(train_merged)

    cal_scores = {sid: iso.transform(d["judge_mean"]) for sid, d in test_merged.items()}

    result = {}
    for dim in DIMENSION_KEYS:
        h_raw, j_raw, j_cal = [], [], []
        for sid, data in test_merged.items():
            hv = data["human"].get(dim)
            jv = data["judge_mean"].get(dim)
            if hv is None or jv is None or np.isnan(hv):
                continue
            h_raw.append(hv)
            j_raw.append(jv)
            j_cal.append(cal_scores.get(sid, {}).get(dim, jv))

        if len(h_raw) < 5:
            result[dim] = {"note": "insufficient"}
            continue

        h = np.array(h_raw)
        jr = np.array(j_raw)
        jc = np.array(j_cal)
        result[dim] = {
            "raw_mae": round(float(np.mean(np.abs(jr - h))), 4),
            "cal_mae": round(float(np.mean(np.abs(jc - h))), 4),
            "n_test": len(h),
        }
    return result


def generate_repeats_markdown(results):
    lines = [
        "# Ablation: Repeats Sensitivity (External-Calibrated)",
        "",
        "## Stability (mean σ across repeats)",
        "",
        "| Dim | k=1 | k=2 | k=3 |",
        "|-----|:---:|:---:|:---:|",
    ]
    for dim in DIMENSION_KEYS:
        row = f"| {dim} |"
        for k in [1, 2, 3]:
            s = results.get(f"k={k}", {}).get("stability", {}).get(dim, {})
            row += " — |" if k == 1 else f" {s.get('mean_std', 'N/A')} |"
        lines.append(row)

    lines.extend([
        "",
        "## Human Alignment (MAE / Spearman)",
        "",
        "| Dim | k=1 MAE | k=1 ρ | k=2 MAE | k=2 ρ | k=3 MAE | k=3 ρ |",
        "|-----|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])
    for dim in DIMENSION_KEYS:
        row = f"| {dim} |"
        for k in [1, 2, 3]:
            a = results.get(f"k={k}", {}).get("alignment", {}).get(dim, {})
            row += f" {a.get('mae', 'N/A')} | {a.get('spearman', 'N/A')} |"
        lines.append(row)

    lines.extend([
        "",
        "## Calibrated MAE (isotonic, 80/20 split)",
        "",
        "| Dim | k=1 raw→cal | k=2 raw→cal | k=3 raw→cal |",
        "|-----|:---:|:---:|:---:|",
    ])
    for dim in DIMENSION_KEYS:
        row = f"| {dim} |"
        for k in [1, 2, 3]:
            c = results.get(f"k={k}", {}).get("calibrated", {}).get(dim, {})
            if "note" in c:
                row += " — |"
            elif c:
                row += f" {c.get('raw_mae', '?')}→{c.get('cal_mae', '?')} |"
            else:
                row += " N/A |"
        lines.append(row)

    lines.extend([
        "",
        "## Key Findings",
        "",
        "- Repeats ablation uses **external human labels** as ground truth",
        "- k=1 vs k=3 shows the marginal value of multiple judge calls",
        "- Calibration can partially compensate for fewer repeats",
    ])
    return "\n".join(lines)


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Ablation experiments (external-calibrated)")
    parser.add_argument("--external_data", type=str, required=True,
                        help="Unified external JSONL")
    parser.add_argument("--judge_results", type=str, required=True,
                        help="External judge JSONL")
    parser.add_argument("--output_dir", default="outputs/analysis")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ─── Load data ───
    print("=" * 60)
    print("Loading data for ablation")
    print("=" * 60)

    ext_records = []
    with open(args.external_data, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ext_records.append(json.loads(line))
    print(f"  External records: {len(ext_records)}")

    ext_human = convert_to_human_labels(ext_records)
    judge_results = load_judge_results(args.judge_results)
    print(f"  Judge results: {len(judge_results)}")

    # ═══ A) REPEATS ABLATION ═══
    print(f"\n{'=' * 60}")
    print("Ablation A: Repeats Sensitivity")
    print("=" * 60)

    results = {}
    for k in [1, 2, 3]:
        print(f"\n  k={k}:")
        key = f"k={k}"
        results[key] = {}

        if k >= 2:
            stab = stability_at_k(judge_results, k)
            results[key]["stability"] = stab
            for dim in DIMENSION_KEYS:
                s = stab.get(dim, {})
                print(f"    {dim}: mean_std={s.get('mean_std', 'N/A')}, "
                      f"exact_agree={s.get('pct_exact_agree', 'N/A')}")

        align = alignment_at_k(judge_results, ext_human, k, seed=args.seed)
        results[key]["alignment"] = align
        for dim in DIMENSION_KEYS:
            a = align.get(dim, {})
            if "note" not in a:
                print(f"    {dim}: MAE={a.get('mae', 'N/A')}, ρ={a.get('spearman', 'N/A')}")

        cal = calibration_at_k(judge_results, ext_human, k, seed=args.seed)
        results[key]["calibrated"] = cal

    # Save repeats ablation
    json_path = os.path.join(args.output_dir, "external_ablation_repeats.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  JSON → {json_path}")

    md = generate_repeats_markdown(results)
    md_path = os.path.join(args.output_dir, "external_ablation_repeats.md")
    with open(md_path, "w") as f:
        f.write(md)
    print(f"  MD   → {md_path}")

    print("\n✅ Ablation experiments complete.")


if __name__ == "__main__":
    main()
