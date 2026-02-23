"""
Ablation B: Temperature and repeats sensitivity analysis.

Uses existing judge results (3 repeats at temp=0.3) to analyse:
  1. repeats=1 vs repeats=3: How much does averaging help stability & alignment?
  2. If API access available: temp=0 vs temp=0.3 vs temp=0.7 comparison

Outputs:
  - outputs/analysis/ablation_repeats.json
  - outputs/analysis/ablation_repeats.md

Usage:
    python experiments/run_ablation_repeats.py \
        --human outputs/labels/human/ratings_r1.csv,outputs/labels/human/ratings_r2.csv
"""
import sys, os, json, argparse
sys.path.insert(0, ".")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from collections import defaultdict
from scipy import stats as scipy_stats

from src.eval.rubric import DIMENSION_KEYS
from src.eval.human_labels_schema import load_human_labels_csv
from src.eval.llm_judge import load_judge_results
from src.eval.calibrate import merge_human_and_judge, IsotonicCalibrator


JUDGE_FILES = [
    "outputs/judge/gpt2_vanilla_judge.jsonl",
    "outputs/judge/gpt2_finetuned_judge.jsonl",
    "outputs/judge/empathy_chain_judge.jsonl",
]


def simulate_k_repeats(judge_results: list[dict], k: int, seed: int = 42) -> list[dict]:
    """Simulate using only k repeats by subsampling from existing repeats.

    For each (sample_id, model) group, select k repeats (deterministically).
    Returns a new list with synthetic judge results where scores are the mean of k repeats.
    """
    rng = np.random.RandomState(seed)

    by_group = defaultdict(list)
    for r in judge_results:
        if "scores" in r:
            key = f"{r['sample_id']}_{r.get('model', '')}"
            by_group[key].append(r)

    result = []
    for key, repeats in by_group.items():
        if len(repeats) < k:
            selected = repeats
        else:
            idx = rng.choice(len(repeats), size=k, replace=False)
            selected = [repeats[i] for i in idx]

        # Create a synthetic "averaged" record
        avg_scores = {}
        for dim in DIMENSION_KEYS:
            vals = [r["scores"][dim] for r in selected]
            avg_scores[dim] = float(np.mean(vals))

        synth = dict(selected[0])  # copy metadata from first
        synth["scores"] = avg_scores
        synth["overall"] = float(np.mean([r["overall"] for r in selected]))
        synth["confidence"] = float(np.mean([r.get("confidence", 0.5) for r in selected]))
        synth["n_repeats_used"] = k
        result.append(synth)

    return result


def compute_stability_at_k(judge_results: list[dict], k: int) -> dict:
    """Compute stability metrics when using k repeats."""
    by_group = defaultdict(list)
    for r in judge_results:
        if "scores" in r:
            key = f"{r['sample_id']}_{r.get('model', '')}"
            by_group[key].append(r)

    per_dim_stds = {dim: [] for dim in DIMENSION_KEYS}
    for key, repeats in by_group.items():
        # Use only first k repeats
        subset = repeats[:k]
        if len(subset) < 2:
            continue
        for dim in DIMENSION_KEYS:
            vals = [r["scores"][dim] for r in subset]
            per_dim_stds[dim].append(float(np.std(vals)))

    stability = {}
    for dim in DIMENSION_KEYS:
        s = per_dim_stds[dim]
        if s:
            stability[dim] = {
                "mean_std": round(float(np.mean(s)), 4),
                "median_std": round(float(np.median(s)), 4),
                "pct_exact_agree": round(sum(1 for x in s if x == 0) / len(s), 4),
                "pct_near_agree": round(
                    sum(1 for x in s if x <= 0.5) / len(s), 4
                ),
            }
    return stability


def compute_alignment_at_k(
    judge_results: list[dict],
    human_labels: list[dict],
    k: int,
    seed: int = 42,
) -> dict:
    """Compute human alignment when using only k repeats for judge averaging."""
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
        sp, _ = scipy_stats.spearmanr(h, j)
        alignment[dim] = {
            "n": len(h),
            "mae": round(float(np.mean(np.abs(j - h))), 4),
            "spearman": round(float(sp), 4),
            "bias": round(float(np.mean(j - h)), 4),
        }

    return alignment


def compute_calibrated_at_k(
    judge_results: list[dict],
    human_labels: list[dict],
    k: int,
    seed: int = 42,
) -> dict:
    """Fit and evaluate calibration using only k repeats (leave-one-out style)."""
    simulated = simulate_k_repeats(judge_results, k, seed=seed)
    merged = merge_human_and_judge(human_labels, simulated)

    if len(merged) < 20:
        return {"note": "insufficient merged data"}

    # Simple 80/20 split for quick ablation
    sids = sorted(merged.keys())
    n_train = int(len(sids) * 0.8)
    train_ids, test_ids = set(sids[:n_train]), set(sids[n_train:])

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


def generate_markdown(results: dict) -> str:
    lines = [
        "# Ablation B: Temperature / Repeats Sensitivity",
        "",
        "## Effect of Number of Repeats (k)",
        "",
        "### Stability (mean σ across repeats)",
        "",
        "| Dim | k=1 | k=2 | k=3 |",
        "|-----|:---:|:---:|:---:|",
    ]
    for dim in DIMENSION_KEYS:
        row = f"| {dim} |"
        for k in [1, 2, 3]:
            s = results.get(f"k={k}", {}).get("stability", {}).get(dim, {})
            row += f" — |" if k == 1 else f" {s.get('mean_std', 'N/A')} |"
        lines.append(row)

    lines.extend([
        "",
        "### Human Alignment (MAE / Spearman)",
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
        "### Calibrated MAE (isotonic, 80/20 split)",
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
        "1. **How many repeats are needed?** Compare k=1 vs k=3 stability and alignment.",
        "2. **Is k=2 a good cost-performance trade-off?**",
        "3. **Does calibration compensate for fewer repeats?**",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human", default="outputs/human_annotation/simulated_human_labels.csv")
    parser.add_argument("--output_dir", default="outputs/analysis")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    all_human = []
    for p in args.human.split(","):
        p = p.strip()
        if os.path.exists(p):
            all_human.extend(load_human_labels_csv(p))
    print(f"Human labels: {len(all_human)}")

    all_judge = []
    for path in JUDGE_FILES:
        if os.path.exists(path):
            all_judge.extend(load_judge_results(path))
    print(f"Judge results: {len(all_judge)}")

    # Analyse different k values
    results = {}
    for k in [1, 2, 3]:
        print(f"\n{'='*60}")
        print(f"Analysing k={k} repeats")
        print(f"{'='*60}")

        key = f"k={k}"
        results[key] = {}

        # Stability (only meaningful for k >= 2 with raw repeats)
        if k >= 2:
            stab = compute_stability_at_k(all_judge, k)
            results[key]["stability"] = stab
            for dim in DIMENSION_KEYS:
                s = stab.get(dim, {})
                print(f"  {dim}: mean_std={s.get('mean_std', 'N/A')}, "
                      f"exact_agree={s.get('pct_exact_agree', 'N/A')}")

        # Alignment with human
        align = compute_alignment_at_k(all_judge, all_human, k)
        results[key]["alignment"] = align
        for dim in DIMENSION_KEYS:
            a = align.get(dim, {})
            if "note" not in a:
                print(f"  {dim}: MAE={a.get('mae', 'N/A')}, ρ={a.get('spearman', 'N/A')}")

        # Calibrated performance
        cal = compute_calibrated_at_k(all_judge, all_human, k)
        results[key]["calibrated"] = cal

    # Save
    json_path = os.path.join(args.output_dir, "ablation_repeats.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nJSON → {json_path}")

    md = generate_markdown(results)
    md_path = os.path.join(args.output_dir, "ablation_repeats.md")
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Markdown → {md_path}")


if __name__ == "__main__":
    main()
