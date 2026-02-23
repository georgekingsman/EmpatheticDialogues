"""
Judge ↔ Human systematic alignment analysis.

Produces:
  1. Per-dimension: Spearman, Kendall, MAE, RMSE, bias (mean(judge-human))
  2. Error decomposition by:
     a) Response length (short/medium/long)
     b) Safety-relevant vs not (keyword detection)
     c) Model group (vanilla / finetuned / empathy)
  3. Top-N error cases (high disagreement) with full context

Outputs:
  - outputs/analysis/judge_vs_human_raw.json     (structured metrics)
  - outputs/analysis/judge_vs_human_raw.md        (markdown narrative + tables)
  - outputs/analysis/error_cases.json             (flagged high-disagreement samples)

Usage:
    python experiments/judge_vs_human_analysis.py \
        --human outputs/labels/human/ratings_r1.csv,outputs/labels/human/ratings_r2.csv \
        --generations outputs/generations/gpt2_vanilla.jsonl,outputs/generations/gpt2_finetuned.jsonl,outputs/generations/empathy_chain.jsonl
"""
import sys, os, json, csv, argparse, re
sys.path.insert(0, ".")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from collections import defaultdict
from scipy import stats as scipy_stats
from src.eval.rubric import DIMENSION_KEYS
from src.eval.human_labels_schema import load_human_labels_csv
from src.eval.llm_judge import load_judge_results
from src.eval.calibrate import merge_human_and_judge

# Safety-related keywords for bucketing
SAFETY_KEYWORDS = re.compile(
    r"suicid|self.harm|kill\s+(my|your)self|medication|overdose|"
    r"hurt\s+(my|your)self|crisis|emergency|die|death|abuse|"
    r"violence|panic\s+attack|danger|hospital",
    re.IGNORECASE,
)

JUDGE_FILES = {
    "gpt2_vanilla":   "outputs/judge/gpt2_vanilla_judge.jsonl",
    "gpt2_finetuned": "outputs/judge/gpt2_finetuned_judge.jsonl",
    "empathy_chain":  "outputs/judge/empathy_chain_judge.jsonl",
}

GENERATION_FILES = {
    "gpt2_vanilla":   "outputs/generations/gpt2_vanilla.jsonl",
    "gpt2_finetuned": "outputs/generations/gpt2_finetuned.jsonl",
    "empathy_chain":  "outputs/generations/empathy_chain.jsonl",
}


# ===================================================================
# Core alignment metrics
# ===================================================================

def compute_alignment_metrics(merged: dict[str, dict]) -> dict:
    """Per-dimension: Spearman, Kendall, MAE, RMSE, bias, std(err)."""
    results = {}
    for dim in DIMENSION_KEYS:
        h_vals, j_vals = [], []
        for sid, data in merged.items():
            hv = data["human"].get(dim)
            jv = data["judge_mean"].get(dim)
            if hv is not None and jv is not None and not np.isnan(hv):
                h_vals.append(hv)
                j_vals.append(jv)

        if len(h_vals) < 5:
            results[dim] = {"n": len(h_vals), "note": "insufficient data"}
            continue

        h, j = np.array(h_vals), np.array(j_vals)
        errors = j - h

        sp_res = scipy_stats.spearmanr(h, j)
        kt_res = scipy_stats.kendalltau(h, j)

        results[dim] = {
            "n": len(h),
            "spearman": round(float(sp_res[0]), 4),       # type: ignore[arg-type]
            "spearman_p": round(float(sp_res[1]), 6),     # type: ignore[arg-type]
            "kendall": round(float(kt_res[0]), 4),        # type: ignore[arg-type]
            "kendall_p": round(float(kt_res[1]), 6),      # type: ignore[arg-type]
            "mae": round(float(np.mean(np.abs(errors))), 4),
            "rmse": round(float(np.sqrt(np.mean(errors ** 2))), 4),
            "bias": round(float(np.mean(errors)), 4),  # positive = judge > human
            "bias_std": round(float(np.std(errors)), 4),
            "human_mean": round(float(np.mean(h)), 3),
            "judge_mean": round(float(np.mean(j)), 3),
        }

    return results


# ===================================================================
# Error decomposition
# ===================================================================

def _categorise_response_length(gen: dict) -> str:
    length = len(gen.get("response", ""))
    if length < 150:
        return "short (<150 chars)"
    elif length < 400:
        return "medium (150-400)"
    else:
        return "long (>400 chars)"


def _categorise_safety(gen: dict) -> str:
    text = gen.get("user_statement", "") + " " + gen.get("response", "")
    return "safety-relevant" if SAFETY_KEYWORDS.search(text) else "non-safety"


def _categorise_model(gen: dict) -> str:
    return gen.get("model", "unknown")


def decompose_errors(
    merged: dict[str, dict],
    gen_by_id: dict[str, dict],
    category_fn,
    category_name: str,
) -> dict:
    """Decompose judge-human errors by a bucketing function."""

    buckets: dict[str, dict[str, list]] = defaultdict(lambda: {d: [] for d in DIMENSION_KEYS})
    for sid, data in merged.items():
        gen = gen_by_id.get(sid)
        if gen is None:
            continue
        cat = category_fn(gen)
        for dim in DIMENSION_KEYS:
            hv = data["human"].get(dim)
            jv = data["judge_mean"].get(dim)
            if hv is not None and jv is not None and not np.isnan(hv):
                buckets[cat][dim].append(jv - hv)

    result = {}
    for cat, dims in sorted(buckets.items()):
        cat_res = {}
        for dim in DIMENSION_KEYS:
            errs = np.array(dims[dim])
            if len(errs) == 0:
                cat_res[dim] = {"n": 0}
                continue
            cat_res[dim] = {
                "n": len(errs),
                "mae": round(float(np.mean(np.abs(errs))), 4),
                "bias": round(float(np.mean(errs)), 4),
                "rmse": round(float(np.sqrt(np.mean(errs ** 2))), 4),
            }
        result[cat] = cat_res

    return {category_name: result}


# ===================================================================
# Error case extraction
# ===================================================================

def extract_error_cases(
    merged: dict[str, dict],
    gen_by_id: dict[str, dict],
    n_cases: int = 20,
) -> list[dict]:
    """Extract the top-N highest-disagreement samples with full context.

    Selects cases where |judge_overall - human_overall| is largest.
    Includes both 'judge too high' and 'judge too low' cases.
    """
    scored = []
    for sid, data in merged.items():
        h_overall = np.mean([data["human"].get(d, 0) for d in DIMENSION_KEYS])
        j_overall = np.mean([data["judge_mean"].get(d, 0) for d in DIMENSION_KEYS])
        diff = j_overall - h_overall
        scored.append((sid, abs(diff), diff))

    # Sort by |diff| desc
    scored.sort(key=lambda x: x[1], reverse=True)

    cases = []
    for sid, abs_diff, signed_diff in scored[:n_cases]:
        gen = gen_by_id.get(sid, {})
        data = merged[sid]
        case = {
            "sample_id": sid,
            "model": gen.get("model", "unknown"),
            "user_statement": gen.get("user_statement", "")[:300],
            "response": gen.get("response", "")[:500],
            "human_scores": data["human"],
            "judge_scores": data["judge_mean"],
            "judge_std": data["judge_std"],
            "signed_diff": round(signed_diff, 3),
            "direction": "judge_too_high" if signed_diff > 0 else "judge_too_low",
            "per_dim_diff": {
                dim: round(data["judge_mean"].get(dim, 0) - data["human"].get(dim, 0), 3)
                for dim in DIMENSION_KEYS
            },
        }
        cases.append(case)

    return cases


# ===================================================================
# Markdown report
# ===================================================================

def generate_markdown(
    alignment: dict,
    decompositions: dict,
    error_cases: list[dict],
    n_samples: int,
) -> str:
    lines = [
        "# Judge ↔ Human Alignment Analysis (Pre-Calibration)",
        "",
        f"**Matched samples**: {n_samples}",
        "",
        "## 1. Overall Alignment Metrics",
        "",
        "| Dimension | Spearman | Kendall | MAE | RMSE | Bias (J−H) | n |",
        "|-----------|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for dim in DIMENSION_KEYS:
        a = alignment.get(dim, {})
        if "note" in a:
            lines.append(f"| {dim} | — | — | — | — | — | {a.get('n',0)} |")
            continue
        lines.append(
            f"| {dim} | {a['spearman']:.3f} | {a['kendall']:.3f} | "
            f"{a['mae']:.3f} | {a['rmse']:.3f} | {a['bias']:+.3f} | {a['n']} |"
        )

    lines.extend(["", "> Bias > 0 → judge scores higher than human; Bias < 0 → judge harsher.", ""])

    # Decompositions
    for dec_name, dec_data in decompositions.items():
        lines.extend([
            f"## 2. Error Decomposition: {dec_name}",
            "",
        ])
        # Build table
        cats = sorted(dec_data.keys())
        lines.append("| Bucket | Dim | n | MAE | Bias | RMSE |")
        lines.append("|--------|-----|--:|:---:|:---:|:---:|")
        for cat in cats:
            for dim in DIMENSION_KEYS:
                d = dec_data[cat].get(dim, {})
                if d.get("n", 0) == 0:
                    continue
                lines.append(
                    f"| {cat} | {dim} | {d['n']} | {d['mae']:.3f} | {d['bias']:+.3f} | {d['rmse']:.3f} |"
                )
        lines.append("")

    # Error cases
    lines.extend([
        "## 3. Top Error Cases",
        "",
        f"Showing top {len(error_cases)} highest-disagreement samples.",
        "",
    ])
    for i, c in enumerate(error_cases[:10]):
        lines.extend([
            f"### Case {i+1} ({c['direction']}, Δ = {c['signed_diff']:+.2f})",
            f"- **Model**: {c['model']}",
            f"- **User**: {c['user_statement'][:200]}...",
            f"- **Response**: {c['response'][:200]}...",
            f"- **Human**: {c['human_scores']}",
            f"- **Judge**: {c['judge_scores']}",
            f"- **Per-dim Δ**: {c['per_dim_diff']}",
            "",
        ])

    return "\n".join(lines)


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Judge ↔ Human alignment analysis")
    parser.add_argument(
        "--human",
        default="outputs/human_annotation/simulated_human_labels.csv",
        help="Comma-separated paths to human label CSVs (will be concatenated)",
    )
    parser.add_argument("--output_dir", default="outputs/analysis")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load human labels ---
    all_human = []
    for p in args.human.split(","):
        p = p.strip()
        if os.path.exists(p):
            h = load_human_labels_csv(p)
            all_human.extend(h)
            print(f"  Loaded {len(h)} human labels from {p}")
    print(f"  Total human labels: {len(all_human)}")

    # --- Load judge results ---
    all_judge = []
    for tag, path in JUDGE_FILES.items():
        if os.path.exists(path):
            results = load_judge_results(path)
            all_judge.extend(results)
            print(f"  Loaded {len(results)} judge results from {tag}")
    print(f"  Total judge results: {len(all_judge)}")

    # --- Load generations (for metadata) ---
    all_gens = []
    for tag, path in GENERATION_FILES.items():
        if os.path.exists(path):
            with open(path) as f:
                gens = [json.loads(l) for l in f]
                all_gens.extend(gens)
    gen_by_id = {g["id"]: g for g in all_gens}
    print(f"  Loaded {len(all_gens)} generations")

    # --- Merge ---
    merged = merge_human_and_judge(all_human, all_judge)
    print(f"\n  Merged: {len(merged)} common samples")

    # --- Alignment metrics ---
    print("\n" + "=" * 60)
    print("Per-Dimension Alignment")
    print("=" * 60)
    alignment = compute_alignment_metrics(merged)
    for dim in DIMENSION_KEYS:
        a = alignment.get(dim, {})
        if "note" in a:
            print(f"  {dim}: {a['note']}")
        else:
            print(f"  {dim:15s}: Spearman={a['spearman']:.3f}  "
                  f"MAE={a['mae']:.3f}  bias={a['bias']:+.3f}")

    # --- Error decomposition ---
    print("\n" + "=" * 60)
    print("Error Decomposition")
    print("=" * 60)
    decompositions = {}
    for name, fn in [
        ("Response Length", _categorise_response_length),
        ("Safety Relevance", _categorise_safety),
        ("Model Group", _categorise_model),
    ]:
        dec = decompose_errors(merged, gen_by_id, fn, name)
        decompositions.update(dec)
        print(f"\n  {name}:")
        for cat, dims in dec[name].items():
            n = dims.get(DIMENSION_KEYS[0], {}).get("n", 0)
            avg_mae = np.mean([
                dims.get(d, {}).get("mae", 0) for d in DIMENSION_KEYS
            ])
            print(f"    {cat}: n={n}, avg_MAE={avg_mae:.3f}")

    # --- Error cases ---
    print("\n" + "=" * 60)
    print("Extracting error cases")
    print("=" * 60)
    error_cases = extract_error_cases(merged, gen_by_id, n_cases=20)
    for i, c in enumerate(error_cases[:5]):
        print(f"  #{i+1}: {c['direction']} Δ={c['signed_diff']:+.2f} "
              f"model={c['model']}  sid={c['sample_id'][:12]}")

    # --- Save ---
    report = {
        "n_human_labels": len(all_human),
        "n_judge_results": len(all_judge),
        "n_merged": len(merged),
        "alignment": alignment,
        "error_decomposition": decompositions,
        "n_error_cases": len(error_cases),
    }
    json_path = os.path.join(args.output_dir, "judge_vs_human_raw.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nJSON report → {json_path}")

    # Error cases separate file
    cases_path = os.path.join(args.output_dir, "error_cases.json")
    with open(cases_path, "w") as f:
        json.dump(error_cases, f, indent=2, default=str, ensure_ascii=False)
    print(f"Error cases → {cases_path}")

    # Markdown
    md = generate_markdown(alignment, decompositions, error_cases, len(merged))
    md_path = os.path.join(args.output_dir, "judge_vs_human_raw.md")
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Markdown report → {md_path}")


if __name__ == "__main__":
    main()
