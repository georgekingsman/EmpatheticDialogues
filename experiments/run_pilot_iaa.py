"""
Run IAA analysis on pilot annotations and decide GO / NO-GO for full annotation.

Usage:
    python experiments/run_pilot_iaa.py \
        --r1 outputs/human_annotation/pilot/pilot_annotation_R1.csv \
        --r2 outputs/human_annotation/pilot/pilot_annotation_R2.csv \
        --mapping outputs/human_annotation/pilot/_pilot_mapping.json \
        --duplicates outputs/human_annotation/pilot/_duplicate_pairs.json \
        --kappa_threshold 0.4

Outputs:
    outputs/analysis/pilot_iaa_report.json   (full metrics)
    outputs/analysis/pilot_iaa_report.md     (human-readable summary)
"""
import sys, os, csv, json, argparse
sys.path.insert(0, ".")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from src.eval.human_labels_schema import (
    compute_iaa_extended,
    compute_self_consistency,
    iaa_go_nogo,
)
from src.eval.rubric import DIMENSION_KEYS


def load_pilot_csv(path: str) -> list[dict]:
    """Load pilot annotation CSV (uses eval_id, no sample_id yet)."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in DIMENSION_KEYS + ["overall"]:
                if key in row and row[key]:
                    try:
                        row[key] = int(row[key])
                    except ValueError:
                        pass
            records.append(row)
    return records


def resolve_eval_ids(labels: list[dict], mapping: list[dict]) -> list[dict]:
    """Replace eval_id with the real sample_id using the mapping file.

    Keeps duplicate entries with their own sample_id (e.g. dup_xxx)
    so that self-consistency can compare original vs duplicate.
    """
    eval_to_info = {}
    for m in mapping:
        eval_to_info[m["eval_id"]] = m

    resolved = []
    for lab in labels:
        eid = lab.get("eval_id") or lab.get("sample_id")
        info = eval_to_info.get(eid, {})
        new_lab = dict(lab)
        # Keep dup_xxx as-is — do NOT collapse to original
        new_lab["sample_id"] = info.get("sample_id", eid)
        new_lab["model_tag"] = info.get("model_tag", "unknown")
        new_lab["is_duplicate_of"] = info.get("is_duplicate_of")
        resolved.append(new_lab)
    return resolved


def split_labels(labels: list[dict]):
    """Separate duplicate entries from regular ones."""
    regular, duplicates = [], []
    for lab in labels:
        if lab.get("is_duplicate_of"):
            duplicates.append(lab)
        else:
            regular.append(lab)
    return regular, duplicates


def generate_markdown_report(
    iaa: dict, decisions: dict, self_con: dict, n_samples: int
) -> str:
    """Generate a human-readable Markdown report."""
    lines = [
        "# Pilot IAA Report",
        "",
        f"**Samples**: {n_samples} (2 raters)",
        "",
        "## Per-Dimension Agreement",
        "",
        "| Dimension | Weighted κ (linear) | Krippendorff α | Spearman | MAE | Exact Agree | Verdict |",
        "|-----------|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for dim in DIMENSION_KEYS:
        d = iaa.get(dim, {})
        dec = decisions.get(dim, {})
        lines.append(
            f"| {dim} | {d.get('weighted_kappa_linear', 'N/A')} | "
            f"{d.get('krippendorff_alpha', 'N/A')} | "
            f"{d.get('spearman', 'N/A')} | "
            f"{d.get('mae', 'N/A')} | "
            f"{d.get('exact_agreement', 'N/A')} | "
            f"**{dec.get('verdict', 'N/A')}** |"
        )

    lines.extend([
        "",
        "## Score Distributions per Rater",
        "",
    ])
    for dim in DIMENSION_KEYS:
        d = iaa.get(dim, {})
        dist1 = d.get("score_dist_r1", {})
        dist2 = d.get("score_dist_r2", {})
        lines.append(f"**{dim}**: R1={dist1}, R2={dist2}")

    if self_con:
        lines.extend([
            "",
            "## Self-Consistency (hidden duplicates)",
            "",
        ])
        for ann, score in self_con.items():
            lines.append(f"- {ann}: mean |Δ| = {score}")

    overall = decisions.get("_overall", {})
    lines.extend([
        "",
        "## Decision",
        "",
        f"**{overall.get('recommendation', 'N/A')}**",
        "",
        "### Per-dimension actions:",
    ])
    for dim in DIMENSION_KEYS:
        dec = decisions.get(dim, {})
        lines.append(f"- **{dim}**: {dec.get('verdict', 'N/A')} — {dec.get('action', '')}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Pilot IAA analysis")
    parser.add_argument("--r1", default="outputs/human_annotation/pilot/pilot_annotation_R1.csv",
                        help="Rater 1 filled CSV")
    parser.add_argument("--r2", default="outputs/human_annotation/pilot/pilot_annotation_R2.csv",
                        help="Rater 2 filled CSV")
    parser.add_argument("--mapping", default="outputs/human_annotation/pilot/_pilot_mapping.json")
    parser.add_argument("--duplicates", default="outputs/human_annotation/pilot/_duplicate_pairs.json")
    parser.add_argument("--kappa_threshold", type=float, default=0.4)
    parser.add_argument("--output_dir", default="outputs/analysis")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load mapping
    with open(args.mapping) as f:
        mapping = json.load(f)

    # Load duplicate pairs (may not exist)
    dup_pairs = []
    if os.path.exists(args.duplicates):
        with open(args.duplicates) as f:
            dup_pairs = json.load(f)

    # Load and resolve annotations
    print("=" * 60)
    print("Loading pilot annotations")
    print("=" * 60)
    labels_r1 = load_pilot_csv(args.r1)
    labels_r2 = load_pilot_csv(args.r2)

    # Resolve eval_id → sample_id (keeps dup_xxx distinct)
    labels_r1 = resolve_eval_ids(labels_r1, mapping)
    labels_r2 = resolve_eval_ids(labels_r2, mapping)
    print(f"  R1: {len(labels_r1)} rows,  R2: {len(labels_r2)} rows")

    # Split out duplicates for self-consistency vs IAA
    regular_r1, dup_r1 = split_labels(labels_r1)
    regular_r2, dup_r2 = split_labels(labels_r2)
    print(f"  Regular: R1={len(regular_r1)}, R2={len(regular_r2)}")
    print(f"  Duplicates: R1={len(dup_r1)}, R2={len(dup_r2)}")

    all_labels = labels_r1 + labels_r2          # includes dups (for self-consistency)
    regular_labels = regular_r1 + regular_r2    # excludes dups (for IAA)

    # Compute extended IAA (on regular labels only — no duplicates)
    print("\n" + "=" * 60)
    print("Computing inter-annotator agreement")
    print("=" * 60)
    iaa = compute_iaa_extended(regular_labels)

    for dim in DIMENSION_KEYS:
        d = iaa.get(dim, {})
        print(f"  {dim:15s}: weighted_κ={d.get('weighted_kappa_linear', 'N/A'):>6}  "
              f"α={d.get('krippendorff_alpha', 'N/A'):>6}  "
              f"Spearman={d.get('spearman', 'N/A'):>6}  "
              f"MAE={d.get('mae', 'N/A')}")

    # Self-consistency: compare scores where same annotator rated
    # the original and its hidden duplicate
    # dup_pairs format: [[orig_sample_id, dup_sample_id], ...]
    # In all_labels, orig has sample_id=orig_sample_id, dup has sample_id=dup_sample_id
    print("\n  Self-consistency on duplicates:")
    self_con = compute_self_consistency(all_labels, dup_pairs)
    for ann, score in self_con.items():
        print(f"    {ann}: mean |Δ| = {score}")

    # GO / NO-GO decision
    print("\n" + "=" * 60)
    print("GO / NO-GO Decision")
    print("=" * 60)
    decisions = iaa_go_nogo(iaa, kappa_threshold=args.kappa_threshold)
    for dim in DIMENSION_KEYS:
        dec = decisions[dim]
        print(f"  {dim:15s}: {dec['verdict']:8s} (κ_w={dec['weighted_kappa']})  → {dec['action']}")
    overall = decisions["_overall"]
    print(f"\n  >>> {overall['recommendation']}")

    # Save JSON report
    report = {
        "n_pilot_samples": len(regular_r1),
        "n_duplicates": len(dup_r1),
        "kappa_threshold": args.kappa_threshold,
        "iaa_extended": iaa,
        "self_consistency": self_con,
        "go_nogo_decisions": decisions,
    }
    json_path = os.path.join(args.output_dir, "pilot_iaa_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nJSON report → {json_path}")

    # Save Markdown report
    md = generate_markdown_report(iaa, decisions, self_con, len(regular_r1))
    md_path = os.path.join(args.output_dir, "pilot_iaa_report.md")
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Markdown report → {md_path}")


if __name__ == "__main__":
    main()
