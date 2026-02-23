"""
Generate full 600-sample annotation batch for the GO path.

After pilot IAA passes (κ_w ≥ 0.4 on all dims), this script generates
the complete annotation package: 600 samples (200 per model) with
hidden duplicates for ongoing self-consistency QC.

Usage:
    python experiments/generate_full_annotation.py \
        [--n_per_model 200] \
        [--n_duplicates 30] \
        [--output_dir outputs/human_annotation/full]

Inputs:
    - outputs/generations/gpt2_vanilla.jsonl
    - outputs/generations/gpt2_finetuned.jsonl
    - outputs/generations/empathy_chain.jsonl
    - outputs/judge/*.jsonl          (for uncertainty-based prioritisation)

Outputs:
    - full_annotation/full_samples.csv             (all samples with context)
    - full_annotation/full_annotation_R1.csv       (blank for Rater 1)
    - full_annotation/full_annotation_R2.csv       (blank for Rater 2)
    - full_annotation/_full_mapping.json           (eval_id → sample_id)
    - full_annotation/_full_duplicate_pairs.json   (QC duplicate pairs)
"""
import sys, os, csv, json, argparse, hashlib
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, ".")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np

np.random.seed(42)

MODELS = ["gpt2_vanilla", "gpt2_finetuned", "empathy_chain"]
GEN_DIR = "outputs/generations"
JUDGE_DIR = "outputs/judge"


def load_generations(model_tag: str) -> list[dict]:
    path = os.path.join(GEN_DIR, f"{model_tag}.jsonl")
    if not os.path.exists(path):
        print(f"  Warning: {path} not found, skipping")
        return []
    records = []
    with open(path) as f:
        for line in f:
            r = json.loads(line.strip())
            r["model_tag"] = model_tag
            records.append(r)
    return records


def load_judge_uncertainty(model_tag: str) -> dict:
    """Load judge results and compute per-sample uncertainty (std across repeats)."""
    path = os.path.join(JUDGE_DIR, f"{model_tag}_judge.jsonl")
    if not os.path.exists(path):
        return {}
    from collections import defaultdict
    by_sid = defaultdict(list)
    with open(path) as f:
        for line in f:
            r = json.loads(line.strip())
            if "scores" in r:
                scores = r["scores"]
                mean_score = np.mean([scores.get(d, 3) for d in
                                      ["emotion", "validation", "helpfulness", "safety"]])
                by_sid[r["sample_id"]].append(mean_score)

    uncertainty = {}
    for sid, vals in by_sid.items():
        if len(vals) > 1:
            uncertainty[sid] = float(np.std(vals))
        else:
            uncertainty[sid] = 0.0
    return uncertainty


def select_samples(
    all_records: list[dict],
    n_per_model: int,
    uncertainty: dict,
) -> list[dict]:
    """Select samples per model, prioritising high-uncertainty ones.

    Strategy: top 30% by uncertainty, rest random stratified.
    """
    by_model = defaultdict(list)
    for r in all_records:
        by_model[r["model_tag"]].append(r)

    selected = []
    for model in MODELS:
        pool = by_model.get(model, [])
        if not pool:
            continue

        if len(pool) <= n_per_model:
            selected.extend(pool)
            continue

        # Sort by uncertainty (higher first)
        pool.sort(key=lambda r: -uncertainty.get(
            r.get("sample_id", r.get("id", "")), 0.0
        ))

        n_uncertain = int(n_per_model * 0.3)
        n_random = n_per_model - n_uncertain

        uncertain_picks = pool[:n_uncertain]
        rest = pool[n_uncertain:]
        np.random.shuffle(rest)
        random_picks = rest[:n_random]

        selected.extend(uncertain_picks + random_picks)
        print(f"  {model}: {len(uncertain_picks)} uncertainty + {len(random_picks)} random = {len(uncertain_picks) + len(random_picks)}")

    return selected


def add_hidden_duplicates(
    selected: list[dict],
    n_duplicates: int,
) -> tuple[list[dict], list[list[str]]]:
    """Add hidden duplicates for self-consistency QC."""
    dup_candidates = list(selected)
    np.random.shuffle(dup_candidates)
    n_dup = min(n_duplicates, len(dup_candidates))

    duplicates = []
    dup_pairs = []
    seen_sids = set()
    for r in dup_candidates:
        if len(duplicates) >= n_dup:
            break
        sid = r.get("sample_id", r.get("id", ""))
        if sid in seen_sids:
            continue
        seen_sids.add(sid)

        dup_id = f"dup_{sid[:8]}"
        dup_record = dict(r)
        dup_record["sample_id"] = dup_id
        dup_record["_is_duplicate_of"] = sid
        duplicates.append(dup_record)
        dup_pairs.append([sid, dup_id])

    return selected + duplicates, dup_pairs


def build_annotation_package(
    records: list[dict],
    dup_pairs: list[list[str]],
    output_dir: str,
):
    """Build the full annotation package: samples CSV, blank sheets, mapping."""
    os.makedirs(output_dir, exist_ok=True)

    # Shuffle for blinding
    np.random.shuffle(records)

    # Assign eval_ids
    mapping = []
    for i, r in enumerate(records):
        eval_id = f"full_{i:04d}"
        sid = r.get("sample_id", r.get("id", ""))
        entry = {
            "eval_id": eval_id,
            "sample_id": sid,
            "model_tag": r.get("model_tag", ""),
        }
        if r.get("_is_duplicate_of"):
            entry["is_duplicate_of"] = r["_is_duplicate_of"]
        mapping.append(entry)
        r["eval_id"] = eval_id

    # Save full samples CSV
    sample_fields = ["eval_id", "context", "response"]
    # Detect field names from data
    if records:
        r0 = records[0]
        if "prompt" in r0 and "context" not in r0:
            for r in records:
                r["context"] = r.get("prompt", "")
        if "generated" in r0 and "response" not in r0:
            for r in records:
                r["response"] = r.get("generated", "")

    samples_path = os.path.join(output_dir, "full_samples.csv")
    with open(samples_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sample_fields, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(r)
    print(f"  Samples → {samples_path} ({len(records)} rows)")

    # Blank annotation sheets
    ann_fields = ["eval_id", "annotator_id", "emotion", "validation",
                  "helpfulness", "safety", "overall", "notes"]
    for ann_id in ["R1", "R2"]:
        rows = [{"eval_id": r["eval_id"], "annotator_id": ann_id} for r in records]
        path = os.path.join(output_dir, f"full_annotation_{ann_id}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=ann_fields)
            w.writeheader()
            for row in rows:
                w.writerow({k: row.get(k, "") for k in ann_fields})
        print(f"  Annotation sheet → {path}")

    # Mapping
    map_path = os.path.join(output_dir, "_full_mapping.json")
    with open(map_path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"  Mapping → {map_path} ({len(mapping)} entries)")

    # Duplicate pairs
    dup_path = os.path.join(output_dir, "_full_duplicate_pairs.json")
    with open(dup_path, "w") as f:
        json.dump(dup_pairs, f, indent=2)
    print(f"  Duplicate pairs → {dup_path} ({len(dup_pairs)} pairs)")

    return mapping


def main():
    parser = argparse.ArgumentParser(description="Generate full annotation batch")
    parser.add_argument("--n_per_model", type=int, default=200)
    parser.add_argument("--n_duplicates", type=int, default=30,
                        help="Hidden duplicates for self-consistency QC")
    parser.add_argument("--output_dir", default="outputs/human_annotation/full")
    args = parser.parse_args()

    print("=" * 60)
    print("Generating Full Annotation Batch")
    print("=" * 60)

    # Load all generations
    all_records = []
    uncertainty = {}
    for model in MODELS:
        gens = load_generations(model)
        unc = load_judge_uncertainty(model)
        print(f"  {model}: {len(gens)} generations, {len(unc)} with judge scores")
        all_records.extend(gens)
        uncertainty.update(unc)

    # Select samples
    print(f"\n  Selecting {args.n_per_model} per model ({args.n_per_model * len(MODELS)} total)...")
    selected = select_samples(all_records, args.n_per_model, uncertainty)
    print(f"  Selected: {len(selected)} samples")

    # Add hidden duplicates
    final, dup_pairs = add_hidden_duplicates(selected, args.n_duplicates)
    print(f"  After duplicates: {len(final)} samples ({len(dup_pairs)} duplicates)")

    # Build package
    print(f"\n  Building annotation package → {args.output_dir}/")
    mapping = build_annotation_package(final, dup_pairs, args.output_dir)

    # Summary
    n_dup = sum(1 for m in mapping if m.get("is_duplicate_of"))
    n_regular = len(mapping) - n_dup
    print(f"\n{'='*60}")
    print(f"Full annotation package ready!")
    print(f"{'='*60}")
    print(f"  Regular samples: {n_regular}")
    print(f"  Hidden duplicates: {n_dup}")
    print(f"  Total rows: {len(mapping)}")
    print(f"""
  Distribute to annotators:
    R1: {args.output_dir}/full_samples.csv + full_annotation_R1.csv
    R2: {args.output_dir}/full_samples.csv + full_annotation_R2.csv

  After annotations return, run:
    python experiments/run_pilot_iaa.py \\
      --r1 {args.output_dir}/full_annotation_R1_FILLED.csv \\
      --r2 {args.output_dir}/full_annotation_R2_FILLED.csv \\
      --mapping {args.output_dir}/_full_mapping.json \\
      --duplicates {args.output_dir}/_full_duplicate_pairs.json \\
      --output_dir outputs/analysis/full_iaa
""")


if __name__ == "__main__":
    main()
