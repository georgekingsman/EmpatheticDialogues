"""
Generate a PILOT annotation batch (150 samples) for 2 annotators.

Selects samples via stratified sampling across 3 models (50 each),
prioritising high-uncertainty samples from judge analysis.

Outputs:
  - outputs/human_annotation/pilot/pilot_samples.csv       (context for reading)
  - outputs/human_annotation/pilot/pilot_annotation_R1.csv  (blank sheet rater 1)
  - outputs/human_annotation/pilot/pilot_annotation_R2.csv  (blank sheet rater 2)
  - outputs/human_annotation/pilot/_pilot_mapping.json      (eval_id → sample_id + model)
"""
import sys, os, csv, json, random
sys.path.insert(0, ".")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from collections import defaultdict
from src.eval.rubric import DIMENSION_KEYS

random.seed(42)
np.random.seed(42)

GENERATION_FILES = {
    "gpt2_vanilla":    "outputs/generations/gpt2_vanilla.jsonl",
    "gpt2_finetuned":  "outputs/generations/gpt2_finetuned.jsonl",
    "empathy_chain":   "outputs/generations/empathy_chain.jsonl",
}

JUDGE_FILES = {
    "gpt2_vanilla":    "outputs/judge/gpt2_vanilla_judge.jsonl",
    "gpt2_finetuned":  "outputs/judge/gpt2_finetuned_judge.jsonl",
    "empathy_chain":   "outputs/judge/empathy_chain_judge.jsonl",
}

N_PER_MODEL = 50       # 50 × 3 models = 150 pilot samples
N_DUPLICATES = 10       # hidden duplicates for self-consistency QC
ANNOTATOR_IDS = ["R1", "R2"]


def _load_generations(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def _load_judge_uncertainty(path):
    """Return dict[sample_id → mean_std_across_dims] for uncertainty ranking."""
    records = [json.loads(l) for l in open(path) if "scores" in l]
    by_id = defaultdict(list)
    for r in records:
        if "scores" in r:
            by_id[r["sample_id"]].append(r)
    uncertainty = {}
    for sid, repeats in by_id.items():
        stds = []
        for dim in DIMENSION_KEYS:
            vals = [r["scores"][dim] for r in repeats]
            if len(vals) >= 2:
                stds.append(float(np.std(vals)))
        uncertainty[sid] = float(np.mean(stds)) if stds else 0.0
    return uncertainty


def main():
    out_dir = "outputs/human_annotation/pilot"
    os.makedirs(out_dir, exist_ok=True)

    # Collect candidates per model, ordered by judge uncertainty (high first)
    selected = []  # list of {sample_id, model_tag, user_statement, response}
    for tag, gen_path in GENERATION_FILES.items():
        gens = _load_generations(gen_path)
        gen_by_id = {g["id"]: g for g in gens}

        uncertainty = _load_judge_uncertainty(JUDGE_FILES[tag])

        # Sort by uncertainty desc; take top N_PER_MODEL
        ranked_ids = sorted(gen_by_id.keys(), key=lambda s: uncertainty.get(s, 0), reverse=True)
        chosen_ids = ranked_ids[:N_PER_MODEL]

        for sid in chosen_ids:
            g = gen_by_id[sid]
            selected.append({
                "sample_id": sid,
                "model_tag": tag,
                "user_statement": g["user_statement"],
                "response": g["response"],
            })

    # Shuffle for blind evaluation
    random.shuffle(selected)

    # Add hidden duplicates (randomly pick N_DUPLICATES from selected, re-insert)
    dup_sources = random.sample(selected, min(N_DUPLICATES, len(selected)))
    dup_mapping = []
    for src in dup_sources:
        dup_id = f"dup_{src['sample_id'][:8]}"
        dup_entry = dict(src)
        dup_entry["sample_id"] = dup_id
        dup_entry["_original_sample_id"] = src["sample_id"]
        selected.append(dup_entry)
        dup_mapping.append((src["sample_id"], dup_id))

    # Re-shuffle so duplicates aren't at the end
    random.shuffle(selected)

    # Assign eval_ids
    for i, rec in enumerate(selected):
        rec["eval_id"] = f"pilot_{i:04d}"

    # --- Save context sheet (what annotators read) ---
    ctx_path = os.path.join(out_dir, "pilot_samples.csv")
    with open(ctx_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["eval_id", "user_statement", "response"])
        writer.writeheader()
        for rec in selected:
            writer.writerow({
                "eval_id": rec["eval_id"],
                "user_statement": rec["user_statement"],
                "response": rec["response"],
            })
    print(f"Pilot context sheet ({len(selected)} samples) → {ctx_path}")

    # --- Save blank annotation sheets ---
    ann_fields = ["eval_id", "annotator_id", "emotion", "validation",
                  "helpfulness", "safety", "overall", "notes"]
    for ann_id in ANNOTATOR_IDS:
        ann_path = os.path.join(out_dir, f"pilot_annotation_{ann_id}.csv")
        with open(ann_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=ann_fields)
            writer.writeheader()
            for rec in selected:
                writer.writerow({
                    "eval_id": rec["eval_id"],
                    "annotator_id": ann_id,
                    "emotion": "", "validation": "", "helpfulness": "",
                    "safety": "", "overall": "", "notes": "",
                })
        print(f"Blank annotation sheet → {ann_path}")

    # --- Save internal mapping ---
    mapping = []
    for rec in selected:
        m = {
            "eval_id": rec["eval_id"],
            "sample_id": rec["sample_id"],
            "model_tag": rec["model_tag"],
        }
        if "_original_sample_id" in rec:
            m["is_duplicate_of"] = rec["_original_sample_id"]
        mapping.append(m)

    mapping_path = os.path.join(out_dir, "_pilot_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"Internal mapping → {mapping_path}")

    # --- Save duplicate pairs (for self-consistency check) ---
    dup_path = os.path.join(out_dir, "_duplicate_pairs.json")
    with open(dup_path, "w") as f:
        json.dump(dup_mapping, f, indent=2)
    print(f"Duplicate pairs ({len(dup_mapping)}) → {dup_path}")

    print(f"\n=== PILOT READY ===")
    print(f"Total samples: {len(selected)} ({N_PER_MODEL}×3 models + {N_DUPLICATES} hidden duplicates)")
    print(f"Give each annotator:")
    print(f"  1. {ctx_path}   (read this)")
    print(f"  2. pilot_annotation_Rx.csv  (fill scores here)")
    print(f"  3. docs/rubric_v2.md + docs/annotation_guide_v2.md")


if __name__ == "__main__":
    main()
