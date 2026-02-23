"""
Simulate realistic pilot annotations for pipeline testing.

Creates two annotator CSVs with:
  - Moderate agreement (κ_w ≈ 0.3–0.5, realistic for pilot)
  - Some dimension-dependent difficulty (safety easier, helpfulness harder)
  - Some annotator bias (R2 slightly more generous)

This is ONLY for testing the IAA pipeline end-to-end.
Delete the outputs and re-run with real annotations.
"""
import sys, os, csv, json, random
sys.path.insert(0, ".")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np

random.seed(123)
np.random.seed(123)

PILOT_DIR = "outputs/human_annotation/pilot"

# Dimension-specific agreement levels (how much noise per dim)
# Lower noise = higher agreement
DIM_NOISE = {
    "emotion": 0.9,       # moderate agreement
    "validation": 1.0,    # harder to agree on
    "helpfulness": 1.1,   # hardest
    "safety": 0.6,        # easiest to agree on
}

# Annotator bias (added to base score)
ANNOTATOR_BIAS = {
    "R1": 0.0,
    "R2": 0.3,   # R2 slightly more generous
}


def simulate_score(base: float, noise_std: float, bias: float) -> int:
    raw = base + bias + np.random.normal(0, noise_std)
    return int(np.clip(round(raw), 1, 5))


def main():
    # Load mapping to get sample info
    with open(os.path.join(PILOT_DIR, "_pilot_mapping.json")) as f:
        mapping = json.load(f)

    # Load judge results for base scores
    judge_means = {}
    for tag in ["gpt2_vanilla", "gpt2_finetuned", "empathy_chain"]:
        path = f"outputs/judge/{tag}_judge.jsonl"
        if not os.path.exists(path):
            continue
        from collections import defaultdict
        by_sid = defaultdict(list)
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if "scores" in r:
                    by_sid[r["sample_id"]].append(r["scores"])
        for sid, scores_list in by_sid.items():
            dims = {}
            for dim in ["emotion", "validation", "helpfulness", "safety"]:
                vals = [s[dim] for s in scores_list]
                dims[dim] = float(np.mean(vals))
            judge_means[sid] = dims

    # Generate annotations for each annotator
    for ann_id in ["R1", "R2"]:
        in_path = os.path.join(PILOT_DIR, f"pilot_annotation_{ann_id}.csv")
        out_path = os.path.join(PILOT_DIR, f"pilot_annotation_{ann_id}_SIMULATED.csv")

        rows = []
        with open(in_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                eval_id = row["eval_id"]

                # Find original sample_id
                info = next((m for m in mapping if m["eval_id"] == eval_id), None)
                if not info:
                    continue

                sid = info.get("is_duplicate_of", info["sample_id"])
                base_scores = judge_means.get(sid, {
                    "emotion": 1.5, "validation": 1.5,
                    "helpfulness": 1.5, "safety": 2.0,
                })

                new_row = {
                    "eval_id": eval_id,
                    "annotator_id": ann_id,
                }
                dim_scores = []
                for dim in ["emotion", "validation", "helpfulness", "safety"]:
                    score = simulate_score(
                        base_scores.get(dim, 2.0),
                        DIM_NOISE[dim],
                        ANNOTATOR_BIAS[ann_id],
                    )
                    new_row[dim] = score
                    dim_scores.append(score)

                new_row["overall"] = simulate_score(
                    float(np.mean(dim_scores)), 0.7, ANNOTATOR_BIAS[ann_id]
                )
                new_row["notes"] = ""
                rows.append(new_row)

        # Write
        fields = ["eval_id", "annotator_id", "emotion", "validation",
                  "helpfulness", "safety", "overall", "notes"]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"Simulated {len(rows)} annotations → {out_path}")


if __name__ == "__main__":
    main()
