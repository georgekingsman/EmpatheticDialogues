"""
Simulate human annotations for testing the calibration pipeline.

Creates a synthetic human annotation CSV that:
- Partially agrees with judge (correlation ~0.6-0.8)
- Has some systematic bias (humans tend to score slightly higher than LLM judge)
- Covers 100 samples from each model (300 total)

This is for PIPELINE TESTING ONLY. Real human annotations are needed for the paper.
"""
import sys, os, json, csv, random
sys.path.insert(0, ".")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from collections import defaultdict
from src.eval.rubric import DIMENSION_KEYS

random.seed(42)
np.random.seed(42)

JUDGE_FILES = {
    "gpt2_vanilla": "outputs/judge/gpt2_vanilla_judge.jsonl",
    "gpt2_finetuned": "outputs/judge/gpt2_finetuned_judge.jsonl",
    "empathy_chain": "outputs/judge/empathy_chain_judge.jsonl",
}


def simulate_human_score(judge_score: float, noise_std: float = 0.8, bias: float = 0.3) -> int:
    """Simulate a human score given a judge score.
    
    Adds positive bias (humans slightly more generous) and random noise.
    """
    human = judge_score + bias + np.random.normal(0, noise_std)
    return int(np.clip(round(human), 1, 5))


def main():
    os.makedirs("outputs/human_annotation", exist_ok=True)
    
    all_labels = []
    
    for tag, path in JUDGE_FILES.items():
        records = [json.loads(l) for l in open(path)]
        
        # Average judge scores by sample_id
        by_sample = defaultdict(list)
        for r in records:
            if "scores" in r:
                by_sample[r["sample_id"]].append(r)
        
        # Take first 100 samples
        sample_ids = list(by_sample.keys())[:100]
        
        for sid in sample_ids:
            repeats = by_sample[sid]
            judge_means = {}
            for dim in DIMENSION_KEYS:
                vals = [r["scores"][dim] for r in repeats]
                judge_means[dim] = float(np.mean(vals))
            
            # Simulate two annotators
            for annotator in ["annotator_A", "annotator_B"]:
                label = {
                    "sample_id": sid,
                    "annotator_id": annotator,
                }
                for dim in DIMENSION_KEYS:
                    label[dim] = simulate_human_score(judge_means[dim])
                label["overall"] = simulate_human_score(
                    np.mean([judge_means[d] for d in DIMENSION_KEYS])
                )
                label["notes"] = f"simulated_{tag}"
                all_labels.append(label)
    
    # Save as CSV
    fields = ["sample_id", "annotator_id"] + DIMENSION_KEYS + ["overall", "notes"]
    out_path = "outputs/human_annotation/simulated_human_labels.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for lab in all_labels:
            writer.writerow(lab)
    
    print(f"Generated {len(all_labels)} simulated annotations → {out_path}")
    print(f"  Samples: {len(all_labels) // 2} (2 annotators each)")
    
    # Quick stats
    for dim in DIMENSION_KEYS:
        vals = [lab[dim] for lab in all_labels]
        print(f"  {dim}: mean={np.mean(vals):.2f}, std={np.std(vals):.2f}")


if __name__ == "__main__":
    main()
