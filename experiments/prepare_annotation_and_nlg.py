"""
Generate blank human annotation sheets and compute NLG metrics.

Creates:
  - outputs/human_annotation/  — blank CSV sheets for each model
  - outputs/nlg_metrics.json   — BLEU/ROUGE for fine-tuned models vs references
"""
import sys, os
sys.path.insert(0, ".")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
from src.eval.human_labels_schema import generate_blank_annotation_sheet
from src.eval.metrics import compute_nlg_metrics

GENERATION_FILES = {
    "gpt2_vanilla": "outputs/generations/gpt2_vanilla.jsonl",
    "gpt2_finetuned": "outputs/generations/gpt2_finetuned.jsonl",
    "empathy_chain": "outputs/generations/empathy_chain.jsonl",
}


def main():
    os.makedirs("outputs/human_annotation", exist_ok=True)

    # ---------------------------------------------------------------
    # 1. Generate blank annotation sheets
    # ---------------------------------------------------------------
    print("=" * 60)
    print("Generating blank human annotation sheets")
    print("=" * 60)
    for tag, path in GENERATION_FILES.items():
        out_csv = f"outputs/human_annotation/{tag}_annotation.csv"
        generate_blank_annotation_sheet(path, out_csv, annotator_id="annotator_A")
        # Also create sheet for annotator B
        out_csv_b = f"outputs/human_annotation/{tag}_annotation_B.csv"
        generate_blank_annotation_sheet(path, out_csv_b, annotator_id="annotator_B")

    # Also create a combined annotation sheet with all models interleaved
    # (this is useful for blind evaluation where annotators don't know which model)
    print("\nCreating combined blind annotation sheet...")
    combined = []
    for tag, path in GENERATION_FILES.items():
        with open(path) as f:
            for line in f:
                rec = json.loads(line.strip())
                combined.append({
                    "sample_id": rec["id"],
                    "model_tag": tag,  # for internal tracking, strip before giving to annotators
                    "user_statement": rec["user_statement"],
                    "response": rec["response"],
                })

    # Shuffle for blind evaluation
    import random
    random.seed(42)
    random.shuffle(combined)

    # Save combined context sheet (for annotators to read)
    context_path = "outputs/human_annotation/blind_evaluation_samples.csv"
    import csv
    with open(context_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["eval_id", "user_statement", "response"])
        writer.writeheader()
        for i, rec in enumerate(combined):
            writer.writerow({
                "eval_id": f"eval_{i:04d}",
                "user_statement": rec["user_statement"],
                "response": rec["response"],
            })
    print(f"  Saved {len(combined)} blind samples → {context_path}")

    # Save internal mapping (eval_id → sample_id + model)
    mapping_path = "outputs/human_annotation/_internal_mapping.json"
    mapping = []
    for i, rec in enumerate(combined):
        mapping.append({
            "eval_id": f"eval_{i:04d}",
            "sample_id": rec["sample_id"],
            "model_tag": rec["model_tag"],
        })
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"  Internal mapping → {mapping_path}")

    # Save blank annotation sheet for blind eval
    blind_annotation_path = "outputs/human_annotation/blind_annotation_A.csv"
    with open(blind_annotation_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "eval_id", "annotator_id", "emotion", "validation",
            "helpfulness", "safety", "overall", "notes"
        ])
        writer.writeheader()
        for i in range(len(combined)):
            writer.writerow({
                "eval_id": f"eval_{i:04d}",
                "annotator_id": "annotator_A",
                "emotion": "", "validation": "", "helpfulness": "",
                "safety": "", "overall": "", "notes": "",
            })
    print(f"  Blank blind annotation → {blind_annotation_path}")

    # ---------------------------------------------------------------
    # 2. Compute NLG metrics (BLEU/ROUGE)
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Computing NLG metrics (BLEU / ROUGE)")
    print("=" * 60)

    nlg_results = {}
    for tag, path in GENERATION_FILES.items():
        with open(path) as f:
            records = [json.loads(line) for line in f]

        predictions = [r["response"] for r in records]
        references = [r["reference"] for r in records]

        # Skip if all references are empty
        if all(not ref.strip() for ref in references):
            print(f"  {tag}: no references available, skipping")
            nlg_results[tag] = {"note": "no references"}
            continue

        metrics = compute_nlg_metrics(predictions, references)
        nlg_results[tag] = metrics
        print(f"  {tag}: BLEU={metrics.get('bleu',0):.4f}  ROUGE-1={metrics.get('rouge1',0):.4f}  ROUGE-2={metrics.get('rouge2',0):.4f}  ROUGE-L={metrics.get('rougeL',0):.4f}")

    # Save
    nlg_path = "outputs/nlg_metrics.json"
    with open(nlg_path, "w") as f:
        json.dump(nlg_results, f, indent=2)
    print(f"\nSaved NLG metrics → {nlg_path}")

    print("\nDone! Annotation sheets ready in outputs/human_annotation/")


if __name__ == "__main__":
    main()
