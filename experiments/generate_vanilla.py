"""
Generate responses from GPT-2 vanilla (no fine-tuning) as a lower-bound baseline.

This can run immediately without training since it uses the pre-trained model.
"""
import sys
sys.path.insert(0, ".")

import torch
from src.data.build_dataset import load_jsonl, split_records
from src.models.baseline_gpt2 import GPT2BaselineModel
from src.inference.generate import generate_batch, save_jsonl

def main():
    seed = 42
    n_samples = 200

    # Load test data
    records = load_jsonl("data/formatted_Psych_data.jsonl")
    splits = split_records(records, seed=seed)
    test_records = splits["test"][:n_samples]
    print(f"Using {len(test_records)} test samples")

    # Load vanilla GPT-2 (no fine-tuning)
    print("Loading GPT-2 vanilla...")
    model = GPT2BaselineModel("gpt2")

    # Use MPS if available
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        model.to(torch.device("mps"))
        print("Using MPS device")

    # Generate
    print("Generating responses...")
    results = generate_batch(
        model,
        test_records,
        model_name="gpt2-vanilla",
        checkpoint="pretrained",
        seed=seed,
        max_new_tokens=128,
        temperature=0.7,
        top_p=0.9,
    )

    save_jsonl(results, "outputs/generations/gpt2_vanilla.jsonl")
    print(f"Done! Saved {len(results)} generations")

    # Show a few samples
    for r in results[:3]:
        print(f"\n--- Sample {r['id']} ---")
        print(f"User: {r['user_statement'][:100]}...")
        print(f"Response: {r['response'][:150]}...")

if __name__ == "__main__":
    main()
