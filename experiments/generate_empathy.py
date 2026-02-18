"""Generate responses from the fine-tuned empathy chain model."""
import sys
sys.path.insert(0, ".")

import torch
from src.data.build_dataset import load_jsonl, split_records
from src.models.empathy_chain import CBTEmpatheticModel
from src.inference.generate import generate_batch, save_jsonl

def main():
    seed = 42
    n_samples = 200

    records = load_jsonl("data/formatted_Psych_data.jsonl")
    splits = split_records(records, seed=seed)
    test_records = splits["test"][:n_samples]
    print(f"Using {len(test_records)} test samples")

    print("Loading fine-tuned empathy chain model...")
    model = CBTEmpatheticModel("gpt2", hidden_dim=768)
    state = torch.load("checkpoints/empathy_best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        model.to(torch.device("mps"))
        print("Using MPS device")

    print("Generating responses...")
    results = generate_batch(
        model,
        test_records,
        model_name="gpt2-empathy-chain",
        checkpoint="checkpoints/empathy_best.pt",
        seed=seed,
        max_new_tokens=128,
        temperature=0.7,
        top_p=0.9,
    )

    save_jsonl(results, "outputs/generations/empathy_chain.jsonl")

    for r in results[:3]:
        print(f"\n--- Sample {r['id']} ---")
        print(f"User: {r['user_statement'][:100]}...")
        print(f"Response: {r['response'][:200]}...")

if __name__ == "__main__":
    main()
