"""Quick test: empathy chain generation after fix."""
import sys, os
sys.path.insert(0, ".")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from src.data.build_dataset import load_jsonl, split_records
from src.models.empathy_chain import CBTEmpatheticModel

records = load_jsonl("data/formatted_Psych_data.jsonl")
splits = split_records(records, seed=42)
test_5 = splits["test"][:5]

model = CBTEmpatheticModel("gpt2", hidden_dim=768)
state = torch.load("checkpoints/empathy_best.pt", map_location="cpu", weights_only=True)
model.load_state_dict(state)
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    model.to(torch.device("mps"))
model.eval()

from src.data.templates import build_inference_prompt

for i, rec in enumerate(test_5):
    prompt = build_inference_prompt(rec["current_statement"])
    tok = model.tokenizer(prompt, return_tensors="pt")
    print(f"\n--- Sample {i} (prompt_len={tok['input_ids'].shape[1]}) ---")
    resp = model.generate_response(prompt, max_new_tokens=128, temperature=0.7, top_p=0.9)
    print(f"Response ({len(resp.split())} words): {resp[:200]}")
