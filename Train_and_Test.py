import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
from Model_Integration import CBT_EmpatheticModel
from transformers import AutoTokenizer

class TherapistQADataset(Dataset):
    def __init__(self, path):
        with open(path, "r", encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            "input_text": sample["current_statement"],
            "target_text": sample["therapist_response"]
        }

def collate_fn(batch, tokenizer, max_length=128):
    input_texts = [item["input_text"] for item in batch]
    target_texts = [item["target_text"] for item in batch]
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    targets = tokenizer(target_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    return inputs["input_ids"], inputs["attention_mask"], targets["input_ids"]

def main():
    model_name = "uer/gpt2-chinese-cluecorpussmall"
    hidden_dim = 768
    batch_size = 4
    epochs = 3
    lr = 5e-5

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = CBT_EmpatheticModel(model_name, hidden_dim)
    model.train()

    dataset = TherapistQADataset("./data/formatted_Psych_data.jsonl")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            collate_fn=lambda b: collate_fn(b, tokenizer))

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    device = torch.device("cpu")
    model.to(device)

    for epoch in range(epochs):
        for i, (input_ids, attention_mask, target_ids) in enumerate(dataloader):
            input_ids, attention_mask, target_ids = input_ids.to(device), attention_mask.to(device), target_ids.to(device)

            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask)

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = target_ids[:, 1:].contiguous()

            min_len = min(shift_logits.size(1), shift_labels.size(1))
            shift_logits = shift_logits[:, :min_len, :]
            shift_labels = shift_labels[:, :min_len]

            loss = criterion(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1)
            )
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "./cbt_gpt2_model.pt")
    print("✅ 共情链增强模型训练完成并保存")

if __name__ == "__main__":
    main()
