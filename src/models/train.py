"""
Training script — supports both baseline GPT-2 and Empathy Chain model.

Key improvement over original: proper label masking (-100 on prompt tokens)
so loss is computed ONLY on the therapist response.

Usage:
    # Train baseline
    python -m src.models.train --model_type baseline --epochs 3

    # Train empathy chain
    python -m src.models.train --model_type empathy --epochs 3
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.data.build_dataset import EmpathyDataset, load_jsonl, split_records


def train(args):
    # ---- Device ----
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ---- Model ----
    if args.model_type == "baseline":
        from src.models.baseline_gpt2 import GPT2BaselineModel

        model = GPT2BaselineModel(args.model_name)
        tokenizer = model.tokenizer
    else:
        from src.models.empathy_chain import CBTEmpatheticModel

        model = CBTEmpatheticModel(args.model_name, hidden_dim=args.hidden_dim)
        tokenizer = model.tokenizer

    model.to(device)
    model.train()

    # ---- Data ----
    records = load_jsonl(args.data_path)
    splits = split_records(records, seed=args.seed)
    print(f"Train: {len(splits['train'])}  Val: {len(splits['val'])}  Test: {len(splits['test'])}")

    train_ds = EmpathyDataset(splits["train"], tokenizer, max_length=args.max_length)
    val_ds = EmpathyDataset(splits["val"], tokenizer, max_length=args.max_length)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False
    )

    # ---- Optimizer ----
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # ---- Training loop ----
    best_val_loss = float("inf")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_train_batches = len(train_loader)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            loss_val = loss.item()
            if not (loss_val == loss_val):  # NaN check
                print(f"  ⚠ NaN loss at batch {n_batches+1}, skipping", flush=True)
                continue
            total_loss += loss_val
            n_batches += 1

            if n_batches % 100 == 0 or n_batches == n_train_batches:
                avg = total_loss / n_batches
                print(f"  Epoch {epoch} [{n_batches}/{n_train_batches}] loss={avg:.4f}", flush=True)

        train_loss = total_loss / max(n_batches, 1)
        elapsed = time.time() - t0

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs["loss"].item()
                val_batches += 1

        val_loss = val_loss / max(val_batches, 1)

        print(
            f"Epoch {epoch}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"time={elapsed:.1f}s"
        )

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = out_dir / f"{args.model_type}_best.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  → saved best checkpoint: {ckpt_path}")

    # Save final
    final_path = out_dir / f"{args.model_type}_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"Training complete. Final checkpoint: {final_path}")


def main():
    parser = argparse.ArgumentParser(description="Train baseline or empathy model")
    parser.add_argument("--model_type", choices=["baseline", "empathy"], default="baseline")
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--data_path", type=str, default="data/formatted_Psych_data.jsonl")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
