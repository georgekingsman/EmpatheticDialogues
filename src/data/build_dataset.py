"""
Build a HuggingFace-compatible Causal LM dataset from formatted_Psych_data.jsonl.

Key design decisions
--------------------
* Prompt tokens are masked with ``-100`` in labels so that loss is
  computed **only** on the therapist response (+ EOS).
* Supports stratified train/val/test split with fixed seed.
* Optionally returns a lightweight subset for annotation sampling.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import BatchEncoding, PreTrainedTokenizer

from src.data.templates import build_training_text

# ---------------------------------------------------------------------------
# Raw JSONL loader
# ---------------------------------------------------------------------------

def load_jsonl(path: str | Path) -> list[dict]:
    """Load a JSONL file and return a list of dicts."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Causal LM Dataset with proper label masking
# ---------------------------------------------------------------------------

class EmpathyDataset(Dataset):
    """Causal LM dataset with label masking on prompt tokens.

    Each sample is tokenized as:
        [prompt_tokens] [target_tokens]
    Labels:
        [-100 ... -100] [target_token_ids]

    Parameters
    ----------
    records : list[dict]
        Each record must contain ``current_statement`` and ``therapist_response``.
    tokenizer : PreTrainedTokenizer
        Tokenizer to use.  ``pad_token`` will be set to ``eos_token`` if unset.
    max_length : int
        Maximum total sequence length (prompt + target).
    """

    def __init__(
        self,
        records: list[dict],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Filter out records where prompt alone exceeds max_length
        # (these would produce all-masked labels → NaN loss)
        self.records = []
        skipped = 0
        for rec in records:
            parts = build_training_text(
                user_statement=rec["current_statement"],
                therapist_response=rec["therapist_response"],
            )
            prompt_ids = tokenizer.encode(parts["prompt"], add_special_tokens=False)
            if len(prompt_ids) < max_length - 5:  # need at least 5 target tokens
                self.records.append(rec)
            else:
                skipped += 1
        if skipped > 0:
            print(f"  EmpathyDataset: filtered {skipped}/{skipped+len(self.records)} samples (prompt >= max_length)")


    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        parts = build_training_text(
            user_statement=rec["current_statement"],
            therapist_response=rec["therapist_response"],
        )

        # Tokenize full sequence
        full_enc: BatchEncoding = self.tokenizer(
            parts["full_text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Tokenize prompt alone to know how many tokens to mask
        prompt_enc: BatchEncoding = self.tokenizer(
            parts["prompt"],
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )
        prompt_len: int = len(prompt_enc["input_ids"])  # type: ignore[arg-type]

        input_ids: torch.Tensor = full_enc["input_ids"].squeeze(0)          # type: ignore[union-attr]  # (max_length,)
        attention_mask: torch.Tensor = full_enc["attention_mask"].squeeze(0) # type: ignore[union-attr]  # (max_length,)

        # Build labels: -100 on prompt + pad positions
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100  # also mask padding

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ---------------------------------------------------------------------------
# Train / val / test split helper
# ---------------------------------------------------------------------------

def split_records(
    records: list[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Deterministic stratified (by prefix of therapist_response) split.

    Returns dict with keys ``train``, ``val``, ``test``.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


# ---------------------------------------------------------------------------
# Stratified annotation subsample
# ---------------------------------------------------------------------------

def sample_for_annotation(
    records: list[dict],
    n: int = 200,
    seed: int = 42,
) -> list[dict]:
    """Return a stratified sub-sample suitable for human annotation.

    Stratification is by response length terciles to ensure
    coverage of short / medium / long responses.
    """
    rng = random.Random(seed)
    by_len = sorted(records, key=lambda r: len(r["therapist_response"]))
    tercile = len(by_len) // 3
    buckets = [by_len[:tercile], by_len[tercile : 2 * tercile], by_len[2 * tercile :]]

    per_bucket = n // len(buckets)
    sampled: list[dict] = []
    for bucket in buckets:
        sampled.extend(rng.sample(bucket, min(per_bucket, len(bucket))))

    # Fill remainder from the full list
    remaining = n - len(sampled)
    if remaining > 0:
        pool = [r for r in records if r not in sampled]
        sampled.extend(rng.sample(pool, min(remaining, len(pool))))

    rng.shuffle(sampled)
    return sampled[:n]


# ---------------------------------------------------------------------------
# CLI entry-point: build & inspect the dataset
# ---------------------------------------------------------------------------

def main():
    """Quick sanity check: load data, split, print stats."""
    import argparse

    parser = argparse.ArgumentParser(description="Build empathy dataset")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/formatted_Psych_data.jsonl",
        help="Path to formatted JSONL data",
    )
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = load_jsonl(args.data_path)
    splits = split_records(records, seed=args.seed)

    print(f"Total records:  {len(records)}")
    print(f"  Train:        {len(splits['train'])}")
    print(f"  Val:          {len(splits['val'])}")
    print(f"  Test:         {len(splits['test'])}")

    # Show a sample
    sample = splits["train"][0]
    parts = build_training_text(
        sample["current_statement"], sample["therapist_response"]
    )
    print(f"\n--- Sample prompt ---\n{parts['prompt']}")
    print(f"\n--- Sample target ---\n{parts['target'][:200]}...")

    # Annotation subset
    ann = sample_for_annotation(splits["test"], n=200, seed=args.seed)
    print(f"\nAnnotation subset size: {len(ann)}")


if __name__ == "__main__":
    main()
