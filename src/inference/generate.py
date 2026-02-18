"""
Unified generation interface.

All models go through the same ``generate_batch()`` wrapper which records:
  model_name / checkpoint / seed / decoding params / prompt / response / runtime

Output is a list of dicts ready for JSONL serialisation.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from src.data.templates import build_inference_prompt


# ---------------------------------------------------------------------------
# Single-response helper
# ---------------------------------------------------------------------------

def _generate_one(
    model,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    do_sample: bool,
) -> tuple[str, float]:
    """Call model.generate_response and return (response, elapsed_seconds)."""
    t0 = time.perf_counter()
    response = model.generate_response(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        do_sample=do_sample,
    )
    elapsed = time.perf_counter() - t0
    return response, elapsed


# ---------------------------------------------------------------------------
# Batch generation (main entry-point)
# ---------------------------------------------------------------------------

def generate_batch(
    model,
    records: list[dict],
    *,
    model_name: str = "unknown",
    checkpoint: str = "",
    seed: int = 0,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    do_sample: bool = True,
) -> list[dict]:
    """Generate responses for a list of records and return annotated results.

    Parameters
    ----------
    model : nn.Module
        Any model with a ``generate_response(prompt, ...)`` method.
    records : list[dict]
        Each record must have ``current_statement`` (and optionally an ``id``).
    model_name : str
        Identifier written to output metadata.
    checkpoint : str
        Path or tag for the model checkpoint.
    seed : int
        Random seed (set before generation).
    max_new_tokens, temperature, top_p, top_k, do_sample
        Standard decoding hyperparameters.

    Returns
    -------
    list[dict]  – one dict per record with full provenance metadata.
    """
    torch.manual_seed(seed)
    model.eval()

    results: list[dict] = []
    ts = datetime.now(timezone.utc).isoformat()

    for i, rec in enumerate(records):
        user_text = rec["current_statement"]
        prompt = build_inference_prompt(user_text)

        response, elapsed = _generate_one(
            model, prompt, max_new_tokens, temperature, top_p, top_k, do_sample
        )

        # Deterministic sample ID
        sample_id = rec.get("id", hashlib.md5(user_text.encode()).hexdigest()[:12])

        results.append(
            {
                "id": sample_id,
                "prompt": prompt,
                "user_statement": user_text,
                "response": response,
                "reference": rec.get("therapist_response", ""),
                "model": model_name,
                "checkpoint": checkpoint,
                "seed": seed,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "runtime_s": round(elapsed, 3),
                "ts": ts,
            }
        )
        if (i + 1) % 50 == 0:
            print(f"  [{model_name}] generated {i + 1}/{len(records)}")

    return results


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

def save_jsonl(records: list[dict], path: str | Path) -> None:
    """Append-safe JSONL writer."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Saved {len(records)} records → {path}")


def load_jsonl(path: str | Path) -> list[dict]:
    """Load a JSONL file."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# API-based generation (for strong baselines like GPT-4 / DeepSeek)
# ---------------------------------------------------------------------------

def generate_via_api(
    records: list[dict],
    *,
    api_fn,
    model_name: str = "gpt-4",
    seed: int = 0,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 256,
    system_prompt: str = (
        "You are a compassionate and professional therapist. "
        "Respond empathetically to the user's statement."
    ),
) -> list[dict]:
    """Generate responses via an external API function.

    ``api_fn(system_prompt, user_message, **kwargs) -> str`` is the caller's
    responsibility to implement (wrapping OpenAI / Anthropic / DeepSeek SDK).

    Parameters
    ----------
    api_fn : callable
        Signature: ``api_fn(system: str, user: str, **kw) -> str``
    """
    results: list[dict] = []
    ts = datetime.now(timezone.utc).isoformat()

    for i, rec in enumerate(records):
        user_text = rec["current_statement"]
        prompt = build_inference_prompt(user_text)

        t0 = time.perf_counter()
        try:
            response = api_fn(
                system_prompt,
                user_text,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_new_tokens,
            )
        except Exception as e:
            response = f"[API_ERROR] {e}"
        elapsed = time.perf_counter() - t0

        sample_id = rec.get("id", hashlib.md5(user_text.encode()).hexdigest()[:12])
        results.append(
            {
                "id": sample_id,
                "prompt": prompt,
                "user_statement": user_text,
                "response": response,
                "reference": rec.get("therapist_response", ""),
                "model": model_name,
                "checkpoint": "api",
                "seed": seed,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": -1,
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
                "runtime_s": round(elapsed, 3),
                "ts": ts,
            }
        )
        if (i + 1) % 20 == 0:
            print(f"  [{model_name}] generated {i + 1}/{len(records)}")

    return results


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main():
    """Generate responses from local models and save to JSONL."""
    import argparse

    parser = argparse.ArgumentParser(description="Batch generation interface")
    parser.add_argument("--data_path", type=str, default="data/formatted_Psych_data.jsonl")
    parser.add_argument("--model_type", choices=["baseline", "empathy"], default="baseline")
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--output", type=str, default="outputs/generations/baseline.jsonl")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load data
    from src.data.build_dataset import load_jsonl as load_data, split_records

    all_records = load_data(args.data_path)
    splits = split_records(all_records, seed=args.seed)
    test_records = splits["test"][: args.n_samples]

    # Load model
    if args.model_type == "baseline":
        from src.models.baseline_gpt2 import GPT2BaselineModel

        model = GPT2BaselineModel(args.model_name)
    else:
        from src.models.empathy_chain import CBTEmpatheticModel

        model = CBTEmpatheticModel(args.model_name)

    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state)

    # Generate
    results = generate_batch(
        model,
        test_records,
        model_name=args.model_name,
        checkpoint=args.checkpoint,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    save_jsonl(results, args.output)


if __name__ == "__main__":
    main()
