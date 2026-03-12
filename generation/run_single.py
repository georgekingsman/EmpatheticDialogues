"""
Condition A — Single-Agent generation.

One LLM generates a supportive reply directly. No checking step.

Usage:
    python generation/run_single.py \
        --input  data/scenarios/benchmark.jsonl \
        --output outputs/generations/single_agent.jsonl \
        --model  deepseek-chat \
        --backend deepseek
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts" / "maker"

def load_maker_prompt() -> str:
    path = PROMPT_DIR / "system_prompt.txt"
    return path.read_text(encoding="utf-8").strip()


# ---------------------------------------------------------------------------
# API wrappers (reuse pattern from src/eval/llm_judge.py)
# ---------------------------------------------------------------------------

def _make_client(backend: str):
    from openai import OpenAI
    if backend == "openai":
        return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    elif backend == "deepseek":
        return OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")


def call_llm(client, model: str, system: str, user: str,
             temperature: float = 0.7, max_tokens: int = 512,
             max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"  [retry] attempt {attempt+1} failed: {e}. Waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Unreachable")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str | Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Saved {len(records)} records → {path}")


def append_jsonl(record: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_done_ids(path: str | Path) -> set:
    """Return set of scenario IDs already in the output file."""
    done = set()
    p = Path(path)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                done.add(json.loads(line).get("id", ""))
    return done


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_single(
    scenarios: list[dict],
    client,
    model: str,
    system_prompt: str,
    output_path: str | Path,
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> list[dict]:
    done_ids = load_done_ids(output_path)
    if done_ids:
        print(f"  Resuming: {len(done_ids)} already done, skipping...")

    results = []
    ts = datetime.now(timezone.utc).isoformat()

    for i, sc in enumerate(scenarios):
        user_msg = sc["user_utterance"]
        sid = sc.get("id", hashlib.md5(user_msg.encode()).hexdigest()[:12])

        if sid in done_ids:
            continue

        t0 = time.perf_counter()
        try:
            response = call_llm(client, model, system_prompt, user_msg,
                                temperature=temperature, max_tokens=max_tokens)
        except Exception as e:
            response = f"[API_ERROR] {e}"
        elapsed = time.perf_counter() - t0

        record = {
            "id": sid,
            "condition": "single_agent",
            "user_utterance": user_msg,
            "risk_level": sc.get("risk_level", ""),
            "response": response,
            "model": model,
            "temperature": temperature,
            "runtime_s": round(elapsed, 3),
            "ts": ts,
        }
        results.append(record)
        append_jsonl(record, output_path)

        if (i + 1) % 10 == 0:
            print(f"  [single] {i + 1}/{len(scenarios)}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Condition A: Single-Agent generation")
    parser.add_argument("--input", type=str, required=True, help="Benchmark JSONL")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL")
    parser.add_argument("--model", type=str, default="deepseek-chat")
    parser.add_argument("--backend", choices=["openai", "deepseek"], default="deepseek")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=512)
    args = parser.parse_args()

    scenarios = load_jsonl(args.input)
    print(f"Loaded {len(scenarios)} scenarios from {args.input}")

    client = _make_client(args.backend)
    system_prompt = load_maker_prompt()

    results = run_single(
        scenarios, client, args.model, system_prompt,
        output_path=args.output,
        temperature=args.temperature, max_tokens=args.max_tokens,
    )
    total = len(load_done_ids(args.output))
    print(f"Done. Total records in {args.output}: {total}")


if __name__ == "__main__":
    main()
