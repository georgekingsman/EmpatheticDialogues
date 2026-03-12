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
             temperature: float = 0.7, max_tokens: int = 512) -> str:
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


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_single(
    scenarios: list[dict],
    client,
    model: str,
    system_prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> list[dict]:
    results = []
    ts = datetime.now(timezone.utc).isoformat()

    for i, sc in enumerate(scenarios):
        user_msg = sc["user_utterance"]
        sid = sc.get("id", hashlib.md5(user_msg.encode()).hexdigest()[:12])

        t0 = time.perf_counter()
        try:
            response = call_llm(client, model, system_prompt, user_msg,
                                temperature=temperature, max_tokens=max_tokens)
        except Exception as e:
            response = f"[API_ERROR] {e}"
        elapsed = time.perf_counter() - t0

        results.append({
            "id": sid,
            "condition": "single_agent",
            "user_utterance": user_msg,
            "risk_level": sc.get("risk_level", ""),
            "response": response,
            "model": model,
            "temperature": temperature,
            "runtime_s": round(elapsed, 3),
            "ts": ts,
        })

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
        temperature=args.temperature, max_tokens=args.max_tokens,
    )
    save_jsonl(results, args.output)


if __name__ == "__main__":
    main()
