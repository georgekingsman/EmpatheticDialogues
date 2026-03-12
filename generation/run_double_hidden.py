"""
Condition B — Double-AI with Hidden Checker.

Maker generates a draft → Checker reviews silently → Policy layer decides
approve / revise / abstain / escalate → user sees only the final response.

Usage:
    python generation/run_double_hidden.py \
        --input  data/scenarios/benchmark.jsonl \
        --output outputs/generations/double_hidden.jsonl \
        --model  deepseek-chat \
        --backend deepseek
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

PROMPTS_ROOT = Path(__file__).resolve().parent.parent / "prompts"


def load_maker_prompt() -> str:
    return (PROMPTS_ROOT / "maker" / "system_prompt.txt").read_text(encoding="utf-8").strip()


def load_checker_prompt() -> str:
    return (PROMPTS_ROOT / "checker" / "system_prompt.txt").read_text(encoding="utf-8").strip()


# ---------------------------------------------------------------------------
# API helpers (same pattern as run_single.py)
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
# JSON extraction (reused from src/eval/llm_judge.py pattern)
# ---------------------------------------------------------------------------

def extract_json(text: str) -> dict | None:
    text = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


# ---------------------------------------------------------------------------
# I/O
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
# Core pipeline
# ---------------------------------------------------------------------------

def run_double_hidden(
    scenarios: list[dict],
    client,
    model: str,
    maker_prompt: str,
    checker_prompt: str,
    temperature: float = 0.7,
    checker_temperature: float = 0.3,
    max_tokens: int = 512,
    max_revisions: int = 1,
) -> list[dict]:
    """Run Maker → Checker → Policy for each scenario.

    On ``revise``, the Maker is re-prompted once with checker feedback.
    """
    # Import policy logic
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from checker.checker_schema import validate_checker_output
    from checker.policy_rules import apply_policy

    results = []
    ts = datetime.now(timezone.utc).isoformat()

    for i, sc in enumerate(scenarios):
        user_msg = sc["user_utterance"]
        sid = sc.get("id", hashlib.md5(user_msg.encode()).hexdigest()[:12])

        # ---- Step 1: Maker draft ----
        t0 = time.perf_counter()
        try:
            draft = call_llm(client, model, maker_prompt, user_msg,
                             temperature=temperature, max_tokens=max_tokens)
        except Exception as e:
            draft = f"[MAKER_ERROR] {e}"

        # ---- Step 2: Checker review ----
        checker_input = (
            f"### User message\n{user_msg}\n\n"
            f"### Draft assistant response\n{draft}\n\n"
            "Evaluate the draft and output ONLY the JSON object."
        )
        try:
            checker_raw = call_llm(client, model, checker_prompt, checker_input,
                                   temperature=checker_temperature, max_tokens=512)
        except Exception as e:
            checker_raw = f"[CHECKER_ERROR] {e}"

        parsed = extract_json(checker_raw)
        checker_result = validate_checker_output(parsed) if parsed else None

        # ---- Step 3: Policy ----
        if checker_result is None:
            # Checker parse failure → fall back to approve
            policy = {"action": "approve", "pass_through": True,
                      "revision_note": "", "user_message": "", "resources": ""}
            checker_result = {"scores": {}, "decision": "parse_error",
                              "flags": [], "revision_guidance": "",
                              "escalation_resources": ""}
        else:
            policy = apply_policy(checker_result)

        # ---- Step 4: Handle revise (one retry) ----
        final_response = draft
        revision_count = 0

        if policy["action"] == "revise" and max_revisions > 0:
            revision_note = policy["revision_note"]
            revision_prompt = (
                f"Your previous response was reviewed and the following "
                f"improvement was suggested:\n\n{revision_note}\n\n"
                f"Original user message: {user_msg}\n\n"
                f"Please provide an improved response."
            )
            try:
                final_response = call_llm(
                    client, model, maker_prompt, revision_prompt,
                    temperature=temperature, max_tokens=max_tokens,
                )
                revision_count = 1
            except Exception:
                pass  # keep original draft

        elif policy["action"] == "abstain":
            final_response = policy["user_message"]

        elif policy["action"] == "escalate":
            final_response = (
                policy["user_message"] + "\n\n" + policy["resources"]
            )

        elapsed = time.perf_counter() - t0

        results.append({
            "id": sid,
            "condition": "double_hidden",
            "user_utterance": user_msg,
            "risk_level": sc.get("risk_level", ""),
            "maker_draft": draft,
            "response": final_response,
            "checker_decision": checker_result["decision"],
            "checker_scores": checker_result.get("scores", {}),
            "checker_flags": checker_result.get("flags", []),
            "policy_action": policy["action"],
            "revision_count": revision_count,
            "model": model,
            "temperature": temperature,
            "runtime_s": round(elapsed, 3),
            "ts": ts,
        })

        if (i + 1) % 10 == 0:
            print(f"  [double_hidden] {i + 1}/{len(scenarios)}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Condition B: Double-AI with Hidden Checker")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model", type=str, default="deepseek-chat")
    parser.add_argument("--backend", choices=["openai", "deepseek"], default="deepseek")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--checker_temperature", type=float, default=0.3)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--max_revisions", type=int, default=1)
    args = parser.parse_args()

    scenarios = load_jsonl(args.input)
    print(f"Loaded {len(scenarios)} scenarios from {args.input}")

    client = _make_client(args.backend)
    maker_prompt = load_maker_prompt()
    checker_prompt = load_checker_prompt()

    results = run_double_hidden(
        scenarios, client, args.model, maker_prompt, checker_prompt,
        temperature=args.temperature,
        checker_temperature=args.checker_temperature,
        max_tokens=args.max_tokens,
        max_revisions=args.max_revisions,
    )
    save_jsonl(results, args.output)


if __name__ == "__main__":
    main()
