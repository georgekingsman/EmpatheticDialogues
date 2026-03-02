"""
LLM-as-a-Judge evaluation pipeline.

Design goals
------------
* Structured JSON output only (no free-form prose).
* Multi-sample runs for stability analysis (same sample, K seeds).
* Supports any OpenAI-compatible API (GPT-4, DeepSeek, local vLLM, etc.).
* Each judge invocation produces one JSON record.

Usage
-----
    python -m src.eval.llm_judge \\
        --generations outputs/generations/baseline.jsonl \\
        --output outputs/judge/baseline_judge.jsonl \\
        --judge_model gpt-4 \\
        --n_repeats 3
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.eval.rubric import DIMENSION_KEYS, RUBRIC_DIMENSIONS, rubric_to_text, validate_scores

# ---------------------------------------------------------------------------
# Prompt template (strong JSON constraint)
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator of empathetic and supportive dialogue responses.
You will score a therapist/assistant response to a user's emotional statement
using the rubric below.  Output **ONLY** a single JSON object — no markdown
fences, no extra text.

{rubric}

Output exactly this JSON schema (nothing else):
{{
  "scores": {{"emotion": <1-5>, "validation": <1-5>, "helpfulness": <1-5>, "safety": <1-5>}},
  "overall": <1-5>,
  "confidence": <0.0-1.0>,
  "notes": "<brief rationale, 1-2 sentences>"
}}
"""

JUDGE_USER_TEMPLATE = """\
### User statement
{user_statement}

### Assistant response
{response}

Evaluate the response and output ONLY the JSON object.
"""


def build_judge_messages(user_statement: str, response: str) -> list[dict]:
    """Build the chat messages list for the judge LLM."""
    rubric_text = rubric_to_text()
    system = JUDGE_SYSTEM_PROMPT.format(rubric=rubric_text)
    user = JUDGE_USER_TEMPLATE.format(
        user_statement=user_statement,
        response=response,
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# JSON extraction / parsing
# ---------------------------------------------------------------------------

def extract_json(text: str) -> dict | None:
    """Try to extract a JSON object from possibly noisy model output."""
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?", "", text).strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find first { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def validate_judge_output(parsed: dict) -> dict | None:
    """Validate and normalise a parsed judge response.

    Returns None if structurally invalid.
    """
    if not isinstance(parsed, dict):
        return None

    scores = parsed.get("scores", {})
    if not validate_scores(scores):
        return None

    raw_overall = parsed.get("overall")
    if isinstance(raw_overall, (int, float)) and int(raw_overall) in range(1, 6):
        overall = int(raw_overall)
    else:
        # Fall back to mean of dimension scores
        overall = round(sum(scores[k] for k in DIMENSION_KEYS) / len(DIMENSION_KEYS))

    confidence = parsed.get("confidence", 0.5)
    try:
        confidence = float(confidence)
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        confidence = 0.5

    notes = parsed.get("notes", "")

    return {
        "scores": {k: int(scores[k]) for k in DIMENSION_KEYS},
        "overall": int(overall),
        "confidence": round(confidence, 3),
        "notes": str(notes)[:500],
    }


# ---------------------------------------------------------------------------
# Single judge call
# ---------------------------------------------------------------------------

def judge_one(
    user_statement: str,
    response: str,
    api_fn: Callable,
    judge_model: str = "gpt-4",
    temperature: float = 0.3,
    max_retries: int = 2,
) -> dict:
    """Call the judge LLM once and return structured scores.

    Parameters
    ----------
    api_fn : callable
        ``api_fn(messages, model, temperature) -> str``
        Must return the raw text completion.
    """
    messages = build_judge_messages(user_statement, response)

    for attempt in range(max_retries + 1):
        try:
            raw = api_fn(messages, model=judge_model, temperature=temperature)
        except Exception as e:
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
            return {"error": str(e), "raw": ""}

        parsed = extract_json(raw)
        if parsed is not None:
            validated = validate_judge_output(parsed)
            if validated is not None:
                validated["raw"] = raw
                return validated

        # Retry on parse failure
        if attempt < max_retries:
            time.sleep(1)

    return {"error": "json_parse_failure", "raw": raw}


# ---------------------------------------------------------------------------
# Batch judge (main pipeline)
# ---------------------------------------------------------------------------

def judge_batch(
    generations: list[dict],
    api_fn: Callable,
    *,
    judge_model: str = "gpt-4",
    temperature: float = 0.3,
    n_repeats: int = 3,
    delay_between: float = 0.5,
) -> list[dict]:
    """Run the judge over all generations, optionally with multiple repeats.

    Parameters
    ----------
    generations : list[dict]
        Each dict must have ``id``, ``user_statement``, ``response``, ``model``.
    api_fn : callable
        API calling function (see ``judge_one``).
    n_repeats : int
        Number of independent judge calls per sample (for stability analysis).
    delay_between : float
        Seconds to sleep between API calls (rate limiting).

    Returns
    -------
    list[dict] with keys: sample_id, model, repeat_idx, scores, overall,
    confidence, notes, judge_model, judge_temp, ts.
    """
    results: list[dict] = []
    ts = datetime.now(timezone.utc).isoformat()

    total = len(generations) * n_repeats
    done = 0

    for gen in generations:
        for rep in range(n_repeats):
            result = judge_one(
                user_statement=gen["user_statement"],
                response=gen["response"],
                api_fn=api_fn,
                judge_model=judge_model,
                temperature=temperature,
            )

            record = {
                "sample_id": gen["id"],
                "model": gen.get("model", "unknown"),
                "repeat_idx": rep,
                "judge_model": judge_model,
                "judge_temp": temperature,
                "ts": ts,
            }

            if "error" in result:
                record["error"] = result["error"]
                record["raw"] = result.get("raw", "")
            else:
                record.update(
                    {
                        "scores": result["scores"],
                        "overall": result["overall"],
                        "confidence": result["confidence"],
                        "notes": result["notes"],
                    }
                )

            results.append(record)
            done += 1
            if done % 20 == 0:
                print(f"  Judge progress: {done}/{total}")

            if delay_between > 0:
                time.sleep(delay_between)

    return results


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def save_judge_results(results: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Saved {len(results)} judge results → {path}")


def load_judge_results(path: str | Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Example API wrapper (OpenAI-compatible)
# ---------------------------------------------------------------------------

def openai_api_fn(messages: list[dict], model: str = "gpt-4", temperature: float = 0.3) -> str:
    """OpenAI-compatible API wrapper.  Requires OPENAI_API_KEY env var."""
    try:
        from openai import OpenAI  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError("pip install openai")

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=300,
    )
    return resp.choices[0].message.content


def deepseek_api_fn(messages: list[dict], model: str = "deepseek-chat", temperature: float = 0.3) -> str:
    """DeepSeek API wrapper.  Requires DEEPSEEK_API_KEY env var."""
    try:
        from openai import OpenAI  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError("pip install openai")

    client = OpenAI(
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=300,
    )
    return resp.choices[0].message.content


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="LLM-as-a-Judge pipeline")
    parser.add_argument("--generations", type=str, required=True, help="Path to generation JSONL")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL path")
    parser.add_argument("--judge_model", type=str, default="gpt-4")
    parser.add_argument("--judge_backend", choices=["openai", "deepseek"], default="openai")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--n_repeats", type=int, default=3)
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args()

    from src.inference.generate import load_jsonl

    generations = load_jsonl(args.generations)
    print(f"Loaded {len(generations)} generations from {args.generations}")

    if args.judge_backend == "openai":
        api_fn = openai_api_fn
    else:
        api_fn = deepseek_api_fn

    results = judge_batch(
        generations,
        api_fn=api_fn,
        judge_model=args.judge_model,
        temperature=args.temperature,
        n_repeats=args.n_repeats,
        delay_between=args.delay,
    )

    save_judge_results(results, args.output)


if __name__ == "__main__":
    main()
