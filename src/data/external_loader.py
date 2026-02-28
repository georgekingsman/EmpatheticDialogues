"""
External dataset loader for human-anchored calibration (Route B).

Loads publicly available empathetic / counseling dialogue datasets that include
human quality ratings, and normalises them into a unified JSONL format:

    {"item_id": ..., "prompt": ..., "response": ..., "human_overall": 1-5,
     "human_scores": {"emotion": ..., "validation": ..., ...},     # optional
     "source": "dataset_name"}

Supported datasets (auto-detected or specified via --dataset):
    1. EmpatheticDialogues-EVAL  – custom CSV/JSONL with human ratings
    2. EPITOME                   – empathy/emotional-reaction/exploration labels (0/1/2)
    3. Generic CSV/JSONL         – any file with prompt + response + numeric score

If the external data uses a different scale (e.g., 0–2 or 1–7), the loader
maps it to our 1–5 Likert scale via linear rescaling.

Usage:
    python -m src.data.external_loader \\
        --input data/external/my_dataset.csv \\
        --dataset generic \\
        --output data/external/unified.jsonl \\
        --score_col overall_quality \\
        --prompt_col context \\
        --response_col reply
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Scale mapping utilities
# ---------------------------------------------------------------------------

def rescale_to_1_5(value: float, src_min: float, src_max: float) -> float:
    """Linearly rescale *value* from [src_min, src_max] → [1, 5]."""
    if src_max == src_min:
        return 3.0
    normed = (value - src_min) / (src_max - src_min)
    return round(1.0 + normed * 4.0, 2)


def _make_id(text: str) -> str:
    """Deterministic short hash for item_id generation."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Format-specific loaders
# ---------------------------------------------------------------------------

def load_generic_csv(
    path: str | Path,
    *,
    prompt_col: str = "prompt",
    response_col: str = "response",
    score_col: str = "overall",
    score_min: float = 1.0,
    score_max: float = 5.0,
    id_col: str | None = None,
    delimiter: str = ",",
) -> list[dict]:
    """Load any CSV with prompt/response/score columns.

    Scores outside [score_min, score_max] are clipped before rescaling.
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            prompt = row.get(prompt_col, "").strip()
            response = row.get(response_col, "").strip()
            raw_score = row.get(score_col, "")
            if not prompt or not response or not raw_score:
                continue
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                continue

            score = max(score_min, min(score_max, score))
            overall = rescale_to_1_5(score, score_min, score_max)

            item_id = row.get(id_col, "") if id_col else ""
            if not item_id:
                item_id = _make_id(f"{prompt}||{response}")

            records.append({
                "item_id": str(item_id),
                "prompt": prompt,
                "response": response,
                "human_overall": overall,
                "source": Path(path).stem,
            })
    return records


def load_generic_jsonl(
    path: str | Path,
    *,
    prompt_col: str = "prompt",
    response_col: str = "response",
    score_col: str = "overall",
    score_min: float = 1.0,
    score_max: float = 5.0,
    id_col: str | None = None,
) -> list[dict]:
    """Load any JSONL with prompt/response/score fields."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt = str(row.get(prompt_col, "")).strip()
            response = str(row.get(response_col, "")).strip()
            raw_score = row.get(score_col)
            if not prompt or not response or raw_score is None:
                continue
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                continue

            score = max(score_min, min(score_max, score))
            overall = rescale_to_1_5(score, score_min, score_max)

            item_id = str(row.get(id_col, "")) if id_col else ""
            if not item_id:
                item_id = _make_id(f"{prompt}||{response}")

            records.append({
                "item_id": str(item_id),
                "prompt": prompt,
                "response": response,
                "human_overall": overall,
                "source": Path(path).stem,
            })
    return records


def load_epitome(path: str | Path) -> list[dict]:
    """Load EPITOME-style dataset (CSV with empathy dimensions scored 0/1/2).

    Expected columns: seeker_post, response_post,
                      emotional_reactions, explorations, interpretations
    Rescales 0–6 (sum of 3 binary dimensions) → 1–5.
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt = row.get("seeker_post", "").strip()
            response = row.get("response_post", "").strip()
            if not prompt or not response:
                continue
            try:
                er = int(row.get("emotional_reactions", 0))
                ex = int(row.get("explorations", 0))
                ip = int(row.get("interpretations", 0))
            except (TypeError, ValueError):
                continue

            # Sum of three 0–2 dimensions → 0–6, rescale to 1–5
            total = er + ex + ip
            overall = rescale_to_1_5(total, 0, 6)

            item_id = _make_id(f"{prompt}||{response}")
            records.append({
                "item_id": item_id,
                "prompt": prompt,
                "response": response,
                "human_overall": overall,
                "human_scores": {
                    "emotional_reactions": er,
                    "explorations": ex,
                    "interpretations": ip,
                },
                "source": "epitome",
            })
    return records


def load_empatheticdialogues_eval(path: str | Path) -> list[dict]:
    """Load EmpatheticDialogues-EVAL or similar pre-rated CSV.

    Expected columns: context / prompt, response, empathy_score (1-5)
    Also accepts user_statement + therapist_response naming.
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Try multiple column name conventions
            prompt = (
                row.get("context", "")
                or row.get("prompt", "")
                or row.get("user_statement", "")
                or row.get("current_statement", "")
            ).strip()

            response = (
                row.get("response", "")
                or row.get("therapist_response", "")
                or row.get("reply", "")
            ).strip()

            raw = (
                row.get("empathy_score", "")
                or row.get("overall", "")
                or row.get("quality", "")
                or row.get("rating", "")
            )

            if not prompt or not response or not raw:
                continue
            try:
                score = float(raw)
            except (TypeError, ValueError):
                continue

            # Assume 1–5 scale by default; if we detect 0–4 or 1–7 etc.,
            # the caller should use generic loader with explicit bounds
            overall = max(1.0, min(5.0, score))

            item_id = row.get("id", "") or row.get("item_id", "")
            if not item_id:
                item_id = _make_id(f"{prompt}||{response}")

            records.append({
                "item_id": str(item_id),
                "prompt": prompt,
                "response": response,
                "human_overall": round(overall, 2),
                "source": "empatheticdialogues_eval",
            })
    return records


# ---------------------------------------------------------------------------
# Unified loader (auto-detect or explicit)
# ---------------------------------------------------------------------------

DATASET_LOADERS = {
    "epitome": load_epitome,
    "empatheticdialogues_eval": load_empatheticdialogues_eval,
}


def load_external(
    path: str | Path,
    *,
    dataset: str = "generic",
    prompt_col: str = "prompt",
    response_col: str = "response",
    score_col: str = "overall",
    score_min: float = 1.0,
    score_max: float = 5.0,
    id_col: str | None = None,
) -> list[dict]:
    """Unified entry point: load any external dataset into standard format.

    Parameters
    ----------
    path : path to CSV or JSONL file
    dataset : one of "generic", "epitome", "empatheticdialogues_eval"
    prompt_col, response_col, score_col : column names for generic loader
    score_min, score_max : source scale bounds (for generic loader rescaling)
    id_col : column name for item ID (optional, auto-generated if absent)

    Returns
    -------
    list[dict] with keys: item_id, prompt, response, human_overall (1–5), source
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"External data not found: {path}")

    # Named dataset loaders
    if dataset in DATASET_LOADERS:
        records = DATASET_LOADERS[dataset](path)
    elif str(path).endswith(".jsonl"):
        records = load_generic_jsonl(
            path,
            prompt_col=prompt_col,
            response_col=response_col,
            score_col=score_col,
            score_min=score_min,
            score_max=score_max,
            id_col=id_col,
        )
    else:
        # Default: CSV
        records = load_generic_csv(
            path,
            prompt_col=prompt_col,
            response_col=response_col,
            score_col=score_col,
            score_min=score_min,
            score_max=score_max,
            id_col=id_col,
        )

    print(f"Loaded {len(records)} records from {path} (dataset={dataset})")
    return records


def save_unified_jsonl(records: list[dict], path: str | Path) -> None:
    """Write records to unified JSONL format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Saved {len(records)} unified records → {path}")


def convert_to_generation_format(records: list[dict]) -> list[dict]:
    """Convert unified external records to the generation format expected by
    ``judge_batch`` (i.e. with ``id``, ``user_statement``, ``response``, ``model``).
    """
    generations = []
    for rec in records:
        generations.append({
            "id": rec["item_id"],
            "user_statement": rec["prompt"],
            "response": rec["response"],
            "model": f"external_{rec.get('source', 'unknown')}",
        })
    return generations


def convert_to_human_labels(records: list[dict], annotator_id: str = "external_human") -> list[dict]:
    """Convert unified external records to the human-labels format expected by
    ``merge_human_and_judge``.

    Since external data usually only has an overall score, we replicate it
    across all 4 dimensions as a reasonable proxy.
    """
    labels = []
    for rec in records:
        overall = rec["human_overall"]
        # If external data has per-dimension human scores, use them;
        # otherwise replicate overall to all dimensions
        h_scores = rec.get("human_scores", {})
        from src.eval.rubric import DIMENSION_KEYS
        label = {
            "sample_id": rec["item_id"],
            "annotator_id": annotator_id,
        }
        for dim in DIMENSION_KEYS:
            label[dim] = h_scores.get(dim, overall)
        label["overall"] = overall
        label["notes"] = f"external:{rec.get('source', 'unknown')}"
        labels.append(label)
    return labels


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Load external dataset and convert to unified JSONL"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to external CSV or JSONL file")
    parser.add_argument("--output", type=str, default="data/external/unified.jsonl",
                        help="Output path for unified JSONL")
    parser.add_argument("--dataset", type=str, default="generic",
                        choices=["generic", "epitome", "empatheticdialogues_eval"],
                        help="Dataset format (default: generic)")
    parser.add_argument("--prompt_col", default="prompt")
    parser.add_argument("--response_col", default="response")
    parser.add_argument("--score_col", default="overall")
    parser.add_argument("--score_min", type=float, default=1.0)
    parser.add_argument("--score_max", type=float, default=5.0)
    parser.add_argument("--id_col", default=None)
    args = parser.parse_args()

    records = load_external(
        args.input,
        dataset=args.dataset,
        prompt_col=args.prompt_col,
        response_col=args.response_col,
        score_col=args.score_col,
        score_min=args.score_min,
        score_max=args.score_max,
        id_col=args.id_col,
    )

    save_unified_jsonl(records, args.output)

    # Show sample
    if records:
        print("\nSample record:")
        print(json.dumps(records[0], indent=2, ensure_ascii=False))
        print(f"\nScore distribution:")
        scores = [r["human_overall"] for r in records]
        import numpy as np
        print(f"  min={min(scores):.2f}  max={max(scores):.2f}  "
              f"mean={np.mean(scores):.2f}  std={np.std(scores):.2f}")


if __name__ == "__main__":
    main()
