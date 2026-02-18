"""
Schema and utilities for human annotation labels.

Defines the expected CSV / JSONL schema, validation logic, and
inter-annotator agreement computation.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.eval.rubric import DIMENSION_KEYS, VALID_SCORES

# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------

HUMAN_LABEL_FIELDS = [
    "sample_id",        # matches generation output "id"
    "annotator_id",     # anonymised annotator identifier
    "emotion",          # 1–5
    "validation",       # 1–5
    "helpfulness",      # 1–5
    "safety",           # 1–5
    "overall",          # 1–5 (optional holistic score)
    "notes",            # free-text rationale (optional)
]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_label(label: dict) -> list[str]:
    """Return a list of validation errors (empty if valid)."""
    errors = []
    if not label.get("sample_id"):
        errors.append("missing sample_id")
    if not label.get("annotator_id"):
        errors.append("missing annotator_id")
    for key in DIMENSION_KEYS:
        val = label.get(key)
        if val is None:
            errors.append(f"missing {key}")
        else:
            try:
                val = int(val)
            except (TypeError, ValueError):
                errors.append(f"invalid type for {key}: {val}")
                continue
            if val not in VALID_SCORES:
                errors.append(f"{key}={val} out of range 1-5")
    return errors


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_human_labels_csv(path: str | Path) -> list[dict]:
    """Load a CSV of human annotations and validate."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            # Cast numeric fields
            for key in DIMENSION_KEYS + ["overall"]:
                if key in row and row[key]:
                    try:
                        row[key] = int(row[key])
                    except ValueError:
                        pass
            errs = validate_label(row)
            if errs:
                print(f"  Warning: row {i}: {errs}")
            records.append(row)
    return records


def save_human_labels_csv(records: list[dict], path: str | Path) -> None:
    """Save annotations to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HUMAN_LABEL_FIELDS)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec.get(k, "") for k in HUMAN_LABEL_FIELDS})
    print(f"Saved {len(records)} human labels → {path}")


def generate_blank_annotation_sheet(
    generation_path: str | Path,
    output_path: str | Path,
    annotator_id: str = "",
) -> None:
    """Create a blank CSV annotation sheet from generation JSONL."""
    records: list[dict] = []
    with open(generation_path, "r") as f:
        for line in f:
            gen = json.loads(line.strip())
            records.append(
                {
                    "sample_id": gen["id"],
                    "annotator_id": annotator_id,
                    "emotion": "",
                    "validation": "",
                    "helpfulness": "",
                    "safety": "",
                    "overall": "",
                    "notes": "",
                }
            )
    save_human_labels_csv(records, output_path)


# ---------------------------------------------------------------------------
# Inter-annotator agreement
# ---------------------------------------------------------------------------

def compute_iaa(
    labels: list[dict],
) -> dict[str, float]:
    """Compute pairwise inter-annotator agreement (Cohen's kappa per dimension).

    Expects labels list with multiple annotator_ids per sample_id.
    Returns dict[dimension_key → kappa].
    """
    from collections import defaultdict

    try:
        from sklearn.metrics import cohen_kappa_score
    except ImportError:
        print("sklearn required for IAA computation. pip install scikit-learn")
        return {}

    # Group by sample_id
    by_sample: dict[str, list[dict]] = defaultdict(list)
    for lab in labels:
        by_sample[lab["sample_id"]].append(lab)

    # Collect pairs for samples with exactly 2 annotators
    a1_scores: dict[str, list[int]] = {k: [] for k in DIMENSION_KEYS}
    a2_scores: dict[str, list[int]] = {k: [] for k in DIMENSION_KEYS}

    for sid, anns in by_sample.items():
        if len(anns) != 2:
            continue
        for key in DIMENSION_KEYS:
            try:
                a1_scores[key].append(int(anns[0][key]))
                a2_scores[key].append(int(anns[1][key]))
            except (KeyError, TypeError, ValueError):
                continue

    kappas = {}
    for key in DIMENSION_KEYS:
        if len(a1_scores[key]) >= 10:
            kappas[key] = round(cohen_kappa_score(a1_scores[key], a2_scores[key]), 4)
        else:
            kappas[key] = float("nan")

    return kappas
