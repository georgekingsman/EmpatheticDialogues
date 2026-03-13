#!/usr/bin/env python3
"""Generate human annotation spreadsheet for ceiling-effect verification.

Produces a CSV with 30 stratified samples × 3 conditions = 90 rows,
with the response text pre-filled and columns for human scoring.

Usage:
    python results/prepare_human_review.py
"""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "results" / "offline_eval_v1"
OUT = DATA / "human_review_sheet.csv"

REVIEW_DIMS = ["emotion", "validation", "safety"]


def main():
    # Load review sample IDs from ceiling audit
    audit = json.load(open(DATA / "ceiling_audit.json"))
    sample_ids = [r["sample_id"] for r in audit["human_review_sample"]]

    # Load scenarios
    scenarios = {}
    with open(DATA / "scenarios.csv") as f:
        for row in csv.DictReader(f):
            scenarios[row["id"]] = row

    # Load generations for all 3 conditions
    generations = {}
    for fname, cond in [("outputs_A.jsonl", "A_single_agent"),
                         ("outputs_B.jsonl", "B_double_hidden"),
                         ("outputs_C.jsonl", "C_double_visible")]:
        with open(DATA / fname) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                generations[(cond, r["id"])] = r

    # Load LLM judge scores for comparison
    judge = {}
    with open(DATA / "judge_scores.csv") as f:
        for row in csv.DictReader(f):
            cond_key = {"single_agent": "A_single_agent",
                        "double_hidden": "B_double_hidden",
                        "double_visible": "C_double_visible"}[row["condition"]]
            judge[(cond_key, row["sample_id"])] = row

    # Write annotation CSV
    fieldnames = [
        "row_id",
        "sample_id",
        "risk_level",
        "condition",
        "user_utterance",
        "response_text",
        # Human annotation columns (to be filled)
        "human_emotion",       # 1-5
        "human_validation",    # 1-5
        "human_safety",        # 1-5
        "human_notes",
        # LLM judge scores (for later comparison, hidden from annotator)
        "llm_emotion",
        "llm_validation",
        "llm_safety",
    ]

    rows = []
    row_id = 0
    for sid in sorted(sample_ids):
        for cond in ["A_single_agent", "B_double_hidden", "C_double_visible"]:
            row_id += 1
            gen = generations.get((cond, sid), {})
            jdg = judge.get((cond, sid), {})
            scenario = scenarios.get(sid, {})

            rows.append({
                "row_id": row_id,
                "sample_id": sid,
                "risk_level": scenario.get("risk_level", ""),
                "condition": cond,
                "user_utterance": gen.get("user_utterance",
                                          scenario.get("user_utterance", "")),
                "response_text": gen.get("response", ""),
                "human_emotion": "",
                "human_validation": "",
                "human_safety": "",
                "human_notes": "",
                "llm_emotion": jdg.get("emotion", ""),
                "llm_validation": jdg.get("validation", ""),
                "llm_safety": jdg.get("safety", ""),
            })

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"[✓] Human review sheet: {OUT}")
    print(f"    {len(rows)} rows ({len(sample_ids)} samples × 3 conditions)")
    print(f"    Dims to annotate: {REVIEW_DIMS}")
    print()
    print("Instructions for annotators:")
    print("  1. Score each response on emotion, validation, safety (1-5)")
    print("  2. Do NOT look at llm_* columns until after scoring")
    print("  3. Use human_notes for any qualitative observations")
    print("  4. The condition column should be HIDDEN from annotators")
    print("     (create a blind version without condition/llm columns)")

    # Also create a blind version (no condition label, no LLM scores)
    blind_out = DATA / "human_review_blind.csv"
    blind_fields = ["row_id", "sample_id", "risk_level",
                     "user_utterance", "response_text",
                     "human_emotion", "human_validation",
                     "human_safety", "human_notes"]
    with open(blind_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=blind_fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row[k] for k in blind_fields})

    print(f"\n[✓] Blind version (for annotators): {blind_out}")


if __name__ == "__main__":
    main()
