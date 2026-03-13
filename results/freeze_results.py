#!/usr/bin/env python3
"""Freeze offline evaluation v1 results into results/offline_eval_v1/.

Run once:  python results/freeze_results.py
"""
import csv
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "offline_eval_v1"
OUT.mkdir(parents=True, exist_ok=True)

# ── 1. scenarios.csv ──────────────────────────────────────────────
src = ROOT / "data" / "scenarios" / "benchmark.jsonl"
rows = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
cols = ["id", "risk_level", "user_utterance", "topic", "context_notes",
        "recommended_strategy", "escalation_expected", "boundary_notes"]
with open(OUT / "scenarios.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
print(f"[✓] scenarios.csv  ({len(rows)} rows)")

# ── 2. Copy generation JSONL files ───────────────────────────────
for tag, dst_name in [("single_agent", "outputs_A.jsonl"),
                       ("double_hidden", "outputs_B.jsonl"),
                       ("double_visible", "outputs_C.jsonl")]:
    shutil.copy2(ROOT / "outputs" / "generations" / f"{tag}.jsonl", OUT / dst_name)
    print(f"[✓] {dst_name}")

# ── 3. judge_scores.csv ──────────────────────────────────────────
eval_path = ROOT / "outputs" / "analysis" / "offline_evaluation.jsonl"
records = [json.loads(line) for line in eval_path.read_text().splitlines() if line.strip()]
dims = ["emotion", "validation", "helpfulness", "safety", "boundary_adherence", "escalation"]
judge_cols = ["sample_id", "condition", "risk_level", "repeat_idx"] + dims + ["overall", "confidence", "notes"]
with open(OUT / "judge_scores.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=judge_cols)
    w.writeheader()
    for r in records:
        row = {
            "sample_id": r["sample_id"],
            "condition": r["condition"],
            "risk_level": r["risk_level"],
            "repeat_idx": r.get("repeat_idx", 0),
            "overall": r.get("overall", ""),
            "confidence": r.get("confidence", ""),
            "notes": r.get("notes", ""),
        }
        row.update(r["scores"])
        w.writerow(row)
print(f"[✓] judge_scores.csv  ({len(records)} rows)")

# ── 4. checker_actions.csv ────────────────────────────────────────
checker_cols = ["sample_id", "condition", "risk_level", "checker_decision",
                "policy_action", "revision_count",
                "emotional_appropriateness", "safety_boundaries",
                "uncertainty_grounding", "escalation_need",
                "checker_flags"]
with open(OUT / "checker_actions.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=checker_cols)
    w.writeheader()
    for tag, dst_name in [("double_hidden", "outputs_B.jsonl"),
                           ("double_visible", "outputs_C.jsonl")]:
        gen_path = ROOT / "outputs" / "generations" / f"{tag}.jsonl"
        for line in gen_path.read_text().splitlines():
            if not line.strip():
                continue
            g = json.loads(line)
            row = {
                "sample_id": g["id"],
                "condition": g["condition"],
                "risk_level": g["risk_level"],
                "checker_decision": g.get("checker_decision", ""),
                "policy_action": g.get("policy_action", ""),
                "revision_count": g.get("revision_count", 0),
                "checker_flags": json.dumps(g.get("checker_flags", [])),
            }
            cs = g.get("checker_scores", {})
            row["emotional_appropriateness"] = cs.get("emotional_appropriateness", "")
            row["safety_boundaries"] = cs.get("safety_boundaries", "")
            row["uncertainty_grounding"] = cs.get("uncertainty_grounding", "")
            row["escalation_need"] = cs.get("escalation_need", "")
            w.writerow(row)
n_checker = sum(1 for _ in open(OUT / "checker_actions.csv")) - 1
print(f"[✓] checker_actions.csv  ({n_checker} rows)")

print("\n✅  All results frozen to", OUT)
