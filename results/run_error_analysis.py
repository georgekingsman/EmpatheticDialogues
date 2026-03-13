#!/usr/bin/env python3
"""Error analysis for offline evaluation v1.

Identifies four error types:
  Type A — Single Agent misses escalation on high-risk scenarios
  Type B — Checker makes response more template/cold (Emotion/Validation drops)
  Type C — Checker over-escalates medium-risk (false positive escalation)
  Type D — Visible Checker makes response sound bureaucratic

For each type, selects 2–3 concrete examples for qualitative analysis.

Usage:
    python results/run_error_analysis.py
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "results" / "offline_eval_v1"
OUT = DATA / "error_analysis.json"

DIMS = ["emotion", "validation", "helpfulness", "safety",
        "boundary_adherence", "escalation"]


def load_judge():
    by_id = defaultdict(dict)
    with open(DATA / "judge_scores.csv") as f:
        for r in csv.DictReader(f):
            by_id[r["sample_id"]][r["condition"]] = {
                "risk_level": r["risk_level"],
                **{d: float(r[d]) for d in DIMS},
                "overall": float(r["overall"]) if r["overall"] else None,
                "notes": r.get("notes", ""),
            }
    return dict(by_id)


def load_generations():
    gens = defaultdict(dict)
    for fname, cond in [("outputs_A.jsonl", "single_agent"),
                         ("outputs_B.jsonl", "double_hidden"),
                         ("outputs_C.jsonl", "double_visible")]:
        with open(DATA / fname) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                gens[r["id"]][cond] = r
    return dict(gens)


def load_checker_actions():
    actions = defaultdict(dict)
    with open(DATA / "checker_actions.csv") as f:
        for r in csv.DictReader(f):
            actions[r["sample_id"]][r["condition"]] = r
    return dict(actions)


def run():
    judge = load_judge()
    gens = load_generations()
    checker = load_checker_actions()

    errors = {
        "type_A": {"description": "Single Agent misses escalation on high-risk",
                    "examples": []},
        "type_B": {"description": "Checker reduces warmth (Emotion/Validation drop)",
                    "examples": []},
        "type_C": {"description": "Checker over-escalates medium-risk (false positive)",
                    "examples": []},
        "type_D": {"description": "Visible Checker makes response sound bureaucratic",
                    "examples": []},
    }

    # ── Type A: Single Agent low escalation on high-risk ──
    type_a_candidates = []
    for sid, conds in judge.items():
        if "single_agent" not in conds:
            continue
        a = conds["single_agent"]
        if a["risk_level"] != "high":
            continue
        if a["escalation"] <= 3:
            score_gap = 0
            for c in ["double_hidden", "double_visible"]:
                if c in conds:
                    score_gap = max(score_gap, conds[c]["escalation"] - a["escalation"])
            type_a_candidates.append((sid, a["escalation"], score_gap))

    type_a_candidates.sort(key=lambda t: (-t[2], t[1]))
    for sid, esc_score, gap in type_a_candidates[:3]:
        example = {
            "sample_id": sid,
            "risk_level": "high",
            "A_escalation": esc_score,
            "escalation_gap_vs_checker": gap,
        }
        if sid in gens and "single_agent" in gens[sid]:
            g = gens[sid]["single_agent"]
            example["user_utterance"] = g.get("user_utterance", "")
            example["A_response"] = g.get("response", "")
        for c in ["double_hidden", "double_visible"]:
            if sid in gens and c in gens[sid]:
                example[f"{c}_response"] = gens[sid][c].get("response", "")
            if sid in judge and c in judge[sid]:
                example[f"{c}_escalation"] = judge[sid][c]["escalation"]
        if sid in judge and "single_agent" in judge[sid]:
            example["A_notes"] = judge[sid]["single_agent"]["notes"]
        errors["type_A"]["examples"].append(example)

    # ── Type B: Checker drops Emotion/Validation ──
    type_b_candidates = []
    for sid, conds in judge.items():
        if "single_agent" not in conds:
            continue
        a = conds["single_agent"]
        for c in ["double_hidden", "double_visible"]:
            if c not in conds:
                continue
            chk = conds[c]
            emo_drop = a["emotion"] - chk["emotion"]
            val_drop = a["validation"] - chk["validation"]
            warmth_drop = emo_drop + val_drop
            if warmth_drop >= 1.0:
                type_b_candidates.append((sid, c, emo_drop, val_drop, warmth_drop))

    type_b_candidates.sort(key=lambda t: -t[4])
    seen_b = set()
    for sid, cond, emo_drop, val_drop, warmth_drop in type_b_candidates:
        if sid in seen_b:
            continue
        seen_b.add(sid)
        if len(errors["type_B"]["examples"]) >= 3:
            break
        example = {
            "sample_id": sid,
            "risk_level": judge[sid]["single_agent"]["risk_level"],
            "condition": cond,
            "emotion_drop": emo_drop,
            "validation_drop": val_drop,
            "warmth_composite_drop": warmth_drop,
        }
        if sid in gens:
            if "single_agent" in gens[sid]:
                example["A_response"] = gens[sid]["single_agent"].get("response", "")
            if cond in gens[sid]:
                example["checker_response"] = gens[sid][cond].get("response", "")
                example["user_utterance"] = gens[sid][cond].get("user_utterance", "")
        errors["type_B"]["examples"].append(example)

    # ── Type C: Checker escalates medium-risk (false positive) ──
    type_c_candidates = []
    for sid, conds in checker.items():
        for c in ["double_hidden", "double_visible"]:
            if c not in conds:
                continue
            act = conds[c]
            if act["risk_level"] == "medium" and act.get("checker_decision") == "escalate":
                type_c_candidates.append((sid, c))

    for sid, cond in type_c_candidates[:3]:
        example = {
            "sample_id": sid,
            "risk_level": "medium",
            "condition": cond,
            "checker_decision": "escalate",
        }
        if sid in gens and cond in gens[sid]:
            example["user_utterance"] = gens[sid][cond].get("user_utterance", "")
            example["response"] = gens[sid][cond].get("response", "")
            example["maker_draft"] = gens[sid][cond].get("maker_draft", "")
        if sid in judge and cond in judge[sid]:
            example["judge_scores"] = {d: judge[sid][cond][d] for d in DIMS}
        errors["type_C"]["examples"].append(example)

    # ── Type D: Visible Checker sounds bureaucratic ──
    # Identify cases where C has lower emotion than B (visible indicator hurts warmth)
    type_d_candidates = []
    for sid, conds in judge.items():
        if "double_hidden" not in conds or "double_visible" not in conds:
            continue
        b = conds["double_hidden"]
        c = conds["double_visible"]
        emo_diff = b["emotion"] - c["emotion"]
        val_diff = b["validation"] - c["validation"]
        if emo_diff > 0 or val_diff > 0:
            type_d_candidates.append((sid, emo_diff, val_diff, emo_diff + val_diff))

    type_d_candidates.sort(key=lambda t: -t[3])
    for sid, emo_diff, val_diff, total in type_d_candidates[:3]:
        example = {
            "sample_id": sid,
            "risk_level": judge[sid]["double_visible"]["risk_level"],
            "B_emotion": judge[sid]["double_hidden"]["emotion"],
            "C_emotion": judge[sid]["double_visible"]["emotion"],
            "B_validation": judge[sid]["double_hidden"]["validation"],
            "C_validation": judge[sid]["double_visible"]["validation"],
        }
        if sid in gens:
            if "double_hidden" in gens[sid]:
                example["B_response"] = gens[sid]["double_hidden"].get("response", "")
            if "double_visible" in gens[sid]:
                example["C_response"] = gens[sid]["double_visible"].get("response", "")
                example["visible_indicator"] = gens[sid]["double_visible"].get("visible_indicator", "")
                example["visible_detail"] = gens[sid]["double_visible"].get("visible_detail", "")
                example["user_utterance"] = gens[sid]["double_visible"].get("user_utterance", "")
        errors["type_D"]["examples"].append(example)

    # ── Summary counts ──
    summary = {
        "type_A_total_missed_escalations": len(type_a_candidates),
        "type_B_total_warmth_drops": len(set(t[0] for t in type_b_candidates)),
        "type_C_total_false_escalations": len(type_c_candidates),
        "type_D_total_visible_worse_than_hidden": len(type_d_candidates),
    }
    errors["summary"] = summary

    # ── Write ──
    OUT.write_text(json.dumps(errors, indent=2, ensure_ascii=False))
    print(f"[✓] Error analysis written to {OUT}\n")

    print("=" * 70)
    print("ERROR ANALYSIS SUMMARY")
    print("=" * 70)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print()
    for etype in ["type_A", "type_B", "type_C", "type_D"]:
        info = errors[etype]
        print(f"  {etype}: {info['description']}")
        print(f"    Examples: {len(info['examples'])}")
        for ex in info["examples"]:
            print(f"      - {ex['sample_id']} (risk={ex.get('risk_level', '?')})")
        print()

    return errors


if __name__ == "__main__":
    run()
