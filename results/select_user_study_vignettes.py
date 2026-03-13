#!/usr/bin/env python3
"""Select 12 vignettes for user study and create stimulus package."""

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "results" / "offline_eval_v2_final"

def load_all():
    scenarios = []
    with open(DATA / "scenarios.csv") as f:
        for r in csv.DictReader(f):
            scenarios.append(r)
    
    scores = {}
    with open(DATA / "judge_scores_main.csv") as f:
        for r in csv.DictReader(f):
            key = (r["sample_id"], r["condition"])
            scores[key] = {d: float(r[d]) for d in 
                ["emotion","validation","helpfulness","safety","boundary_adherence","escalation"]}
    
    checker = {}
    with open(DATA / "checker_actions.csv") as f:
        for r in csv.DictReader(f):
            checker[(r["sample_id"], r["condition"])] = r["checker_decision"]
    
    outputs = {"single_agent": {}, "double_hidden": {}, "double_visible": {}}
    for cond, fname in [("single_agent", "outputs_A.jsonl"),
                         ("double_hidden", "outputs_B_hidden.jsonl"),
                         ("double_visible", "outputs_C_visible.jsonl")]:
        with open(DATA / fname) as f:
            for line in f:
                obj = json.loads(line)
                outputs[cond][obj["id"]] = obj
    
    return scenarios, scores, checker, outputs

def main():
    scenarios, scores, checker, outputs = load_all()
    
    by_risk = defaultdict(list)
    for s in scenarios:
        sid = s["id"]
        risk = s["risk_level"]
        topic = s["topic"]
        
        a = scores.get((sid, "single_agent"), {})
        b = scores.get((sid, "double_hidden"), {})
        c = scores.get((sid, "double_visible"), {})
        
        b_dec = checker.get((sid, "double_hidden"), "n/a")
        c_dec = checker.get((sid, "double_visible"), "n/a")
        
        emo_diff = a.get("emotion", 0) - b.get("emotion", 0)
        esc_diff = b.get("escalation", 0) - a.get("escalation", 0)
        
        by_risk[risk].append({
            "id": sid, "risk": risk, "topic": topic,
            "utterance": s["user_utterance"][:80],
            "full_utterance": s["user_utterance"],
            "a_emo": a.get("emotion", 0), "b_emo": b.get("emotion", 0),
            "a_esc": a.get("escalation", 0), "b_esc": b.get("escalation", 0),
            "b_dec": b_dec, "c_dec": c_dec,
            "emo_diff": emo_diff, "esc_diff": esc_diff,
            "discrim": abs(emo_diff) + abs(esc_diff),
        })
    
    print("=== CANDIDATE ANALYSIS ===\n")
    for risk in ["low", "medium", "high"]:
        items = sorted(by_risk[risk], key=lambda x: x["discrim"], reverse=True)
        print(f"--- {risk.upper()} RISK ({len(items)} total) ---")
        topics_seen = set()
        for item in items[:10]:
            flag = " *" if item["topic"] not in topics_seen else ""
            topics_seen.add(item["topic"])
            print(f"  {item['id']} [{item['topic']}]{flag} "
                  f"emo_diff={item['emo_diff']:.0f} esc_diff={item['esc_diff']:.0f} "
                  f"B_dec={item['b_dec']} | {item['utterance']}")
        print()
    
    # Selection criteria:
    # LOW (4): diverse topics, all approved, minimal condition differences
    # MEDIUM (4): diverse topics, mostly approved, some discrimination 
    # HIGH (4): diverse topics, mostly escalated, maximum discrimination
    
    # Pick specific IDs based on topic diversity and discrimination
    selected = {
        "low": [],
        "medium": [],
        "high": [],
    }
    
    # For all risk levels: pick 4 with max topic diversity, preferring higher discrimination
    for risk in ["low", "medium", "high"]:
        items = sorted(by_risk[risk], key=lambda x: x["discrim"], reverse=True)
        for item in items:
            topics_so_far = {s["topic"] for s in selected[risk]}
            if item["topic"] not in topics_so_far:
                selected[risk].append(item)
            if len(selected[risk]) >= 4:
                break
        # If we still need more (not enough unique topics), fill from top discriminators
        if len(selected[risk]) < 4:
            used_ids = {s["id"] for s in selected[risk]}
            for item in items:
                if item["id"] not in used_ids:
                    selected[risk].append(item)
                if len(selected[risk]) >= 4:
                    break
    
    print("\n=== SELECTED 12 VIGNETTES ===\n")
    all_selected = []
    for risk in ["low", "medium", "high"]:
        print(f"--- {risk.upper()} ---")
        for item in selected[risk]:
            print(f"  {item['id']} [{item['topic']}] B_dec={item['b_dec']} C_dec={item['c_dec']} "
                  f"| {item['utterance']}")
            all_selected.append(item)
    
    # Build stimulus package
    stimuli = []
    for item in all_selected:
        sid = item["id"]
        a_out = outputs["single_agent"].get(sid, {})
        b_out = outputs["double_hidden"].get(sid, {})
        c_out = outputs["double_visible"].get(sid, {})
        
        stimulus = {
            "vignette_id": sid,
            "risk_level": item["risk"],
            "topic": item["topic"],
            "user_utterance": item["full_utterance"],
            "responses": {
                "A_single_agent": a_out.get("response", ""),
                "B_hidden_checker": b_out.get("response", ""),
                "C_visible_checker": {
                    "response": c_out.get("response", ""),
                    "safety_indicator": "✓ Safety reviewed — " + (
                        "This response has been reviewed for safety and appropriateness."
                        if checker.get((sid, "double_visible")) == "approve"
                        else "This response includes professional resources based on a safety review."
                    ),
                },
            },
            "judge_scores": {
                "A": scores.get((sid, "single_agent"), {}),
                "B": scores.get((sid, "double_hidden"), {}),
                "C": scores.get((sid, "double_visible"), {}),
            },
            "checker_decisions": {
                "B": checker.get((sid, "double_hidden"), "n/a"),
                "C": checker.get((sid, "double_visible"), "n/a"),
            },
        }
        stimuli.append(stimulus)
    
    out_path = ROOT / "results" / "offline_eval_v2_final" / "user_study_stimuli.json"
    with open(out_path, "w") as f:
        json.dump(stimuli, f, indent=2, ensure_ascii=False)
    print(f"\n[✓] Saved {len(stimuli)} stimuli to {out_path.name}")
    
    # Print IDs for reference
    ids = [s["vignette_id"] for s in stimuli]
    print(f"\nSelected IDs: {ids}")

if __name__ == "__main__":
    main()
