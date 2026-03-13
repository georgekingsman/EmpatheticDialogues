#!/usr/bin/env python3
"""
Generate simulated user study data to validate the analysis script.

Simulates N=36 participants (3 per cell × 12 cells) with response
patterns based on offline evaluation priors:
- A: highest empathy/warmth, lower safety/boundary
- B: moderate empathy, higher safety
- C: moderate empathy, highest safety + transparency
- High-risk scenarios show larger condition effects

Outputs:
  results/user_study_data.csv      (per-vignette Likert ratings)
  results/user_study_post.csv      (post-study measures)
"""

import csv
import json
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "results" / "offline_eval_v2_final"
OUT = ROOT / "results"

random.seed(42)
np.random.seed(42)

# Condition effect profiles (mean shift from baseline of 4.5 on 1-7 scale)
# Based on offline priors, scaled to 7-point
CONDITION_EFFECTS = {
    "A": {
        "Q1_empathy": +0.8, "Q2_warmth": +0.7, "Q3_safety": -0.1,
        "Q4_boundary": -0.2, "Q5_transparency": -0.3, "Q6_trust": +0.1,
        "Q7_rely": +0.2, "Q8_seekhelp": -0.3,
    },
    "B": {
        "Q1_empathy": +0.2, "Q2_warmth": +0.1, "Q3_safety": +0.3,
        "Q4_boundary": +0.3, "Q5_transparency": -0.1, "Q6_trust": +0.2,
        "Q7_rely": +0.1, "Q8_seekhelp": +0.1,
    },
    "C": {
        "Q1_empathy": +0.1, "Q2_warmth": +0.0, "Q3_safety": +0.4,
        "Q4_boundary": +0.5, "Q5_transparency": +0.8, "Q6_trust": +0.3,
        "Q7_rely": +0.2, "Q8_seekhelp": +0.4,
    },
}

# Risk level modifiers (added to condition effects)
RISK_EFFECTS = {
    "low":    {"Q3_safety": +0.3, "Q8_seekhelp": -0.5},
    "medium": {"Q3_safety": +0.0, "Q8_seekhelp": +0.0},
    "high":   {"Q3_safety": -0.3, "Q8_seekhelp": +0.5, "Q1_empathy": -0.2},
}

ITEMS = ["Q1_empathy", "Q2_warmth", "Q3_safety", "Q4_boundary",
         "Q5_transparency", "Q6_trust", "Q7_rely", "Q8_seekhelp"]


def clamp(val, lo=1, hi=7):
    return max(lo, min(hi, round(val)))


def main():
    # Load counterbalancing
    with open(DATA / "counterbalancing_matrix.json") as f:
        cb = json.load(f)

    cells = cb["counterbalancing_cells"]
    vignettes = cb["vignettes"]

    rows = []
    participants = []
    pid = 0

    for cell in cells:
        cell_id = cell["cell_id"]
        # 3 participants per cell
        for rep in range(3):
            pid += 1
            participant_id = f"P{pid:03d}"
            # Random participant intercept (individual tendency)
            p_intercept = np.random.normal(0, 0.4)

            for vid, assignment in cell["assignments"].items():
                condition = assignment["condition"]
                risk = assignment["risk_level"]
                sid = assignment["scenario_id"]
                topic = assignment["topic"]

                row = {
                    "participant_id": participant_id,
                    "cell_id": cell_id,
                    "vignette_id": vid,
                    "scenario_id": sid,
                    "condition": condition,
                    "risk_level": risk,
                    "topic": topic,
                    "attention_check_passed": 1,
                    "completion_time_s": int(np.random.normal(1200, 200)),
                }

                for item in ITEMS:
                    base = 4.5
                    cond_eff = CONDITION_EFFECTS[condition].get(item, 0)
                    risk_eff = RISK_EFFECTS.get(risk, {}).get(item, 0)
                    # Condition × Risk interaction: effects amplified in high-risk
                    interaction = 0
                    if risk == "high":
                        interaction = cond_eff * 0.3  # 30% amplification
                    noise = np.random.normal(0, 0.7)
                    val = base + cond_eff + risk_eff + interaction + p_intercept + noise
                    row[item] = clamp(val)

                rows.append(row)

            # Post-study data
            participants.append({
                "participant_id": participant_id,
                "overall_satisfaction": clamp(np.random.normal(5.0, 0.8)),
                "crisis_comfort": clamp(np.random.normal(3.5, 1.2)),
                "mental_workload": clamp(np.random.normal(3.0, 1.0)),
                "referral_correct_count": min(7, max(0, int(np.random.normal(5.5, 1.0)))),
                "age_group": random.choice(["18-24", "25-34", "35-44", "45-54"]),
                "gender": random.choice(["Woman", "Man", "Non-binary"]),
                "mh_service_use": random.choice(["Yes", "No", "Prefer not to say"]),
                "chatbot_experience": random.choice(["Yes, regularly", "Yes, a few times", "No"]),
                "open_feedback_safety": "",
                "open_feedback_improve": "",
            })

    # Write per-vignette data
    fieldnames = ["participant_id", "cell_id", "vignette_id", "scenario_id",
                  "condition", "risk_level", "topic",
                  "attention_check_passed", "completion_time_s"] + ITEMS
    with open(OUT / "user_study_data.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Write post-study data
    post_fields = list(participants[0].keys())
    with open(OUT / "user_study_post.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=post_fields)
        writer.writeheader()
        writer.writerows(participants)

    print(f"Generated {len(rows)} observation rows for {pid} participants")
    print(f"  Per-vignette: {OUT / 'user_study_data.csv'}")
    print(f"  Post-study:   {OUT / 'user_study_post.csv'}")
    print(f"\nNow run: python results/analyse_user_study.py")


if __name__ == "__main__":
    main()
