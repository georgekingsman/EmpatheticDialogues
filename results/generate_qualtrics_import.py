#!/usr/bin/env python3
"""
Generate Qualtrics-ready stimulus files for each counterbalancing cell.

For each of the 12 cells, produces a JSON with vignette order and the
exact text (user_utterance + AI response + optional safety indicator)
to display. This simplifies Qualtrics setup: import the JSON for each
cell as embedded data and pipe the fields into survey questions.

Output: results/qualtrics_import/cell_XX.json (12 files)
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "results" / "offline_eval_v2_final"
OUT = ROOT / "results" / "qualtrics_import"

CONDITION_LABELS = {
    "A": "Single Agent",
    "B": "Hidden Checker",
    "C": "Visible Checker",
}

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # Load stimuli
    with open(DATA / "user_study_stimuli.json") as f:
        stimuli = json.load(f)
    stim_by_id = {s["vignette_id"]: s for s in stimuli}

    # Load counterbalancing matrix
    with open(DATA / "counterbalancing_matrix.json") as f:
        cb = json.load(f)

    cells = cb["counterbalancing_cells"]

    for cell in cells:
        cell_id = cell["cell_id"]
        cell_data = {
            "cell_id": cell_id,
            "vignettes": [],
        }

        for vid, assignment in cell["assignments"].items():
            sid = assignment["scenario_id"]
            condition = assignment["condition"]
            risk = assignment["risk_level"]
            topic = assignment["topic"]

            stim = stim_by_id.get(sid, {})
            user_utterance = stim.get("user_utterance", "")

            # Get the right response based on condition
            responses = stim.get("responses", {})
            if condition == "A":
                response_text = responses.get("A_single_agent", "")
                safety_indicator = ""
                show_indicator = False
            elif condition == "B":
                response_text = responses.get("B_hidden_checker", "")
                safety_indicator = ""
                show_indicator = False
            else:  # C
                c_data = responses.get("C_visible_checker", {})
                if isinstance(c_data, dict):
                    response_text = c_data.get("response", "")
                    safety_indicator = c_data.get("safety_indicator", "")
                else:
                    response_text = str(c_data)
                    safety_indicator = ""
                show_indicator = True

            vignette = {
                "vignette_id": vid,
                "scenario_id": sid,
                "condition": condition,
                "condition_label": CONDITION_LABELS[condition],
                "risk_level": risk,
                "topic": topic,
                "user_utterance": user_utterance,
                "response_text": response_text,
                "show_safety_indicator": show_indicator,
                "safety_indicator_text": safety_indicator,
                # Qualtrics embedded data field names
                "qualtrics_fields": {
                    f"utterance_{vid}": user_utterance,
                    f"response_{vid}": response_text,
                    f"indicator_{vid}": safety_indicator if show_indicator else "",
                    f"show_indicator_{vid}": "1" if show_indicator else "0",
                    f"condition_{vid}": condition,
                    f"risk_{vid}": risk,
                },
            }
            cell_data["vignettes"].append(vignette)

        out_path = OUT / f"cell_{cell_id:02d}.json"
        with open(out_path, "w") as f:
            json.dump(cell_data, f, indent=2, ensure_ascii=False)

    # Also generate a flat CSV for easy Qualtrics import
    # Each row = one cell, columns = embedded data fields for all 12 vignettes
    import csv
    csv_path = OUT / "qualtrics_embedded_data.csv"
    all_fields = set()
    rows = []
    for cell in cells:
        cell_id = cell["cell_id"]
        with open(OUT / f"cell_{cell_id:02d}.json") as f:
            cd = json.load(f)
        row = {"cell_id": cell_id}
        for v in cd["vignettes"]:
            row.update(v["qualtrics_fields"])
            all_fields.update(v["qualtrics_fields"].keys())
        rows.append(row)

    fieldnames = ["cell_id"] + sorted(all_fields)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(cells)} cell files in {OUT.relative_to(ROOT)}/")
    print(f"Flat CSV: {csv_path.relative_to(ROOT)}")
    print(f"\nQualtrics setup:")
    print(f"  1. Import qualtrics_embedded_data.csv as contact list")
    print(f"  2. Set cell_id as panel variable (URL param or manual)")
    print(f"  3. Pipe ${{e://Field/utterance_V01}} etc. into question text")
    print(f"  4. Use display logic: show_indicator_VXX == 1 for safety badge")


if __name__ == "__main__":
    main()
