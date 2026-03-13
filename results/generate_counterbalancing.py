#!/usr/bin/env python3
"""
Generate Latin-square counterbalancing matrix for 12-vignette × 3-condition user study.

Design:
- 12 vignettes (V01–V12): 4 low, 4 medium, 4 high risk
- 3 conditions: A (single), B (hidden checker), C (visible checker)
- Within-subjects: each participant sees all 12 vignettes
- Each participant sees exactly 4 vignettes per condition
- Within each risk level (4 vignettes), each condition appears at least once
- 12 counterbalancing cells → target 3 participants each = 36 total

Output: counterbalancing_matrix.json + human-readable summary
"""

import json
import itertools
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "offline_eval_v2_final"

VIGNETTES = [
    # LOW (V01-V04)
    {"vid": "V01", "sid": "S018", "risk": "low",    "topic": "work_stress"},
    {"vid": "V02", "sid": "S002", "risk": "low",    "topic": "loneliness"},
    {"vid": "V03", "sid": "S004", "risk": "low",    "topic": "academic_pressure"},
    {"vid": "V04", "sid": "S006", "risk": "low",    "topic": "relationship"},
    # MEDIUM (V05-V08)
    {"vid": "V05", "sid": "S047", "risk": "medium", "topic": "eating_disorder"},
    {"vid": "V06", "sid": "S033", "risk": "medium", "topic": "substance"},
    {"vid": "V07", "sid": "S040", "risk": "medium", "topic": "depression"},
    {"vid": "V08", "sid": "S045", "risk": "medium", "topic": "anxiety"},
    # HIGH (V09-V12)
    {"vid": "V09", "sid": "S063", "risk": "high",   "topic": "self_harm"},
    {"vid": "V10", "sid": "S076", "risk": "high",   "topic": "suicidal_ideation"},
    {"vid": "V11", "sid": "S081", "risk": "high",   "topic": "trauma"},
    {"vid": "V12", "sid": "S065", "risk": "high",   "topic": "domestic_violence"},
]

CONDITIONS = ["A", "B", "C"]

def generate_balanced_latin_square():
    """
    Create 12 counterbalancing cells.
    
    For each risk block of 4 vignettes, we use a balanced 4×3 assignment:
    - 3 possible balanced patterns for assigning 3 conditions to 4 slots 
      (each condition appears at ~1.33 times, so we use patterns like ABCA, BCAB, CABC)
    - We cross the 3 risk blocks to get 3^3 = 27, but select 12 that are balanced.
    
    Simpler approach: use 3 base rotations per risk block, cross them.
    """
    # For 4 vignettes and 3 conditions, balanced assignment = each condition appears 
    # at least once, two conditions appear once, one appears twice.
    # All permutations of [A,A,B,C], [B,B,A,C], [C,C,A,B] etc.
    
    # 3 balanced blocks per risk level:
    risk_assignments = [
        # Block 1: A appears twice
        ["A", "A", "B", "C"],
        # Block 2: B appears twice  
        ["B", "A", "B", "C"],
        # Block 3: C appears twice
        ["C", "A", "B", "C"],
    ]
    # Actually, let's ensure that across all 3 risk levels, each participant sees
    # exactly 4 of each condition (4A + 4B + 4C = 12 total).
    
    # Strategy: For each risk level (4 vignettes), assign conditions.
    # Constraint: total across 3 risk levels must be 4A, 4B, 4C.
    # Valid per-risk assignments (1-2 of each condition in 4 slots):
    # Type X: [X, X, Y, Z] where one condition gets 2, others get 1
    
    # If risk1 gives 2A,1B,1C → need 2 more A, 3 more B, 3 more C from risk2+risk3
    # risk2: 1A,2B,1C → remaining: 1A,1B,2C → risk3: 1A,1B,2C ✓
    # risk2: 1A,1B,2C → remaining: 1A,2B,1C → risk3: 1A,2B,1C ✓
    # risk2: 2A,1B,1C → remaining: 0A,2B,2C → risk3: 0A,2B,2C ✗ (need each ≥1)
    
    # Valid combinations (which condition gets 2 in each risk level):
    # (A,B,C), (A,C,B), (B,A,C), (B,C,A), (C,A,B), (C,B,A) = 6 base combos
    # But position within risk block also varies.
    
    # Simpler: enumerate all valid assignments and pick 12 that are presentation-balanced.
    
    # For a risk block of 4 vignettes, list all assignments where each condition ≥1:
    def valid_risk_assignments():
        """Return all ways to assign 3 conditions to 4 slots with each condition ≥ 1."""
        results = []
        for combo in itertools.product(CONDITIONS, repeat=4):
            counts = {c: combo.count(c) for c in CONDITIONS}
            if all(counts[c] >= 1 for c in CONDITIONS):
                results.append(list(combo))
        return results
    
    all_risk_patterns = valid_risk_assignments()  # 36 patterns
    
    # Find all valid full assignments (3 risk blocks) where total = 4A,4B,4C
    valid_full = []
    for p1 in all_risk_patterns:
        for p2 in all_risk_patterns:
            for p3 in all_risk_patterns:
                full = p1 + p2 + p3
                counts = {c: full.count(c) for c in CONDITIONS}
                if counts["A"] == 4 and counts["B"] == 4 and counts["C"] == 4:
                    valid_full.append((tuple(p1), tuple(p2), tuple(p3)))
    
    print(f"Total valid full assignments: {len(valid_full)}")
    
    # Select 12 that maximize presentation balance:
    # Each vignette position should see each condition ~4 times across 12 cells
    # (12 cells / 3 conditions = 4 per condition per position)
    
    # Greedy selection: pick assignments that improve position balance
    import random
    random.seed(42)
    
    best_selection = None
    best_score = float('inf')
    
    for _ in range(5000):
        random.shuffle(valid_full)
        selected = []
        position_counts = [[0, 0, 0] for _ in range(12)]  # 12 positions × 3 conditions
        
        for assignment in valid_full:
            full = list(assignment[0]) + list(assignment[1]) + list(assignment[2])
            # Check if adding this would keep all position counts ≤ 4
            ok = True
            for i, c in enumerate(full):
                ci = CONDITIONS.index(c)
                if position_counts[i][ci] >= 4:
                    ok = False
                    break
            if ok:
                selected.append(assignment)
                for i, c in enumerate(full):
                    ci = CONDITIONS.index(c)
                    position_counts[i][ci] += 1
            if len(selected) >= 12:
                break
        
        if len(selected) >= 12:
            # Score: variance of position counts (want all to be 4)
            score = sum((position_counts[i][j] - 4)**2 
                       for i in range(12) for j in range(3))
            if score < best_score:
                best_score = score
                best_selection = selected[:]
    
    return best_selection

def main():
    cells = generate_balanced_latin_square()
    
    if not cells:
        print("ERROR: Could not find balanced selection")
        return
    
    print(f"\nSelected {len(cells)} counterbalancing cells (balance score: lower=better)")
    
    # Build the matrix
    matrix = []
    for cell_idx, (p1, p2, p3) in enumerate(cells):
        full = list(p1) + list(p2) + list(p3)
        cell = {
            "cell_id": cell_idx + 1,
            "assignments": {}
        }
        for i, v in enumerate(VIGNETTES):
            cell["assignments"][v["vid"]] = {
                "scenario_id": v["sid"],
                "condition": full[i],
                "risk_level": v["risk"],
                "topic": v["topic"],
            }
        matrix.append(cell)
    
    # Print summary
    print("\n=== COUNTERBALANCING MATRIX ===\n")
    header = "Cell | " + " | ".join(v["vid"] for v in VIGNETTES) + " | #A #B #C"
    print(header)
    print("-" * len(header))
    for cell in matrix:
        row = f" {cell['cell_id']:2d}  | "
        conditions = []
        for v in VIGNETTES:
            c = cell["assignments"][v["vid"]]["condition"]
            conditions.append(c)
            row += f" {c}  | "
        counts = {c: conditions.count(c) for c in CONDITIONS}
        row += f" {counts['A']}  {counts['B']}  {counts['C']}"
        print(row)
    
    # Check position balance
    print("\n=== POSITION BALANCE ===")
    print("(How many times each condition appears at each vignette position across 12 cells)")
    for i, v in enumerate(VIGNETTES):
        counts = {c: 0 for c in CONDITIONS}
        for cell in matrix:
            c = cell["assignments"][v["vid"]]["condition"]
            counts[c] += 1
        print(f"  {v['vid']} ({v['risk']:6s} {v['topic']:20s}): A={counts['A']} B={counts['B']} C={counts['C']}")
    
    # Save
    output = {
        "design": {
            "n_vignettes": 12,
            "n_conditions": 3,
            "n_cells": len(matrix),
            "target_n_per_cell": 3,
            "target_total_n": 36,
            "conditions": {
                "A": "Single Agent (no checker)",
                "B": "Double AI - Hidden Checker",
                "C": "Double AI - Visible Checker",
            },
        },
        "vignettes": VIGNETTES,
        "counterbalancing_cells": matrix,
    }
    
    out_path = OUT / "counterbalancing_matrix.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[✓] Saved to {out_path.name}")

if __name__ == "__main__":
    main()
