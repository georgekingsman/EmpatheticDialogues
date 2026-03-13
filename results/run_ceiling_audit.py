#!/usr/bin/env python3
"""Ceiling-effect & judge-bias audit for offline evaluation v1.

Checks:
  1. Score distribution histograms per condition × dimension
  2. Ceiling rate (% of scores = 5) per condition × dimension
  3. Zero-variance detection (all scores identical)
  4. Score entropy analysis (low entropy → suspicious uniformity)
  5. Cross-condition monotonicity check (does judge always rank A > B > C?)
  6. Generates human-review sample list (stratified by risk)

Usage:
    python results/run_ceiling_audit.py
"""

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "results" / "offline_eval_v1"
OUT = DATA / "ceiling_audit.json"

DIMS = ["emotion", "validation", "helpfulness", "safety",
        "boundary_adherence", "escalation"]
CONDITIONS = ["single_agent", "double_hidden", "double_visible"]
COND_LABELS = {"single_agent": "A", "double_hidden": "B", "double_visible": "C"}


def load_scores():
    """Return list of dicts from judge_scores.csv."""
    rows = []
    with open(DATA / "judge_scores.csv") as f:
        for row in csv.DictReader(f):
            for d in DIMS:
                row[d] = float(row[d])
            rows.append(row)
    return rows


def entropy(values):
    """Shannon entropy (bits) for a discrete distribution."""
    c = Counter(values)
    n = len(values)
    if n == 0:
        return 0.0
    h = 0.0
    for cnt in c.values():
        p = cnt / n
        if p > 0:
            h -= p * math.log2(p)
    return h


def run_audit():
    rows = load_scores()
    report = {}

    # ── 1. Ceiling rates ──
    ceiling = {}
    for cond in CONDITIONS:
        cl = COND_LABELS[cond]
        ceiling[cl] = {}
        cond_rows = [r for r in rows if r["condition"] == cond]
        for dim in DIMS:
            vals = [r[dim] for r in cond_rows]
            n = len(vals)
            at_5 = sum(1 for v in vals if v == 5.0)
            at_1 = sum(1 for v in vals if v == 1.0)
            ceiling[cl][dim] = {
                "n": n,
                "pct_at_5": round(100 * at_5 / n, 1) if n else 0,
                "pct_at_1": round(100 * at_1 / n, 1) if n else 0,
                "mean": round(np.mean(vals), 3),
                "std": round(np.std(vals), 3),
                "unique_values": sorted(set(vals)),
                "entropy_bits": round(entropy(vals), 3),
                "zero_variance": bool(np.std(vals) == 0.0),
            }
    report["ceiling_rates"] = ceiling

    # ── 2. Flagged dimensions (zero variance or > 95 % at ceiling) ──
    flags = []
    for cl in ceiling:
        for dim in DIMS:
            c = ceiling[cl][dim]
            if c["zero_variance"]:
                flags.append(f"{cl}.{dim}: ZERO VARIANCE (all scores = {c['mean']})")
            elif c["pct_at_5"] >= 95:
                flags.append(f"{cl}.{dim}: {c['pct_at_5']}% at ceiling (5)")
    report["flags"] = flags

    # ── 3. Score distributions ──
    distributions = {}
    for cond in CONDITIONS:
        cl = COND_LABELS[cond]
        distributions[cl] = {}
        cond_rows = [r for r in rows if r["condition"] == cond]
        for dim in DIMS:
            vals = [r[dim] for r in cond_rows]
            dist = Counter(vals)
            distributions[cl][dim] = {str(k): v for k, v in sorted(dist.items())}
    report["score_distributions"] = distributions

    # ── 4. Risk-stratified ceiling rates ──
    risk_ceiling = {}
    for risk in ["low", "medium", "high"]:
        risk_ceiling[risk] = {}
        for cond in CONDITIONS:
            cl = COND_LABELS[cond]
            risk_ceiling[risk][cl] = {}
            sub = [r for r in rows if r["condition"] == cond and r["risk_level"] == risk]
            for dim in DIMS:
                vals = [r[dim] for r in sub]
                n = len(vals)
                at_5 = sum(1 for v in vals if v == 5.0)
                risk_ceiling[risk][cl][dim] = {
                    "n": n,
                    "pct_at_5": round(100 * at_5 / n, 1) if n else 0,
                    "mean": round(np.mean(vals), 3),
                    "std": round(np.std(vals), 3),
                    "zero_variance": bool(np.std(vals) == 0.0),
                }
    report["ceiling_by_risk"] = risk_ceiling

    # ── 5. Cross-condition monotonicity (per sample) ──
    # For each sample, does A always get highest score? (sign of judge bias)
    by_sample = defaultdict(dict)
    for r in rows:
        by_sample[r["sample_id"]][r["condition"]] = r

    monotone_stats = {}
    for dim in DIMS:
        a_always_max = 0
        a_always_min = 0
        n_complete = 0
        for sid, conds in by_sample.items():
            if len(conds) < 3:
                continue
            n_complete += 1
            sa = conds["single_agent"][dim]
            sb = conds["double_hidden"][dim]
            sc = conds["double_visible"][dim]
            if sa >= sb and sa >= sc:
                a_always_max += 1
            if sa <= sb and sa <= sc:
                a_always_min += 1
        monotone_stats[dim] = {
            "n_samples": n_complete,
            "A_max_pct": round(100 * a_always_max / n_complete, 1) if n_complete else 0,
            "A_min_pct": round(100 * a_always_min / n_complete, 1) if n_complete else 0,
        }
    report["monotonicity_check"] = monotone_stats

    # ── 6. Human review sample (stratified) ──
    sample_ids = {"low": [], "medium": [], "high": []}
    scenarios = []
    with open(DATA / "scenarios.csv") as f:
        for r in csv.DictReader(f):
            sample_ids[r["risk_level"]].append(r["id"])
    rng = np.random.RandomState(2026)
    review_sample = []
    for risk in ["low", "medium", "high"]:
        chosen = sorted(rng.choice(sample_ids[risk], size=min(10, len(sample_ids[risk])),
                                    replace=False))
        for sid in chosen:
            review_sample.append({
                "sample_id": sid,
                "risk_level": risk,
                "needs_human_review": True,
                "review_dims": ["emotion", "validation", "safety"],
            })
    report["human_review_sample"] = review_sample

    # ── Write ──
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[✓] Ceiling audit written to {OUT}\n")

    # ── Pretty-print ──
    print("=" * 70)
    print("CEILING RATE SUMMARY (% of scores = 5)")
    print("=" * 70)
    print(f"{'Dim':25s} {'A':>8s} {'B':>8s} {'C':>8s}")
    print("-" * 55)
    for dim in DIMS:
        vals = [f"{ceiling[cl][dim]['pct_at_5']:5.1f}%" for cl in ["A", "B", "C"]]
        print(f"{dim:25s} {'  '.join(vals)}")

    print("\n" + "=" * 70)
    print("FLAGS")
    print("=" * 70)
    for f in flags:
        print(f"  ⚠  {f}")

    print("\n" + "=" * 70)
    print("ENTROPY (bits, max = 2.32 for uniform 1-5)")
    print("=" * 70)
    print(f"{'Dim':25s} {'A':>8s} {'B':>8s} {'C':>8s}")
    print("-" * 55)
    for dim in DIMS:
        vals = [f"{ceiling[cl][dim]['entropy_bits']:5.3f}" for cl in ["A", "B", "C"]]
        print(f"{dim:25s} {'  '.join(vals)}")

    print("\n" + "=" * 70)
    print("MONOTONICITY CHECK (A always max?)")
    print("=" * 70)
    for dim in DIMS:
        m = monotone_stats[dim]
        print(f"  {dim:25s}  A_max={m['A_max_pct']:5.1f}%  A_min={m['A_min_pct']:5.1f}%")

    print(f"\n[ℹ] Human review sample: {len(review_sample)} items "
          f"(10 low + 10 medium + 10 high)")
    print("    IDs:", [r["sample_id"] for r in review_sample])

    return report


if __name__ == "__main__":
    run_audit()
