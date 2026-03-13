#!/usr/bin/env python3
"""Statistical significance tests for offline evaluation v1.

Performs:
  1. Wilcoxon signed-rank tests (paired, non-parametric) for each dimension
     across A-vs-B, A-vs-C, B-vs-C
  2. Holm-corrected p-values for multiple comparisons
  3. Effect sizes: rank-biserial correlation r and Cliff's delta
  4. McNemar test for binary escalation correctness
  5. Risk-stratified sub-analyses (focus on high-risk)
  6. Bootstrap 95 % confidence intervals for all means

Usage:
    python results/run_statistics.py
"""

import csv
import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "results" / "offline_eval_v1"
OUT = ROOT / "results" / "offline_eval_v1" / "statistics.json"

DIMS = ["emotion", "validation", "helpfulness", "safety",
        "boundary_adherence", "escalation"]
CONDITIONS = ["single_agent", "double_hidden", "double_visible"]
COND_LABELS = {"single_agent": "A", "double_hidden": "B", "double_visible": "C"}

# ── Load data ─────────────────────────────────────────────────────

def load_judge_scores():
    """Return {(condition, sample_id): {dim: score}} dict."""
    data = {}
    with open(DATA / "judge_scores.csv") as f:
        for row in csv.DictReader(f):
            key = (row["condition"], row["sample_id"])
            data[key] = {
                "risk_level": row["risk_level"],
                **{d: float(row[d]) for d in DIMS},
            }
    return data


def paired_vectors(data, cond_a, cond_b, dim, risk=None):
    """Extract paired score vectors for two conditions on one dimension."""
    ids_a = {sid for (c, sid) in data if c == cond_a}
    ids_b = {sid for (c, sid) in data if c == cond_b}
    common = sorted(ids_a & ids_b)
    va, vb = [], []
    for sid in common:
        ra = data[(cond_a, sid)]
        rb = data[(cond_b, sid)]
        if risk and ra["risk_level"] != risk:
            continue
        va.append(ra[dim])
        vb.append(rb[dim])
    return np.array(va), np.array(vb)


# ── Statistical tests ────────────────────────────────────────────

def wilcoxon_test(x, y):
    """Wilcoxon signed-rank test; returns (statistic, p_value).
    Falls back gracefully if all differences are zero."""
    d = x - y
    if np.all(d == 0):
        return 0.0, 1.0
    try:
        stat, p = stats.wilcoxon(d, alternative="two-sided")
        return float(stat), float(p)
    except ValueError:
        return 0.0, 1.0


def rank_biserial(x, y):
    """Rank-biserial r as effect size for Wilcoxon (matched-pairs r)."""
    d = x - y
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    r_plus = np.sum(ranks[d > 0])
    r_minus = np.sum(ranks[d < 0])
    return float((r_plus - r_minus) / (r_plus + r_minus)) if (r_plus + r_minus) > 0 else 0.0


def cliffs_delta(x, y):
    """Cliff's delta (non-parametric effect size)."""
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return 0.0
    count = 0
    for xi in x:
        for yj in y:
            if xi > yj:
                count += 1
            elif xi < yj:
                count -= 1
    return float(count / (n1 * n2))


def holm_correction(p_values):
    """Holm-Bonferroni step-down correction for a list of p-values."""
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda t: t[1])
    corrected = [None] * n
    cum_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = p * (n - rank)
        adj = min(adj, 1.0)
        cum_max = max(cum_max, adj)
        corrected[orig_idx] = cum_max
    return corrected


def bootstrap_ci(x, n_boot=10000, ci=0.95, seed=42):
    """Bootstrap confidence interval for the mean."""
    rng = np.random.RandomState(seed)
    x = np.asarray(x, dtype=float)
    means = np.array([rng.choice(x, size=len(x), replace=True).mean()
                      for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(means, [100 * alpha, 100 * (1 - alpha)])
    return float(lo), float(hi)


def mcnemar_test(x_correct, y_correct):
    """McNemar test for paired binary data."""
    b = np.sum((x_correct == 1) & (y_correct == 0))
    c = np.sum((x_correct == 0) & (y_correct == 1))
    n = b + c
    if n == 0:
        return 0.0, 1.0
    if n < 25:
        # exact binomial
        p = float(stats.binomtest(b, n, 0.5).pvalue)
        return float(n), p
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p = float(1 - stats.chi2.cdf(chi2, 1))
    return float(chi2), p


# ── Main analysis ─────────────────────────────────────────────────

def run_analysis():
    data = load_judge_scores()
    results = {}

    # ── Build per-condition vectors ──
    pairs = [
        ("A_vs_B", "single_agent", "double_hidden"),
        ("A_vs_C", "single_agent", "double_visible"),
        ("B_vs_C", "double_hidden", "double_visible"),
    ]

    # ── 1. Overall pairwise tests ──
    all_p_values = []
    pairwise_raw = {}
    for label, c1, c2 in pairs:
        pairwise_raw[label] = {}
        for dim in DIMS:
            x, y = paired_vectors(data, c1, c2, dim)
            stat, p = wilcoxon_test(x, y)
            rb = rank_biserial(x, y)
            cd = cliffs_delta(x, y)
            pairwise_raw[label][dim] = {
                "n": int(len(x)),
                "mean_diff": float(np.mean(x - y)),
                "wilcoxon_stat": stat,
                "p_value_raw": p,
                "rank_biserial_r": rb,
                "cliffs_delta": cd,
            }
            all_p_values.append((label, dim, p))

    # Holm correction across all 18 tests (3 pairs × 6 dims)
    raw_ps = [t[2] for t in all_p_values]
    corrected = holm_correction(raw_ps)
    for i, (label, dim, _) in enumerate(all_p_values):
        pairwise_raw[label][dim]["p_value_holm"] = corrected[i]
        pairwise_raw[label][dim]["sig_holm_05"] = corrected[i] < 0.05
        pairwise_raw[label][dim]["sig_holm_01"] = corrected[i] < 0.01

    results["pairwise_overall"] = pairwise_raw

    # ── 2. Bootstrap CIs for condition means ──
    condition_stats = {}
    for cond in CONDITIONS:
        cond_data = {d: [] for d in DIMS}
        for (c, sid), rec in data.items():
            if c == cond:
                for d in DIMS:
                    cond_data[d].append(rec[d])
        condition_stats[COND_LABELS[cond]] = {}
        for d in DIMS:
            arr = np.array(cond_data[d])
            lo, hi = bootstrap_ci(arr)
            condition_stats[COND_LABELS[cond]][d] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "ci_95_lower": lo,
                "ci_95_upper": hi,
                "median": float(np.median(arr)),
            }
    results["condition_descriptives"] = condition_stats

    # ── 3. Risk-stratified tests (high-risk focus) ──
    risk_results = {}
    for risk in ["low", "medium", "high"]:
        risk_results[risk] = {}
        risk_p_values = []
        for label, c1, c2 in pairs:
            risk_results[risk][label] = {}
            for dim in DIMS:
                x, y = paired_vectors(data, c1, c2, dim, risk=risk)
                stat, p = wilcoxon_test(x, y)
                rb = rank_biserial(x, y)
                cd = cliffs_delta(x, y)
                risk_results[risk][label][dim] = {
                    "n": int(len(x)),
                    "mean_diff": float(np.mean(x - y)) if len(x) > 0 else 0,
                    "wilcoxon_stat": stat,
                    "p_value_raw": p,
                    "rank_biserial_r": rb,
                    "cliffs_delta": cd,
                }
                risk_p_values.append((label, dim, p))
        # Holm correct within each risk stratum
        raw_ps = [t[2] for t in risk_p_values]
        corrected = holm_correction(raw_ps)
        for i, (label, dim, _) in enumerate(risk_p_values):
            risk_results[risk][label][dim]["p_value_holm"] = corrected[i]
            risk_results[risk][label][dim]["sig_holm_05"] = corrected[i] < 0.05

    results["pairwise_by_risk"] = risk_results

    # ── 4. Risk-stratified descriptives ──
    risk_desc = {}
    for risk in ["low", "medium", "high"]:
        risk_desc[risk] = {}
        for cond in CONDITIONS:
            cond_data = {d: [] for d in DIMS}
            for (c, sid), rec in data.items():
                if c == cond and rec["risk_level"] == risk:
                    for d in DIMS:
                        cond_data[d].append(rec[d])
            risk_desc[risk][COND_LABELS[cond]] = {}
            for d in DIMS:
                arr = np.array(cond_data[d])
                lo, hi = bootstrap_ci(arr)
                risk_desc[risk][COND_LABELS[cond]][d] = {
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "ci_95_lower": lo,
                    "ci_95_upper": hi,
                }
    results["condition_descriptives_by_risk"] = risk_desc

    # ── 5. McNemar for escalation binary ──
    # Define "correct escalation" as: escalation score >= 4
    # (appropriate escalation behavior)
    mcnemar_results = {}
    for label, c1, c2 in pairs:
        x, y = paired_vectors(data, c1, c2, "escalation")
        x_bin = (x >= 4).astype(int)
        y_bin = (y >= 4).astype(int)
        stat, p = mcnemar_test(x_bin, y_bin)
        mcnemar_results[label] = {
            "n": int(len(x)),
            f"{COND_LABELS[c1]}_rate": float(x_bin.mean()),
            f"{COND_LABELS[c2]}_rate": float(y_bin.mean()),
            "statistic": stat,
            "p_value": p,
        }
    results["mcnemar_escalation"] = mcnemar_results

    # ── 6. McNemar for high-risk only ──
    mcnemar_highrisk = {}
    for label, c1, c2 in pairs:
        x, y = paired_vectors(data, c1, c2, "escalation", risk="high")
        x_bin = (x >= 4).astype(int)
        y_bin = (y >= 4).astype(int)
        stat, p = mcnemar_test(x_bin, y_bin)
        mcnemar_highrisk[label] = {
            "n": int(len(x)),
            f"{COND_LABELS[c1]}_rate": float(x_bin.mean()),
            f"{COND_LABELS[c2]}_rate": float(y_bin.mean()),
            "statistic": stat,
            "p_value": p,
        }
    results["mcnemar_escalation_highrisk"] = mcnemar_highrisk

    # ── Write output ──
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"[✓] Statistics written to {OUT}\n")

    # ── Pretty-print summary ──
    print("=" * 70)
    print("PAIRWISE COMPARISONS (overall, Holm-corrected)")
    print("=" * 70)
    for label in ["A_vs_B", "A_vs_C", "B_vs_C"]:
        print(f"\n  {label}:")
        for dim in DIMS:
            r = results["pairwise_overall"][label][dim]
            sig = "***" if r["sig_holm_01"] else ("*" if r["sig_holm_05"] else "ns")
            print(f"    {dim:25s}  Δ={r['mean_diff']:+.3f}  "
                  f"p(holm)={r['p_value_holm']:.4f}  r={r['rank_biserial_r']:+.3f}  "
                  f"δ={r['cliffs_delta']:+.3f}  [{sig}]")

    print("\n" + "=" * 70)
    print("HIGH-RISK PAIRWISE (Holm-corrected within stratum)")
    print("=" * 70)
    for label in ["A_vs_B", "A_vs_C", "B_vs_C"]:
        print(f"\n  {label}:")
        for dim in DIMS:
            r = results["pairwise_by_risk"]["high"][label][dim]
            sig = "*" if r.get("sig_holm_05") else "ns"
            print(f"    {dim:25s}  Δ={r['mean_diff']:+.3f}  "
                  f"p(holm)={r['p_value_holm']:.4f}  r={r['rank_biserial_r']:+.3f}  [{sig}]")

    print("\n" + "=" * 70)
    print("BOOTSTRAP 95% CIs (overall)")
    print("=" * 70)
    for cond_label in ["A", "B", "C"]:
        print(f"\n  Condition {cond_label}:")
        for dim in DIMS:
            s = results["condition_descriptives"][cond_label][dim]
            print(f"    {dim:25s}  M={s['mean']:.3f} ±{s['std']:.3f}  "
                  f"95%CI=[{s['ci_95_lower']:.3f}, {s['ci_95_upper']:.3f}]")

    print("\n" + "=" * 70)
    print("McNEMAR — Escalation (score ≥ 4)")
    print("=" * 70)
    for label, r in results["mcnemar_escalation"].items():
        keys = [k for k in r if k.endswith("_rate")]
        rates = " / ".join(f"{k}={r[k]:.3f}" for k in keys)
        print(f"  {label}: {rates}  χ²={r['statistic']:.2f}  p={r['p_value']:.4f}")

    print("\n  High-risk only:")
    for label, r in results["mcnemar_escalation_highrisk"].items():
        keys = [k for k in r if k.endswith("_rate")]
        rates = " / ".join(f"{k}={r[k]:.3f}" for k in keys)
        print(f"  {label}: {rates}  χ²={r['statistic']:.2f}  p={r['p_value']:.4f}")

    return results


if __name__ == "__main__":
    run_analysis()
