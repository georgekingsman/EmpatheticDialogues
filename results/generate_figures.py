#!/usr/bin/env python3
"""Generate 4 core paper figures from offline evaluation v1 results.

Figures saved to results/offline_eval_v1/figures/:
  fig1_overall_6dim.pdf      — Grouped bar chart of 6 dimensions × 3 conditions
  fig2_highrisk_focus.pdf     — High-risk subset: Safety/Boundary/Escalation/Helpfulness
  fig3_checker_decisions.pdf  — Checker decision distribution by risk level
  fig4_tradeoff.pdf           — Empathy composite vs Safety composite scatter

Usage:
    python results/generate_figures.py
"""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "results" / "offline_eval_v1"
FIG_DIR = DATA / "figures"
FIG_DIR.mkdir(exist_ok=True)

DIMS = ["emotion", "validation", "helpfulness", "safety",
        "boundary_adherence", "escalation"]
DIM_LABELS = {
    "emotion": "Emotion",
    "validation": "Validation",
    "helpfulness": "Helpfulness",
    "safety": "Safety",
    "boundary_adherence": "Boundary",
    "escalation": "Escalation",
}
COND_ORDER = ["single_agent", "double_hidden", "double_visible"]
COND_LABELS = {"single_agent": "A: Single Agent",
               "double_hidden": "B: Hidden Checker",
               "double_visible": "C: Visible Checker"}
COLORS = {"single_agent": "#4C72B0",
          "double_hidden": "#DD8452",
          "double_visible": "#55A868"}

# ── Load data ─────────────────────────────────────────────────────

def load_stats():
    with open(DATA / "statistics.json") as f:
        return json.load(f)

def load_judge_rows():
    rows = []
    with open(DATA / "judge_scores.csv") as f:
        for r in csv.DictReader(f):
            for d in DIMS:
                r[d] = float(r[d])
            rows.append(r)
    return rows

def load_checker():
    with open(DATA / "checker_actions.csv") as f:
        return list(csv.DictReader(f))


# ── Style ─────────────────────────────────────────────────────────

def set_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# ── Figure 1: Overall 6-dimension grouped bar chart ──────────────

def fig1_overall(stats):
    desc = stats["condition_descriptives"]
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(DIMS))
    width = 0.25
    offsets = [-width, 0, width]

    for i, cond in enumerate(["A", "B", "C"]):
        cond_full = COND_ORDER[i]
        means = [desc[cond][d]["mean"] for d in DIMS]
        cis_lo = [desc[cond][d]["ci_95_lower"] for d in DIMS]
        cis_hi = [desc[cond][d]["ci_95_upper"] for d in DIMS]
        errs = [[m - lo for m, lo in zip(means, cis_lo)],
                [hi - m for m, hi in zip(means, cis_hi)]]
        ax.bar(x + offsets[i], means, width, label=COND_LABELS[cond_full],
               color=COLORS[cond_full], yerr=errs, capsize=3, edgecolor="white",
               linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([DIM_LABELS[d] for d in DIMS])
    ax.set_ylabel("Mean Score (1–5)")
    ax.set_ylim(2.5, 5.3)
    ax.set_title("Overall Performance Across 6 Evaluation Dimensions (N=90)")
    ax.legend(loc="lower left", frameon=True, fancybox=False)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
    ax.axhline(y=5.0, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)

    # Significance markers
    sig_info = stats["pairwise_overall"]
    for j, dim in enumerate(DIMS):
        markers = []
        for pair, label in [("A_vs_B", "A–B"), ("A_vs_C", "A–C")]:
            if sig_info[pair][dim]["sig_holm_05"]:
                markers.append(label)
        if markers:
            ax.text(j, 5.15, "*", ha="center", va="bottom", fontsize=12,
                    fontweight="bold", color="#333")

    fig.tight_layout()
    path = FIG_DIR / "fig1_overall_6dim.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"[✓] {path.name}")


# ── Figure 2: High-risk focused ──────────────────────────────────

def fig2_highrisk(stats):
    risk_desc = stats["condition_descriptives_by_risk"]["high"]
    focus_dims = ["safety", "boundary_adherence", "escalation", "helpfulness"]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(focus_dims))
    width = 0.25
    offsets = [-width, 0, width]

    for i, cond in enumerate(["A", "B", "C"]):
        cond_full = COND_ORDER[i]
        means = [risk_desc[cond][d]["mean"] for d in focus_dims]
        cis_lo = [risk_desc[cond][d]["ci_95_lower"] for d in focus_dims]
        cis_hi = [risk_desc[cond][d]["ci_95_upper"] for d in focus_dims]
        errs = [[m - lo for m, lo in zip(means, cis_lo)],
                [hi - m for m, hi in zip(means, cis_hi)]]
        ax.bar(x + offsets[i], means, width, label=COND_LABELS[cond_full],
               color=COLORS[cond_full], yerr=errs, capsize=3, edgecolor="white",
               linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([DIM_LABELS[d] for d in focus_dims])
    ax.set_ylabel("Mean Score (1–5)")
    ax.set_ylim(2.5, 5.3)
    ax.set_title("High-Risk Scenarios: Safety-Critical Dimensions (N=30)")
    ax.legend(loc="lower left", frameon=True, fancybox=False)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
    ax.axhline(y=5.0, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)

    fig.tight_layout()
    path = FIG_DIR / "fig2_highrisk_focus.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"[✓] {path.name}")


# ── Figure 3: Checker decision distribution ──────────────────────

def fig3_checker(stats_json):
    # Load from offline_evaluation.json for checker_stats
    with open(ROOT / "outputs" / "analysis" / "offline_evaluation.json") as f:
        eval_data = json.load(f)
    cs = eval_data["checker_stats"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    decisions = ["approve", "revise", "escalate"]
    dec_colors = {"approve": "#55A868", "revise": "#CCBB44", "escalate": "#CC6677"}
    risks = ["low", "medium", "high"]

    for ax_idx, (cond, cond_label) in enumerate([
        ("double_hidden", "B: Hidden Checker"),
        ("double_visible", "C: Visible Checker"),
    ]):
        by_risk = cs[cond]["by_risk"]
        x = np.arange(len(risks))
        bottom = np.zeros(len(risks))

        for dec in decisions:
            vals = []
            for risk in risks:
                vals.append(by_risk[risk].get(dec, 0))
            vals = np.array(vals, dtype=float)
            axes[ax_idx].bar(x, vals, 0.5, bottom=bottom, label=dec.capitalize(),
                              color=dec_colors[dec], edgecolor="white", linewidth=0.5)
            # Add count labels
            for j, v in enumerate(vals):
                if v > 0:
                    axes[ax_idx].text(j, bottom[j] + v / 2, str(int(v)),
                                       ha="center", va="center", fontsize=10,
                                       fontweight="bold", color="white")
            bottom += vals

        axes[ax_idx].set_xticks(x)
        axes[ax_idx].set_xticklabels([r.capitalize() for r in risks])
        axes[ax_idx].set_xlabel("Risk Level")
        axes[ax_idx].set_title(cond_label)

    axes[0].set_ylabel("Count")
    axes[1].legend(loc="upper right", frameon=True, fancybox=False)
    fig.suptitle("Checker Decision Distribution by Risk Level", fontsize=13, y=1.02)
    fig.tight_layout()
    path = FIG_DIR / "fig3_checker_decisions.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"[✓] {path.name}")


# ── Figure 4: Empathy–Safety trade-off ───────────────────────────

def fig4_tradeoff():
    rows = load_judge_rows()

    # Compute per-sample composites
    data = {}  # {(cond, sid): (empathy_composite, safety_composite)}
    for r in rows:
        empathy = (r["emotion"] + r["validation"]) / 2
        safety = (r["safety"] + r["boundary_adherence"] + r["escalation"]) / 3
        data[(r["condition"], r["sample_id"])] = (empathy, safety)

    fig, ax = plt.subplots(figsize=(7, 6))
    markers = {"single_agent": "o", "double_hidden": "s", "double_visible": "D"}

    for cond in COND_ORDER:
        pts = [(e, s) for (c, _), (e, s) in data.items() if c == cond]
        emp = [p[0] for p in pts]
        saf = [p[1] for p in pts]
        ax.scatter(emp, saf, c=COLORS[cond], marker=markers[cond],
                   label=COND_LABELS[cond], alpha=0.6, s=40, edgecolors="white",
                   linewidth=0.5)

    # Condition means
    for cond in COND_ORDER:
        pts = [(e, s) for (c, _), (e, s) in data.items() if c == cond]
        me = np.mean([p[0] for p in pts])
        ms = np.mean([p[1] for p in pts])
        ax.scatter(me, ms, c=COLORS[cond], marker=markers[cond],
                   s=200, edgecolors="black", linewidth=2, zorder=10)
        ax.annotate(COND_LABELS[cond].split(":")[0],
                    (me, ms), textcoords="offset points",
                    xytext=(10, -5), fontsize=9, fontweight="bold")

    ax.set_xlabel("Empathy Composite\n(Emotion + Validation) / 2")
    ax.set_ylabel("Safety Composite\n(Safety + Boundary + Escalation) / 3")
    ax.set_title("Empathy–Safety Trade-off Across Conditions")
    ax.set_xlim(3.0, 5.2)
    ax.set_ylim(3.0, 5.2)
    ax.legend(loc="lower left", frameon=True, fancybox=False)
    ax.axhline(y=5.0, color="gray", linestyle=":", alpha=0.3)
    ax.axvline(x=5.0, color="gray", linestyle=":", alpha=0.3)

    fig.tight_layout()
    path = FIG_DIR / "fig4_tradeoff.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"[✓] {path.name}")


# ── Main ──────────────────────────────────────────────────────────

def main():
    set_style()
    stats = load_stats()
    fig1_overall(stats)
    fig2_highrisk(stats)
    fig3_checker(stats)
    fig4_tradeoff()
    print(f"\n✅  All figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
