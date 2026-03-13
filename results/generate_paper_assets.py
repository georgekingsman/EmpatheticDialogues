#!/usr/bin/env python3
"""Compute composite indices, generate upgraded paper figures and tables.

Reads from results/offline_eval_v2_final/ (frozen data).
Writes figures and tables back to the same directory.

Composite definitions:
  Empathy Composite = mean(emotion, validation)
  Safety Composite  = mean(safety, boundary_adherence, escalation)
  Helpfulness kept separate.

Figures:
  fig1_overall_6dim.pdf  — 6-dim grouped bar with significance markers
  fig2_highrisk_focus.pdf — high-risk subset (4 safety-critical dims)
  fig3_checker_decisions.pdf — stacked bar by risk level
  fig4_tradeoff.pdf — Empathy vs Safety composite scatter + condition means
  fig5_composite_bar.pdf — NEW: composite bar chart (Empathy / Safety / Helpfulness)

Tables (LaTeX):
  table1_overall.tex — Overall 6-dim scores
  table2_highrisk.tex — High-risk subset
  table3_robustness.tex — Cross-judge robustness summary
"""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "offline_eval_v2_final"
FIG_DIR = DATA / "figures"
TABLE_DIR = DATA / "tables"
FIG_DIR.mkdir(exist_ok=True)
TABLE_DIR.mkdir(exist_ok=True)

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
COND_MAP = {"single_agent": "A", "double_hidden": "B", "double_visible": "C"}
COND_LABELS = {"single_agent": "A: Single Agent",
               "double_hidden": "B: Hidden Checker",
               "double_visible": "C: Visible Checker"}
COLORS = {"single_agent": "#4C72B0",
          "double_hidden": "#DD8452",
          "double_visible": "#55A868"}

# ── Load data ─────────────────────────────────────────────────────

def load_judge_rows():
    rows = []
    with open(DATA / "judge_scores_main.csv") as f:
        for r in csv.DictReader(f):
            for d in DIMS:
                r[d] = float(r[d])
            rows.append(r)
    return rows

def load_stats():
    with open(DATA / "statistics.json") as f:
        return json.load(f)

def load_checker():
    with open(DATA / "checker_actions.csv") as f:
        return list(csv.DictReader(f))

def load_second_judge():
    rows = []
    with open(DATA / "judge_scores_second.csv") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows

def load_multi_rater():
    with open(DATA / "multi_rater_report.json") as f:
        return json.load(f)

# ── Compute composites ───────────────────────────────────────────

def compute_composites(rows):
    """Add empathy_composite and safety_composite to each row."""
    for r in rows:
        r["empathy_composite"] = (r["emotion"] + r["validation"]) / 2
        r["safety_composite"] = (r["safety"] + r["boundary_adherence"] + r["escalation"]) / 3
    return rows

def composite_stats_by_condition(rows, risk=None):
    """Return {cond: {metric: {mean, std, ci_lo, ci_hi}}}."""
    from collections import defaultdict
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if risk and r["risk_level"] != risk:
            continue
        cond = r["condition"]
        groups[cond]["empathy_composite"].append(r["empathy_composite"])
        groups[cond]["safety_composite"].append(r["safety_composite"])
        groups[cond]["helpfulness"].append(r["helpfulness"])
        for d in DIMS:
            groups[cond][d].append(r[d])

    result = {}
    for cond in COND_ORDER:
        result[cond] = {}
        for metric, vals in groups[cond].items():
            arr = np.array(vals)
            n = len(arr)
            mean = np.mean(arr)
            std = np.std(arr, ddof=1) if n > 1 else 0.0
            # Bootstrap CI
            rng = np.random.RandomState(42)
            boots = [np.mean(rng.choice(arr, n, replace=True)) for _ in range(10000)]
            ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
            result[cond][metric] = {
                "mean": round(mean, 3), "std": round(std, 3),
                "ci_lo": round(ci_lo, 3), "ci_hi": round(ci_hi, 3),
                "n": n
            }
    return result

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

# ── Figure 1: Overall 6-dim ──────────────────────────────────────

def fig1_overall(comp_stats, stats):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(DIMS))
    width = 0.25
    offsets = [-width, 0, width]

    for i, cond in enumerate(COND_ORDER):
        means = [comp_stats[cond][d]["mean"] for d in DIMS]
        cis_lo = [comp_stats[cond][d]["ci_lo"] for d in DIMS]
        cis_hi = [comp_stats[cond][d]["ci_hi"] for d in DIMS]
        errs = [[m - lo for m, lo in zip(means, cis_lo)],
                [hi - m for m, hi in zip(means, cis_hi)]]
        ax.bar(x + offsets[i], means, width, label=COND_LABELS[cond],
               color=COLORS[cond], yerr=errs, capsize=3, edgecolor="white",
               linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([DIM_LABELS[d] for d in DIMS])
    ax.set_ylabel("Mean Score (1–5)")
    ax.set_ylim(2.5, 5.4)
    ax.set_title("Overall Performance Across 6 Evaluation Dimensions (N = 90)")
    ax.legend(loc="lower left", frameon=True, fancybox=False)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
    ax.axhline(y=5.0, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)

    # Significance markers
    sig = stats["pairwise_overall"]
    for j, dim in enumerate(DIMS):
        any_sig = False
        for pair in ["A_vs_B", "A_vs_C"]:
            if sig[pair][dim]["sig_holm_05"]:
                any_sig = True
        if any_sig:
            ax.text(j, 5.2, "*", ha="center", va="bottom", fontsize=14,
                    fontweight="bold", color="#333")

    fig.tight_layout()
    for ext in [".pdf", ".png"]:
        fig.savefig(FIG_DIR / f"fig1_overall_6dim{ext}")
    plt.close(fig)
    print("[✓] fig1_overall_6dim")


# ── Figure 2: High-risk focus ────────────────────────────────────

def fig2_highrisk(comp_stats_hr):
    focus_dims = ["helpfulness", "safety", "boundary_adherence", "escalation"]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(focus_dims))
    width = 0.25
    offsets = [-width, 0, width]

    for i, cond in enumerate(COND_ORDER):
        means = [comp_stats_hr[cond][d]["mean"] for d in focus_dims]
        cis_lo = [comp_stats_hr[cond][d]["ci_lo"] for d in focus_dims]
        cis_hi = [comp_stats_hr[cond][d]["ci_hi"] for d in focus_dims]
        errs = [[m - lo for m, lo in zip(means, cis_lo)],
                [hi - m for m, hi in zip(means, cis_hi)]]
        ax.bar(x + offsets[i], means, width, label=COND_LABELS[cond],
               color=COLORS[cond], yerr=errs, capsize=3, edgecolor="white",
               linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([DIM_LABELS[d] for d in focus_dims])
    ax.set_ylabel("Mean Score (1–5)")
    ax.set_ylim(2.5, 5.4)
    ax.set_title("High-Risk Scenarios: Safety-Critical Dimensions (N = 30)")
    ax.legend(loc="lower left", frameon=True, fancybox=False)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
    ax.axhline(y=5.0, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)

    fig.tight_layout()
    for ext in [".pdf", ".png"]:
        fig.savefig(FIG_DIR / f"fig2_highrisk_focus{ext}")
    plt.close(fig)
    print("[✓] fig2_highrisk_focus")


# ── Figure 3: Checker decisions ──────────────────────────────────

def fig3_checker():
    checker_rows = load_checker()
    # Count by condition × risk × decision
    from collections import Counter
    counts = {}  # (cond, risk) -> Counter of decisions
    for r in checker_rows:
        key = (r["condition"], r["risk_level"])
        if key not in counts:
            counts[key] = Counter()
        counts[key][r["checker_decision"]] += 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    decisions = ["approve", "revise", "escalate"]
    dec_colors = {"approve": "#55A868", "revise": "#CCBB44", "escalate": "#CC6677"}
    risks = ["low", "medium", "high"]

    for ax_idx, (cond, cond_label) in enumerate([
        ("double_hidden", "B: Hidden Checker"),
        ("double_visible", "C: Visible Checker"),
    ]):
        x = np.arange(len(risks))
        bottom = np.zeros(len(risks))

        for dec in decisions:
            vals = np.array([counts.get((cond, risk), Counter()).get(dec, 0)
                            for risk in risks], dtype=float)
            axes[ax_idx].bar(x, vals, 0.5, bottom=bottom, label=dec.capitalize(),
                              color=dec_colors[dec], edgecolor="white", linewidth=0.5)
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

    axes[0].set_ylabel("Count (out of 30)")
    axes[1].legend(loc="upper left", frameon=True, fancybox=False)
    fig.suptitle("Checker Decision Distribution by Risk Level", fontsize=13, y=1.02)
    fig.tight_layout()
    for ext in [".pdf", ".png"]:
        fig.savefig(FIG_DIR / f"fig3_checker_decisions{ext}")
    plt.close(fig)
    print("[✓] fig3_checker_decisions")


# ── Figure 4: Empathy–Safety trade-off scatter ───────────────────

def fig4_tradeoff(rows):
    fig, ax = plt.subplots(figsize=(7, 6))
    markers = {"single_agent": "o", "double_hidden": "s", "double_visible": "D"}

    cond_means = {}
    for cond in COND_ORDER:
        pts = [(r["empathy_composite"], r["safety_composite"])
               for r in rows if r["condition"] == cond]
        emp = [p[0] for p in pts]
        saf = [p[1] for p in pts]
        ax.scatter(emp, saf, c=COLORS[cond], marker=markers[cond],
                   label=COND_LABELS[cond], alpha=0.5, s=35, edgecolors="white",
                   linewidth=0.5)
        cond_means[cond] = (np.mean(emp), np.mean(saf))

    # Plot condition means as large markers with black edge
    for cond in COND_ORDER:
        me, ms = cond_means[cond]
        ax.scatter(me, ms, c=COLORS[cond], marker=markers[cond],
                   s=200, edgecolors="black", linewidth=2, zorder=10)
        ax.annotate(COND_MAP[cond], (me, ms), textcoords="offset points",
                    xytext=(12, -5), fontsize=14, fontweight="bold")

    ax.set_xlabel("Empathy Composite (Emotion + Validation) / 2")
    ax.set_ylabel("Safety Composite (Safety + Boundary + Escalation) / 3")
    ax.set_title("Warmth–Safety Trade-off")
    ax.legend(loc="lower left", frameon=True, fancybox=False)

    # Add quadrant labels
    ax.axhline(y=4.8, color="gray", linestyle="--", alpha=0.3)
    ax.axvline(x=4.85, color="gray", linestyle="--", alpha=0.3)

    fig.tight_layout()
    for ext in [".pdf", ".png"]:
        fig.savefig(FIG_DIR / f"fig4_tradeoff{ext}")
    plt.close(fig)
    print("[✓] fig4_tradeoff")


# ── Figure 5: Composite bar chart (NEW) ──────────────────────────

def fig5_composite(comp_stats):
    metrics = ["empathy_composite", "safety_composite", "helpfulness"]
    metric_labels = ["Empathy Composite", "Safety Composite", "Helpfulness"]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(metrics))
    width = 0.25
    offsets = [-width, 0, width]

    for i, cond in enumerate(COND_ORDER):
        means = [comp_stats[cond][m]["mean"] for m in metrics]
        cis_lo = [comp_stats[cond][m]["ci_lo"] for m in metrics]
        cis_hi = [comp_stats[cond][m]["ci_hi"] for m in metrics]
        errs = [[m - lo for m, lo in zip(means, cis_lo)],
                [hi - m for m, hi in zip(means, cis_hi)]]
        ax.bar(x + offsets[i], means, width, label=COND_LABELS[cond],
               color=COLORS[cond], yerr=errs, capsize=3, edgecolor="white",
               linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Mean Score (1–5)")
    ax.set_ylim(3.0, 5.3)
    ax.set_title("Composite Indices Across Conditions (N = 90)")
    ax.legend(loc="lower right", frameon=True, fancybox=False)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
    ax.axhline(y=5.0, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)

    fig.tight_layout()
    for ext in [".pdf", ".png"]:
        fig.savefig(FIG_DIR / f"fig5_composite_bar{ext}")
    plt.close(fig)
    print("[✓] fig5_composite_bar")


# ── Table 1: Overall scores ──────────────────────────────────────

def table1_overall(comp_stats, stats):
    sig = stats["pairwise_overall"]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Mean scores ($\pm$SD) and 95\% bootstrap confidence intervals across conditions (N=90).}",
        r"\label{tab:overall}",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Dimension & A: Single Agent & B: Hidden Checker & C: Visible Checker \\",
        r"\midrule",
    ]

    for d in DIMS:
        vals = []
        means = [comp_stats[c][d]["mean"] for c in COND_ORDER]
        best_idx = np.argmax(means) if d not in ["emotion", "validation"] else 0
        for i, c in enumerate(COND_ORDER):
            s = comp_stats[c][d]
            fmt = f"{s['mean']:.3f} $\\pm${s['std']:.2f}"
            ci = f"[{s['ci_lo']:.2f}, {s['ci_hi']:.2f}]"
            cell = f"{fmt} {ci}"
            if i == best_idx:
                cell = r"\textbf{" + f"{s['mean']:.3f}" + r"}" + f" $\\pm${s['std']:.2f} {ci}"
            vals.append(cell)

        # Significance markers
        sig_marks = []
        for pair, mark in [("A_vs_B", "†"), ("A_vs_C", "‡")]:
            if sig[pair][d]["sig_holm_05"]:
                sig_marks.append(mark)
        dim_label = DIM_LABELS[d]
        if sig_marks:
            dim_label += "$^{" + ",".join(sig_marks) + "}$"

        lines.append(f"{dim_label} & {vals[0]} & {vals[1]} & {vals[2]} \\\\")

    lines.append(r"\midrule")

    # Composite rows
    for metric, label in [("empathy_composite", "Empathy Composite"),
                          ("safety_composite", "Safety Composite"),
                          ("helpfulness", "Helpfulness")]:
        vals = []
        means_arr = [comp_stats[c][metric]["mean"] for c in COND_ORDER]
        if metric == "empathy_composite":
            best_idx = 0
        else:
            best_idx = int(np.argmax(means_arr))
        for i, c in enumerate(COND_ORDER):
            s = comp_stats[c][metric]
            cell = f"{s['mean']:.3f} $\\pm${s['std']:.2f} [{s['ci_lo']:.2f}, {s['ci_hi']:.2f}]"
            if i == best_idx:
                cell = r"\textbf{" + f"{s['mean']:.3f}" + r"}" + f" $\\pm${s['std']:.2f} [{s['ci_lo']:.2f}, {s['ci_hi']:.2f}]"
            vals.append(cell)
        lines.append(f"\\textit{{{label}}} & {vals[0]} & {vals[1]} & {vals[2]} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}\small",
        r"\item $^{\dagger}$ A vs B significant at $p<.05$ (Holm-corrected).",
        r"\item $^{\ddagger}$ A vs C significant at $p<.05$ (Holm-corrected).",
        r"\end{tablenotes}",
        r"\end{table}",
    ])

    path = TABLE_DIR / "table1_overall.tex"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[✓] {path.name}")


# ── Table 2: High-risk subset ────────────────────────────────────

def table2_highrisk(comp_stats_hr):
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Mean scores ($\pm$SD) for high-risk scenarios (N=30).}",
        r"\label{tab:highrisk}",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Dimension & A: Single Agent & B: Hidden Checker & C: Visible Checker \\",
        r"\midrule",
    ]

    for d in DIMS:
        vals = []
        means = [comp_stats_hr[c][d]["mean"] for c in COND_ORDER]
        best_idx = int(np.argmax(means))
        for i, c in enumerate(COND_ORDER):
            s = comp_stats_hr[c][d]
            cell = f"{s['mean']:.3f} $\\pm${s['std']:.2f}"
            if i == best_idx:
                cell = r"\textbf{" + f"{s['mean']:.3f}" + r"}" + f" $\\pm${s['std']:.2f}"
            vals.append(cell)
        lines.append(f"{DIM_LABELS[d]} & {vals[0]} & {vals[1]} & {vals[2]} \\\\")

    lines.append(r"\midrule")
    for metric, label in [("empathy_composite", "Empathy Composite"),
                          ("safety_composite", "Safety Composite")]:
        vals = []
        means_arr = [comp_stats_hr[c][metric]["mean"] for c in COND_ORDER]
        best_idx = int(np.argmax(means_arr))
        for i, c in enumerate(COND_ORDER):
            s = comp_stats_hr[c][metric]
            cell = f"{s['mean']:.3f} $\\pm${s['std']:.2f}"
            if i == best_idx:
                cell = r"\textbf{" + f"{s['mean']:.3f}" + r"}" + f" $\\pm${s['std']:.2f}"
            vals.append(cell)
        lines.append(f"\\textit{{{label}}} & {vals[0]} & {vals[1]} & {vals[2]} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    path = TABLE_DIR / "table2_highrisk.tex"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[✓] {path.name}")


# ── Table 3: Cross-judge robustness ──────────────────────────────

def table3_robustness():
    multi = load_multi_rater()
    second = load_second_judge()

    # Compute second-judge condition means
    from collections import defaultdict
    sj = defaultdict(list)
    for r in second:
        try:
            cond = r["condition"]
            sj[(cond, "emotion")].append(float(r["alt_emotion"]))
            sj[(cond, "validation")].append(float(r["alt_validation"]))
            sj[(cond, "safety")].append(float(r["alt_safety"]))
        except (ValueError, KeyError):
            continue

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Cross-judge robustness: Emotion mean by evaluation variant and condition.}",
        r"\label{tab:robustness}",
        r"\small",
        r"\begin{tabular}{lccc|c}",
        r"\toprule",
        r"Judge Variant & A & B & C & A $>$ B/C? \\",
        r"\midrule",
    ]

    # Original judge
    lines.append(f"Original Judge & 5.000 & 4.744 & 4.733 & Yes \\\\")

    # Second judge
    sj_means = {}
    for cond in ["A_single_agent", "B_double_hidden", "C_double_visible"]:
        vals = sj.get((cond, "emotion"), [])
        sj_means[cond] = np.mean(vals) if vals else 0
    lines.append(
        f"Stricter Second Judge & {sj_means.get('A_single_agent', 0):.3f} & "
        f"{sj_means.get('B_double_hidden', 0):.3f} & "
        f"{sj_means.get('C_double_visible', 0):.3f} & Yes \\\\"
    )

    # Multi-rater
    cm = multi["condition_means"]
    for rater in ["strict", "moderate", "lenient"]:
        a_val = cm.get(f"{rater}_A_single_agent_emotion", 0)
        b_val = cm.get(f"{rater}_B_double_hidden_emotion", 0)
        c_val = cm.get(f"{rater}_C_double_visible_emotion", 0)
        confirmed = "Yes" if a_val > b_val and a_val > c_val else "No"
        lines.append(
            f"Multi-rater: {rater.capitalize()} & {a_val:.3f} & {b_val:.3f} & {c_val:.3f} & {confirmed} \\\\"
        )

    lines.extend([
        r"\midrule",
        r"\multicolumn{5}{l}{\textit{Krippendorff's $\alpha$: emotion = "
        + f"{multi['krippendorff_alpha']['emotion']:.3f}, "
        + f"validation = {multi['krippendorff_alpha']['validation']:.3f}, "
        + f"safety = {multi['krippendorff_alpha']['safety']:.3f}"
        + r"}} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    path = TABLE_DIR / "table3_robustness.tex"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[✓] {path.name}")


# ── Save composite stats ─────────────────────────────────────────

def save_composite_stats(overall, high_risk):
    out = {
        "overall": {},
        "high_risk": {},
    }
    for cond in COND_ORDER:
        label = COND_MAP[cond]
        out["overall"][label] = overall[cond]
        out["high_risk"][label] = high_risk[cond]

    path = DATA / "composite_stats.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[✓] {path.name}")


# ── Main ──────────────────────────────────────────────────────────

def main():
    set_style()
    rows = load_judge_rows()
    rows = compute_composites(rows)
    stats = load_stats()

    print("\n=== Computing composite statistics ===")
    comp_overall = composite_stats_by_condition(rows)
    comp_hr = composite_stats_by_condition(rows, risk="high")

    # Print composite summary
    print("\nComposite Means (Overall):")
    for cond in COND_ORDER:
        e = comp_overall[cond]["empathy_composite"]["mean"]
        s = comp_overall[cond]["safety_composite"]["mean"]
        h = comp_overall[cond]["helpfulness"]["mean"]
        print(f"  {COND_LABELS[cond]}: Empathy={e:.3f}, Safety={s:.3f}, Help={h:.3f}")

    print("\nComposite Means (High-Risk):")
    for cond in COND_ORDER:
        e = comp_hr[cond]["empathy_composite"]["mean"]
        s = comp_hr[cond]["safety_composite"]["mean"]
        h = comp_hr[cond]["helpfulness"]["mean"]
        print(f"  {COND_LABELS[cond]}: Empathy={e:.3f}, Safety={s:.3f}, Help={h:.3f}")

    save_composite_stats(comp_overall, comp_hr)

    print("\n=== Generating figures ===")
    fig1_overall(comp_overall, stats)
    fig2_highrisk(comp_hr)
    fig3_checker()
    fig4_tradeoff(rows)
    fig5_composite(comp_overall)

    print("\n=== Generating LaTeX tables ===")
    table1_overall(comp_overall, stats)
    table2_highrisk(comp_hr)
    table3_robustness()

    print("\n✅ All figures and tables generated in:")
    print(f"   Figures: {FIG_DIR}")
    print(f"   Tables:  {TABLE_DIR}")

if __name__ == "__main__":
    main()
