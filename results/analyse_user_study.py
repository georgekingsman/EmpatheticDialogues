#!/usr/bin/env python3
"""
Analyse user study data for the Maker-Checker paper.

Expected input: results/user_study_data.csv with columns:
  participant_id, cell_id, vignette_id, condition, risk_level,
  Q1_empathy, Q2_warmth, Q3_safety, Q4_boundary, Q5_transparency,
  Q6_trust, Q7_rely, Q8_seekhelp,
  attention_check_passed, completion_time_s

Post-study file: results/user_study_post.csv with columns:
  participant_id, overall_satisfaction, crisis_comfort,
  mental_workload, referral_correct_count,
  age_group, gender, mh_service_use, chatbot_experience,
  open_feedback_safety, open_feedback_improve

Outputs:
  results/user_study_results/
    descriptives.json        - Means, SDs, CIs per condition × risk
    lmm_results.json         - LMM coefficients and p-values
    pairwise_contrasts.json  - Holm-corrected pairwise comparisons
    figures/                 - Publication-ready plots
    tables/                  - LaTeX tables
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Attempt to import statistical packages; guide user if missing.
try:
    import statsmodels.formula.api as smf
    from scipy import stats
    HAS_STATS = True
except ImportError:
    HAS_STATS = False
    warnings.warn("statsmodels or scipy not installed. Install with: "
                  "pip install statsmodels scipy")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "results"
OUT_DIR = ROOT / "results" / "user_study_results"

ITEMS = {
    "Q1_empathy": "Perceived Empathy",
    "Q2_warmth": "Perceived Warmth",
    "Q3_safety": "Perceived Safety",
    "Q4_boundary": "Boundary Clarity",
    "Q5_transparency": "Transparency",
    "Q6_trust": "Trust",
    "Q7_rely": "Willingness to Rely",
    "Q8_seekhelp": "Seek Real Help",
}

COMPOSITES = {
    "empathy_composite": ("Q1_empathy", "Q2_warmth"),
    "safety_composite": ("Q3_safety", "Q4_boundary"),
    "trust_composite": ("Q6_trust", "Q7_rely"),
}

CONDITION_LABELS = {"A": "Single Agent", "B": "Hidden Checker", "C": "Visible Checker"}
RISK_ORDER = ["low", "medium", "high"]
CONDITION_ORDER = ["A", "B", "C"]


def load_data():
    """Load and validate per-vignette data."""
    path = DATA_DIR / "user_study_data.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"User study data not found at {path}. "
            "Export from Qualtrics and place the CSV there."
        )
    df = pd.read_csv(path)

    required = ["participant_id", "cell_id", "vignette_id", "condition",
                 "risk_level"] + list(ITEMS.keys())
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Exclusions
    n_before = df["participant_id"].nunique()
    if "attention_check_passed" in df.columns:
        failed = df.groupby("participant_id")["attention_check_passed"].min()
        exclude_attn = set(failed[failed == 0].index)
        df = df[~df["participant_id"].isin(exclude_attn)]
    if "completion_time_s" in df.columns:
        fast = df.groupby("participant_id")["completion_time_s"].sum()
        exclude_fast = set(fast[fast < 300].index)  # < 5 min
        df = df[~df["participant_id"].isin(exclude_fast)]

    n_after = df["participant_id"].nunique()
    print(f"Participants: {n_before} loaded, {n_before - n_after} excluded, "
          f"{n_after} retained")

    # Compute composites
    for name, cols in COMPOSITES.items():
        df[name] = df[list(cols)].mean(axis=1)

    # Reliance Calibration Index
    df["rci"] = df["trust_composite"] * df["Q8_seekhelp"]

    # Categorical coding
    df["condition"] = pd.Categorical(df["condition"], categories=CONDITION_ORDER, ordered=True)
    df["risk_level"] = pd.Categorical(df["risk_level"], categories=RISK_ORDER, ordered=True)

    return df


def load_post_data():
    """Load post-study questionnaire."""
    path = DATA_DIR / "user_study_post.csv"
    if not path.exists():
        print(f"  [skip] Post-study data not found at {path}")
        return None
    return pd.read_csv(path)


def descriptives(df):
    """Compute means, SDs, 95% CIs by condition and condition × risk."""
    results = {}
    for level in ["overall", "by_risk"]:
        group_cols = ["condition"] if level == "overall" else ["condition", "risk_level"]
        measures = list(ITEMS.keys()) + list(COMPOSITES.keys()) + ["rci"]
        agg = df.groupby(group_cols)[measures].agg(["mean", "std", "count"])

        desc = {}
        for cond_key, grp in df.groupby(group_cols):
            key_str = str(cond_key)
            desc[key_str] = {}
            for m in measures:
                vals = grp[m].dropna()
                n = len(vals)
                mean = vals.mean()
                sd = vals.std()
                se = sd / np.sqrt(n) if n > 1 else 0
                ci_lo = mean - 1.96 * se
                ci_hi = mean + 1.96 * se
                desc[key_str][m] = {
                    "mean": round(mean, 3),
                    "sd": round(sd, 3),
                    "ci_lo": round(ci_lo, 3),
                    "ci_hi": round(ci_hi, 3),
                    "n": n,
                }
        results[level] = desc

    return results


def run_lmm(df):
    """Run Linear Mixed-Effects Models for each DV."""
    if not HAS_STATS:
        print("  [skip] statsmodels not available; skipping LMM")
        return {}

    measures = list(ITEMS.keys()) + list(COMPOSITES.keys()) + ["rci"]
    results = {}

    for m in measures:
        formula = f"{m} ~ C(condition, Treatment('A')) * C(risk_level, Treatment('low'))"
        try:
            model = smf.mixedlm(
                formula, df,
                groups=df["participant_id"],
                re_formula="1",
                vc_formula={"vignette_id": "0 + C(vignette_id)"},
            )
            fit = model.fit(reml=True)
            results[m] = {
                "converged": fit.converged,
                "aic": round(fit.aic, 1),
                "bic": round(fit.bic, 1),
                "fixed_effects": {
                    name: {
                        "coef": round(fit.fe_params[name], 4),
                        "se": round(fit.bse_fe[name], 4),
                        "z": round(fit.tvalues[name], 3),
                        "p": round(fit.pvalues[name], 5),
                    }
                    for name in fit.fe_params.index
                },
            }
            print(f"  {m}: converged={fit.converged}")
        except Exception as e:
            print(f"  {m}: FAILED — {e}")
            results[m] = {"error": str(e)}

    return results


def pairwise_contrasts(df):
    """Compute pairwise condition contrasts with Holm correction."""
    if not HAS_STATS:
        print("  [skip] scipy not available; skipping contrasts")
        return {}

    measures = list(ITEMS.keys()) + list(COMPOSITES.keys()) + ["rci"]
    pairs = [("A", "B"), ("A", "C"), ("B", "C")]
    results = {}

    for m in measures:
        contrasts = []
        for c1, c2 in pairs:
            vals1 = df[df["condition"] == c1][m].dropna()
            vals2 = df[df["condition"] == c2][m].dropna()
            t_stat, p_val = stats.ttest_ind(vals1, vals2)
            diff = vals1.mean() - vals2.mean()
            pooled_sd = np.sqrt((vals1.var() + vals2.var()) / 2)
            d = diff / pooled_sd if pooled_sd > 0 else 0
            contrasts.append({
                "pair": f"{c1}-{c2}",
                "diff": round(diff, 4),
                "t": round(t_stat, 3),
                "p_raw": round(p_val, 5),
                "d": round(d, 3),
            })

        # Holm correction
        p_vals = sorted([(c["p_raw"], i) for i, c in enumerate(contrasts)])
        k = len(p_vals)
        for rank, (p, idx) in enumerate(p_vals):
            adjusted = min(p * (k - rank), 1.0)
            contrasts[idx]["p_holm"] = round(adjusted, 5)

        results[m] = contrasts

    return results


def generate_figures(df):
    """Generate publication-ready figures."""
    if not HAS_PLOT:
        print("  [skip] matplotlib not available; skipping figures")
        return

    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "figure.dpi": 300,
    })

    # Figure 1: Composite bar chart (condition × composite)
    composites = list(COMPOSITES.keys())
    cond_means = {c: [] for c in CONDITION_ORDER}
    cond_errs = {c: [] for c in CONDITION_ORDER}
    for comp in composites:
        for cond in CONDITION_ORDER:
            vals = df[df["condition"] == cond][comp].dropna()
            cond_means[cond].append(vals.mean())
            cond_errs[cond].append(1.96 * vals.std() / np.sqrt(len(vals)))

    x = np.arange(len(composites))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4A90D9", "#E67E22", "#2ECC71"]
    for i, cond in enumerate(CONDITION_ORDER):
        ax.bar(x + i * width, cond_means[cond], width, yerr=cond_errs[cond],
               label=f"{cond}: {CONDITION_LABELS[cond]}", color=colors[i],
               capsize=3, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x + width)
    ax.set_xticklabels([c.replace("_", " ").title() for c in composites])
    ax.set_ylabel("Mean Rating (1–7)")
    ax.set_ylim(1, 7.5)
    ax.legend(frameon=False)
    ax.set_title("Composite Scores by Condition (User Study)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"user_composites.{ext}")
    plt.close(fig)
    print(f"  Saved user_composites.pdf/png")

    # Figure 2: RCI × Risk interaction (line plot)
    fig, ax = plt.subplots(figsize=(7, 5))
    for cond, color in zip(CONDITION_ORDER, colors):
        means = []
        errs = []
        for risk in RISK_ORDER:
            vals = df[(df["condition"] == cond) & (df["risk_level"] == risk)]["rci"].dropna()
            means.append(vals.mean())
            errs.append(1.96 * vals.std() / np.sqrt(len(vals)))
        ax.errorbar(RISK_ORDER, means, yerr=errs, marker="o", label=f"{cond}: {CONDITION_LABELS[cond]}",
                    color=color, capsize=4, linewidth=2, markersize=8)
    ax.set_xlabel("Risk Level")
    ax.set_ylabel("Reliance Calibration Index (Trust × Seek-Help)")
    ax.set_title("Reliance Calibration by Condition and Risk Level")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"rci_interaction.{ext}")
    plt.close(fig)
    print(f"  Saved rci_interaction.pdf/png")

    # Figure 3: Per-item heatmap
    fig, ax = plt.subplots(figsize=(10, 4))
    item_names = list(ITEMS.values())
    item_cols = list(ITEMS.keys())
    data_matrix = np.zeros((3, len(item_cols)))
    for i, cond in enumerate(CONDITION_ORDER):
        for j, col in enumerate(item_cols):
            data_matrix[i, j] = df[df["condition"] == cond][col].mean()
    im = ax.imshow(data_matrix, cmap="YlOrRd", aspect="auto", vmin=1, vmax=7)
    ax.set_yticks(range(3))
    ax.set_yticklabels([f"{c}: {CONDITION_LABELS[c]}" for c in CONDITION_ORDER])
    ax.set_xticks(range(len(item_names)))
    ax.set_xticklabels(item_names, rotation=45, ha="right")
    for i in range(3):
        for j in range(len(item_cols)):
            ax.text(j, i, f"{data_matrix[i, j]:.2f}", ha="center", va="center",
                    color="white" if data_matrix[i, j] > 4.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, label="Mean Rating (1–7)")
    ax.set_title("Per-Item Mean Ratings by Condition")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"item_heatmap.{ext}")
    plt.close(fig)
    print(f"  Saved item_heatmap.pdf/png")


def generate_tables(df):
    """Generate LaTeX tables."""
    table_dir = OUT_DIR / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    # Table: Overall means by condition
    measures = list(ITEMS.keys()) + list(COMPOSITES.keys())
    rows = []
    for m in measures:
        label = ITEMS.get(m, m.replace("_", " ").title())
        cells = []
        for cond in CONDITION_ORDER:
            vals = df[df["condition"] == cond][m].dropna()
            mean = vals.mean()
            sd = vals.std()
            cells.append(f"{mean:.2f} $\\pm${sd:.2f}")
        rows.append(f"  {label} & {' & '.join(cells)} \\\\")

    latex = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\caption{User study: Mean ratings ($\\pm$SD) by condition.}\n"
        "\\label{tab:userstudy}\n"
        "\\small\n"
        "\\begin{tabular}{lccc}\n"
        "\\toprule\n"
        "Measure & A: Single Agent & B: Hidden Checker & C: Visible Checker \\\\\n"
        "\\midrule\n"
        + "\n".join(rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    (table_dir / "user_study_overall.tex").write_text(latex)
    print(f"  Saved user_study_overall.tex")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Loading Data ===")
    df = load_data()
    post = load_post_data()

    print(f"\nData shape: {df.shape}")
    print(f"Participants: {df['participant_id'].nunique()}")
    print(f"Conditions: {df['condition'].value_counts().to_dict()}")
    print(f"Risk levels: {df['risk_level'].value_counts().to_dict()}")

    print("\n=== Descriptive Statistics ===")
    desc = descriptives(df)
    with open(OUT_DIR / "descriptives.json", "w") as f:
        json.dump(desc, f, indent=2)
    print(f"  Saved descriptives.json")

    # Print summary table
    print(f"\n  {'Measure':<25s} {'A':>8s} {'B':>8s} {'C':>8s}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}")
    all_measures = list(ITEMS.keys()) + list(COMPOSITES.keys())
    for m in all_measures:
        vals = []
        for cond in CONDITION_ORDER:
            v = desc["overall"].get(str(cond), {}).get(m, {})
            vals.append(f"{v.get('mean', 0):.2f}")
        label = ITEMS.get(m, m.replace("_", " ").title())
        print(f"  {label:<25s} {vals[0]:>8s} {vals[1]:>8s} {vals[2]:>8s}")

    print("\n=== Linear Mixed-Effects Models ===")
    lmm = run_lmm(df)
    with open(OUT_DIR / "lmm_results.json", "w") as f:
        json.dump(lmm, f, indent=2)
    print(f"  Saved lmm_results.json")

    print("\n=== Pairwise Contrasts (Holm-corrected) ===")
    pw = pairwise_contrasts(df)
    with open(OUT_DIR / "pairwise_contrasts.json", "w") as f:
        json.dump(pw, f, indent=2)
    print(f"  Saved pairwise_contrasts.json")

    # Print key contrasts
    for m in ["empathy_composite", "safety_composite", "Q5_transparency", "rci"]:
        if m in pw:
            label = ITEMS.get(m, m.replace("_", " ").title())
            print(f"\n  {label}:")
            for c in pw[m]:
                sig = "***" if c["p_holm"] < .001 else "**" if c["p_holm"] < .01 else "*" if c["p_holm"] < .05 else "ns"
                print(f"    {c['pair']}: d={c['d']:.2f}, p_holm={c['p_holm']:.4f} {sig}")

    print("\n=== Generating Figures ===")
    generate_figures(df)

    print("\n=== Generating Tables ===")
    generate_tables(df)

    # Post-study summary
    if post is not None:
        print("\n=== Post-Study Summary ===")
        for col in ["overall_satisfaction", "crisis_comfort", "mental_workload"]:
            if col in post.columns:
                print(f"  {col}: M={post[col].mean():.2f}, SD={post[col].std():.2f}")

    print(f"\n[✓] All outputs saved to {OUT_DIR.relative_to(ROOT)}/")
    print("\nTo interpret results relative to hypotheses:")
    print("  H1 (Safety: B,C > A):  Check Q3_safety & Q4_boundary A-B, A-C contrasts")
    print("  H2 (Transparency: C > A,B):  Check Q5_transparency A-C, B-C contrasts")
    print("  H3 (Warmth: A > B,C):  Check Q1_empathy & Q2_warmth A-B, A-C contrasts")
    print("  H4 (Calibrated reliance):  Check rci Condition × Risk interaction in LMM")


if __name__ == "__main__":
    main()
