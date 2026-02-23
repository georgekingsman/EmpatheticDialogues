"""
Schema and utilities for human annotation labels.

Defines the expected CSV / JSONL schema, validation logic, and
inter-annotator agreement computation (Cohen's κ, weighted κ,
Krippendorff's α, Spearman/Kendall correlation).
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.rubric import DIMENSION_KEYS, VALID_SCORES

# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------

HUMAN_LABEL_FIELDS = [
    "sample_id",        # matches generation output "id"
    "annotator_id",     # anonymised annotator identifier
    "emotion",          # 1–5
    "validation",       # 1–5
    "helpfulness",      # 1–5
    "safety",           # 1–5
    "overall",          # 1–5 (optional holistic score)
    "notes",            # free-text rationale (optional)
]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_label(label: dict) -> list[str]:
    """Return a list of validation errors (empty if valid)."""
    errors = []
    if not label.get("sample_id"):
        errors.append("missing sample_id")
    if not label.get("annotator_id"):
        errors.append("missing annotator_id")
    for key in DIMENSION_KEYS:
        val = label.get(key)
        if val is None:
            errors.append(f"missing {key}")
        else:
            try:
                val = int(val)
            except (TypeError, ValueError):
                errors.append(f"invalid type for {key}: {val}")
                continue
            if val not in VALID_SCORES:
                errors.append(f"{key}={val} out of range 1-5")
    return errors


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_human_labels_csv(path: str | Path) -> list[dict]:
    """Load a CSV of human annotations and validate."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            # Cast numeric fields
            for key in DIMENSION_KEYS + ["overall"]:
                if key in row and row[key]:
                    try:
                        row[key] = int(row[key])
                    except ValueError:
                        pass
            errs = validate_label(row)
            if errs:
                print(f"  Warning: row {i}: {errs}")
            records.append(row)
    return records


def save_human_labels_csv(records: list[dict], path: str | Path) -> None:
    """Save annotations to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HUMAN_LABEL_FIELDS)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec.get(k, "") for k in HUMAN_LABEL_FIELDS})
    print(f"Saved {len(records)} human labels → {path}")


def generate_blank_annotation_sheet(
    generation_path: str | Path,
    output_path: str | Path,
    annotator_id: str = "",
) -> None:
    """Create a blank CSV annotation sheet from generation JSONL."""
    records: list[dict] = []
    with open(generation_path, "r") as f:
        for line in f:
            gen = json.loads(line.strip())
            records.append(
                {
                    "sample_id": gen["id"],
                    "annotator_id": annotator_id,
                    "emotion": "",
                    "validation": "",
                    "helpfulness": "",
                    "safety": "",
                    "overall": "",
                    "notes": "",
                }
            )
    save_human_labels_csv(records, output_path)


# ---------------------------------------------------------------------------
# Inter-annotator agreement (extended: κ, weighted κ, α, correlation)
# ---------------------------------------------------------------------------

def _collect_paired_scores(
    labels: list[dict],
) -> tuple[dict[str, list[int]], dict[str, list[int]], int]:
    """Group labels by sample_id and collect paired scores for 2-annotator case.

    Returns (a1_scores, a2_scores, n_paired_samples).
    """
    by_sample: dict[str, list[dict]] = defaultdict(list)
    for lab in labels:
        by_sample[lab["sample_id"]].append(lab)

    a1: dict[str, list[int]] = {k: [] for k in DIMENSION_KEYS}
    a2: dict[str, list[int]] = {k: [] for k in DIMENSION_KEYS}
    n_paired = 0

    for sid, anns in by_sample.items():
        if len(anns) < 2:
            continue
        # Use first two annotators (sorted by annotator_id for reproducibility)
        anns_sorted = sorted(anns, key=lambda x: str(x.get("annotator_id", "")))
        n_paired += 1
        for key in DIMENSION_KEYS:
            try:
                a1[key].append(int(anns_sorted[0][key]))
                a2[key].append(int(anns_sorted[1][key]))
            except (KeyError, TypeError, ValueError):
                continue

    return a1, a2, n_paired


def compute_iaa(
    labels: list[dict],
) -> dict[str, float]:
    """Compute pairwise inter-annotator agreement (Cohen's kappa per dimension).

    Backward-compatible: returns dict[dimension_key → kappa].
    """
    try:
        from sklearn.metrics import cohen_kappa_score
    except ImportError:
        print("sklearn required for IAA computation. pip install scikit-learn")
        return {}

    a1, a2, _ = _collect_paired_scores(labels)

    kappas = {}
    for key in DIMENSION_KEYS:
        if len(a1[key]) >= 10:
            kappas[key] = round(cohen_kappa_score(a1[key], a2[key]), 4)
        else:
            kappas[key] = float("nan")

    return kappas


def compute_iaa_extended(
    labels: list[dict],
) -> dict[str, dict[str, Any]]:
    """Compute a full suite of inter-annotator agreement metrics.

    For each dimension returns:
        - cohens_kappa: unweighted Cohen's κ
        - weighted_kappa: linear-weighted Cohen's κ (better for ordinal Likert)
        - quadratic_kappa: quadratic-weighted Cohen's κ
        - krippendorff_alpha: ordinal Krippendorff's α (works for 2+ raters)
        - spearman: Spearman rank correlation
        - kendall: Kendall's τ
        - pearson: Pearson correlation
        - mae: mean absolute difference between raters
        - exact_agreement: proportion of exact matches
        - near_agreement: proportion of matches within ±1
        - n_samples: number of paired samples used
        - score_distribution: per-rater histogram {1:x, 2:y, ...}
    """
    try:
        from sklearn.metrics import cohen_kappa_score
    except ImportError:
        print("sklearn required. pip install scikit-learn")
        return {}

    from scipy import stats as scipy_stats

    a1, a2, n_paired = _collect_paired_scores(labels)

    results: dict[str, dict] = {}

    for key in DIMENSION_KEYS:
        s1, s2 = np.array(a1[key]), np.array(a2[key])
        n = len(s1)
        if n < 10:
            results[key] = {"n_samples": n, "note": "insufficient data (<10)"}
            continue

        dim_res: dict[str, Any] = {"n_samples": n}

        # Cohen's κ variants
        dim_res["cohens_kappa"] = round(float(cohen_kappa_score(s1, s2)), 4)
        dim_res["weighted_kappa_linear"] = round(
            float(cohen_kappa_score(s1, s2, weights="linear")), 4
        )
        dim_res["weighted_kappa_quadratic"] = round(
            float(cohen_kappa_score(s1, s2, weights="quadratic")), 4
        )

        # Krippendorff's alpha (ordinal)
        dim_res["krippendorff_alpha"] = round(
            _krippendorff_alpha_ordinal(s1, s2), 4
        )

        # Correlations
        sp, sp_p = scipy_stats.spearmanr(s1, s2)
        kt, kt_p = scipy_stats.kendalltau(s1, s2)
        pr, pr_p = scipy_stats.pearsonr(s1, s2)
        dim_res["spearman"] = round(float(sp), 4)
        dim_res["spearman_p"] = round(float(sp_p), 6)
        dim_res["kendall"] = round(float(kt), 4)
        dim_res["kendall_p"] = round(float(kt_p), 6)
        dim_res["pearson"] = round(float(pr), 4)
        dim_res["pearson_p"] = round(float(pr_p), 6)

        # Agreement rates
        dim_res["mae"] = round(float(np.mean(np.abs(s1 - s2))), 4)
        dim_res["exact_agreement"] = round(float(np.mean(s1 == s2)), 4)
        dim_res["near_agreement"] = round(float(np.mean(np.abs(s1 - s2) <= 1)), 4)

        # Score distributions
        dim_res["score_dist_r1"] = {
            int(v): int(c) for v, c in zip(*np.unique(s1, return_counts=True))
        }
        dim_res["score_dist_r2"] = {
            int(v): int(c) for v, c in zip(*np.unique(s2, return_counts=True))
        }

        results[key] = dim_res

    # Add overall summary
    results["_summary"] = {
        "n_paired_samples": n_paired,
        "mean_weighted_kappa": round(
            float(np.nanmean([
                results[k].get("weighted_kappa_linear", float("nan"))
                for k in DIMENSION_KEYS
            ])),
            4,
        ),
        "mean_krippendorff_alpha": round(
            float(np.nanmean([
                results[k].get("krippendorff_alpha", float("nan"))
                for k in DIMENSION_KEYS
            ])),
            4,
        ),
    }

    return results


def _krippendorff_alpha_ordinal(
    r1: np.ndarray, r2: np.ndarray
) -> float:
    """Compute Krippendorff's alpha for ordinal data (2 raters).

    Uses the ordinal distance metric: d²(v,v') = Σ_{k=min(v,v')}^{max(v,v')} n_k - (n_v + n_v')/2
    Simplified for 2 raters.
    """
    n = len(r1)
    if n < 2:
        return float("nan")

    # Stack into reliability matrix (2 raters × n items)
    all_vals = np.concatenate([r1, r2])
    vals, counts = np.unique(all_vals, return_counts=True)
    val_to_idx = {int(v): i for i, v in enumerate(vals)}

    # Observed disagreement
    d_obs = 0.0
    for i in range(n):
        d_obs += (float(r1[i]) - float(r2[i])) ** 2
    d_obs /= n

    # Expected disagreement (by chance)
    total = 2 * n
    d_exp = 0.0
    for i, vi in enumerate(vals):
        for j, vj in enumerate(vals):
            if i >= j:
                continue
            d_exp += counts[i] * counts[j] * (float(vi) - float(vj)) ** 2
    d_exp = d_exp * 2 / (total * (total - 1))

    if d_exp == 0:
        return 1.0 if d_obs == 0 else 0.0

    return float(1.0 - d_obs / d_exp)


def compute_self_consistency(
    labels: list[dict],
    duplicate_pairs: list[tuple[str, str]] | None = None,
) -> dict[str, float]:
    """Compute per-annotator self-consistency on embedded duplicates.

    Parameters
    ----------
    duplicate_pairs : list of (sample_id_original, sample_id_duplicate)
        If None, returns empty dict.

    Returns
    -------
    dict[annotator_id → mean_absolute_diff_across_dims]
    """
    if not duplicate_pairs:
        return {}

    pair_set = {(a, b) for a, b in duplicate_pairs}
    pair_set |= {(b, a) for a, b in duplicate_pairs}

    by_annotator: dict[str, dict[str, dict]] = defaultdict(dict)
    for lab in labels:
        by_annotator[lab["annotator_id"]][lab["sample_id"]] = lab

    result = {}
    for ann_id, samples in by_annotator.items():
        diffs = []
        for orig, dup in duplicate_pairs:
            if orig in samples and dup in samples:
                for dim in DIMENSION_KEYS:
                    try:
                        d = abs(int(samples[orig][dim]) - int(samples[dup][dim]))
                        diffs.append(d)
                    except (KeyError, TypeError, ValueError):
                        continue
        if diffs:
            result[ann_id] = round(float(np.mean(diffs)), 4)

    return result


def iaa_go_nogo(
    iaa_results: dict[str, dict],
    kappa_threshold: float = 0.4,
) -> dict:
    """Evaluate whether IAA is sufficient for full annotation.

    Returns decision dict with per-dimension verdicts and overall recommendation.
    """
    decisions = {}
    all_ok = True

    for dim in DIMENSION_KEYS:
        dim_res = iaa_results.get(dim, {})
        wk = dim_res.get("weighted_kappa_linear", float("nan"))

        if np.isnan(wk):
            verdict = "SKIP"
            action = "Insufficient data"
        elif wk >= kappa_threshold:
            verdict = "GO"
            action = "Proceed to full annotation"
        elif wk >= 0.25:
            verdict = "REVISE"
            action = "Add rubric examples / boundary rules, re-pilot 50 samples"
            all_ok = False
        else:
            verdict = "REWRITE"
            action = "Rewrite rubric/guide for this dimension before continuing"
            all_ok = False

        decisions[dim] = {
            "weighted_kappa": wk,
            "verdict": verdict,
            "action": action,
        }

    decisions["_overall"] = {
        "recommendation": "GO — proceed to full 600-sample annotation"
        if all_ok
        else "HOLD — address dimensions with REVISE/REWRITE before continuing",
        "all_dimensions_pass": all_ok,
    }

    return decisions
