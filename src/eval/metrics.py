"""
Evaluation metrics for empathetic dialogue — both automatic and meta-evaluation.

Includes:
    - Traditional NLG metrics (BLEU, ROUGE) for reference comparison
    - Judge reliability / stability metrics
    - Active sampling utility (uncertainty-driven annotation selection)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from src.eval.rubric import DIMENSION_KEYS


# ===================================================================
# Traditional NLG metrics (reference-based)
# ===================================================================

def compute_nlg_metrics(
    predictions: list[str],
    references: list[str],
) -> dict[str, float]:
    """Compute BLEU and ROUGE scores.

    Requires ``evaluate`` library: ``pip install evaluate``.
    """
    try:
        import evaluate
    except ImportError:
        print("pip install evaluate rouge_score")
        return {}

    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")

    bleu_result = bleu.compute(
        predictions=predictions,
        references=[[r] for r in references],
    )

    rouge_result = rouge.compute(
        predictions=predictions,
        references=references,
    )

    return {
        "bleu": round(bleu_result["bleu"], 4),
        "rouge1": round(rouge_result["rouge1"], 4),
        "rouge2": round(rouge_result["rouge2"], 4),
        "rougeL": round(rouge_result["rougeL"], 4),
    }


# ===================================================================
# Judge reliability metrics
# ===================================================================

def compute_judge_stability(judge_results: list[dict]) -> dict:
    """Analyse judge self-consistency across repeated evaluations.

    Groups by (sample_id, model) and computes:
        - per-dimension standard deviation across repeats
        - per-dimension Krippendorff's alpha (if enough repeats)
        - overall agreement rate (same score ±1)

    Parameters
    ----------
    judge_results : list[dict]
        Must contain ``sample_id``, ``repeat_idx``, ``scores``.

    Returns
    -------
    dict with overall and per-dimension statistics.
    """
    # Group by (sample_id, model)
    groups: dict[str, list[dict]] = defaultdict(list)
    for res in judge_results:
        if "scores" not in res:
            continue
        key = f"{res['sample_id']}_{res.get('model', '')}"
        groups[key].append(res)

    per_dim_stds: dict[str, list[float]] = {dim: [] for dim in DIMENSION_KEYS}
    exact_agree_counts: dict[str, int] = {dim: 0 for dim in DIMENSION_KEYS}
    near_agree_counts: dict[str, int] = {dim: 0 for dim in DIMENSION_KEYS}
    total_pairs = 0

    for key, repeats in groups.items():
        if len(repeats) < 2:
            continue
        for dim in DIMENSION_KEYS:
            vals = [r["scores"][dim] for r in repeats if dim in r["scores"]]
            if len(vals) >= 2:
                per_dim_stds[dim].append(float(np.std(vals)))
                # Pairwise agreement
                for i in range(len(vals)):
                    for j in range(i + 1, len(vals)):
                        if vals[i] == vals[j]:
                            exact_agree_counts[dim] += 1
                        if abs(vals[i] - vals[j]) <= 1:
                            near_agree_counts[dim] += 1
        # Count total pairs for first dimension (same for all)
        n = len([r for r in repeats if "scores" in r])
        total_pairs += n * (n - 1) // 2

    result = {"n_groups": len(groups), "total_pairs": total_pairs, "per_dimension": {}}

    for dim in DIMENSION_KEYS:
        stds = per_dim_stds[dim]
        result["per_dimension"][dim] = {
            "mean_std": round(float(np.mean(stds)), 4) if stds else None,
            "median_std": round(float(np.median(stds)), 4) if stds else None,
            "exact_agreement_rate": (
                round(exact_agree_counts[dim] / total_pairs, 4) if total_pairs > 0 else None
            ),
            "near_agreement_rate": (
                round(near_agree_counts[dim] / total_pairs, 4) if total_pairs > 0 else None
            ),
        }

    return result


def compute_judge_human_correlation(
    human_labels: list[dict],
    judge_results: list[dict],
) -> dict[str, dict[str, float]]:
    """Compute per-dimension Spearman and Kendall correlation between
    human labels and judge scores.

    Uses the mean score when multiple annotators/repeats exist.
    """
    from scipy import stats as scipy_stats

    # Aggregate human by sample_id
    human_by_id: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for lab in human_labels:
        for dim in DIMENSION_KEYS:
            if dim in lab:
                try:
                    human_by_id[lab["sample_id"]][dim].append(int(lab[dim]))
                except (ValueError, TypeError):
                    pass

    # Aggregate judge by sample_id
    judge_by_id: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for res in judge_results:
        if "scores" not in res:
            continue
        for dim in DIMENSION_KEYS:
            if dim in res["scores"]:
                judge_by_id[res["sample_id"]][dim].append(float(res["scores"][dim]))

    results = {}
    common_ids = set(human_by_id.keys()) & set(judge_by_id.keys())

    for dim in DIMENSION_KEYS:
        h_vals, j_vals = [], []
        for sid in common_ids:
            if dim in human_by_id[sid] and dim in judge_by_id[sid]:
                h_vals.append(np.mean(human_by_id[sid][dim]))
                j_vals.append(np.mean(judge_by_id[sid][dim]))

        if len(h_vals) < 5:
            results[dim] = {"n": len(h_vals), "note": "insufficient data"}
            continue

        spearman, sp_p = scipy_stats.spearmanr(h_vals, j_vals)
        kendall, kt_p = scipy_stats.kendalltau(h_vals, j_vals)

        results[dim] = {
            "n": len(h_vals),
            "spearman": round(float(spearman), 4),
            "spearman_p": round(float(sp_p), 6),
            "kendall": round(float(kendall), 4),
            "kendall_p": round(float(kt_p), 6),
        }

    return results


# ===================================================================
# Error decomposition by category
# ===================================================================

def error_by_category(
    merged: dict[str, dict],
    generations: list[dict],
    category_fn=None,
) -> dict[str, dict]:
    """Decompose judge–human error by a category (e.g., response length, topic).

    Parameters
    ----------
    merged : output of calibrate.merge_human_and_judge
    generations : list of generation records (for metadata like response length)
    category_fn : callable(generation_record) -> str
        Assigns each sample to a category. Default: response length bucket.

    Returns
    -------
    dict[category → {dim → MAE}]
    """
    if category_fn is None:
        def category_fn(gen):
            length = len(gen.get("response", ""))
            if length < 100:
                return "short"
            elif length < 300:
                return "medium"
            else:
                return "long"

    gen_by_id = {g["id"]: g for g in generations}

    by_cat: dict[str, dict[str, list]] = defaultdict(lambda: {dim: [] for dim in DIMENSION_KEYS})

    for sid, data in merged.items():
        gen = gen_by_id.get(sid)
        if gen is None:
            continue
        cat = category_fn(gen)
        for dim in DIMENSION_KEYS:
            hv = data["human"].get(dim)
            jv = data["judge_mean"].get(dim)
            if hv is not None and jv is not None and not np.isnan(hv):
                by_cat[cat][dim].append(abs(jv - hv))

    result = {}
    for cat, dim_errors in by_cat.items():
        result[cat] = {
            dim: round(float(np.mean(errs)), 4) if errs else None
            for dim, errs in dim_errors.items()
        }
        result[cat]["n_samples"] = max(len(v) for v in dim_errors.values()) if dim_errors else 0

    return result


# ===================================================================
# Active sampling (uncertainty-driven annotation selection)
# ===================================================================

def select_for_annotation(
    judge_results: list[dict],
    n: int = 50,
    strategy: str = "uncertainty",
) -> list[str]:
    """Select sample IDs that would benefit most from human annotation.

    Strategies:
        "uncertainty" – highest judge self-disagreement (std across repeats)
        "low_confidence" – lowest judge self-reported confidence
        "random" – uniform random baseline

    Returns
    -------
    list of sample_id strings, ordered by priority.
    """
    # Aggregate
    by_id: dict[str, list[dict]] = defaultdict(list)
    for res in judge_results:
        if "scores" in res:
            by_id[res["sample_id"]].append(res)

    scored: list[tuple[str, float]] = []

    for sid, repeats in by_id.items():
        if strategy == "uncertainty":
            stds = []
            for dim in DIMENSION_KEYS:
                vals = [r["scores"][dim] for r in repeats]
                if len(vals) >= 2:
                    stds.append(float(np.std(vals)))
            score = float(np.mean(stds)) if stds else 0.0

        elif strategy == "low_confidence":
            confs = [r.get("confidence", 0.5) for r in repeats]
            score = -float(np.mean(confs))  # lower confidence → higher priority

        elif strategy == "random":
            score = float(np.random.random())

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        scored.append((sid, score))

    # Sort descending by priority score
    scored.sort(key=lambda x: x[1], reverse=True)
    return [sid for sid, _ in scored[:n]]
