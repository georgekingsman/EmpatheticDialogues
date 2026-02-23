"""
NO-GO Recovery Workflow — run when pilot IAA fails (κ_w < threshold).

This script automates the recovery process:
1. Load pilot IAA report to identify failing dimensions
2. Extract the top-N hardest/most-disagreed cases per failing dimension
3. Generate a "disagreement case pack" for the alignment meeting
4. Create a rubric revision template highlighting gaps
5. Generate a 50-sample mini-pilot batch for re-testing after rubric revision

Usage:
    python experiments/nogo_recovery.py \
        --iaa_report outputs/pilot_iaa_TEST/pilot_iaa_report.json \
        --pilot_samples outputs/human_annotation/pilot/pilot_samples.csv \
        --r1 outputs/human_annotation/pilot/pilot_annotation_R1_SIMULATED.csv \
        --r2 outputs/human_annotation/pilot/pilot_annotation_R2_SIMULATED.csv \
        --mapping outputs/human_annotation/pilot/_pilot_mapping.json \
        --output_dir outputs/nogo_recovery
"""
import sys, os, csv, json, argparse, hashlib, random
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, ".")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from src.eval.rubric import DIMENSION_KEYS


# ─── Helpers ──────────────────────────────────────────────────────────────

def load_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def save_csv(rows, path, fieldnames):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


# ─── Step 1: Identify failing dimensions ─────────────────────────────────

def get_failing_dims(report: dict, threshold: float = 0.4) -> dict:
    """Return dict of dim → {kappa, verdict} for failing dimensions."""
    decisions = report.get("go_nogo_decisions", {})
    fails = {}
    for dim in DIMENSION_KEYS:
        dec = decisions.get(dim, {})
        v = dec.get("verdict", "")
        if v in ("REWRITE", "REVISE"):
            fails[dim] = {
                "weighted_kappa": dec.get("weighted_kappa"),
                "verdict": v,
            }
    return fails


# ─── Step 2: Extract highest-disagreement cases ──────────────────────────

def extract_disagreements(
    r1_labels: list[dict],
    r2_labels: list[dict],
    mapping: list[dict],
    pilot_samples: list[dict],
    failing_dims: dict,
    top_n: int = 15,
) -> dict:
    """For each failing dim, find samples with largest |R1 - R2| difference.

    Returns dict of dim → list of case dicts (sorted by disagreement).
    """
    eval_to_info = {m["eval_id"]: m for m in mapping}

    # Index annotations by eval_id
    r1_by_eid = {r["eval_id"]: r for r in r1_labels}
    r2_by_eid = {r["eval_id"]: r for r in r2_labels}

    # Index pilot_samples by eval_id
    samples_by_eid = {}
    for s in pilot_samples:
        eid = s.get("eval_id")
        if eid:
            samples_by_eid[eid] = s

    result = {}
    for dim in failing_dims:
        cases = []
        for eid in r1_by_eid:
            if eid not in r2_by_eid:
                continue
            info = eval_to_info.get(eid, {})
            # Skip duplicates
            if info.get("is_duplicate_of"):
                continue

            try:
                s1 = int(r1_by_eid[eid].get(dim, 0))
                s2 = int(r2_by_eid[eid].get(dim, 0))
            except (ValueError, TypeError):
                continue

            if s1 == 0 or s2 == 0:
                continue

            diff = abs(s1 - s2)
            sample = samples_by_eid.get(eid, {})
            cases.append({
                "eval_id": eid,
                "sample_id": info.get("sample_id", ""),
                "model_tag": info.get("model_tag", ""),
                "dimension": dim,
                "r1_score": s1,
                "r2_score": s2,
                "abs_diff": diff,
                "context": sample.get("context", sample.get("prompt", ""))[:200],
                "response": sample.get("response", sample.get("generated", ""))[:300],
            })

        cases.sort(key=lambda x: (-x["abs_diff"], x["eval_id"]))
        result[dim] = cases[:top_n]

    return result


# ─── Step 3: Generate disagreement case pack ─────────────────────────────

def generate_disagreement_pack(
    disagreements: dict,
    output_dir: str,
):
    """Write a CSV and Markdown for the alignment meeting."""
    all_cases = []
    for dim, cases in disagreements.items():
        all_cases.extend(cases)

    # CSV for structured review
    fields = ["eval_id", "sample_id", "model_tag", "dimension",
              "r1_score", "r2_score", "abs_diff", "context", "response",
              "consensus_score", "resolution_note"]  # last 2 for meeting
    csv_path = os.path.join(output_dir, "disagreement_cases.csv")
    save_csv(all_cases, csv_path, fields)

    # Markdown for the meeting
    md_lines = [
        "# 标注对齐会(Alignment Meeting) — 分歧案例讨论",
        "",
        "**Instructions**: For each case below, discuss why R1 and R2 disagree,",
        "then agree on the correct score and what rubric rule should have been applied.",
        "",
    ]
    for dim, cases in disagreements.items():
        md_lines.append(f"## Dimension: {dim} ({len(cases)} cases)")
        md_lines.append("")
        for i, c in enumerate(cases, 1):
            md_lines.extend([
                f"### Case {i}: {c['eval_id']} (|Δ| = {c['abs_diff']})",
                f"- **Model**: {c['model_tag']}",
                f"- **R1 score**: {c['r1_score']}",
                f"- **R2 score**: {c['r2_score']}",
                f"- **Context**: {c['context']}",
                f"- **Response**: {c['response']}",
                "",
                "**Consensus score**: ___",
                "",
                "**Which rubric rule applies?** ___",
                "",
                "**Rubric gap (if any)?** ___",
                "",
                "---",
                "",
            ])

    md_path = os.path.join(output_dir, "alignment_meeting_cases.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"  Disagreement CSV → {csv_path}")
    print(f"  Meeting Markdown → {md_path}")


# ─── Step 4: Rubric revision template ────────────────────────────────────

def generate_rubric_revision_template(
    failing_dims: dict,
    disagreements: dict,
    output_dir: str,
):
    """Create a rubric revision template highlighting gaps and patterns."""
    lines = [
        "# Rubric Revision Template (v2 → v3)",
        "",
        "Fill in after the alignment meeting. Focus on failing dimensions.",
        "",
    ]
    for dim, info in failing_dims.items():
        cases = disagreements.get(dim, [])
        # Analyze disagreement patterns
        diffs = [c["abs_diff"] for c in cases]
        score_pairs = [(c["r1_score"], c["r2_score"]) for c in cases]

        lines.extend([
            f"## {dim.upper()} (κ_w = {info['weighted_kappa']:.4f}, verdict: {info['verdict']})",
            "",
            f"**N cases with large disagreement**: {len(cases)}",
        ])

        if diffs:
            lines.append(f"**Mean |Δ|**: {np.mean(diffs):.2f}, Max |Δ|**: {max(diffs)}")

        # Find boundary confusion patterns
        confusion = defaultdict(int)
        for s1, s2 in score_pairs:
            key = f"{min(s1,s2)}↔{max(s1,s2)}"
            confusion[key] += 1
        top_confusions = sorted(confusion.items(), key=lambda x: -x[1])[:3]
        lines.append(f"**Common confusion boundaries**: {top_confusions}")

        lines.extend([
            "",
            "### Current rubric gaps:",
            "- [ ] _Fill: what rule is unclear/missing?_",
            "",
            "### Proposed additions:",
            "- [ ] _Fill: new boundary rule or example_",
            "",
            "### Proposed anchor examples:",
            "| Score | Example | Key distinguisher |",
            "|-------|---------|-------------------|",
            "| 1 | _fill_ | _fill_ |",
            "| 2 | _fill_ | _fill_ |",
            "| 3 | _fill_ | _fill_ |",
            "| 4 | _fill_ | _fill_ |",
            "| 5 | _fill_ | _fill_ |",
            "",
            "---",
            "",
        ])

    path = os.path.join(output_dir, "rubric_revision_template.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Rubric revision template → {path}")


# ─── Step 5: Generate mini-pilot 50 samples ──────────────────────────────

def generate_mini_pilot(
    pilot_samples: list[dict],
    disagreements: dict,
    mapping: list[dict],
    output_dir: str,
    n_samples: int = 50,
    n_trap: int = 5,
):
    """Generate 50-sample mini-pilot for re-testing after rubric revision.

    Composition:
    - 10 "trap" cases: highest-disagreement from pilot (already seen, known answer)
    - 40 fresh cases from the pilot pool, stratified by model
    """
    eval_to_info = {m["eval_id"]: m for m in mapping}
    samples_by_eid = {}
    for s in pilot_samples:
        eid = s.get("eval_id")
        if eid:
            samples_by_eid[eid] = s

    # Collect trap cases: top-N most disagreed overall
    all_disagree = []
    for dim, cases in disagreements.items():
        for c in cases:
            all_disagree.append(c)
    # Deduplicate by eval_id, keep max diff
    best_diff = {}
    for c in all_disagree:
        eid = c["eval_id"]
        if eid not in best_diff or c["abs_diff"] > best_diff[eid]["abs_diff"]:
            best_diff[eid] = c
    trap_eids = sorted(best_diff.keys(), key=lambda e: -best_diff[e]["abs_diff"])[:n_trap]

    # Collect fresh cases: the rest of the pilot, stratified
    pilot_eids = [s.get("eval_id") for s in pilot_samples if s.get("eval_id")]
    used_eids = set(trap_eids)
    # Also exclude duplicates
    dup_eids = set()
    for m in mapping:
        if m.get("is_duplicate_of"):
            dup_eids.add(m["eval_id"])
    available = [eid for eid in pilot_eids if eid not in used_eids and eid not in dup_eids]

    # Stratify by model
    by_model = defaultdict(list)
    for eid in available:
        info = eval_to_info.get(eid, {})
        by_model[info.get("model_tag", "unknown")].append(eid)

    n_fresh = n_samples - len(trap_eids)
    models = sorted(by_model.keys())
    per_model = n_fresh // len(models) if models else n_fresh
    fresh_eids = []
    for model in models:
        pool = by_model[model]
        random.shuffle(pool)
        fresh_eids.extend(pool[:per_model])
    # Fill remainder
    remaining = [eid for eid in available if eid not in set(fresh_eids)]
    random.shuffle(remaining)
    fresh_eids.extend(remaining[:n_fresh - len(fresh_eids)])

    # Combine and shuffle
    mini_eids = trap_eids + fresh_eids
    random.shuffle(mini_eids)

    # Assign new mini-pilot eval_ids
    mini_mapping = []
    mini_samples = []
    for i, eid in enumerate(mini_eids):
        new_eid = f"mini_{i:04d}"
        sample = samples_by_eid.get(eid, {})
        info = eval_to_info.get(eid, {})

        mini_samples.append({**sample, "eval_id": new_eid})
        mini_mapping.append({
            "mini_eval_id": new_eid,
            "original_eval_id": eid,
            "sample_id": info.get("sample_id", ""),
            "model_tag": info.get("model_tag", ""),
            "is_trap": eid in set(trap_eids),
        })

    # Save
    mini_dir = os.path.join(output_dir, "mini_pilot")
    os.makedirs(mini_dir, exist_ok=True)

    # Mini-pilot samples
    if mini_samples:
        fields = list(mini_samples[0].keys())
        save_csv(mini_samples, os.path.join(mini_dir, "mini_pilot_samples.csv"), fields)

    # Mini-pilot blank annotation sheets
    ann_fields = ["eval_id", "annotator_id", "emotion", "validation",
                  "helpfulness", "safety", "overall", "notes"]
    for ann_id in ["R1", "R2"]:
        rows = [{"eval_id": s["eval_id"], "annotator_id": ann_id} for s in mini_samples]
        save_csv(rows, os.path.join(mini_dir, f"mini_annotation_{ann_id}.csv"), ann_fields)

    # Mini-pilot mapping
    save_json(mini_mapping, os.path.join(mini_dir, "_mini_pilot_mapping.json"))

    trap_count = sum(1 for m in mini_mapping if m["is_trap"])
    print(f"  Mini-pilot: {len(mini_samples)} samples ({trap_count} traps) → {mini_dir}/")
    return mini_mapping


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NO-GO Recovery Workflow")
    parser.add_argument("--iaa_report", required=True,
                        help="Path to pilot_iaa_report.json")
    parser.add_argument("--pilot_samples",
                        default="outputs/human_annotation/pilot/pilot_samples.csv")
    parser.add_argument("--r1", required=True, help="Rater 1 filled CSV")
    parser.add_argument("--r2", required=True, help="Rater 2 filled CSV")
    parser.add_argument("--mapping",
                        default="outputs/human_annotation/pilot/_pilot_mapping.json")
    parser.add_argument("--output_dir", default="outputs/nogo_recovery")
    parser.add_argument("--mini_n", type=int, default=50,
                        help="Mini-pilot sample count")
    parser.add_argument("--trap_n", type=int, default=5,
                        help="Number of trap/reference cases in mini-pilot")
    parser.add_argument("--top_disagree", type=int, default=15,
                        help="Top-N disagreement cases per dimension")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Identify failures
    print("=" * 60)
    print("NO-GO Recovery Workflow")
    print("=" * 60)

    report = load_json(args.iaa_report)
    failing = get_failing_dims(report)

    if not failing:
        print("\n  All dimensions PASS — no recovery needed!")
        print("  → Proceed to full annotation with: python experiments/generate_full_annotation.py")
        return

    print(f"\n  Failing dimensions: {list(failing.keys())}")
    for dim, info in failing.items():
        print(f"    {dim}: κ_w={info['weighted_kappa']:.4f} ({info['verdict']})")

    # Step 2: Extract disagreements
    print(f"\n  Extracting top-{args.top_disagree} disagreements per dimension...")
    mapping = load_json(args.mapping)
    pilot_samples = load_csv(args.pilot_samples)
    r1_labels = load_csv(args.r1)
    r2_labels = load_csv(args.r2)

    disagreements = extract_disagreements(
        r1_labels, r2_labels, mapping, pilot_samples,
        failing, top_n=args.top_disagree,
    )
    for dim, cases in disagreements.items():
        if cases:
            avg_diff = np.mean([c["abs_diff"] for c in cases])
            print(f"    {dim}: {len(cases)} cases, avg |Δ|={avg_diff:.2f}")

    # Step 3: Generate meeting pack
    print("\n  Generating alignment meeting materials...")
    generate_disagreement_pack(disagreements, args.output_dir)

    # Step 4: Rubric revision template
    print("\n  Generating rubric revision template...")
    generate_rubric_revision_template(failing, disagreements, args.output_dir)

    # Step 5: Generate mini-pilot
    print(f"\n  Generating {args.mini_n}-sample mini-pilot...")
    generate_mini_pilot(
        pilot_samples, disagreements, mapping,
        args.output_dir,
        n_samples=args.mini_n,
        n_trap=args.trap_n,
    )

    # Summary
    print("\n" + "=" * 60)
    print("Recovery Package Ready")
    print("=" * 60)
    print(f"""
Next steps:
  1. 标注对齐会 (30 min):
     - Review: {args.output_dir}/alignment_meeting_cases.md
     - Discuss disagreements, agree on consensus scores
     - Fill in consensus_score and resolution_note columns

  2. Rubric revision:
     - Fill: {args.output_dir}/rubric_revision_template.md
     - Update: docs/rubric_v2.md → rubric_v3.md
     - Update: docs/annotation_guide_v2.md → include new examples

  3. Mini-pilot re-test:
     - Distribute: {args.output_dir}/mini_pilot/mini_annotation_R1.csv
     - Distribute: {args.output_dir}/mini_pilot/mini_annotation_R2.csv
     - After filled, run:
       python experiments/run_pilot_iaa.py \\
         --r1 {args.output_dir}/mini_pilot/mini_annotation_R1_FILLED.csv \\
         --r2 {args.output_dir}/mini_pilot/mini_annotation_R2_FILLED.csv \\
         --mapping {args.output_dir}/mini_pilot/_mini_pilot_mapping.json \\
         --output_dir {args.output_dir}/mini_pilot_iaa

  4. If mini-pilot passes → proceed to full 600
     If fails again → one more alignment round (max 2 attempts)
""")


if __name__ == "__main__":
    main()
