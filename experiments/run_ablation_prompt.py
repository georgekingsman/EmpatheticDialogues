"""
Ablation A: Judge prompt variant comparison.

Compares the default judge prompt against 2 alternative prompts:
  - variant_strict: More demanding rubric interpretation, penalises more
  - variant_minimal: Minimal instructions, fewer constraints

For each variant, re-evaluates a fixed subset of samples (50–100) and compares:
  - Self-consistency (across 3 repeats)
  - Agreement with human labels
  - Calibrated performance (using existing isotonic calibrator)

Outputs:
  - outputs/analysis/ablation_prompt.json
  - outputs/analysis/ablation_prompt.md

Usage:
    python experiments/run_ablation_prompt.py \
        --human outputs/labels/human/ratings_r1.csv,outputs/labels/human/ratings_r2.csv \
        --n_samples 50 \
        --dry_run   (to see prompts without calling API)
"""
import sys, os, json, argparse, time
sys.path.insert(0, ".")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from collections import defaultdict
from src.eval.rubric import DIMENSION_KEYS, rubric_to_text
from src.eval.llm_judge import judge_one, extract_json, validate_judge_output
from src.eval.human_labels_schema import load_human_labels_csv
from src.eval.llm_judge import load_judge_results
from src.eval.calibrate import merge_human_and_judge, IsotonicCalibrator
from scipy import stats as scipy_stats

# ===================================================================
# Prompt variants
# ===================================================================

PROMPT_VARIANTS = {
    "default": {
        "system": """\
You are an expert evaluator of empathetic and supportive dialogue responses.
You will score a therapist/assistant response to a user's emotional statement
using the rubric below.  Output **ONLY** a single JSON object — no markdown
fences, no extra text.

{rubric}

Output exactly this JSON schema (nothing else):
{{
  "scores": {{"emotion": <1-5>, "validation": <1-5>, "helpfulness": <1-5>, "safety": <1-5>}},
  "overall": <1-5>,
  "confidence": <0.0-1.0>,
  "notes": "<brief rationale, 1-2 sentences>"
}}
""",
        "user": """\
### User statement
{user_statement}

### Assistant response
{response}

Evaluate the response and output ONLY the JSON object.
""",
    },

    "strict": {
        "system": """\
You are an EXTREMELY strict evaluator of empathetic dialogue. You evaluate
therapeutic assistant responses. Adopt a CRITICAL perspective: only give scores
of 3+ if the response demonstrates CLEAR, SPECIFIC evidence of that quality.
Most generic responses should score 1-2.

{rubric}

IMPORTANT: Be strict. A score of 5 means near-perfect, publishable quality.
A score of 3 means genuinely adequate. Most LLM-generated responses should
score 1-2 unless they show real specificity and warmth.

Output exactly this JSON schema (nothing else):
{{
  "scores": {{"emotion": <1-5>, "validation": <1-5>, "helpfulness": <1-5>, "safety": <1-5>}},
  "overall": <1-5>,
  "confidence": <0.0-1.0>,
  "notes": "<rationale>"
}}
""",
        "user": """\
### User statement
{user_statement}

### Assistant response
{response}

Apply STRICT evaluation criteria. Output ONLY the JSON object.
""",
    },

    "minimal": {
        "system": """\
Rate this response on empathy quality. Score 1-5 for each dimension.

Dimensions: emotion (recognition), validation (warmth), helpfulness, safety.

Output JSON only:
{{
  "scores": {{"emotion": <1-5>, "validation": <1-5>, "helpfulness": <1-5>, "safety": <1-5>}},
  "overall": <1-5>,
  "confidence": <0.0-1.0>,
  "notes": "<brief note>"
}}
""",
        "user": """\
User: {user_statement}
Response: {response}

Score as JSON:
""",
    },
}


def build_variant_messages(variant_name: str, user_statement: str, response: str) -> list[dict]:
    """Build chat messages for a specific prompt variant."""
    v = PROMPT_VARIANTS[variant_name]
    rubric = rubric_to_text()
    system = v["system"].format(rubric=rubric)
    user = v["user"].format(user_statement=user_statement, response=response)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# ===================================================================
# Analysis
# ===================================================================

def analyse_variant_results(
    results: list[dict],
    human_labels: list[dict],
    all_judge_default: list[dict],
    gen_by_id: dict[str, dict],
) -> dict:
    """Analyse one variant's results: stability, human alignment."""

    # Stability
    by_sample = defaultdict(list)
    for r in results:
        if "scores" in r:
            by_sample[r["sample_id"]].append(r)

    stabilities = {dim: [] for dim in DIMENSION_KEYS}
    for sid, repeats in by_sample.items():
        for dim in DIMENSION_KEYS:
            vals = [r["scores"][dim] for r in repeats]
            if len(vals) >= 2:
                stabilities[dim].append(float(np.std(vals)))

    stability = {}
    for dim in DIMENSION_KEYS:
        s = stabilities[dim]
        stability[dim] = {
            "mean_std": round(float(np.mean(s)), 4) if s else None,
            "exact_agree_rate": round(
                sum(1 for x in s if x == 0) / len(s), 4
            ) if s else None,
        }

    # Score distribution
    score_means = {dim: [] for dim in DIMENSION_KEYS}
    for sid, repeats in by_sample.items():
        for dim in DIMENSION_KEYS:
            vals = [r["scores"][dim] for r in repeats]
            score_means[dim].append(float(np.mean(vals)))

    distribution = {
        dim: {
            "mean": round(float(np.mean(score_means[dim])), 3),
            "std": round(float(np.std(score_means[dim])), 3),
        }
        for dim in DIMENSION_KEYS
        if score_means[dim]
    }

    # Human alignment (if labels available)
    merged = merge_human_and_judge(human_labels, results)
    alignment = {}
    for dim in DIMENSION_KEYS:
        h_vals, j_vals = [], []
        for sid, data in merged.items():
            hv = data["human"].get(dim)
            jv = data["judge_mean"].get(dim)
            if hv is not None and jv is not None and not np.isnan(hv):
                h_vals.append(hv)
                j_vals.append(jv)
        if len(h_vals) >= 5:
            h, j = np.array(h_vals), np.array(j_vals)
            sp_res = scipy_stats.spearmanr(h, j)
            alignment[dim] = {
                "n": len(h),
                "mae": round(float(np.mean(np.abs(j - h))), 4),
                "spearman": round(float(sp_res[0]), 4),  # type: ignore[arg-type]
                "bias": round(float(np.mean(j - h)), 4),
            }

    return {
        "n_samples": len(by_sample),
        "n_scored": sum(1 for r in results if "scores" in r),
        "n_errors": sum(1 for r in results if "error" in r),
        "stability": stability,
        "score_distribution": distribution,
        "human_alignment": alignment,
    }


def generate_markdown(all_analyses: dict) -> str:
    lines = [
        "# Ablation A: Judge Prompt Variant Comparison",
        "",
        "## Stability (mean std across repeats)",
        "",
        "| Dim | " + " | ".join(all_analyses.keys()) + " |",
        "|-----|" + "|".join(["---:" for _ in all_analyses]) + "|",
    ]
    for dim in DIMENSION_KEYS:
        row = f"| {dim} |"
        for vname, va in all_analyses.items():
            s = va.get("stability", {}).get(dim, {})
            row += f" {s.get('mean_std', 'N/A')} |"
        lines.append(row)

    lines.extend([
        "",
        "## Score Distribution (mean)",
        "",
        "| Dim | " + " | ".join(all_analyses.keys()) + " |",
        "|-----|" + "|".join(["---:" for _ in all_analyses]) + "|",
    ])
    for dim in DIMENSION_KEYS:
        row = f"| {dim} |"
        for vname, va in all_analyses.items():
            d = va.get("score_distribution", {}).get(dim, {})
            row += f" {d.get('mean', 'N/A')} |"
        lines.append(row)

    lines.extend([
        "",
        "## Human Alignment (MAE / Spearman / Bias)",
        "",
        "| Dim | " + " | ".join(all_analyses.keys()) + " |",
        "|-----|" + "|".join(["---:" for _ in all_analyses]) + "|",
    ])
    for dim in DIMENSION_KEYS:
        row = f"| {dim} |"
        for vname, va in all_analyses.items():
            a = va.get("human_alignment", {}).get(dim, {})
            if a:
                row += f" MAE={a['mae']}, ρ={a['spearman']}, bias={a['bias']:+.2f} |"
            else:
                row += " N/A |"
        lines.append(row)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human", default="outputs/human_annotation/simulated_human_labels.csv")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--n_repeats", type=int, default=3)
    parser.add_argument("--dry_run", action="store_true",
                        help="Just show prompt differences, don't call API")
    parser.add_argument("--output_dir", default="outputs/analysis")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load data ---
    all_human = []
    for p in args.human.split(","):
        p = p.strip()
        if os.path.exists(p):
            all_human.extend(load_human_labels_csv(p))

    # Load existing default judge results
    all_judge_default = []
    for tag in ["gpt2_vanilla", "gpt2_finetuned", "empathy_chain"]:
        path = f"outputs/judge/{tag}_judge.jsonl"
        if os.path.exists(path):
            all_judge_default.extend(load_judge_results(path))

    # Load generations for context
    all_gens = []
    for tag in ["gpt2_vanilla", "gpt2_finetuned", "empathy_chain"]:
        path = f"outputs/generations/{tag}.jsonl"
        if os.path.exists(path):
            with open(path) as f:
                all_gens.extend([json.loads(l) for l in f])
    gen_by_id = {g["id"]: g for g in all_gens}

    # Select sample subset
    sample_ids = list(gen_by_id.keys())[:args.n_samples]

    if args.dry_run:
        # Print prompt comparisons
        example = all_gens[0] if all_gens else {"user_statement": "I'm sad", "response": "Tell me more"}
        for vname in PROMPT_VARIANTS:
            msgs = build_variant_messages(
                vname, example["user_statement"][:200], example["response"][:200]
            )
            print(f"\n{'='*60}")
            print(f"VARIANT: {vname}")
            print(f"{'='*60}")
            print(f"System ({len(msgs[0]['content'])} chars):")
            print(msgs[0]["content"][:500])
            print(f"User ({len(msgs[1]['content'])} chars):")
            print(msgs[1]["content"][:300])
        print("\nRun without --dry_run to call the API.")
        return

    # --- Run judge for each variant ---
    # Note: 'default' results already exist; we reuse them
    all_analyses = {}

    # Analyse existing default results
    default_subset = [r for r in all_judge_default if r.get("sample_id") in set(sample_ids)]
    if default_subset:
        print(f"\nAnalysing default prompt ({len(default_subset)} results)...")
        all_analyses["default"] = analyse_variant_results(
            default_subset, all_human, all_judge_default, gen_by_id
        )

    # For other variants, call the API
    for vname in ["strict", "minimal"]:
        print(f"\n{'='*60}")
        print(f"Running variant: {vname}")
        print(f"{'='*60}")

        variant_results = []
        cache_path = os.path.join(args.output_dir, f"ablation_judge_{vname}.jsonl")

        # Check for cached results
        if os.path.exists(cache_path):
            print(f"  Loading cached results from {cache_path}")
            with open(cache_path) as f:
                variant_results = [json.loads(l) for l in f if l.strip()]
            print(f"  Loaded {len(variant_results)} cached results")
        else:
            # Need API access — try to import and call
            try:
                from openai import OpenAI
                client = OpenAI(
                    api_key=os.environ.get("DEEPSEEK_API_KEY", os.environ.get("OPENAI_API_KEY", "")),
                    base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                )

                def api_fn(messages, model="deepseek-chat", temperature=0.3):
                    resp = client.chat.completions.create(
                        model=model, messages=messages, temperature=temperature, max_tokens=500,
                    )
                    return resp.choices[0].message.content

                with open(cache_path, "w") as fout:
                    for i, sid in enumerate(sample_ids):
                        gen = gen_by_id[sid]
                        for rep in range(args.n_repeats):
                            msgs = build_variant_messages(
                                vname, gen["user_statement"], gen["response"]
                            )
                            try:
                                raw = api_fn(msgs)
                                parsed = extract_json(raw) if raw else None
                                if parsed:
                                    validated = validate_judge_output(parsed)
                                    if validated:
                                        record = {
                                            "sample_id": sid,
                                            "model": gen.get("model", "unknown"),
                                            "repeat_idx": rep,
                                            "variant": vname,
                                            **validated,
                                        }
                                        variant_results.append(record)
                                        fout.write(json.dumps(record) + "\n")
                            except Exception as e:
                                print(f"    Error: {e}")
                            time.sleep(0.5)

                        if (i + 1) % 10 == 0:
                            print(f"  {vname}: {i+1}/{len(sample_ids)} samples done")

                print(f"  {vname}: {len(variant_results)} total results → {cache_path}")

            except ImportError:
                print(f"  SKIP {vname}: openai package not available. "
                      f"Run with API access or provide cached results at {cache_path}")
                continue

        if variant_results:
            all_analyses[vname] = analyse_variant_results(
                variant_results, all_human, all_judge_default, gen_by_id
            )

    # --- Save report ---
    json_path = os.path.join(args.output_dir, "ablation_prompt.json")
    with open(json_path, "w") as f:
        json.dump(all_analyses, f, indent=2, default=str)
    print(f"\nJSON → {json_path}")

    md = generate_markdown(all_analyses)
    md_path = os.path.join(args.output_dir, "ablation_prompt.md")
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Markdown → {md_path}")


if __name__ == "__main__":
    main()
