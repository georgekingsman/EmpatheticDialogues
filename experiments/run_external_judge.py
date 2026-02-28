"""
Step 2: Run LLM-as-a-Judge on external dataset.

Reads a unified external JSONL (produced by src.data.external_loader),
invokes the existing judge pipeline (src.eval.llm_judge), and writes
per-sample judge scores to outputs/judge_external/.

Usage:
    python experiments/run_external_judge.py \\
        --input data/external/unified.jsonl \\
        --dataset my_dataset \\
        --judge_model deepseek-chat \\
        --judge_backend deepseek \\
        --n_repeats 3

Outputs:
    outputs/judge_external/<dataset>_<judge_model>.jsonl
"""

import sys, os, json, argparse
sys.path.insert(0, ".")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path

from src.data.external_loader import load_external, convert_to_generation_format
from src.eval.llm_judge import (
    judge_batch,
    save_judge_results,
    load_judge_results,
    openai_api_fn,
    deepseek_api_fn,
)


def main():
    parser = argparse.ArgumentParser(description="Run LLM judge on external dataset")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to unified external JSONL (from external_loader)")
    parser.add_argument("--dataset", type=str, default="external",
                        help="Dataset name tag for output file naming")
    parser.add_argument("--judge_model", type=str, default="deepseek-chat",
                        help="Judge model name (default: deepseek-chat)")
    parser.add_argument("--judge_backend", choices=["openai", "deepseek"],
                        default="deepseek",
                        help="API backend (default: deepseek)")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--n_repeats", type=int, default=3,
                        help="Number of judge repeats per sample (default: 3)")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Delay between API calls in seconds")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of samples to judge (for testing)")
    parser.add_argument("--output_dir", type=str, default="outputs/judge_external",
                        help="Output directory")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing partial output file")

    # Also support raw external file loading params
    parser.add_argument("--dataset_format", type=str, default="generic",
                        choices=["generic", "epitome", "empatheticdialogues_eval"],
                        help="Format of input file (if not pre-converted JSONL)")
    parser.add_argument("--prompt_col", default="prompt")
    parser.add_argument("--response_col", default="response")
    parser.add_argument("--score_col", default="overall")
    parser.add_argument("--score_min", type=float, default=1.0)
    parser.add_argument("--score_max", type=float, default=5.0)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Determine output path ---
    judge_tag = args.judge_model.replace("/", "_").replace("-", "_")
    output_path = os.path.join(args.output_dir, f"{args.dataset}_{judge_tag}.jsonl")

    # --- Load data ---
    print("=" * 60)
    print("Loading external data")
    print("=" * 60)

    input_path = Path(args.input)
    if input_path.suffix == ".jsonl":
        # Try loading as pre-converted unified JSONL first
        records = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        # Check if it's already in unified format
        if records and "item_id" in records[0]:
            print(f"  Loaded {len(records)} pre-converted records")
        else:
            # Load via external_loader
            records = load_external(
                args.input,
                dataset=args.dataset_format,
                prompt_col=args.prompt_col,
                response_col=args.response_col,
                score_col=args.score_col,
                score_min=args.score_min,
                score_max=args.score_max,
            )
    else:
        records = load_external(
            args.input,
            dataset=args.dataset_format,
            prompt_col=args.prompt_col,
            response_col=args.response_col,
            score_col=args.score_col,
            score_min=args.score_min,
            score_max=args.score_max,
        )

    # Convert to generation format for judge_batch
    generations = convert_to_generation_format(records)

    if args.max_samples:
        generations = generations[:args.max_samples]
        print(f"  Limited to {args.max_samples} samples")

    # --- Resume support ---
    done_ids = set()
    existing_results = []
    if args.resume and os.path.exists(output_path):
        existing_results = load_judge_results(output_path)
        for r in existing_results:
            done_ids.add(r["sample_id"])
        print(f"  Resuming: {len(done_ids)} samples already judged")
        generations = [g for g in generations if g["id"] not in done_ids]

    print(f"  Samples to judge: {len(generations)}")
    print(f"  Repeats: {args.n_repeats}")
    print(f"  Total API calls: {len(generations) * args.n_repeats}")
    print(f"  Judge model: {args.judge_model} (backend: {args.judge_backend})")
    print(f"  Output: {output_path}")

    if not generations:
        print("\n  No samples to judge (all done or empty input). Exiting.")
        return

    # --- Select API function ---
    if args.judge_backend == "openai":
        api_fn = openai_api_fn
    else:
        api_fn = deepseek_api_fn

    # --- Run judge ---
    print(f"\n{'=' * 60}")
    print("Running LLM judge")
    print("=" * 60)

    results = judge_batch(
        generations,
        api_fn=api_fn,
        judge_model=args.judge_model,
        temperature=args.temperature,
        n_repeats=args.n_repeats,
        delay_between=args.delay,
    )

    # Merge with any existing results
    all_results = existing_results + results

    # --- Save ---
    save_judge_results(all_results, output_path)

    # --- Summary ---
    n_success = sum(1 for r in results if "scores" in r)
    n_error = sum(1 for r in results if "error" in r)
    print(f"\n{'=' * 60}")
    print("Summary")
    print("=" * 60)
    print(f"  New results: {len(results)} ({n_success} success, {n_error} errors)")
    print(f"  Total results: {len(all_results)}")
    print(f"  Output: {output_path}")

    if n_success > 0:
        # Quick score distribution
        import numpy as np
        overalls = [r["overall"] for r in results if "overall" in r]
        print(f"  Overall score dist: mean={np.mean(overalls):.2f}, "
              f"std={np.std(overalls):.2f}, "
              f"min={min(overalls)}, max={max(overalls)}")

    print("\n✅ External judge scoring complete.")


if __name__ == "__main__":
    main()
