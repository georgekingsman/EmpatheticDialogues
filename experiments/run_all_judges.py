"""
Run LLM-as-a-Judge on all 3 model generation files using DeepSeek Chat.

Each sample is judged n_repeats times (default 3) for stability analysis.
Output: one JSONL per model in outputs/judge/
"""
import sys, os
sys.path.insert(0, ".")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["DEEPSEEK_API_KEY"] = os.environ.get("DEEPSEEK_API_KEY", "sk-f01763598a4f41d3ac32874aafc40d62")

from src.eval.llm_judge import judge_batch, deepseek_api_fn, save_judge_results
from src.inference.generate import load_jsonl

GENERATION_FILES = {
    "gpt2_vanilla": "outputs/generations/gpt2_vanilla.jsonl",
    "gpt2_finetuned": "outputs/generations/gpt2_finetuned.jsonl",
    "empathy_chain": "outputs/generations/empathy_chain.jsonl",
}

JUDGE_MODEL = "deepseek-chat"
N_REPEATS = 3
TEMPERATURE = 0.3
DELAY = 0.3  # seconds between API calls

def main():
    os.makedirs("outputs/judge", exist_ok=True)

    for model_tag, gen_path in GENERATION_FILES.items():
        print(f"\n{'='*60}")
        print(f"Judging: {model_tag} ({gen_path})")
        print(f"{'='*60}")

        generations = load_jsonl(gen_path)
        print(f"  Loaded {len(generations)} samples, {N_REPEATS} repeats each = {len(generations)*N_REPEATS} calls")

        results = judge_batch(
            generations,
            api_fn=deepseek_api_fn,
            judge_model=JUDGE_MODEL,
            temperature=TEMPERATURE,
            n_repeats=N_REPEATS,
            delay_between=DELAY,
        )

        out_path = f"outputs/judge/{model_tag}_judge.jsonl"
        save_judge_results(results, out_path)

        # Quick stats
        scored = [r for r in results if "scores" in r]
        errors = [r for r in results if "error" in r]
        if scored:
            avg_overall = sum(r["overall"] for r in scored) / len(scored)
            avg_dims = {}
            for dim in ["emotion", "validation", "helpfulness", "safety"]:
                avg_dims[dim] = sum(r["scores"][dim] for r in scored) / len(scored)
            print(f"  Results: {len(scored)} scored, {len(errors)} errors")
            print(f"  Avg overall: {avg_overall:.2f}")
            for dim, val in avg_dims.items():
                print(f"    {dim}: {val:.2f}")
        else:
            print(f"  WARNING: No successfully scored results! {len(errors)} errors")

    print("\n\nAll judge evaluations complete!")
    print("Output files:")
    for model_tag in GENERATION_FILES:
        print(f"  outputs/judge/{model_tag}_judge.jsonl")

if __name__ == "__main__":
    main()
