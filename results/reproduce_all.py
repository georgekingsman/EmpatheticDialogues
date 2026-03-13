#!/usr/bin/env python3
"""Master reproducibility script — runs the full pipeline from frozen data.

Usage:
    python results/reproduce_all.py [--skip-generation] [--skip-judge]

This script reads from results/offline_eval_v2_final/ and regenerates:
  1. Statistical analysis → statistics.json
  2. Composite indices → composite_stats.json
  3. Paper figures (5) → figures/
  4. LaTeX tables (3) → tables/

If --skip-generation and --skip-judge are NOT passed, the script also
regenerates outputs and judge scores (requires DEEPSEEK_API_KEY).

The default mode (with both flags) is a fast local rebuild from frozen data.
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

def run_step(label, cmd, cwd=ROOT):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, cwd=str(cwd))
    if result.returncode != 0:
        print(f"  ✗ FAILED: {label}")
        sys.exit(1)
    print(f"  ✓ {label}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip response generation (use frozen outputs)")
    parser.add_argument("--skip-judge", action="store_true",
                        help="Skip judge scoring (use frozen scores)")
    args = parser.parse_args()

    print("=" * 60)
    print("  REPRODUCIBILITY PIPELINE")
    print("  Frozen data: results/offline_eval_v2_final/")
    print("=" * 60)

    if not args.skip_generation:
        run_step("Generate Condition A (Single Agent)",
                 "python generation/run_single.py")
        run_step("Generate Condition B (Hidden Checker)",
                 "python generation/run_double_hidden.py")
        run_step("Generate Condition C (Visible Checker)",
                 "python generation/run_double_visible.py")

    if not args.skip_judge:
        run_step("Run Primary Judge",
                 "python results/run_statistics.py")
        run_step("Run Second Judge (Stricter)",
                 "python results/run_second_judge.py")
        run_step("Run Multi-Rater Simulation",
                 "python results/run_multi_rater.py")

    # Always regenerate analysis from frozen data
    run_step("Generate Composite Stats + Figures + Tables",
             "python results/generate_paper_assets.py")

    print("\n" + "=" * 60)
    print("  ✅ PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\n  Figures: results/offline_eval_v2_final/figures/")
    print(f"  Tables:  results/offline_eval_v2_final/tables/")
    print(f"  Stats:   results/offline_eval_v2_final/composite_stats.json")
    print()

if __name__ == "__main__":
    main()
