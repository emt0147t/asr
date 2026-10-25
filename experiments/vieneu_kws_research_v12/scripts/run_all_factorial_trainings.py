#!/usr/bin/env python3
"""
Phase 25 Factorial Experiment Orchestrator
Repository: D:\\trainidentificationaudio
Target: experiments/vieneu_kws_research_v12/scripts/run_all_factorial_trainings.py

Orchestrates all 12 training runs of the 2x2 Factorial Experiment:
Conditions: D0_H0, D0_H1, D1_H0, D1_H1
Seeds: 42, 43, 44
Total runs: 12
"""

import sys
import io
import time
import json
from pathlib import Path
import pandas as pd

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "scripts"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "training_phase25"
REPORTS_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "reports"

sys.path.insert(0, str(SCRIPTS_DIR))
from train_phase25_factorial import train_factorial_run

CONDITIONS = ["D0_H0", "D0_H1", "D1_H0", "D1_H1"]
SEEDS = [42, 43, 44]


def main():
    print("=" * 80)
    print("PHASE 25: EXECUTING COMPLETE 2x2 FACTORIAL TRAINING EXPERIMENT (12 RUNS)")
    print("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    t_global_start = time.perf_counter()

    total_runs = len(CONDITIONS) * len(SEEDS)
    current_run = 0

    for cond in CONDITIONS:
        for seed in SEEDS:
            current_run += 1
            print(f"\n>>> RUN {current_run}/{total_runs}: Condition = {cond}, Seed = {seed} <<<")
            
            # Check if already completed
            summary_file = OUTPUT_DIR / f"{cond}_seed{seed}" / "run_summary.json"
            if summary_file.exists():
                print(f"Run {cond}_seed{seed} already completed. Loading existing summary.")
                with open(summary_file, "r", encoding="utf-8") as f:
                    summary = json.load(f)
            else:
                summary = train_factorial_run(condition=cond, seed=seed)

            all_summaries.append(summary)

    total_elapsed = time.perf_counter() - t_global_start
    print("\n" + "=" * 80)
    print(f"ALL 12 FACTORIAL RUNS COMPLETED IN {total_elapsed/60:.2f} MINUTES")
    print("=" * 80)

    # Save aggregated execution ledger
    ledger_path = REPORTS_DIR / "phase25_factorial_execution_ledger.json"
    with open(ledger_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "phase": "Phase 25 Gate E Execution",
                "total_runs": len(all_summaries),
                "conditions": CONDITIONS,
                "seeds": SEEDS,
                "total_execution_time_seconds": round(total_elapsed, 2)
            },
            "runs": all_summaries
        }, f, indent=2)

    # Print summary table
    print("\nExecution Summary Table:")
    print(f"{'Condition':<10} | {'Seed':<5} | {'Best Ep':<7} | {'Val Acc':<8} | {'Val Macro-F1':<12} | {'Time (s)':<8} | {'Checkpoint SHA':<16}")
    print("-" * 80)
    for s in all_summaries:
        print(f"{s['condition']:<10} | {s['seed']:<5} | {s['best_epoch']:<7} | {s['best_val_accuracy']*100:.2f}%  | {s['best_val_macro_f1']*100:.2f}%       | {s['training_duration_seconds']:<8.1f} | {s['checkpoint_sha256'][:16]}...")

    print(f"\nExecution ledger saved to: {ledger_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
