"""
Reproducible Entry Point 3: Extended Streaming Benchmark
Workspace: D:\\trainidentificationaudio

Runs the 3.00-hour (10,800s) target-absent stream evaluation and the 200-event target-present
stream benchmark, saving all stream-level and event-level CSV files.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve()))
from experiments.synthetic_kws_phase20_22.phase21_robustness.phase21_streaming_benchmark import run_phase21_benchmark

if __name__ == "__main__":
    print("Running reproducible streaming benchmark...")
    run_phase21_benchmark()
    print("Streaming benchmark complete.")
