#!/usr/bin/env python3
"""
Phase 12 Methodological Latency Benchmark Tool.
Workspace: D:\trainidentificationaudio
Target: experiments/vieneu_kws_research_v12/bundle/benchmark_latency.py
"""

import sys
import time
import json
from pathlib import Path
import numpy as np
import torch

from infer import KWSInferenceEngine, DEFAULT_CKPT, DEFAULT_THRESHOLD

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_WAV = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v10" / "human_review" / "audio" / "review_clip_001.wav"


def run_latency_benchmark(num_warmup=100, num_trials=1000, device="cpu", output_json=None):
    torch.set_num_threads(4)
    engine = KWSInferenceEngine(checkpoint_path=DEFAULT_CKPT, theta_cmd=DEFAULT_THRESHOLD, device=device)

    print(f"Benchmarking inference latency on {device} ({num_warmup} warm-up, {num_trials} benchmark trials)...")

    # Warm-up
    for _ in range(num_warmup):
        _ = engine.predict(FIXTURE_WAV)

    # Benchmark loop
    timings_ms = []
    for _ in range(num_trials):
        t0 = time.perf_counter()
        _ = engine.predict(FIXTURE_WAV)
        dt = (time.perf_counter() - t0) * 1000.0
        timings_ms.append(dt)

    timings_ms = np.array(timings_ms)

    metrics = {
        "device": device,
        "threads": torch.get_num_threads(),
        "num_warmup": num_warmup,
        "num_trials": num_trials,
        "mean_ms": round(float(np.mean(timings_ms)), 3),
        "std_ms": round(float(np.std(timings_ms)), 3),
        "median_ms": round(float(np.median(timings_ms)), 3),
        "p90_ms": round(float(np.percentile(timings_ms, 90)), 3),
        "p95_ms": round(float(np.percentile(timings_ms, 95)), 3),
        "p99_ms": round(float(np.percentile(timings_ms, 99)), 3),
        "min_ms": round(float(np.min(timings_ms)), 3),
        "max_ms": round(float(np.max(timings_ms)), 3),
        "target_ms": 50.0,
        "status": "PASSED_LATENCY_BUDGET" if float(np.mean(timings_ms)) <= 50.0 else "EXCEEDED_LATENCY_BUDGET"
    }

    print("\n--- Latency Benchmark Results ---")
    print(f"Mean Latency:   {metrics['mean_ms']} ± {metrics['std_ms']} ms")
    print(f"Median:         {metrics['median_ms']} ms")
    print(f"95th Percentile:{metrics['p95_ms']} ms")
    print(f"99th Percentile:{metrics['p99_ms']} ms")
    print(f"Status:         {metrics['status']} (Budget: <= 50.0 ms)")

    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved benchmark results to {output_json}")

    return metrics


if __name__ == "__main__":
    out_path = Path(__file__).resolve().parent / "latency_benchmark.json"
    run_latency_benchmark(output_json=out_path)
