#!/usr/bin/env python3
"""
Phase 13.4: CPU Latency Benchmarking Script for Vietnamese KWS Release Candidate.
Workspace: D:\trainidentificationaudio
Target: experiments/vieneu_kws_research_v13_4/bundle/benchmark_latency.py

Benchmarks single-clip WAV-to-prediction CPU latency across development fixtures
for both PyTorch and ONNX Runtime execution backends.
"""

import sys
import time
import json
import statistics
import numpy as np
from pathlib import Path

BUNDLE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BUNDLE_DIR.parents[2]
DEV_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_1" / "fixtures"

from infer import VietnameseKWSInferencer


def benchmark_backend(backend: str, iterations: int = 100, warmup: int = 10) -> dict:
    inferencer = VietnameseKWSInferencer(backend=backend)
    fixture_files = sorted(list(DEV_DIR.glob("*.wav")))
    assert len(fixture_files) >= 5, f"Expected at least 5 fixtures, found {len(fixture_files)}"

    # 1. Warm-up runs
    for i in range(warmup):
        f = fixture_files[i % len(fixture_files)]
        inferencer.predict(f)

    # 2. Benchmark runs
    latencies_ms = []
    for i in range(iterations):
        f = fixture_files[i % len(fixture_files)]
        t0 = time.perf_counter()
        inferencer.predict(f)
        dt = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt)

    lat_np = np.array(latencies_ms)
    mean_lat = float(np.mean(lat_np))
    median_lat = float(np.median(lat_np))
    p95_lat = float(np.percentile(lat_np, 95))
    min_lat = float(np.min(lat_np))
    max_lat = float(np.max(lat_np))
    std_lat = float(np.std(lat_np))

    return {
        "backend": backend,
        "iterations": iterations,
        "warmup_runs": warmup,
        "fixtures_used": len(fixture_files),
        "mean_ms": round(mean_lat, 2),
        "median_p50_ms": round(median_lat, 2),
        "p95_ms": round(p95_lat, 2),
        "min_ms": round(min_lat, 2),
        "max_ms": round(max_lat, 2),
        "std_ms": round(std_lat, 2),
        "meets_50ms_sla": (mean_lat <= 50.0) and (p95_lat <= 50.0)
    }


def run_benchmark():
    print("=" * 70)
    print("VIETNAMESE KWS LATENCY BENCHMARK (Phase 13.4 Candidate)")
    print("=" * 70)

    results = {}
    for b in ["pytorch", "onnx"]:
        print(f"Benchmarking backend: {b} (10 warmup, 100 iterations)...")
        res = benchmark_backend(b, iterations=100, warmup=10)
        results[b] = res
        print(f"  Mean: {res['mean_ms']} ms | Median: {res['median_p50_ms']} ms | p95: {res['p95_ms']} ms | Max: {res['max_ms']} ms | SLA Pass: {res['meets_50ms_sla']}")

    out_json = BUNDLE_DIR / "latency_benchmark_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Latency benchmark complete. Results saved to: {out_json.name}")
    return results


if __name__ == "__main__":
    run_benchmark()
