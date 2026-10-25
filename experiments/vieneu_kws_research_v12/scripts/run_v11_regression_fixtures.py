#!/usr/bin/env python3
"""
Phase 18 Gate B & D: Legacy v11 Regression Test Harness.
Workspace: D:\trainidentificationaudio
Target: experiments/vieneu_kws_research_v12/scripts/run_v11_regression_fixtures.py

Runs the default legacy v11 pipeline on recorded fixtures and verifies zero regression.
Modes:
  --record: Captures pre-change baseline to JSON.
  --verify: Compares current run against recorded baseline with 100% exact equality requirement.
"""

import sys
import io
import json
import hashlib
import argparse
from pathlib import Path
from typing import Dict, Any, List

# Ensure UTF-8 console output
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEMO_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4" / "demo"
BUNDLE_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4" / "bundle"
for p in [str(DEMO_DIR), str(BUNDLE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import soundfile as sf

from live_mic_engine import LiveKWSPipeline, CLASSES, TARGET_CLASSES, DEFAULT_THRESHOLD
from infer import preprocess_audio

V11_ONNX_PATH = BUNDLE_DIR / "onnx" / "vietnamese_kws_v11.onnx"
EXPECTED_V11_ONNX_HASH = "54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99"

V11_CKPT_PATH = BUNDLE_DIR / "checkpoint" / "variant_v11_hard_negative_weighted_seed42.pt"
EXPECTED_V11_CKPT_HASH = "3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2"

BASELINE_OUTPUT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "vieneu_kws_research_v12"
    / "reports"
    / "v11_pre_change_baseline.json"
)

FIXTURE_DIRS = [
    PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "fixtures",
    PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_1" / "fixtures"
]


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def collect_fixtures() -> List[Path]:
    fixtures = []
    for fdir in FIXTURE_DIRS:
        if fdir.exists():
            fixtures.extend(sorted(fdir.glob("*.wav")))
    # Deduplicate by name if any
    unique_fixtures = []
    seen_names = set()
    for f in fixtures:
        if f.name not in seen_names:
            seen_names.add(f.name)
            unique_fixtures.append(f)
    return sorted(unique_fixtures, key=lambda p: p.name)


def run_fixtures(pipeline: LiveKWSPipeline, fixtures: List[Path]) -> List[Dict[str, Any]]:
    results = []
    for wav_path in fixtures:
        audio = preprocess_audio(wav_path)
        pred_class, conf, probs, dt_ms = pipeline._run_in_memory_inference(audio)
        rec = {
            "fixture": wav_path.name,
            "path": str(wav_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "predicted_class": pred_class,
            "confidence": round(float(conf), 4),
            "probabilities": {k: round(float(v), 4) for k, v in probs.items()}
        }
        results.append(rec)
    return results


def main():
    parser = argparse.ArgumentParser(description="Legacy v11 Regression Test Harness")
    parser.add_argument("--record", action="store_true", help="Record pre-change baseline JSON")
    parser.add_argument("--verify", action="store_true", help="Verify current pipeline matches baseline JSON")
    args = parser.parse_args()

    # Default to verify if baseline exists and neither flag specified
    if not args.record and not args.verify:
        if BASELINE_OUTPUT_PATH.exists():
            args.verify = True
        else:
            args.record = True

    print("=" * 80)
    print("PHASE 18: LEGACY v11 REGRESSION HARNESS")
    print(f"Action: {'RECORD BASELINE' if args.record else 'VERIFY ZERO REGRESSION'}")
    print("=" * 80)

    # 1. Verify frozen artifacts integrity
    assert V11_ONNX_PATH.exists(), f"v11 ONNX missing: {V11_ONNX_PATH}"
    actual_onnx_hash = compute_sha256(V11_ONNX_PATH)
    assert actual_onnx_hash == EXPECTED_V11_ONNX_HASH, (
        f"v11 ONNX hash mismatch!\nExpected: {EXPECTED_V11_ONNX_HASH}\nActual:   {actual_onnx_hash}"
    )

    assert V11_CKPT_PATH.exists(), f"v11 Checkpoint missing: {V11_CKPT_PATH}"
    actual_ckpt_hash = compute_sha256(V11_CKPT_PATH)
    assert actual_ckpt_hash == EXPECTED_V11_CKPT_HASH, (
        f"v11 Checkpoint hash mismatch!\nExpected: {EXPECTED_V11_CKPT_HASH}\nActual:   {actual_ckpt_hash}"
    )
    print(f"[OK] Verified v11 ONNX:       {actual_onnx_hash[:16]}...")
    print(f"[OK] Verified v11 Checkpoint: {actual_ckpt_hash[:16]}...")

    # 2. Collect test fixtures
    fixtures = collect_fixtures()
    print(f"[OK] Collected {len(fixtures)} test fixtures across {len(FIXTURE_DIRS)} directories.")

    # 3. Instantiate default legacy LiveKWSPipeline
    pipeline = LiveKWSPipeline(
        mode="segmented",
        backend="onnx",
        model_path=V11_ONNX_PATH,
        theta_cmd=DEFAULT_THRESHOLD
    )

    # 4. Run inference
    current_results = run_fixtures(pipeline, fixtures)

    if args.record:
        record_payload = {
            "description": "Pre-change legacy v11 baseline on test fixtures",
            "v11_onnx_sha256": actual_onnx_hash,
            "v11_checkpoint_sha256": actual_ckpt_hash,
            "theta_cmd": DEFAULT_THRESHOLD,
            "num_fixtures": len(fixtures),
            "results": current_results
        }
        BASELINE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(BASELINE_OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(record_payload, f, indent=2, ensure_ascii=False)
        print(f"\n[OK] Baseline successfully recorded to: {BASELINE_OUTPUT_PATH.name}")
        for r in current_results:
            print(f"  {r['fixture']:<42} -> {r['predicted_class']:<6} (conf={r['confidence']:.4f})")
        return

    if args.verify:
        if not BASELINE_OUTPUT_PATH.exists():
            raise FileNotFoundError(f"Baseline file does not exist: {BASELINE_OUTPUT_PATH}")
        with open(BASELINE_OUTPUT_PATH, "r", encoding="utf-8") as f:
            baseline = json.load(f)

        assert baseline["v11_onnx_sha256"] == actual_onnx_hash
        baseline_results = {r["fixture"]: r for r in baseline["results"]}

        mismatches = []
        for curr in current_results:
            fname = curr["fixture"]
            if fname not in baseline_results:
                mismatches.append(f"Fixture {fname} not found in baseline")
                continue
            base = baseline_results[fname]
            if curr["predicted_class"] != base["predicted_class"]:
                mismatches.append(
                    f"Label mismatch on {fname}: current={curr['predicted_class']} vs base={base['predicted_class']}"
                )
            # Probabilities comparison (within 1e-4)
            for c in CLASSES:
                diff = abs(curr["probabilities"][c] - base["probabilities"][c])
                if diff > 1e-4:
                    mismatches.append(
                        f"Prob diff on {fname} class {c}: {curr['probabilities'][c]} vs {base['probabilities'][c]} (diff={diff:.6f})"
                    )

        if mismatches:
            print(f"\n[FAIL] Found {len(mismatches)} regressions against baseline:")
            for m in mismatches:
                print(f"  - {m}")
            raise SystemExit(1)
        else:
            print(f"\n[PASS] 100% PERFECT MATCH across all {len(fixtures)} fixtures!")
            print("Zero regressions observed in legacy v11 behavior.")


if __name__ == "__main__":
    main()
