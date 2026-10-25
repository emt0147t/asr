#!/usr/bin/env python3
"""
Phase 19: Controlled Offline Profile B Validation Harness.
Workspace: D:\trainidentificationaudio
Target: experiments/vieneu_kws_research_v12/scripts/run_phase19_offline_validation.py

Executes:
  - Gate C: Controlled development replay on vad_dev_corpus (60 sequences) and development fixtures.
  - Gate D: Observed test reproduction on vad_test_corpus (60 sequences) comparing with Phase 16 audit baseline.
Outputs full event ledger and metrics to profile_b_offline_validation_results.json.
"""

import sys
import io
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Ensure UTF-8 console output
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEMO_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4" / "demo"
BUNDLE_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4" / "bundle"
V12_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12"

for p in [str(DEMO_DIR), str(BUNDLE_DIR), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd
import soundfile as sf

from live_mic_engine import (
    LiveKWSPipeline,
    DetectionEvent,
    CLASSES,
    TARGET_CLASSES,
    DEFAULT_THRESHOLD
)
from infer import preprocess_audio

E1_ONNX_PATH = V12_DIR / "onnx" / "vietnamese_kws_v12_e1.onnx"
VAD_CONFIG_PATH = V12_DIR / "configs" / "selected_phase15_vad_config.json"
DEV_MANIFEST_PATH = V12_DIR / "manifests" / "vad_dev_corpus_manifest.csv"
TEST_MANIFEST_PATH = V12_DIR / "manifests" / "vad_test_corpus_manifest.csv"
OUTPUT_REPORT_PATH = V12_DIR / "reports" / "profile_b_offline_validation_results.json"

EXPECTED_E1_ONNX_HASH = "20ecc2964d88d5a9d1d15b1c665e9697a32d702b1e3dd12ab1e5f829aff016b8"
EXPECTED_VAD_CONFIG_HASH = "af5c3891fe73f831bb2972f2ce4722b3cbb6b669ad306bd408b8e941ee7de9b3"
EXPECTED_DEV_MANIFEST_HASH = "1ebf11f8c7048e162e04d24df92b40e75b026c7059d9a9af5afcbab1274587c3"
EXPECTED_TEST_MANIFEST_HASH = "d0342e3f80c1307801c81637ba5b02e002c598aded35577b525afc106a8b5876"


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def calculate_temporal_iou(s1: float, e1: float, s2: float, e2: float) -> float:
    inter_s = max(s1, s2)
    inter_e = min(e1, e2)
    intersection = max(0.0, inter_e - inter_s)
    union = max(e1, e2) - min(s1, s2)
    return intersection / union if union > 1e-9 else 0.0


def replay_corpus_through_pipeline(
    manifest_csv: Path,
    pipeline: LiveKWSPipeline,
    chunk_samples: int = 320
) -> Dict[str, Any]:
    """
    Feeds a sequential speech corpus chunk-by-chunk through LiveKWSPipeline (Profile B)
    using the active chunk callback and event callback architecture.
    """
    df = pd.read_csv(manifest_csv)
    split_name = df["corpus_split"].iloc[0]

    total_target_events = 0
    total_matched_commands = 0
    total_missed_commands = 0
    total_substitutions = 0
    total_oov_false_activations = 0
    total_silence_false_activations = 0
    total_duplicate_triggers = 0
    exact_correct_sequences = 0

    all_latencies_ms = []
    total_negative_seconds = 0.0
    total_oov_seconds = 0.0
    total_silence_seconds = 0.0

    gap_stats = {
        gap: {"target_events": 0, "matched": 0, "exact_seqs": 0, "total_seqs": 0, "fa_total": 0}
        for gap in [100, 150, 200, 250, 300, 400]
    }
    cond_stats = {
        cond: {"target_events": 0, "matched": 0, "exact_seqs": 0, "total_seqs": 0, "fa_total": 0}
        for cond in ["clean", "snr20", "snr15", "snr10", "snr05"]
    }

    per_sequence_records = []

    for idx, row in df.iterrows():
        seq_id = row["sequence_id"]
        wav_path = PROJECT_ROOT / row["audio_path"]
        ref_events = json.loads(row["events_json"])
        nominal_gap = int(row["nominal_gap_ms"])
        cond = row["condition"]
        total_duration = float(row["duration_seconds"])

        # Reference targets
        target_ground_truth = [e["label"] for e in ref_events if e["is_target"]]
        n_targets_in_seq = len(target_ground_truth)
        total_target_events += n_targets_in_seq

        # Negative durations
        oov_dur = sum(e["duration_sec"] for e in ref_events if not e["is_target"])
        target_dur = sum(e["duration_sec"] for e in ref_events if e["is_target"])
        silence_dur = max(0.0, total_duration - (target_dur + oov_dur))
        total_oov_seconds += oov_dur
        total_silence_seconds += silence_dur
        total_negative_seconds += (oov_dur + silence_dur)

        # Re-arm pipeline and clear segmenter
        pipeline.segmenter.reset()
        pipeline.recent_events = []
        pipeline.is_running = True

        emitted_events: List[DetectionEvent] = []

        def seq_callback(ev: DetectionEvent):
            emitted_events.append(ev)

        pipeline.event_callback = seq_callback

        audio, sr = sf.read(str(wav_path), dtype="float32")
        assert sr == 16000

        # Feed audio chunk-by-chunk
        for c_start in range(0, len(audio), chunk_samples):
            chunk = audio[c_start : c_start + chunk_samples]
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
            pipeline._on_new_audio_chunk(chunk)

        # Feed 360ms trailing silence to allow segment finalization
        trailing_chunk = np.zeros(chunk_samples, dtype=np.float32)
        n_trailing = int(round((0.360 * 16000) / chunk_samples))
        for _ in range(n_trailing):
            pipeline._on_new_audio_chunk(trailing_chunk)

        # Commands accepted and emitted by pipeline
        emitted_commands = [ev.command for ev in emitted_events]

        # Check exact sequence match
        is_exact = (emitted_commands == target_ground_truth)
        if is_exact:
            exact_correct_sequences += 1

        matched_target_refs = set()
        matched_pred_cmds = set()
        seq_fa_oov = 0
        seq_fa_silence = 0
        seq_duplicates = 0

        # Temporal matching of emitted events against reference events
        for p_idx, ev in enumerate(emitted_events):
            p_cls = ev.command
            p_s = ev.segment_start_sec
            p_e = ev.segment_end_sec

            best_iou = 0.0
            best_r_idx = -1
            best_overlap = 0.0

            for r_idx, r_evt in enumerate(ref_events):
                r_s, r_e = r_evt["start_sec"], r_evt["end_sec"]
                iou = calculate_temporal_iou(p_s, p_e, r_s, r_e)
                inter_s = max(p_s, r_s)
                inter_e = min(p_e, r_e)
                overlap = max(0.0, inter_e - inter_s)
                if iou > best_iou:
                    best_iou = iou
                    best_r_idx = r_idx
                    best_overlap = overlap

            # Overlap threshold: IoU >= 0.20 or overlap >= 80ms
            if best_r_idx >= 0 and (best_iou >= 0.20 or best_overlap >= 0.080):
                ref_match = ref_events[best_r_idx]

                if ref_match["is_target"]:
                    if best_r_idx not in matched_target_refs:
                        matched_target_refs.add(best_r_idx)
                        matched_pred_cmds.add(p_idx)
                        if p_cls == ref_match["label"]:
                            total_matched_commands += 1
                            lat_ms = (p_e - ref_match["start_sec"]) * 1000.0
                            all_latencies_ms.append(lat_ms)
                        else:
                            total_substitutions += 1
                    else:
                        total_duplicate_triggers += 1
                        seq_duplicates += 1
                else:
                    # Overlapping an OOV event -> OOV false activation
                    total_oov_false_activations += 1
                    seq_fa_oov += 1
            else:
                # Silence or ambient noise false activation
                total_silence_false_activations += 1
                seq_fa_silence += 1

        seq_missed = n_targets_in_seq - len(matched_target_refs)
        total_missed_commands += seq_missed

        gap_stats[nominal_gap]["target_events"] += n_targets_in_seq
        gap_stats[nominal_gap]["matched"] += len([r for r in matched_target_refs if ref_events[r]["label"] in emitted_commands])
        gap_stats[nominal_gap]["exact_seqs"] += int(is_exact)
        gap_stats[nominal_gap]["total_seqs"] += 1
        gap_stats[nominal_gap]["fa_total"] += (seq_fa_oov + seq_fa_silence)

        cond_stats[cond]["target_events"] += n_targets_in_seq
        cond_stats[cond]["matched"] += len([r for r in matched_target_refs if ref_events[r]["label"] in emitted_commands])
        cond_stats[cond]["exact_seqs"] += int(is_exact)
        cond_stats[cond]["total_seqs"] += 1
        cond_stats[cond]["fa_total"] += (seq_fa_oov + seq_fa_silence)

        per_sequence_records.append({
            "sequence_id": seq_id,
            "nominal_gap_ms": nominal_gap,
            "condition": cond,
            "target_ground_truth": target_ground_truth,
            "emitted_commands": emitted_commands,
            "is_exact_match": is_exact,
            "n_targets": n_targets_in_seq,
            "n_emitted_cmds": len(emitted_commands),
            "fa_oov": seq_fa_oov,
            "fa_silence": seq_fa_silence,
            "duplicates": seq_duplicates,
            "emitted_events": [
                {
                    "command": ev.command,
                    "confidence": ev.confidence,
                    "start_sec": ev.segment_start_sec,
                    "end_sec": ev.segment_end_sec,
                    "duration_ms": ev.segment_duration_ms
                }
                for ev in emitted_events
            ]
        })

    # Event-level summary metrics
    n_total_emitted_cmds = total_matched_commands + total_substitutions + total_oov_false_activations + total_silence_false_activations + total_duplicate_triggers
    cmd_precision = total_matched_commands / max(1, n_total_emitted_cmds)
    cmd_recall = total_matched_commands / max(1, total_target_events)
    cmd_f1 = (2.0 * cmd_precision * cmd_recall) / max(1e-9, cmd_precision + cmd_recall)

    exact_accuracy = exact_correct_sequences / len(df)
    total_negative_minutes = total_negative_seconds / 60.0
    total_oov_minutes = total_oov_seconds / 60.0
    total_silence_minutes = total_silence_seconds / 60.0

    total_false_activations = total_oov_false_activations + total_silence_false_activations
    fa_per_negative_minute = total_false_activations / max(1e-6, total_negative_minutes)
    fa_per_oov_minute = total_oov_false_activations / max(1e-6, total_oov_minutes)

    median_latency_ms = float(np.median(all_latencies_ms)) if all_latencies_ms else 0.0
    p95_latency_ms = float(np.percentile(all_latencies_ms, 95)) if all_latencies_ms else 0.0

    return {
        "corpus_split": split_name,
        "manifest_path": str(manifest_csv.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "total_sequences": len(df),
        "exact_correct_sequences": exact_correct_sequences,
        "exact_sequence_accuracy": round(exact_accuracy, 4),
        "total_target_events": total_target_events,
        "matched_commands": total_matched_commands,
        "missed_commands": total_missed_commands,
        "substitutions": total_substitutions,
        "oov_false_activations": total_oov_false_activations,
        "silence_false_activations": total_silence_false_activations,
        "duplicate_triggers": total_duplicate_triggers,
        "command_precision": round(cmd_precision, 4),
        "command_recall": round(cmd_recall, 4),
        "command_f1": round(cmd_f1, 4),
        "total_negative_minutes": round(total_negative_minutes, 2),
        "total_oov_minutes": round(total_oov_minutes, 2),
        "total_silence_minutes": round(total_silence_minutes, 2),
        "false_activations_per_negative_min": round(fa_per_negative_minute, 2),
        "false_activations_per_oov_min": round(fa_per_oov_minute, 2),
        "median_latency_ms": round(median_latency_ms, 2),
        "p95_latency_ms": round(p95_latency_ms, 2),
        "gap_breakdown": gap_stats,
        "condition_breakdown": cond_stats,
        "per_sequence_records": per_sequence_records
    }


def main():
    print("=" * 80)
    print("PHASE 19: CONTROLLED OFFLINE PROFILE B VALIDATION")
    print("=" * 80)

    # 1. Cryptographic artifact verification
    print("\n--- Verifying Artifact Cryptographic Hashes ---")
    assert E1_ONNX_PATH.exists()
    assert VAD_CONFIG_PATH.exists()
    assert DEV_MANIFEST_PATH.exists()
    assert TEST_MANIFEST_PATH.exists()

    actual_onnx_hash = compute_sha256(E1_ONNX_PATH)
    actual_vad_hash = compute_sha256(VAD_CONFIG_PATH)
    actual_dev_hash = compute_sha256(DEV_MANIFEST_PATH)
    actual_test_hash = compute_sha256(TEST_MANIFEST_PATH)

    assert actual_onnx_hash == EXPECTED_E1_ONNX_HASH
    assert actual_vad_hash == EXPECTED_VAD_CONFIG_HASH
    assert actual_dev_hash == EXPECTED_DEV_MANIFEST_HASH
    assert actual_test_hash == EXPECTED_TEST_MANIFEST_HASH

    print(f"[OK] E1 ONNX Hash:        {actual_onnx_hash}")
    print(f"[OK] VAD Config Hash:     {actual_vad_hash}")
    print(f"[OK] Dev Manifest Hash:   {actual_dev_hash}")
    print(f"[OK] Test Manifest Hash:  {actual_test_hash}")

    # 2. Instantiate integrated Profile B pipeline
    print("\n--- Instantiating Integrated Profile B Pipeline ---")
    pipeline = LiveKWSPipeline(
        mode="segmented",
        profile="experimental_e1_speechband",
        backend="onnx",
        model_path=E1_ONNX_PATH,
        vad_config_path=VAD_CONFIG_PATH,
        model_hash=actual_onnx_hash,
        vad_config_hash=actual_vad_hash,
        theta_cmd=DEFAULT_THRESHOLD
    )
    print(f"[OK] Profile:       {pipeline.profile_name}")
    print(f"[OK] VAD Algorithm: {pipeline.vad_algorithm}")
    print(f"[OK] Threshold:     {pipeline.theta_cmd}")

    # 3. Gate C: Controlled Development Replay (vad_dev_corpus)
    print("\n" + "=" * 80)
    print("GATE C: CONTROLLED DEVELOPMENT REPLAY (vad_dev_corpus, 60 sequences)")
    print("=" * 80)
    dev_results = replay_corpus_through_pipeline(DEV_MANIFEST_PATH, pipeline, chunk_samples=320)
    print(f"  Exact-Sequence Accuracy:   {dev_results['exact_correct_sequences']} / {dev_results['total_sequences']} ({dev_results['exact_sequence_accuracy'] * 100:.2f}%)")
    print(f"  Command Precision:         {dev_results['command_precision'] * 100:.2f}%")
    print(f"  Command Recall:            {dev_results['command_recall'] * 100:.2f}%")
    print(f"  Command F1:                {dev_results['command_f1']:.4f}")
    print(f"  OOV False Activations:     {dev_results['oov_false_activations']}")
    print(f"  Silence False Activations: {dev_results['silence_false_activations']}")
    print(f"  P95 Latency:               {dev_results['p95_latency_ms']:.1f} ms")

    # 4. Gate D: Observed Test Reproduction (vad_test_corpus)
    print("\n" + "=" * 80)
    print("GATE D: OBSERVED TEST REPRODUCTION (vad_test_corpus, 60 sequences)")
    print("Label: OBSERVED_TEST_REPRODUCTION_NOT_INDEPENDENT_VALIDATION")
    print("=" * 80)
    test_results = replay_corpus_through_pipeline(TEST_MANIFEST_PATH, pipeline, chunk_samples=320)
    print(f"  Exact-Sequence Accuracy:   {test_results['exact_correct_sequences']} / {test_results['total_sequences']} ({test_results['exact_sequence_accuracy'] * 100:.2f}%)")
    print(f"  Command Precision:         {test_results['command_precision'] * 100:.2f}%")
    print(f"  Command Recall:            {test_results['command_recall'] * 100:.2f}%")
    print(f"  Command F1:                {test_results['command_f1']:.4f}")
    print(f"  Matched Commands:          {test_results['matched_commands']} / {test_results['total_target_events']}")
    print(f"  Missed Commands:           {test_results['missed_commands']}")
    print(f"  Substitutions:             {test_results['substitutions']}")
    print(f"  OOV False Activations:     {test_results['oov_false_activations']}")
    print(f"  Silence False Activations: {test_results['silence_false_activations']}")
    print(f"  Duplicate Triggers:        {test_results['duplicate_triggers']}")
    print(f"  Median Latency:            {test_results['median_latency_ms']:.1f} ms")
    print(f"  P95 Latency:               {test_results['p95_latency_ms']:.1f} ms")

    # Audit comparison against Phase 16 baseline
    print("\n--- Comparing Gate D with Phase 16 Baseline ---")
    p16_ref = {
        "exact_sequence_accuracy": 0.5167,
        "matched_commands": 157,
        "missed_commands": 33,
        "substitutions": 2,
        "oov_false_activations": 6,
        "silence_false_activations": 0,
        "duplicate_triggers": 0,
        "command_precision": 0.9515,
        "command_recall": 0.8177,
        "command_f1": 0.8796,
        "p95_latency_ms": 517.62
    }

    parity_ok = True
    for key, ref_val in p16_ref.items():
        curr_val = test_results[key]
        if isinstance(ref_val, float):
            diff = abs(curr_val - ref_val)
            match = (diff < 1e-3) or (key == "p95_latency_ms" and diff < 1.0)
        else:
            match = (curr_val == ref_val)
        status_str = "MATCH" if match else "DIFF"
        if not match:
            parity_ok = False
        print(f"  [{status_str}] {key:<28}: current={curr_val} vs ref={ref_val}")

    # Output full report JSON
    payload = {
        "metadata": {
            "phase": 19,
            "title": "Controlled Offline Profile B Validation",
            "date": "2026-10-24",
            "governance_labels": [
                "EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED",
                "NOT_VALIDATED_ON_NATURAL_SPEECH",
                "NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION",
                "NOT_VALIDATED_PENDING_DATA"
            ],
            "evaluation_label": "OBSERVED_TEST_REPRODUCTION_NOT_INDEPENDENT_VALIDATION",
            "artifacts_verified": {
                "e1_onnx": actual_onnx_hash,
                "vad_config": actual_vad_hash,
                "dev_manifest": actual_dev_hash,
                "test_manifest": actual_test_hash
            }
        },
        "gate_c_dev_replay": dev_results,
        "gate_d_test_reproduction": test_results,
        "gate_d_parity_with_phase16": {
            "status": "PASS" if parity_ok else "INVESTIGATE",
            "reference_metrics": p16_ref
        }
    }

    OUTPUT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Validation results saved to: {OUTPUT_REPORT_PATH.name}")

    if parity_ok:
        print("\n>>> REPRODUCTION SUCCESS: Integrated Profile B reproduced Phase 16 metrics exactly! <<<")
    else:
        print("\n>>> DISCREPANCY DETECTED: Review per-sequence records. <<<")


if __name__ == "__main__":
    main()
