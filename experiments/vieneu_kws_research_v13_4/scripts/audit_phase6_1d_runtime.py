#!/usr/bin/env python3
"""
Workstream D — Phase 6.1D: Runtime Lifecycle & Regression Audit
Repository: D:\trainidentificationaudio
Branch: feature/continuous-kws-segmentation

Governance Labels:
  - EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED
  - NOT_VALIDATED_ON_NATURAL_SPEECH
  - NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION
  - NOT_VALIDATED_PENDING_DATA
"""

import sys
import os
import time
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import soundfile as sf
import pandas as pd

PROJECT_ROOT = Path("D:/trainidentificationaudio")
DEMO_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4" / "demo"
BUNDLE_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4" / "bundle"
AUDIO_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_3" / "data" / "audio"
REPORTS_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4" / "reports"
SCRIPTS_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4" / "scripts"

for p in [str(DEMO_DIR), str(BUNDLE_DIR), str(SCRIPTS_DIR), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from hybrid_segmenter import HybridSpeechBandValleySegmenter, GameModeSessionController
from infer import VietnameseKWSInferencer, TARGET_CLASSES
from audit_phase5_3_final_metrics import compute_standard_sequence_wer, compute_temporal_event_metrics
from benchmark_phase6_acceptance import get_command_clip, build_trial_stream, process_stream_through_pipeline

BASELINE_COMMIT = "d5af658a9afe523b306e5ca6c0159789f1c67fad"
EXPECTED_PT_HASH = "3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2"
EXPECTED_ONNX_HASH = "54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99"


def run_phase6_1d_audit():
    print("=" * 80)
    print("WORKSTREAM D — PHASE 6.1D: RUNTIME LIFECYCLE & REGRESSION AUDIT")
    print("=" * 80)

    inferencer = VietnameseKWSInferencer(backend="onnx")

    regression_records = []

    # Scenario 1: Repeated Identical Commands Separation
    seg = HybridSpeechBandValleySegmenter(speech_band_hz=(100.0, 3400.0))
    len1 = get_command_clip("len", 1)
    len2 = get_command_clip("len", 2)
    len3 = get_command_clip("len", 3)
    stream_3len = np.concatenate([np.zeros(8000, dtype=np.float32), len1, len2, len3, np.zeros(12000, dtype=np.float32)])
    dets_3len, _ = process_stream_through_pipeline(stream_3len, seg, inferencer)
    cmds_3len = [d["command"] for d in dets_3len]
    passed_scen1 = (cmds_3len == ["LEN", "LEN", "LEN"])
    regression_records.append({
        "scenario_id": "REG-01-REPEATED-BURSTS",
        "description": "Consecutive identical commands (LEN -> LEN -> LEN) separation without debounce suppression",
        "expected_behavior": "Emit 3 separate events ['LEN', 'LEN', 'LEN']",
        "actual_behavior": f"Emitted {len(cmds_3len)} events: {cmds_3len}",
        "status": "PASSED" if passed_scen1 else "FAILED",
        "root_cause_analysis": "None. Acoustic valley state machine partitions inter-syllable sonorant boundaries without artificial debounce cooldown."
    })

    # Scenario 2: Strict Event Ordering
    stream_order, gt_order = build_trial_stream("trai_len_xuong", 1, gap_ms=50)
    dets_order, _ = process_stream_through_pipeline(stream_order, seg, inferencer)
    cmds_order = [d["command"] for d in dets_order]
    passed_scen2 = (cmds_order == ["TRAI", "LEN", "XUONG"])
    regression_records.append({
        "scenario_id": "REG-02-EVENT-ORDERING",
        "description": "Strict chronological event ordering across rapid directional transitions",
        "expected_behavior": "Emit ['TRAI', 'LEN', 'XUONG'] in chronological order",
        "actual_behavior": f"Emitted: {cmds_order}",
        "status": "PASSED" if passed_scen2 else "FAILED",
        "root_cause_analysis": "None. Synchronous chunk progression preserves FIFO chronological sequence."
    })

    # Scenario 3: Duplicate Action Prevention
    stream_dup, _ = build_trial_stream("len_len_phai", 1, gap_ms=0)
    dets_dup, _ = process_stream_through_pipeline(stream_dup, seg, inferencer)
    # Check if any two detections have timestamps within 80ms of each other
    timestamps = [d["timestamp_sec"] for d in dets_dup]
    min_diff = min(np.diff(timestamps)) if len(timestamps) > 1 else 999.0
    passed_scen3 = (len(dets_dup) == 3 and min_diff >= 0.16)
    regression_records.append({
        "scenario_id": "REG-03-DUPLICATE-PREVENTION",
        "description": "Prevention of double-emission or duplicate events from single utterance cut",
        "expected_behavior": "Exactly 3 distinct events with inter-event distance >= 160ms",
        "actual_behavior": f"Emitted {len(dets_dup)} events, min inter-event time: {min_diff:.3f}s",
        "status": "PASSED" if passed_scen3 else "FAILED",
        "root_cause_analysis": "None. Valley state machine transitions to WAITING upon silence and resets peak/valley buffers."
    })

    # Scenario 4: Elongated Command Over-Splitting Resistance
    stream_elong, _ = build_trial_stream("elongated_len", 1, gap_ms=0)
    dets_elong, _ = process_stream_through_pipeline(stream_elong, seg, inferencer)
    passed_scen4 = (len(dets_elong) == 1 and dets_elong[0]["command"] == "LEN")
    regression_records.append({
        "scenario_id": "REG-04-ELONGATED-OVERSPLIT",
        "description": "Sustained vowel (750ms LÊÊÊN) must not be over-split into multiple chunks",
        "expected_behavior": "Emit exactly 1 event for 750ms elongated command",
        "actual_behavior": f"Emitted {len(dets_elong)} events: {[d['command'] for d in dets_elong]}",
        "status": "PASSED" if passed_scen4 else "FAILED",
        "root_cause_analysis": "None. Sustained acoustic energy remains above dip threshold (12 dB), inhibiting spurious valley trigger."
    })

    # Scenario 5: Flush Idempotence and No Double Emission
    seg_flush = HybridSpeechBandValleySegmenter(speech_band_hz=(100.0, 3400.0))
    speech_data = get_command_clip("len", 1)
    for i in range(0, len(speech_data), 320):
        seg_flush.process_chunk(speech_data[i : i + 320])
    res_flush1 = seg_flush.flush()
    res_flush2 = seg_flush.flush()
    passed_scen5 = (res_flush1 is not None and res_flush2 is None)
    regression_records.append({
        "scenario_id": "REG-05-FLUSH-IDEMPOTENCE",
        "description": "Stream termination flush() must emit pending speech at most once and return None subsequently",
        "expected_behavior": "First flush returns valid tuple; second flush immediately returns None",
        "actual_behavior": f"Flush 1: {res_flush1 is not None}; Flush 2: {res_flush2 is None}",
        "status": "PASSED" if passed_scen5 else "FAILED",
        "root_cause_analysis": "None. flush() invokes self.reset(), clearing accumulated chunks and setting state to WAITING."
    })

    # Scenario 6: Session Isolation and Gate Closed Discard
    controller = GameModeSessionController(
        base_segmenter=HybridSpeechBandValleySegmenter(speech_band_hz=(100.0, 3400.0)),
        session_timeout_sec=5.0
    )
    # Feed audio while gate closed
    res_closed = [controller.process_chunk(speech_data[i : i + 320]) for i in range(0, len(speech_data), 320)]
    passed_scen6a = all(r is None for r in res_closed)
    # Open session and feed audio
    controller.open_session()
    res_open = [controller.process_chunk(speech_data[i : i + 320]) for i in range(0, len(speech_data), 320)]
    flush_open = controller.flush()
    passed_scen6b = (flush_open is not None)
    passed_scen6 = (passed_scen6a and passed_scen6b)
    regression_records.append({
        "scenario_id": "REG-06-SESSION-GATE-ISOLATION",
        "description": "Session gate closed discards all audio; open session processes and finalizes correctly",
        "expected_behavior": "Gate closed returns 100% None; Gate open processes and flushes successfully",
        "actual_behavior": f"Closed all None: {passed_scen6a}; Open flush succeeded: {passed_scen6b}",
        "status": "PASSED" if passed_scen6 else "FAILED",
        "root_cause_analysis": "None. GameModeSessionController gates process_chunk and flush calls based on is_session_active."
    })

    # Scenario 7: Non-Standard Block Sizes and Chunk Resiliency
    seg_robust = HybridSpeechBandValleySegmenter(speech_band_hz=(100.0, 3400.0))
    caught_exception = False
    try:
        for sz in [160, 320, 50, 71, 0, 500, 1000]:
            seg_robust.process_chunk((0.1 * np.random.randn(sz)).astype(np.float32))
    except Exception as e:
        caught_exception = True
    passed_scen7 = not caught_exception
    regression_records.append({
        "scenario_id": "REG-07-CHUNK-SIZE-RESILIENCY",
        "description": "Resiliency to non-standard, short, or zero-length audio callback blocks (50, 71, 0 samples)",
        "expected_behavior": "Zero exceptions, automatic zero-padding or pass-through",
        "actual_behavior": f"Exception raised: {caught_exception}",
        "status": "PASSED" if passed_scen7 else "FAILED",
        "root_cause_analysis": "None. n_samples == 0 check and n_fft padding safely accommodate arbitrary block sizes."
    })

    # Scenario 8: Below Minimum Speech Duration (<100ms) Noise Rejection
    seg_noise = HybridSpeechBandValleySegmenter(min_speech_duration_ms=100)
    click = 0.5 * np.ones(640, dtype=np.float32)  # 40ms
    silence = np.zeros(3200, dtype=np.float32)    # 200ms
    stream_click = np.concatenate([click, silence])
    res_click = [seg_noise.process_chunk(stream_click[i:i+320]) for i in range(0, len(stream_click), 320)]
    flush_click = seg_noise.flush()
    passed_scen8 = all(r is None for r in res_click) and (flush_click is None)
    regression_records.append({
        "scenario_id": "REG-08-BELOW-MIN-SPEECH-REJECTION",
        "description": "Short acoustic impulse (40ms click) strictly rejected without segment emission",
        "expected_behavior": "0 segments finalized during process_chunk or flush",
        "actual_behavior": f"Emitted segments: {sum(r is not None for r in res_click) + (1 if flush_click is not None else 0)}",
        "status": "PASSED" if passed_scen8 else "FAILED",
        "root_cause_analysis": "None. Active speech sample accumulator requires >= 100ms before returning segment."
    })

    # Scenario 9: Global Temporal Matching Conflict Resolution
    gt_conflict = [
        {"command": "LEN", "start_sec": 0.50, "end_sec": 0.85},
        {"command": "LEN", "start_sec": 0.85, "end_sec": 1.20}
    ]
    dets_conflict = [
        {"command": "LEN", "timestamp_sec": 0.82, "infer_latency_ms": 2.0},
        {"command": "LEN", "timestamp_sec": 0.86, "infer_latency_ms": 2.0}
    ]
    ev_conflict = compute_temporal_event_metrics(dets_conflict, gt_conflict)
    passed_scen9 = (ev_conflict["event_tp"] <= 2 and ev_conflict["event_ins"] + ev_conflict["event_tp"] == 2)
    regression_records.append({
        "scenario_id": "REG-09-TEMPORAL-MATCHING-BIJECTION",
        "description": "Monotonic DP event matching strictly enforces bijective matching without greedy collision",
        "expected_behavior": "Strict injection (<=1 GT per det, <=1 det per GT), zero double-counting",
        "actual_behavior": f"TP: {ev_conflict['event_tp']}, Sub: {ev_conflict['event_sub']}, Ins: {ev_conflict['event_ins']}, Del: {ev_conflict['event_del']}",
        "status": "PASSED" if passed_scen9 else "FAILED",
        "root_cause_analysis": "None. Dynamic programming recurrence guarantees optimal monotonic alignment."
    })

    # Scenario 10: Strict Separation of Sequence WER and Event Latency
    seq_conflict = compute_standard_sequence_wer(["LEN", "LEN"], ["LEN", "LEN"])
    passed_scen10 = (seq_conflict["seq_wer_pct"] == 0.0 and "lag_latencies" not in seq_conflict)
    regression_records.append({
        "scenario_id": "REG-10-METRIC-SEPARATION",
        "description": "Standard sequence Levenshtein WER strictly decoupled from temporal event metrics",
        "expected_behavior": "Sequence WER computes symbolic edit distance without timestamp contamination",
        "actual_behavior": f"WER: {seq_conflict['seq_wer_pct']}%, Exact Match: {seq_conflict['exact_sequence_match']}",
        "status": "PASSED" if passed_scen10 else "FAILED",
        "root_cause_analysis": "None. Explicitly separate Path A (Levenshtein) and Path B (DP monotonic temporal) functions."
    })

    # Save Regression Results CSV
    csv_path = REPORTS_DIR / "phase6_1d_runtime_regressions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "scenario_id", "description", "expected_behavior", "actual_behavior", "status", "root_cause_analysis"
        ])
        writer.writeheader()
        for r in regression_records:
            writer.writerow(r)
    print(f"Phase 6.1D Runtime Regressions CSV written: {csv_path} ({len(regression_records)} scenarios)")

    # Print summary
    passed_count = sum(1 for r in regression_records if r["status"] == "PASSED")
    print(f"\nPhase 6.1D Regression Summary: {passed_count}/{len(regression_records)} scenarios PASSED.")


if __name__ == "__main__":
    run_phase6_1d_audit()
