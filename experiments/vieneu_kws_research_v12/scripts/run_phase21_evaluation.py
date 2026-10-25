#!/usr/bin/env python3
"""
Phase 21 Step 6: End-to-End Evaluation of Frozen E1 on New Synthetic Corpora.
Repository: D:\trainidentificationaudio
Target: experiments/vieneu_kws_research_v12/scripts/run_phase21_evaluation.py

Evaluates:
  1. synthetic_eval_dev_v1 (20 sequences)
  2. synthetic_eval_test_v1 (40 sequences)

Strict Governance:
  - Model: Frozen E1 ONNX (vietnamese_kws_v12_e1.onnx)
  - Frontend: CanonicalLogMelFrontendNumPy (Slaney filterbank parity)
  - VAD: SpeechBandRatioVAD (selected_phase15_vad_config.json)
  - Decision Rule: theta_cmd = 0.20, REJECT participates in argmax
  - ZERO retraining, ZERO threshold tuning, ZERO word blacklisting.
"""

import sys
import io
import json
import hashlib
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import soundfile as sf
import onnxruntime as ort

# Ensure UTF-8 console output
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANIFESTS_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "manifests"
DEV_MANIFEST_CSV = MANIFESTS_DIR / "synthetic_eval_dev_manifest.csv"
TEST_MANIFEST_CSV = MANIFESTS_DIR / "synthetic_eval_test_manifest.csv"

E1_ONNX_PATH = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "onnx" / "vietnamese_kws_v12_e1.onnx"
EXPECTED_E1_ONNX_HASH = "20ecc2964d88d5a9d1d15b1c665e9697a32d702b1e3dd12ab1e5f829aff016b8"

VAD_CONFIG_PATH = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "configs" / "selected_phase15_vad_config.json"
EXPECTED_VAD_CONFIG_HASH = "af5c3891fe73f831bb2972f2ce4722b3cbb6b669ad306bd408b8e941ee7de9b3"

REPORT_OUT_JSON = (
    PROJECT_ROOT
    / "experiments"
    / "vieneu_kws_research_v12"
    / "reports"
    / "phase21_synthetic_evaluation_results.json"
)

# Insert scripts directory for frontend and VAD
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "phase15_vad"))

from audit_frontend_parity_phase20 import CanonicalLogMelFrontendNumPy
from vad_segmenters import SpeechBandRatioVAD

CLASSES = ["TRAI", "PHAI", "LEN", "XUONG", "REJECT"]
TARGET_CLASSES = ["TRAI", "PHAI", "LEN", "XUONG"]
DEFAULT_THRESHOLD = 0.20
DEV_VOICES = ["Anh Khôi", "Minh Quân"]
TEST_VOICES = ["Ái Hân", "Hữu Quân", "Xuân Tiên", "Mạnh Dũng"]


def compute_file_sha256(path: Path) -> str:
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


def evaluate_corpus_on_e1_pipeline(
    manifest_csv: Path,
    session: ort.InferenceSession,
    frontend: CanonicalLogMelFrontendNumPy,
    vad_params: Dict[str, Any],
    chunk_samples: int = 320
) -> Dict[str, Any]:
    """Runs end-to-end streaming evaluation using SpeechBandRatioVAD + E1 ONNX."""
    df = pd.read_csv(manifest_csv)
    split_name = df["corpus_split"].iloc[0]

    total_target_events = 0
    matched_commands = 0
    missed_commands = 0
    substitutions = 0
    oov_false_activations = 0
    silence_false_activations = 0
    duplicate_triggers = 0
    exact_correct_sequences = 0

    all_latencies_ms = []
    total_negative_seconds = 0.0
    total_oov_seconds = 0.0
    total_silence_seconds = 0.0

    # Categorized false activations
    fa_by_oov_type = {"phonetic_confuser": 0, "test_unseen_general": 0, "dev_general_oov": 0}
    # Performance by voice
    voice_stats: Dict[str, Dict[str, int]] = {}
    # Performance by gap
    gap_stats = {
        gap: {"target_events": 0, "matched": 0, "exact_seqs": 0, "total_seqs": 0, "fa_total": 0}
        for gap in [100, 150, 200, 250, 300, 400]
    }
    # Performance by condition
    cond_stats = {
        cond: {"target_events": 0, "matched": 0, "exact_seqs": 0, "total_seqs": 0, "fa_total": 0}
        for cond in ["clean", "snr20", "snr15", "snr10", "snr05"]
    }

    # Confusion Matrix: rows=ground truth, cols=predicted (0..4)
    # 0: TRAI, 1: PHAI, 2: LEN, 3: XUONG, 4: REJECT (Miss)
    conf_matrix = np.zeros((5, 5), dtype=int)

    per_sequence_records = []

    for idx, row in df.iterrows():
        seq_id = row["sequence_id"]
        wav_path = PROJECT_ROOT / row["audio_path"]
        ref_events = json.loads(row["events_json"])
        nominal_gap = int(row["nominal_gap_ms"])
        cond = row["condition"]
        total_duration = float(row["duration_seconds"])

        target_ground_truth = [e["label"] for e in ref_events if e["is_target"]]
        n_targets_in_seq = len(target_ground_truth)
        total_target_events += n_targets_in_seq

        # Track durations
        target_dur_sum = sum(e["duration_sec"] for e in ref_events if e["is_target"])
        oov_dur_sum = sum(e["duration_sec"] for e in ref_events if not e["is_target"])
        silence_dur = max(0.0, total_duration - target_dur_sum - oov_dur_sum)

        total_negative_seconds += (oov_dur_sum + silence_dur)
        total_oov_seconds += oov_dur_sum
        total_silence_seconds += silence_dur

        # Initialize fresh VAD for sequence
        vad = SpeechBandRatioVAD(**vad_params)
        audio, sr = sf.read(str(wav_path), dtype="float32")
        assert sr == 16000

        emitted_events = []

        # Stream chunk-by-chunk
        for c_start in range(0, len(audio), chunk_samples):
            chunk = audio[c_start : c_start + chunk_samples]
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
            chunk_time = c_start / float(sr)
            seg = vad.process_chunk(chunk, chunk_time)
            if seg is not None:
                audio_1s = seg["segment_audio_1s"]
                spec = frontend.compute(audio_1s)
                ort_out = session.run(None, {"log_mel": spec})
                probs = ort_out[1][0]
                top_idx = int(np.argmax(probs))

                # Canonical 5-class decision contract
                if top_idx < 4 and probs[top_idx] >= DEFAULT_THRESHOLD:
                    pred_cmd = TARGET_CLASSES[top_idx]
                    conf = float(probs[top_idx])
                    emitted_events.append({
                        "command": pred_cmd,
                        "confidence": round(conf, 4),
                        "start_sec": seg["start_sec"],
                        "end_sec": seg["end_sec"],
                        "duration_ms": round(seg["duration_sec"] * 1000.0, 1)
                    })

        # 360ms trailing silence flush
        trailing_chunk = np.zeros(chunk_samples, dtype=np.float32)
        n_trailing = int(round((0.360 * 16000) / chunk_samples))
        for _ in range(n_trailing):
            seg = vad.process_chunk(trailing_chunk, len(audio) / float(sr))
            if seg is not None:
                audio_1s = seg["segment_audio_1s"]
                spec = frontend.compute(audio_1s)
                ort_out = session.run(None, {"log_mel": spec})
                probs = ort_out[1][0]
                top_idx = int(np.argmax(probs))
                if top_idx < 4 and probs[top_idx] >= DEFAULT_THRESHOLD:
                    pred_cmd = TARGET_CLASSES[top_idx]
                    conf = float(probs[top_idx])
                    emitted_events.append({
                        "command": pred_cmd,
                        "confidence": round(conf, 4),
                        "start_sec": seg["start_sec"],
                        "end_sec": seg["end_sec"],
                        "duration_ms": round(seg["duration_sec"] * 1000.0, 1)
                    })

        emitted_commands = [ev["command"] for ev in emitted_events]
        is_exact = (emitted_commands == target_ground_truth)
        if is_exact:
            exact_correct_sequences += 1

        # Match emitted commands against reference events
        matched_target_refs = set()
        matched_pred_cmds = set()
        seq_fa_oov = 0
        seq_fa_silence = 0
        seq_duplicates = 0

        for p_idx, ev in enumerate(emitted_events):
            p_cls = ev["command"]
            p_s = ev["start_sec"]
            p_e = ev["end_sec"]

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

            # Overlap criteria: IoU >= 0.20 or overlap >= 80ms
            if best_r_idx >= 0 and (best_iou >= 0.20 or best_overlap >= 0.080):
                ref_match = ref_events[best_r_idx]
                v_name = ref_match.get("voice", "unknown")
                if v_name not in voice_stats:
                    voice_stats[v_name] = {"targets": 0, "matched": 0, "misses": 0, "fa": 0}

                if ref_match["is_target"]:
                    if best_r_idx not in matched_target_refs:
                        matched_target_refs.add(best_r_idx)
                        matched_pred_cmds.add(p_idx)
                        r_label = ref_match["label"]
                        r_idx_num = CLASSES.index(r_label)
                        p_idx_num = CLASSES.index(p_cls)

                        if p_cls == r_label:
                            matched_commands += 1
                            conf_matrix[r_idx_num, p_idx_num] += 1
                            voice_stats[v_name]["matched"] += 1
                            lat_ms = (p_e - ref_match["start_sec"]) * 1000.0
                            all_latencies_ms.append(lat_ms)
                        else:
                            substitutions += 1
                            conf_matrix[r_idx_num, p_idx_num] += 1
                    else:
                        # Duplicate trigger for already matched command
                        duplicate_triggers += 1
                        seq_duplicates += 1
                else:
                    # OOV event triggered a command
                    oov_false_activations += 1
                    seq_fa_oov += 1
                    voice_stats[v_name]["fa"] += 1
                    oov_t = ref_match.get("oov_type", "general_oov")
                    fa_by_oov_type[oov_t] = fa_by_oov_type.get(oov_t, 0) + 1
                    conf_matrix[4, CLASSES.index(p_cls)] += 1
            else:
                # Spurious activation during silence or noise
                silence_false_activations += 1
                seq_fa_silence += 1
                conf_matrix[4, CLASSES.index(p_cls)] += 1

        # Count missed commands
        for r_idx, r_evt in enumerate(ref_events):
            if r_evt["is_target"]:
                v_name = r_evt.get("voice", "unknown")
                if v_name not in voice_stats:
                    voice_stats[v_name] = {"targets": 0, "matched": 0, "misses": 0, "fa": 0}
                voice_stats[v_name]["targets"] += 1

                if r_idx not in matched_target_refs:
                    missed_commands += 1
                    voice_stats[v_name]["misses"] += 1
                    r_idx_num = CLASSES.index(r_evt["label"])
                    conf_matrix[r_idx_num, 4] += 1  # Miss placed into REJECT column

        # Gap and condition updates
        seq_fa_total = seq_fa_oov + seq_fa_silence
        if nominal_gap in gap_stats:
            gap_stats[nominal_gap]["total_seqs"] += 1
            gap_stats[nominal_gap]["target_events"] += n_targets_in_seq
            gap_stats[nominal_gap]["matched"] += len(matched_target_refs)
            gap_stats[nominal_gap]["exact_seqs"] += int(is_exact)
            gap_stats[nominal_gap]["fa_total"] += seq_fa_total

        if cond in cond_stats:
            cond_stats[cond]["total_seqs"] += 1
            cond_stats[cond]["target_events"] += n_targets_in_seq
            cond_stats[cond]["matched"] += len(matched_target_refs)
            cond_stats[cond]["exact_seqs"] += int(is_exact)
            cond_stats[cond]["fa_total"] += seq_fa_total

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
            "emitted_events": emitted_events
        })

    # Summary calculations
    total_emitted = matched_commands + substitutions + oov_false_activations + silence_false_activations
    precision = (matched_commands / total_emitted) if total_emitted > 0 else 0.0
    recall = (matched_commands / total_target_events) if total_target_events > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    exact_acc = exact_correct_sequences / len(df)

    p50_lat = float(np.percentile(all_latencies_ms, 50)) if all_latencies_ms else 0.0
    p95_lat = float(np.percentile(all_latencies_ms, 95)) if all_latencies_ms else 0.0

    neg_mins = total_negative_seconds / 60.0
    fa_per_neg_min = (oov_false_activations + silence_false_activations) / neg_mins if neg_mins > 0 else 0.0

    return {
        "corpus_split": split_name,
        "manifest_path": str(manifest_csv),
        "total_sequences": len(df),
        "exact_correct_sequences": exact_correct_sequences,
        "exact_sequence_accuracy": round(exact_acc, 4),
        "total_target_events": total_target_events,
        "matched_commands": matched_commands,
        "missed_commands": missed_commands,
        "substitutions": substitutions,
        "oov_false_activations": oov_false_activations,
        "oov_false_activations_by_type": fa_by_oov_type,
        "silence_false_activations": silence_false_activations,
        "duplicate_triggers": duplicate_triggers,
        "command_precision": round(precision, 4),
        "command_recall": round(recall, 4),
        "command_f1": round(f1, 4),
        "total_negative_minutes": round(neg_mins, 2),
        "false_activations_per_negative_minute": round(fa_per_neg_min, 2),
        "median_latency_ms": round(p50_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "confusion_matrix": conf_matrix.tolist(),
        "voice_breakdown": voice_stats,
        "gap_breakdown": gap_stats,
        "condition_breakdown": cond_stats,
        "per_sequence_records": per_sequence_records
    }


def main():
    print("=" * 80)
    print("PHASE 21: EVALUATION OF FROZEN E1 ON NEW SYNTHETIC CORPORA")
    print("=" * 80)

    # 1. Verify frozen artifact hashes
    actual_onnx_hash = compute_file_sha256(E1_ONNX_PATH)
    actual_vad_hash = compute_file_sha256(VAD_CONFIG_PATH)

    assert actual_onnx_hash == EXPECTED_E1_ONNX_HASH, f"E1 ONNX modified! {actual_onnx_hash}"
    assert actual_vad_hash == EXPECTED_VAD_CONFIG_HASH, f"VAD config modified! {actual_vad_hash}"
    print(f"[OK] Verified E1 ONNX SHA-256:       {actual_onnx_hash[:16]}...")
    print(f"[OK] Verified VAD Config SHA-256:    {actual_vad_hash[:16]}...")

    # Load VAD parameters
    with open(VAD_CONFIG_PATH, "r", encoding="utf-8") as f:
        vad_meta = json.load(f)
    vad_params = dict(vad_meta["parameters"])
    vad_params.pop("algorithm", None)
    if isinstance(vad_params.get("speech_band_hz"), list):
        vad_params["speech_band_hz"] = tuple(vad_params["speech_band_hz"])

    # 2. Instantiate Session and Frontend
    session = ort.InferenceSession(str(E1_ONNX_PATH), providers=["CPUExecutionProvider"])
    frontend = CanonicalLogMelFrontendNumPy()
    print("[OK] Initialized CanonicalLogMelFrontendNumPy and ONNX Session.")

    # 3. Evaluate Development Corpus (synthetic_eval_dev_v1, 20 seqs)
    print("\n" + "=" * 80)
    print("EVALUATING SYNTHETIC EVAL DEV CORPUS (synthetic_eval_dev_v1, 20 sequences)")
    print("=" * 80)
    dev_results = evaluate_corpus_on_e1_pipeline(DEV_MANIFEST_CSV, session, frontend, vad_params)
    print(f"  Exact-Sequence Accuracy:   {dev_results['exact_correct_sequences']} / {dev_results['total_sequences']} ({dev_results['exact_sequence_accuracy']*100:.2f}%)")
    print(f"  Command Precision:         {dev_results['command_precision']*100:.2f}%")
    print(f"  Command Recall:            {dev_results['command_recall']*100:.2f}%")
    print(f"  Command F1:                {dev_results['command_f1']:.4f}")
    print(f"  Matched / Missed / Subs:   {dev_results['matched_commands']} / {dev_results['missed_commands']} / {dev_results['substitutions']}")
    print(f"  OOV False Activations:     {dev_results['oov_false_activations']}")
    print(f"  Silence False Activations: {dev_results['silence_false_activations']}")
    print(f"  Duplicate Triggers:        {dev_results['duplicate_triggers']}")
    print(f"  P95 Latency:               {dev_results['p95_latency_ms']} ms")

    # 4. Evaluate Locked Test Corpus (synthetic_eval_test_v1, 40 seqs)
    print("\n" + "=" * 80)
    print("EVALUATING LOCKED SYNTHETIC TEST CORPUS (synthetic_eval_test_v1, 40 sequences)")
    print("=" * 80)
    test_results = evaluate_corpus_on_e1_pipeline(TEST_MANIFEST_CSV, session, frontend, vad_params)
    print(f"  Exact-Sequence Accuracy:   {test_results['exact_correct_sequences']} / {test_results['total_sequences']} ({test_results['exact_sequence_accuracy']*100:.2f}%)")
    print(f"  Command Precision:         {test_results['command_precision']*100:.2f}%")
    print(f"  Command Recall:            {test_results['command_recall']*100:.2f}%")
    print(f"  Command F1:                {test_results['command_f1']:.4f}")
    print(f"  Matched / Missed / Subs:   {test_results['matched_commands']} / {test_results['missed_commands']} / {test_results['substitutions']}")
    print(f"  OOV False Activations:     {test_results['oov_false_activations']} (Confusers: {test_results['oov_false_activations_by_type'].get('phonetic_confuser', 0)}, General: {test_results['oov_false_activations_by_type'].get('test_unseen_general', 0)})")
    print(f"  Silence False Activations: {test_results['silence_false_activations']}")
    print(f"  Duplicate Triggers:        {test_results['duplicate_triggers']}")
    print(f"  False Act / Neg Minute:    {test_results['false_activations_per_negative_minute']}")
    print(f"  Median / P95 Latency:      {test_results['median_latency_ms']} ms / {test_results['p95_latency_ms']} ms")

    # 5. Save Output JSON
    full_report = {
        "metadata": {
            "timestamp": "2026-10-24T20:20:00+07:00",
            "phase": "Phase 21 Synthetic Evaluation",
            "e1_onnx_sha256": actual_onnx_hash,
            "vad_config_sha256": actual_vad_hash,
            "generator_evaluated": "VieNeu-TTS v3 Nano",
            "unseen_test_voices": TEST_VOICES,
            "threshold_theta_cmd": DEFAULT_THRESHOLD
        },
        "dev_corpus_results": dev_results,
        "test_corpus_results": test_results
    }

    REPORT_OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Wrote Phase 21 evaluation results to {REPORT_OUT_JSON}")


if __name__ == "__main__":
    main()
