#!/usr/bin/env python3
"""
Phase 28-31 Stage A: Phase 27 Event-Ledger and Metric Audit
Repository: D:\\trainidentificationaudio
Target: experiments/vieneu_kws_research_v12/scripts/audit_phase27_stage_a.py
"""

import sys
import io
import json
import numpy as np
import pandas as pd
from pathlib import Path
import soundfile as sf
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANIFESTS_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "manifests"
REPORTS_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "reports"
CANDIDATE_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "candidates" / "phase26_candidate_d0_h1_seed44"

sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "phase15_vad"))
from train_phase25_factorial import CorrectedVietnameseKWSNet, LogMelFrontend, CLASSES, CLASS_TO_IDX
from vad_segmenters import SpeechBandRatioVAD

ISOLATED_MANIFEST = MANIFESTS_DIR / "edge_tts_eval_isolated_test_manifest.csv"
STREAMING_MANIFEST = MANIFESTS_DIR / "edge_tts_eval_streaming_manifest.csv"
STREAMING_EVENTS = MANIFESTS_DIR / "edge_tts_eval_streaming_events.json"
VAD_CONFIG = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "configs" / "selected_phase15_vad_config.json"

MODELS = {
    "Candidate_D0_H1_seed44": CANDIDATE_DIR / "candidate_checkpoint.pt",
    "Baseline_E1_seed42": PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "checkpoints" / "exp_v12_e1_baseline_scale_seed42.pt",
    "Baseline_v11_seed42": PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4" / "bundle" / "checkpoint" / "variant_v11_hard_negative_weighted_seed42.pt"
}


def load_model(ckpt_path: Path, device: torch.device):
    model = CorrectedVietnameseKWSNet(num_classes=5, dropout_p=0.30).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def main():
    print("=" * 80)
    print("PHASE 28-31 STAGE A: PHASE 27 METRIC & EVENT LEDGER AUDIT")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frontend = LogMelFrontend().to(device)
    frontend.eval()

    df_iso = pd.read_csv(ISOLATED_MANIFEST)
    df_stream = pd.read_csv(STREAMING_MANIFEST)
    with open(STREAMING_EVENTS, "r", encoding="utf-8") as f:
        gt_events_list = json.load(f)

    # Load canonical SpeechBandRatioVAD configuration
    with open(VAD_CONFIG, "r", encoding="utf-8") as f:
        vad_cfg = json.load(f)
    print(f"Loaded canonical VAD config: {VAD_CONFIG.name} (algorithm: {vad_cfg.get('selected_algorithm')})")

    audit_results = {}

    for name, path in MODELS.items():
        print(f"\n--- Auditing {name} ---")
        model = load_model(path, device)

        # 1. Audit Isolated Metrics
        preds = []
        targets = []
        for idx, row in df_iso.iterrows():
            audio, _ = sf.read(str(PROJECT_ROOT / row["audio_path"]), dtype="float32")
            if len(audio) < 16000:
                audio = np.pad(audio, (0, 16000 - len(audio)), mode="constant")
            elif len(audio) > 16000:
                audio = audio[:16000]

            with torch.no_grad():
                spec = frontend(torch.from_numpy(audio).unsqueeze(0).to(device))
                logits = model(spec)
                probs = F.softmax(logits, dim=-1)[0]
                target_probs = probs[:4]
                max_p, max_idx = torch.max(target_probs, dim=-1)
                pred = int(max_idx.item()) if max_p.item() >= 0.20 else 4

            preds.append(pred)
            targets.append(CLASS_TO_IDX[row["label"]])

        y_true = np.array(targets)
        y_pred = np.array(preds)
        p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2, 3, 4], average=None, zero_division=0)
        macro_f1 = float(np.mean(f1[:4]))
        acc = float(np.mean(y_true == y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])

        oov_mask = (y_true == 4)
        oov_fa = int(np.sum(y_pred[oov_mask] != 4))
        oov_total = int(np.sum(oov_mask))

        phonetic_mask = (df_iso["subgroup"] == "test_oov_phonetic").values
        phonetic_fa = int(np.sum(y_pred[phonetic_mask] != 4))
        phonetic_total = int(np.sum(phonetic_mask))

        general_mask = (df_iso["subgroup"] == "test_oov_general").values
        general_fa = int(np.sum(y_pred[general_mask] != 4))
        general_total = int(np.sum(general_mask))

        print(f"  [Isolated Verified] Macro-F1: {macro_f1*100:.2f}% | Acc: {acc*100:.2f}% | OOV FA: {oov_fa}/{oov_total} ({oov_fa/oov_total*100:.2f}%)")
        print(f"                      Phonetic OOV FA: {phonetic_fa}/{phonetic_total} ({phonetic_fa/phonetic_total*100:.2f}%) | General OOV FA: {general_fa}/{general_total} ({general_fa/general_total*100:.2f}%)")

        # 2. Audit Streaming Metrics under Canonical SpeechBandRatioVAD Segmenter
        # Group GT events by sequence
        events_by_seq = {}
        for ev in gt_events_list:
            events_by_seq.setdefault(ev["sequence_id"], []).append(ev)

        segmented_stream_exact = 0
        segmented_stream_tp = 0
        segmented_stream_gt_cmds = 0
        segmented_stream_oov_fa = 0
        segmented_stream_silence_fa = 0
        segmented_stream_detected = 0

        total_oov_time_s = 0.0
        total_silence_time_s = 0.0

        for idx, row in df_stream.iterrows():
            seq_id = row["sequence_id"]
            wav_path = PROJECT_ROOT / row["audio_path"]
            audio, _ = sf.read(str(wav_path), dtype="float32")
            seq_len = len(audio)

            gt_events = events_by_seq.get(seq_id, [])
            gt_cmds = [ev for ev in gt_events if ev["label"] != "REJECT"]
            gt_oovs = [ev for ev in gt_events if ev["label"] == "REJECT"]

            segmented_stream_gt_cmds += len(gt_cmds)
            total_oov_time_s += sum(ev["duration"] for ev in gt_oovs)
            total_silence_time_s += max(0.0, (seq_len / 16000.0) - sum(ev["duration"] for ev in gt_events))

            # Instantiate SpeechBandRatioVAD for this sequence
            vad_params = vad_cfg.get("parameters", {})
            vad = SpeechBandRatioVAD(
                sample_rate=16000,
                energy_threshold_db=vad_params.get("energy_threshold_db", -46.0),
                min_speech_band_ratio=vad_params.get("min_speech_band_ratio", 0.32),
                hangover_ms=vad_params.get("hangover_ms", 160),
                preroll_ms=vad_params.get("preroll_ms", 150),
                min_speech_duration_ms=vad_params.get("min_speech_duration_ms", 100),
                max_segment_duration_ms=vad_params.get("max_segment_duration_ms", 1200)
            )

            # Process 20ms chunks (320 samples)
            chunk_size = 320
            detected_segments = []

            for c_start in range(0, seq_len, chunk_size):
                chunk = audio[c_start : c_start + chunk_size]
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode="constant")
                c_time = c_start / 16000.0
                event = vad.process_chunk(chunk, c_time)
                if event is not None:
                    # An utterance segment was finalized by VAD!
                    seg_1s = event["segment_audio_1s"]
                    with torch.no_grad():
                        spec = frontend(torch.from_numpy(seg_1s).unsqueeze(0).to(device))
                        logits = model(spec)
                        probs = F.softmax(logits, dim=-1)[0]
                        target_probs = probs[:4]
                        max_p, max_idx = torch.max(target_probs, dim=-1)
                        if max_p.item() >= 0.20:
                            detected_segments.append({
                                "label": CLASSES[max_idx.item()],
                                "start_time": event["start_sec"],
                                "end_time": event["end_sec"],
                                "conf": float(max_p.item())
                            })

            segmented_stream_detected += len(detected_segments)

            # Sequence match
            detected_seq = [d["label"] for d in detected_segments]
            expected_seq = eval(row["expected_command_sequence"]) if isinstance(row["expected_command_sequence"], str) else row["expected_command_sequence"]
            if detected_seq == expected_seq:
                segmented_stream_exact += 1

            # Match events
            matched_gt = set()
            for d in detected_segments:
                dt = d["start_time"]
                dlbl = d["label"]
                match_found = False
                for i, gt in enumerate(gt_cmds):
                    if i not in matched_gt and gt["label"] == dlbl:
                        if (gt["start_time"] - 0.30) <= dt <= (gt["end_time"] + 0.50):
                            match_found = True
                            matched_gt.add(i)
                            segmented_stream_tp += 1
                            break

                if not match_found:
                    in_oov = any(gt["start_time"] - 0.20 <= dt <= gt["end_time"] + 0.30 for gt in gt_oovs)
                    if in_oov:
                        segmented_stream_oov_fa += 1
                    else:
                        segmented_stream_silence_fa += 1

        oov_time_min = total_oov_time_s / 60.0
        silence_time_min = total_silence_time_s / 60.0

        vad_recall = segmented_stream_tp / segmented_stream_gt_cmds if segmented_stream_gt_cmds > 0 else 0.0
        vad_precision = segmented_stream_tp / segmented_stream_detected if segmented_stream_detected > 0 else 0.0
        vad_f1 = (2 * vad_precision * vad_recall) / (vad_precision + vad_recall) if (vad_precision + vad_recall) > 0 else 0.0

        oov_fa_rate_min = segmented_stream_oov_fa / oov_time_min if oov_time_min > 0 else 0.0
        silence_fa_rate_min = segmented_stream_silence_fa / silence_time_min if silence_time_min > 0 else 0.0

        print(f"  [Streaming with SpeechBandRatioVAD]")
        print(f"      Exact Sequence Accuracy: {segmented_stream_exact}/{len(df_stream)} ({segmented_stream_exact/len(df_stream)*100:.2f}%)")
        print(f"      Event Precision: {vad_precision*100:.2f}% | Event Recall: {vad_recall*100:.2f}% | Event F1: {vad_f1*100:.2f}%")
        print(f"      Total Detections: {segmented_stream_detected} (TP: {segmented_stream_tp}, OOV FA: {segmented_stream_oov_fa}, Silence FA: {segmented_stream_silence_fa})")
        print(f"      OOV FA/min: {oov_fa_rate_min:.2f} (denom: {oov_time_min:.2f} min) | Silence FA/min: {silence_fa_rate_min:.2f} (denom: {silence_time_min:.2f} min)")

        audit_results[name] = {
            "isolated": {
                "macro_f1": macro_f1,
                "accuracy": acc,
                "confusion_matrix": cm.tolist(),
                "oov_total": oov_total,
                "oov_fa": oov_fa,
                "oov_fa_rate": oov_fa / oov_total,
                "phonetic_oov_fa": phonetic_fa,
                "phonetic_oov_total": phonetic_total,
                "phonetic_oov_fa_rate": phonetic_fa / phonetic_total,
                "general_oov_fa": general_fa,
                "general_oov_total": general_total,
                "general_oov_fa_rate": general_fa / general_total,
                "per_class": {
                    CLASSES[i]: {
                        "precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i]), "support": int(s[i])
                    } for i in range(5)
                }
            },
            "streaming_canonical_vad": {
                "exact_sequence_accuracy": segmented_stream_exact / len(df_stream),
                "exact_sequence_matches": segmented_stream_exact,
                "event_precision": vad_precision,
                "event_recall": vad_recall,
                "event_f1": vad_f1,
                "total_gt_commands": segmented_stream_gt_cmds,
                "total_detected_commands": segmented_stream_detected,
                "true_positives": segmented_stream_tp,
                "oov_false_activations": segmented_stream_oov_fa,
                "silence_noise_false_activations": segmented_stream_silence_fa,
                "fa_per_minute_oov": oov_fa_rate_min,
                "fa_per_minute_silence": silence_fa_rate_min,
                "oov_duration_min": oov_time_min,
                "silence_duration_min": silence_time_min
            }
        }

    # Save Stage A audit results
    out_audit = REPORTS_DIR / "phase28_stage_a_metric_audit.json"
    with open(out_audit, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)

    print(f"\n[OK] Stage A audit complete. Saved to {out_audit.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
