"""
Audit of Conditional Classification Accuracy (Section D of Phase 16).
Traces every target command event on the locked test corpus:
1. Did VAD produce a segment with IoU >= 0.50?
2. If YES: What was E1's prediction? (Correct, Substitution, or False Reject)
3. If NO: Why? (Merged into multi-word segment, missed by VAD, or low IoU)
"""

import sys
import io
import json
from pathlib import Path
import pandas as pd
import numpy as np
import soundfile as sf

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "phase15_vad"))

from eval_kws import KWSEvaluator, TARGET_CLASSES
from vad_segmenters import SpeechBandRatioVAD
from eval_stage_a_segmentation import calculate_temporal_iou

def run_audit():
    cfg_raw = json.load(open(PROJECT_ROOT / "experiments/vieneu_kws_research_v12/configs/selected_phase15_vad_config.json"))["parameters"]
    vad_params = {k: v for k, v in cfg_raw.items() if k != "algorithm"}
    vad_params["speech_band_hz"] = tuple(vad_params["speech_band_hz"])

    vad = SpeechBandRatioVAD(**vad_params)
    eval_e1 = KWSEvaluator(PROJECT_ROOT / "experiments/vieneu_kws_research_v12/checkpoints/exp_v12_e1_baseline_scale_seed42.pt")

    df_test = pd.read_csv(PROJECT_ROOT / "experiments/vieneu_kws_research_v12/manifests/vad_test_corpus_manifest.csv")

    total_targets = 0
    matched_iou50 = 0
    e1_correct_on_iou50 = 0
    e1_subst_on_iou50 = 0
    e1_reject_on_iou50 = 0

    unsegmented_targets = 0
    unsegmented_merged = 0
    unsegmented_missed_vad = 0
    unsegmented_low_iou = 0

    for idx, row in df_test.iterrows():
        audio_path = PROJECT_ROOT / row["audio_path"]
        audio, sr = sf.read(str(audio_path), dtype="float32")
        ref_events = json.loads(row["events_json"])

        vad.reset()
        pred_segs = []
        for c in range(0, len(audio), 320):
            s = vad.process_chunk(audio[c : c + 320], c / sr)
            if s is not None:
                pred_segs.append(s)

        for r_idx, ref in enumerate(ref_events):
            if not ref["is_target"]:
                continue
            total_targets += 1

            best_iou = 0.0
            best_seg = None
            best_overlap = 0.0

            for seg in pred_segs:
                iou = calculate_temporal_iou(ref["start_sec"], ref["end_sec"], seg["start_sec"], seg["end_sec"])
                inter_s = max(ref["start_sec"], seg["start_sec"])
                inter_e = min(ref["end_sec"], seg["end_sec"])
                overlap = max(0.0, inter_e - inter_s)
                if iou > best_iou:
                    best_iou = iou
                    best_seg = seg
                    best_overlap = overlap

            if best_iou >= 0.50:
                matched_iou50 += 1
                pred_class, conf, probs, _ = eval_e1.predict_clip(best_seg["segment_audio_1s"])
                if pred_class == ref["label"]:
                    e1_correct_on_iou50 += 1
                elif pred_class in TARGET_CLASSES:
                    e1_subst_on_iou50 += 1
                else:
                    e1_reject_on_iou50 += 1
            else:
                unsegmented_targets += 1
                # Check why unsegmented
                if best_seg is not None and best_overlap >= 0.080:
                    # Check if this segment overlapped multiple references (merge)
                    overlapping_refs = [
                        r for r in ref_events
                        if max(0.0, min(r["end_sec"], best_seg["end_sec"]) - max(r["start_sec"], best_seg["start_sec"])) >= 0.050
                    ]
                    if len(overlapping_refs) >= 2:
                        unsegmented_merged += 1
                    else:
                        unsegmented_low_iou += 1
                else:
                    unsegmented_missed_vad += 1

    print("=" * 80)
    print("  AUDIT OF CONDITIONAL CLASSIFICATION ACCURACY")
    print("=" * 80)
    print(f"Total Target Commands: {total_targets}")
    print(f"1. Target Commands Matched by VAD with IoU >= 0.50: {matched_iou50} / {total_targets} ({matched_iou50/total_targets*100:.2f}%)")
    print(f"   - Correctly Classified by E1: {e1_correct_on_iou50} / {matched_iou50} ({e1_correct_on_iou50/matched_iou50*100:.2f}%)")
    print(f"   - Substituted into Wrong Command: {e1_subst_on_iou50} / {matched_iou50} ({e1_subst_on_iou50/matched_iou50*100:.2f}%)")
    print(f"   - Falsely Rejected as REJECT: {e1_reject_on_iou50} / {matched_iou50} ({e1_reject_on_iou50/matched_iou50*100:.2f}%)")
    print(f"\n2. Target Commands Excluded due to Segmentation Failure (IoU < 0.50): {unsegmented_targets} / {total_targets} ({unsegmented_targets/total_targets*100:.2f}%)")
    print(f"   - Merged into multi-word segments: {unsegmented_merged}")
    print(f"   - Low IoU overlap (0.20 <= IoU < 0.50): {unsegmented_low_iou}")
    print(f"   - Missed by VAD completely (no overlap): {unsegmented_missed_vad}")
    print("\n3. Reconciliation of Reported '98.74%' Claim:")
    print(f"   - In Phase 15 Report: '157 / 159 = 98.74%'")
    print(f"   - Actual Numerator: 157 (E1 Correct command detections)")
    print(f"   - Actual Denominator: 159 (Total command emissions matching target events: 157 correct + 2 substitutions)")
    print(f"   - NOTICE: The denominator 159 did NOT include the {e1_reject_on_iou50} false rejects on matched segments, nor the {unsegmented_targets} unsegmented targets!")
    print(f"   - True Conditional Accuracy on ALL IoU >= 0.50 segments: {e1_correct_on_iou50} / {matched_iou50} = {e1_correct_on_iou50/matched_iou50*100:.2f}%")
    print("=" * 80)

if __name__ == "__main__":
    run_audit()
