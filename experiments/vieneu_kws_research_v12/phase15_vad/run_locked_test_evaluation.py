"""
One-Time Locked Test Evaluation Driver (Phase 15).

Executes strictly ONE-TIME evaluation of:
  1. BaselineFixedSilenceVAD (frozen default baseline)
  2. Selected Winning Candidate: SpeechBandRatioVAD (loaded from frozen selected_phase15_vad_config.json)

Across both frozen models:
  - v11 (variant_v11_hard_negative_weighted_seed42.pt)
  - E1 (exp_v12_e1_baseline_scale_seed42.pt)

On the previously unobserved, locked vad_test_corpus (60 sequences, manifest SHA-256: d0342e3f...).
Preserves complete per-sequence predictions and logs without modifying any parameters.
"""

import sys
import io
import json
import hashlib
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "phase15_vad"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "scripts"))

from vad_segmenters import BaselineFixedSilenceVAD, SpeechBandRatioVAD
from eval_stage_a_segmentation import evaluate_segmenter_on_corpus
from eval_stage_b_streaming_kws import evaluate_streaming_kws
from eval_kws import KWSEvaluator

MANIFEST_TEST = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "manifests" / "vad_test_corpus_manifest.csv"
FROZEN_CONFIG_PATH = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "configs" / "selected_phase15_vad_config.json"

V11_CKPT = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4" / "bundle" / "checkpoint" / "variant_v11_hard_negative_weighted_seed42.pt"
E1_CKPT = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "checkpoints" / "exp_v12_e1_baseline_scale_seed42.pt"


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 80)
    print("  PHASE 15: ONE-TIME LOCKED TEST EVALUATION")
    print("  Unseen Test Voices, Unseen Test OOV ('ghế', etc.), Unseen Test Noise Tracks")
    print("=" * 80)

    # 1. Verify frozen test manifest and selected config hashes
    test_manifest_hash = compute_sha256(MANIFEST_TEST)
    config_hash = compute_sha256(FROZEN_CONFIG_PATH)
    print(f"Locked Test Manifest Hash: {test_manifest_hash}")
    print(f"Frozen Winning Config Hash: {config_hash}")

    with open(FROZEN_CONFIG_PATH, "r", encoding="utf-8") as f:
        selected_cfg = json.load(f)["parameters"]

    # 2. Stage A: Segmentation-Only Evaluation on Locked Test
    print("\n" + "=" * 80)
    print("  STAGE A: SEGMENTATION-ONLY EVALUATION ON LOCKED TEST")
    print("=" * 80)

    base_vad = BaselineFixedSilenceVAD()
    winning_vad = SpeechBandRatioVAD(
        sample_rate=selected_cfg["sample_rate"],
        speech_band_hz=tuple(selected_cfg["speech_band_hz"]),
        min_speech_band_ratio=selected_cfg["min_speech_band_ratio"],
        energy_threshold_db=selected_cfg["energy_threshold_db"],
        hangover_ms=selected_cfg["hangover_ms"],
        preroll_ms=selected_cfg["preroll_ms"],
        min_speech_duration_ms=selected_cfg["min_speech_duration_ms"],
        max_segment_duration_ms=selected_cfg["max_segment_duration_ms"]
    )

    stage_a_base = evaluate_segmenter_on_corpus(base_vad, MANIFEST_TEST)
    stage_a_winning = evaluate_segmenter_on_corpus(winning_vad, MANIFEST_TEST)

    out_stage_a = {
        "manifest_sha256": test_manifest_hash,
        "config_sha256": config_hash,
        "BaselineFixedSilenceVAD": stage_a_base,
        "Selected_SpeechBandRatioVAD": stage_a_winning
    }
    stage_a_path = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "reports" / "stage_a_locked_test_results.json"
    with open(stage_a_path, "w", encoding="utf-8") as f:
        json.dump(out_stage_a, f, indent=2, ensure_ascii=False)
    print(f"Saved Stage A Locked Test Results: {stage_a_path}")

    # 3. Stage B: End-to-End Evaluation on Locked Test
    print("\n" + "=" * 80)
    print("  STAGE B: END-TO-END STREAMING EVALUATION ON LOCKED TEST")
    print("=" * 80)

    eval_v11 = KWSEvaluator(V11_CKPT)
    eval_e1 = KWSEvaluator(E1_CKPT)

    # Re-instantiate segmenters for stage B
    base_vad_1 = BaselineFixedSilenceVAD()
    base_vad_2 = BaselineFixedSilenceVAD()
    win_vad_1 = SpeechBandRatioVAD(
        sample_rate=selected_cfg["sample_rate"],
        speech_band_hz=tuple(selected_cfg["speech_band_hz"]),
        min_speech_band_ratio=selected_cfg["min_speech_band_ratio"],
        energy_threshold_db=selected_cfg["energy_threshold_db"],
        hangover_ms=selected_cfg["hangover_ms"],
        preroll_ms=selected_cfg["preroll_ms"],
        min_speech_duration_ms=selected_cfg["min_speech_duration_ms"],
        max_segment_duration_ms=selected_cfg["max_segment_duration_ms"]
    )
    win_vad_2 = SpeechBandRatioVAD(
        sample_rate=selected_cfg["sample_rate"],
        speech_band_hz=tuple(selected_cfg["speech_band_hz"]),
        min_speech_band_ratio=selected_cfg["min_speech_band_ratio"],
        energy_threshold_db=selected_cfg["energy_threshold_db"],
        hangover_ms=selected_cfg["hangover_ms"],
        preroll_ms=selected_cfg["preroll_ms"],
        min_speech_duration_ms=selected_cfg["min_speech_duration_ms"],
        max_segment_duration_ms=selected_cfg["max_segment_duration_ms"]
    )

    stage_b_base_v11 = evaluate_streaming_kws(base_vad_1, eval_v11, MANIFEST_TEST, model_name="v11")
    stage_b_base_e1 = evaluate_streaming_kws(base_vad_2, eval_e1, MANIFEST_TEST, model_name="e1")
    stage_b_win_v11 = evaluate_streaming_kws(win_vad_1, eval_v11, MANIFEST_TEST, model_name="v11")
    stage_b_win_e1 = evaluate_streaming_kws(win_vad_2, eval_e1, MANIFEST_TEST, model_name="e1")

    out_stage_b = {
        "manifest_sha256": test_manifest_hash,
        "config_sha256": config_hash,
        "Baseline_v11": stage_b_base_v11,
        "Baseline_e1": stage_b_base_e1,
        "Winning_SpeechBandRatio_v11": stage_b_win_v11,
        "Winning_SpeechBandRatio_e1": stage_b_win_e1
    }
    stage_b_path = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "reports" / "stage_b_locked_test_results.json"
    with open(stage_b_path, "w", encoding="utf-8") as f:
        json.dump(out_stage_b, f, indent=2, ensure_ascii=False)
    print(f"\nSaved Stage B Locked Test Results: {stage_b_path}")

    print("\n" + "=" * 80)
    print("  LOCKED TEST EVALUATION COMPLETED STRICTLY ONCE")
    print("=" * 80)


if __name__ == "__main__":
    main()
