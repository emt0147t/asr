#!/usr/bin/env python3
"""
Phase 13.1 Preprocessing Remediation Comparison Script.
Workspace: D:\trainidentificationaudio
Target: experiments/vieneu_kws_research_v13_1/compare_preprocessing_remediation.py

Compares:
- Pre-remediation Phase 12 engine (unconditional -1.0 dBFS normalization)
- Post-remediation Phase 13.1 engine (clamping only when max > 1.0; no floor amplification)
- Upstream training reference (train_v11.py BoundedDatasetV11 logic)
across the 8 non-locked remediation fixtures.
"""

import sys
import json
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Phase 12 Inference Engine
from experiments.vieneu_kws_research_v12.bundle.infer import (
    KWSInferenceEngine as EnginePhase12,
    preprocess_audio_file as preprocess_p12
)

# Phase 13.1 Remediated Engine
from experiments.vieneu_kws_research_v13_1.bundle.infer import (
    KWSInferenceEngine as EnginePhase13_1,
    preprocess_audio_file as preprocess_p13_1,
    CLASSES,
    TARGET_CLASSES,
    DEFAULT_THRESHOLD
)

FIXTURES_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_1" / "fixtures"
OUTPUT_REPORT = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_1" / "preprocessing_comparison_results.json"


def training_ref_preprocess(wav_path: Path, target_samples: int = 16000) -> np.ndarray:
    """Exact preprocessing extracted from train_v11.py BoundedDatasetV11."""
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    if len(audio) < target_samples:
        pad = target_samples - len(audio)
        audio = np.pad(audio, (0, pad), mode="constant")
    elif len(audio) > target_samples:
        audio = audio[:target_samples]

    max_val = np.max(np.abs(audio)) if len(audio) > 0 else 0.0
    if max_val > 1.0:
        audio = audio / max_val
    return audio.astype(np.float32)


def main():
    print("=" * 100)
    print("PHASE 13.1: PREPROCESSING REMEDIATION NUMERICAL COMPARISON")
    print("=" * 100)

    fixtures = sorted(FIXTURES_DIR.glob("*.wav"))
    if not fixtures:
        raise FileNotFoundError(f"No fixtures found in {FIXTURES_DIR}")

    ckpt_path = (
        PROJECT_ROOT
        / "experiments"
        / "vieneu_kws_research_v13_1"
        / "bundle"
        / "checkpoint"
        / "variant_v11_hard_negative_weighted_seed42.pt"
    )

    engine_p12 = EnginePhase12(checkpoint_path=ckpt_path, theta_cmd=0.20, device="cpu")
    engine_p13 = EnginePhase13_1(checkpoint_path=ckpt_path, theta_cmd=0.20, device="cpu")

    results = []

    header = f"{'Fixture':36s} | {'Raw Peak':8s} | {'P12 Peak':8s} | {'P13 Peak':8s} | {'P12 Pred':10s} | {'P13 Pred':10s} | {'Ref Pred':10s} | {'P13 vs Ref'}"
    print(header)
    print("-" * len(header))

    for wav_path in fixtures:
        raw_audio, sr = sf.read(wav_path, dtype="float32")
        raw_peak = float(np.max(np.abs(raw_audio))) if len(raw_audio) > 0 else 0.0

        # Preprocessed waveforms
        wave_p12 = preprocess_p12(wav_path)
        wave_p13 = preprocess_p13_1(wav_path)
        wave_ref = training_ref_preprocess(wav_path)

        peak_p12 = float(np.max(np.abs(wave_p12)))
        peak_p13 = float(np.max(np.abs(wave_p13)))
        peak_ref = float(np.max(np.abs(wave_ref)))

        # Waveform difference between P13 and Training Reference
        wave_diff_ref = float(np.max(np.abs(wave_p13 - wave_ref)))

        # Predictions
        pred_p12 = engine_p12.predict(wav_path)
        pred_p13 = engine_p13.predict(wav_path)

        # Run model on training reference waveform
        with torch.no_grad():
            x_ref = torch.from_numpy(wave_ref).unsqueeze(0)
            specs_ref = engine_p13.frontend(x_ref)
            logits_ref = engine_p13.model(specs_ref)
            probs_ref = F.softmax(logits_ref, dim=-1).cpu().numpy()[0]
        cmd_probs_ref = probs_ref[:4]
        raw_argmax_ref = int(np.argmax(cmd_probs_ref))
        max_prob_ref = float(cmd_probs_ref[raw_argmax_ref])
        label_ref = TARGET_CLASSES[raw_argmax_ref] if max_prob_ref >= 0.20 else "REJECT"

        p13_matches_ref = (pred_p13["predicted_class"] == label_ref) and (wave_diff_ref < 1e-5)

        record = {
            "fixture": wav_path.name,
            "raw_peak": round(raw_peak, 5),
            "p12_peak": round(peak_p12, 5),
            "p13_peak": round(peak_p13, 5),
            "training_ref_peak": round(peak_ref, 5),
            "p13_vs_ref_waveform_max_diff": wave_diff_ref,
            "p12_prediction": pred_p12["predicted_class"],
            "p12_confidence": pred_p12["confidence"],
            "p12_rejection_status": pred_p12["rejection_status"],
            "p12_probabilities": pred_p12["probabilities"],
            "p13_prediction": pred_p13["predicted_class"],
            "p13_confidence": pred_p13["confidence"],
            "p13_rejection_status": pred_p13["rejection_status"],
            "p13_probabilities": pred_p13["probabilities"],
            "training_ref_prediction": label_ref,
            "training_ref_confidence": round(max_prob_ref, 4),
            "p13_matches_training_ref": p13_matches_ref
        }
        results.append(record)

        match_str = "[EXACT MATCH]" if p13_matches_ref else "[MISMATCH]"
        print(f"{wav_path.name:36s} | {raw_peak:8.4f} | {peak_p12:8.4f} | {peak_p13:8.4f} | {pred_p12['predicted_class']:10s} | {pred_p13['predicted_class']:10s} | {label_ref:10s} | {match_str}")

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 100)
    print(f"Results saved to: {OUTPUT_REPORT.name}")
    print("=" * 100)


if __name__ == "__main__":
    main()
