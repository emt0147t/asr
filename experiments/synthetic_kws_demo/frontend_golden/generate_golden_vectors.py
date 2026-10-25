"""
Deterministic Golden-Vector Generation Pipeline for Vietnamese KWS.
Generates immutable reference audio fixtures, preprocessed Log-Mel tensors,
and reference predictions from frozen PyTorch and ONNX Runtime models.
Governance Invariant: HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED.
"""

import sys
import os
import shutil
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List

if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

import numpy as np
import soundfile as sf
import torch
import onnxruntime as ort

from experiments.synthetic_kws_v3.models_v3 import VietnameseTCResNet
from experiments.synthetic_kws_demo.export.export_and_benchmark import (
    ExportableTCResNet,
    CHECKPOINT_PATH,
    EXPECTED_CHECKPOINT_SHA256,
    ONNX_PATH,
)
from src.preprocessing.feature_extractor import LogMelFeatureExtractor
from experiments.synthetic_kws_v4.onset_aware_generator import (
    generate_vietnamese_onset_aware_utterance,
    generate_synthetic_silence_clip,
)
from experiments.synthetic_kws_demo.classifier import CLASS_LABELS


GOLDEN_DIR = Path("experiments/synthetic_kws_demo/frontend_golden")
WAVS_DIR = GOLDEN_DIR / "wavs"
TENSORS_DIR = GOLDEN_DIR / "tensors"
MANIFEST_PATH = GOLDEN_DIR / "golden_manifest.json"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest().upper()


def generate_golden_dataset():
    print("=" * 72)
    print("PHASE 7: Generating Immutable Frontend Golden Vectors...")
    print("=" * 72)

    # 1. Verify Checkpoint Integrity
    assert CHECKPOINT_PATH.exists(), f"Source checkpoint missing: {CHECKPOINT_PATH}"
    actual_ckpt_sha = sha256_file(CHECKPOINT_PATH)
    if actual_ckpt_sha != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(
            f"Checkpoint SHA-256 verification failed!\n"
            f"Expected: {EXPECTED_CHECKPOINT_SHA256}\n"
            f"Actual:   {actual_ckpt_sha}"
        )
    print(f"  [OK] Source checkpoint verified: {actual_ckpt_sha}")

    assert ONNX_PATH.exists(), f"Exported ONNX model missing: {ONNX_PATH}"
    print(f"  [OK] ONNX model verified: {ONNX_PATH}")

    # 2. Setup Directories
    WAVS_DIR.mkdir(parents=True, exist_ok=True)
    TENSORS_DIR.mkdir(parents=True, exist_ok=True)

    # 3. Load Models & Feature Extractor
    base_model = VietnameseTCResNet(num_classes=6, dropout_p=0.0)
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    base_model.load_state_dict(state_dict)
    base_model.eval()
    pt_model = ExportableTCResNet(base_model)
    pt_model.eval()

    ort_session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    extractor = LogMelFeatureExtractor(sample_rate=16000, n_mels=40)
    extractor.eval()

    # 4. Generate/Acquire 12 Representative Test Cases
    cases_config = [
        # Canonical Demo Fixtures (1-6)
        {
            "case_id": "case_01_trai_demo",
            "name": "TRAI Canonical Demonstration Fixture",
            "category": "canonical_demo",
            "intended_label": "TRAI",
            "source_type": "fixture_copy",
            "source_path": Path("experiments/synthetic_kws_demo/fixtures/demo_01_trai.wav"),
            "description": "Native onset-aware stop closure (25ms) + transient burst (15ms) for 'trái'."
        },
        {
            "case_id": "case_02_phai_demo",
            "name": "PHAI Canonical Demonstration Fixture",
            "category": "canonical_demo",
            "intended_label": "PHAI",
            "source_type": "fixture_copy",
            "source_path": Path("experiments/synthetic_kws_demo/fixtures/demo_02_phai.wav"),
            "description": "Continuous labiodental frication (50ms) without closure for 'phải'."
        },
        {
            "case_id": "case_03_len_demo",
            "name": "LEN Canonical Demonstration Fixture",
            "category": "canonical_demo",
            "intended_label": "LEN",
            "source_type": "fixture_copy",
            "source_path": Path("experiments/synthetic_kws_demo/fixtures/demo_03_len.wav"),
            "description": "Level tone formant resonance for 'lên'."
        },
        {
            "case_id": "case_04_xuong_demo",
            "name": "XUONG Canonical Demonstration Fixture",
            "category": "canonical_demo",
            "intended_label": "XUONG",
            "source_type": "fixture_copy",
            "source_path": Path("experiments/synthetic_kws_demo/fixtures/demo_04_xuong.wav"),
            "description": "Sibilant fricative onset + dipping contour for 'xuống'."
        },
        {
            "case_id": "case_05_unknown_demo",
            "name": "UNKNOWN Canonical Demonstration Fixture",
            "category": "canonical_demo",
            "intended_label": "UNKNOWN",
            "source_type": "fixture_copy",
            "source_path": Path("experiments/synthetic_kws_demo/fixtures/demo_05_unknown.wav"),
            "description": "Out-of-vocabulary non-target confuser ('chào')."
        },
        {
            "case_id": "case_06_silence_demo",
            "name": "SILENCE Canonical Demonstration Fixture",
            "category": "canonical_demo",
            "intended_label": "SILENCE",
            "source_type": "fixture_copy",
            "source_path": Path("experiments/synthetic_kws_demo/fixtures/demo_06_silence.wav"),
            "description": "Full-buffer unshifted ambient background dither noise."
        },
        # Timing Variants (7-8)
        {
            "case_id": "case_07_trai_leading_pad",
            "name": "TRAI with 200ms Leading Zero Padding",
            "category": "timing_variant",
            "intended_label": "TRAI",
            "source_type": "synthesize_trai_leading",
            "description": "Speech utterance delayed by 200ms (3,200 zero samples leading) right-padded to 16,000 samples."
        },
        {
            "case_id": "case_08_phai_centered",
            "name": "PHAI with Centered Zero Padding",
            "category": "timing_variant",
            "intended_label": "PHAI",
            "source_type": "synthesize_phai_centered",
            "description": "Speech utterance centered with equal 4,000 samples leading and 4,000 samples trailing padding."
        },
        # Numerical Edge Cases (9-11)
        {
            "case_id": "case_09_silence_pure_zeros",
            "name": "Pure Digital Silence (Exact Zeros)",
            "category": "numerical_edge_case",
            "intended_label": "SILENCE",
            "source_type": "synthesize_pure_zeros",
            "description": "16,000 exact 0.0 samples. Tests log epsilon stabilizer ln(1e-6) = -13.81551 across all bins."
        },
        {
            "case_id": "case_10_low_amplitude_ambient",
            "name": "Ultra-Low-Amplitude Gaussian Ambient Floor",
            "category": "numerical_edge_case",
            "intended_label": "SILENCE",
            "source_type": "synthesize_low_ambient",
            "description": "Gaussian noise floor with sigma=1e-5 (max abs < 1e-4). Tests low-energy dynamic range."
        },
        {
            "case_id": "case_11_transient_impulse",
            "name": "Single-Sample Unit Impulse Spike",
            "category": "numerical_edge_case",
            "intended_label": "UNKNOWN",
            "source_type": "synthesize_unit_impulse",
            "description": "Single impulse spike at sample index 400 (x[400]=0.95). Tests wideband transient filterbank response."
        },
        # Holdout Anchor (12)
        {
            "case_id": "case_12_holdout_trai_native",
            "name": "Historical V4 Holdout TRAI Anchor",
            "category": "holdout_anchor",
            "intended_label": "TRAI",
            "source_type": "holdout_copy",
            "source_path": Path("experiments/synthetic_kws_v4/data/fresh_holdout/fresh_v4_cmd_trai_SPK_CALIB_031_take000.wav"),
            "description": "Bit-exact anchor clip from the frozen V4 fresh synthetic holdout cohort (SPK_CALIB_031)."
        }
    ]

    golden_records = []

    for item in cases_config:
        cid = item["case_id"]
        wav_fname = f"{cid}.wav"
        target_wav_p = WAVS_DIR / wav_fname

        # Synthesize or copy audio
        stype = item["source_type"]
        if stype in ["fixture_copy", "holdout_copy"]:
            src = item["source_path"]
            assert src.exists(), f"Source audio missing: {src}"
            shutil.copy2(src, target_wav_p)
            audio, sr = sf.read(str(target_wav_p), dtype="float32")
        elif stype == "synthesize_trai_leading":
            raw = generate_vietnamese_onset_aware_utterance("trái", sr=16000, f0_base=130.0, duration_sec=0.5, dialect="Northern")
            pad_l = 3200
            pad_r = 16000 - len(raw) - pad_l
            audio = np.pad(raw, (pad_l, pad_r)).astype(np.float32)
            sr = 16000
            sf.write(str(target_wav_p), audio, sr, subtype="PCM_16")
        elif stype == "synthesize_phai_centered":
            raw = generate_vietnamese_onset_aware_utterance("phải", sr=16000, f0_base=130.0, duration_sec=0.5, dialect="Northern")
            pad_l = (16000 - len(raw)) // 2
            pad_r = 16000 - len(raw) - pad_l
            audio = np.pad(raw, (pad_l, pad_r)).astype(np.float32)
            sr = 16000
            sf.write(str(target_wav_p), audio, sr, subtype="PCM_16")
        elif stype == "synthesize_pure_zeros":
            audio = np.zeros(16000, dtype=np.float32)
            sr = 16000
            sf.write(str(target_wav_p), audio, sr, subtype="PCM_16")
        elif stype == "synthesize_low_ambient":
            rng = np.random.RandomState(20261016)
            audio = (rng.randn(16000) * 1e-5).astype(np.float32)
            sr = 16000
            sf.write(str(target_wav_p), audio, sr, subtype="PCM_16")
        elif stype == "synthesize_unit_impulse":
            audio = np.zeros(16000, dtype=np.float32)
            audio[400] = 0.95
            sr = 16000
            sf.write(str(target_wav_p), audio, sr, subtype="PCM_16")
        else:
            raise ValueError(f"Unknown source type: {stype}")

        # Always read audio directly from the final written PCM_16 WAV on disk
        audio, sr = sf.read(str(target_wav_p), dtype="float32")
        assert len(audio) == 16000 and sr == 16000
        wav_sha = sha256_file(target_wav_p)

        # Preprocess to Log-Mel Spectrogram
        audio_t = torch.from_numpy(audio).float()
        with torch.no_grad():
            log_mel_t = extractor(audio_t)  # Shape: (1, 1, 40, 101)
            pt_logits, pt_probs = pt_model(log_mel_t)

        log_mel_np = log_mel_t.cpu().numpy()  # float32 (1, 1, 40, 101)
        assert log_mel_np.shape == (1, 1, 40, 101)
        assert not np.isnan(log_mel_np).any()
        assert not np.isinf(log_mel_np).any()

        # Save immutable golden tensor
        tensor_fname = f"{cid}_logmel.npy"
        target_tensor_p = TENSORS_DIR / tensor_fname
        np.save(str(target_tensor_p), log_mel_np)
        tensor_sha = sha256_file(target_tensor_p)

        # PyTorch predictions
        pt_logits_vec = pt_logits.cpu().numpy()[0].tolist()
        pt_probs_vec = pt_probs.cpu().numpy()[0].tolist()
        pt_pred_cid = int(np.argmax(pt_probs_vec))

        # ONNX Runtime predictions
        ort_logits_np, ort_probs_np = ort_session.run(None, {"log_mel": log_mel_np})
        ort_logits_vec = ort_logits_np[0].tolist()
        ort_probs_vec = ort_probs_np[0].tolist()
        ort_pred_cid = int(np.argmax(ort_probs_vec))

        # Parity calculation
        argmax_match = (pt_pred_cid == ort_pred_cid)
        max_logits_diff = float(np.max(np.abs(np.array(pt_logits_vec) - np.array(ort_logits_vec))))
        max_prob_diff = float(np.max(np.abs(np.array(pt_probs_vec) - np.array(ort_probs_vec))))
        mean_prob_diff = float(np.mean(np.abs(np.array(pt_probs_vec) - np.array(ort_probs_vec))))

        record = {
            "case_id": cid,
            "case_name": item["name"],
            "category": item["category"],
            "intended_label": item["intended_label"],
            "description": item["description"],
            "wav_artifact": {
                "filename": wav_fname,
                "path": str(target_wav_p),
                "sha256": wav_sha,
                "sample_rate": 16000,
                "samples": len(audio),
                "duration_sec": 1.0,
                "channels": 1
            },
            "tensor_artifact": {
                "filename": tensor_fname,
                "path": str(target_tensor_p),
                "sha256": tensor_sha,
                "shape": list(log_mel_np.shape),
                "dtype": str(log_mel_np.dtype),
                "min_value": float(np.min(log_mel_np)),
                "max_value": float(np.max(log_mel_np)),
                "mean_value": float(np.mean(log_mel_np)),
                "std_value": float(np.std(log_mel_np))
            },
            "pytorch_reference": {
                "predicted_label": CLASS_LABELS[pt_pred_cid],
                "predicted_class_id": pt_pred_cid,
                "confidence": float(pt_probs_vec[pt_pred_cid]),
                "logits": [round(x, 7) for x in pt_logits_vec],
                "probabilities": [round(x, 7) for x in pt_probs_vec]
            },
            "onnx_reference": {
                "predicted_label": CLASS_LABELS[ort_pred_cid],
                "predicted_class_id": ort_pred_cid,
                "confidence": float(ort_probs_vec[ort_pred_cid]),
                "logits": [round(x, 7) for x in ort_logits_vec],
                "probabilities": [round(x, 7) for x in ort_probs_vec]
            },
            "parity": {
                "argmax_agreement": argmax_match,
                "max_abs_logits_diff": max_logits_diff,
                "max_abs_prob_diff": max_prob_diff,
                "mean_abs_prob_diff": mean_prob_diff
            }
        }
        golden_records.append(record)

        print(f"  [{cid:<27}] | Pred: {CLASS_LABELS[pt_pred_cid]:<7} ({pt_probs_vec[pt_pred_cid]:.4f}) | "
              f"Tensor SHA: {tensor_sha[:16]}... | ParityDiff: {max_prob_diff:.2e}")

    # Build master manifest
    manifest_data = {
        "manifest_version": "1.0.0",
        "generated_at": "2026-10-16",
        "governance": "HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED. Golden vectors are synthetic reference fixtures.",
        "verified_frontend_contract": {
            "sample_rate": 16000,
            "channels": 1,
            "samples": 16000,
            "duration_sec": 1.0,
            "stft_parameters": {
                "n_fft": 512,
                "win_length": 400,
                "hop_length": 160,
                "window_function": "Periodic Hann Window (torch.hann_window(400, periodic=True))",
                "center": True,
                "pad_mode": "reflect",
                "normalized": False,
                "power": 2.0
            },
            "mel_filterbank_parameters": {
                "n_mels": 40,
                "f_min": 20.0,
                "f_max": 8000.0,
                "mel_scale": "slaney",
                "norm": "slaney (area-normalized)"
            },
            "compression_parameters": {
                "log_type": "natural_logarithm (ln)",
                "epsilon": 1.0e-6,
                "formula": "ln(mel_power_spectrum + 1.0e-6)"
            },
            "output_tensor_spec": {
                "shape": [1, 1, 40, 101],
                "layout": "(Batch, Channels, Mel_Bands, Time_Frames)",
                "dtype": "float32"
            },
            "class_ordering": {str(i): label for i, label in enumerate(CLASS_LABELS)}
        },
        "source_checkpoint": {
            "path": str(CHECKPOINT_PATH),
            "sha256": actual_ckpt_sha
        },
        "onnx_model": {
            "path": str(ONNX_PATH),
            "sha256": sha256_file(ONNX_PATH)
        },
        "total_cases": len(golden_records),
        "cases": golden_records
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"\nSaved golden manifest with {len(golden_records)} cases to: {MANIFEST_PATH}")
    return MANIFEST_PATH


if __name__ == "__main__":
    generate_golden_dataset()
