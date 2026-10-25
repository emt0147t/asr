#!/usr/bin/env python3
"""
Phase 13.2.1: Frontend Preprocessing Compatibility Audit.
Workspace: D:\trainidentificationaudio
Target: experiments/vieneu_kws_research_v13_2/audit_frontend_compatibility.py

Audits the exact frontend implementations:
1. Training Frontend (train_v11.py): n_fft=512, win_length=400, hop_length=160
2. Remediated Inference Frontend (v13_1/bundle/infer.py): n_fft=512, win_length=400, hop_length=160
3. Phase 92 Portable / Baseline Frontend (v9/artifacts/inference.py): n_fft=400, win_length=400, hop_length=160
4. Portable 512 NumPy Frontend: Pure-NumPy implementation of the authoritative 512-point frontend

Evaluates on non-locked synthetic development fixtures and records numerical differences,
logits deviations, and decision parity.
"""

import sys
import json
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.transforms as T

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 1. Training frontend from train_v11.py
from experiments.vieneu_kws_research_v11.train_v11 import (
    LogMelFrontend as TrainingLogMelFrontend,
    CorrectedVietnameseKWSNet
)

# 2. Remediated Phase 13.1 inference engine
from experiments.vieneu_kws_research_v13_1.bundle.infer import (
    KWSInferenceEngine as RemediatedEngine,
    LogMelFrontend as RemediatedLogMelFrontend,
    preprocess_audio_file as preprocess_remediated,
    TARGET_CLASSES,
    DEFAULT_THRESHOLD
)

# 3. Phase 92 / Phase 9 baseline frontend (n_fft=400)
from experiments.vieneu_kws_research_v9.artifacts.inference import (
    LogMelFrontend as Phase92LogMelFrontend
)

# Portable Slaney filterbank generator from Phase 92
from experiments.vieneu_kws_research_v10.parity.portable_frontend import (
    hz_to_mel_slaney,
    mel_to_hz_slaney
)

FIXTURES_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_1" / "fixtures"
CKPT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "vieneu_kws_research_v12"
    / "bundle"
    / "checkpoint"
    / "variant_v11_hard_negative_weighted_seed42.pt"
)
OUTPUT_JSON = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_2" / "reports" / "frontend_audit_results.json"


def create_slaney_filterbank_512(n_mels=40, sample_rate=16000, f_min=20.0, f_max=8000.0) -> np.ndarray:
    """Constructs 40x257 triangular Slaney filterbank matrix for n_fft=512."""
    n_freqs = 512 // 2 + 1  # 257
    fft_freqs = np.linspace(0.0, sample_rate / 2.0, n_freqs, dtype=np.float64)

    mel_min = hz_to_mel_slaney(f_min)
    mel_max = hz_to_mel_slaney(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2, dtype=np.float64)
    hz_points = mel_to_hz_slaney(mel_points)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for m in range(n_mels):
        left = hz_points[m]
        center = hz_points[m + 1]
        right = hz_points[m + 2]
        enorm = 2.0 / (right - left)

        up_idx = np.logical_and(fft_freqs >= left, fft_freqs <= center)
        if center > left:
            fb[m, up_idx] = (fft_freqs[up_idx] - left) / (center - left) * enorm

        down_idx = np.logical_and(fft_freqs >= center, fft_freqs <= right)
        if right > center:
            fb[m, down_idx] = (right - fft_freqs[down_idx]) / (right - center) * enorm
    return fb


class PortableLogMelFrontend512:
    """Pure NumPy portable implementation of torchaudio LogMelFrontend(n_fft=512, win_length=400, hop_length=160)."""
    def __init__(self, sample_rate=16000, n_fft=512, win_length=400, hop_length=160, n_mels=40, eps=1e-6):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.eps = eps
        # Periodic Hann window matching torch.hann_window(400, periodic=True)
        self.window = (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(win_length) / win_length)).astype(np.float32)
        # Center Hann window of length 400 inside 512: (512 - 400) // 2 = 56 on both sides
        self.padded_window = np.pad(self.window, (56, 56), mode="constant")
        self.filterbank = create_slaney_filterbank_512(n_mels=n_mels, sample_rate=sample_rate)

    def compute(self, wav: np.ndarray) -> np.ndarray:
        # Reflect pad by n_fft // 2 = 256 on both sides (center=True)
        pad_len = self.n_fft // 2
        padded = np.pad(wav, (pad_len, pad_len), mode="reflect")

        num_frames = (len(wav)) // self.hop_length + 1  # 101 frames
        mel_spectrogram = np.zeros((self.n_mels, num_frames), dtype=np.float32)

        for i in range(num_frames):
            start = i * self.hop_length
            # Frame of 512 multiplied by centered window
            frame_win = padded[start : start + self.n_fft] * self.padded_window
            rfft_out = np.fft.rfft(frame_win, n=self.n_fft)
            power_spec = (np.abs(rfft_out) ** 2).astype(np.float32)  # [257]
            mel_spectrogram[:, i] = np.dot(self.filterbank, power_spec)

        log_mel = np.log(mel_spectrogram + self.eps)
        return log_mel[np.newaxis, np.newaxis, :, :].astype(np.float32)  # [1, 1, 40, 101]


def run_audit():
    print("=" * 100)
    print("PHASE 13.2.1: FRONTEND COMPATIBILITY AUDIT")
    print("=" * 100)

    # 1. Load Model Checkpoint
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model = CorrectedVietnameseKWSNet(num_classes=5)
    model.load_state_dict(state_dict)
    model.eval()

    # 2. Instantiate Frontends
    fe_training = TrainingLogMelFrontend(n_fft=512, win_length=400, hop_length=160).eval()
    fe_remediated = RemediatedLogMelFrontend(n_fft=512, win_length=400, hop_length=160).eval()
    fe_phase92 = Phase92LogMelFrontend(n_fft=400, win_length=400, hop_length=160).eval()
    fe_portable_512 = PortableLogMelFrontend512(n_fft=512, win_length=400, hop_length=160)

    fixtures = [
        FIXTURES_DIR / "dev_01_digital_silence.wav",
        FIXTURES_DIR / "dev_02_quiet_ambient_room.wav",
        FIXTURES_DIR / "dev_05_normal_len_peak089.wav",
        FIXTURES_DIR / "dev_06_normal_xuong_peak089.wav",
        FIXTURES_DIR / "dev_07_high_amplitude_trai_peak180.wav"
    ]

    results = []

    print(f"\n{'Fixture':32s} | {'Frontend':22s} | {'Spec Shape':14s} | {'MaxDiff vs Tr':14s} | {'Logits MaxDiff':14s} | {'Prediction':10s} | {'Status'}")
    print("-" * 125)

    for wav_path in fixtures:
        audio = preprocess_remediated(wav_path)
        x_torch = torch.from_numpy(audio).unsqueeze(0)  # [1, 16000]

        with torch.no_grad():
            # A. Training Frontend (n_fft=512)
            spec_tr = fe_training(x_torch)
            logits_tr = model(spec_tr)
            probs_tr = F.softmax(logits_tr, dim=-1).numpy()[0]
            cmd_tr = probs_tr[:4]
            pred_tr = TARGET_CLASSES[cmd_tr.argmax()] if cmd_tr.max() >= DEFAULT_THRESHOLD else "REJECT"

            # B. Remediated Phase 13.1 Frontend (n_fft=512)
            spec_rem = fe_remediated(x_torch)
            logits_rem = model(spec_rem)
            probs_rem = F.softmax(logits_rem, dim=-1).numpy()[0]
            cmd_rem = probs_rem[:4]
            pred_rem = TARGET_CLASSES[cmd_rem.argmax()] if cmd_rem.max() >= DEFAULT_THRESHOLD else "REJECT"

            # C. Phase 92 Frontend (n_fft=400)
            spec_p92 = fe_phase92(x_torch)
            logits_p92 = model(spec_p92)
            probs_p92 = F.softmax(logits_p92, dim=-1).numpy()[0]
            cmd_p92 = probs_p92[:4]
            pred_p92 = TARGET_CLASSES[cmd_p92.argmax()] if cmd_p92.max() >= DEFAULT_THRESHOLD else "REJECT"

            # D. Portable 512 NumPy Frontend
            spec_np512 = torch.from_numpy(fe_portable_512.compute(audio))
            logits_np512 = model(spec_np512)
            probs_np512 = F.softmax(logits_np512, dim=-1).numpy()[0]
            cmd_np512 = probs_np512[:4]
            pred_np512 = TARGET_CLASSES[cmd_np512.argmax()] if cmd_np512.max() >= DEFAULT_THRESHOLD else "REJECT"

        # Compare differences relative to Training Frontend
        diff_rem_spec = float(torch.max(torch.abs(spec_rem - spec_tr)))
        diff_rem_logits = float(torch.max(torch.abs(logits_rem - logits_tr)))

        diff_p92_spec = float(torch.max(torch.abs(spec_p92 - spec_tr)))
        diff_p92_logits = float(torch.max(torch.abs(logits_p92 - logits_tr)))

        diff_np512_spec = float(torch.max(torch.abs(spec_np512 - spec_tr)))
        diff_np512_logits = float(torch.max(torch.abs(logits_np512 - logits_tr)))

        # Print comparison rows
        print(f"{wav_path.name:32s} | {'Training (512)':22s} | {str(list(spec_tr.shape)):14s} | {'0.0 (Ref)':14s} | {'0.0 (Ref)':14s} | {pred_tr:10s} | REFERENCE")
        print(f"{'':32s} | {'Remediated 13.1 (512)':22s} | {str(list(spec_rem.shape)):14s} | {diff_rem_spec:14.2e} | {diff_rem_logits:14.2e} | {pred_rem:10s} | {'EXACT MATCH' if pred_rem == pred_tr else 'MISMATCH'}")
        print(f"{'':32s} | {'Phase 92 (400)':22s} | {str(list(spec_p92.shape)):14s} | {diff_p92_spec:14.2e} | {diff_p92_logits:14.2e} | {pred_p92:10s} | {'MATCH' if pred_p92 == pred_tr else 'DECISION DRIFT'}")
        print(f"{'':32s} | {'Portable NumPy (512)':22s} | {str(list(spec_np512.shape)):14s} | {diff_np512_spec:14.2e} | {diff_np512_logits:14.2e} | {pred_np512:10s} | {'PORTABLE MATCH' if pred_np512 == pred_tr else 'MISMATCH'}")
        print("-" * 125)

        record = {
            "fixture": wav_path.name,
            "training_frontend_512": {
                "spec_shape": list(spec_tr.shape),
                "prediction": pred_tr,
                "confidence": float(cmd_tr.max())
            },
            "remediated_frontend_13_1": {
                "spec_shape": list(spec_rem.shape),
                "spec_max_diff_vs_tr": diff_rem_spec,
                "logits_max_diff_vs_tr": diff_rem_logits,
                "prediction": pred_rem,
                "confidence": float(cmd_rem.max()),
                "matches_training_decision": bool(pred_rem == pred_tr)
            },
            "phase92_frontend_400": {
                "spec_shape": list(spec_p92.shape),
                "spec_max_diff_vs_tr": diff_p92_spec,
                "logits_max_diff_vs_tr": diff_p92_logits,
                "prediction": pred_p92,
                "confidence": float(cmd_p92.max()),
                "matches_training_decision": bool(pred_p92 == pred_tr)
            },
            "portable_numpy_512": {
                "spec_shape": list(spec_np512.shape),
                "spec_max_diff_vs_tr": diff_np512_spec,
                "logits_max_diff_vs_tr": diff_np512_logits,
                "prediction": pred_np512,
                "confidence": float(cmd_np512.max()),
                "matches_training_decision": bool(pred_np512 == pred_tr)
            }
        }
        results.append(record)

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nAudit results successfully saved to: {OUTPUT_JSON.name}")


if __name__ == "__main__":
    run_audit()
