#!/usr/bin/env python3
"""
Phase 13.1 Remediated Standalone Inference Engine for Vietnamese Bounded Keyword Spotter.
Workspace: D:\trainidentificationaudio
Target: experiments/vieneu_kws_research_v13_1/bundle/infer.py

Applies the corrected preprocessing specification matching train_v11.py and
Phase 91 DEPLOYMENT_CONTRACT.md:
- Clamping only when max(abs(audio)) > 1.0
- Noise floor amplification strictly prohibited
- Target length: 16,000 samples (1.0s at 16 kHz)
- Operating threshold: theta_cmd = 0.20
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from scipy import signal

CLASSES = ["TRAI", "PHAI", "LEN", "XUONG", "REJECT"]
TARGET_CLASSES = ["TRAI", "PHAI", "LEN", "XUONG"]
DEFAULT_THRESHOLD = 0.20
DEFAULT_CKPT = Path(__file__).resolve().parent / "checkpoint" / "variant_v11_hard_negative_weighted_seed42.pt"


class LogMelFrontend(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=512, win_length=400, hop_length=160, n_mels=40, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.mel = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm="slaney",
            n_mels=n_mels,
            mel_scale="slaney",
            f_min=20.0,
            f_max=8000.0
        )

    def forward(self, x):
        mel_spec = self.mel(x)
        log_mel = torch.log(mel_spec + self.eps)
        if log_mel.dim() == 3:
            log_mel = log_mel.unsqueeze(1)
        return log_mel


class CorrectedVietnameseKWSNet(nn.Module):
    def __init__(self, num_classes=5, dropout_p=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # (40, 101) -> (20, 50)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # (20, 50) -> (10, 25)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # (10, 25) -> (5, 12)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))      # (5, 12) -> (1, 1)

        self.dropout = nn.Dropout(p=dropout_p)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)

        out = self.pool1(F.relu(self.bn1(self.conv1(x))))
        out = self.pool2(F.relu(self.bn2(self.conv2(out))))
        out = self.pool3(F.relu(self.bn3(self.conv3(out))))
        out = self.global_pool(F.relu(self.bn4(self.conv4(out))))

        out = torch.flatten(out, 1)
        out = self.dropout(F.relu(self.fc1(out)))
        logits = self.fc2(out)
        return logits


def preprocess_audio_file(wav_path: Path, target_sr: int = 16000, target_samples: int = 16000) -> np.ndarray:
    """
    Authoritative reference preprocessing matching train_v11.py and DEPLOYMENT_CONTRACT.md.
    - Downmix to mono if multi-channel.
    - Sinc/polyphase resample to 16 kHz if sample rate differs.
    - Pad right or center-crop to 16,000 samples.
    - Clamped ONLY if max(abs(audio)) > 1.0 (zero floor amplification).
    """
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)

    if sr != target_sr:
        num_target = int(round(len(audio) * float(target_sr) / sr))
        audio = signal.resample(audio, num_target)

    # Pad right with zeros if shorter than 16,000 samples (matching BoundedDatasetV11 and audio_validator.py)
    if len(audio) < target_samples:
        pad_right = target_samples - len(audio)
        audio = np.pad(audio, (0, pad_right), mode="constant", constant_values=0.0)
    elif len(audio) > target_samples:
        audio = audio[:target_samples]

    # Peak normalization: Clamped ONLY if amplitude exceeds 1.0 (prevents digital clipping, preserves quiet noise floor)
    max_val = np.max(np.abs(audio)) if len(audio) > 0 else 0.0
    if max_val > 1.0:
        audio = audio / max_val

    return audio.astype(np.float32)


class KWSInferenceEngine:
    def __init__(self, checkpoint_path: Path = DEFAULT_CKPT, theta_cmd: float = DEFAULT_THRESHOLD, device: str = "cpu"):
        self.device = torch.device(device)
        self.theta_cmd = theta_cmd
        self.frontend = LogMelFrontend().to(self.device)
        self.frontend.eval()

        self.model = CorrectedVietnameseKWSNet(num_classes=5).to(self.device)
        ckpt_data = torch.load(checkpoint_path, map_location=self.device)
        state_dict = ckpt_data["model_state_dict"] if "model_state_dict" in ckpt_data else ckpt_data
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, wav_path: Path) -> dict:
        t0 = time.perf_counter()
        audio = preprocess_audio_file(wav_path)
        x = torch.from_numpy(audio).unsqueeze(0).to(self.device)

        with torch.no_grad():
            specs = self.frontend(x)
            logits = self.model(specs)
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        dt_ms = (time.perf_counter() - t0) * 1000.0

        cmd_probs = probs[:4]
        raw_argmax_id = int(np.argmax(cmd_probs))
        raw_argmax_class = TARGET_CLASSES[raw_argmax_id]
        max_prob = float(cmd_probs[raw_argmax_id])

        if max_prob >= self.theta_cmd:
            predicted_class = raw_argmax_class
            rejection_status = "ACCEPTED"
        else:
            predicted_class = "REJECT"
            rejection_status = "REJECTED_LOW_CONFIDENCE"

        return {
            "wav_path": str(wav_path),
            "predicted_class": predicted_class,
            "confidence": round(max_prob, 4),
            "rejection_status": rejection_status,
            "raw_argmax_class": raw_argmax_class,
            "theta_cmd": self.theta_cmd,
            "probabilities": {CLASSES[i]: round(float(probs[i]), 4) for i in range(5)},
            "latency_ms": round(dt_ms, 2)
        }


def main():
    parser = argparse.ArgumentParser(description="Remediated Vietnamese KWS Inference (Phase 13.1)")
    parser.add_argument("--wav", type=str, required=True, help="Path to input audio WAV file")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CKPT), help="Path to model checkpoint .pt")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Command decision threshold theta_cmd")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    engine = KWSInferenceEngine(checkpoint_path=Path(args.checkpoint), theta_cmd=args.threshold, device=args.device)
    res = engine.predict(Path(args.wav))

    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print(f"File:       {res['wav_path']}")
        print(f"Prediction: {res['predicted_class']} (Confidence: {res['confidence']:.4f})")
        print(f"Status:     {res['rejection_status']} (Threshold: {res['theta_cmd']})")
        print(f"Latency:    {res['latency_ms']:.2f} ms")
        print(f"Probs:      {res['probabilities']}")


if __name__ == "__main__":
    main()
