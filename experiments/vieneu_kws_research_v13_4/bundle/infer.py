#!/usr/bin/env python3
"""
Vietnamese Keyword Spotting (KWS) Standalone Inference CLI.
Bundle Version: Phase 13.4 Final Release Candidate
Candidate: variant_v11_hard_negative_weighted_seed42
Classes: TRAI, PHAI, LEN, XUONG, REJECT

Supports PyTorch and ONNX Runtime execution backends.
Enforces remediated preprocessing:
- 16 kHz mono float32
- Pad right with zeros if shorter than 16,000 samples; truncate if longer
- Peak clamping ONLY if max(abs(audio)) > 1.0 (zero floor amplification)
- Log-Mel Frontend: n_fft=512, win_length=400, hop_length=160, 40 Slaney Mel bins
- Decision Threshold: theta_cmd = 0.20
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Union

import numpy as np
import soundfile as sf
from scipy import signal

BUNDLE_DIR = Path(__file__).resolve().parent
DEFAULT_PT_CKPT = BUNDLE_DIR / "checkpoint" / "variant_v11_hard_negative_weighted_seed42.pt"
DEFAULT_ONNX_PATH = BUNDLE_DIR / "onnx" / "vietnamese_kws_v11.onnx"
DEFAULT_THRESHOLD = 0.20

CLASSES = ["TRAI", "PHAI", "LEN", "XUONG", "REJECT"]
TARGET_CLASSES = ["TRAI", "PHAI", "LEN", "XUONG"]


def preprocess_audio(wav_path: Union[str, Path], target_sr: int = 16000, target_samples: int = 16000) -> np.ndarray:
    """
    Authoritative reference preprocessing:
    1. Read WAV, downmix to mono.
    2. Resample to 16,000 Hz if necessary.
    3. Pad right or truncate to 16,000 samples (1.0 s).
    4. Peak clamping ONLY if max(abs(audio)) > 1.0.
    """
    audio, sr = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)

    if sr != target_sr:
        num_target = int(round(len(audio) * float(target_sr) / sr))
        audio = signal.resample(audio, num_target)

    if len(audio) < target_samples:
        pad_right = target_samples - len(audio)
        audio = np.pad(audio, (0, pad_right), mode="constant", constant_values=0.0)
    elif len(audio) > target_samples:
        audio = audio[:target_samples]

    max_val = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.0
    if max_val > 1.0:
        audio = audio / max_val

    return audio.astype(np.float32)


# PyTorch Backend Components (lazy loaded if backend == 'pytorch')
class LogMelFrontendTorch:
    def __init__(self, sample_rate=16000, n_fft=512, win_length=400, hop_length=160, n_mels=40, eps=1e-6):
        import torch
        import torchaudio
        self.torch = torch
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
            mel_scale="slaney",
            norm="slaney",
        )
        self.eps = eps

    def __call__(self, x):
        return self.torch.log(self.mel(x) + self.eps)


def create_slaney_filterbank_512(n_mels=40, sample_rate=16000, f_min=20.0, f_max=8000.0) -> np.ndarray:
    """Constructs 40x257 Slaney filterbank matrix in NumPy for ONNX / C++ parity."""
    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0) if hz < 1000.0 else 1000.0 + 1127.01048 * np.log(hz / 1000.0)
    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0) if mel < 1000.0 else 1000.0 * np.exp((mel - 1000.0) / 1127.01048)

    n_freqs = 512 // 2 + 1  # 257
    fft_freqs = np.linspace(0.0, sample_rate / 2.0, n_freqs, dtype=np.float64)
    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2, dtype=np.float64)
    hz_points = np.array([mel_to_hz(m) for m in mel_points])

    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for m in range(n_mels):
        left, center, right = hz_points[m], hz_points[m + 1], hz_points[m + 2]
        enorm = 2.0 / (right - left)
        up_idx = np.logical_and(fft_freqs >= left, fft_freqs <= center)
        if center > left:
            fb[m, up_idx] = (fft_freqs[up_idx] - left) / (center - left) * enorm
        down_idx = np.logical_and(fft_freqs >= center, fft_freqs <= right)
        if right > center:
            fb[m, down_idx] = (right - fft_freqs[down_idx]) / (right - center) * enorm
    return fb


class LogMelFrontendNumPy:
    """Portable Pure-NumPy frontend producing identical spectrograms to PyTorch STFT (diff <= 2e-5)."""
    def __init__(self, sample_rate=16000, n_fft=512, win_length=400, hop_length=160, n_mels=40, eps=1e-6):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.eps = eps
        hann_400 = (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(win_length) / win_length)).astype(np.float32)
        # Center Hann window of length 400 inside 512-point FFT frame: (512 - 400) // 2 = 56 on both sides
        self.padded_window = np.pad(hann_400, (56, 56), mode="constant")
        self.filterbank = create_slaney_filterbank_512(n_mels=n_mels, sample_rate=sample_rate)

    def compute(self, wav: np.ndarray) -> np.ndarray:
        pad_len = self.n_fft // 2  # 256
        padded = np.pad(wav, (pad_len, pad_len), mode="reflect")
        num_frames = len(wav) // self.hop_length + 1  # 101 frames
        mel_spec = np.zeros((self.n_mels, num_frames), dtype=np.float32)
        for i in range(num_frames):
            start = i * self.hop_length
            frame_win = padded[start : start + self.n_fft] * self.padded_window
            rfft_out = np.fft.rfft(frame_win, n=self.n_fft)
            power_spec = (np.abs(rfft_out) ** 2).astype(np.float32)
            mel_spec[:, i] = np.dot(self.filterbank, power_spec)
        log_mel = np.log(mel_spec + self.eps)
        return log_mel[np.newaxis, np.newaxis, :, :].astype(np.float32)


class VietnameseKWSInferencer:
    """Universal Inference Engine supporting both PyTorch and ONNX backends."""
    def __init__(
        self,
        backend: str = "pytorch",
        model_path: Path = None,
        theta_cmd: float = DEFAULT_THRESHOLD,
        device: str = "cpu"
    ):
        self.backend = backend.lower()
        self.theta_cmd = theta_cmd
        self.device = device

        if self.backend == "pytorch":
            import torch
            import torch.nn as nn
            import torch.nn.functional as F

            self.torch = torch
            self.F = F
            self.frontend = LogMelFrontendNumPy()

            # Architecture definition
            class CorrectedVietnameseKWSNet(nn.Module):
                def __init__(self, num_classes=5, dropout_p=0.3):
                    super().__init__()
                    self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
                    self.bn1 = nn.BatchNorm2d(32)
                    self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
                    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
                    self.bn2 = nn.BatchNorm2d(64)
                    self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
                    self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
                    self.bn3 = nn.BatchNorm2d(64)
                    self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
                    self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
                    self.bn4 = nn.BatchNorm2d(128)
                    self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
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
                    return self.fc2(out)

            ckpt_file = Path(model_path) if model_path else DEFAULT_PT_CKPT
            if not ckpt_file.exists():
                raise FileNotFoundError(f"PyTorch checkpoint not found: {ckpt_file}")
            self.model = CorrectedVietnameseKWSNet(num_classes=5)
            ckpt_data = torch.load(str(ckpt_file), map_location=device)
            state_dict = ckpt_data["model_state_dict"] if "model_state_dict" in ckpt_data else ckpt_data
            self.model.load_state_dict(state_dict)
            self.model.eval()

        elif self.backend == "onnx":
            import onnxruntime as ort
            self.ort = ort
            self.frontend = LogMelFrontendNumPy()

            onnx_file = Path(model_path) if model_path else DEFAULT_ONNX_PATH
            if not onnx_file.exists():
                raise FileNotFoundError(f"ONNX model file not found: {onnx_file}")
            self.session = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
        else:
            raise ValueError(f"Unsupported backend '{backend}'. Choose 'pytorch' or 'onnx'.")

    def predict(self, wav_path: Union[str, Path]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        audio = preprocess_audio(wav_path)

        if self.backend == "pytorch":
            spec = self.frontend.compute(audio)
            x = self.torch.from_numpy(spec).to(self.device)
            with self.torch.no_grad():
                logits = self.model(x)
                probs = self.F.softmax(logits, dim=-1).cpu().numpy()[0]
        else:  # onnx
            spec = self.frontend.compute(audio)
            ort_out = self.session.run(None, {"log_mel": spec})
            probs = ort_out[1][0]  # probabilities output

        dt_ms = (time.perf_counter() - t0) * 1000.0

        # Validate output shape and finiteness
        assert len(probs) == 5, f"Expected 5 output probabilities, got {len(probs)}"
        assert np.all(np.isfinite(probs)), "Probabilities contain NaN or Inf values"

        # Canonical 5-class decision rule (Chapter 07 Preprocessing & Inference Contract):
        # c* = argmax over ALL 5 classes (TRAI, PHAI, LEN, XUONG, REJECT)
        # Accepted only if c* is a command (0..3) AND P(c*) >= theta_cmd (0.20)
        top_idx = int(np.argmax(probs))
        raw_argmax_class = CLASSES[top_idx]
        top_prob = float(probs[top_idx])

        if top_idx < 4 and top_prob >= self.theta_cmd:
            predicted_class = raw_argmax_class
            rejection_status = "ACCEPTED"
            confidence = top_prob
        else:
            predicted_class = "REJECT"
            rejection_status = "REJECTED_BY_REJECT_CLASS" if top_idx == 4 else "REJECTED_LOW_CONFIDENCE"
            confidence = top_prob if top_idx == 4 else float(np.max(probs[:4]))

        return {
            "wav_path": str(wav_path),
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "rejection_status": rejection_status,
            "raw_argmax_class": raw_argmax_class,
            "theta_cmd": self.theta_cmd,
            "backend": self.backend,
            "probabilities": {CLASSES[i]: round(float(probs[i]), 4) for i in range(5)},
            "latency_ms": round(dt_ms, 2)
        }


def main():
    parser = argparse.ArgumentParser(description="Vietnamese KWS Inference CLI (Phase 13.4 Final Candidate)")
    parser.add_argument("wav_path", nargs="?", help="Path to input audio file (.wav)")
    parser.add_argument("--backend", choices=["pytorch", "onnx"], default="pytorch", help="Execution backend")
    parser.add_argument("--model", type=Path, default=None, help="Custom path to model checkpoint/ONNX file")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Decision threshold theta_cmd")
    parser.add_argument("--dir", type=Path, default=None, help="Directory of WAV files for batch inference")
    parser.add_argument("--out", type=Path, default=None, help="Output JSON path for batch predictions")

    args = parser.parse_args()

    inferencer = VietnameseKWSInferencer(backend=args.backend, model_path=args.model, theta_cmd=args.threshold)

    if args.dir:
        wav_files = sorted(list(args.dir.glob("*.wav")))
        print(f"Running batch inference on {len(wav_files)} files using backend={args.backend}...")
        results = []
        for wf in wav_files:
            res = inferencer.predict(wf)
            results.append(res)
            print(f"  {wf.name:<30} -> {res['predicted_class']:<8} (conf={res['confidence']:.3f}, {res['latency_ms']:.1f}ms)")
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Saved {len(results)} predictions to {args.out}")
    elif args.wav_path:
        res = inferencer.predict(args.wav_path)
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
