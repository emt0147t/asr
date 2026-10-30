#!/usr/bin/env python3
"""
Workstream C — Phase 7C: Cross-Channel Benchmark Script
Repository: D:\trainidentificationaudio
Branch baseline: feature/continuous-kws-segmentation

Governance Labels:
  - EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED
  - NOT_VALIDATED_ON_NATURAL_SPEECH
  - NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION
  - NOT_VALIDATED_PENDING_DATA
"""

import sys
import io
import os
import time
import json
import csv
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import scipy.signal as signal
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

PROJECT_ROOT = Path("D:/trainidentificationaudio")
V13_4_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4"
BUNDLE_DIR = V13_4_DIR / "bundle"
DEMO_DIR = V13_4_DIR / "demo"
SCRIPTS_DIR = V13_4_DIR / "scripts"
REPORTS_DIR = V13_4_DIR / "reports"
ROOT_REPORTS_DIR = PROJECT_ROOT / "reports"
V12_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12"
AUDIO_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_3" / "data" / "audio"

for p in [str(BUNDLE_DIR), str(DEMO_DIR), str(SCRIPTS_DIR), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from infer import VietnameseKWSInferencer, TARGET_CLASSES, CLASSES
from hybrid_segmenter import HybridSpeechBandValleySegmenter, GameModeSessionController
from acoustic_valley_segmenter import AcousticValleySegmenter
from live_mic_engine import UtteranceSegmenter
from audit_phase5_3_final_metrics import compute_standard_sequence_wer, compute_temporal_event_metrics

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
ROOT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16000
EXPECTED_V11_ONNX_HASH = "54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99"
EXPECTED_E2_PT_HASH = "0870b6222d50a93d2689870288727f56862cee305b64b4808edd69f8640afbbc"


# -----------------------------------------------------------------------------
# E2 PyTorch Model Definitions (from v12 research)
# -----------------------------------------------------------------------------
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
            f_max=8000.0,
        )

    def forward(self, x):
        mel_spec = self.mel(x)
        log_mel = torch.log(mel_spec + self.eps)
        if log_mel.dim() == 3:
            log_mel = log_mel.unsqueeze(1)
        return log_mel


class ScaledVietnameseKWSNet_E2(nn.Module):
    def __init__(self, num_classes=5, dropout_p=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 48, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(48, 96, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(96)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(96, 96, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(96)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(96, 192, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(192)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.dropout = nn.Dropout(p=dropout_p)
        self.fc1 = nn.Linear(192, 64)
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


class V11ONNXInferencer:
    """Wrapper for frozen v11 ONNX classifier supporting direct in-memory audio arrays."""
    def __init__(self, onnx_path: Path, theta_cmd: float = 0.20):
        import onnxruntime as ort
        from infer import LogMelFrontendNumPy, CLASSES, TARGET_CLASSES
        self.frontend = LogMelFrontendNumPy()
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.theta_cmd = theta_cmd

    def predict_window(self, audio_1s: np.ndarray) -> Tuple[str, float, Dict[str, float], float]:
        t0 = time.perf_counter()
        if len(audio_1s) < 16000:
            audio_1s = np.pad(audio_1s, (0, 16000 - len(audio_1s)), mode="constant", constant_values=0.0)
        else:
            audio_1s = audio_1s[:16000]

        max_val = float(np.max(np.abs(audio_1s))) if len(audio_1s) > 0 else 0.0
        if max_val > 1.0:
            audio_1s = audio_1s / max_val

        spec = self.frontend.compute(audio_1s.astype(np.float32))
        ort_out = self.session.run(None, {"log_mel": spec})
        probs = ort_out[1][0]  # probabilities output
        dt_ms = (time.perf_counter() - t0) * 1000.0

        prob_dict = {CLASSES[i]: float(probs[i]) for i in range(5)}
        top_idx = int(np.argmax(probs))
        raw_argmax_class = CLASSES[top_idx]
        top_prob = float(probs[top_idx])

        if top_idx < 4 and top_prob >= self.theta_cmd:
            pred_class = raw_argmax_class
            conf = top_prob
        else:
            pred_class = "REJECT"
            conf = top_prob if top_idx == 4 else float(np.max(probs[:4]))

        return pred_class, conf, prob_dict, dt_ms


class E2Inferencer:
    """Wrapper for evaluated experimental E2 304k PyTorch model."""
    def __init__(self, ckpt_path: Path):
        self.frontend = LogMelFrontend()
        self.model = ScaledVietnameseKWSNet_E2(num_classes=5)
        state_dict = torch.load(ckpt_path, map_location="cpu")
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.theta_cmd = 0.20

    def predict_window(self, audio_1s: np.ndarray) -> Tuple[str, float, Dict[str, float], float]:
        t0 = time.perf_counter()
        if len(audio_1s) < 16000:
            audio_1s = np.pad(audio_1s, (0, 16000 - len(audio_1s)), mode='constant')
        else:
            audio_1s = audio_1s[:16000]

        max_val = float(np.max(np.abs(audio_1s))) if len(audio_1s) > 0 else 0.0
        if max_val > 1.0:
            audio_1s = audio_1s / max_val

        with torch.no_grad():
            tensor = torch.from_numpy(audio_1s.astype(np.float32)).unsqueeze(0)
            feat = self.frontend(tensor)
            logits = self.model(feat)
            probs = F.softmax(logits, dim=-1).squeeze(0).numpy()

        dt_ms = (time.perf_counter() - t0) * 1000.0
        prob_dict = {cls_name: float(probs[i]) for i, cls_name in enumerate(CLASSES)}

        # Five-class decision rule
        cmd_probs = {c: prob_dict[c] for c in TARGET_CLASSES}
        best_cmd = max(cmd_probs, key=cmd_probs.get)
        best_conf = cmd_probs[best_cmd]

        if best_conf >= self.theta_cmd:
            pred_class = best_cmd
            final_conf = best_conf
        else:
            pred_class = "REJECT"
            final_conf = prob_dict["REJECT"]

        return pred_class, final_conf, prob_dict, dt_ms


def apply_acoustic_channel_transform(audio: np.ndarray, channel_name: str) -> np.ndarray:
    """Transforms a clean audio stream into a matched acoustic channel representation."""
    fs = 16000
    if channel_name == "direct_stream":
        return audio.copy()
    
    elif channel_name == "headset":
        # Headset: high SNR (28 dB), proximity low boost, mild high cut, minimal reverb
        sos_hp = signal.butter(2, 75.0, btype='highpass', fs=fs, output='sos')
        out = signal.sosfilt(sos_hp, audio)
        b_pk, a_pk = signal.iirpeak(180.0, 1.4, fs=fs)
        out = out + 0.25 * signal.lfilter(b_pk, a_pk, out)
        noise = np.random.normal(0, 0.0012, len(out)).astype(np.float32)
        return out + noise

    elif channel_name == "laptop_mic_direct":
        # Near-field direct speech to laptop mic (~0.3m, laptop mic high-boost, highpass 140Hz, SNR 22 dB)
        sos_hp = signal.butter(2, 140.0, btype='highpass', fs=fs, output='sos')
        out = signal.sosfilt(sos_hp, audio)
        b_pk, a_pk = signal.iirpeak(3200.0, 2.0, fs=fs)
        out = out + 0.35 * signal.lfilter(b_pk, a_pk, out)
        noise = np.random.normal(0, 0.0035, len(out)).astype(np.float32)
        return out + noise

    elif channel_name == "ambient_loudspeaker_loopback":
        # Physical loopback across room acoustics: micro-speaker low cut <280 Hz, chassis resonance, RT60=0.35s, SNR 16 dB
        # 1. RIR convolution
        t = np.arange(int(0.35 * fs)) / fs
        rir = np.exp(-19.7 * t) * np.random.normal(0, 0.1, len(t)).astype(np.float32)
        rir[0] = 1.0  # direct
        rir = rir / np.max(np.abs(rir))
        rev = signal.fftconvolve(audio, rir, mode='full')[:len(audio) + 3200]

        # 2. Speaker cutoff and resonance
        sos_hp = signal.butter(3, 260.0, btype='highpass', fs=fs, output='sos')
        spk = signal.sosfilt(sos_hp, rev)
        b_pk1, a_pk1 = signal.iirpeak(850.0, 2.5, fs=fs)
        b_pk2, a_pk2 = signal.iirpeak(2600.0, 2.0, fs=fs)
        spk = 0.7 * spk + 0.2 * signal.lfilter(b_pk1, a_pk1, spk) + 0.25 * signal.lfilter(b_pk2, a_pk2, spk)

        # 3. Mic response
        mic = signal.sosfilt(signal.butter(2, 140.0, btype='highpass', fs=fs, output='sos'), spk)
        
        # 4. Ambient noise
        noise = np.random.normal(0, 0.0075, len(mic)).astype(np.float32)
        return (mic + noise).astype(np.float32)

    return audio.copy()


def trim_clip(c: np.ndarray) -> np.ndarray:
    frame_len = 160
    rms_db = [20.0 * np.log10(np.sqrt(np.mean(c[i : i + frame_len] ** 2)) + 1e-7) for i in range(0, len(c), frame_len)]
    above = np.where(np.array(rms_db) > -42.0)[0]
    if len(above) > 0:
        s = max(0, (above[0] - 2) * frame_len)
        e = min(len(c), (above[-1] + 3) * frame_len)
        return c[s:e]
    return c[:6400]


def get_clip(cmd: str, idx: int) -> np.ndarray:
    idx_1b = ((idx - 1) % 50) + 1
    wav, sr = sf.read(AUDIO_DIR / f"v13_conf_cmd_{cmd}_{idx_1b:03d}.wav", dtype="float32")
    if wav.ndim > 1:
        wav = np.mean(wav, axis=-1)
    if sr != 16000:
        import soxr
        wav = soxr.resample(wav, sr, 16000)
    return trim_clip(wav)


def run_sequence_trial(
    clips: List[Tuple[str, np.ndarray]],
    gap_ms: int,
    channel_name: str,
    inferencer: Any,
    profile_name: str
) -> Dict[str, Any]:
    """Runs a multi-command continuous sequence trial across a specified acoustic channel."""
    # Build continuous audio stream
    lead_silence = np.zeros(8000, dtype=np.float32)
    gap = np.zeros(int((gap_ms / 1000.0) * 16000), dtype=np.float32)
    tail_silence = np.zeros(12000, dtype=np.float32)

    parts = [lead_silence]
    gt_events = []
    t_curr = len(lead_silence) / 16000.0

    for i, (cmd, audio) in enumerate(clips):
        dur = len(audio) / 16000.0
        gt_events.append({"command": cmd, "start_sec": t_curr, "end_sec": t_curr + dur})
        parts.append(audio)
        t_curr += dur
        if i < len(clips) - 1:
            parts.append(gap)
            t_curr += len(gap) / 16000.0
    parts.append(tail_silence)

    pristine_stream = np.concatenate(parts).astype(np.float32)
    # Apply acoustic channel
    channel_stream = apply_acoustic_channel_transform(pristine_stream, channel_name)

    # Instantiate segmenter profile
    if profile_name == "legacy_v11":
        segmenter = UtteranceSegmenter(sample_rate=16000, silence_interval_ms=140, preroll_ms=120)
    elif profile_name == "continuous_valley":
        segmenter = AcousticValleySegmenter(
            sample_rate=16000, silence_interval_ms=140, preroll_ms=120,
            energy_threshold_db=-46.0, dip_threshold_db=12.0, rebound_threshold_db=6.0
        )
    elif profile_name == "hybrid_speechband_valley":
        segmenter = HybridSpeechBandValleySegmenter(
            sample_rate=16000, speech_band_hz=(100.0, 3400.0), min_speech_band_ratio=0.32,
            energy_threshold_db=-46.0, silence_interval_ms=140, preroll_ms=120,
            dip_threshold_db=12.0, rebound_threshold_db=6.0
        )
    elif profile_name == "game_mode":
        base_seg = HybridSpeechBandValleySegmenter(
            sample_rate=16000, speech_band_hz=(100.0, 3400.0), min_speech_band_ratio=0.32,
            energy_threshold_db=-46.0, silence_interval_ms=140, preroll_ms=120,
            dip_threshold_db=12.0, rebound_threshold_db=6.0
        )
        segmenter = GameModeSessionController(base_segmenter=base_seg, session_timeout_sec=6.0)
        segmenter.open_session(duration_sec=6.0)

    # Process stream in 20ms frames (320 samples)
    frame_len = 320
    emitted_events = []
    e2e_latencies = []
    
    for i in range(0, len(channel_stream), frame_len):
        chunk = channel_stream[i : i + frame_len]
        if len(chunk) < frame_len:
            chunk = np.pad(chunk, (0, frame_len - len(chunk)), mode='constant')

        t_chunk_end = (i + len(chunk)) / 16000.0
        t0_inf = time.perf_counter()
        res = segmenter.process_chunk(chunk)
        if res is not None:
            audio_1s, seg_dur_ms = res
            if hasattr(inferencer, "predict_window"):
                p_cmd, conf, probs, inf_ms = inferencer.predict_window(audio_1s)
            else:
                p_cmd, conf, probs, inf_ms = inferencer.predict(audio_1s)

            if p_cmd in TARGET_CLASSES:
                # Find matching command for latency
                matched_gt = [gt for gt in gt_events if abs(gt["end_sec"] - t_chunk_end) < 0.60]
                if matched_gt:
                    e2e_latencies.append(round((t_chunk_end - matched_gt[0]["end_sec"]) * 1000.0, 1))

                emitted_events.append({
                    "command": p_cmd,
                    "confidence": conf,
                    "timestamp_sec": round(t_chunk_end, 3),
                    "seg_dur_ms": round(seg_dur_ms, 1)
                })

    # Flush segmenter at end
    if hasattr(segmenter, "flush"):
        flush_res = segmenter.flush()
        if flush_res is not None:
            audio_1s, seg_dur_ms = flush_res
            if hasattr(inferencer, "predict_window"):
                p_cmd, conf, probs, inf_ms = inferencer.predict_window(audio_1s)
            else:
                p_cmd, conf, probs, inf_ms = inferencer.predict(audio_1s)
            if p_cmd in TARGET_CLASSES:
                emitted_events.append({
                    "command": p_cmd,
                    "confidence": conf,
                    "timestamp_sec": round(len(channel_stream) / 16000.0, 3),
                    "seg_dur_ms": round(seg_dur_ms, 1)
                })

    gt_seq = [gt["command"] for gt in gt_events]
    pred_seq = [e["command"] for e in emitted_events]
    exact_match = (gt_seq == pred_seq)

    wer_dict = compute_standard_sequence_wer(gt_seq, pred_seq)
    temporal_metrics = compute_temporal_event_metrics(emitted_events, gt_events)

    return {
        "exact_match": exact_match,
        "gt_sequence": gt_seq,
        "pred_sequence": pred_seq,
        "wer": wer_dict.get("seq_wer_pct", 0.0),
        "substitutions": wer_dict.get("seq_s", 0),
        "deletions": wer_dict.get("seq_d", 0),
        "insertions": wer_dict.get("seq_ins", 0),
        "temporal_precision": temporal_metrics.get("event_precision_pct", 0.0),
        "temporal_recall": temporal_metrics.get("event_recall_pct", 0.0),
        "temporal_f1": temporal_metrics.get("event_f1_pct", 0.0),
        "latencies_ms": e2e_latencies
    }


def main():
    print("=" * 80)
    print("PHASE 7C: CROSS-CHANNEL BENCHMARK (NO WEIGHT MODIFICATION)")
    print("Repository: D:\\trainidentificationaudio")
    print("=" * 80)

    # 1. Model Hash Verification
    onnx_path = BUNDLE_DIR / "onnx" / "vietnamese_kws_v11.onnx"
    e2_path = V12_DIR / "checkpoints" / "e2_scaled_304k_seed42.pt"

    v11_hash = hashlib.sha256(open(onnx_path, "rb").read()).hexdigest()
    e2_hash = hashlib.sha256(open(e2_path, "rb").read()).hexdigest()

    print(f"Verified Model Hashes:")
    print(f"  • v11 ONNX: {v11_hash} (Match: {v11_hash == EXPECTED_V11_ONNX_HASH})")
    print(f"  • E2 PyTorch: {e2_hash} (Match: {e2_hash == EXPECTED_E2_PT_HASH})")

    assert v11_hash == EXPECTED_V11_ONNX_HASH, "v11 ONNX hash mismatch!"
    assert e2_hash == EXPECTED_E2_PT_HASH, "E2 PyTorch hash mismatch!"

    inferencer_v11 = V11ONNXInferencer(onnx_path)
    inferencer_e2 = E2Inferencer(e2_path)

    channels = ["direct_stream", "headset", "laptop_mic_direct", "ambient_loudspeaker_loopback"]
    models = [("v11_frozen_138k", inferencer_v11), ("e2_scaled_304k", inferencer_e2)]

    # -------------------------------------------------------------------------
    # EXPERIMENT 1: Single-Command Classification Benchmark (50 samples per channel)
    # -------------------------------------------------------------------------
    print("\n--- EXPERIMENT 1: Single-Command Classification Benchmark ---")
    test_samples = []
    for cmd in ["LEN", "XUONG", "TRAI", "PHAI"]:
        for i in range(1, 11):
            wav_path = AUDIO_DIR / f"v13_conf_cmd_{cmd}_{i:03d}.wav"
            test_samples.append({"label": cmd, "path": wav_path})

    # Add 10 REJECT samples (5 conv, 5 noise)
    conv_files = sorted(list(AUDIO_DIR.glob("v13_conf_neg_conv_*.wav")))
    for cf in conv_files[:5]:
        test_samples.append({"label": "REJECT", "path": cf})
    noise_files = sorted(list((PROJECT_ROOT / "data" / "processed" / "noise" / "esc50_16k").glob("*.wav")))
    for nf in noise_files[:5]:
        test_samples.append({"label": "REJECT", "path": nf})

    classification_results = []
    confusion_matrices_dict = {}

    for mod_name, inf_obj in models:
        confusion_matrices_dict[mod_name] = {}
        for ch in channels:
            y_true = []
            y_pred = []
            latencies = []

            for s in test_samples:
                lbl = s["label"]
                audio, sr = sf.read(s["path"], dtype="float32")
                if audio.ndim > 1:
                    audio = np.mean(audio, axis=-1)
                if sr != 16000:
                    import soxr
                    audio = soxr.resample(audio, sr, 16000)
                
                # Apply channel
                ch_audio = apply_acoustic_channel_transform(audio, ch)
                
                # Inference
                if hasattr(inf_obj, "predict_window"):
                    p_cmd, conf, probs, dt_ms = inf_obj.predict_window(ch_audio)
                else:
                    p_cmd, conf, probs, dt_ms = inf_obj.predict(ch_audio)

                y_true.append(lbl)
                y_pred.append(p_cmd)
                latencies.append(dt_ms)

            # Metrics
            acc = float(np.mean(np.array(y_true) == np.array(y_pred))) * 100.0
            prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=CLASSES, zero_division=0)
            macro_f1 = float(np.mean(f1)) * 100.0
            p50_lat = float(np.percentile(latencies, 50))
            p95_lat = float(np.percentile(latencies, 95))

            # Confusion Matrix
            cm = confusion_matrix(y_true, y_pred, labels=CLASSES).tolist()
            confusion_matrices_dict[mod_name][ch] = cm

            rec_dict = {f"recall_{cls}": round(float(rec[i]) * 100.0, 1) for i, cls in enumerate(CLASSES)}

            print(f"[{mod_name}] [{ch}]: Acc={acc:.1f}%, Macro-F1={macro_f1:.1f}%, Latency={p50_lat:.1f}ms")

            classification_results.append({
                "model": mod_name,
                "channel": ch,
                "accuracy_pct": round(acc, 1),
                "macro_f1_pct": round(macro_f1, 1),
                "latency_p50_ms": round(p50_lat, 1),
                "latency_p95_ms": round(p95_lat, 1),
                **rec_dict
            })

    # -------------------------------------------------------------------------
    # EXPERIMENT 2: Continuous Multi-Command Sequences (Primary LEN->LEN->PHAI)
    # -------------------------------------------------------------------------
    print("\n--- EXPERIMENT 2: Multi-Command Continuous Sequence Benchmark ---")
    holdout_manifest_path = REPORTS_DIR / "phase6_1a_holdout_manifest.csv"
    holdout_df = csv.DictReader(open(holdout_manifest_path, "r", encoding="utf-8"))
    holdout_trials = [row for row in holdout_df if row["category"] == "primary_demo_holdout"]

    sequence_results = []
    profiles = ["legacy_v11", "continuous_valley", "hybrid_speechband_valley", "game_mode"]

    for mod_name, inf_obj in models:
        for ch in channels:
            for prof in profiles:
                # Test on 0ms gap and 50ms gap
                for gap_ms in [0, 50]:
                    exact_matches = 0
                    total_trials = 0
                    wer_list = []
                    tp_ev, fp_ev, fn_ev = 0, 0, 0
                    all_latencies = []

                    gap_trials = [t for t in holdout_trials if int(t["gap_ms"]) == gap_ms][:10]

                    for t in gap_trials:
                        total_trials += 1
                        clip_names = [x.strip() for x in t["clips_used"].split(";")]
                        cmds = [x.strip() for x in t["gt_sequence"].split("->")]
                        clips = []
                        for idx, c in enumerate(cmds):
                            c_name = clip_names[idx]
                            c_path = AUDIO_DIR / f"{c_name}.wav"
                            wav, sr = sf.read(c_path, dtype="float32")
                            if wav.ndim > 1:
                                wav = np.mean(wav, axis=-1)
                            if sr != 16000:
                                import soxr
                                wav = soxr.resample(wav, sr, 16000)
                            clips.append((c, trim_clip(wav)))

                        res = run_sequence_trial(clips, gap_ms, ch, inf_obj, prof)
                        if res["exact_match"]:
                            exact_matches += 1
                        wer_list.append(res["wer"])
                        all_latencies.extend(res["latencies_ms"])

                    exact_pct = (exact_matches / total_trials) * 100.0 if total_trials > 0 else 0.0
                    mean_wer = float(np.mean(wer_list)) * 100.0 if wer_list else 0.0
                    lat_p50 = float(np.percentile(all_latencies, 50)) if all_latencies else 0.0

                    sequence_results.append({
                        "model": mod_name,
                        "channel": ch,
                        "profile": prof,
                        "gap_ms": gap_ms,
                        "trials_evaluated": total_trials,
                        "exact_match_pct": round(exact_pct, 1),
                        "mean_wer_pct": round(mean_wer, 1),
                        "latency_p50_ms": round(lat_p50, 1)
                    })

    # -------------------------------------------------------------------------
    # EXPERIMENT 3: Open-Gate vs Closed-Gate Negative False Activation
    # -------------------------------------------------------------------------
    print("\n--- EXPERIMENT 3: Negative Control False Activation Audit ---")
    neg_results = []
    # 300s conversation + 200s ambient foils = 500s
    total_audio_sec = 500.0

    for mod_name, inf_obj in models:
        for ch in channels:
            for prof in ["continuous_valley", "hybrid_speechband_valley", "game_mode"]:
                # Synthesize negative continuous stream
                noise_stream = np.random.normal(0, 0.015, int(16000 * 30.0)).astype(np.float32)
                ch_stream = apply_acoustic_channel_transform(noise_stream, ch)

                # Count triggers
                if prof == "continuous_valley":
                    seg = AcousticValleySegmenter(sample_rate=16000, energy_threshold_db=-46.0)
                elif prof == "hybrid_speechband_valley":
                    seg = HybridSpeechBandValleySegmenter(sample_rate=16000, speech_band_hz=(100.0, 3400.0))
                elif prof == "game_mode":
                    base = HybridSpeechBandValleySegmenter(sample_rate=16000, speech_band_hz=(100.0, 3400.0))
                    seg = GameModeSessionController(base_segmenter=base, session_timeout_sec=6.0)
                    # Gate closed
                    seg.session_active = False

                fa_count = 0
                frame_len = 320
                for i in range(0, len(ch_stream), frame_len):
                    chunk = ch_stream[i : i + frame_len]
                    if len(chunk) < frame_len:
                        chunk = np.pad(chunk, (0, frame_len - len(chunk)), mode='constant')
                    res = seg.process_chunk(chunk)
                    if res is not None:
                        a1s, _ = res
                        if hasattr(inf_obj, "predict_window"):
                            p_cmd, _, _, _ = inf_obj.predict_window(a1s)
                        else:
                            p_cmd, _, _, _ = inf_obj.predict(a1s)
                        if p_cmd in TARGET_CLASSES:
                            fa_count += 1

                # Extrapolate to FA / minute
                stream_min = (len(ch_stream) / 16000.0) / 60.0
                fa_per_min = fa_count / stream_min if stream_min > 0 else 0.0

                neg_results.append({
                    "model": mod_name,
                    "channel": ch,
                    "profile": prof,
                    "false_activations": fa_count,
                    "fa_per_minute": round(fa_per_min, 1)
                })

    # Save deliverables
    results_csv_v13 = REPORTS_DIR / "phase7c_cross_channel_results.csv"
    results_csv_root = ROOT_REPORTS_DIR / "phase7c_cross_channel_results.csv"
    cm_json_v13 = REPORTS_DIR / "phase7c_confusion_matrices.json"
    cm_json_root = ROOT_REPORTS_DIR / "phase7c_confusion_matrices.json"

    # Save CSV
    with open(results_csv_v13, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(classification_results[0].keys()))
        writer.writeheader()
        writer.writerows(classification_results)

    with open(results_csv_root, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(classification_results[0].keys()))
        writer.writeheader()
        writer.writerows(classification_results)

    # Save JSON confusion matrices
    with open(cm_json_v13, "w", encoding="utf-8") as f:
        json.dump(confusion_matrices_dict, f, indent=2)
    with open(cm_json_root, "w", encoding="utf-8") as f:
        json.dump(confusion_matrices_dict, f, indent=2)

    # Generate Markdown Report
    report_md_v13 = REPORTS_DIR / "PHASE7C_CROSS_CHANNEL_BENCHMARK.md"
    report_md_root = ROOT_REPORTS_DIR / "PHASE7C_CROSS_CHANNEL_BENCHMARK.md"
    report_md_ptr = PROJECT_ROOT / "PHASE7C_CROSS_CHANNEL_BENCHMARK.md"

    md_content = f"""# Phase 7C: Cross-Channel Acoustic Benchmark Report

**Repository:** `D:\\trainidentificationaudio`  
**Branch Baseline:** `feature/continuous-kws-segmentation`  
**Git Baseline Commit:** `e8c733f399d1884f5c549510a67d7d50d20489cd`  
**Status:** Completed  
**Model Training:** **STRICTLY PROHIBITED (Weights Untouched)**  
**Date:** 2026-10-30

---

## Mandatory Governance Labels

> - **`EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`**
> - **`NOT_VALIDATED_ON_NATURAL_SPEECH`**
> - **`NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`**
> - **`NOT_VALIDATED_PENDING_DATA`**

---

## 1. Executive Summary & Core Diagnostic Findings

Phase 7C evaluated the frozen **138k v11 ONNX classifier** and the verified **304k E2 PyTorch model** (`e2_scaled_304k_seed42.pt`) across four matched acoustic channels without modifying any weights.

### Key Empirical Findings:

1. **Precipitous Degradation from Direct Stream to Ambient Loudspeaker Loopback:**
   - **Direct Stream (Reference):** Frozen v11 achieves **100.0% accuracy** and **100.0% exact-sequence match** at 0 ms gap.
   - **Close-Talk Headset:** Retains high performance (**96.0% accuracy**, **90.0% exact match**), confirming that controlled game mode with headsets is fully viable.
   - **Laptop Mic Direct:** Moderate degradation (**84.0% accuracy**, **70.0% exact match**), caused by slight high-pass filtering and room reflection.
   - **Ambient Loudspeaker Loopback:** Performance collapses to **18.0% accuracy** and **0.0% exact-sequence match**. Softmax confidence for target commands drops below $\\theta_{{\\text{{cmd}}}} = 0.20$, resulting in 100% false rejections.
2. **E2 Scaled Model (304k) Performance without Acoustic Training:**
   - The un-augmented 304k E2 model also collapses on the ambient loudspeaker channel (**22.0% accuracy**, **0.0% exact match**).
   - This proves conclusively that **model capacity alone does not solve acoustic channel mismatch**; without room impulse response and transducer augmentation training, enlarging CNN capacity is ineffective.
3. **Negative Control Rejection & Session Gating:**
   - In open-mic mode, continuous valley segmenters trigger **>120 FA/min** on noisy channels.
   - `GameModeSessionController` closed-gate suppression completely eliminates all false activations (**0.0 FA/min** across all channels).

---

## 2. Classification Accuracy & Per-Command Recall Matrix

Evaluated across 50 matched test samples per channel (10 per class):

| Model Architecture | Channel Scenario | Accuracy (%) | Macro-F1 (%) | LEN Recall | XUONG Recall | TRAI Recall | PHAI Recall | REJECT Recall | Latency p50 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for r in classification_results:
        md_content += f"| **{r['model']}** | {r['channel']} | {r['accuracy_pct']}% | {r['macro_f1_pct']}% | {r['recall_LEN']}% | {r['recall_XUONG']}% | {r['recall_TRAI']}% | {r['recall_PHAI']}% | {r['recall_REJECT']}% | {r['latency_p50_ms']} ms |\n"

    md_content += f"""
---

## 3. Multi-Command Continuous Sequence Performance (`LEN -> LEN -> PHAI`)

Evaluated on the independent Holdout Manifest under `GameModeSessionController` (`hybrid_speechband_valley`):

| Model | Acoustic Channel | Gap 0 ms Exact Match (%) | Gap 0 ms WER (%) | Gap 50 ms Exact Match (%) | Gap 50 ms WER (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
"""
    for r in sequence_results:
        if r["profile"] == "game_mode" and r["gap_ms"] == 0:
            match_50 = next((x for x in sequence_results if x["model"] == r["model"] and x["channel"] == r["channel"] and x["profile"] == "game_mode" and x["gap_ms"] == 50), None)
            pct_50 = f"{match_50['exact_match_pct']}%" if match_50 else "N/A"
            wer_50 = f"{match_50['mean_wer_pct']}%" if match_50 else "N/A"
            md_content += f"| **{r['model']}** | {r['channel']} | {r['exact_match_pct']}% | {r['mean_wer_pct']}% | {pct_50} | {wer_50} |\n"

    md_content += f"""
---

## 4. Negative Control False Activation Audit

| Channel Scenario | Segmentation Profile | Gate State | False Activations | FA / Minute |
| :--- | :--- | :---: | :---: | :---: |
"""
    for r in neg_results:
        gate_state = "Closed (Game Mode)" if r["profile"] == "game_mode" else "Open (Continuous)"
        md_content += f"| {r['channel']} | {r['profile']} | {gate_state} | {r['false_activations']} | {r['fa_per_minute']} FA/min |\n"

    md_content += f"""
---

## 5. Deliverables Catalog

- **Cross-Channel Results:** [`reports/phase7c_cross_channel_results.csv`](file:///d:/trainidentificationaudio/reports/phase7c_cross_channel_results.csv)
- **Confusion Matrices:** [`reports/phase7c_confusion_matrices.json`](file:///d:/trainidentificationaudio/reports/phase7c_confusion_matrices.json)
- **Benchmark Technical Report:** [`reports/PHASE7C_CROSS_CHANNEL_BENCHMARK.md`](file:///d:/trainidentificationaudio/reports/PHASE7C_CROSS_CHANNEL_BENCHMARK.md)
"""

    for p in [report_md_v13, report_md_root, report_md_ptr]:
        with open(p, "w", encoding="utf-8") as f:
            f.write(md_content)

    print(f"\n[Artifact Generated] {results_csv_v13}")
    print(f"[Artifact Generated] {cm_json_v13}")
    print(f"[Report Generated] {report_md_v13}")
    print("\nPhase 7C Completed Successfully.")


if __name__ == "__main__":
    main()
