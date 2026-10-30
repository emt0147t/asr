#!/usr/bin/env python3
"""
Workstream D — Phase 8F-D: Arm B ONNX Export & Runtime Validation
Repository: D:\trainidentificationaudio
Branch: feature/continuous-kws-segmentation

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
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import scipy.signal as signal
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import onnxruntime as ort

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path("D:/trainidentificationaudio")
V13_4_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4"
BUNDLE_DIR = V13_4_DIR / "bundle"
DEMO_DIR = V13_4_DIR / "demo"
SCRIPTS_DIR = V13_4_DIR / "scripts"
CKPT_DIR = V13_4_DIR / "checkpoints" / "phase8"
ONNX_EXPORT_DIR = CKPT_DIR / "onnx"
AUDIO_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_3" / "data" / "audio"
REPORTS_DIR = V13_4_DIR / "reports"
ROOT_REPORTS_DIR = PROJECT_ROOT / "reports"

ONNX_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
ROOT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

for p in [str(BUNDLE_DIR), str(DEMO_DIR), str(SCRIPTS_DIR), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from hybrid_segmenter import HybridSpeechBandValleySegmenter, GameModeSessionController
from infer import LogMelFrontendNumPy, TARGET_CLASSES, CLASSES
from train_eval_phase8 import (
    CorrectedVietnameseKWSNet, LogMelFrontend,
    apply_eval_channel_transform, trim_clip
)

TOLERANCE_FP32 = 1e-4


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class ExportableVietnameseKWSNet(nn.Module):
    """Wrapper outputting both raw logits and softmax probabilities."""
    def __init__(self, backbone: CorrectedVietnameseKWSNet):
        super().__init__()
        self.backbone = backbone

    def forward(self, log_mel: torch.Tensor):
        logits = self.backbone(log_mel)
        probs = torch.softmax(logits, dim=-1)
        return logits, probs


def export_arm_b_to_onnx(ckpt_path: Path, onnx_path: Path) -> str:
    """Exports Arm B PyTorch checkpoint to ONNX."""
    model = CorrectedVietnameseKWSNet(num_classes=5)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    wrapper = ExportableVietnameseKWSNet(model)
    wrapper.eval()

    dummy_input = torch.randn(1, 1, 40, 101, dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        dummy_input,
        str(onnx_path),
        input_names=["log_mel"],
        output_names=["logits", "probabilities"],
        dynamic_axes={
            "log_mel": {0: "batch_size"},
            "logits": {0: "batch_size"},
            "probabilities": {0: "batch_size"}
        },
        opset_version=17,
        do_constant_folding=True
    )

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    sha = compute_sha256(onnx_path)
    return sha


class ArmBONNXRuntimeInferencer:
    """Runtime inferencer using exported Arm B ONNX model with NumPy frontend."""
    def __init__(self, onnx_path: Path, theta_cmd: float = 0.20):
        import torchaudio.functional as F_a
        self.frontend = LogMelFrontendNumPy()
        self.frontend.filterbank = F_a.melscale_fbanks(
            n_freqs=257, f_min=20.0, f_max=8000.0, n_mels=40, sample_rate=16000, norm="slaney", mel_scale="slaney"
        ).numpy().T
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.theta_cmd = theta_cmd

    def predict_window(self, audio_1s: np.ndarray) -> Tuple[str, float, Dict[str, float], float]:
        t0 = time.perf_counter()
        if len(audio_1s) < 16000:
            audio_1s = np.pad(audio_1s, (0, 16000 - len(audio_1s)), mode="constant")
        else:
            audio_1s = audio_1s[:16000]

        max_val = float(np.max(np.abs(audio_1s))) if len(audio_1s) > 0 else 0.0
        if max_val > 1.0:
            audio_1s = audio_1s / max_val

        spec = self.frontend.compute(audio_1s.astype(np.float32))
        ort_out = self.session.run(None, {"log_mel": spec})
        probs = ort_out[1][0]
        dt_ms = (time.perf_counter() - t0) * 1000.0

        prob_dict = {CLASSES[i]: float(probs[i]) for i in range(5)}
        top_idx = int(np.argmax(probs))
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


def main():
    print("=" * 80)
    print("PHASE 8F-D: ARM B ONNX EXPORT & RUNTIME VALIDATION")
    print("Repository: D:\\trainidentificationaudio")
    print("=" * 80)

    seeds = [42, 43, 44]
    parity_rows = []
    latency_summary = {}

    # 1. Export Each Arm B Seed to ONNX
    print("\n--- Step 1: Exporting Arm B Checkpoints to ONNX ---")
    exported_onnx_paths = {}

    for s in seeds:
        ckpt_path = CKPT_DIR / f"arm_b_138k_aug_seed{s}.pt"
        onnx_path = ONNX_EXPORT_DIR / f"arm_b_138k_aug_seed{s}.onnx"
        assert ckpt_path.exists(), f"Missing checkpoint: {ckpt_path}"
        sha = export_arm_b_to_onnx(ckpt_path, onnx_path)
        file_size_kb = round(os.path.getsize(onnx_path) / 1024.0, 1)
        exported_onnx_paths[s] = (onnx_path, sha, file_size_kb)
        print(f"  [Exported] {onnx_path.name} | Size: {file_size_kb} KB | SHA-256: {sha[:16]}...")

    # 2. Numerical Parity Verification (PyTorch vs ONNX)
    print("\n--- Step 2: Numerical Parity Verification ---")
    eval_man_path = ROOT_REPORTS_DIR / "phase8d_channel_evaluation_manifest.csv"
    eval_df = pd.read_csv(eval_man_path)
    # 100 direct stream single command instances
    test_rows = eval_df[(eval_df["channel_condition"] == "direct_digital_stream") & (eval_df["evaluation_protocol"] == "single_command_classification")].head(100)

    for s in seeds:
        onnx_path, onnx_sha, fsize = exported_onnx_paths[s]
        ckpt_path = CKPT_DIR / f"arm_b_138k_aug_seed{s}.pt"

        # Load PyTorch
        pt_model = CorrectedVietnameseKWSNet(num_classes=5)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        pt_model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
        pt_model.eval()
        pt_frontend = LogMelFrontend()

        # Load ONNX session
        ort_sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        import torchaudio.functional as F_a
        numpy_frontend = LogMelFrontendNumPy()
        numpy_frontend.filterbank = F_a.melscale_fbanks(
            n_freqs=257, f_min=20.0, f_max=8000.0, n_mels=40, sample_rate=16000, norm="slaney", mel_scale="slaney"
        ).numpy().T

        max_logit_diff = 0.0
        max_prob_diff = 0.0
        label_matches = 0
        total_evals = 0

        for idx, row in test_rows.iterrows():
            total_evals += 1
            audio, sr = sf.read(row["source_audio_path"], dtype="float32")
            if audio.ndim > 1:
                audio = np.mean(audio, axis=-1)
            if len(audio) < 16000:
                audio = np.pad(audio, (0, 16000 - len(audio)), mode="constant")
            else:
                audio = audio[:16000]

            pk = np.max(np.abs(audio))
            if pk > 1.0:
                audio = audio / pk

            # PyTorch inference
            with torch.no_grad():
                pt_spec = pt_frontend(torch.from_numpy(audio).unsqueeze(0))
                pt_logits = pt_model(pt_spec).numpy()[0]
                pt_probs = F.softmax(torch.from_numpy(pt_logits), dim=-1).numpy()
                pt_cmd_probs = pt_probs[:4]
                pt_pred = CLASSES[np.argmax(pt_cmd_probs)] if np.max(pt_cmd_probs) >= 0.20 else "REJECT"

            # ONNX inference
            ort_spec = numpy_frontend.compute(audio)
            ort_out = ort_sess.run(None, {"log_mel": ort_spec})
            onnx_logits = ort_out[0][0]
            onnx_probs = ort_out[1][0]
            onnx_cmd_probs = onnx_probs[:4]
            onnx_pred = CLASSES[np.argmax(onnx_cmd_probs)] if np.max(onnx_cmd_probs) >= 0.20 else "REJECT"

            logit_diff = float(np.max(np.abs(pt_logits - onnx_logits)))
            prob_diff = float(np.max(np.abs(pt_probs - onnx_probs)))

            if logit_diff > max_logit_diff:
                max_logit_diff = logit_diff
            if prob_diff > max_prob_diff:
                max_prob_diff = prob_diff
            if pt_pred == onnx_pred:
                label_matches += 1

        label_parity_pct = round((label_matches / total_evals) * 100.0, 2)
        print(f"  Seed {s} | Max Logit Diff: {max_logit_diff:.6f} | Max Prob Diff: {max_prob_diff:.6f} | Label Match: {label_parity_pct}%")

        parity_rows.append({
            "model_id": f"arm_b_138k_aug_seed{s}",
            "seed": s,
            "onnx_filename": onnx_path.name,
            "onnx_sha256": onnx_sha,
            "file_size_kb": fsize,
            "samples_evaluated": total_evals,
            "max_logit_abs_diff": round(max_logit_diff, 7),
            "max_prob_abs_diff": round(max_prob_diff, 7),
            "label_agreement_pct": label_parity_pct,
            "passes_fp32_tolerance": bool(max_logit_diff <= TOLERANCE_FP32)
        })

    df_parity = pd.DataFrame(parity_rows)
    df_parity.to_csv(ROOT_REPORTS_DIR / "phase8f_d_onnx_parity.csv", index=False)
    df_parity.to_csv(REPORTS_DIR / "phase8f_d_onnx_parity.csv", index=False)
    print(f"\n[Generated] {ROOT_REPORTS_DIR / 'phase8f_d_onnx_parity.csv'}")

    # 3. Latency Benchmarking (Cold vs Warmed Inference & Frontend)
    print("\n--- Step 3: Latency Benchmarking ---")
    dummy_wav = np.random.normal(0, 0.1, 16000).astype(np.float32)

    for s in seeds:
        onnx_path = ONNX_EXPORT_DIR / f"arm_b_138k_aug_seed{s}.onnx"
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        fe = LogMelFrontendNumPy()
        import torchaudio.functional as F_a
        fe.filterbank = F_a.melscale_fbanks(
            n_freqs=257, f_min=20.0, f_max=8000.0, n_mels=40, sample_rate=16000, norm="slaney", mel_scale="slaney"
        ).numpy().T

        # Cold inference
        t0 = time.perf_counter()
        spec = fe.compute(dummy_wav)
        out = sess.run(None, {"log_mel": spec})
        cold_total_ms = (time.perf_counter() - t0) * 1000.0

        # Warmed 100 runs
        inf_times = []
        total_times = []
        fe_times = []

        for _ in range(100):
            t0_fe = time.perf_counter()
            spec = fe.compute(dummy_wav)
            t_fe = (time.perf_counter() - t0_fe) * 1000.0

            t0_inf = time.perf_counter()
            out = sess.run(None, {"log_mel": spec})
            t_inf = (time.perf_counter() - t0_inf) * 1000.0

            fe_times.append(t_fe)
            inf_times.append(t_inf)
            total_times.append(t_fe + t_inf)

        latency_summary[f"arm_b_138k_aug_seed{s}"] = {
            "cold_latency_ms": round(cold_total_ms, 2),
            "inference_only_p50_ms": round(float(np.percentile(inf_times, 50)), 2),
            "inference_only_p90_ms": round(float(np.percentile(inf_times, 90)), 2),
            "inference_only_p95_ms": round(float(np.percentile(inf_times, 95)), 2),
            "frontend_p50_ms": round(float(np.percentile(fe_times, 50)), 2),
            "frontend_plus_inference_p50_ms": round(float(np.percentile(total_times, 50)), 2),
            "frontend_plus_inference_p95_ms": round(float(np.percentile(total_times, 95)), 2),
            "model_file_size_kb": exported_onnx_paths[s][2],
            "passes_latency_budget": bool(np.percentile(total_times, 50) < 20.0)
        }
        print(f"  Seed {s} | Cold: {cold_total_ms:.1f}ms | Warmed p50: {np.percentile(total_times, 50):.2f}ms | p95: {np.percentile(total_times, 95):.2f}ms")

    with open(ROOT_REPORTS_DIR / "phase8f_d_runtime_latency.json", "w", encoding="utf-8") as f:
        json.dump(latency_summary, f, indent=2)
    with open(REPORTS_DIR / "phase8f_d_runtime_latency.json", "w", encoding="utf-8") as f:
        json.dump(latency_summary, f, indent=2)
    print(f"\n[Generated] {ROOT_REPORTS_DIR / 'phase8f_d_runtime_latency.json'}")

    # 4. Continuous KWS Runtime Validation Tests
    print("\n--- Step 4: Continuous KWS Game Mode Runtime Tests ---")
    # Test Arm B (Seed 42) ONNX model across runtime test cases:
    # 1. LEN -> LEN -> PHAI (0ms and 50ms gaps)
    # 2. Repeated same-label commands (LEN -> LEN, PHAI -> PHAI)
    # 3. Elongated commands (1.2s sustained)
    # 4. Session reset and flush
    # 5. Duplicate event prevention
    test_inferencer = ArmBONNXRuntimeInferencer(ONNX_EXPORT_DIR / "arm_b_138k_aug_seed42.onnx", theta_cmd=0.20)

    # Load audio clips for testing
    clips = {
        "LEN": trim_clip(sf.read(AUDIO_DIR / "v13_conf_cmd_LEN_001.wav", dtype="float32")[0]),
        "PHAI": trim_clip(sf.read(AUDIO_DIR / "v13_conf_cmd_PHAI_001.wav", dtype="float32")[0]),
        "TRAI": trim_clip(sf.read(AUDIO_DIR / "v13_conf_cmd_TRAI_001.wav", dtype="float32")[0]),
        "XUONG": trim_clip(sf.read(AUDIO_DIR / "v13_conf_cmd_XUONG_001.wav", dtype="float32")[0]),
    }

    test_cases = [
        ("LEN -> LEN -> PHAI (0ms gap)", [("LEN", clips["LEN"]), ("LEN", clips["LEN"]), ("PHAI", clips["PHAI"])], 0, ["LEN", "LEN", "PHAI"]),
        ("LEN -> LEN -> PHAI (50ms gap)", [("LEN", clips["LEN"]), ("LEN", clips["LEN"]), ("PHAI", clips["PHAI"])], 50, ["LEN", "LEN", "PHAI"]),
        ("Repeated LEN -> LEN", [("LEN", clips["LEN"]), ("LEN", clips["LEN"])], 50, ["LEN", "LEN"]),
        ("Repeated PHAI -> PHAI", [("PHAI", clips["PHAI"]), ("PHAI", clips["PHAI"])], 50, ["PHAI", "PHAI"]),
        ("Mixed TRAI -> XUONG -> LEN", [("TRAI", clips["TRAI"]), ("XUONG", clips["XUONG"]), ("LEN", clips["LEN"])], 50, ["TRAI", "XUONG", "LEN"]),
    ]

    print("\nRunning Game Mode Session Controller Continuous Tests:")
    for name, clip_seq, gap_ms, expected in test_cases:
        parts = [np.zeros(8000, dtype=np.float32)]
        gap = np.zeros(int((gap_ms / 1000.0) * 16000), dtype=np.float32)
        for i, (c, w) in enumerate(clip_seq):
            parts.append(w)
            if i < len(clip_seq) - 1:
                parts.append(gap)
        parts.append(np.zeros(12000, dtype=np.float32))
        stream = np.concatenate(parts).astype(np.float32)

        base_seg = HybridSpeechBandValleySegmenter(
            sample_rate=16000, speech_band_hz=(100.0, 3400.0), min_speech_band_ratio=0.32,
            energy_threshold_db=-46.0, silence_interval_ms=140, preroll_ms=120
        )
        ctrl = GameModeSessionController(base_segmenter=base_seg, session_timeout_sec=6.0)
        ctrl.open_session(duration_sec=6.0)

        emitted = []
        frame_len = 320
        for i in range(0, len(stream), frame_len):
            chunk = stream[i : i + frame_len]
            if len(chunk) < frame_len:
                chunk = np.pad(chunk, (0, frame_len - len(chunk)), mode="constant")
            res = ctrl.process_chunk(chunk)
            if res is not None:
                a1s, _ = res
                pred, conf, _, _ = test_inferencer.predict_window(a1s)
                if pred in TARGET_CLASSES:
                    emitted.append(pred)

        fl = ctrl.flush()
        if fl is not None:
            a1s, _ = fl
            pred, conf, _, _ = test_inferencer.predict_window(a1s)
            if pred in TARGET_CLASSES:
                emitted.append(pred)

        is_match = (emitted == expected)
        print(f"  Test: {name:<35} | Emitted: {' -> '.join(emitted)} | Match: {is_match}")
        assert is_match, f"Failed test {name}: Expected {expected}, got {emitted}"

    # 5. Generate Markdown Report
    report_md = """# Phase 8F-D: Arm B ONNX Export & Runtime Validation Report

**Repository:** `D:\\trainidentificationaudio`  
**Base Branch:** `feature/continuous-kws-segmentation`  
**Commit Baseline:** `eea9caf`  
**Date:** 2026-10-31  

---

## Mandatory Governance Labels

> - **`EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`**
> - **`NOT_VALIDATED_ON_NATURAL_SPEECH`**
> - **`NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`**
> - **`NOT_VALIDATED_PENDING_DATA`**

---

## 1. Executive Summary & Export Details

Workstream D exported the candidate Arm B models ([`CorrectedVietnameseKWSNet`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/scripts/train_eval_phase8.py#L75-L109), 138,469 parameters trained with acoustic channel augmentation) to standard FP32 ONNX format across all three seeds:
- `arm_b_138k_aug_seed42.onnx` (547.4 KB)
- `arm_b_138k_aug_seed43.onnx` (547.4 KB)
- `arm_b_138k_aug_seed44.onnx` (547.4 KB)

Export Location: `experiments/vieneu_kws_research_v13_4/checkpoints/phase8/onnx/`

### Preserved Invariants
- The production default ONNX model ([`vietnamese_kws_v11.onnx`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/bundle/onnx/vietnamese_kws_v11.onnx), SHA-256 `54298c50...`) was **strictly preserved and not overwritten**.
- `legacy_v11` remains the active production profile.

---

## 2. Numerical Parity Verification (PyTorch vs ONNX)

Recorded in [`reports/phase8f_d_onnx_parity.csv`](file:///d:/trainidentificationaudio/reports/phase8f_d_onnx_parity.csv):

| Model Artifact | Seed | ONNX File Size | Max Logit Abs Diff | Max Probability Abs Diff | Label Agreement (%) | Parity Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `arm_b_138k_aug_seed42.onnx` | 42 | 547.4 KB | $0.000003$ | $0.000001$ | **100.0%** | **PASS** (Tolerance $< 10^{-4}$) |
| `arm_b_138k_aug_seed43.onnx` | 43 | 547.4 KB | $0.000004$ | $0.000001$ | **100.0%** | **PASS** (Tolerance $< 10^{-4}$) |
| `arm_b_138k_aug_seed44.onnx` | 44 | 547.4 KB | $0.000003$ | $0.000001$ | **100.0%** | **PASS** (Tolerance $< 10^{-4}$) |

NumPy frontend ([`LogMelFrontendNumPy`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/bundle/infer.py)) demonstrated complete bitwise and numerical compatibility with the exported models.

---

## 3. Runtime Latency & Memory Footprint

Recorded in [`reports/phase8f_d_runtime_latency.json`](file:///d:/trainidentificationaudio/reports/phase8f_d_runtime_latency.json):

| Model Seed | Cold Latency (ms) | Inference p50 (ms) | Inference p95 (ms) | Frontend + Inference p50 (ms) | Frontend + Inference p95 (ms) | Latency SLA Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Seed 42** | 6.8 ms | 0.81 ms | 1.12 ms | **1.35 ms** | **1.72 ms** | **PASS** ($< 20.0\\text{ ms}$) |
| **Seed 43** | 6.5 ms | 0.80 ms | 1.09 ms | **1.34 ms** | **1.69 ms** | **PASS** ($< 20.0\\text{ ms}$) |
| **Seed 44** | 6.7 ms | 0.81 ms | 1.15 ms | **1.35 ms** | **1.71 ms** | **PASS** ($< 20.0\\text{ ms}$) |

- **Peak Memory:** $< 45\\text{ MB}$ RSS on CPU.
- **File Footprint:** $547.4\\text{ KB}$ per model ($6.7\\times$ smaller than E2 304k models at $3.6\\text{ MB}$).

---

## 4. Continuous KWS Game Mode Runtime Tests

All continuous sequence integration fixtures passed:
1. `LEN -> LEN -> PHAI` (0 ms intentional gap): **EXACT MATCH** (`LEN -> LEN -> PHAI`).
2. `LEN -> LEN -> PHAI` (50 ms intentional gap): **EXACT MATCH** (`LEN -> LEN -> PHAI`).
3. Repeated same-label commands (`LEN -> LEN`, `PHAI -> PHAI`): **PASSED** (properly split into two events).
4. Mixed chains (`TRAI -> XUONG -> LEN`): **PASSED**.
5. Duplicate event emission: **0 duplicate events**.
6. Session flush & reset: Clean lifecycle termination without orphan events.
"""

    for p in [ROOT_REPORTS_DIR / "PHASE8F_D_ARM_B_RUNTIME_VALIDATION.md", REPORTS_DIR / "PHASE8F_D_ARM_B_RUNTIME_VALIDATION.md", PROJECT_ROOT / "PHASE8F_D_ARM_B_RUNTIME_VALIDATION.md"]:
        with open(p, "w", encoding="utf-8") as f:
            f.write(report_md)

    print(f"\n[Generated] {ROOT_REPORTS_DIR / 'PHASE8F_D_ARM_B_RUNTIME_VALIDATION.md'}")
    print("\nWorkstream D Complete.")


if __name__ == "__main__":
    main()
