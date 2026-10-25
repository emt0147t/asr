"""
Reproducible ONNX Export and Numerical Runtime-Parity Audit for Vietnamese KWS.
Primary Candidate Model: V4 Arm B (arm_b_onset_no_resample.pt)
Governance Invariant: HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED.
Operates exclusively on synthetic acoustic fixtures and synthetic tensors.
"""

import sys
import os
import time
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple

if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import onnxruntime as ort

from experiments.synthetic_kws_v3.models_v3 import VietnameseTCResNet
from src.preprocessing.feature_extractor import LogMelFeatureExtractor
from experiments.synthetic_kws_demo.classifier import AudioValidator, CLASS_LABELS


EXPECTED_CHECKPOINT_SHA256 = "6A0FD55CD6F16F5B683951127076EEC72506C31EE670F896AAEACE5981CF1D96"
CHECKPOINT_PATH = Path("experiments/synthetic_kws_v4/checkpoints/arm_b_onset_no_resample.pt")
EXPORT_DIR = Path("experiments/synthetic_kws_demo/export")
ONNX_FILENAME = "arm_b_tcresnet14.onnx"
ONNX_PATH = EXPORT_DIR / ONNX_FILENAME
CONFIG_PATH = EXPORT_DIR / "model_config.json"
REPORT_PATH = EXPORT_DIR / "parity_report.json"


class ExportableTCResNet(nn.Module):
    """
    Export wrapper for VietnameseTCResNet.
    Accepts Log-Mel feature tensor (Batch, 1, 40, 101) or (Batch, 40, 101).
    Returns both uncalibrated logits and normalized class probabilities.
    """
    def __init__(self, base_model: VietnameseTCResNet):
        super().__init__()
        self.base_model = base_model

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.base_model(x)  # (Batch, 6)
        probabilities = F.softmax(logits, dim=-1)  # (Batch, 6)
        return logits, probabilities


def compute_sha256(filepath: Path) -> str:
    return hashlib.sha256(filepath.read_bytes()).hexdigest().upper()


def verify_checkpoint() -> str:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Source checkpoint not found at: {CHECKPOINT_PATH}")
    actual_sha = compute_sha256(CHECKPOINT_PATH)
    if actual_sha != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(
            f"Checkpoint SHA-256 verification failed!\n"
            f"Expected: {EXPECTED_CHECKPOINT_SHA256}\n"
            f"Actual:   {actual_sha}"
        )
    return actual_sha


def export_model_to_onnx(opset_version: int = 17) -> Tuple[str, Dict[str, Any]]:
    print("=" * 72)
    print("STEP 1: Freezing and Verifying Source Model Checkpoint...")
    ckpt_sha = verify_checkpoint()
    print(f"  Checkpoint verified: {CHECKPOINT_PATH}")
    print(f"  SHA-256: {ckpt_sha}")

    # Instantiate PyTorch model and load weights
    base_model = VietnameseTCResNet(num_classes=6, dropout_p=0.0)
    ckpt_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state_dict = ckpt_dict["model_state_dict"] if "model_state_dict" in ckpt_dict else ckpt_dict
    base_model.load_state_dict(state_dict)
    base_model.eval()

    export_model = ExportableTCResNet(base_model)
    export_model.eval()

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.zeros(1, 1, 40, 101, dtype=torch.float32)

    print(f"STEP 2: Exporting PyTorch model to ONNX (opset {opset_version})...")
    torch.onnx.export(
        export_model,
        dummy_input,
        str(ONNX_PATH),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["log_mel"],
        output_names=["logits", "probabilities"],
        dynamic_axes={
            "log_mel": {0: "batch_size"},
            "logits": {0: "batch_size"},
            "probabilities": {0: "batch_size"},
        }
    )

    # Validate ONNX model with checker
    onnx_proto = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(onnx_proto)
    onnx_sha = compute_sha256(ONNX_PATH)
    onnx_file_size = ONNX_PATH.stat().st_size

    print(f"  ONNX Export successful: {ONNX_PATH}")
    print(f"  ONNX File Size: {onnx_file_size} bytes ({onnx_file_size / 1024:.2f} KB)")
    print(f"  ONNX SHA-256: {onnx_sha}")

    # Build model configuration metadata
    model_config = {
        "model_name": "VietnameseTCResNet-14",
        "description": "Six-class keyword spotting model for Vietnamese navigation commands.",
        "governance": "HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED. Trained on synthetic audio.",
        "source_checkpoint": {
            "path": str(CHECKPOINT_PATH),
            "sha256": ckpt_sha,
            "architecture": "VietnameseTCResNet (TC-ResNet-14)",
            "num_classes": 6,
            "dropout_p": 0.0,
            "training_regime": "V4 Arm B (onset-aware acoustic features, native left-aligned, no resample)"
        },
        "onnx_export": {
            "filename": ONNX_FILENAME,
            "path": str(ONNX_PATH),
            "sha256": onnx_sha,
            "file_size_bytes": onnx_file_size,
            "opset_version": opset_version,
            "producer_name": "PyTorch ONNX Exporter",
            "torch_version": torch.__version__,
            "onnx_version": onnx.__version__,
            "onnxruntime_version": ort.__version__,
            "inputs": [
                {
                    "name": "log_mel",
                    "shape": ["batch_size", 1, 40, 101],
                    "dtype": "float32",
                    "description": "Normalized Log-Mel spectrogram computed from 16kHz mono audio (1.0s)."
                }
            ],
            "outputs": [
                {
                    "name": "logits",
                    "shape": ["batch_size", 6],
                    "dtype": "float32",
                    "description": "Uncalibrated linear projection logits for the 6 classes."
                },
                {
                    "name": "probabilities",
                    "shape": ["batch_size", 6],
                    "dtype": "float32",
                    "description": "Softmax class probabilities normalized across the 6 classes."
                }
            ],
            "classes": {str(i): label for i, label in enumerate(CLASS_LABELS)}
        },
        "mobile_preprocessing_contract": {
            "required_sample_rate": 16000,
            "required_channels": 1,
            "required_samples": 16000,
            "duration_sec": 1.0,
            "n_fft": 512,
            "win_length": 400,
            "hop_length": 160,
            "n_mels": 40,
            "f_min": 20.0,
            "f_max": 8000.0,
            "window_type": "Hamming (or Hann as verified)",
            "mel_scale": "Slaney",
            "mel_norm": "Slaney (area normalized)",
            "log_transform": "ln(mel + 1e-6)",
            "expected_tensor_shape": [1, 1, 40, 101],
            "note": "Mobile runtimes must independently replicate Log-Mel extraction before feeding this model."
        }
    }

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2)
    print(f"  Configuration saved to: {CONFIG_PATH}")

    return onnx_sha, model_config


def run_runtime_parity_audit(model_config: Dict[str, Any]) -> Dict[str, Any]:
    print("\n" + "=" * 72)
    print("STEP 3: Running Runtime Parity Audit...")

    # Load PyTorch model
    base_model = VietnameseTCResNet(num_classes=6, dropout_p=0.0)
    ckpt_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state_dict = ckpt_dict["model_state_dict"] if "model_state_dict" in ckpt_dict else ckpt_dict
    base_model.load_state_dict(state_dict)
    base_model.eval()
    pt_model = ExportableTCResNet(base_model)
    pt_model.eval()

    # Load ONNX Runtime session
    ort_opts = ort.SessionOptions()
    ort_opts.inter_op_num_threads = 1
    ort_opts.intra_op_num_threads = 1
    ort_session = ort.InferenceSession(str(ONNX_PATH), ort_opts, providers=["CPUExecutionProvider"])

    # Load canonical feature extractor
    extractor = LogMelFeatureExtractor(sample_rate=16000, n_mels=40)
    extractor.eval()

    audit_results: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "parity_threshold_criterion": {
            "max_abs_prob_diff_threshold": 1e-4,
            "class_agreement_threshold": 1.0
        },
        "cohort_1_demo_fixtures": {},
        "cohort_2_fresh_holdout_300": {},
        "cohort_3_deterministic_synthetic_tensors_50": {},
        "latency_benchmark": {},
        "summary": {}
    }

    # ------------------------------------------------------------------------
    # Cohort 1: 6 Synthetic Demonstration Fixtures
    # ------------------------------------------------------------------------
    print("\n--- Auditing Cohort 1: 6 Synthetic Demo Fixtures ---")
    fixtures_dir = Path("experiments/synthetic_kws_demo/fixtures")
    demo_files = [
        ("demo_01_trai.wav", "TRAI"),
        ("demo_02_phai.wav", "PHAI"),
        ("demo_03_len.wav", "LEN"),
        ("demo_04_xuong.wav", "XUONG"),
        ("demo_05_unknown.wav", "UNKNOWN"),
        ("demo_06_silence.wav", "SILENCE"),
    ]

    demo_comparisons = []
    cohort1_max_diff = 0.0
    cohort1_sum_diff = 0.0
    cohort1_agreements = 0

    for fname, intended_label in demo_files:
        wav_p = fixtures_dir / fname
        audio, sr = AudioValidator.validate_and_load(wav_p)
        audio_t = torch.from_numpy(audio).float()
        with torch.no_grad():
            log_mel_t = extractor(audio_t)  # (1, 1, 40, 101)
            pt_logits, pt_probs = pt_model(log_mel_t)

        pt_logits_np = pt_logits.cpu().numpy()[0]
        pt_probs_np = pt_probs.cpu().numpy()[0]
        pt_pred_cid = int(np.argmax(pt_probs_np))

        # ONNX Runtime inference
        ort_inputs = {"log_mel": log_mel_t.cpu().numpy()}
        ort_logits_np, ort_probs_np = ort_session.run(None, ort_inputs)
        ort_logits_vec = ort_logits_np[0]
        ort_probs_vec = ort_probs_np[0]
        ort_pred_cid = int(np.argmax(ort_probs_vec))

        # Parity metrics
        abs_prob_diffs = np.abs(pt_probs_np - ort_probs_vec)
        max_abs_diff = float(np.max(abs_prob_diffs))
        mean_abs_diff = float(np.mean(abs_prob_diffs))
        agreement = (pt_pred_cid == ort_pred_cid)

        if agreement:
            cohort1_agreements += 1
        cohort1_max_diff = max(cohort1_max_diff, max_abs_diff)
        cohort1_sum_diff += mean_abs_diff

        demo_comparisons.append({
            "fixture": fname,
            "intended_label": intended_label,
            "pytorch_predicted_label": CLASS_LABELS[pt_pred_cid],
            "onnx_predicted_label": CLASS_LABELS[ort_pred_cid],
            "class_agreement": agreement,
            "pytorch_confidence": float(pt_probs_np[pt_pred_cid]),
            "onnx_confidence": float(ort_probs_vec[ort_pred_cid]),
            "max_abs_probability_diff": max_abs_diff,
            "mean_abs_probability_diff": mean_abs_diff,
            "pytorch_probabilities": {CLASS_LABELS[i]: float(pt_probs_np[i]) for i in range(6)},
            "onnx_probabilities": {CLASS_LABELS[i]: float(ort_probs_vec[i]) for i in range(6)}
        })

        print(f"  {fname:<20} | PT: {CLASS_LABELS[pt_pred_cid]:<7} ({pt_probs_np[pt_pred_cid]:.4f}) | "
              f"ONNX: {CLASS_LABELS[ort_pred_cid]:<7} ({ort_probs_vec[ort_pred_cid]:.4f}) | "
              f"MaxDiff: {max_abs_diff:.2e} | Agreement: {agreement}")

    audit_results["cohort_1_demo_fixtures"] = {
        "count": len(demo_files),
        "class_agreements": cohort1_agreements,
        "agreement_rate": cohort1_agreements / len(demo_files),
        "overall_max_abs_prob_diff": cohort1_max_diff,
        "overall_mean_abs_prob_diff": cohort1_sum_diff / len(demo_files),
        "fixtures": demo_comparisons
    }

    # ------------------------------------------------------------------------
    # Cohort 2: 300 V4 Fresh Synthetic Holdout Clips
    # ------------------------------------------------------------------------
    print("\n--- Auditing Cohort 2: 300 V4 Fresh Synthetic Holdout Clips ---")
    holdout_manifest_p = Path("experiments/synthetic_kws_v4/fresh_holdout_manifest_6class.csv")
    df_holdout = pd.read_csv(holdout_manifest_p)

    cohort2_agreements = 0
    cohort2_max_diff = 0.0
    cohort2_sum_diff = 0.0
    per_class_max_diff = {c: 0.0 for c in CLASS_LABELS}

    for idx, row in df_holdout.iterrows():
        fpath = Path(row["processed_path"])
        audio, sr = sf.read(str(fpath), dtype="float32")
        audio_t = torch.from_numpy(audio).float()
        with torch.no_grad():
            log_mel_t = extractor(audio_t)
            pt_logits, pt_probs = pt_model(log_mel_t)

        pt_probs_np = pt_probs.cpu().numpy()[0]
        pt_pred_cid = int(np.argmax(pt_probs_np))

        ort_inputs = {"log_mel": log_mel_t.cpu().numpy()}
        _, ort_probs_np = ort_session.run(None, ort_inputs)
        ort_probs_vec = ort_probs_np[0]
        ort_pred_cid = int(np.argmax(ort_probs_vec))

        abs_diffs = np.abs(pt_probs_np - ort_probs_vec)
        max_d = float(np.max(abs_diffs))
        mean_d = float(np.mean(abs_diffs))

        if pt_pred_cid == ort_pred_cid:
            cohort2_agreements += 1

        cohort2_max_diff = max(cohort2_max_diff, max_d)
        cohort2_sum_diff += mean_d

        for i, clabel in enumerate(CLASS_LABELS):
            per_class_max_diff[clabel] = max(per_class_max_diff[clabel], float(abs_diffs[i]))

    cohort2_count = len(df_holdout)
    cohort2_mean_diff = cohort2_sum_diff / cohort2_count
    print(f"  Total clips evaluated: {cohort2_count}")
    print(f"  Class agreement: {cohort2_agreements}/{cohort2_count} ({cohort2_agreements/cohort2_count*100:.2f}%)")
    print(f"  Overall max absolute probability difference: {cohort2_max_diff:.2e}")
    print(f"  Overall mean absolute probability difference: {cohort2_mean_diff:.2e}")
    for clabel, max_d in per_class_max_diff.items():
        print(f"    Class {clabel:<7} max diff: {max_d:.2e}")

    audit_results["cohort_2_fresh_holdout_300"] = {
        "count": cohort2_count,
        "manifest": str(holdout_manifest_p),
        "class_agreements": cohort2_agreements,
        "agreement_rate": cohort2_agreements / cohort2_count,
        "overall_max_abs_prob_diff": cohort2_max_diff,
        "overall_mean_abs_prob_diff": cohort2_mean_diff,
        "per_class_max_abs_prob_diff": per_class_max_diff
    }

    # ------------------------------------------------------------------------
    # Cohort 3: 50 Deterministic Synthetic Tensors (Documented Seed 20261016)
    # ------------------------------------------------------------------------
    print("\n--- Auditing Cohort 3: 50 Deterministic Synthetic Tensors (Seed 20261016) ---")
    rng = np.random.RandomState(20261016)
    torch.manual_seed(20261016)

    cohort3_agreements = 0
    cohort3_max_diff = 0.0
    cohort3_sum_diff = 0.0

    for i in range(50):
        # Varying statistical regimes: normal, uniform, silence floor, high dynamic range
        regime = i % 5
        if regime == 0:
            feat = rng.randn(1, 1, 40, 101).astype(np.float32)
        elif regime == 1:
            feat = rng.uniform(-10.0, 5.0, (1, 1, 40, 101)).astype(np.float32)
        elif regime == 2:
            # Ambient low floor
            feat = (rng.randn(1, 1, 40, 101) * 0.1 - 12.0).astype(np.float32)
        elif regime == 3:
            # All zeros (flat baseline)
            feat = np.zeros((1, 1, 40, 101), dtype=np.float32)
        else:
            # High amplitude burst
            feat = (rng.randn(1, 1, 40, 101) * 5.0).astype(np.float32)

        feat_t = torch.from_numpy(feat)
        with torch.no_grad():
            pt_logits, pt_probs = pt_model(feat_t)

        pt_probs_np = pt_probs.cpu().numpy()[0]
        pt_pred_cid = int(np.argmax(pt_probs_np))

        _, ort_probs_np = ort_session.run(None, {"log_mel": feat})
        ort_probs_vec = ort_probs_np[0]
        ort_pred_cid = int(np.argmax(ort_probs_vec))

        abs_diffs = np.abs(pt_probs_np - ort_probs_vec)
        max_d = float(np.max(abs_diffs))
        mean_d = float(np.mean(abs_diffs))

        if pt_pred_cid == ort_pred_cid:
            cohort3_agreements += 1

        cohort3_max_diff = max(cohort3_max_diff, max_d)
        cohort3_sum_diff += mean_d

    cohort3_count = 50
    cohort3_mean_diff = cohort3_sum_diff / cohort3_count
    print(f"  Total synthetic tensors evaluated: {cohort3_count}")
    print(f"  Class agreement: {cohort3_agreements}/{cohort3_count} ({cohort3_agreements/cohort3_count*100:.2f}%)")
    print(f"  Overall max absolute probability difference: {cohort3_max_diff:.2e}")
    print(f"  Overall mean absolute probability difference: {cohort3_mean_diff:.2e}")

    audit_results["cohort_3_deterministic_synthetic_tensors_50"] = {
        "count": cohort3_count,
        "random_seed": 20261016,
        "class_agreements": cohort3_agreements,
        "agreement_rate": cohort3_agreements / cohort3_count,
        "overall_max_abs_prob_diff": cohort3_max_diff,
        "overall_mean_abs_prob_diff": cohort3_mean_diff
    }

    # ------------------------------------------------------------------------
    # Latency Benchmarking (Isolated Forward Pass)
    # ------------------------------------------------------------------------
    print("\n--- Benchmarking Isolated Forward-Pass Latency (CPU) ---")
    benchmark_tensor = np.zeros((1, 1, 40, 101), dtype=np.float32)
    benchmark_t = torch.from_numpy(benchmark_tensor)

    # PyTorch warm-up & timing
    for _ in range(25):
        with torch.no_grad():
            _ = pt_model(benchmark_t)

    pt_times = []
    for _ in range(100):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = pt_model(benchmark_t)
        t1 = time.perf_counter()
        pt_times.append((t1 - t0) * 1000.0)

    # ONNX Runtime warm-up & timing
    ort_feed = {"log_mel": benchmark_tensor}
    for _ in range(25):
        _ = ort_session.run(None, ort_feed)

    ort_times = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = ort_session.run(None, ort_feed)
        t1 = time.perf_counter()
        ort_times.append((t1 - t0) * 1000.0)

    pt_mean = float(np.mean(pt_times))
    pt_p95 = float(np.percentile(pt_times, 95))
    ort_mean = float(np.mean(ort_times))
    ort_p95 = float(np.percentile(ort_times, 95))

    print(f"  PyTorch CPU forward pass:     mean = {pt_mean:.3f} ms | p95 = {pt_p95:.3f} ms (100 runs)")
    print(f"  ONNX Runtime CPU forward pass: mean = {ort_mean:.3f} ms | p95 = {ort_p95:.3f} ms (100 runs)")
    print(f"  Speedup ratio (PyTorch / ORT): {pt_mean / ort_mean:.2f}x")

    audit_results["latency_benchmark"] = {
        "runs": 100,
        "warmup_runs": 25,
        "measurement_boundary": "Isolated model forward pass on CPU (excludes audio file I/O and Log-Mel STFT)",
        "hardware_environment": "Desktop x86_64 CPU (single-thread execution)",
        "pytorch_forward_mean_ms": pt_mean,
        "pytorch_forward_p95_ms": pt_p95,
        "onnxruntime_forward_mean_ms": ort_mean,
        "onnxruntime_forward_p95_ms": ort_p95,
        "speedup_ratio": pt_mean / ort_mean
    }

    # ------------------------------------------------------------------------
    # Overall Parity Summary & Decision Gate
    # ------------------------------------------------------------------------
    all_agreements = (
        cohort1_agreements == len(demo_files) and
        cohort2_agreements == cohort2_count and
        cohort3_agreements == cohort3_count
    )
    max_global_diff = max(cohort1_max_diff, cohort2_max_diff, cohort3_max_diff)
    parity_passed = all_agreements and (max_global_diff <= 1e-4)

    audit_results["summary"] = {
        "total_test_samples": len(demo_files) + cohort2_count + cohort3_count,
        "total_class_agreements": cohort1_agreements + cohort2_agreements + cohort3_agreements,
        "overall_agreement_rate": (cohort1_agreements + cohort2_agreements + cohort3_agreements) / (len(demo_files) + cohort2_count + cohort3_count),
        "global_max_abs_probability_diff": max_global_diff,
        "parity_threshold_met": parity_passed,
        "status": "PARITY_AUDIT_PASS" if parity_passed else "PARITY_AUDIT_FAIL",
        "governance_notice": "HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED. Parity audit conducted solely on synthetic fixtures."
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)
    print(f"\nParity report saved to: {REPORT_PATH}")
    print(f"Overall Parity Status: {audit_results['summary']['status']}")
    print(f"Global Max Absolute Difference: {max_global_diff:.2e} (Threshold: <= 1e-4)")

    return audit_results


def main():
    onnx_sha, model_config = export_model_to_onnx(opset_version=17)
    parity_results = run_runtime_parity_audit(model_config)
    print("\nPhase 6 ONNX Export and Runtime-Parity Audit Completed Successfully.")


if __name__ == "__main__":
    main()
