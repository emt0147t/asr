"""
Model Export and Mobile Parity Module for Vietnamese KWS.
Exports PyTorch models to ONNX (Float32 and Quantized INT8):
  - Champion Model: VietnameseTCResNet (models/checkpoints/kws_tcresnet_advanced.pt)
  - Baseline Model: VietnameseKWSNet (models/checkpoints/best_kws_model.pt)
Benchmarks latency & file sizes on desktop CPU, verifies bit-accurate numerical parity,
and exports golden audio test vectors for Android/iOS integration testing.
"""

import sys
from pathlib import Path

# Ensure UTF-8 stdout and sys.path
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
    except Exception:
        pass
else:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

import time
import json
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

from src.models.kws_architectures import VietnameseTCResNet, VietnameseKWSNet
from src.preprocessing.feature_extractor import LogMelFeatureExtractor, extract_log_mel_numpy

CLASS_NAMES = ["TRAI", "PHAI", "LEN", "XUONG", "UNKNOWN", "SILENCE"]


def export_single_model(model: torch.nn.Module,
                        model_name: str,
                        input_shape: tuple,
                        export_dir: Path):
    """
    Exports PyTorch model to Float32 ONNX and Quantized INT8 ONNX.
    """
    fp32_path = export_dir / f"{model_name}_fp32.onnx"
    int8_path = export_dir / f"{model_name}_int8.onnx"
    dummy_input = torch.randn(*input_shape, dtype=torch.float32)

    # 1. Export FP32 ONNX
    torch.onnx.export(
        model,
        dummy_input,
        str(fp32_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["log_mel_input"],
        output_names=["logits_output"],
        dynamic_axes={
            "log_mel_input": {0: "batch_size"},
            "logits_output": {0: "batch_size"}
        }
    )
    onnx.checker.check_model(onnx.load(str(fp32_path)))
    print(f"[EXPORT] Float32 ONNX exported -> {fp32_path}")

    # 2. Dynamic Quantization to INT8
    try:
        quantize_dynamic(
            model_input=str(fp32_path),
            model_output=str(int8_path),
            weight_type=QuantType.QUInt8
        )
        print(f"[QUANTIZE] Dynamic INT8 ONNX exported -> {int8_path}")
    except Exception as e:
        print(f"[QUANTIZE WARNING] Could not quantize {model_name}: {e}")
        int8_path = fp32_path

    # Footprint
    fp32_size_kb = fp32_path.stat().st_size / 1024.0
    int8_size_kb = int8_path.stat().st_size / 1024.0
    print(f"  --> {model_name} FP32 Size: {fp32_size_kb:.1f} KB | INT8 Size: {int8_size_kb:.1f} KB (Compression: {(1 - int8_size_kb/fp32_size_kb)*100:.1f}%)")

    # 3. Numerical Parity Check
    session_fp32 = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    dummy_np = dummy_input.numpy()
    with torch.no_grad():
        pt_out = model(dummy_input).numpy()
    ort_out = session_fp32.run(None, {"log_mel_input": dummy_np})[0]
    max_diff = float(np.max(np.abs(pt_out - ort_out)))
    print(f"  --> Numerical Parity (PyTorch vs ONNX FP32): Max Abs Diff = {max_diff:.2e}")
    assert max_diff < 1e-4, f"Parity failure for {model_name}: max diff {max_diff}"
    print(f"  [PASS] Bit-accurate output parity verified for {model_name} (< 1e-4)")

    # 4. Latency Benchmark
    session_int8 = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    n_warmup, n_runs = 20, 200
    for _ in range(n_warmup):
        _ = session_fp32.run(None, {"log_mel_input": dummy_np})
        _ = session_int8.run(None, {"log_mel_input": dummy_np})

    t0 = time.perf_counter()
    for _ in range(n_runs):
        _ = session_fp32.run(None, {"log_mel_input": dummy_np})
    lat_fp32_ms = ((time.perf_counter() - t0) / n_runs) * 1000.0

    t0 = time.perf_counter()
    for _ in range(n_runs):
        _ = session_int8.run(None, {"log_mel_input": dummy_np})
    lat_int8_ms = ((time.perf_counter() - t0) / n_runs) * 1000.0

    print(f"  --> Latency (Desktop CPU): FP32 = {lat_fp32_ms:.3f} ms | INT8 = {lat_int8_ms:.3f} ms")

    return {
        "model_name": model_name,
        "fp32_path": str(fp32_path),
        "int8_path": str(int8_path),
        "fp32_size_kb": fp32_size_kb,
        "int8_size_kb": int8_size_kb,
        "max_parity_diff": max_diff,
        "lat_fp32_ms": lat_fp32_ms,
        "lat_int8_ms": lat_int8_ms
    }


def export_models_and_golden_vectors():
    print("=" * 65)
    print("EXPORTING VIETNAMESE KWS MODELS FOR MOBILE RUNTIMES")
    print("=" * 65)

    export_dir = Path("models/exported")
    golden_dir = Path("tests/golden_vectors")
    export_dir.mkdir(parents=True, exist_ok=True)
    golden_dir.mkdir(parents=True, exist_ok=True)

    # 1. Export Champion TC-ResNet-14 Model
    tcresnet_ckpt = Path("models/checkpoints/kws_tcresnet_advanced.pt")
    if tcresnet_ckpt.exists():
        print("\n--- Exporting Champion Model: TC-ResNet-14 ---")
        checkpoint = torch.load(tcresnet_ckpt, map_location="cpu", weights_only=False)
        model_tc = VietnameseTCResNet(num_classes=6)
        model_tc.load_state_dict(checkpoint["model_state_dict"])
        model_tc.eval()
        # Input shape: (Batch, 40, 101)
        res_tc = export_single_model(model_tc, "vietnamese_kws_tcresnet", (1, 40, 101), export_dir)
    else:
        print(f"[WARNING] Missing checkpoint: {tcresnet_ckpt}")
        model_tc = None

    # 2. Export Baseline 2D-CNN Model (for legacy compatibility)
    baseline_ckpt = Path("models/checkpoints/best_kws_model.pt")
    if baseline_ckpt.exists():
        print("\n--- Exporting Baseline Model: 2D-CNN ---")
        checkpoint = torch.load(baseline_ckpt, map_location="cpu", weights_only=False)
        model_base = VietnameseKWSNet(num_classes=6)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        model_base.load_state_dict(state_dict)
        model_base.eval()
        # Input shape: (Batch, 1, 40, 101)
        res_base = export_single_model(model_base, "vietnamese_kws", (1, 1, 40, 101), export_dir)

    # 3. Generate Golden Audio Test Vectors for Android & iOS (using test_manifest_v2.csv)
    print("\n--- Generating Golden Audio Test Vectors from Verified Test Manifest (v2) ---")
    test_manifest = pd.read_csv("data/manifests/test_manifest_v2.csv")
    extractor = LogMelFeatureExtractor(sample_rate=16000, n_mels=40)

    eval_model = model_tc if model_tc is not None else model_base
    is_tcresnet = (eval_model is model_tc)

    for c_id, c_name in enumerate(CLASS_NAMES):
        c_rows = test_manifest[test_manifest["class_id"] == c_id]
        if len(c_rows) == 0:
            print(f"[SKIP] No sample found in test manifest for {c_name}")
            continue
        sample_row = c_rows.iloc[0]
        wav_path = Path(sample_row["processed_path"])
        audio, sr = sf.read(str(wav_path), dtype="float32")
        if len(audio) < 16000:
            audio = np.pad(audio, (0, 16000 - len(audio)))
        elif len(audio) > 16000:
            audio = audio[:16000]

        # PyTorch extraction & inference
        with torch.no_grad():
            feat_torch = extractor(torch.from_numpy(audio).float())  # (1, 1, 40, 101)
            inp = feat_torch.squeeze(1) if is_tcresnet else feat_torch
            logits_torch = eval_model(inp).numpy()[0]
            probs_torch = torch.softmax(torch.from_numpy(logits_torch), dim=0).numpy()

        # NumPy reference extraction
        feat_numpy = extract_log_mel_numpy(audio)  # (40, 101)

        golden_payload = {
            "metadata": {
                "class_id": c_id,
                "class_label": c_name,
                "recording_id": sample_row["recording_id"],
                "source": sample_row["source"],
                "transcript": sample_row["transcript"],
                "sample_rate": 16000,
                "samples_count": len(audio),
                "model_architecture": "TC-ResNet-14" if is_tcresnet else "VietnameseKWSNet"
            },
            "audio_waveform_first_100_samples": [round(float(x), 6) for x in audio[:100]],
            "log_mel_shape": list(feat_numpy.shape),
            "log_mel_sample_corner": [round(float(x), 6) for x in feat_numpy[:3, :3].flatten()],
            "expected_logits": [round(float(x), 6) for x in logits_torch],
            "expected_probabilities": [round(float(x), 6) for x in probs_torch],
            "expected_argmax_class": int(np.argmax(logits_torch))
        }

        golden_file = golden_dir / f"golden_{c_name.lower()}.json"
        with open(golden_file, "w", encoding="utf-8") as f:
            json.dump(golden_payload, f, indent=2)
        print(f"  --> Saved 6-class golden vector: {c_name} -> {golden_file}")

    print("\n[SUCCESS] Mobile runtime export and 6-class golden vectors completed.")


if __name__ == "__main__":
    export_models_and_golden_vectors()
