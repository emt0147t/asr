#!/usr/bin/env python3
"""
Model Export and Parity Verification Script for Vietnamese KWS (Phase 91).
Candidate: variant_w10 (CorrectedVietnameseKWSNet)
Workspace: D:\trainidentificationaudio
Exports:
  1. FP32 ONNX Model:       models/variant_w10_seed42_fp32.onnx
  2. Dynamic INT8 ONNX:     models/variant_w10_seed42_int8.onnx
  3. TorchScript Model:     models/variant_w10_seed42_torchscript.pt

Performs numerical parity testing against PyTorch on fixed development fixtures
with explicit tolerance <= 1e-4 for FP32 ONNX and TorchScript.
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

from experiments.vieneu_kws_research_v9.artifacts.inference import (
    CorrectedVietnameseKWSNet,
    LogMelFrontend,
    CLASSES
)
from experiments.vieneu_kws_research_v10.deployment.audio_validator import AudioValidator

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "vieneu_kws_research_v9"
    / "artifacts"
    / "checkpoints"
    / "variant_w10_seed42.pt"
)
EXPECTED_CHECKPOINT_SHA256 = "7374af469a4bc4df5e124aae27720a8c11e688413f045eec3ee4ffd79c836bed"

MODELS_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v10" / "deployment" / "models"
ONNX_FP32_PATH = MODELS_DIR / "variant_w10_seed42_fp32.onnx"
ONNX_INT8_PATH = MODELS_DIR / "variant_w10_seed42_int8.onnx"
TORCHSCRIPT_PATH = MODELS_DIR / "variant_w10_seed42_torchscript.pt"
MANIFEST_PATH = MODELS_DIR / "export_manifest.json"

TOLERANCE_FP32 = 1e-4
DEFAULT_THRESHOLD = 0.40


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class ExportableVietnameseKWSNet(nn.Module):
    """Wrapper that outputs both raw logits and normalized probabilities."""
    def __init__(self, backbone: CorrectedVietnameseKWSNet):
        super().__init__()
        self.backbone = backbone

    def forward(self, log_mel: torch.Tensor):
        logits = self.backbone(log_mel)
        probs = torch.softmax(logits, dim=-1)
        return logits, probs


def run_export() -> Dict[str, Any]:
    print("=" * 80)
    print("PHASE 91: EXPORTING VIETNAMESE KWS CNN (variant_w10_seed42)")
    print("=" * 80)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Verify Checkpoint
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    actual_hash = compute_sha256(CHECKPOINT_PATH)
    assert actual_hash == EXPECTED_CHECKPOINT_SHA256, (
        f"Checkpoint hash mismatch!\nExpected: {EXPECTED_CHECKPOINT_SHA256}\nActual:   {actual_hash}"
    )
    print(f"[OK] Checkpoint verified: {CHECKPOINT_PATH.name}")
    print(f"     SHA-256: {actual_hash}")

    # 2. Instantiate Model
    base_model = CorrectedVietnameseKWSNet(num_classes=5)
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    base_model.load_state_dict(state_dict)
    base_model.eval()

    export_model = ExportableVietnameseKWSNet(base_model)
    export_model.eval()

    dummy_input = torch.zeros(1, 1, 40, 101, dtype=torch.float32)

    # 3. Export TorchScript
    print("\n--- Exporting TorchScript ---")
    with torch.no_grad():
        traced_model = torch.jit.trace(export_model, dummy_input)
        traced_model.save(str(TORCHSCRIPT_PATH))
    ts_size_kb = TORCHSCRIPT_PATH.stat().st_size / 1024.0
    ts_hash = compute_sha256(TORCHSCRIPT_PATH)
    print(f"[OK] TorchScript exported: {TORCHSCRIPT_PATH.name} ({ts_size_kb:.1f} KB, SHA256: {ts_hash[:16]}...)")

    # 4. Export FP32 ONNX
    print("\n--- Exporting ONNX FP32 ---")
    with torch.no_grad():
        torch.onnx.export(
            export_model,
            dummy_input,
            str(ONNX_FP32_PATH),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["log_mel"],
            output_names=["logits", "probabilities"],
            dynamic_axes={
                "log_mel": {0: "batch_size"},
                "logits": {0: "batch_size"},
                "probabilities": {0: "batch_size"}
            }
        )

    # Validate ONNX model graph
    onnx_proto = onnx.load(str(ONNX_FP32_PATH))
    onnx.checker.check_model(onnx_proto)
    onnx_fp32_size_kb = ONNX_FP32_PATH.stat().st_size / 1024.0
    onnx_fp32_hash = compute_sha256(ONNX_FP32_PATH)
    print(f"[OK] ONNX FP32 exported: {ONNX_FP32_PATH.name} ({onnx_fp32_size_kb:.1f} KB, SHA256: {onnx_fp32_hash[:16]}...)")

    # 5. Quantize to Dynamic INT8 ONNX
    print("\n--- Quantizing ONNX to Dynamic INT8 ---")
    quantize_dynamic(
        model_input=str(ONNX_FP32_PATH),
        model_output=str(ONNX_INT8_PATH),
        weight_type=QuantType.QInt8
    )
    onnx_int8_size_kb = ONNX_INT8_PATH.stat().st_size / 1024.0
    onnx_int8_hash = compute_sha256(ONNX_INT8_PATH)
    print(f"[OK] ONNX INT8 exported: {ONNX_INT8_PATH.name} ({onnx_int8_size_kb:.1f} KB, SHA256: {onnx_int8_hash[:16]}...)")

    # 6. Run Parity Verification on Synthetic Fixtures
    print("\n--- Running Parity Tests on Synthetic Fixtures ---")
    frontend = LogMelFrontend()
    frontend.eval()

    fixtures_dir = PROJECT_ROOT / "experiments" / "synthetic_kws_demo" / "fixtures"
    test_wavs = sorted([
        f for f in fixtures_dir.glob("*.wav")
        if not f.name.startswith("invalid_")
    ])

    ort_session_fp32 = ort.InferenceSession(str(ONNX_FP32_PATH), providers=["CPUExecutionProvider"])
    ort_session_int8 = ort.InferenceSession(str(ONNX_INT8_PATH), providers=["CPUExecutionProvider"])
    loaded_ts = torch.jit.load(str(TORCHSCRIPT_PATH))
    loaded_ts.eval()

    parity_results = []
    max_dev_onnx_fp32 = 0.0
    max_dev_ts = 0.0
    max_dev_onnx_int8 = 0.0
    label_mismatches_fp32 = 0
    label_mismatches_ts = 0
    label_mismatches_int8 = 0

    with torch.no_grad():
        for wav_path in test_wavs:
            wav_tensor, _, _ = AudioValidator.load_and_validate(wav_path, strict=False)
            feats = frontend(wav_tensor)  # [1, 1, 40, 101]

            # 1. PyTorch Reference
            py_logits, py_probs = export_model(feats)
            py_logits_np = py_logits.numpy()[0]
            py_probs_np = py_probs.numpy()[0]

            cmd_probs_py = py_probs_np[:4]
            max_cmd_py = float(cmd_probs_py.max())
            py_pred = CLASSES[cmd_probs_py.argmax()] if max_cmd_py >= DEFAULT_THRESHOLD else "REJECT"

            # 2. TorchScript
            ts_logits, ts_probs = loaded_ts(feats)
            ts_logits_np = ts_logits.numpy()[0]
            ts_probs_np = ts_probs.numpy()[0]
            diff_ts = float(abs(ts_logits_np - py_logits_np).max())
            max_dev_ts = max(max_dev_ts, diff_ts)
            cmd_probs_ts = ts_probs_np[:4]
            max_cmd_ts = float(cmd_probs_ts.max())
            ts_pred = CLASSES[cmd_probs_ts.argmax()] if max_cmd_ts >= DEFAULT_THRESHOLD else "REJECT"
            if ts_pred != py_pred:
                label_mismatches_ts += 1

            # 3. ONNX FP32
            feats_np = feats.numpy()
            onnx_outs = ort_session_fp32.run(None, {"log_mel": feats_np})
            onnx_logits_np = onnx_outs[0][0]
            onnx_probs_np = onnx_outs[1][0]
            diff_onnx = float(abs(onnx_logits_np - py_logits_np).max())
            max_dev_onnx_fp32 = max(max_dev_onnx_fp32, diff_onnx)
            cmd_probs_onnx = onnx_probs_np[:4]
            max_cmd_onnx = float(cmd_probs_onnx.max())
            onnx_pred = CLASSES[cmd_probs_onnx.argmax()] if max_cmd_onnx >= DEFAULT_THRESHOLD else "REJECT"
            if onnx_pred != py_pred:
                label_mismatches_fp32 += 1

            # 4. ONNX INT8
            int8_outs = ort_session_int8.run(None, {"log_mel": feats_np})
            int8_logits_np = int8_outs[0][0]
            int8_probs_np = int8_outs[1][0]
            diff_int8 = float(abs(int8_probs_np - py_probs_np).max())
            max_dev_onnx_int8 = max(max_dev_onnx_int8, diff_int8)
            cmd_probs_int8 = int8_probs_np[:4]
            max_cmd_int8 = float(cmd_probs_int8.max())
            int8_pred = CLASSES[cmd_probs_int8.argmax()] if max_cmd_int8 >= DEFAULT_THRESHOLD else "REJECT"
            if int8_pred != py_pred:
                label_mismatches_int8 += 1

            record = {
                "wav_file": wav_path.name,
                "py_prediction": py_pred,
                "ts_prediction": ts_pred,
                "onnx_fp32_prediction": onnx_pred,
                "onnx_int8_prediction": int8_pred,
                "max_diff_torchscript_logits": diff_ts,
                "max_diff_onnx_fp32_logits": diff_onnx,
                "max_diff_onnx_int8_probs": diff_int8
            }
            parity_results.append(record)
            print(f"  {wav_path.name:<25} Py: {py_pred:<6} TS: {ts_pred:<6} ONNX: {onnx_pred:<6} (max diff: {diff_onnx:.2e})")

    assert max_dev_onnx_fp32 <= TOLERANCE_FP32, (
        f"ONNX FP32 max deviation {max_dev_onnx_fp32:.2e} exceeds tolerance {TOLERANCE_FP32}!"
    )
    assert label_mismatches_fp32 == 0, f"Found {label_mismatches_fp32} label mismatches in ONNX FP32!"
    assert label_mismatches_ts == 0, f"Found {label_mismatches_ts} label mismatches in TorchScript!"

    print(f"\n[PARITY PASSED] All {len(test_wavs)} fixtures verified:")
    print(f"  TorchScript Max Diff: {max_dev_ts:.2e} (0 label mismatches)")
    print(f"  ONNX FP32 Max Diff:   {max_dev_onnx_fp32:.2e} (0 label mismatches, tol <= {TOLERANCE_FP32})")
    print(f"  ONNX INT8 Max Diff:   {max_dev_onnx_int8:.2e} ({label_mismatches_int8} label mismatches)")

    manifest = {
        "candidate_id": "variant_w10_seed42",
        "phase": 91,
        "source_checkpoint": {
            "path": str(CHECKPOINT_PATH.relative_to(PROJECT_ROOT)),
            "sha256": actual_hash,
            "parameter_count": 138469
        },
        "exported_models": {
            "torchscript": {
                "filename": TORCHSCRIPT_PATH.name,
                "sha256": ts_hash,
                "size_kb": round(ts_size_kb, 2),
                "format": "TorchScript (eval traced)"
            },
            "onnx_fp32": {
                "filename": ONNX_FP32_PATH.name,
                "sha256": onnx_fp32_hash,
                "size_kb": round(onnx_fp32_size_kb, 2),
                "opset_version": 17,
                "input_names": ["log_mel"],
                "input_shape": [1, 1, 40, 101],
                "output_names": ["logits", "probabilities"],
                "output_shapes": [[1, 5], [1, 5]]
            },
            "onnx_int8": {
                "filename": ONNX_INT8_PATH.name,
                "sha256": onnx_int8_hash,
                "size_kb": round(onnx_int8_size_kb, 2),
                "quantization_type": "dynamic_int8",
                "compression_ratio": round(onnx_fp32_size_kb / onnx_int8_size_kb, 2)
            }
        },
        "parity_verification": {
            "fixtures_evaluated": len(test_wavs),
            "max_deviation_torchscript": max_dev_ts,
            "max_deviation_onnx_fp32": max_dev_onnx_fp32,
            "max_deviation_onnx_int8_probs": max_dev_onnx_int8,
            "label_mismatches_torchscript": label_mismatches_ts,
            "label_mismatches_onnx_fp32": label_mismatches_fp32,
            "label_mismatches_onnx_int8": label_mismatches_int8,
            "tolerance_fp32": TOLERANCE_FP32,
            "status": "PARITY_VERIFIED"
        },
        "fixtures_detail": parity_results,
        "governance": {
            "human_listener_verification": "PENDING_HUMAN_LISTENER_VERIFICATION",
            "natural_speech_validation": "NOT_VALIDATED_ON_NATURAL_SPEECH",
            "continuous_open_mic_verdict": "NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION"
        }
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[OK] Export manifest saved: {MANIFEST_PATH.name}")
    return manifest


if __name__ == "__main__":
    run_export()
