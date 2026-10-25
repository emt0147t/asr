#!/usr/bin/env python3
"""
Phase 12 Model Export and Numerical Parity Verification.
Workspace: D:\trainidentificationaudio
Target: experiments/vieneu_kws_research_v12/onnx/export_onnx_v12.py

Exports frozen Phase 11 candidate (variant_v11_hard_negative_weighted_seed42.pt)
to ONNX (FP32 & Dynamic INT8), verifying numerical parity against PyTorch reference
on Phase 12 development fixtures with tolerance <= 1e-4 for FP32.
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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

from experiments.vieneu_kws_research_v12.bundle.infer import (
    CorrectedVietnameseKWSNet,
    LogMelFrontend,
    preprocess_audio_file,
    CLASSES,
    TARGET_CLASSES,
    DEFAULT_THRESHOLD
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "vieneu_kws_research_v12"
    / "bundle"
    / "checkpoint"
    / "variant_v11_hard_negative_weighted_seed42.pt"
)
EXPECTED_CHECKPOINT_SHA256 = "3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2"

ONNX_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "onnx"
ONNX_FP32_PATH = ONNX_DIR / "variant_v11_kws_model_fp32.onnx"
ONNX_INT8_PATH = ONNX_DIR / "variant_v11_kws_model_int8.onnx"
MANIFEST_PATH = ONNX_DIR / "onnx_export_manifest.json"
FIXTURES_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "fixtures"

TOLERANCE_FP32 = 1e-4


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class ExportableVietnameseKWSNet(nn.Module):
    """Wrapper that outputs both raw logits and softmax probabilities."""
    def __init__(self, backbone: CorrectedVietnameseKWSNet):
        super().__init__()
        self.backbone = backbone

    def forward(self, log_mel: torch.Tensor):
        logits = self.backbone(log_mel)
        probs = torch.softmax(logits, dim=-1)
        return logits, probs


def export_and_verify():
    print("=" * 80)
    print("PHASE 12: ONNX EXPORT & NUMERICAL PARITY VERIFICATION")
    print("=" * 80)

    ONNX_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Verify Checkpoint
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    actual_hash = compute_sha256(CHECKPOINT_PATH)
    assert actual_hash == EXPECTED_CHECKPOINT_SHA256, (
        f"Checkpoint hash mismatch!\nExpected: {EXPECTED_CHECKPOINT_SHA256}\nActual:   {actual_hash}"
    )
    print(f"[OK] Verified Checkpoint: {CHECKPOINT_PATH.name}")
    print(f"     SHA-256: {actual_hash}")

    # 2. Instantiate Model and Load Weights
    base_model = CorrectedVietnameseKWSNet(num_classes=5)
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    base_model.load_state_dict(state_dict)
    base_model.eval()

    export_model = ExportableVietnameseKWSNet(base_model)
    export_model.eval()

    dummy_input = torch.zeros(1, 1, 40, 101, dtype=torch.float32)

    # 3. Export FP32 ONNX
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

    onnx_proto = onnx.load(str(ONNX_FP32_PATH))
    onnx.checker.check_model(onnx_proto)
    onnx_fp32_size_kb = ONNX_FP32_PATH.stat().st_size / 1024.0
    onnx_fp32_hash = compute_sha256(ONNX_FP32_PATH)
    print(f"[OK] ONNX FP32 exported: {ONNX_FP32_PATH.name} ({onnx_fp32_size_kb:.1f} KB, SHA-256: {onnx_fp32_hash})")

    # 4. Quantize to Dynamic INT8 ONNX
    print("\n--- Quantizing ONNX to Dynamic INT8 ---")
    quantize_dynamic(
        model_input=str(ONNX_FP32_PATH),
        model_output=str(ONNX_INT8_PATH),
        weight_type=QuantType.QInt8
    )
    onnx_int8_size_kb = ONNX_INT8_PATH.stat().st_size / 1024.0
    onnx_int8_hash = compute_sha256(ONNX_INT8_PATH)
    print(f"[OK] ONNX INT8 exported: {ONNX_INT8_PATH.name} ({onnx_int8_size_kb:.1f} KB, SHA-256: {onnx_int8_hash})")

    # 5. Parity Verification across Development Fixtures
    print("\n--- Running Parity Tests on Development Fixtures ---")
    frontend = LogMelFrontend()
    frontend.eval()

    fixtures = sorted(FIXTURES_DIR.glob("*.wav"))
    if not fixtures:
        raise FileNotFoundError(f"No fixtures found in {FIXTURES_DIR}")

    ort_session_fp32 = ort.InferenceSession(str(ONNX_FP32_PATH), providers=["CPUExecutionProvider"])
    ort_session_int8 = ort.InferenceSession(str(ONNX_INT8_PATH), providers=["CPUExecutionProvider"])

    parity_records = []
    max_dev_onnx_fp32_logits = 0.0
    max_dev_onnx_fp32_probs = 0.0
    label_mismatches_fp32 = 0
    label_mismatches_int8 = 0

    with torch.no_grad():
        for wav_path in fixtures:
            audio_np = preprocess_audio_file(wav_path)
            x_tensor = torch.from_numpy(audio_np).unsqueeze(0)
            log_mel_tensor = frontend(x_tensor)  # [1, 1, 40, 101]

            # PyTorch Reference
            py_logits, py_probs = export_model(log_mel_tensor)
            py_logits_np = py_logits.numpy()[0]
            py_probs_np = py_probs.numpy()[0]

            cmd_probs_py = py_probs_np[:4]
            max_cmd_py = float(cmd_probs_py.max())
            py_pred = TARGET_CLASSES[cmd_probs_py.argmax()] if max_cmd_py >= DEFAULT_THRESHOLD else "REJECT"

            # ONNX FP32
            feats_np = log_mel_tensor.numpy()
            onnx_outs = ort_session_fp32.run(None, {"log_mel": feats_np})
            onnx_logits_np = onnx_outs[0][0]
            onnx_probs_np = onnx_outs[1][0]

            diff_logits = float(np.max(np.abs(onnx_logits_np - py_logits_np)))
            diff_probs = float(np.max(np.abs(onnx_probs_np - py_probs_np)))
            max_dev_onnx_fp32_logits = max(max_dev_onnx_fp32_logits, diff_logits)
            max_dev_onnx_fp32_probs = max(max_dev_onnx_fp32_probs, diff_probs)

            cmd_probs_onnx = onnx_probs_np[:4]
            max_cmd_onnx = float(cmd_probs_onnx.max())
            onnx_pred = TARGET_CLASSES[cmd_probs_onnx.argmax()] if max_cmd_onnx >= DEFAULT_THRESHOLD else "REJECT"
            if onnx_pred != py_pred:
                label_mismatches_fp32 += 1

            # ONNX INT8
            int8_outs = ort_session_int8.run(None, {"log_mel": feats_np})
            int8_probs_np = int8_outs[1][0]
            cmd_probs_int8 = int8_probs_np[:4]
            max_cmd_int8 = float(cmd_probs_int8.max())
            int8_pred = TARGET_CLASSES[cmd_probs_int8.argmax()] if max_cmd_int8 >= DEFAULT_THRESHOLD else "REJECT"
            if int8_pred != py_pred:
                label_mismatches_int8 += 1

            record = {
                "wav_file": wav_path.name,
                "py_prediction": py_pred,
                "onnx_fp32_prediction": onnx_pred,
                "onnx_int8_prediction": int8_pred,
                "onnx_fp32_max_prob": float(max_cmd_onnx),
                "py_max_prob": float(max_cmd_py),
                "max_diff_fp32_logits": diff_logits,
                "max_diff_fp32_probs": diff_probs
            }
            parity_records.append(record)
            print(f"  {wav_path.name:32s} | Py: {py_pred:6s} | ONNX: {onnx_pred:6s} | ProbDiff: {diff_probs:.2e}")

    print("\n--- Summary ---")
    print(f"Total Fixtures Evaluated:           {len(fixtures)}")
    print(f"Max Absolute Diff (FP32 Logits):    {max_dev_onnx_fp32_logits:.2e}")
    print(f"Max Absolute Diff (FP32 Probs):     {max_dev_onnx_fp32_probs:.2e} (tolerance <= {TOLERANCE_FP32})")
    print(f"Label Mismatches (PyTorch vs FP32): {label_mismatches_fp32}")
    print(f"Label Mismatches (PyTorch vs INT8): {label_mismatches_int8}")

    assert max_dev_onnx_fp32_probs <= TOLERANCE_FP32, (
        f"FP32 numerical parity exceeded tolerance: {max_dev_onnx_fp32_probs} > {TOLERANCE_FP32}"
    )
    assert label_mismatches_fp32 == 0, f"Label mismatches between PyTorch and ONNX FP32: {label_mismatches_fp32}"

    manifest_data = {
        "status": "PARITY_VERIFIED",
        "source_checkpoint": {
            "path": str(CHECKPOINT_PATH),
            "sha256": actual_hash,
            "architecture": "CorrectedVietnameseKWSNet",
            "parameters": sum(p.numel() for p in base_model.parameters())
        },
        "models": {
            "onnx_fp32": {
                "path": str(ONNX_FP32_PATH),
                "size_kb": onnx_fp32_size_kb,
                "sha256": onnx_fp32_hash,
                "opset_version": 17,
                "max_prob_deviation": max_dev_onnx_fp32_probs,
                "max_logits_deviation": max_dev_onnx_fp32_logits,
                "label_mismatch_count": label_mismatches_fp32,
                "tolerance_applied": TOLERANCE_FP32,
                "parity_passed": True
            },
            "onnx_int8": {
                "path": str(ONNX_INT8_PATH),
                "size_kb": onnx_int8_size_kb,
                "sha256": onnx_int8_hash,
                "quantization": "dynamic_int8",
                "label_mismatch_count": label_mismatches_int8
            }
        },
        "parity_records": parity_records
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"\n[SUCCESS] Manifest saved to: {MANIFEST_PATH.name}")
    return manifest_data


if __name__ == "__main__":
    export_and_verify()
