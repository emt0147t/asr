#!/usr/bin/env python3
"""
Phase 18 Gate A: E1 Checkpoint to ONNX Export and Numerical Parity Verification.
Workspace: D:\trainidentificationaudio
Target: experiments/vieneu_kws_research_v12/scripts/export_e1_onnx.py

Exports frozen E1 candidate (exp_v12_e1_baseline_scale_seed42.pt)
to ONNX (vietnamese_kws_v12_e1.onnx), verifying numerical parity against PyTorch reference
on Phase 12 development fixtures with strict tolerance <= 1e-5.
"""

import os
import sys
import io
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List

# Ensure UTF-8 console output
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import onnxruntime as ort

from experiments.vieneu_kws_research_v13_4.bundle.infer import (
    LogMelFrontendNumPy,
    preprocess_audio,
    CLASSES,
    TARGET_CLASSES,
    DEFAULT_THRESHOLD
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "vieneu_kws_research_v12"
    / "checkpoints"
    / "exp_v12_e1_baseline_scale_seed42.pt"
)
EXPECTED_CHECKPOINT_SHA256 = "f3c0fe8f7ccebb8fe354cb50acb247e0674232044706db0187feb92b1706b878"

ONNX_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "onnx"
ONNX_E1_PATH = ONNX_DIR / "vietnamese_kws_v12_e1.onnx"
PARITY_REPORT_PATH = ONNX_DIR / "e1_onnx_parity_report.json"
FIXTURES_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "fixtures"

TOLERANCE_PARITY = 1e-5


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


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


class ExportableVietnameseKWSNet(nn.Module):
    """Wrapper that outputs both raw logits and softmax probabilities."""
    def __init__(self, backbone: CorrectedVietnameseKWSNet):
        super().__init__()
        self.backbone = backbone

    def forward(self, log_mel: torch.Tensor):
        logits = self.backbone(log_mel)
        probs = torch.softmax(logits, dim=-1)
        return logits, probs


def export_and_verify() -> Dict[str, Any]:
    print("=" * 80)
    print("PHASE 18 GATE A: E1 ONNX EXPORT & NUMERICAL PARITY VERIFICATION")
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
    print(f"\n--- Exporting ONNX FP32 to {ONNX_E1_PATH.name} ---")
    with torch.no_grad():
        torch.onnx.export(
            export_model,
            dummy_input,
            str(ONNX_E1_PATH),
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

    # Check ONNX validity
    onnx_proto = onnx.load(str(ONNX_E1_PATH))
    onnx.checker.check_model(onnx_proto)
    onnx_size_kb = ONNX_E1_PATH.stat().st_size / 1024.0
    onnx_hash = compute_sha256(ONNX_E1_PATH)
    print(f"[OK] ONNX model exported: {ONNX_E1_PATH.name} ({onnx_size_kb:.1f} KB)")
    print(f"     SHA-256: {onnx_hash}")

    # 4. Parity Verification across Development Fixtures
    print("\n--- Running Parity Tests on Development Fixtures ---")
    frontend = LogMelFrontendNumPy()

    fixtures = sorted(FIXTURES_DIR.glob("*.wav"))
    if not fixtures:
        raise FileNotFoundError(f"No fixtures found in {FIXTURES_DIR}")
    print(f"Found {len(fixtures)} fixtures in {FIXTURES_DIR.name}/")

    ort_session = ort.InferenceSession(str(ONNX_E1_PATH), providers=["CPUExecutionProvider"])

    fixture_records = []
    max_logit_diff = 0.0
    max_prob_diff = 0.0
    label_mismatches = 0

    with torch.no_grad():
        for wav_path in fixtures:
            # Preprocess audio (16,000 samples, clamp > 1.0)
            audio_np = preprocess_audio(wav_path)
            # LogMelFrontendNumPy produces [1, 1, 40, 101]
            spec_np = frontend.compute(audio_np)
            spec_tensor = torch.from_numpy(spec_np)

            # PyTorch Reference forward pass
            py_logits_tensor, py_probs_tensor = export_model(spec_tensor)
            py_logits = py_logits_tensor.numpy()[0]
            py_probs = py_probs_tensor.numpy()[0]

            # ONNX Runtime forward pass
            onnx_outs = ort_session.run(None, {"log_mel": spec_np})
            onnx_logits = onnx_outs[0][0]
            onnx_probs = onnx_outs[1][0]

            # Parity errors
            diff_logits = float(np.max(np.abs(onnx_logits - py_logits)))
            diff_probs = float(np.max(np.abs(onnx_probs - py_probs)))
            max_logit_diff = max(max_logit_diff, diff_logits)
            max_prob_diff = max(max_prob_diff, diff_probs)

            # Canonical decision rule
            py_top_idx = int(np.argmax(py_probs))
            py_top_prob = float(py_probs[py_top_idx])
            py_pred = TARGET_CLASSES[py_top_idx] if (py_top_idx < 4 and py_top_prob >= DEFAULT_THRESHOLD) else "REJECT"

            onnx_top_idx = int(np.argmax(onnx_probs))
            onnx_top_prob = float(onnx_probs[onnx_top_idx])
            onnx_pred = TARGET_CLASSES[onnx_top_idx] if (onnx_top_idx < 4 and onnx_top_prob >= DEFAULT_THRESHOLD) else "REJECT"

            matched = (py_pred == onnx_pred)
            if not matched:
                label_mismatches += 1

            rec = {
                "fixture": wav_path.name,
                "py_pred": py_pred,
                "onnx_pred": onnx_pred,
                "label_matched": matched,
                "max_abs_logit_diff": diff_logits,
                "max_abs_prob_diff": diff_probs,
                "py_probs": {CLASSES[i]: float(py_probs[i]) for i in range(5)},
                "onnx_probs": {CLASSES[i]: float(onnx_probs[i]) for i in range(5)}
            }
            fixture_records.append(rec)
            status = "PASS" if (diff_logits <= TOLERANCE_PARITY and diff_probs <= TOLERANCE_PARITY and matched) else "FAIL"
            print(f"  [{status}] {wav_path.name:<35} | Pred: {onnx_pred:<6} | MaxLogitDiff: {diff_logits:.2e} | MaxProbDiff: {diff_probs:.2e}")

    parity_passed = (
        max_logit_diff <= TOLERANCE_PARITY and
        max_prob_diff <= TOLERANCE_PARITY and
        label_mismatches == 0
    )

    summary = {
        "gate": "GATE_A_E1_ONNX_EXPORT_AND_PARITY",
        "status": "PASS" if parity_passed else "FAIL",
        "checkpoint_source": str(CHECKPOINT_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "checkpoint_sha256": actual_hash,
        "onnx_output": str(ONNX_E1_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "onnx_sha256": onnx_hash,
        "onnx_size_kb": round(onnx_size_kb, 2),
        "tolerance_threshold": TOLERANCE_PARITY,
        "max_abs_logit_diff": max_logit_diff,
        "max_abs_prob_diff": max_prob_diff,
        "total_fixtures": len(fixtures),
        "label_mismatches": label_mismatches,
        "fixtures": fixture_records
    }

    with open(PARITY_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("GATE A SUMMARY:")
    print(f"  Status:             {summary['status']}")
    print(f"  Max Logit Diff:     {max_logit_diff:.2e} (Threshold: <= {TOLERANCE_PARITY})")
    print(f"  Max Prob Diff:      {max_prob_diff:.2e} (Threshold: <= {TOLERANCE_PARITY})")
    print(f"  Label Mismatches:   {label_mismatches} / {len(fixtures)}")
    print(f"  ONNX SHA-256:       {onnx_hash}")
    print(f"  Report written to:  {PARITY_REPORT_PATH.name}")
    print("=" * 80)

    if not parity_passed:
        raise RuntimeError(f"Gate A failed! Parity difference exceeds {TOLERANCE_PARITY}")

    return summary


if __name__ == "__main__":
    export_and_verify()
