#!/usr/bin/env python3
"""
Phase 13.4: ONNX Export and Numerical Parity Verification for Vietnamese KWS Candidate v11.
Workspace: D:\trainidentificationaudio
Target: experiments/vieneu_kws_research_v13_4/export/export_v11_onnx.py

Exports the frozen candidate checkpoint (variant_v11_hard_negative_weighted_seed42.pt)
directly to standard FP32 ONNX format and verifies numerical parity on development fixtures.
"""

import os
import sys
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import onnx
import onnxruntime as ort

from experiments.vieneu_kws_research_v13_1.bundle.infer import (
    CorrectedVietnameseKWSNet,
    LogMelFrontend,
    preprocess_audio_file,
    TARGET_CLASSES,
    DEFAULT_THRESHOLD,
    CLASSES
)

BUNDLE_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4" / "bundle"
CHECKPOINT_PATH = BUNDLE_DIR / "checkpoint" / "variant_v11_hard_negative_weighted_seed42.pt"
EXPECTED_CHECKPOINT_SHA256 = "3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2"

ONNX_DIR = BUNDLE_DIR / "onnx"
ONNX_FILE = ONNX_DIR / "vietnamese_kws_v11.onnx"
PARITY_MANIFEST_FILE = ONNX_DIR / "export_parity_manifest.json"
DEV_FIXTURES_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_1" / "fixtures"

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


def run_export_and_verify():
    ONNX_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PHASE 13.4: ONNX EXPORT AND NUMERICAL PARITY VERIFICATION")
    print("=" * 80)

    # 1. Verify Checkpoint Integrity
    assert CHECKPOINT_PATH.exists(), f"Checkpoint missing: {CHECKPOINT_PATH}"
    actual_ckpt_sha = compute_sha256(CHECKPOINT_PATH)
    assert actual_ckpt_sha == EXPECTED_CHECKPOINT_SHA256, (
        f"Checkpoint SHA-256 mismatch!\nExpected: {EXPECTED_CHECKPOINT_SHA256}\nActual:   {actual_ckpt_sha}"
    )
    print(f"[OK] Checkpoint integrity verified: {actual_ckpt_sha}")

    # 2. Instantiate and Load PyTorch Model
    model = CorrectedVietnameseKWSNet(num_classes=5)
    ckpt_data = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state_dict = ckpt_data["model_state_dict"] if "model_state_dict" in ckpt_data else ckpt_data
    model.load_state_dict(state_dict)
    model.eval()

    wrapper = ExportableVietnameseKWSNet(model)
    wrapper.eval()

    # 3. Export to ONNX
    print(f"Exporting model to {ONNX_FILE}...")
    dummy_input = torch.randn(1, 1, 40, 101, dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        dummy_input,
        str(ONNX_FILE),
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

    # Check onnx model structure
    onnx_model = onnx.load(str(ONNX_FILE))
    onnx.checker.check_model(onnx_model)
    onnx_sha = compute_sha256(ONNX_FILE)
    print(f"[OK] ONNX export succeeded. SHA-256: {onnx_sha}")

    # 4. Parity Testing on Dev Fixtures
    print("\n--- Verifying Numerical Parity on Dev Fixtures ---")
    frontend = LogMelFrontend()
    frontend.eval()

    ort_session = ort.InferenceSession(str(ONNX_FILE), providers=["CPUExecutionProvider"])

    fixture_files = sorted(list(DEV_FIXTURES_DIR.glob("*.wav")))
    assert len(fixture_files) >= 5, f"Expected at least 5 dev fixtures, found {len(fixture_files)}"

    fixture_results = []
    max_logits_diff_overall = 0.0
    max_probs_diff_overall = 0.0

    for fpath in fixture_files:
        audio = preprocess_audio_file(fpath)
        x = torch.from_numpy(audio).unsqueeze(0)

        with torch.no_grad():
            spec = frontend(x)
            pt_logits, pt_probs = wrapper(spec)

        pt_logits_np = pt_logits.cpu().numpy()
        pt_probs_np = pt_probs.cpu().numpy()

        ort_out = ort_session.run(None, {"log_mel": spec.cpu().numpy()})
        ort_logits_np = ort_out[0]
        ort_probs_np = ort_out[1]

        diff_logits = float(np.max(np.abs(pt_logits_np - ort_logits_np)))
        diff_probs = float(np.max(np.abs(pt_probs_np - ort_probs_np)))

        max_logits_diff_overall = max(max_logits_diff_overall, diff_logits)
        max_probs_diff_overall = max(max_probs_diff_overall, diff_probs)

        # Classification Parity
        cmd_probs_pt = pt_probs_np[0][:4]
        pred_pt = TARGET_CLASSES[int(np.argmax(cmd_probs_pt))] if np.max(cmd_probs_pt) >= DEFAULT_THRESHOLD else "REJECT"

        cmd_probs_ort = ort_probs_np[0][:4]
        pred_ort = TARGET_CLASSES[int(np.argmax(cmd_probs_ort))] if np.max(cmd_probs_ort) >= DEFAULT_THRESHOLD else "REJECT"

        parity_passed = (diff_logits <= TOLERANCE_FP32) and (diff_probs <= TOLERANCE_FP32) and (pred_pt == pred_ort)

        fixture_results.append({
            "fixture": fpath.name,
            "max_abs_diff_logits": diff_logits,
            "max_abs_diff_probs": diff_probs,
            "prediction_pytorch": pred_pt,
            "prediction_onnx": pred_ort,
            "parity_passed": parity_passed
        })

        print(f"  {fpath.name:<35} | Logits Diff: {diff_logits:.2e} | Probs Diff: {diff_probs:.2e} | Pred: {pred_ort} | Pass: {parity_passed}")

    assert all(r["parity_passed"] for r in fixture_results), "ONNX numerical parity check failed on dev fixtures!"
    print(f"\n[OK] All {len(fixture_results)} fixtures passed numerical parity (Max logits diff: {max_logits_diff_overall:.2e} <= {TOLERANCE_FP32}).")

    parity_manifest = {
        "status": "ONNX_PARITY_VERIFIED",
        "checkpoint_source": "variant_v11_hard_negative_weighted_seed42.pt",
        "checkpoint_sha256": actual_ckpt_sha,
        "onnx_model_file": ONNX_FILE.name,
        "onnx_sha256": onnx_sha,
        "onnx_opset": 17,
        "tolerance": TOLERANCE_FP32,
        "max_logits_diff_observed": max_logits_diff_overall,
        "max_probs_diff_observed": max_probs_diff_overall,
        "dev_fixtures_evaluated": len(fixture_results),
        "all_fixtures_passed": True,
        "fixtures": fixture_results,
        "export_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

    with open(PARITY_MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(parity_manifest, f, ensure_ascii=False, indent=2)
    print(f"[OK] Parity manifest written to {PARITY_MANIFEST_FILE.name}")

    return parity_manifest


if __name__ == "__main__":
    run_export_and_verify()
