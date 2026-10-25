#!/usr/bin/env python3
"""
Phase 27 Candidate Artifact Audit and ONNX Export with Numerical Parity Verification
Repository: D:\\trainidentificationaudio
Target: experiments/vieneu_kws_research_v12/scripts/audit_and_export_candidate_onnx.py
"""

import sys
import io
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import onnx
import onnxruntime as ort

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CANDIDATE_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "candidates" / "phase26_candidate_d0_h1_seed44"
CKPT_PATH = CANDIDATE_DIR / "candidate_checkpoint.pt"
ONNX_PATH = CANDIDATE_DIR / "vietnamese_kws_phase26_candidate.onnx"

sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "scripts"))
from train_phase25_factorial import CorrectedVietnameseKWSNet, LogMelFrontend, CLASSES, CLASS_TO_IDX


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 80)
    print("PHASE 27: CANDIDATE ARTIFACT AUDIT & ONNX EXPORT PARITY")
    print("=" * 80)

    # 1. Load checkpoint
    ckpt_sha = compute_sha256(CKPT_PATH)
    expected_sha = "00b4fc71f863788648d5ea388a99c4cbd4441d9b9f63adc5e56154e329df63f3"
    assert ckpt_sha == expected_sha, f"Checkpoint hash mismatch: {ckpt_sha} != {expected_sha}"
    print(f"[OK] Checkpoint verified: {CKPT_PATH.name} (SHA-256: {ckpt_sha})")

    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model = CorrectedVietnameseKWSNet(num_classes=5, dropout_p=0.30)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_params == 138469, f"Parameter count mismatch: {total_params} != 138469"
    print(f"[OK] Architecture: CorrectedVietnameseKWSNet with {total_params:,} trainable parameters.")

    # 2. Export to ONNX
    dummy_input = torch.randn(1, 1, 40, 101, dtype=torch.float32)
    print(f"\nExporting to ONNX: {ONNX_PATH.name}...")

    torch.onnx.export(
        model,
        dummy_input,
        str(ONNX_PATH),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}}
    )

    onnx_sha = compute_sha256(ONNX_PATH)
    print(f"[OK] ONNX export complete (SHA-256: {onnx_sha})")

    # 3. Check ONNX model validity
    onnx_model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(onnx_model)
    print("[OK] ONNX model graph structure validated successfully.")

    # 4. Numerical Parity Verification
    print("\n--- Verifying PyTorch to ONNX Numerical Parity ---")
    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])

    rng = np.random.RandomState(42027)
    max_abs_diff = 0.0
    max_rel_diff = 0.0

    for i in range(100):
        # Generate random spectrogram-like inputs
        test_in = rng.randn(1, 1, 40, 101).astype(np.float32)

        # PyTorch inference
        with torch.no_grad():
            torch_out = model(torch.from_numpy(test_in)).numpy()

        # ONNX inference
        onnx_out = session.run(["logits"], {"input": test_in})[0]

        abs_diff = np.max(np.abs(torch_out - onnx_out))
        rel_diff = np.max(np.abs(torch_out - onnx_out) / (np.abs(torch_out) + 1e-7))

        max_abs_diff = max(max_abs_diff, abs_diff)
        max_rel_diff = max(max_rel_diff, rel_diff)

    print(f"Max Absolute Logit Difference across 100 random vectors: {max_abs_diff:.2e}")
    print(f"Max Relative Difference: {max_rel_diff:.2e}")
    assert max_abs_diff < 1e-4, f"Parity violation: max_abs_diff {max_abs_diff} >= 1e-4"
    print("[PASS] PyTorch and ONNX predictions are numerically identical within float32 tolerance.")

    # Save export metadata
    export_info = {
        "candidate_checkpoint_sha256": ckpt_sha,
        "candidate_onnx_sha256": onnx_sha,
        "opset_version": 14,
        "input_shape": [1, 1, 40, 101],
        "output_shape": [1, 5],
        "classes": CLASSES,
        "max_abs_diff": float(max_abs_diff),
        "parity_verified": True
    }
    with open(CANDIDATE_DIR / "onnx_export_info.json", "w", encoding="utf-8") as f:
        json.dump(export_info, f, indent=2)

    print(f"\n[OK] Parity audit complete. Output saved to {CANDIDATE_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
