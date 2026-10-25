"""
Verification Script for Frontend Golden Vectors and Numerical Parity.
Recomputes features and model inferences from source WAVs, compares against
immutable golden artifacts, and verifies exact reproducibility.
Governance Invariant: HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED.
"""

import sys
import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Any

if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

import numpy as np
import soundfile as sf
import torch
import onnxruntime as ort

from experiments.synthetic_kws_v3.models_v3 import VietnameseTCResNet
from experiments.synthetic_kws_demo.export.export_and_benchmark import (
    ExportableTCResNet,
    CHECKPOINT_PATH,
    EXPECTED_CHECKPOINT_SHA256,
    ONNX_PATH,
)
from src.preprocessing.feature_extractor import LogMelFeatureExtractor
from experiments.synthetic_kws_demo.classifier import CLASS_LABELS


GOLDEN_DIR = Path("experiments/synthetic_kws_demo/frontend_golden")
MANIFEST_PATH = GOLDEN_DIR / "golden_manifest.json"
REPORT_PATH = GOLDEN_DIR / "feature_parity_report.json"


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest().upper()


def verify_golden_suite() -> Dict[str, Any]:
    print("=" * 72)
    print("PHASE 7: Verifying Frontend Golden Vectors & Numerical Reproducibility...")
    print("=" * 72)

    assert MANIFEST_PATH.exists(), f"Golden manifest missing: {MANIFEST_PATH}"
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # 1. Model Checkpoint Verification
    ckpt_sha = sha256_file(CHECKPOINT_PATH)
    assert ckpt_sha == EXPECTED_CHECKPOINT_SHA256, f"Checkpoint hash mismatch: {ckpt_sha}"
    assert ONNX_PATH.exists(), f"ONNX model missing: {ONNX_PATH}"
    onnx_sha = sha256_file(ONNX_PATH)
    assert onnx_sha == manifest["onnx_model"]["sha256"], f"ONNX model hash mismatch: {onnx_sha}"

    # 2. Setup Reference Models
    base_model = VietnameseTCResNet(num_classes=6, dropout_p=0.0)
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    base_model.load_state_dict(state_dict)
    base_model.eval()
    pt_model = ExportableTCResNet(base_model)
    pt_model.eval()

    ort_session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    extractor = LogMelFeatureExtractor(sample_rate=16000, n_mels=40)
    extractor.eval()

    cases = manifest["cases"]
    verification_results = []
    global_max_feature_diff = 0.0
    global_max_prob_diff = 0.0
    all_agreements = True
    all_reproducible = True

    for c in cases:
        cid = c["case_id"]
        wav_p = Path(c["wav_artifact"]["path"])
        tensor_p = Path(c["tensor_artifact"]["path"])

        assert wav_p.exists(), f"WAV missing: {wav_p}"
        assert tensor_p.exists(), f"Saved tensor missing: {tensor_p}"

        # Check hashes
        act_wav_sha = sha256_file(wav_p)
        act_tensor_sha = sha256_file(tensor_p)
        assert act_wav_sha == c["wav_artifact"]["sha256"], f"WAV SHA mismatch on {cid}"
        assert act_tensor_sha == c["tensor_artifact"]["sha256"], f"Tensor SHA mismatch on {cid}"

        # Load saved golden tensor
        saved_log_mel = np.load(str(tensor_p))

        # Recompute Log-Mel features from WAV
        audio, sr = sf.read(str(wav_p), dtype="float32")
        assert len(audio) == 16000 and sr == 16000
        audio_t = torch.from_numpy(audio).float()
        with torch.no_grad():
            recomputed_t = extractor(audio_t)
            pt_logits, pt_probs = pt_model(recomputed_t)

        recomputed_np = recomputed_t.cpu().numpy()

        # Check finite values
        assert not np.isnan(recomputed_np).any()
        assert not np.isinf(recomputed_np).any()
        assert recomputed_np.shape == (1, 1, 40, 101)

        # Numerical comparison between recomputed and saved golden tensor
        feat_abs_diff = np.abs(recomputed_np - saved_log_mel)
        max_feat_diff = float(np.max(feat_abs_diff))
        mean_feat_diff = float(np.mean(feat_abs_diff))
        global_max_feature_diff = max(global_max_feature_diff, max_feat_diff)

        # Feature reproducibility check (exact or floating-point machine precision)
        is_feat_reproducible = (max_feat_diff <= 1e-6)
        if not is_feat_reproducible:
            all_reproducible = False

        # Run ONNX Runtime on recomputed feature tensor
        ort_logits_np, ort_probs_np = ort_session.run(None, {"log_mel": recomputed_np})
        ort_probs_vec = ort_probs_np[0]
        pt_probs_vec = pt_probs.cpu().numpy()[0]

        pt_cid = int(np.argmax(pt_probs_vec))
        ort_cid = int(np.argmax(ort_probs_vec))
        argmax_match = (pt_cid == ort_cid)
        if not argmax_match:
            all_agreements = False

        max_prob_diff = float(np.max(np.abs(pt_probs_vec - ort_probs_vec)))
        global_max_prob_diff = max(global_max_prob_diff, max_prob_diff)

        case_res = {
            "case_id": cid,
            "case_name": c["case_name"],
            "category": c["category"],
            "feature_shape_match": (recomputed_np.shape == (1, 1, 40, 101)),
            "feature_dtype_match": (str(recomputed_np.dtype) == "float32"),
            "feature_max_abs_diff_against_saved": max_feat_diff,
            "feature_mean_abs_diff_against_saved": mean_feat_diff,
            "feature_is_reproducible": is_feat_reproducible,
            "pytorch_argmax_label": CLASS_LABELS[pt_cid],
            "onnx_argmax_label": CLASS_LABELS[ort_cid],
            "argmax_agreement": argmax_match,
            "max_abs_prob_diff": max_prob_diff,
            "status": "PASS" if (is_feat_reproducible and argmax_match and max_prob_diff <= 1e-4) else "FAIL"
        }
        verification_results.append(case_res)

        print(f"  [{cid:<27}] | FeatDiff: {max_feat_diff:.2e} | ProbDiff: {max_prob_diff:.2e} | "
              f"PT: {CLASS_LABELS[pt_cid]:<7} | ONNX: {CLASS_LABELS[ort_cid]:<7} | Status: {case_res['status']}")

    suite_status = "GOLDEN_VERIFICATION_PASS" if (all_reproducible and all_agreements and global_max_prob_diff <= 1e-4) else "GOLDEN_VERIFICATION_FAIL"

    report_data = {
        "report_version": "1.0.0",
        "timestamp": "2026-10-16",
        "governance": "HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED.",
        "summary": {
            "total_golden_cases": len(cases),
            "reproducible_feature_cases": sum(1 for r in verification_results if r["feature_is_reproducible"]),
            "argmax_agreement_cases": sum(1 for r in verification_results if r["argmax_agreement"]),
            "global_max_feature_diff_against_saved": global_max_feature_diff,
            "global_max_prob_diff_pytorch_vs_onnx": global_max_prob_diff,
            "parity_criterion_threshold": 1.0e-4,
            "feature_tolerance_threshold": 1.0e-6,
            "suite_status": suite_status
        },
        "verified_contract_summary": manifest["verified_frontend_contract"],
        "cases": verification_results
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print("\n" + "=" * 72)
    print(f"Verification Suite Status: {suite_status}")
    print(f"Global Max Feature Difference against Golden Tensors: {global_max_feature_diff:.2e}")
    print(f"Global Max Probability Difference (PT vs ORT):       {global_max_prob_diff:.2e}")
    print(f"Report saved to: {REPORT_PATH}")
    print("=" * 72)

    return report_data


if __name__ == "__main__":
    verify_golden_suite()
