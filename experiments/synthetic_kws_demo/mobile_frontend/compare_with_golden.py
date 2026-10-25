"""
Golden Parity & Numerical Verification Script for KWS Mobile Frontend (Phase 8).
Compares the C/C++ frontend against all 12 frozen golden vectors and feeds
C/C++ features to the frozen ONNX model.

Governance Invariant: HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED.
Golden artifacts in frontend_golden/ are strictly READ-ONLY and preserved.
"""

import sys
import json
import ctypes
import hashlib
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import soundfile as sf
import onnxruntime as ort

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.synthetic_kws_demo.classifier import PredictionResult
from experiments.synthetic_kws_demo.game_adapter import CommandToActionAdapter, GameAction


GOLDEN_DIR = PROJECT_ROOT / "experiments" / "synthetic_kws_demo" / "frontend_golden"
MANIFEST_PATH = GOLDEN_DIR / "golden_manifest.json"
ONNX_PATH = PROJECT_ROOT / "experiments" / "synthetic_kws_demo" / "export" / "arm_b_tcresnet14.onnx"
CHECKPOINT_PATH = PROJECT_ROOT / "experiments" / "synthetic_kws_v4" / "checkpoints" / "arm_b_onset_no_resample.pt"
DLL_PATH = PROJECT_ROOT / "experiments" / "synthetic_kws_demo" / "mobile_frontend" / "bin" / "kws_mobile_frontend.dll"
REPORT_PATH = PROJECT_ROOT / "experiments" / "synthetic_kws_demo" / "mobile_frontend" / "mobile_parity_report.json"

EXPECTED_CHECKPOINT_SHA256 = "6A0FD55CD6F16F5B683951127076EEC72506C31EE670F896AAEACE5981CF1D96"
EXPECTED_ONNX_SHA256 = "8A5AC4C105DF52F1749AE11B6BB2124B3A672486601EFF308ACC7E257FF97625"
EXPECTED_MANIFEST_SHA256 = "5904A5503CE5528ECF88A4A670C6726D80B74285B34EE6B910CF2FA48F60E5C3"


def compute_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load_cxx_frontend():
    assert DLL_PATH.exists(), f"C++ frontend DLL missing: {DLL_PATH}"
    lib = ctypes.CDLL(str(DLL_PATH))
    lib.kws_frontend_init.argtypes = []
    lib.kws_frontend_init.restype = ctypes.c_int
    lib.kws_frontend_compute.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float)
    ]
    lib.kws_frontend_compute.restype = ctypes.c_int
    lib.kws_frontend_get_error_string.argtypes = [ctypes.c_int]
    lib.kws_frontend_get_error_string.restype = ctypes.c_char_p

    init_res = lib.kws_frontend_init()
    assert init_res == 0, f"kws_frontend_init failed: {init_res}"
    return lib


def run_mobile_golden_parity() -> Dict[str, Any]:
    print("=" * 72)
    print("PHASE 8: Portable C/C++ Log-Mel Frontend Parity Verification")
    print("Governance Invariant: HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED")
    print("=" * 72)

    # 1. Baseline Model & Golden Manifest Verification
    assert CHECKPOINT_PATH.exists(), f"Checkpoint missing: {CHECKPOINT_PATH}"
    actual_ckpt_sha = compute_sha256(CHECKPOINT_PATH)
    assert actual_ckpt_sha == EXPECTED_CHECKPOINT_SHA256, f"Checkpoint SHA mismatch: {actual_ckpt_sha}"

    assert ONNX_PATH.exists(), f"ONNX model missing: {ONNX_PATH}"
    actual_onnx_sha = compute_sha256(ONNX_PATH)
    assert actual_onnx_sha == EXPECTED_ONNX_SHA256, f"ONNX SHA mismatch: {actual_onnx_sha}"

    assert MANIFEST_PATH.exists(), f"Golden manifest missing: {MANIFEST_PATH}"
    actual_manifest_sha = compute_sha256(MANIFEST_PATH)
    assert actual_manifest_sha == EXPECTED_MANIFEST_SHA256, f"Golden manifest SHA mismatch: {actual_manifest_sha}"

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Snapshot all golden file hashes before verification
    golden_hashes_before = {}
    for case in manifest["cases"]:
        wav_p = PROJECT_ROOT / case["wav_artifact"]["path"]
        tensor_p = PROJECT_ROOT / case["tensor_artifact"]["path"]
        assert wav_p.exists(), f"WAV missing: {wav_p}"
        assert tensor_p.exists(), f"Tensor missing: {tensor_p}"
        golden_hashes_before[str(wav_p)] = compute_sha256(wav_p)
        golden_hashes_before[str(tensor_p)] = compute_sha256(tensor_p)
        assert golden_hashes_before[str(wav_p)] == case["wav_artifact"]["sha256"]
        assert golden_hashes_before[str(tensor_p)] == case["tensor_artifact"]["sha256"]

    # 2. Initialize C++ Frontend & ONNX Session
    cxx_lib = load_cxx_frontend()
    ort_session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    game_adapter = CommandToActionAdapter(confidence_threshold=0.50)

    class_ordering = manifest["verified_frontend_contract"]["class_ordering"]
    ordered_labels = [class_ordering[str(i)] for i in range(len(class_ordering))]

    # 3. Iterate over all 12 Golden Cases
    case_reports: List[Dict[str, Any]] = []
    global_max_feature_diff = 0.0
    global_max_prob_diff = 0.0
    all_features_within_threshold = True
    all_argmax_agreements = True
    edge_cases_safe = True

    print(f"\nEvaluating {manifest['total_cases']} golden cases:")
    print(f"{'Case ID':<30} | {'Max Feat Diff':<13} | {'Max Prob Diff':<13} | {'Pred':<8} | {'Conf':<6} | {'Action':<10} | {'Status'}")
    print("-" * 105)

    for case in manifest["cases"]:
        cid = case["case_id"]
        wav_p = PROJECT_ROOT / case["wav_artifact"]["path"]
        tensor_p = PROJECT_ROOT / case["tensor_artifact"]["path"]

        # Read audio WAV
        audio, sr = sf.read(str(wav_p), dtype="float32")
        assert len(audio) == 16000 and sr == 16000, f"Invalid audio dimensions in {cid}"

        # Run C++ Frontend
        cxx_features = np.zeros((40, 101), dtype=np.float32)
        c_status = cxx_lib.kws_frontend_compute(
            audio.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            16000,
            cxx_features.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
        assert c_status == 0, f"kws_frontend_compute failed on {cid}: code {c_status}"

        # Load golden reference tensor: shape [1, 1, 40, 101]
        golden_tensor = np.load(str(tensor_p))
        golden_2d = golden_tensor[0, 0]

        # Compute numerical difference metrics
        abs_diff = np.abs(cxx_features - golden_2d)
        max_feat_diff = float(np.max(abs_diff))
        mean_feat_diff = float(np.mean(abs_diff))
        rel_diff = np.abs(cxx_features - golden_2d) / (np.abs(golden_2d) + 1e-6)
        max_rel_diff = float(np.max(rel_diff))

        global_max_feature_diff = max(global_max_feature_diff, max_feat_diff)
        feat_pass = (max_feat_diff <= 1.0e-4)
        if not feat_pass:
            all_features_within_threshold = False

        # Run ONNX inference on C++ features
        cxx_tensor_4d = cxx_features.reshape(1, 1, 40, 101)
        ort_logits, ort_probs = ort_session.run(None, {"log_mel": cxx_tensor_4d})
        cxx_probs_vec = ort_probs[0]

        cxx_pred_idx = int(np.argmax(cxx_probs_vec))
        cxx_pred_label = ordered_labels[cxx_pred_idx]
        cxx_confidence = float(cxx_probs_vec[cxx_pred_idx])

        ref_probs_vec = np.array(case["onnx_reference"]["probabilities"], dtype=np.float32)
        prob_abs_diff = np.abs(cxx_probs_vec - ref_probs_vec)
        max_prob_diff = float(np.max(prob_abs_diff))
        global_max_prob_diff = max(global_max_prob_diff, max_prob_diff)

        ref_pred_label = case["onnx_reference"]["predicted_label"]
        argmax_match = (cxx_pred_label == ref_pred_label)
        if not argmax_match:
            all_argmax_agreements = False

        # Adapter / Game Action verification
        pred_res = PredictionResult(
            is_valid=True,
            wav_path=str(wav_p),
            predicted_label=cxx_pred_label,
            predicted_class_id=cxx_pred_idx,
            confidence=cxx_confidence,
            class_probabilities={ordered_labels[i]: float(cxx_probs_vec[i]) for i in range(len(ordered_labels))},
            checkpoint_path=str(CHECKPOINT_PATH),
            checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256,
            preprocessing_latency_ms=0.0,
            inference_latency_ms=0.0,
            total_latency_ms=0.0,
            error_message=None
        )
        adapted_action = game_adapter.adapt(pred_res)

        # Safety verification for artificial edge cases
        is_edge_case = cid in ["case_09_silence_pure_zeros", "case_10_low_amplitude_ambient", "case_11_transient_impulse"]
        if is_edge_case:
            # Must produce NO_ACTION and confidence < 0.50
            if adapted_action.action != GameAction.NO_ACTION or cxx_confidence >= 0.50:
                edge_cases_safe = False

        case_pass = feat_pass and argmax_match and (max_prob_diff <= 1.0e-4)
        if is_edge_case:
            case_pass = case_pass and (adapted_action.action == GameAction.NO_ACTION)

        status_str = "PASS" if case_pass else "FAIL"
        print(f"{cid:<30} | {max_feat_diff:<13.6e} | {max_prob_diff:<13.6e} | {cxx_pred_label:<8} | {cxx_confidence:<6.4f} | {adapted_action.action.value:<10} | {status_str}")

        case_reports.append({
            "case_id": cid,
            "case_name": case["case_name"],
            "category": case["category"],
            "feature_max_abs_diff": max_feat_diff,
            "feature_mean_abs_diff": mean_feat_diff,
            "feature_max_rel_diff": max_rel_diff,
            "feature_parity_pass": feat_pass,
            "cxx_predicted_label": cxx_pred_label,
            "ref_predicted_label": ref_pred_label,
            "argmax_agreement": argmax_match,
            "max_prob_diff": max_prob_diff,
            "cxx_confidence": cxx_confidence,
            "adapted_action": adapted_action.action.value,
            "is_edge_case": is_edge_case,
            "edge_case_safe": (adapted_action.action == GameAction.NO_ACTION) if is_edge_case else True,
            "status": status_str
        })

    # 4. Invariant Check: Verify all golden hashes are unchanged
    for path_str, expected_sha in golden_hashes_before.items():
        curr_sha = compute_sha256(Path(path_str))
        assert curr_sha == expected_sha, f"CRITICAL: Golden file modified: {path_str}"

    curr_manifest_sha = compute_sha256(MANIFEST_PATH)
    assert curr_manifest_sha == EXPECTED_MANIFEST_SHA256, "CRITICAL: golden_manifest.json modified"

    # Overall Verdict: Host-level cross-implementation parity
    overall_status = "HOST_CROSS_IMPLEMENTATION_PARITY_PASS" if (
        all_features_within_threshold and
        all_argmax_agreements and
        edge_cases_safe and
        global_max_feature_diff <= 1.0e-4 and
        global_max_prob_diff <= 1.0e-4
    ) else "HOST_CROSS_IMPLEMENTATION_PARITY_FAIL"

    report = {
        "report_version": "1.0.0",
        "timestamp": "2026-10-16",
        "governance": "HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED.",
        "summary": {
            "suite_status": overall_status,
            "total_golden_cases": len(case_reports),
            "feature_parity_cases": sum(1 for c in case_reports if c["feature_parity_pass"]),
            "argmax_agreement_cases": sum(1 for c in case_reports if c["argmax_agreement"]),
            "edge_cases_verified_safe": sum(1 for c in case_reports if c["is_edge_case"] and c["edge_case_safe"]),
            "global_max_feature_abs_diff": global_max_feature_diff,
            "global_max_prob_diff": global_max_prob_diff,
            "feature_diff_threshold": 1.0e-4,
            "prob_diff_threshold": 1.0e-4,
            "confidence_threshold": 0.50
        },
        "cxx_frontend_library": {
            "path": str(DLL_PATH),
            "sha256": compute_sha256(DLL_PATH),
            "fft_algorithm": "Cooley-Tukey Radix-2 In-Place DIT FFT (512-point, 9-bit bit-reversal)",
            "window_spec": "Periodic Hann (400-point, zero-padded with 56 leading and 56 trailing zeros to 512)",
            "mel_filterbank": "40 Slaney Area-Normalized Filters (precomputed static float table)",
            "padding_mode": "Symmetric Reflect (256 samples on each side)",
            "log_compression": "Natural logarithm ln(mel_power + 1.0e-6)"
        },
        "cases": case_reports
    }

    # Write report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("-" * 105)
    print(f"Overall Suite Verdict: {overall_status}")
    print(f"Global Max Feature Abs Diff: {global_max_feature_diff:.6e} (Threshold: <= 1.000000e-04)")
    print(f"Global Max Prob Diff:        {global_max_prob_diff:.6e} (Threshold: <= 1.000000e-04)")
    print(f"Report written to: {REPORT_PATH}")
    print("=" * 72)

    return report


if __name__ == "__main__":
    report = run_mobile_golden_parity()
    if report["summary"]["suite_status"] != "HOST_CROSS_IMPLEMENTATION_PARITY_PASS":
        sys.exit(1)
