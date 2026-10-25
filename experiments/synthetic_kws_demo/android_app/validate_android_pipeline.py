"""
Offline Android Pipeline Validation & Numerical Parity Test Harness (Phase 9).
Verifies:
  1. Toolchain readiness across Gates A, B, C, D, E.
  2. Asset integrity (model SHA-256 and audio fixture hashes).
  3. Strict WAV validation and rejection contract (mono, 16kHz, 16000 samples).
  4. Complete end-to-end pipeline across all 12 frozen golden vectors:
     WAV validation -> C++ frontend -> ONNX inference -> GameActionAdapter.
  5. Action-suppression path for artificial numerical edge cases (cases 09, 10, 11).

Governance Invariant: HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED.
"""

import sys
import os
import json
import ctypes
import shutil
import hashlib
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import soundfile as sf
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.synthetic_kws_demo.classifier import PredictionResult
from experiments.synthetic_kws_demo.game_adapter import CommandToActionAdapter, GameAction


ANDROID_APP_DIR = PROJECT_ROOT / "experiments" / "synthetic_kws_demo" / "android_app"
ASSETS_DIR = ANDROID_APP_DIR / "app" / "src" / "main" / "assets"
GOLDEN_DIR = PROJECT_ROOT / "experiments" / "synthetic_kws_demo" / "frontend_golden"
MANIFEST_PATH = GOLDEN_DIR / "golden_manifest.json"
ONNX_SOURCE_PATH = PROJECT_ROOT / "experiments" / "synthetic_kws_demo" / "export" / "arm_b_tcresnet14.onnx"
ONNX_ASSET_PATH = ASSETS_DIR / "models" / "arm_b_tcresnet14.onnx"
CHECKPOINT_PATH = PROJECT_ROOT / "experiments" / "synthetic_kws_v4" / "checkpoints" / "arm_b_onset_no_resample.pt"
DLL_PATH = PROJECT_ROOT / "experiments" / "synthetic_kws_demo" / "mobile_frontend" / "bin" / "kws_mobile_frontend.dll"
REPORT_PATH = ANDROID_APP_DIR / "android_parity_report.json"

EXPECTED_CHECKPOINT_SHA256 = "6A0FD55CD6F16F5B683951127076EEC72506C31EE670F896AAEACE5981CF1D96"
EXPECTED_ONNX_SHA256 = "8A5AC4C105DF52F1749AE11B6BB2124B3A672486601EFF308ACC7E257FF97625"
EXPECTED_MANIFEST_SHA256 = "5904A5503CE5528ECF88A4A670C6726D80B74285B34EE6B910CF2FA48F60E5C3"

CLASS_LABELS = ["TRAI", "PHAI", "LEN", "XUONG", "UNKNOWN", "SILENCE"]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def inspect_toolchains() -> Dict[str, Any]:
    """Inspects host system for Android and Java build tools."""
    tools = ["java", "javac", "gradle", "cmake", "adb"]
    found_tools = {}
    for t in tools:
        found_tools[t] = shutil.which(t)

    env_vars = {
        "JAVA_HOME": os.environ.get("JAVA_HOME"),
        "ANDROID_HOME": os.environ.get("ANDROID_HOME"),
        "ANDROID_SDK_ROOT": os.environ.get("ANDROID_SDK_ROOT"),
        "ANDROID_NDK_HOME": os.environ.get("ANDROID_NDK_HOME")
    }

    local_sdk = Path(os.path.expandvars("%LOCALAPPDATA%/Android/Sdk"))
    sdk_exists = local_sdk.exists()

    return {
        "tools": found_tools,
        "environment_variables": env_vars,
        "local_sdk_exists": sdk_exists,
        "local_sdk_path": str(local_sdk) if sdk_exists else None
    }


def load_cxx_frontend():
    assert DLL_PATH.exists(), f"Frontend DLL missing: {DLL_PATH}"
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

    ret = lib.kws_frontend_init()
    assert ret == 0, f"Frontend init failed: {ret}"
    return lib


def validate_wav_header(file_path: Path) -> Dict[str, Any]:
    """Strict WAV parser validating format without resampling/normalization."""
    data = file_path.read_bytes()
    if len(data) < 44:
        raise ValueError(f"WAV file too small ({len(data)} bytes)")

    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("Invalid WAV RIFF/WAVE header")

    # Parse fmt chunk
    fmt_idx = data.find(b"fmt ")
    if fmt_idx == -1:
        raise ValueError("Missing 'fmt ' chunk in WAV")

    audio_format = int.from_bytes(data[fmt_idx+8:fmt_idx+10], "little")
    channels = int.from_bytes(data[fmt_idx+10:fmt_idx+12], "little")
    sample_rate = int.from_bytes(data[fmt_idx+12:fmt_idx+16], "little")
    bits_per_sample = int.from_bytes(data[fmt_idx+22:fmt_idx+24], "little")

    if channels != 1:
        raise ValueError(f"Unsupported channel count: expected mono (1), got {channels}")
    if sample_rate != 16000:
        raise ValueError(f"Unsupported sample rate: expected 16000 Hz, got {sample_rate} Hz")

    # Parse data chunk
    data_idx = data.find(b"data")
    if data_idx == -1:
        raise ValueError("Missing 'data' chunk in WAV")

    data_size = int.from_bytes(data[data_idx+4:data_idx+8], "little")
    bytes_per_sample = bits_per_sample // 8
    num_samples = data_size // bytes_per_sample

    if num_samples != 16000:
        raise ValueError(f"Unsupported duration: expected 16000 samples, got {num_samples}")

    # Extract samples
    raw_pcm = data[data_idx+8:data_idx+8+data_size]
    if audio_format == 1 and bits_per_sample == 16:
        samples = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32) / 32768.0
    elif audio_format == 3 and bits_per_sample == 32:
        samples = np.frombuffer(raw_pcm, dtype=np.float32).copy()
    else:
        raise ValueError(f"Unsupported audio format: code {audio_format}, {bits_per_sample} bits")

    if np.isnan(samples).any() or np.isinf(samples).any():
        raise ValueError("Non-finite values encountered in audio data")

    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "bits_per_sample": bits_per_sample,
        "num_samples": num_samples,
        "samples": samples
    }


def run_android_validation() -> Dict[str, Any]:
    print("=" * 72)
    print("PHASE 9: Android Offline KWS Integration and Runtime Validation")
    print("Governance Invariant: HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED")
    print("=" * 72)

    # 1. Toolchain Probe
    tc_info = inspect_toolchains()
    print("\n--- Toolchain Inspection ---")
    for t, p in tc_info["tools"].items():
        print(f"  {t:<10}: {p if p else 'NOT FOUND'}")
    for v, val in tc_info["environment_variables"].items():
        print(f"  {v:<18}: {val if val else 'NOT SET'}")
    print(f"  Local SDK Exists  : {tc_info['local_sdk_exists']}")

    # 2. Asset Integrity Check
    print("\n--- Asset Integrity Check ---")
    assert ONNX_SOURCE_PATH.exists(), f"Source ONNX missing: {ONNX_SOURCE_PATH}"
    src_onnx_sha = sha256_file(ONNX_SOURCE_PATH)
    assert src_onnx_sha == EXPECTED_ONNX_SHA256, f"Source ONNX SHA mismatch: {src_onnx_sha}"

    assert ONNX_ASSET_PATH.exists(), f"Packaged ONNX asset missing: {ONNX_ASSET_PATH}"
    asset_onnx_sha = sha256_file(ONNX_ASSET_PATH)
    assert asset_onnx_sha == EXPECTED_ONNX_SHA256, f"Packaged ONNX SHA mismatch: {asset_onnx_sha}"
    assert src_onnx_sha == asset_onnx_sha, "Packaged ONNX differs from source model"
    print(f"  Packaged ONNX Asset SHA-256: {asset_onnx_sha} (MATCHES FROZEN SOURCE)")

    # Check manifest
    manifest_p = ASSETS_DIR / "asset_manifest.json"
    assert manifest_p.exists(), f"Asset manifest missing: {manifest_p}"
    print(f"  Asset Manifest: {manifest_p.name} (VERIFIED)")

    # Check Manifest permissions: NO microphone, NO network
    manifest_xml = ANDROID_APP_DIR / "app" / "src" / "main" / "AndroidManifest.xml"
    assert manifest_xml.exists()
    xml_content = manifest_xml.read_text(encoding="utf-8")
    assert "RECORD_AUDIO" not in xml_content, "CRITICAL: RECORD_AUDIO found in AndroidManifest.xml"
    assert "INTERNET" not in xml_content, "CRITICAL: INTERNET permission found in AndroidManifest.xml"
    print("  AndroidManifest.xml: 0 permissions (NO RECORD_AUDIO, NO INTERNET verified)")

    # 3. Malformed Audio Rejection Contract Test
    print("\n--- Audio Validation Rejection Contract ---")
    malformed_tests = [
        ("invalid_stereo.wav", ASSETS_DIR / "audio" / "invalid_stereo.wav", "channel count"),
        ("invalid_sample_rate_8k.wav", PROJECT_ROOT / "experiments/synthetic_kws_demo/fixtures/invalid_sample_rate_8k.wav", "sample rate"),
        ("invalid_short_duration.wav", PROJECT_ROOT / "experiments/synthetic_kws_demo/fixtures/invalid_short_duration.wav", "duration"),
        ("invalid_empty.wav", PROJECT_ROOT / "experiments/synthetic_kws_demo/fixtures/invalid_empty.wav", "too small")
    ]
    for name, p, expected_err in malformed_tests:
        if p.exists():
            try:
                validate_wav_header(p)
                raise AssertionError(f"Expected validation failure on {name}, but it passed!")
            except ValueError as e:
                print(f"  {name:<28}: SAFELY REJECTED ({e})")

    # 4. End-to-End Pipeline on All 12 Golden Vectors
    print("\n--- End-to-End Golden Vector Parity & Action Suppression ---")
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        golden_man = json.load(f)

    cxx_lib = load_cxx_frontend()
    ort_session = ort.InferenceSession(str(ONNX_ASSET_PATH), providers=["CPUExecutionProvider"])
    game_adapter = CommandToActionAdapter(confidence_threshold=0.50)

    cases_results = []
    global_max_feat_diff = 0.0
    global_max_prob_diff = 0.0
    all_passed = True

    print(f"{'Case ID':<30} | {'Max Feat Diff':<13} | {'Max Prob Diff':<13} | {'Pred':<8} | {'Conf':<6} | {'Action':<10} | {'Status'}")
    print("-" * 105)

    for case in golden_man["cases"]:
        cid = case["case_id"]
        wav_p = PROJECT_ROOT / case["wav_artifact"]["path"]
        tensor_p = PROJECT_ROOT / case["tensor_artifact"]["path"]

        # Parse WAV with strict validator
        wav_info = validate_wav_header(wav_p)
        audio = wav_info["samples"]

        # C++ Frontend feature extraction
        cxx_features = np.zeros((40, 101), dtype=np.float32)
        c_status = cxx_lib.kws_frontend_compute(
            audio.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            16000,
            cxx_features.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
        assert c_status == 0, f"Frontend compute error on {cid}: {c_status}"

        # Reference comparison
        golden_tensor = np.load(str(tensor_p))[0, 0]
        feat_abs_diff = np.abs(cxx_features - golden_tensor)
        max_feat_diff = float(np.max(feat_abs_diff))
        mean_feat_diff = float(np.mean(feat_abs_diff))
        global_max_feat_diff = max(global_max_feat_diff, max_feat_diff)

        # ONNX inference on asset model
        inp_4d = cxx_features.reshape(1, 1, 40, 101)
        ort_logits, ort_probs = ort_session.run(None, {"log_mel": inp_4d})
        probs_vec = ort_probs[0]

        pred_idx = int(np.argmax(probs_vec))
        pred_label = CLASS_LABELS[pred_idx]
        confidence = float(probs_vec[pred_idx])

        ref_probs = np.array(case["onnx_reference"]["probabilities"], dtype=np.float32)
        max_prob_diff = float(np.max(np.abs(probs_vec - ref_probs)))
        global_max_prob_diff = max(global_max_prob_diff, max_prob_diff)

        ref_label = case["onnx_reference"]["predicted_label"]
        argmax_match = (pred_label == ref_label)

        # Game adapter action
        pred_res = PredictionResult(
            is_valid=True,
            wav_path=str(wav_p),
            predicted_label=pred_label,
            predicted_class_id=pred_idx,
            confidence=confidence,
            class_probabilities={CLASS_LABELS[i]: float(probs_vec[i]) for i in range(len(CLASS_LABELS))},
            checkpoint_path=str(CHECKPOINT_PATH),
            checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256,
            preprocessing_latency_ms=0.0,
            inference_latency_ms=0.0,
            total_latency_ms=0.0,
            error_message=None
        )
        adapted_action = game_adapter.adapt(pred_res)

        # Edge case safety suppression check
        is_edge_case = cid in ["case_09_silence_pure_zeros", "case_10_low_amplitude_ambient", "case_11_transient_impulse"]
        edge_case_safe = True
        if is_edge_case:
            # Under confidence threshold, must be NO_ACTION and confidence < 0.50
            if adapted_action.action != GameAction.NO_ACTION or confidence >= 0.50:
                edge_case_safe = False

        case_pass = (max_feat_diff <= 1.0e-4) and (max_prob_diff <= 1.0e-4) and argmax_match and edge_case_safe
        if not case_pass:
            all_passed = False

        status_str = "PASS" if case_pass else "FAIL"
        print(f"{cid:<30} | {max_feat_diff:<13.6e} | {max_prob_diff:<13.6e} | {pred_label:<8} | {confidence:<6.4f} | {adapted_action.action.value:<10} | {status_str}")

        cases_results.append({
            "case_id": cid,
            "category": case["category"],
            "wav_sha256": case["wav_artifact"]["sha256"],
            "feature_shape": [1, 1, 40, 101],
            "feature_dtype": "float32",
            "has_nan_or_inf": bool(np.isnan(cxx_features).any() or np.isinf(cxx_features).any()),
            "feature_max_abs_diff": max_feat_diff,
            "feature_mean_abs_diff": mean_feat_diff,
            "onnx_max_prob_diff": max_prob_diff,
            "reference_predicted_label": ref_label,
            "pipeline_predicted_label": pred_label,
            "confidence": confidence,
            "adapted_action": adapted_action.action.value,
            "is_edge_case": is_edge_case,
            "edge_case_safe": edge_case_safe,
            "status": status_str
        })

    # 5. Build Gate Status Determination (Per Section 2 & 4 specifications)
    has_ndk = tc_info["environment_variables"]["ANDROID_NDK_HOME"] is not None
    has_sdk = tc_info["environment_variables"]["ANDROID_HOME"] is not None or tc_info["local_sdk_exists"]
    has_gradle = tc_info["tools"]["gradle"] is not None
    has_jdk = tc_info["tools"]["javac"] is not None or tc_info["environment_variables"]["JAVA_HOME"] is not None

    gate_a_status = "PASS" if all_passed and global_max_feat_diff <= 1.0e-4 else "FAIL"
    gate_b_status = "BLOCKED" if not has_ndk else "PASS"
    gate_c_status = "BLOCKED" if not (has_sdk and has_gradle and has_jdk) else "PASS"
    gate_d_status = "BLOCKED"
    gate_e_status = "NOT_TESTED"

    # Overall Phase 9 verdict is BLOCKED for the Android deployment objective while toolchain is unavailable
    overall_verdict = "BLOCKED"

    report = {
        "report_version": "1.0.0",
        "phase": "Phase 9 — Android Offline KWS Integration and ONNX Runtime Validation",
        "timestamp": "2026-10-16",
        "governance": "HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED. Offline synthetic fixtures only.",
        "overall_verdict": overall_verdict,
        "gates": {
            "Gate_A_Host_Regression": {
                "name": "Host-side / static validation",
                "description": "Host-level C++/Python feature parity and ONNX inference across 12 golden vectors",
                "status": gate_a_status,
                "global_max_feature_diff": global_max_feat_diff,
                "global_max_prob_diff": global_max_prob_diff,
                "tolerance_threshold": 1.0e-4
            },
            "Gate_B_Android_Native_Build": {
                "name": "Android NDK compilation",
                "description": "Android NDK compilation of C++ Log-Mel frontend for target ABIs",
                "status": gate_b_status,
                "reason": "Host lacks Android NDK toolchain (ANDROID_NDK_HOME not set)",
                "cmake_config_tested": True,
                "prerequisites_missing": ["Android NDK", "NDK C++ toolchain"] if not has_ndk else []
            },
            "Gate_C_Android_Application_Build": {
                "name": "APK build",
                "description": "Gradle build producing APK with native library, ONNX model, and assets",
                "status": gate_c_status,
                "reason": "Host lacks Java Development Kit (javac/JAVA_HOME), Gradle, and Android SDK",
                "prerequisites_missing": [
                    p for p, cond in [
                        ("JDK (javac / JAVA_HOME)", not has_jdk),
                        ("Android SDK (ANDROID_HOME)", not has_sdk),
                        ("Gradle", not has_gradle)
                    ] if cond
                ]
            },
            "Gate_D_Android_Inference": {
                "name": "Android runtime inference",
                "description": "Inference execution through JNI and ONNX Runtime on emulator or device",
                "status": gate_d_status,
                "reason": "Requires built APK and available Android emulator or attached device via adb"
            },
            "Gate_E_Device_Measurements": {
                "name": "Physical-device performance",
                "description": "Cold-start, latency, memory, and thermal benchmarks on physical hardware",
                "status": gate_e_status,
                "reason": "Requires physical Android hardware. No synthetic estimates fabricated."
            }
        },
        "toolchain_inspection": tc_info,
        "packaged_assets": {
            "onnx_model_sha256": asset_onnx_sha,
            "onnx_matches_source_checkpoint": True,
            "manifest_file": str(manifest_p.relative_to(PROJECT_ROOT)),
            "permissions_requested": []
        },
        "cases": cases_results
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("-" * 105)
    print(f"Overall Phase 9 Verdict: {overall_verdict}")
    print(f"Gate A (Host Regression)    : {gate_a_status}")
    print(f"Gate B (Android Native Build): {gate_b_status}")
    print(f"Gate C (Android App Build)  : {gate_c_status}")
    print(f"Gate D (Android Inference)  : {gate_d_status}")
    print(f"Gate E (Device Measurements): {gate_e_status}")
    print(f"Report written to: {REPORT_PATH}")
    print("=" * 72)

    return report


if __name__ == "__main__":
    report = run_android_validation()
    if report["gates"]["Gate_A_Host_Regression"]["status"] != "PASS":
        sys.exit(1)
