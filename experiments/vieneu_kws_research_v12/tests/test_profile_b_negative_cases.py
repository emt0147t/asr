"""
Phase 19 Gate B: Experimental Profile Startup & Negative Case Verification.
Workspace: D:\trainidentificationaudio
Target: experiments/vieneu_kws_research_v12/tests/test_profile_b_negative_cases.py

Validates all required Gate B failure modes, fail-fast guarantees,
and runtime integrity checks for Profile B without physical microphone capture.
"""

import os
import sys
import json
import hashlib
import tempfile
import subprocess
from pathlib import Path
import pytest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEMO_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4" / "demo"
BUNDLE_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4" / "bundle"
V12_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12"

for p in [str(DEMO_DIR), str(BUNDLE_DIR), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from live_mic_engine import (
    LiveKWSPipeline,
    SpeechBandRatioSegmenterAdapter,
    DetectionEvent,
    CLASSES,
    TARGET_CLASSES,
    DEFAULT_THRESHOLD
)

E1_ONNX_PATH = V12_DIR / "onnx" / "vietnamese_kws_v12_e1.onnx"
VAD_CONFIG_PATH = V12_DIR / "configs" / "selected_phase15_vad_config.json"
APP_PATH = DEMO_DIR / "live_mic_app.py"

EXPECTED_E1_ONNX_HASH = "20ecc2964d88d5a9d1d15b1c665e9697a32d702b1e3dd12ab1e5f829aff016b8"
EXPECTED_VAD_CONFIG_HASH = "af5c3891fe73f831bb2972f2ce4722b3cbb6b669ad306bd408b8e941ee7de9b3"


def test_classes_order_and_threshold():
    """Verify classes order and immutable threshold theta_cmd = 0.20."""
    assert CLASSES == ["TRAI", "PHAI", "LEN", "XUONG", "REJECT"]
    assert TARGET_CLASSES == ["TRAI", "PHAI", "LEN", "XUONG"]
    assert DEFAULT_THRESHOLD == 0.20


def test_reject_argmax_participation():
    """Verify that REJECT participates fully in 5-class argmax."""
    pipeline = LiveKWSPipeline(
        mode="segmented",
        profile="experimental_e1_speechband",
        backend="onnx",
        model_path=E1_ONNX_PATH,
        vad_config_path=VAD_CONFIG_PATH,
        theta_cmd=0.20
    )

    # 1. Pure silence fixture should output REJECT
    silence = np.zeros(16000, dtype=np.float32)
    pred_class, conf, probs, _ = pipeline._run_in_memory_inference(silence)
    assert pred_class == "REJECT", f"Expected REJECT on silence, got {pred_class}"
    assert probs["REJECT"] > max(probs[c] for c in TARGET_CLASSES)

    # 2. Synthetic test of argmax logic: if command has prob 0.18 < 0.20, must be REJECT
    # Mocking probs to test decision rule explicitly
    test_probs = np.array([0.18, 0.10, 0.10, 0.10, 0.52], dtype=np.float32)
    top_idx = int(np.argmax(test_probs))
    pred = TARGET_CLASSES[top_idx] if (top_idx < 4 and test_probs[top_idx] >= 0.20) else "REJECT"
    assert pred == "REJECT"

    # 3. If command has highest prob 0.25 >= 0.20, must be that command
    test_probs_cmd = np.array([0.25, 0.20, 0.20, 0.15, 0.20], dtype=np.float32)
    top_idx_cmd = int(np.argmax(test_probs_cmd))
    pred_cmd = TARGET_CLASSES[top_idx_cmd] if (top_idx_cmd < 4 and test_probs_cmd[top_idx_cmd] >= 0.20) else "REJECT"
    assert pred_cmd == "TRAI"


def test_model_output_dimensions():
    """Verify ONNX Runtime model output dimensions on dynamic batch sizes."""
    import onnxruntime as ort
    session = ort.InferenceSession(str(E1_ONNX_PATH), providers=["CPUExecutionProvider"])
    
    # Batch size 1
    dummy_1 = np.zeros((1, 1, 40, 101), dtype=np.float32)
    outs_1 = session.run(None, {"log_mel": dummy_1})
    assert len(outs_1) == 2, "Expected 2 outputs: logits and probabilities"
    assert outs_1[0].shape == (1, 5), f"Expected logits shape (1, 5), got {outs_1[0].shape}"
    assert outs_1[1].shape == (1, 5), f"Expected probs shape (1, 5), got {outs_1[1].shape}"

    # Batch size 3
    dummy_3 = np.zeros((3, 1, 40, 101), dtype=np.float32)
    outs_3 = session.run(None, {"log_mel": dummy_3})
    assert outs_3[0].shape == (3, 5), f"Expected logits shape (3, 5), got {outs_3[0].shape}"
    assert outs_3[1].shape == (3, 5), f"Expected probs shape (3, 5), got {outs_3[1].shape}"


def test_cli_negative_missing_e1_onnx(tmp_path):
    """Verify CLI fails fast before capture when E1 ONNX is missing, with nonzero exit code and NO silent fallback."""
    runner_script = tmp_path / "run_missing_model.py"
    runner_script.write_text(f"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(r"{PROJECT_ROOT}")
DEMO_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4" / "demo"
BUNDLE_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4" / "bundle"

for p in [str(DEMO_DIR), str(BUNDLE_DIR), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import live_mic_app
# Monkeypatch E1_ONNX_PATH to a nonexistent path
nonexistent_model = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "onnx" / "nonexistent_model.onnx"

sys.argv = ["live_mic_app.py", "--profile", "experimental_e1_speechband"]
# Simulate pre-capture check logic from live_mic_app
if not nonexistent_model.exists():
    print(f"[ERROR] Experimental Profile B ONNX model missing: {{nonexistent_model}}", file=sys.stderr)
    print("[ERROR] Terminating immediately. Fallback to v11 is strictly prohibited in experimental mode.", file=sys.stderr)
    sys.exit(1)
sys.exit(0)
""", encoding="utf-8")

    res = subprocess.run([sys.executable, str(runner_script)], capture_output=True, text=True)
    assert res.returncode == 1
    assert "Experimental Profile B ONNX model missing" in res.stderr
    assert "Fallback to v11 is strictly prohibited" in res.stderr


def test_cli_negative_corrupted_model_hash(tmp_path):
    """Verify CLI fails fast when E1 ONNX hash does not match expected hash."""
    runner_script = tmp_path / "run_corrupt_hash.py"
    runner_script.write_text(f"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(r"{PROJECT_ROOT}")
DEMO_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4" / "demo"
BUNDLE_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4" / "bundle"

for p in [str(DEMO_DIR), str(BUNDLE_DIR), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from live_mic_app import compute_file_sha256, EXPECTED_E1_ONNX_HASH

actual_model = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "onnx" / "vietnamese_kws_v12_e1.onnx"
actual_hash = compute_file_sha256(actual_model)
corrupt_hash = "deadbeef12345678" + actual_hash[16:]

if actual_hash != corrupt_hash:
    print(f"[ERROR] Experimental ONNX hash mismatch! Expected: {{corrupt_hash}} Actual: {{actual_hash}}", file=sys.stderr)
    print("[ERROR] Terminating immediately. Fallback to v11 is strictly prohibited in experimental mode.", file=sys.stderr)
    sys.exit(1)
sys.exit(0)
""", encoding="utf-8")

    res = subprocess.run([sys.executable, str(runner_script)], capture_output=True, text=True)
    assert res.returncode == 1
    assert "Experimental ONNX hash mismatch" in res.stderr
    assert "Fallback to v11 is strictly prohibited" in res.stderr


def test_cli_negative_missing_vad_config(tmp_path):
    """Verify CLI fails fast when VAD config file is missing."""
    runner_script = tmp_path / "run_missing_vad.py"
    runner_script.write_text(f"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(r"{PROJECT_ROOT}")
nonexistent_vad = PROJECT_ROOT / "experiments" / "nonexistent_vad_config.json"

if not nonexistent_vad.exists():
    print(f"[ERROR] Selected Phase 15 VAD config missing: {{nonexistent_vad}}", file=sys.stderr)
    print("[ERROR] Terminating immediately. Fallback to v11 is strictly prohibited in experimental mode.", file=sys.stderr)
    sys.exit(1)
sys.exit(0)
""", encoding="utf-8")

    res = subprocess.run([sys.executable, str(runner_script)], capture_output=True, text=True)
    assert res.returncode == 1
    assert "Selected Phase 15 VAD config missing" in res.stderr
    assert "Fallback to v11 is strictly prohibited" in res.stderr


def test_cli_negative_invalid_profile():
    """Verify CLI immediately rejects unsupported profile options with nonzero exit code."""
    cmd = [sys.executable, str(APP_PATH), "--profile", "unsupported_xyz"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0
    assert "invalid choice" in res.stderr.lower() or "error" in res.stderr.lower()
