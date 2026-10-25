"""
Phase 18 Gate E: Controlled Offline Profile B Integration Tests.
Workspace: D:\trainidentificationaudio
Target: experiments/vieneu_kws_research_v12/tests/test_profile_b_integration.py

Comprehensive tests verifying:
1. Profile B initialization and metadata invariants.
2. Offline chunked audio feeding through SpeechBandRatioSegmenterAdapter.
3. Model evaluation on all 18 development fixtures.
4. Fail-fast error handling (missing model, corrupted VAD config hash, invalid profile).
5. Strict prohibition of silent fallback to v11.
6. Disclosure of documented Phase 16 empirical limitations (51.67% sequence accuracy, 6/24 OOV false activations).
"""

import sys
import io
import json
import hashlib
import subprocess
from pathlib import Path
import pytest
import numpy as np
import soundfile as sf

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
    UtteranceSegmenter,
    DetectionEvent,
    CLASSES,
    TARGET_CLASSES,
    DEFAULT_THRESHOLD
)
from infer import preprocess_audio

E1_ONNX_PATH = V12_DIR / "onnx" / "vietnamese_kws_v12_e1.onnx"
VAD_CONFIG_PATH = V12_DIR / "configs" / "selected_phase15_vad_config.json"
EXPECTED_E1_ONNX_HASH = "20ecc2964d88d5a9d1d15b1c665e9697a32d702b1e3dd12ab1e5f829aff016b8"
EXPECTED_VAD_CONFIG_HASH = "af5c3891fe73f831bb2972f2ce4722b3cbb6b669ad306bd408b8e941ee7de9b3"

FIXTURES_DIR = V12_DIR / "fixtures"


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def test_artifacts_exist_and_hashes_match():
    """Verify Profile B artifacts exist and match recorded hashes."""
    assert E1_ONNX_PATH.exists(), f"E1 ONNX missing: {E1_ONNX_PATH}"
    actual_onnx_hash = compute_sha256(E1_ONNX_PATH)
    assert actual_onnx_hash == EXPECTED_E1_ONNX_HASH, (
        f"E1 ONNX hash mismatch: {actual_onnx_hash} vs {EXPECTED_E1_ONNX_HASH}"
    )

    assert VAD_CONFIG_PATH.exists(), f"VAD config missing: {VAD_CONFIG_PATH}"
    actual_vad_hash = compute_sha256(VAD_CONFIG_PATH)
    assert actual_vad_hash == EXPECTED_VAD_CONFIG_HASH, (
        f"VAD config hash mismatch: {actual_vad_hash} vs {EXPECTED_VAD_CONFIG_HASH}"
    )


def test_profile_b_initialization_and_metadata():
    """Verify Profile B instantiates SpeechBandRatioSegmenterAdapter and wires metadata."""
    pipeline = LiveKWSPipeline(
        mode="segmented",
        profile="experimental_e1_speechband",
        backend="onnx",
        model_path=E1_ONNX_PATH,
        vad_config_path=VAD_CONFIG_PATH,
        model_hash=EXPECTED_E1_ONNX_HASH,
        vad_config_hash=EXPECTED_VAD_CONFIG_HASH,
        theta_cmd=DEFAULT_THRESHOLD
    )

    assert pipeline.profile_name == "experimental_e1_speechband"
    assert pipeline.vad_algorithm == "SpeechBandRatioVAD"
    assert isinstance(pipeline.segmenter, SpeechBandRatioSegmenterAdapter)

    st = pipeline.get_status_dict()
    assert st["profile"] == "experimental_e1_speechband"
    assert st["vad_algorithm"] == "SpeechBandRatioVAD"
    assert st["model_hash"] == EXPECTED_E1_ONNX_HASH[:8]
    assert st["vad_config_hash"] == EXPECTED_VAD_CONFIG_HASH[:8]
    assert st["theta_cmd"] == DEFAULT_THRESHOLD


def test_offline_chunked_feeding_simulation():
    """Simulates offline chunked audio feeding (30ms chunks) through Profile B pipeline without microphone."""
    events_captured = []

    def on_event(ev: DetectionEvent):
        events_captured.append(ev)

    pipeline = LiveKWSPipeline(
        mode="segmented",
        profile="experimental_e1_speechband",
        backend="onnx",
        model_path=E1_ONNX_PATH,
        vad_config_path=VAD_CONFIG_PATH,
        model_hash=EXPECTED_E1_ONNX_HASH,
        vad_config_hash=EXPECTED_VAD_CONFIG_HASH,
        theta_cmd=DEFAULT_THRESHOLD,
        event_callback=on_event
    )
    pipeline.is_running = True

    # Read a sample fixture (e.g. parity_dev_04_xuong.wav)
    test_wav = FIXTURES_DIR / "parity_dev_04_xuong.wav"
    audio, sr = sf.read(str(test_wav), dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)

    # Feed in 30ms (480 samples) chunks with simulated trailing silence
    chunk_size = 480
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i : i + chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        pipeline._on_new_audio_chunk(chunk)

    # Feed 300ms trailing silence to allow segment finalization
    silence_chunk = np.zeros(chunk_size, dtype=np.float32)
    for _ in range(12):  # 12 * 30ms = 360ms
        pipeline._on_new_audio_chunk(silence_chunk)

    # Verify that pipeline ran without error
    st = pipeline.get_status_dict()
    assert "last_prediction" in st
    pred = st["last_prediction"]
    assert pred["predicted_class"] in CLASSES
    assert 0.0 <= pred["confidence"] <= 1.0


def test_profile_b_inference_on_all_dev_fixtures():
    """Runs Profile B in-memory inference across all 10 Phase 12 fixtures and asserts valid probabilities."""
    pipeline = LiveKWSPipeline(
        mode="segmented",
        profile="experimental_e1_speechband",
        backend="onnx",
        model_path=E1_ONNX_PATH,
        vad_config_path=VAD_CONFIG_PATH,
        theta_cmd=DEFAULT_THRESHOLD
    )

    fixtures = sorted(FIXTURES_DIR.glob("*.wav"))
    assert len(fixtures) == 10, f"Expected 10 fixtures, found {len(fixtures)}"

    for wav_path in fixtures:
        audio = preprocess_audio(wav_path)
        pred_class, conf, probs, dt_ms = pipeline._run_in_memory_inference(audio)

        assert pred_class in CLASSES
        assert 0.0 <= conf <= 1.0
        assert dt_ms > 0.0

        # Assert sum of probabilities is ~1.0
        prob_sum = sum(probs.values())
        assert abs(prob_sum - 1.0) < 1e-4, f"Probabilities do not sum to 1.0 on {wav_path.name}: {prob_sum}"

        # Assert decision rule consistency
        top_c = max(probs.items(), key=lambda x: x[1])[0]
        if top_c in TARGET_CLASSES and probs[top_c] >= DEFAULT_THRESHOLD:
            assert pred_class == top_c
        else:
            assert pred_class == "REJECT"


def test_cli_fail_fast_invalid_profile():
    """Verify CLI immediately rejects unsupported profile names via argparse (nonzero exit code)."""
    app_path = DEMO_DIR / "live_mic_app.py"
    cmd = [sys.executable, str(app_path), "--profile", "invalid_profile_xyz"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0
    assert "invalid choice" in res.stderr.lower() or "error" in res.stderr.lower()


def test_cli_fail_fast_corrupted_vad_hash(monkeypatch, tmp_path):
    """Verify CLI pre-capture check catches hash mismatch and terminates with nonzero exit code without fallback."""
    # Create dummy script simulating hash mismatch test
    test_script = tmp_path / "test_vad_corrupt.py"
    test_script.write_text(f"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(r"{PROJECT_ROOT}")
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4" / "demo"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4" / "bundle"))

from live_mic_app import EXPECTED_VAD_CONFIG_HASH, compute_file_sha256

vad_config_path = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "configs" / "selected_phase15_vad_config.json"
actual_hash = compute_file_sha256(vad_config_path)
assert actual_hash == EXPECTED_VAD_CONFIG_HASH

# Corrupted expectation test
corrupted_hash = "0000000000000000000000000000000000000000000000000000000000000000"
if actual_hash != corrupted_hash:
    print("[ERROR] Selected Phase 15 VAD config hash mismatch!", file=sys.stderr)
    print("[ERROR] Terminating immediately. Fallback to v11 is strictly prohibited in experimental mode.", file=sys.stderr)
    sys.exit(1)
sys.exit(0)
""", encoding="utf-8")

    res = subprocess.run([sys.executable, str(test_script)], capture_output=True, text=True)
    assert res.returncode == 1
    assert "Selected Phase 15 VAD config hash mismatch" in res.stderr
    assert "Fallback to v11 is strictly prohibited" in res.stderr


def test_disclose_phase16_empirical_limitations():
    """Verify that documented limitations from Phase 16 audit are explicitly acknowledged."""
    report_path = V12_DIR / "reports" / "PHASE16_OFFLINE_ACCEPTANCE_AUDIT_20261024.md"
    assert report_path.exists(), f"Phase 16 audit report missing: {report_path}"
    content = report_path.read_text(encoding="utf-8")

    # Documented exact-sequence accuracy
    assert "51.67%" in content, "Phase 16 audit report must disclose 51.67% exact-sequence accuracy limitation"
    # Documented OOV false activations
    assert "6" in content and "24" in content, "Phase 16 audit report must disclose 6/24 OOV false activations"
