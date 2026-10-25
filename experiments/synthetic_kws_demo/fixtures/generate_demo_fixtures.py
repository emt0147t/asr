"""
Generates dedicated synthetic demonstration and test fixtures for Phase 5 prototype.
All fixtures are derived from the parametric Onset-Aware Synthetic Acoustic Generator
using a dedicated demonstration speaker ID (SPK_DEMO_001) to ensure disjointness
from training and validation calibration sets.
"""

import sys
import os
import hashlib
from pathlib import Path

if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd
import soundfile as sf

from experiments.synthetic_kws_v4.onset_aware_generator import (
    generate_vietnamese_onset_aware_utterance,
    generate_synthetic_silence_clip
)


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest().upper()


def generate_demo_fixtures():
    fixtures_dir = Path("experiments/synthetic_kws_demo/fixtures")
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    
    np.random.seed(12345)
    
    # Dedicated demo speaker
    spk_demo = {
        "speaker_id": "SPK_DEMO_001",
        "gender": "MALE",
        "f0_base": 130.0,
        "dialect": "Northern"
    }
    
    # 1. Standard 6-class native demonstration fixtures (16,000 Hz, 1 channel, 16,000 samples)
    valid_configs = [
        ("demo_01_trai.wav", "trái", 0, "TRAI", "MOVE_LEFT", 0.50),
        ("demo_02_phai.wav", "phải", 1, "PHAI", "MOVE_RIGHT", 0.50),
        ("demo_03_len.wav", "lên", 2, "LEN", "JUMP", 0.50),
        ("demo_04_xuong.wav", "xuống", 3, "XUONG", "CROUCH", 0.50),
        ("demo_05_unknown.wav", "chào", 4, "UNKNOWN", "NO_ACTION", 0.50),
    ]
    
    records = []
    
    for fname, word, cid, clabel, action, dur in valid_configs:
        raw = generate_vietnamese_onset_aware_utterance(
            word=word, sr=16000, f0_base=spk_demo["f0_base"], duration_sec=dur, dialect=spk_demo["dialect"]
        )
        # Native V4 representation: left-aligned, right-padded to 16,000 samples
        if len(raw) < 16000:
            audio = np.pad(raw, (0, 16000 - len(raw)))
        else:
            audio = raw[:16000]
            
        out_p = fixtures_dir / fname
        sf.write(str(out_p), audio, 16000, subtype="PCM_16")
        
        records.append({
            "fixture_id": fname.split(".")[0],
            "filename": fname,
            "class_id": cid,
            "class_label": clabel,
            "intended_game_action": action,
            "sample_rate": 16000,
            "channels": 1,
            "samples": len(audio),
            "duration_sec": 1.0,
            "is_valid_pcm16": True,
            "sha256": sha256_file(out_p),
            "provenance": "synthetic_onset_aware_demo_fixture",
            "speaker_id": spk_demo["speaker_id"],
            "notes": f"Dedicated {clabel} demonstration fixture synthesized for Phase 5 prototype."
        })
        
    # 6. SILENCE class (ambient dither fixture, full 16,000 buffer)
    silence_audio = generate_synthetic_silence_clip(sr=16000, duration_sec=1.0, mode="dither")
    out_silence = fixtures_dir / "demo_06_silence.wav"
    sf.write(str(out_silence), silence_audio, 16000, subtype="PCM_16")
    records.append({
        "fixture_id": "demo_06_silence",
        "filename": "demo_06_silence.wav",
        "class_id": 5,
        "class_label": "SILENCE",
        "intended_game_action": "NO_ACTION",
        "sample_rate": 16000,
        "channels": 1,
        "samples": len(silence_audio),
        "duration_sec": 1.0,
        "is_valid_pcm16": True,
        "sha256": sha256_file(out_silence),
        "provenance": "synthetic_ambient_demo_fixture",
        "speaker_id": spk_demo["speaker_id"],
        "notes": "Full-buffer synthetic ambient background dither fixture without onset."
    })
    
    # 2. Malformed / Invalid fixtures for error-handling integration testing
    # A. Stereo audio (2 channels)
    stereo_audio = np.stack([audio, audio], axis=-1)
    out_stereo = fixtures_dir / "invalid_stereo.wav"
    sf.write(str(out_stereo), stereo_audio, 16000, subtype="PCM_16")
    records.append({
        "fixture_id": "invalid_stereo",
        "filename": "invalid_stereo.wav",
        "class_id": -1,
        "class_label": "INVALID",
        "intended_game_action": "ERROR_EXPECTED",
        "sample_rate": 16000,
        "channels": 2,
        "samples": 16000,
        "duration_sec": 1.0,
        "is_valid_pcm16": False,
        "sha256": sha256_file(out_stereo),
        "provenance": "test_harness_negative_fixture",
        "speaker_id": "N/A",
        "notes": "Malformed test fixture: 2 channels (stereo), expected rejection by AudioValidator."
    })
    
    # B. Wrong sample rate (8,000 Hz)
    out_sr8k = fixtures_dir / "invalid_sample_rate_8k.wav"
    sf.write(str(out_sr8k), audio[:8000], 8000, subtype="PCM_16")
    records.append({
        "fixture_id": "invalid_sample_rate_8k",
        "filename": "invalid_sample_rate_8k.wav",
        "class_id": -1,
        "class_label": "INVALID",
        "intended_game_action": "ERROR_EXPECTED",
        "sample_rate": 8000,
        "channels": 1,
        "samples": 8000,
        "duration_sec": 1.0,
        "is_valid_pcm16": False,
        "sha256": sha256_file(out_sr8k),
        "provenance": "test_harness_negative_fixture",
        "speaker_id": "N/A",
        "notes": "Malformed test fixture: 8 kHz sample rate, expected rejection by AudioValidator."
    })
    
    # C. Truncated duration (8,000 samples / 0.5s instead of 16,000 samples)
    out_short = fixtures_dir / "invalid_short_duration.wav"
    sf.write(str(out_short), audio[:8000], 16000, subtype="PCM_16")
    records.append({
        "fixture_id": "invalid_short_duration",
        "filename": "invalid_short_duration.wav",
        "class_id": -1,
        "class_label": "INVALID",
        "intended_game_action": "ERROR_EXPECTED",
        "sample_rate": 16000,
        "channels": 1,
        "samples": 8000,
        "duration_sec": 0.5,
        "is_valid_pcm16": False,
        "sha256": sha256_file(out_short),
        "provenance": "test_harness_negative_fixture",
        "speaker_id": "N/A",
        "notes": "Malformed test fixture: 8,000 samples (0.5s), expected rejection by AudioValidator."
    })
    
    # D. Empty / 0-byte file
    out_empty = fixtures_dir / "invalid_empty.wav"
    out_empty.write_bytes(b"")
    records.append({
        "fixture_id": "invalid_empty",
        "filename": "invalid_empty.wav",
        "class_id": -1,
        "class_label": "INVALID",
        "intended_game_action": "ERROR_EXPECTED",
        "sample_rate": 0,
        "channels": 0,
        "samples": 0,
        "duration_sec": 0.0,
        "is_valid_pcm16": False,
        "sha256": sha256_file(out_empty),
        "provenance": "test_harness_negative_fixture",
        "speaker_id": "N/A",
        "notes": "Malformed test fixture: 0-byte empty file, expected rejection by AudioValidator."
    })
    
    df_manifest = pd.DataFrame(records)
    manifest_p = fixtures_dir / "fixture_manifest.csv"
    df_manifest.to_csv(manifest_p, index=False)
    print(f"Generated {len(df_manifest)} fixtures in {fixtures_dir}")
    print(f"Saved manifest to: {manifest_p}")
    return manifest_p


if __name__ == "__main__":
    generate_demo_fixtures()
