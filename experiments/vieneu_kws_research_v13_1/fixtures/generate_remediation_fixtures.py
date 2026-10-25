#!/usr/bin/env python3
"""
Phase 13.1 Non-Locked Synthetic Fixture Generator.
Workspace: D:\trainidentificationaudio
Target: experiments/vieneu_kws_research_v13_1/fixtures/generate_remediation_fixtures.py

Generates non-locked development fixtures covering:
1. Digital silence (exact zeros)
2. Quiet ambient noise (peak ~0.005)
3. Attenuated commands (peaks 0.05 - 0.10)
4. Normal-amplitude commands (peak ~0.891)
5. High-amplitude inputs (peak > 1.0) for clamping tests
"""

import sys
from pathlib import Path
import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIXTURES_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_1" / "fixtures"
SOURCE_DEV_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "fixtures"


def main():
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating Phase 13.1 remediation development fixtures...")

    # 1. Digital silence
    silence = np.zeros(16000, dtype=np.float32)
    sf.write(FIXTURES_DIR / "dev_01_digital_silence.wav", silence, 16000, subtype="PCM_16")
    print("  [OK] dev_01_digital_silence.wav (zeros)")

    # 2. Quiet ambient noise (peak ~0.005)
    rng = np.random.RandomState(42)
    ambient = rng.normal(0, 0.0012, 16000).astype(np.float32)
    ambient = np.clip(ambient, -0.005, 0.005)
    sf.write(FIXTURES_DIR / "dev_02_quiet_ambient_room.wav", ambient, 16000, subtype="PCM_16")
    print(f"  [OK] dev_02_quiet_ambient_room.wav (peak: {np.max(np.abs(ambient)):.5f})")

    # Load source dev commands
    trai_audio, sr = sf.read(SOURCE_DEV_DIR / "parity_dev_01_trai.wav", dtype="float32")
    phai_audio, sr = sf.read(SOURCE_DEV_DIR / "parity_dev_02_phai.wav", dtype="float32")
    len_audio, sr = sf.read(SOURCE_DEV_DIR / "parity_dev_03_len.wav", dtype="float32")
    xuong_audio, sr = sf.read(SOURCE_DEV_DIR / "parity_dev_04_xuong.wav", dtype="float32")

    # 3. Attenuated TRAI (peak 0.10)
    att_trai = trai_audio * (0.10 / np.max(np.abs(trai_audio)))
    sf.write(FIXTURES_DIR / "dev_03_attenuated_trai_peak010.wav", att_trai, 16000, subtype="PCM_16")
    print(f"  [OK] dev_03_attenuated_trai_peak010.wav (peak: {np.max(np.abs(att_trai)):.3f})")

    # 4. Attenuated PHAI (peak 0.05)
    att_phai = phai_audio * (0.05 / np.max(np.abs(phai_audio)))
    sf.write(FIXTURES_DIR / "dev_04_attenuated_phai_peak005.wav", att_phai, 16000, subtype="PCM_16")
    print(f"  [OK] dev_04_attenuated_phai_peak005.wav (peak: {np.max(np.abs(att_phai)):.3f})")

    # 5. Normal-amplitude LEN (peak 0.891)
    norm_len = len_audio * (0.89125 / np.max(np.abs(len_audio)))
    sf.write(FIXTURES_DIR / "dev_05_normal_len_peak089.wav", norm_len, 16000, subtype="PCM_16")
    print(f"  [OK] dev_05_normal_len_peak089.wav (peak: {np.max(np.abs(norm_len)):.3f})")

    # 6. Normal-amplitude XUONG (peak 0.891)
    norm_xuong = xuong_audio * (0.89125 / np.max(np.abs(xuong_audio)))
    sf.write(FIXTURES_DIR / "dev_06_normal_xuong_peak089.wav", norm_xuong, 16000, subtype="PCM_16")
    print(f"  [OK] dev_06_normal_xuong_peak089.wav (peak: {np.max(np.abs(norm_xuong)):.3f})")

    # 7. High-amplitude TRAI (peak 1.8)
    high_trai = trai_audio * (1.8 / np.max(np.abs(trai_audio)))
    # Write as float32 WAV so samples > 1.0 are preserved without DAC clipping
    sf.write(FIXTURES_DIR / "dev_07_high_amplitude_trai_peak180.wav", high_trai, 16000, subtype="FLOAT")
    print(f"  [OK] dev_07_high_amplitude_trai_peak180.wav (peak: {np.max(np.abs(high_trai)):.3f})")

    # 8. High-amplitude XUONG (peak 2.5)
    high_xuong = xuong_audio * (2.5 / np.max(np.abs(xuong_audio)))
    sf.write(FIXTURES_DIR / "dev_08_high_amplitude_xuong_peak250.wav", high_xuong, 16000, subtype="FLOAT")
    print(f"  [OK] dev_08_high_amplitude_xuong_peak250.wav (peak: {np.max(np.abs(high_xuong)):.3f})")

    print("[SUCCESS] All 8 remediation fixtures generated.")


if __name__ == "__main__":
    main()
