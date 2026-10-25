"""
Pre-synthesizes missing OOV confusers (cửa, bàn, nhà for train; ghế for test_unseen)
using local VieNeu-TTS v3 Turbo.
"""

import sys
import os
import io
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy import signal

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
VIENEU_PKG_DIR = PROJECT_ROOT / ".venv_vieneu" / "Lib" / "site-packages"
if str(VIENEU_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(VIENEU_PKG_DIR))

from vieneu import Vieneu

def generate_confusers():
    staging_dir = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "audio" / "staging_oov"
    staging_dir.mkdir(parents=True, exist_ok=True)

    tts = Vieneu(mode="v3turbo")

    train_words = [
        ("cửa", "Adam"), ("cửa", "Hải Đăng"), ("cửa", "Minh Đức"), ("cửa", "Thanh Bình"),
        ("bàn", "Adam"), ("bàn", "Hải Đăng"), ("bàn", "Minh Đức"), ("bàn", "Thanh Bình"),
        ("nhà", "Adam"), ("nhà", "Hải Đăng"), ("nhà", "Minh Đức"), ("nhà", "Thanh Bình")
    ]

    test_words = [
        ("ghế", "Kim Thanh"), ("ghế", "Quốc Tuấn"), ("ghế", "Quỳnh Anh"), ("ghế", "Đức Trí")
    ]

    all_words = [(w, v, "train") for w, v in train_words] + [(w, v, "test") for w, v in test_words]

    for word, voice, split in all_words:
        raw_48k = tts.infer(word, voice=voice)
        num_target = int(round(len(raw_48k) * 16000.0 / 48000.0))
        resampled = signal.resample(raw_48k, num_target).astype(np.float32)
        
        peak = np.max(np.abs(resampled))
        if peak > 0.95:
            resampled = resampled * (0.95 / peak)
        elif peak < 0.20 and peak > 0.01:
            resampled = resampled * (0.70 / peak)
            
        if len(resampled) < 16000:
            lead_pad = min(2400, (16000 - len(resampled)) // 2)
            trail_pad = 16000 - len(resampled) - lead_pad
            kws_1s = np.pad(resampled, (lead_pad, trail_pad))
        else:
            kws_1s = resampled[:16000]
            
        safe_voice = voice.replace(" ", "_")
        out_path = staging_dir / f"oov_{split}_{word}_{safe_voice}.wav"
        sf.write(str(out_path), kws_1s.astype(np.float32), 16000)
        print(f"Synthesized: {out_path.name} ({word}, voice={voice}, split={split})")

if __name__ == "__main__":
    generate_confusers()
