"""
Phase C: Independent Determinism and Reproducibility Verification
Regenerates sample clips from scratch using OnnxV3LiteEngine with identical seeds and parameters,
comparing maximum absolute sample difference and SHA-256 hashes against saved audio files.
"""

import os
import sys
import hashlib
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal
from vieneu import Vieneu

sys.stdout.reconfigure(encoding="utf-8")

def compute_sha256(filepath):
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def verify_determinism():
    manifest_p = os.path.join(os.path.dirname(__file__), "manifest.csv")
    df = pd.read_csv(manifest_p)

    test_sids = [
        df[df["class_label"] == "TRAI"].iloc[0]["sample_id"],
        df[df["class_label"] == "PHAI"].iloc[0]["sample_id"],
        df[df["class_label"] == "LEN"].iloc[0]["sample_id"],
        df[df["class_label"] == "XUONG"].iloc[0]["sample_id"]
    ]

    tts = Vieneu()
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    results = []
    print("=" * 70)
    print("Testing Independent Regeneration Reproducibility (4 classes)")
    print("=" * 70)

    for sid in test_sids:
        row = df[df["sample_id"] == sid].iloc[0]
        vid = row["voice_id"]
        text = row["text"]
        cls_label = row["class_label"]
        seed = int(row["seed"])
        temp = float(row["temperature"])
        top_p = float(row["top_p"])
        rep_pen = float(row["repetition_penalty"])
        top_k = 25

        raw_path = os.path.join(base_dir, row["raw_wav_path"])
        kws_path = os.path.join(base_dir, row["kws_wav_path"])

        raw_saved, sr_raw = sf.read(raw_path)
        kws_saved, sr_kws = sf.read(kws_path)

        # Regenerate independently
        np.random.seed(seed)
        raw_new = tts.infer(
            text,
            voice=vid,
            temperature=temp,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=rep_pen
        )

        # Check raw WAV byte-level match via in-memory PCM-16 write
        import io
        raw_buf = io.BytesIO()
        sf.write(raw_buf, raw_new, sr_raw, subtype="PCM_16", format="WAV")
        raw_bytes = raw_buf.getvalue()
        raw_hash_new = hashlib.sha256(raw_bytes).hexdigest()
        raw_hash_match = (raw_hash_new == row["raw_sha256"])

        raw_pcm_read, _ = sf.read(io.BytesIO(raw_bytes))
        max_raw_diff = float(np.max(np.abs(raw_saved - raw_pcm_read)))

        # Derived KWS alignment
        resampled_16k = signal.resample_poly(raw_new, 1, 3).astype(np.float32)
        n_resampled = len(resampled_16k)
        kws_samples = 16000
        kws_new = np.zeros(kws_samples, dtype=np.float32)
        onset_sample = int(0.150 * 16000)
        if onset_sample + n_resampled > kws_samples - 800:
            onset_sample = max(800, kws_samples - n_resampled - 800)
        kws_new[onset_sample : onset_sample + n_resampled] = resampled_16k

        peak = float(np.max(np.abs(kws_new)))
        if peak > 0.95:
            kws_new = kws_new * (0.95 / peak)

        kws_buf = io.BytesIO()
        sf.write(kws_buf, kws_new, 16000, subtype="PCM_16", format="WAV")
        kws_bytes = kws_buf.getvalue()
        kws_hash_new = hashlib.sha256(kws_bytes).hexdigest()
        kws_hash_match = (kws_hash_new == row["kws_sha256"])

        kws_pcm_read, _ = sf.read(io.BytesIO(kws_bytes))
        max_kws_diff = float(np.max(np.abs(kws_saved - kws_pcm_read)))

        bit_exact = (raw_hash_match and kws_hash_match)
        status = "BIT_EXACT" if bit_exact else ("NUMERICALLY_CLOSE" if max_raw_diff < 1e-4 else "DIVERGENT")
        res = {
            "sample_id": sid,
            "class_label": cls_label,
            "voice_id": vid,
            "seed": seed,
            "raw_len_saved": len(raw_saved),
            "raw_len_new": len(raw_new),
            "max_raw_diff": max_raw_diff,
            "max_kws_diff": max_kws_diff,
            "status": status
        }
        results.append(res)
        print(f"[{status}] {sid} ({cls_label}, {vid}): max_raw_diff = {max_raw_diff:.8e}, max_kws_diff = {max_kws_diff:.8e}")

    all_bit_exact = all(r["status"] == "BIT_EXACT" for r in results)
    print("=" * 70)
    print(f"Overall Determinism Verdict: {'PASS (Bit-Exact Determinism Confirmed)' if all_bit_exact else 'INSPECT'}")
    print("=" * 70)
    return all_bit_exact

if __name__ == "__main__":
    success = verify_determinism()
    sys.exit(0 if success else 1)
