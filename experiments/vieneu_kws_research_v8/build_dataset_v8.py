import os
import sys
import json
import csv
import time
import hashlib
import numpy as np
import soundfile as sf
from scipy import signal

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = r"D:\trainidentificationaudio"
sys.path.insert(0, BASE_DIR)

V6_EXP_DIR = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v6")
V8_EXP_DIR = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v8")

DATA_DIR = os.path.join(V8_EXP_DIR, "data")
AUDIO_RAW_DIR = os.path.join(DATA_DIR, "audio", "raw_48k")
AUDIO_KWS_DIR = os.path.join(DATA_DIR, "audio", "kws_16k")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")

# 4 Held-Out Voices for Test
TEST_VOICES = ["Kim Thanh", "Quốc Tuấn", "Quỳnh Anh", "Đức Trí"]

TARGET_COMMAND_TEXTS = {
    "TRAI": "trái",
    "PHAI": "phải",
    "LEN": "lên",
    "XUONG": "xuống"
}

# 80 Brand-New Disjoint Non-Target Vietnamese Texts
from scratch_prepare_v8_texts import NEW_TEST_TEXTS


def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def detect_speech_bounds(audio_16k: np.ndarray, sr: int = 16000):
    frame_len = int(0.025 * sr)
    hop_len = int(0.010 * sr)
    num_frames = (len(audio_16k) - frame_len) // hop_len + 1
    if num_frames <= 0:
        return 0.0, len(audio_16k) / sr

    frames = np.array([audio_16k[i * hop_len : i * hop_len + frame_len] for i in range(num_frames)])
    energy = np.sum(frames ** 2, axis=1)
    max_e = np.max(energy) if len(energy) > 0 else 0
    if max_e < 1e-7:
        return 0.0, len(audio_16k) / sr

    thresh = max_e * 0.01
    active = np.where(energy > thresh)[0]
    if len(active) == 0:
        return 0.0, len(audio_16k) / sr

    onset_s = round(float(active[0] * hop_len) / sr, 3)
    offset_s = round(float((active[-1] * hop_len) + frame_len) / sr, 3)
    return onset_s, round(offset_s - onset_s, 3)


def format_to_1s(audio: np.ndarray, orig_sr: int, target_sr: int = 16000):
    if orig_sr != target_sr:
        num_target_samples = int(round(len(audio) * float(target_sr) / orig_sr))
        audio = signal.resample(audio, num_target_samples)

    peak = np.max(np.abs(audio))
    if peak > 0:
        target_peak = 10.0 ** (-1.0 / 20.0)
        audio = audio * (target_peak / peak)

    target_samples = 16000
    if len(audio) < target_samples:
        pad_total = target_samples - len(audio)
        pad_front = pad_total // 2
        pad_back = pad_total - pad_front
        audio = np.pad(audio, (pad_front, pad_back), mode="constant", constant_values=0.0)
    elif len(audio) > target_samples:
        frame_len = int(0.025 * target_sr)
        hop_len = int(0.010 * target_sr)
        num_frames = (len(audio) - frame_len) // hop_len + 1
        energies = [np.sum(audio[i * hop_len : i * hop_len + frame_len] ** 2) for i in range(num_frames)]
        center_frame = int(np.argmax(energies))
        center_sample = center_frame * hop_len + frame_len // 2
        start = max(0, center_sample - target_samples // 2)
        if start + target_samples > len(audio):
            start = len(audio) - target_samples
        audio = audio[start : start + target_samples]

    onset_s, dur_s = detect_speech_bounds(audio, target_sr)
    return audio.astype(np.float32), onset_s, dur_s


def generate_synthetic_background(duration_samples: int, family: str, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    if family == "near_silence":
        audio = rng.normal(0, 0.0008, duration_samples)
    elif family == "hvac_hum":
        t = np.arange(duration_samples) / 16000.0
        audio = (
            0.025 * np.sin(2 * np.pi * 50 * t) +
            0.015 * np.sin(2 * np.pi * 100 * t) +
            0.008 * np.sin(2 * np.pi * 150 * t) +
            rng.normal(0, 0.004, duration_samples)
        )
    elif family == "office_ambient":
        audio = rng.normal(0, 0.012, duration_samples)
        b, a = signal.butter(4, [200.0 / 8000.0, 3500.0 / 8000.0], btype="band")
        audio = signal.lfilter(b, a, audio)
    elif family == "airflow_pink":
        white = rng.normal(0, 0.015, duration_samples)
        b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        a = [1.0, -2.494956002, 2.017265875, -0.522189400]
        audio = signal.lfilter(b, a, white)
    else:
        audio = rng.normal(0, 0.002, duration_samples)
    peak = np.max(np.abs(audio))
    if peak > 0.08:
        audio = audio * (0.08 / peak)
    return audio.astype(np.float32)


def build_test_dataset():
    print("=" * 80)
    print("Phase 74: Synthesizing Fresh Locked Benchmark Dataset v8")
    print("=" * 80)

    os.makedirs(AUDIO_RAW_DIR, exist_ok=True)
    os.makedirs(AUDIO_KWS_DIR, exist_ok=True)
    os.makedirs(SPLITS_DIR, exist_ok=True)

    print("\n--- Initializing VieNeu TTS Engine (v3turbo) ---")
    from vieneu import Vieneu
    tts = Vieneu(mode="v3turbo")

    test_records = []
    t0_start = time.time()

    # Part A: 160 Fresh Target Commands (40 clips per keyword)
    print("\n--- Synthesizing 160 Fresh Target Command Clips ---")
    label_map = {"TRAI": 0, "PHAI": 1, "LEN": 2, "XUONG": 3}
    for kw, text in TARGET_COMMAND_TEXTS.items():
        cid = label_map[kw]
        for v_idx, vid in enumerate(TEST_VOICES):
            for r in range(10):
                sid = f"v8_cmd_test_{kw.lower()}_{v_idx:02d}_{r:02d}"
                raw_path = os.path.join(AUDIO_RAW_DIR, f"{sid}_raw48k.wav")
                kws_path = os.path.join(AUDIO_KWS_DIR, f"{sid}_kws16k.wav")

                temp = round(0.70 + 0.02 * (r % 5), 2)
                top_p = round(0.90 + 0.01 * (r % 5), 2)

                if not os.path.exists(kws_path):
                    raw_audio = tts.infer(text, voice=vid, temperature=temp, top_p=top_p, repetition_penalty=1.20)
                    kws_audio, onset_s, dur_s = format_to_1s(raw_audio, 48000, 16000)
                    sf.write(raw_path, raw_audio, 48000, subtype="PCM_16")
                    sf.write(kws_path, kws_audio, 16000, subtype="PCM_16")
                else:
                    kws_audio, _ = sf.read(kws_path, dtype="float32")
                    onset_s, dur_s = detect_speech_bounds(kws_audio, 16000)

                kws_h = compute_sha256(kws_path)
                test_records.append({
                    "sample_id": sid,
                    "dataset_version": "research_v8.0_bounded",
                    "split": "test",
                    "voice_id": vid,
                    "canonical_class": kw,
                    "class_id": cid,
                    "bounded_label": kw,
                    "bounded_class_id": cid,
                    "confusion_family": "target_command",
                    "transcription": text,
                    "raw_wav_path": os.path.relpath(raw_path, BASE_DIR),
                    "kws_wav_path": os.path.relpath(kws_path, BASE_DIR),
                    "duration_samples": 16000,
                    "sample_rate": 16000,
                    "speech_onset_sec": onset_s,
                    "speech_duration_sec": dur_s,
                    "sha256": kws_h
                })
    print(f"Synthesized {len(test_records)} command clips.")

    # Part B: 480 Fresh Non-Target Distractor Clips (cycling 80 new texts across 4 held-out voices)
    print("\n--- Synthesizing 480 Fresh Non-Target Distractor Clips ---")
    for i in range(480):
        t_info = NEW_TEST_TEXTS[i % len(NEW_TEST_TEXTS)]
        vid = TEST_VOICES[i % len(TEST_VOICES)]
        r_idx = i // len(TEST_VOICES)

        sid = f"v8_unk_test_{t_info['confusion_family'][:4]}_{i:03d}"
        raw_path = os.path.join(AUDIO_RAW_DIR, f"{sid}_raw48k.wav")
        kws_path = os.path.join(AUDIO_KWS_DIR, f"{sid}_kws16k.wav")

        temp = round(0.70 + 0.02 * (r_idx % 5), 2)
        top_p = round(0.90 + 0.01 * (r_idx % 5), 2)

        if not os.path.exists(kws_path):
            raw_audio = tts.infer(t_info["text"], voice=vid, temperature=temp, top_p=top_p, repetition_penalty=1.20)
            kws_audio, onset_s, dur_s = format_to_1s(raw_audio, 48000, 16000)
            sf.write(raw_path, raw_audio, 48000, subtype="PCM_16")
            sf.write(kws_path, kws_audio, 16000, subtype="PCM_16")
        else:
            kws_audio, _ = sf.read(kws_path, dtype="float32")
            onset_s, dur_s = detect_speech_bounds(kws_audio, 16000)

        kws_h = compute_sha256(kws_path)
        test_records.append({
            "sample_id": sid,
            "dataset_version": "research_v8.0_bounded",
            "split": "test",
            "voice_id": vid,
            "canonical_class": "UNKNOWN",
            "class_id": 4,
            "bounded_label": "REJECT",
            "bounded_class_id": 4,
            "confusion_family": t_info["confusion_family"],
            "transcription": t_info["text"],
            "raw_wav_path": os.path.relpath(raw_path, BASE_DIR),
            "kws_wav_path": os.path.relpath(kws_path, BASE_DIR),
            "duration_samples": 16000,
            "sample_rate": 16000,
            "speech_onset_sec": onset_s,
            "speech_duration_sec": dur_s,
            "sha256": kws_h
        })

        if (i + 1) % 80 == 0:
            print(f"  Synthesized {i+1}/480 non-target clips ({time.time()-t0_start:.1f}s elapsed)...")

    # Part C: 80 Fresh Synthetic Silence Clips (20 per acoustic profile)
    print("\n--- Generating 80 Fresh Synthetic Silence Clips ---")
    noise_families = ["near_silence", "hvac_hum", "office_ambient", "airflow_pink"]
    for fam in noise_families:
        for r in range(20):
            sid = f"v8_sil_test_{fam[:4]}_{r:02d}"
            kws_path = os.path.join(AUDIO_KWS_DIR, f"{sid}_kws16k.wav")

            if not os.path.exists(kws_path):
                audio = generate_synthetic_background(16000, fam, seed=84000 + hash(fam) % 10000 + r)
                sf.write(kws_path, audio, 16000, subtype="PCM_16")

            kws_h = compute_sha256(kws_path)
            test_records.append({
                "sample_id": sid,
                "dataset_version": "research_v8.0_bounded",
                "split": "test",
                "voice_id": fam,
                "canonical_class": "SILENCE",
                "class_id": 5,
                "bounded_label": "REJECT",
                "bounded_class_id": 4,
                "confusion_family": fam,
                "transcription": f"[{fam}]",
                "raw_wav_path": os.path.relpath(kws_path, BASE_DIR),
                "kws_wav_path": os.path.relpath(kws_path, BASE_DIR),
                "duration_samples": 16000,
                "sample_rate": 16000,
                "speech_onset_sec": 0.0,
                "speech_duration_sec": 1.0,
                "sha256": kws_h
            })

    print(f"Total fresh test records: {len(test_records)}")
    assert len(test_records) == 720, f"Expected 720 test records, got {len(test_records)}"

    # Save test manifest
    test_manifest_path = os.path.join(SPLITS_DIR, "test_manifest.csv")
    fieldnames = list(test_records[0].keys())
    with open(test_manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(test_records)
    print(f"Wrote Test Manifest: {test_manifest_path} (SHA: {compute_sha256(test_manifest_path)})")

    # Copy / reference training and validation manifests from v6/v7
    v6_splits = os.path.join(V6_EXP_DIR, "data", "splits")
    train_manifest_v8 = os.path.join(SPLITS_DIR, "train_manifest.csv")
    val_manifest_v8 = os.path.join(SPLITS_DIR, "val_manifest.csv")

    train_rows = []
    with open(os.path.join(v6_splits, "train_manifest.csv"), "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["dataset_version"] = "research_v8.0_bounded"
            train_rows.append(r)

    val_rows = []
    with open(os.path.join(v6_splits, "val_manifest.csv"), "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["dataset_version"] = "research_v8.0_bounded"
            val_rows.append(r)

    with open(train_manifest_v8, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(train_rows[0].keys()))
        writer.writeheader()
        writer.writerows(train_rows)
    print(f"Wrote Train Manifest: {train_manifest_v8} ({len(train_rows)} clips)")

    with open(val_manifest_v8, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(val_rows[0].keys()))
        writer.writeheader()
        writer.writerows(val_rows)
    print(f"Wrote Val Manifest: {val_manifest_v8} ({len(val_rows)} clips)")

    # Full combined manifest
    full_manifest_path = os.path.join(DATA_DIR, "manifest.csv")
    all_rows = train_rows + val_rows + test_records
    # Union of all keys preserving order
    all_fieldnames = []
    for r in all_rows:
        for k in r.keys():
            if k not in all_fieldnames:
                all_fieldnames.append(k)

    with open(full_manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_fieldnames, restval="")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Wrote Full Manifest: {full_manifest_path} ({len(all_rows)} total clips)")


if __name__ == "__main__":
    build_test_dataset()
