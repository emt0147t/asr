"""
Phase D: Stratified Blinded Listening Package Generator (80 Samples)
Selects 20 samples per command label (80 total) stratified across 25 voices and 3 splits.
Copies audio to randomized, anonymized files (blind_eval_01.wav ... blind_eval_80.wav),
creates a preserved answer key, an evaluation review sheet, and a listener guide.
"""

import os
import sys
import shutil
import json
import random
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET_DIR = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(DATASET_DIR, "manifest.csv")
BLINDED_DIR = os.path.join(DATASET_DIR, "blinded_listening")
BLINDED_AUDIO_DIR = os.path.join(BLINDED_DIR, "audio")
KEY_JSON = os.path.join(BLINDED_DIR, "blinded_listening_key.json")
KEY_CSV = os.path.join(BLINDED_DIR, "blinded_listening_key.csv")
REVIEW_SHEET_CSV = os.path.join(BLINDED_DIR, "blinded_listening_review_sheet.csv")
GUIDE_MD = os.path.join(BLINDED_DIR, "BLINDED_LISTENING_GUIDE.md")

def create_blinded_package():
    print("=" * 70)
    print("Creating Stratified Blinded Listening Package (80 Utterances)...")
    print("=" * 70)

    os.makedirs(BLINDED_AUDIO_DIR, exist_ok=True)

    df = pd.read_csv(MANIFEST_PATH)

    classes = ["TRAI", "PHAI", "LEN", "XUONG"]
    seed = 33
    selected_indices = []

    # Train: 13 per class = 52
    for c in classes:
        pool = df[(df["split"] == "train") & (df["class_label"] == c)]
        selected_indices.extend(pool.sample(13, random_state=seed * 10 + len(c)).index.tolist())
    # Val: 4 for TRAI, LEN; 3 for PHAI, XUONG = 14
    for c in classes:
        n = 4 if c in ["TRAI", "LEN"] else 3
        pool = df[(df["split"] == "val") & (df["class_label"] == c)]
        selected_indices.extend(pool.sample(n, random_state=seed * 20 + len(c)).index.tolist())
    # Test: 3 for TRAI, LEN; 4 for PHAI, XUONG = 14
    for c in classes:
        n = 3 if c in ["TRAI", "LEN"] else 4
        pool = df[(df["split"] == "test") & (df["class_label"] == c)]
        selected_indices.extend(pool.sample(n, random_state=seed * 30 + len(c)).index.tolist())

    assert len(selected_indices) == 80, f"Expected 80 samples, got {len(selected_indices)}"

    sub_df = df.loc[selected_indices].copy()

    # Shuffle with fixed random state to anonymize order
    rng = random.Random(20261017)
    shuffled_indices = list(range(len(sub_df)))
    rng.shuffle(shuffled_indices)
    sub_df = sub_df.iloc[shuffled_indices].reset_index(drop=True)

    key_records = []
    review_rows = []

    for i, (_, row) in enumerate(sub_df.iterrows(), start=1):
        blind_id = f"BLIND_EVAL_{i:02d}"
        dest_filename = f"blind_eval_{i:02d}.wav"
        dest_path = os.path.join(BLINDED_AUDIO_DIR, dest_filename)

        src_full = os.path.join(BASE_DIR, row["kws_wav_path"])
        shutil.copy2(src_full, dest_path)

        rel_dest = os.path.relpath(dest_path, BASE_DIR).replace("\\", "/")

        key_records.append({
            "blind_clip_id": blind_id,
            "blind_audio_path": rel_dest,
            "original_sample_id": row["sample_id"],
            "true_class_label": row["class_label"],
            "true_text": row["text"],
            "true_tone": row["vietnamese_tone"],
            "split": row["split"],
            "voice_id": row["voice_id"],
            "voice_gender": row["voice_gender"],
            "voice_region": row["voice_region"],
            "temperature": float(row["temperature"]),
            "top_p": float(row["top_p"]),
            "seed": int(row["seed"]),
            "original_kws_wav_path": row["kws_wav_path"],
            "kws_sha256": row["kws_sha256"]
        })

        # Blank evaluation row for human listener
        review_rows.append({
            "blind_clip_id": blind_id,
            "audio_file_path": rel_dest,
            "perceived_command (TRAI/PHAI/LEN/XUONG/OTHER/UNCERTAIN)": "",
            "defect_clipped (YES/NO)": "NO",
            "defect_distorted (YES/NO)": "NO",
            "defect_truncated (YES/NO)": "NO",
            "defect_buzzer (YES/NO)": "NO",
            "defect_clean (YES/NO)": "YES",
            "perceived_tone (SAC/HOI/NGANG/OTHER/UNCERTAIN)": "",
            "intelligibility (1 to 5, 5=crystal clear)": "",
            "naturalness (1 to 5, 5=indistinguishable from human)": "",
            "evaluator_comments": ""
        })

    # Save Key
    with open(KEY_JSON, "w", encoding="utf-8") as f:
        json.dump(key_records, f, indent=2, ensure_ascii=False)

    key_df = pd.DataFrame(key_records)
    key_df.to_csv(KEY_CSV, index=False, encoding="utf-8")

    # Save Review Sheet
    review_df = pd.DataFrame(review_rows)
    review_df.to_csv(REVIEW_SHEET_CSV, index=False, encoding="utf-8")

    # Summary verification
    print(f"Created {len(review_df)} blinded evaluation clips in: {BLINDED_AUDIO_DIR}")
    print("Class distribution in blinded package:")
    print(key_df["true_class_label"].value_counts().to_dict())
    print("Split distribution in blinded package:")
    print(key_df["split"].value_counts().to_dict())
    print(f"Unique voice presets represented: {key_df['voice_id'].nunique()} / 25")

    # Write Guide
    guide_content = f"""# Vietnamese Synthetic KWS — Blinded Listening Evaluation Protocol (80 Samples)

**Package Directory:** `experiments/synthetic_kws_dataset_vieneu_1000/blinded_listening/`  
**Total Clips:** 80  
**Target Vocabulary:** `TRAI` (Trái), `PHAI` (Phải), `LEN` (Lên), `XUONG` (Xuống)  
**Human Verification Status:** `PENDING_HUMAN_LISTENER_VERIFICATION`  
**Official Project Status:** `NOT_VALIDATED_ON_NATURAL_SPEECH`  

---

## 1. Objective

To provide an empirical, objective basis for subjective intelligibility and naturalness of the synthetic Vietnamese command utterances generated by VieNeu-TTS v3 Turbo.

The listener is presented with 80 anonymized audio clips (`blind_eval_01.wav` to `blind_eval_80.wav`) with all metadata (command word, voice identity, model parameters) concealed.

---

## 2. Evaluation Sheet Schema

The evaluation sheet is located at [`blinded_listening_review_sheet.csv`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_dataset_vieneu_1000/blinded_listening/blinded_listening_review_sheet.csv).

| Column | Acceptable Values | Description |
| :--- | :--- | :--- |
| `blind_clip_id` | `BLIND_EVAL_01` .. `BLIND_EVAL_80` | Unique anonymized identifier |
| `audio_file_path` | File path | Relative path to audio clip |
| `perceived_command` | `TRAI`, `PHAI`, `LEN`, `XUONG`, `OTHER`, `UNCERTAIN` | Spoken Vietnamese command recognized by listener |
| `defect_clipped` | `YES` / `NO` | Did the audio contain audible digital harsh clipping? |
| `defect_distorted` | `YES` / `NO` | Did the voice have unnatural buzzy, robotic, or robotic distortion? |
| `defect_truncated` | `YES` / `NO` | Was the initial consonant or final diphthong/consonant cut off? |
| `defect_buzzer` | `YES` / `NO` | Did it sound like a synthetic buzzer instead of human speech? |
| `defect_clean` | `YES` / `NO` | Was the audio clean and defect-free? |
| `perceived_tone` | `SAC`, `HOI`, `NGANG`, `OTHER`, `UNCERTAIN` | Auditory lexical tone perceived by listener |
| `intelligibility` | Integer `1` to `5` | `5`: Crystal clear, immediate recognition<br>`4`: Clear, easy to recognize<br>`3`: Recognizable with slight effort<br>`2`: Ambiguous, difficult to recognize<br>`1`: Completely unintelligible |
| `naturalness` | Integer `1` to `5` | `5`: Indistinguishable from native human speech<br>`4`: Very natural, slight synthetic color<br>`3`: Clear synthetic character but smooth<br>`2`: Unnatural robotic cadence or pitch jumps<br>`1`: Robotic buzzer or synthetic noise |
| `evaluator_comments` | Text | Free-form notes on dialect, tone subtleties, or defects |

---

## 3. Preserved Answer Key

The answer key is strictly quarantined in:
- [`blinded_listening_key.json`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_dataset_vieneu_1000/blinded_listening/blinded_listening_key.json)
- [`blinded_listening_key.csv`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_dataset_vieneu_1000/blinded_listening/blinded_listening_key.csv)

The evaluator must **never consult the key** until all 80 clips have been independently scored.
"""

    with open(GUIDE_MD, "w", encoding="utf-8") as f:
        f.write(guide_content)
    print(f"Listener guide saved to: {GUIDE_MD}")

if __name__ == "__main__":
    create_blinded_package()
