"""
Phase 76: Blinded Human Listening Evaluation Package Preparation.
Workspace: D:\trainidentificationaudio
Experiment: experiments/vieneu_kws_research_v9/

Builds:
1. 120-clip stratified synthetic review package from validation split.
2. Randomized anonymized audio files in human_review/audio/ (review_clip_001.wav - review_clip_120.wav).
3. Reviewer-facing CSV (human_review/reviewer_form.csv) strictly stripped of labels/transcripts/predictions.
4. Internal answer key (human_review/review_manifest.csv).
5. Reviewer guidelines (human_review/README.md).
6. Protocol report (reports/PHASE76_BLINDED_LISTENING_PROTOCOL.md).
"""

import os
import sys
import shutil
import hashlib
import json
import csv
import random

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, BASE_DIR)

V8_DIR = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v8")
V9_DIR = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v9")
REVIEW_DIR = os.path.join(V9_DIR, "human_review")
AUDIO_DEST_DIR = os.path.join(REVIEW_DIR, "audio")
REPORTS_DIR = os.path.join(V9_DIR, "reports")

os.makedirs(AUDIO_DEST_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def sha256_file(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def run_phase76():
    print("=" * 80)
    print("Phase 76: Blinded Human Listening Evaluation Package Preparation")
    print("=" * 80)

    val_csv = os.path.join(V8_DIR, "data", "splits", "val_manifest.csv")
    assert os.path.exists(val_csv), f"Validation manifest missing: {val_csv}"

    with open(val_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        val_rows = list(reader)

    print(f"Loaded {len(val_rows)} rows from validation manifest.")

    # Stratified selection of 120 clips:
    # 40 target commands: 10 TRAI, 10 PHAI, 10 LEN, 10 XUONG
    # 35 minimal pairs (confusable non-targets: tone, consonant, vowel/coda)
    # 30 general non-target phrase contexts
    # 15 background silence/noise clips
    rng = random.Random(76)

    trai_clips = [r for r in val_rows if r["class_label"] == "TRAI"]
    phai_clips = [r for r in val_rows if r["class_label"] == "PHAI"]
    len_clips = [r for r in val_rows if r["class_label"] == "LEN"]
    xuong_clips = [r for r in val_rows if r["class_label"] == "XUONG"]

    min_pair_clips = [
        r for r in val_rows
        if r["confusion_family"] in ["minimal_pair_tone", "minimal_pair_consonant", "minimal_pair_vowel_coda"]
    ]
    phrase_clips = [r for r in val_rows if r["confusion_family"] == "phonetic_phrase_context"]
    bg_clips = [
        r for r in val_rows
        if r["confusion_family"] in ["near_silence", "hvac_hum", "office_ambient", "airflow_pink"]
    ]

    selected_trai = rng.sample(trai_clips, 10)
    selected_phai = rng.sample(phai_clips, 10)
    selected_len = rng.sample(len_clips, 10)
    selected_xuong = rng.sample(xuong_clips, 10)
    selected_min = rng.sample(min_pair_clips, 35)
    selected_phrase = rng.sample(phrase_clips, 30)
    selected_bg = rng.sample(bg_clips, 15)

    all_selected = (
        selected_trai + selected_phai + selected_len + selected_xuong +
        selected_min + selected_phrase + selected_bg
    )
    assert len(all_selected) == 120, f"Expected 120 clips, got {len(all_selected)}"

    # Shuffle thoroughly with fixed seed
    rng.shuffle(all_selected)

    review_manifest_rows = []
    reviewer_form_rows = []

    for idx, item in enumerate(all_selected, start=1):
        clip_id = f"review_clip_{idx:03d}"
        src_wav = os.path.join(BASE_DIR, item["kws_wav_path"])
        dest_filename = f"{clip_id}.wav"
        dest_wav = os.path.join(AUDIO_DEST_DIR, dest_filename)

        shutil.copy2(src_wav, dest_wav)
        clip_sha = sha256_file(dest_wav)

        rel_dest_path = os.path.relpath(dest_wav, BASE_DIR).replace("\\", "/")

        # Internal manifest with full ground truth (SECRET / ANSWER KEY)
        review_manifest_rows.append({
            "review_clip_id": clip_id,
            "original_sample_id": item["sample_id"],
            "voice_id": item["voice_id"],
            "canonical_class": item["class_label"],
            "bounded_label": item["bounded_label"],
            "confusion_family": item["confusion_family"],
            "transcription": item["text"],
            "audio_file": dest_filename,
            "relative_path": rel_dest_path,
            "sha256": clip_sha
        })

        # Reviewer-facing form (STRICTLY BLINDED: NO LABELS, NO TRANSCRIPTS, NO PREDICTIONS)
        reviewer_form_rows.append({
            "review_clip_id": clip_id,
            "audio_path": f"audio/{dest_filename}",
            "perceived_text": "",
            "intelligibility_1to5": "",
            "naturalness_1to5": "",
            "pronunciation_correct_yn": "",
            "confidence_1to5": "",
            "comments": ""
        })

    # Write internal answer key
    manifest_csv = os.path.join(REVIEW_DIR, "review_manifest.csv")
    with open(manifest_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "review_clip_id", "original_sample_id", "voice_id", "canonical_class",
            "bounded_label", "confusion_family", "transcription", "audio_file",
            "relative_path", "sha256"
        ])
        writer.writeheader()
        writer.writerows(review_manifest_rows)
    print(f"[Phase 76] Wrote internal answer key: {manifest_csv}")

    # Write reviewer form (blank ratings)
    form_csv = os.path.join(REVIEW_DIR, "reviewer_form.csv")
    with open(form_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "review_clip_id", "audio_path", "perceived_text", "intelligibility_1to5",
            "naturalness_1to5", "pronunciation_correct_yn", "confidence_1to5", "comments"
        ])
        writer.writeheader()
        writer.writerows(reviewer_form_rows)
    print(f"[Phase 76] Wrote reviewer form: {form_csv}")

    # Write human_review/README.md
    readme_content = """# Blinded Human Listening Evaluation — Reviewer Guide

**Workspace:** `D:\\trainidentificationaudio`  
**Package Directory:** `experiments/vieneu_kws_research_v9/human_review/`  
**Review Set Size:** 120 audio clips (1.0 second each, 16 kHz mono)  
**Status Banner:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Objective and Ethics

This listening test evaluates the intelligibility, acoustic naturalness, and phonetic correctness of synthesized Vietnamese speech audio clips.

To preserve scientific validity, this test is **strictly double-blinded**:
- Reviewers are provided only audio files identified by randomized IDs (`review_clip_001.wav` through `review_clip_120.wav`).
- You are not provided the intended transcript, category label, or any model prediction.
- Please do not consult any answer key or other participants while conducting your review.

---

## 2. Reviewer Instructions

1. Open `reviewer_form.csv` in your preferred spreadsheet or text editor.
2. Listen to each audio clip located in the `audio/` subfolder using headphones or high-quality speakers in a quiet environment.
3. For each clip, enter your genuine human perception across the following columns:
   - **`perceived_text`**: Type the Vietnamese words or syllables you heard. If you hear only silence or noise, type `[SILENCE]`. If you hear speech but cannot decipher the word, type `[UNINTELLIGIBLE]`.
   - **`intelligibility_1to5`**: Rating of speech clarity:
     - `1`: Completely unintelligible (cannot recognize any phoneme).
     - `2`: Barely intelligible (heavy distortion; mostly guessing).
     - `3`: Moderately intelligible (can understand with effort).
     - `4`: Mostly clear (minor acoustic degradation).
     - `5`: Completely clear (effortless comprehension).
   - **`naturalness_1to5`**: Rating of acoustic and prosodic naturalness:
     - `1`: Extremely robotic, buzzy, or synthetic.
     - `2`: Noticeably synthetic with unnatural cadence/tone.
     - `3`: Acceptable synthetic speech.
     - `4`: Very close to human speech with minor synthetic artifacts.
     - `5`: Completely natural, indistinguishable from a human recording.
   - **`pronunciation_correct_yn`**: Vietnamese tonal and phonetic correctness:
     - `Y`: Correct Vietnamese tone and pronunciation.
     - `N`: Defective tone, clipped coda, or unnatural phoneme substitution.
     - `Unsure`: Ambiguous or unintelligible.
   - **`confidence_1to5`**: How confident are you in your transcription:
     - `1`: Pure guess.
     - `5`: Absolutely certain.
   - **`comments`**: (Optional) Note any specific acoustic artifacts (clipping, tone drop, robotization, background noise).

---

## 3. Important Integrity Policies

- **Never fabricate responses:** If no human listener has completed the evaluation, the form must remain blank, and the project status must remain `PENDING_HUMAN_LISTENER_VERIFICATION`.
- Software unit tests verifying file existence or CSV parsing do **not** constitute completed human evaluations.
"""
    with open(os.path.join(REVIEW_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"[Phase 76] Wrote reviewer guide: {os.path.join(REVIEW_DIR, 'README.md')}")

    # Write reports/PHASE76_BLINDED_LISTENING_PROTOCOL.md
    protocol_content = """# Phase 76 — Blinded Human Listening Evaluation Protocol

**Workspace:** `D:\\trainidentificationaudio`  
**Package:** `experiments/vieneu_kws_research_v9/human_review/`  
**Dataset Size:** 120 randomized clips  
**Status Banner:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Study Rationale & Non-Fabrication Mandate

Prior phases achieved high synthetic test scores (99.8% recall, 2.0% FAR). However, synthetic speech generated by neural TTS can exhibit subtle acoustic artifacts (such as pitch drift, unnatural tone contours, or robotic codas) that neural networks might exploit or overlook.

To establish genuine phonetic validity, a blinded human perceptual audit is required.

> [!CAUTION]
> **Zero Fabrication Mandate:**  
> Reviewer responses must be collected exclusively from real human listeners. Automated model predictions or intended text scripts must **never** be copied into the reviewer form. Until completed reviewer submissions are gathered and validated, the official research status remains strictly:
>
> `PENDING_HUMAN_LISTENER_VERIFICATION`

---

## 2. Stratified Sampling Methodology

The review set comprises 120 clips sampled from the validation split (`val_manifest.csv`, $N=360$), ensuring complete independence from the frozen Phase 74 test set:

| Acoustic Sub-Group | Stratum Description | Number of Clips | Percentage |
| :--- | :--- | :--- | :--- |
| **Target Commands** | 10 `TRAI`, 10 `PHAI`, 10 `LEN`, 10 `XUONG` across 4 voices (`Bảo Ngọc`, `Diễm My`, `Hoàng Kiên`, `Tuấn Tú`) | 40 clips | 33.3% |
| **Confusable Non-Targets** | Phonetic minimal pairs (tone, initial consonant, vowel/coda) | 35 clips | 29.2% |
| **General Non-Targets** | Natural conversational phrase contexts | 30 clips | 25.0% |
| **Acoustic Backgrounds** | Near-silence, HVAC hum, office ambient, airflow pink noise | 15 clips | 12.5% |
| **Total Review Set** | **Stratified Perceptual Audit Sample** | **120 clips** | **100.0%** |

---

## 3. Double-Blind Blinding Protocol

To eliminate confirmation bias:
1. **Randomized File Identifiers:** Clips are renamed sequentially from `review_clip_001.wav` to `review_clip_120.wav` using a fixed random shuffle (seed 76).
2. **Reviewer Form Sanitization:** `reviewer_form.csv` contains only the clip identifier and relative audio path. All labels, transcriptions, acoustic families, voice metadata, and model outputs are stripped.
3. **Internal Key Isolation:** The master key mapping clip IDs to ground-truth metadata is sequestered in `review_manifest.csv` and must remain inaccessible to reviewers during evaluation.

---

## 4. Multi-Reviewer Adjudication Protocol

When human listener responses are received:
1. **Inter-Rater Agreement:** Cohen's $\\kappa$ (for 2 reviewers) or Fleiss' $\\kappa$ (for $>2$ reviewers) will be calculated on perceived transcription agreement and binary pronunciation correctness ($Y/N$).
2. **Adjudication Procedure:** Any clip with rater disagreement on target command identity (`TRAI`, `PHAI`, `LEN`, `XUONG`, or Non-Target) will be submitted to a third native Vietnamese speaker for blinded resolution.
3. **Current Operating Status:** Exactly **0 human submissions** are currently on file. A unit test confirming CSV schema validity is not a completed listening study. Therefore, all fields in `reviewer_form.csv` remain unpopulated, and verification status is **PENDING**.
"""
    with open(os.path.join(REPORTS_DIR, "PHASE76_BLINDED_LISTENING_PROTOCOL.md"), "w", encoding="utf-8") as f:
        f.write(protocol_content)
    print(f"[Phase 76] Wrote protocol report: {os.path.join(REPORTS_DIR, 'PHASE76_BLINDED_LISTENING_PROTOCOL.md')}")
    print("Phase 76 package preparation completed successfully.")


if __name__ == "__main__":
    run_phase76()
