# Blinded Human Listening Evaluation — Reviewer Guide

**Workspace:** `D:\trainidentificationaudio`  
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
