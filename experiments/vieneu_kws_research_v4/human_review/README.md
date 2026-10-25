# Blinded Human Listening Review Protocol

**Status:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`  
**Location:** `experiments/vieneu_kws_research_v4/human_review/`  

---

## 1. Overview and Blinding Integrity

To avoid experimental confirmation bias and validate whether synthetic audio artifacts, speech degradation, or phonetic ambiguities explain model behavior, a blinded perceptual listening review protocol has been created.

The review package contains:
- **`human_review_sheet.csv`**: The active rating form provided to human reviewers. It contains **only** blinded IDs (`BLIND_REV_001` to `BLIND_REV_040`), reference audio file names, and timestamp windows. It contains **no** ground-truth labels and **no** model predictions.
- **`human_review_key.csv`**: The locked key mapping each `blind_id` back to its true text, ground-truth class, speaker voice, and error category. This key is sealed until listening scores are recorded.

---

## 2. Instructions for Human Listeners

1. **Environment:** Use circumaural headphones in a quiet room at a calibrated comfortable listening level.
2. **Audio Samples:** Open the indicated audio file in an audio editor or playback tool (e.g., Audacity or Python audio player).
3. **Time Window:** Listen to the audio around the specified `timestamp_window` (each segment is approximately 1.0 second centered on the event).
4. **Scoring Fields:**
   - **`perceived_transcription`**: Transcribe the exact Vietnamese words you hear (e.g., "trái", "phải", "bước lên", "vải", "tải", "xuống").
   - **`is_target_command (TRAI/PHAI/LEN/XUONG/NO)`**: Indicate whether you perceive the segment as a clear standalone command keyword (`TRAI`, `PHAI`, `LEN`, `XUONG`), or a non-target/phrase/distractor (`NO`).
   - **`intelligibility_rating_1_to_5`**:
     - `1`: Unintelligible / heavy noise / cannot determine syllable.
     - `2`: Severely degraded / guessing based on vague formant cues.
     - `3`: Moderately degraded but audible syllable with some effort.
     - `4`: Clear speech with minor background noise.
     - `5`: Pristine, perfectly intelligible native pronunciation.
   - **`reviewer_notes`**: Note any perceived unnatural TTS synthesis artifacts, robotic pitch jumps, or clipped consonants.

---

## 3. Post-Review Reconciliation

Once the human review sheet is completed:
1. Compare `is_target_command` against `human_review_key.csv`.
2. Compute human-vs-model agreement, human-vs-ground-truth agreement, and mean intelligibility per speaker voice.
3. Keep the findings versioned in this folder.
