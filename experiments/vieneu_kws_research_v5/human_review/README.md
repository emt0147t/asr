# Human Listening Review Status and Protocol (Research v5)

**Status:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`  
**Location:** `experiments/vieneu_kws_research_v5/human_review/`  

---

## 1. Governance and Anti-Fabrication Notice

In accordance with strict experimental governance guidelines:
- **No human listening responses are fabricated or assumed.**
- All evaluations in this campaign remain strictly bounded synthetic experimental benchmarks.
- No claim of natural-speech generalization or real-world user validation is made without verified, signed human perceptual audits.

---

## 2. Review Package Architecture

The dataset v5 package maintains a dedicated protocol for external native Vietnamese listening verification:
- **Review Sample Selection:** 40 stratified audio clips (10 true commands, 15 hard-negative minimal pairs, 10 conversational distractors, 5 background noise clips).
- **Blinding Integrity:** Reviewers receive only audio waveforms labeled with anonymized IDs (`V5_BLIND_001` through `V5_BLIND_040`). True transcripts and ground-truth classes remain sealed in the project vault.
- **Evaluation Dimensions:**
  1. Perceived Vietnamese orthographic transcription.
  2. Perceived command intent (`TRAI`, `PHAI`, `LEN`, `XUONG`, or `REJECT / NO_COMMAND`).
  3. Subjective speech naturalness and intelligibility score (1 to 5 Likert scale).
  4. Notable synthesis artifacts (glottal jumps, metallic distortion, phoneme truncation).

All synthetic model results will retain the `PENDING_HUMAN_LISTENER_VERIFICATION` status marker until completed listener worksheets are returned and reconciled.
