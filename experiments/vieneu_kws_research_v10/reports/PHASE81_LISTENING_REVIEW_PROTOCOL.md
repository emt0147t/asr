# Phase 81 — Blinded Human Listening Review Protocol Report

**Workspace:** `D:\trainidentificationaudio`  
**Package:** `experiments/vieneu_kws_research_v10/human_review/`  
**Corpus Size:** 120 randomized clips  
**Status Banners:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Protocol Architecture & Anti-Fabrication Rule

Phase 81 operates the double-blind human perceptual listening evaluation for Vietnamese Keyword Spotting.

### Anti-Fabrication Principle
Reviewer responses must originate exclusively from genuine native Vietnamese human listeners. Automated model outputs, machine transcriptions, or intended ground truths must **never** be entered into reviewer forms. Until genuine human reviewer submissions are gathered and validated, all rating records remain blank, and the status remains:

$$\mathbf{Status:}\quad \mathbf{PENDING\_HUMAN\_LISTENER\_VERIFICATION}$$

---

## 2. Review Corpus & Double-Blind Security

The review package contains 120 clips verified against v9 hashes with 100% cryptographic parity:
- **40 Target Commands:** 10 `TRAI`, 10 `PHAI`, 10 `LEN`, 10 `XUONG` across 4 voices (`Bảo Ngọc`, `Diễm My`, `Hoàng Kiên`, `Tuấn Tú`).
- **35 Confusable Non-Targets:** Minimal pairs across tonal, consonantal, and vowel/coda dimensions.
- **30 General Non-Targets:** Conversational phrase utterances.
- **15 Background Silences:** Near-silence, HVAC hum, office ambient, airflow noise.

### Blinding Guarantees:
1. **Isolated Static Serving:** The web application (`review_app.py`) serves only anonymized audio files from `audio/review_clip_*.wav`.
2. **Zero Metadata Leakage:** The internal master key `review_manifest.csv` is stored outside web server routes and cannot be retrieved via HTTP.
3. **Localhost Binding:** The server binds strictly to `127.0.0.1`, ensuring zero data transmission across external networks.
4. **Resumable State:** Reviews can be paused and resumed at any time, with autosaves written to `human_review/responses/`.

---

## 3. Reviewer Verification & Quality Gate Criteria

When human reviews are submitted, each response will undergo automated quality gates before acceptance:
1. **Completeness:** All 120 clips must have non-empty ratings for intelligibility (1-5), naturalness (1-5), pronunciation (Y/N/Unsure), and confidence (1-5).
2. **Range Validation:** All numerical ratings must strictly adhere to the $[1..5]$ integer scale.
3. **Attention Check Consistency:** Silence and ambient noise clips must be identified with low intelligibility ($\le 2$) or labeled `[SILENCE]`.
4. **Current Submission Total:** Exactly **0 human submissions** are on file. All reviewer response templates remain empty.
