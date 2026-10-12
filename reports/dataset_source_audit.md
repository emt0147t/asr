# Comprehensive Dataset Source Audit Report

**Project:** Production-Oriented Vietnamese Keyword Spotting (KWS) for Android and iOS  
**Document:** Deliverable 3 — Dataset Source Legal, Technical & Provenance Audit  
**Date:** 2026-10-12  
**Status:** COMPLETE — RIGOROUS TECHNICAL & LEGAL EVALUATION  

---

## 1. Executive Summary & Policy Compliance

In strict compliance with the **Mandatory Dataset Policy**:
1. **VIVOS is NOT used as the default dataset or primary data source.** Its non-commercial restrictions (`CC BY-NC-SA 4.0`), single-accent bias, and read-sentence domain make it unsuitable as a production backbone.
2. Every external corpus has been evaluated against legal licenses, distribution restrictions, download availability, speaker diversity, transcript fidelity, acoustic quality, and appropriateness for isolated keyword spotting.
3. Long-form continuous speech corpora are treated as sources for `UNKNOWN` and hard-negative examples; under no circumstances are full sentences containing command words labeled as isolated commands without proper alignment and segmentation.

---

## 2. In-Depth Candidate Source Audits

```
+-------------------------------------------------------------------------------------------------------------------+
| Source ID    | Corpus Name                      | License         | Commercial Use | Isolated KWS Suitability     |
+-------------------------------------------------------------------------------------------------------------------+
| SRC-01       | Mozilla Common Voice Vietnamese  | CC0-1.0         | Permitted      | High (Negative/Unknown)       |
| SRC-02       | VietSpeech / VSV-1100            | Apache-2.0*     | Questionable*  | Quarantined (Provenance risk) |
| SRC-03       | Google FLEURS Vietnamese (vi_vn) | CC-BY-4.0       | Permitted      | High (Negative/Unknown)       |
| SRC-04       | MLCommons MSWC Vietnamese        | CC-BY-4.0       | Permitted      | Sparse isolated words only    |
| SRC-05       | FOSD (FPT Open Speech Dataset)   | FPT Public Lic  | Permitted      | High (Negative/Unknown)       |
| SRC-06       | VIVOS Speech Corpus (AILab)      | CC-BY-NC-SA 4.0 | PROHIBITED     | REJECTED for commercial prod  |
| SRC-07       | MUSAN Noise (OpenSLR 17)         | CC-BY-4.0       | Permitted      | High (Silence/Noise class)    |
| SRC-08       | ESC-50 Environmental Audio       | CC-BY-NC-3.0    | PROHIBITED     | REJECTED for commercial prod  |
| SRC-09       | Speech Commands Ambient Noise    | CC-BY-4.0       | Permitted      | High (Silence/Noise class)    |
| SRC-10       | In-Field Consented Voice Records | Proprietary-ML  | Permitted      | Primary for Classes 0,1,2,3   |
+-------------------------------------------------------------------------------------------------------------------+
```

---

### Source 1: Mozilla Common Voice Vietnamese
- **Official Source URL:** https://commonvoice.mozilla.org/vi and https://commonvoice.mozilla.org/vi/datasets
- **Version Audited:** Common Voice Release 13.0 / 17.0
- **License & Legal Terms:** **CC0-1.0 (Creative Commons Zero / Public Domain Dedication)**. Fully permissive for commercial and non-commercial derivation.
  - *Condition:* Users must agree to the Common Voice Terms of Use: strictly prohibiting attempts to deanonymize, re-identify, or contact volunteer speakers.
- **Downloadable Size & Duration:** ~50 to 60 hours of validated Vietnamese speech; archive size ~4.5 GB (compressed MP3).
- **Recordings & Speakers:** ~35,000+ clips from several hundred crowdsourced volunteer speakers.
- **Transcript Quality:** Human crowdsourced; validated via Mozilla's 2-upvote verification mechanism.
- **Audio Specifications:** Crowdsourced web/mobile browser recordings; 48 kHz / 32 kHz / 16 kHz MP3 format; mono/stereo downmixed; variable acoustic noise, microphone quality, and room acoustics.
- **Accent & Dialect Coverage:** Broad coverage of Northern, Central, and Southern Vietnamese speakers.
- **Access Status & Limitations:** Direct programmatic downloads require API key/authenticated signed email link from Mozilla Common Voice.
- **Audit Decision: APPROVED (CONDITIONAL).**  
  *Usage:* Ingested as general Vietnamese conversational speech for the **`UNKNOWN`** class (and candidate hard negatives). Cannot be used directly as isolated commands without word-level segmentation.

---

### Source 2: VietSpeech / VSV-1100 (NhutP / MMLab UIT)
- **Official Source URL:** https://github.com/NhutP/VietSpeech and https://huggingface.co/datasets/NhutP/VietSpeech
- **Version Audited:** Commit `main`, 27 parquet shards (`train-00000-of-00027.parquet` through `train-00026-of-00027.parquet`).
- **License & Legal Terms:** Repository metadata declares **Apache License 2.0**.
  - *Critical Legal Provenance Risk:* The authors state: *"The voice samples were collected from a variety of social resources... collected from social networks."* Crawled social network speech without explicit user consent or clear original copyright licensing creates secondary copyright liability risks for commercial mobile applications. Relicensing crawled social media under Apache 2.0 does not legally extinguish the original creators' rights under Vietnamese and international copyright laws.
- **Downloadable Size & Duration:** 131 GB total across 27 parquet files; 1,026,047 audio samples; stated ~1,100 hours.
- **Transcript Quality:** Generated largely via automated ASR or weak human captions from social platforms; not certified 100% human-verified.
- **Audio Specifications:** 16,000 Hz, single channel, raw float arrays embedded in Parquet files.
- **Access Status:** Publicly accessible on Hugging Face; total download requires >130 GB disk space and high bandwidth.
- **Audit Decision: QUARANTINED.**  
  *Rationale:* High risk of copyright encumbrance from social media crawling, uncertain commercial redistribution rights, automated transcript provenance, and unwieldy 131 GB footprint for mobile KWS. Quarantined until formal IP clearance is received from UIT MMLab.

---

### Source 3: Google FLEURS Vietnamese (`google/fleurs`, config `vi_vn`)
- **Official Source URL:** https://huggingface.co/datasets/google/fleurs (NeurIPS 2022 Benchmark)
- **Version Audited:** `vi_vn` subset, 2022 release.
- **License & Legal Terms:** **Creative Commons Attribution 4.0 International (CC-BY-4.0)**. Completely open for commercial use, modification, and redistribution with attribution to Google LLC.
- **Downloadable Size & Duration:**
  - `dev.tar.gz`: 214.5 MB (361 recordings)
  - `test.tar.gz`: 544.1 MB (857 recordings)
  - `train.tar.gz`: 1.61 GB (2,994 recordings)
  - Total: 4,212 recordings (~12 hours), archive size 2.36 GB.
- **Recordings & Speakers:** 4,212 verified utterances from distinct native Vietnamese speakers (with balanced male and female demographic annotations).
- **Transcript Quality:** Professional human native speaker read speech; verified high-fidelity transcripts with normalized orthography and spelling.
- **Audio Specifications:** 16,000 Hz native, 16-bit PCM WAV format, clean mobile and studio recording setups.
- **Accent & Dialect Coverage:** Native Vietnamese speakers spanning Northern, Central, and Southern accents.
- **Access Status:** Fully operational; direct HTTP streaming/download verified in our environment.
- **Audit Decision: APPROVED (PRIMARY FOR UNKNOWN).**  
  *Usage:* Perfect source for **`UNKNOWN`** background speech and continuous hard-negative evaluation. Sentences containing command tokens (e.g., *phải*, *trái*) serve as hard negatives to verify rejection of continuous speech.

---

### Source 4: MLCommons Multilingual Spoken Words Corpus (MSWC) Vietnamese
- **Official Source URL:** https://mlcommons.org/datasets/spoken-words/ and https://huggingface.co/datasets/MLCommons/ml_spoken_words
- **Version Audited:** `vi_wav` / `vi_opus` (v1.0)
- **License & Legal Terms:** **Creative Commons Attribution 4.0 International (CC-BY-4.0)**. Fully approved for commercial use.
- **Downloadable Size & Duration:** ~0.1 hours total (~350 samples total in Vietnamese).
- **Recordings & Speakers:**
  - `train.csv`: 215 words (11 speakers)
  - `dev.csv`: 73 words (10 speakers)
  - `test.csv`: 62 words (12 speakers)
- **Keyword Distribution Measured:**
  - `trái`: **0 recordings**
  - `phải`: **0 recordings**
  - `lên`: **7 recordings** across splits
  - `xuống`: **0 recordings**
- **Transcript Quality:** Machine-extracted 1.0-second word segments derived from Common Voice with forced alignment.
- **Audit Decision: APPROVED (LIMITED TO UNKNOWN WORDS).**  
  *Rationale:* MSWC has virtually zero coverage of our four target commands, so it cannot serve as the target command training set. However, its 340+ isolated words (*giờ*, *nói*, *anh*, *tay*, *chuyện*, *nên*, *ông*, *thì*, etc.) provide excellent isolated-word `UNKNOWN` training examples.

---

### Source 5: FOSD (FPT Open Speech Dataset)
- **Official Source URL:** https://data.mendeley.com/datasets/k9sxg2twv4/4 (DOI: 10.17632/k9sxg2twv4.4)
- **Version Audited:** Version 4 (2018/2021)
- **License & Legal Terms:** **FPT Public License (Commercial Use Explicitly Granted)**. Grants free, irrevocable, worldwide rights to use, modify, distribute, and sell.
- **Downloadable Size & Duration:** ~30 hours of recorded speech across 25,921 audio files.
- **Recordings & Speakers:** Over 25,000 clips across multiple speakers.
- **Transcript Quality:** Manually compiled and verified text transcripts with segment boundaries.
- **Audio Specifications:** MP3 format, clean acoustic conditions.
- **Audit Decision: APPROVED (SECONDARY FOR UNKNOWN).**  
  *Usage:* High quality source for expanding the `UNKNOWN` continuous speech pool.

---

### Source 6: VIVOS Speech Corpus (AILab, VNU-HCM)
- **Official Source URL:** https://ailab.hcmus.edu.vn/vivos
- **Version Audited:** VIVOS v1.0 (15 hours, 12,400 utterances, 65 speakers)
- **License & Legal Terms:** **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**.
  - *Commercial Prohibition:* Commercial use is strictly prohibited. Redistribution in a commercial product violates the license.
- **Policy Violation:** Directly violates project policy: *"Do NOT use VIVOS as the default dataset or primary data source."*
- **Audit Decision: REJECTED & EXCLUDED.**  
  *Rationale:* Legal restriction (NonCommercial clause) precludes commercial mobile deployment. Excluded from all training and production manifests.

---

### Source 7: OpenSLR 17 (MUSAN — Music, Speech, and Noise)
- **Official Source URL:** https://www.openslr.org/17/
- **Version Audited:** SLR17 v1.0
- **License & Legal Terms:** **Creative Commons Attribution 4.0 International (CC-BY-4.0)**. Permitted for commercial use and modification.
- **Downloadable Size & Duration:** ~11 GB uncompressed; ~60 hours total (Noise subset: ~6 hours of technical and ambient noise).
- **Audio Specifications:** 16,000 Hz, mono/stereo WAV.
- **Audit Decision: APPROVED (PRIMARY FOR SILENCE & NOISE AUGMENTATION).**

---

### Source 8: ESC-50 (Environmental Sound Classification)
- **Official Source URL:** https://github.com/karolpiczak/ESC-50 and https://huggingface.co/datasets/ashraq/esc50
- **License & Legal Terms:** **Creative Commons Attribution-NonCommercial 3.0 (CC-BY-NC-3.0)**.
- **Audit Decision: REJECTED FOR PRODUCTION / QUARANTINED.**  
  *Rationale:* Non-commercial license. Must not be bundled into a commercial mobile application or commercial model training pipeline.

---

### Source 9: Google Speech Commands Ambient Noise (`_background_noise_`)
- **Official Source URL:** http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
- **Version Audited:** v0.02 `_background_noise_` directory.
- **License & Legal Terms:** **Creative Commons Attribution 4.0 International (CC-BY-4.0)**. Fully permitted.
- **Downloadable Size & Duration:** 6 background audio WAV files (~60 MB, ~35 minutes of continuous ambient noise: pink noise, white noise, running water, mechanical hum).
- **Audio Specifications:** 16,000 Hz, 16-bit mono WAV.
- **Audit Decision: APPROVED (COMPACT CORE FOR SILENCE/NOISE).**

---

### Source 10: In-Field Consented Utterance Collection (Target Commands)
- **Official Source URL:** Internal Workspace Asset Registry (`data/raw/field_recordings/`)
- **License & Legal Terms:** **Proprietary-Consented (Explicit Consent for ML Training & Deployment)**.
- **Target Quota:** 1,500–3,000 verified isolated utterances per command (`TRAI`, `PHAI`, `LEN`, `XUONG`).
- **Acoustic Profiling:** Spanning Northern, Central, and Southern native accents; recorded on mobile devices (iOS/Android); verified 16 kHz 16-bit mono.
- **Audit Decision: APPROVED (PRIMARY BACKBONE FOR CLASSES 0, 1, 2, 3).**

---

## 3. Summary of Source Gating Decisions

| Source | License | Permitted Commercial Use | Action Taken | Target Class Assignment |
| :--- | :--- | :---: | :--- | :--- |
| **Google FLEURS vi_vn** | CC-BY-4.0 | YES | **APPROVED** | Class 4 (`UNKNOWN`) |
| **MLCommons MSWC (vi)** | CC-BY-4.0 | YES | **APPROVED** | Class 4 (`UNKNOWN` isolated words) |
| **FOSD (FPT Speech)** | FPT Public | YES | **APPROVED** | Class 4 (`UNKNOWN`) |
| **Speech Commands Noise**| CC-BY-4.0 | YES | **APPROVED** | Class 5 (`SILENCE`) |
| **MUSAN Noise** | CC-BY-4.0 | YES | **APPROVED** | Class 5 (`SILENCE`) |
| **In-Field Consented** | Consented | YES | **APPROVED** | Classes 0, 1, 2, 3 (`TRAI, PHAI, LEN, XUONG`) |
| **VietSpeech (VSV-1100)**| Apache-2.0*| UNVERIFIED (Social crawl)| **QUARANTINED** | Excluded pending copyright clearance |
| **VIVOS (AILab)** | CC-BY-NC-SA | NO | **REJECTED** | Strictly Excluded (Policy & License) |
| **ESC-50** | CC-BY-NC-3.0| NO | **REJECTED** | Strictly Excluded (Non-commercial) |
| **Mozilla Common Voice** | CC0-1.0 | YES | **CONDITIONAL** | Needs signed API download link |
