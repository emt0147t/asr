# Provenance and Label Policy: Synthetic KWS V5.2 Release Candidate

**Dataset:** `synthetic_kws_v5_2_release_candidate`  
**Total Clips:** 3,000 (16 kHz, mono, 1.0s duration, PCM-16)  
**Date:** 2026-10-16  
**Governance Invariant:** `HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED`

---

## 1. Provenance and Scope

1. **Synthetic-Only Guarantee:**  
   Every audio clip in this dataset was generated parametrically using the isolated V5.2 glottal-filter articulatory synthesis engine. Zero human speech recordings, external audio assets, field recordings, or unverified audio datasets were accessed, ingested, or transformed.
2. **Deterministic Reproducibility:**  
   Every sample has its complete unrounded generation parameters, seeds, speaker vocal tract profiles, and noise parameters explicitly stored in `manifest.csv`. Any sample can be regenerated bit-exact without sequential dependency or a global master seed.

---

## 2. Two-Tier Taxonomy and Label Contract

The dataset enforces a strict two-tier taxonomy:

### Tier 1: Canonical Model Class Label (`class_label`)
Six mutually exclusive target classes matching the production KWS deployment contract:
1. `TRAI` (500 clips): Target Vietnamese keyword "trái" (Turn left).
2. `PHAI` (500 clips): Target Vietnamese keyword "phải" (Turn right).
3. `LEN` (500 clips): Target Vietnamese keyword "lên" (Go up).
4. `XUONG` (500 clips): Target Vietnamese keyword "xuống" (Go down).
5. `UNKNOWN` (500 clips): Non-target speech commands, confusers, and phonetically challenging foils.
6. `SILENCE` (500 clips): Ambient noise, digital silence, and non-target acoustic backgrounds.

### Tier 2: Acoustic Subclass (`acoustic_subclass`)
Describes physical and phonetic provenance:
* **Target Keywords:** `target_keyword`
* **UNKNOWN Subclasses:**
  - `confuser_initial_sharing`: Synthetic words sharing the onset consonant with target words (`trời`, `pha`, `tròn`, `phố`, `lúa`, `xinh`).
  - `confuser_rhyme_sharing`: Synthetic words sharing the rhyme/vowel with target words (`mái`, `tai`, `đèn`, `cuống`, `sen`, `bài`).
  - `open_vocabulary_general`: General conversational Vietnamese words (`chào`, `bạn`, `đi`, `xe`, `nhà`, `số`, `ngày`, `học`, `cơm`, `nước`, `sách`, `bút`).
* **SILENCE Subclasses:**
  - `ambient_room_tone` (60%): Low-pass filtered ambient thermal and room reverberant noise floor.
  - `ambient_low_freq_hum` (25%): 50 Hz / 60 Hz synthetic electrical mains hum with harmonics.
  - `ambient_digital_zero` (15%): Low-level dither floor ($\pm 1$ LSB at 16-bit).

---

## 3. Train / Validation / Test Split Isolation Policy

To strictly prevent data leakage across evaluation sets:
* **70 / 15 / 15 Balance:** 2,100 Train (350 per class), 450 Validation (75 per class), 450 Test (75 per class).
* **Disjoint Speaker Profiles:** 
  - Train: `SPK_TR_01` through `SPK_TR_08`
  - Validation: `SPK_VAL_01`, `SPK_VAL_02` (completely disjoint from train)
  - Test: `SPK_TEST_01`, `SPK_TEST_02` (completely disjoint from train and val)
* **Disjoint Seed Domains:**
  - Train: Seeds 100000–199999, Noise seeds 1000000–1199999
  - Val: Seeds 200000–249999, Noise seeds 2000000–2049999
  - Test: Seeds 300000–349999, Noise seeds 3000000–3049999
