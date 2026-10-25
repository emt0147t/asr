# Six-Class Vietnamese KWS Dataset v2 Audit & Integrity Report

**Dataset Path:** `experiments/synthetic_kws_dataset_vieneu_6class_v2/`  
**Audit Date:** October 2026  
**Auditor:** Automated Integrity Verification Suite  

---

## Summary of Integrity Checks

| Audit Criterion | Expected Standard | Observed Value | Result |
| :--- | :--- | :--- | :---: |
| **Total Clip Count** | Exactly 2,000 | 2,000 | **PASS** |
| **Class Balance** | 250x4 targets + 500 UNK + 500 SIL | Exactly matches | **PASS** |
| **Split Ratio** | 68% / 16% / 16% (1360 / 320 / 320) | 1,360 / 320 / 320 | **PASS** |
| **Unique Sample IDs** | Exactly 2,000 | 2,000 unique IDs | **PASS** |
| **Voice Disjointness** | 17 train, 4 val, 4 test, 0 overlap | 17 / 4 / 4 disjoint | **PASS** |
| **Text Disjointness** | Disjoint UNKNOWN text vocabularies | 60 train, 30 val, 30 test | **PASS** |
| **Lexical Purity** | Zero target words in UNKNOWN | 0 target words found | **PASS** |
| **SHA-256 Collisions** | Zero duplicate waveforms | 2,000 unique hashes | **PASS** |
| **Format Integrity** | 16 kHz mono, 16,000 samples, finite | Verified on all 2,000 files | **PASS** |
| **Peak Amplitude** | $\le 0.99$ | All $\le 0.98$ | **PASS** |

---

## Freezing Decision

The v2 dataset satisfies all governance, balance, diversity, and integrity requirements.
Manifests are officially **FROZEN** for Phase 34 training and Phase 35 evaluation.
