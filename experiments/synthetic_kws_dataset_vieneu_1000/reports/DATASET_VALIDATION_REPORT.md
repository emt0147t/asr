# Vietnamese Synthetic KWS — 1,000-Utterance Dataset Validation Report

**Dataset Directory:** `experiments/synthetic_kws_dataset_vieneu_1000/`  
**Engine:** VieNeu-TTS v3 Turbo (Local CPU ONNX Runtime)  
**SDK Version:** `3.8.3`  
**Total Records:** 1000  
**Date:** October 17, 2026  
**Technical Validation Verdict:** **`PASS`**  
**Perceptual Intelligibility Status:** `PENDING_HUMAN_LISTENER_VERIFICATION`  
**Official Project Status:** `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Class and Split Balance Summary

| Keyword | Text | Vietnamese Tone | Train | Val | Test | Total |
| :--- | :--- | :--- | :-: | :-: | :-: | :-: |
| **TRAI** | Trái | sắc (high-rising) | 170 | 40 | 40 | **250** |
| **PHAI** | Phải | hỏi (mid-falling rising / dipping) | 170 | 40 | 40 | **250** |
| **LEN** | Lên | ngang (mid-level) | 170 | 40 | 40 | **250** |
| **XUONG** | Xuống | sắc (high-rising) | 170 | 40 | 40 | **250** |
| **Total** | | | **680 (68%)** | **160 (16%)** | **160 (16%)** | **1,000 (100%)** |

---

## 2. Split Partition and Disjoint Voice Policy

To prevent synthetic voice leakage across splits, preset voices were partitioned into strictly disjoint subsets:

* **Train Split (17 voices, 680 clips):**
  `Adam, Adam bựa, Hải Đăng, Minh Triết, Minh Đức, Mỹ Duyên, Ngọc Huyền, Ngọc Linh, Quang Sơn, Thanh Bình, Thiền Tâm Đức, Thiện Minh, Thái Sơn, Thục Đoan, Trúc Ly, Xuân Vĩnh, Đoan Trang`
* **Validation Split (4 voices, 160 clips):**
  `Mai Anh, Ngọc Trân, Phạm Tuyên, Thùy Dung`
* **Test Split (4 voices, 160 clips):**
  `Kim Thanh, Quốc Tuấn, Quỳnh Anh, Đức Trí`

* **Voice Overlap Check:**
  - Train $\cap$ Val: `[]` (Zero leakage)
  - Train $\cap$ Test: `[]` (Zero leakage)
  - Val $\cap$ Test: `[]` (Zero leakage)

---

## 3. Signal Integrity and Audio Verification

| Check | Requirement | Observed | Status |
| :--- | :--- | :--- | :-: |
| **Total Sample Count** | Exactly 1,000 pairs | 1000 pairs on disk | **PASS** |
| **Raw Sample Rate** | 48,000 Hz mono 16-bit PCM | 48,000 Hz, 1 channel | **PASS** |
| **KWS Sample Rate & Window** | 16,000 Hz, exactly 16,000 samples (1.0000s) | 16,000 Hz, 16,000 samples | **PASS** |
| **Finite Values** | Zero NaNs, zero Infs | 0 NaNs/Infs | **PASS** |
| **Headroom & Clipping** | Peak < 0.999 (normalized $\le 0.95$) | Max peak = 0.9500, 0 clipped | **PASS** |
| **Phoneme Truncation** | Margins $\ge 50$ ms, full word intact | 0 truncated | **PASS** |
| **Exact Duplicate Raw Hashes** | 0 exact duplicate WAVs | 0 duplicates | **PASS** |
| **Exact Duplicate KWS Hashes** | 0 exact duplicate WAVs | 0 duplicates | **PASS** |
| **SHA-256 Byte Match** | 100% manifest match | 0 mismatches | **PASS** |

---

## 4. Acoustic Distribution Summary

* **Active Speech Duration:** 0.6099 $\pm$ 0.1073 s
* **Speech RMS Level:** -21.88 $\pm$ 2.25 dBFS
* **Mean Spectral Centroid:** 85.7 $\pm$ 43.5 Hz
* **Mean Peak Amplitude:** 0.5086 (Max: 0.9500)

---

## 5. Validation Outcome

Overall Status: **`PASS`**
