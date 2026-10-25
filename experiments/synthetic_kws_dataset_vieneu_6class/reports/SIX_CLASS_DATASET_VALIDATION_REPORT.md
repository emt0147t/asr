# Six-Class Vietnamese KWS Dataset Validation Report

**Date:** 2026-10-18  
**Dataset Path:** `experiments/synthetic_kws_dataset_vieneu_6class/`  
**Master Manifest:** `experiments/synthetic_kws_dataset_vieneu_6class/manifest.csv`  
**Validation Suite:** `tests/test_vieneu_6class_dataset.py`  
**Verification Scope:** Automated integrity, format, class balance, split isolation, and cross-split leakage checks.  

---

## 1. Validation Criteria & Results Summary

| Validation Check | Acceptance Threshold | Measured Value | Result |
| :--- | :--- | :--- | :---: |
| **Total Utterance Count** | Exactly 1,500 rows | 1,500 rows | **PASS** |
| **Master Class Balance** | 250 rows per class (6 classes) | 250 rows / class | **PASS** |
| **Split Count Distribution** | Train: 1,020 | Val: 240 | Test: 240 | Train: 1,020 \| Val: 240 \| Test: 240 | **PASS** |
| **Per-Split Class Balance** | Train: 170/cls \| Val: 40/cls \| Test: 40/cls | Exactly balanced | **PASS** |
| **Voice Preset Isolation** | $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$ | 0 overlapping voices | **PASS** |
| **UNKNOWN Text Isolation** | Disjoint vocabulary across splits | 0 overlapping words | **PASS** |
| **Target Command Protection** | 0 target keywords in UNKNOWN texts | 0 instances | **PASS** |
| **SILENCE Template Isolation** | Disjoint condition models across splits | 0 overlapping models | **PASS** |
| **SHA-256 Hash Uniqueness** | 1,500 unique KWS hashes | 1,500 unique hashes | **PASS** |
| **Cross-Split Hash Collisions** | 0 hash overlaps across splits | 0 hash collisions | **PASS** |
| **Four-Class Baseline Preservation** | 1,000 reference rows match original bit-exact | 1,000 / 1,000 matches | **PASS** |
| **Audio Format Compliance** | 16 kHz, 1.0s, 16,000 samples, mono PCM WAV | 100% compliant | **PASS** |
| **Clipping & Signal Health** | Peak $< 0.999$, finite values, no NaNs/Infs | 100% compliant | **PASS** |

---

## 2. Split Partitioning & Isolation Details

### 2.1 Class Distribution by Split

| Class Label | Intended Text / Condition | Train Split | Validation Split | Held-Out Test Split | Total Utterances |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `TRAI` | "Trái" | 170 | 40 | 40 | **250** |
| `PHAI` | "Phải" | 170 | 40 | 40 | **250** |
| `LEN` | "Lên" | 170 | 40 | 40 | **250** |
| `XUONG` | "Xuống" | 170 | 40 | 40 | **250** |
| `UNKNOWN` | 30 non-target Vietnamese words | 170 | 40 | 40 | **250** |
| `SILENCE` | 11 synthetic acoustic conditions | 170 | 40 | 40 | **250** |
| **Total** | — | **1,020** | **240** | **240** | **1,500** |

### 2.2 Voice Partitioning Audit
- **Train (17 voices)**: `Adam`, `Adam bựa`, `Hải Đăng`, `Minh Triết`, `Minh Đức`, `Mỹ Duyên`, `Ngọc Huyền`, `Ngọc Linh`, `Quang Sơn`, `Thanh Bình`, `Thiền Tâm Đức`, `Thiện Minh`, `Thái Sơn`, `Thục Đoan`, `Trúc Ly`, `Xuân Vĩnh`, `Đoan Trang`.
- **Validation (4 voices)**: `Mai Anh`, `Ngọc Trân`, `Phạm Tuyên`, `Thùy Dung`.
- **Test (4 voices)**: `Kim Thanh`, `Quốc Tuấn`, `Quỳnh Anh`, `Đức Trí`.
- Verified: Zero voice overlap between Train, Validation, and Test splits.

### 2.3 UNKNOWN Vocabulary Partitioning Audit
- **Train ($V_{\text{train}}$)**: `Dừng`, `Bật`, `Tắt`, `Mở`, `Đóng`, `Chạy`, `Vào`, `Ra`, `Trời`, `Phát` (10 words)
- **Validation ($V_{\text{val}}$)**: `Đi`, `Đứng`, `Nghe`, `Nói`, `Hỏi`, `Tìm`, `Chọn`, `Xoay`, `Lùi`, `Quay` (10 words)
- **Test ($V_{\text{test}}$)**: `Chào`, `Cứu`, `Hủy`, `Lưu`, `Xóa`, `Phóng`, `Thu`, `Khóa`, `Được`, `Không` (10 words)
- Verified: $V_{\text{train}} \cap V_{\text{val}} = \emptyset$, $V_{\text{train}} \cap V_{\text{test}} = \emptyset$, $V_{\text{val}} \cap V_{\text{test}} = \emptyset$. Zero target commands present.

### 2.4 SILENCE Condition Partitioning Audit
- **Train Conditions**: Exact digital zero fixture (`vnkws_silence_0001`), thermal dither, stationary room tone ($400\text{ Hz}$ LPF), electrical HVAC hum ($50/60\text{ Hz}$), airflow pink noise.
- **Validation Conditions**: Triangular dither noise, quiet office bandpass tone ($150\text{ Hz}$–$1,200\text{ Hz}$), low-frequency acoustic rumble ($250\text{ Hz}$ LPF).
- **Test Conditions**: Brownian ambient tone ($1/f^2$), comb-filtered room resonance ($20\text{ ms}$ delay), shaped high-pass dither ($2,000\text{ Hz}$ HPF).
- Verified: Condition models are strictly disjoint across splits.

---

## 3. Four-Class Baseline Preservation

The original 1,000 four-class samples in `experiments/synthetic_kws_dataset_vieneu_1000/manifest.csv` were compared against their referenced entries in the six-class manifest:
- Match Rate: **1,000 / 1,000** (100.0%)
- Sample IDs: Identical
- Class Labels: Identical
- Split Assignments: Identical
- SHA-256 Hashes: Identical
- Relative Paths: Resolving cleanly to existing audio files without duplicating storage or altering historical artifacts.

---

## 4. Signal Health & Technical Validation

A comprehensive batch inspection of all 1,500 audio files confirmed:
- Audio Format: RIFF WAV, 16,000 Hz, 16-bit linear PCM, mono.
- Duration: Exactly 16,000 samples ($1.000\text{ s}$).
- Amplitude Bounds:
  - Peak amplitude: $\max = 0.950$, $\min > 0.000$ (except digital zero reference). Zero clipping observed.
  - Active speech RMS: $-20\text{ dBFS}$ to $-28\text{ dBFS}$ for speech; $<-40\text{ dBFS}$ for non-speech silence.
- Boundary Safety: Speech onsets constrained to $\ge 0.05\text{ s}$ and endings to $\le 0.95\text{ s}$, ensuring zero boundary phoneme truncation.

---

## 5. Automated Regression Test Execution

Automated test suite `tests/test_vieneu_6class_dataset.py` executed via `pytest`:
- `test_canonical_label_mapping`: PASSED
- `test_manifest_counts_and_balance`: PASSED
- `test_split_counts_and_balance`: PASSED
- `test_voice_disjointness`: PASSED
- `test_unknown_text_inventory_disjointness`: PASSED
- `test_silence_condition_disjointness`: PASSED
- `test_zero_hash_overlap_across_splits`: PASSED
- `test_preservation_of_four_class_baseline`: PASSED
- `test_audio_format_and_signal_health`: PASSED

All 9/9 focused tests passed with exit code 0.
