# Six-Class Vietnamese KWS Dataset (VieNeu-TTS)

## 1. Overview

This directory contains the six-class Vietnamese keyword spotting (KWS) dataset combining:
1. The accepted 1,000-utterance four-class synthetic command dataset (`TRAI`, `PHAI`, `LEN`, `XUONG`).
2. 250 distinct `UNKNOWN` out-of-vocabulary Vietnamese speech utterances synthesized locally using VieNeu-TTS v3 Turbo.
3. 250 distinct `SILENCE` / non-command audio clips generated across disjoint acoustic condition groups.

Total scale: **1,500 balanced audio clips** ($250$ per class across $6$ classes).

> [!IMPORTANT]
> **Strict Governance Mandates**:
> - **Subjective Human Intelligibility**: **`PENDING_HUMAN_LISTENER_VERIFICATION`** (Phase 25 blinded review pending).
> - **Acoustic Generalization**: **`NOT_VALIDATED_ON_NATURAL_SPEECH`**.
> - **Noise Robustness**: Maintained as an independent evaluation concern; no noise corruption has been injected into the core manifests.

---

## 2. Class Definitions & Split Allocations

| Class Index | Label | Target / Text Inventory | Train (68%) | Val (16%) | Test (16%) | Total |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| `0` | `TRAI` | "Trái" (/ʈaːj˧˥/, sắc tone) | 170 | 40 | 40 | **250** |
| `1` | `PHAI` | "Phải" (/faːj˧˩˨/, hỏi tone) | 170 | 40 | 40 | **250** |
| `2` | `LEN` | "Lên" (/len˧˧/, ngang tone) | 170 | 40 | 40 | **250** |
| `3` | `XUONG` | "Xuống" (/suəŋ˧˥/, sắc tone) | 170 | 40 | 40 | **250** |
| `4` | `UNKNOWN` | 30 disjoint Vietnamese non-target words | 170 | 40 | 40 | **250** |
| `5` | `SILENCE` | 11 disjoint non-speech acoustic conditions | 170 | 40 | 40 | **250** |
| **Total** | — | — | **1,020** | **240** | **240** | **1,500** |

---

## 3. Disjoint Partitioning Guarantees

1. **Speaker Voices (Speech Classes)**:
   - **Train (17 voices)**: `Adam`, `Adam bựa`, `Hải Đăng`, `Minh Triết`, `Minh Đức`, `Mỹ Duyên`, `Ngọc Huyền`, `Ngọc Linh`, `Quang Sơn`, `Thanh Bình`, `Thiền Tâm Đức`, `Thiện Minh`, `Thái Sơn`, `Thục Đoan`, `Trúc Ly`, `Xuân Vĩnh`, `Đoan Trang`.
   - **Validation (4 voices)**: `Mai Anh`, `Ngọc Trân`, `Phạm Tuyên`, `Thùy Dung`.
   - **Test (4 voices)**: `Kim Thanh`, `Quốc Tuấn`, `Quỳnh Anh`, `Đức Trí`.
   - $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$.
2. **UNKNOWN Text Vocabularies**:
   - **Train ($V_{\text{train}}$)**: `Dừng`, `Bật`, `Tắt`, `Mở`, `Đóng`, `Chạy`, `Vào`, `Ra`, `Trời`, `Phát`.
   - **Val ($V_{\text{val}}$)**: `Đi`, `Đứng`, `Nghe`, `Nói`, `Hỏi`, `Tìm`, `Chọn`, `Xoay`, `Lùi`, `Quay`.
   - **Test ($V_{\text{test}}$)**: `Chào`, `Cứu`, `Hủy`, `Lưu`, `Xóa`, `Phóng`, `Thu`, `Khóa`, `Được`, `Không`.
   - $V_{\text{train}} \cap V_{\text{val}} = \emptyset$, $V_{\text{train}} \cap V_{\text{test}} = \emptyset$, $V_{\text{val}} \cap V_{\text{test}} = \emptyset$. No target command words present.
3. **SILENCE Conditions**:
   - **Train**: Digital zero fixture, thermal dither, stationary room tone ($400\text{ Hz}$ LPF), electrical HVAC hum ($50/60\text{ Hz}$), airflow pink noise.
   - **Val**: Triangular dither, quiet office bandpass tone ($150\text{ Hz}$–$1,200\text{ Hz}$), acoustic rumble ($250\text{ Hz}$ LPF).
   - **Test**: Brownian noise ($1/f^2$), comb-filtered room resonance ($20\text{ ms}$ delay), high-pass shaped dither ($2,000\text{ Hz}$ HPF).
4. **Hashes**:
   - All 1,500 audio files possess unique SHA-256 hashes with zero cross-split collisions.

---

## 4. Directory Layout

```
experiments/synthetic_kws_dataset_vieneu_6class/
├── manifest.csv                    # Master dataset manifest (1,500 rows)
├── build_6class_dataset.py         # Generation and assembly script
├── README.md                       # This overview documentation
├── audio/
│   ├── raw_48k/                    # 48 kHz source audio for new UNKNOWN and SILENCE
│   └── kws_16k/                    # 16 kHz 1.0s WAVs for new UNKNOWN and SILENCE
├── splits/
│   ├── train_manifest.csv          # 1,020 rows (170 per class)
│   ├── val_manifest.csv            # 240 rows (40 per class)
│   └── test_manifest.csv           # 240 rows (40 per class)
├── configs/
│   └── dataset_config.json         # Dataset metadata and specifications
├── manifests/
│   └── manifest.csv                # Mirror of master manifest
└── reports/
    ├── SIX_CLASS_DATASET_BUILD_REPORT.md
    ├── SIX_CLASS_DATASET_VALIDATION_REPORT.md
    └── SIX_CLASS_DATA_READINESS_PLAN.md
```
