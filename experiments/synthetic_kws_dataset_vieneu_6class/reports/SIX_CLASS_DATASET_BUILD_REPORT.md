# Six-Class Vietnamese KWS Dataset Build Report

**Date:** 2026-10-18  
**Workspace:** `D:\trainidentificationaudio`  
**Dataset Directory:** `experiments/synthetic_kws_dataset_vieneu_6class/`  
**Dataset Scale:** 1,500 balanced audio utterances (250 per class across 6 classes)  
**Classes:** `TRAI`, `PHAI`, `LEN`, `XUONG`, `UNKNOWN`, `SILENCE`  

---

## 1. Executive Summary & Objectives

Phase 28 expands the verified 1,000-utterance four-class synthetic Vietnamese command dataset into an isolated six-class keyword spotting dataset by incorporating:
1. **`UNKNOWN`**: 250 distinct Vietnamese speech utterances representing non-target commands and everyday short words, synthesized using the local open-source VieNeu-TTS v3 Turbo engine.
2. **`SILENCE`**: 250 distinct non-command audio utterances covering digital zero, thermal dither, stationary room tone, HVAC hum, and ambient acoustic simulations under disjoint condition groups across splits.

The original four-class samples (`TRAI`, `PHAI`, `LEN`, `XUONG`) are preserved bit-for-bit from the accepted Phase 23 dataset without modification or regeneration.

### Strict Governance Status
> [!IMPORTANT]
> - **Human Perceptual Intelligibility Status**: **`PENDING_HUMAN_LISTENER_VERIFICATION`**. High acoustic fidelity and automated metric passes do not replace human perceptual evaluation (Phase 25 blinded review remains pending).
> - **Operational Real-World Generalization**: **`NOT_VALIDATED_ON_NATURAL_SPEECH`**. This dataset establishes multi-class synthetic discrimination; it does not replace field speech testing.
> - **Noise Robustness**: Maintained as an independent evaluation concern; no indiscriminate noise corruption has been injected into the balanced baseline manifests.

---

## 2. Dataset Composition & Class Allocations

The dataset provides exactly 250 balanced examples per class:

| Class Label | Target Semantics | Generation Engine | Source / Provenance | Train (68%) | Val (16%) | Test (16%) | Total |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **`TRAI`** | "Trái" (Left) | VieNeu-TTS v3 Turbo | Referenced from Phase 23 1,000-dataset | 170 | 40 | 40 | **250** |
| **`PHAI`** | "Phải" (Right) | VieNeu-TTS v3 Turbo | Referenced from Phase 23 1,000-dataset | 170 | 40 | 40 | **250** |
| **`LEN`** | "Lên" (Up) | VieNeu-TTS v3 Turbo | Referenced from Phase 23 1,000-dataset | 170 | 40 | 40 | **250** |
| **`XUONG`** | "Xuống" (Down) | VieNeu-TTS v3 Turbo | Referenced from Phase 23 1,000-dataset | 170 | 40 | 40 | **250** |
| **`UNKNOWN`** | Out-of-vocabulary speech | VieNeu-TTS v3 Turbo | Newly synthesized (250 distinct clips) | 170 | 40 | 40 | **250** |
| **`SILENCE`** | Ambient / no command | DSP Audio Generator | Newly generated (250 distinct clips) | 170 | 40 | 40 | **250** |
| **Total** | — | — | — | **1,020** | **240** | **240** | **1,500** |

---

## 3. Speaker Voice Partitioning (Zero-Leakage Architecture)

Speaker voice allocations strictly preserve the authoritative roster verified in Phase 27 across all speech classes:

- **Train Split (17 distinct voice presets)**:
  `Adam`, `Adam bựa`, `Hải Đăng`, `Minh Triết`, `Minh Đức`, `Mỹ Duyên`, `Ngọc Huyền`, `Ngọc Linh`, `Quang Sơn`, `Thanh Bình`, `Thiền Tâm Đức`, `Thiện Minh`, `Thái Sơn`, `Thục Đoan`, `Trúc Ly`, `Xuân Vĩnh`, `Đoan Trang`
  $\rightarrow 17 \text{ voices} \times 10 \text{ samples} = 170 \text{ samples per speech class}$.
- **Validation Split (4 distinct voice presets)**:
  `Mai Anh`, `Ngọc Trân`, `Phạm Tuyên`, `Thùy Dung`
  $\rightarrow 4 \text{ voices} \times 10 \text{ samples} = 40 \text{ samples per speech class}$.
- **Held-Out Test Split (4 distinct voice presets)**:
  `Kim Thanh`, `Quốc Tuấn`, `Quỳnh Anh`, `Đức Trí`
  $\rightarrow 4 \text{ voices} \times 10 \text{ samples} = 40 \text{ samples per speech class}$.

$\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$. Zero voice leakage across splits.

---

## 4. `UNKNOWN` Speech Design & Disjoint Text Inventories

To evaluate acoustic generalization beyond memorizing a fixed distractor vocabulary, the `UNKNOWN` text inventory is strictly disjoint across splits:

### 4.1 Split Text Inventories
1. **Train Vocabulary ($V_{\text{train}}$, 10 distinct words)**:
   - `Dừng` (/zɯŋ/ - stop)
   - `Bật` (/ɓət/ - turn on)
   - `Tắt` (/tat/ - turn off)
   - `Mở` (/məː/ - open)
   - `Đóng` (/ɗawŋm/ - close)
   - `Chạy` (/t͡ɕaj/ - run)
   - `Vào` (/vaːw/ - enter)
   - `Ra` (/zaː/ - exit)
   - `Trời` (/ʈəːj/ - sky/weather; confusable initial retroflex /tr-/)
   - `Phát` (/faːt/ - play/broadcast; confusable initial labiodental /ph-/)
   *Each of the 17 train voices synthesizes these 10 distinct words ($17 \times 10 = 170$ utterances).*

2. **Validation Vocabulary ($V_{\text{val}}$, 10 distinct words)**:
   - `Đi` (/ɗi/ - go)
   - `Đứng` (/ɗɨŋ/ - stand)
   - `Nghe` (/ŋɛ/ - listen)
   - `Nói` (/nɔj/ - speak)
   - `Hỏi` (/hɔj/ - ask; confusable hỏi dipping tone)
   - `Tìm` (/tim/ - search)
   - `Chọn` (/t͡ɕɔn/ - select)
   - `Xoay` (/swaj/ - rotate; confusable initial /x-/)
   - `Lùi` (/luj/ - step back; confusable initial /l-/)
   - `Quay` (/kwaːj/ - turn)
   *Each of the 4 validation voices synthesizes these 10 distinct words ($4 \times 10 = 40$ utterances).*

3. **Held-Out Test Vocabulary ($V_{\text{test}}$, 10 distinct words)**:
   - `Chào` (/t͡ɕaːw/ - hello)
   - `Cứu` (/kiw/ - emergency/help)
   - `Hủy` (/hwi/ - cancel)
   - `Lưu` (/liw/ - save; confusable initial /l-/)
   - `Xóa` (/swaː/ - delete; confusable initial /x-/)
   - `Phóng` (/fawŋm/ - zoom; confusable initial /ph-/)
   - `Thu` (/tʰu/ - shrink/record)
   - `Khóa` (/xwaː/ - lock)
   - `Được` (/ɗɨək/ - okay/allowed)
   - `Không` (/xəwŋm/ - no/cancel)
   *Each of the 4 test voices synthesizes these 10 distinct words ($4 \times 10 = 40$ utterances).*

### 4.2 Disjointness & Safety Verification
- $V_{\text{train}} \cap V_{\text{val}} = \emptyset$
- $V_{\text{train}} \cap V_{\text{test}} = \emptyset$
- $V_{\text{val}} \cap V_{\text{test}} = \emptyset$
- **Target Command Check**: None of the 30 words contain `Trái`, `Phải`, `Lên`, or `Xuống` as standalone words or sub-strings.

---

## 5. `SILENCE` Design & Disjoint Acoustic Conditions

`SILENCE` is operationally defined as non-command audio, including digital reference, thermal dither, and non-speech stationary background ambience. To prevent template leakage, acoustic conditions are partitioned across splits:

- **Train Split (170 samples)**:
  - 1 Exact digital zero reference fixture (`sample_id: vnkws_silence_0001`, exactly 0.0 float32 across all 16,000 samples).
  - 40 Gaussian thermal dither clips (varying RNG seeds, RMS $-76$ to $-67\text{ dBFS}$).
  - 40 Stationary room tone clips (Butterworth low-pass filtered white noise at $400\text{ Hz}$, RMS $-55$ to $-48\text{ dBFS}$).
  - 45 Electrical HVAC equipment hum clips ($50\text{ Hz}$ or $60\text{ Hz}$ fundamental + weak harmonics, RMS $-54$ to $-46\text{ dBFS}$).
  - 44 Gentle airflow pink noise clips ($1/f$ IIR filter roll-off, RMS $-55$ to $-46\text{ dBFS}$).
- **Validation Split (40 samples)**:
  - 15 Triangular dither clips (RMS $-72$ to $-62\text{ dBFS}$).
  - 15 Quiet office bandpass tone clips (Butterworth bandpass $150\text{ Hz}$–$1,200\text{ Hz}$, RMS $-52$ to $-45\text{ dBFS}$).
  - 10 Low-frequency acoustic rumble clips (steep low-pass cutoff at $250\text{ Hz}$, RMS $-50$ to $-42\text{ dBFS}$).
- **Held-Out Test Split (40 samples)**:
  - 15 Brownian ambient tone clips ($1/f^2$ noise, RMS $-52$ to $-44\text{ dBFS}$).
  - 15 Comb-filtered room resonance clips ($20\text{ ms}$ reflection delay, RMS $-50$ to $-44\text{ dBFS}$).
  - 10 High-pass shaped dither clips (high-pass cutoff at $2,000\text{ Hz}$, RMS $-70$ to $-60\text{ dBFS}$).

All 250 SILENCE clips:
- Duration: Exactly 16,000 samples at 16,000 Hz (1.0 s).
- Peak amplitude $< 0.05$ (safely below speech peaks $> 0.20$).
- 100% unique SHA-256 hashes (0 duplicates across or within splits).

---

## 6. Directory Structure & Deliverables

```
experiments/synthetic_kws_dataset_vieneu_6class/
├── manifest.csv                    # Master manifest (1,500 rows)
├── build_6class_dataset.py         # Generation and assembly script
├── audio/
│   ├── raw_48k/                    # 48 kHz source audio for new UNKNOWN and SILENCE
│   └── kws_16k/                    # 16 kHz 1.0s WAVs for new UNKNOWN and SILENCE
├── splits/
│   ├── train_manifest.csv          # 1,020 rows (170 per class)
│   ├── val_manifest.csv            # 240 rows (40 per class)
│   └── test_manifest.csv           # 240 rows (40 per class)
├── configs/
│   └── dataset_config.json         # Dataset metadata and voice/split allocations
├── manifests/
│   └── manifest.csv                # Backup mirror of master manifest
└── reports/
    ├── SIX_CLASS_DATASET_BUILD_REPORT.md
    ├── SIX_CLASS_DATASET_VALIDATION_REPORT.md
    └── SIX_CLASS_DATA_READINESS_PLAN.md
```
