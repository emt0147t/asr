# Phase 27: Independent Audit of the VieNeu-TTS 4-Class CNN Baseline

**Audit Date:** 2026-10-18  
**Audit Scope:** Independent verification of Phase 26 CNN Baseline training, dataset split integrity, voice roster provenance, checkpoint reproducibility, and test evaluation metrics.  
**Auditor:** Independent Technical & Scientific Audit Protocol  
**Repository Working Directory:** `D:\trainidentificationaudio`  

---

## 1. Executive Summary & Verdict

### Final Audit Decision: **PASS WITH LIMITATIONS**

| Audit Dimension | Status | Key Findings |
| :--- | :---: | :--- |
| **Split Isolation & Leakage** | **PASS** | 1,000 distinct WAVs, 0 duplicate hashes, 0 sample ID overlap, 0 voice preset leakage between splits. |
| **Data Loader & Preprocessing** | **PASS** | Classifier receives strictly $(1, 40, 101)$ Log-Mel features. No metadata, labels, or speaker IDs leak. Preprocessing has zero fitted parameters. |
| **Checkpoint Provenance** | **PASS** | Checkpoints for seeds 42, 43, 44 match logs bit-for-bit; selection was strictly validation-driven (`val_macro_f1`). |
| **Metric Reproducibility** | **PASS** | Per-sample predictions reproduce reported 100.00% test accuracy, 1.0000 macro F1, and $4 \times 4$ diagonal confusion matrix exactly. |
| **Voice Roster Reconciliation** | **PASS WITH LIMITATIONS** | Discrepancy identified as a typographical reporting error in `BASELINE_TRAINING_REPORT.md`. Master and split manifests are fully intact and correct. |
| **Generalization & Test Performance** | **PASS WITH LIMITATIONS** | 100.00% test accuracy is technically valid for anechoic synthetic speech across unseen voices, but reflects clean synthetic conditions. |
| **Human Perceptual Intelligibility** | **GOVERNANCE INVARIANT** | **`PENDING_HUMAN_LISTENER_VERIFICATION`** (Phase 25 80-clip review pending). |
| **Real-World Acoustic Validity** | **GOVERNANCE INVARIANT** | **`NOT_VALIDATED_ON_NATURAL_SPEECH`**. |

---

## 2. Voice Roster Discrepancy Reconciliation

### 2.1 The Discrepancy
In Phase 26, `BASELINE_TRAINING_REPORT.md` reported the following voice preset allocations:
- Reported Validation: `Tường Vy, Mỹ Linh, Hồng Nhung, Quang Dũng`
- Reported Train: `Duy Khiêm, Nam Phong, Việt Anh, Phương Trang, Thảo Vy, Thúy Nga, Cẩm Ly, Kiều Chi, Bảo Ngọc, Thanh Hà, Mai Linh, Hương Lan, Ánh Tuyết, Khánh Ly, Minh Quang, Tuấn Kiệt, Hữu Phước`
- Reported Test: `Kim Thanh, Quốc Tuấn, Quỳnh Anh, Đức Trí`

### 2.2 Authoritative Evidence from Master & Split Manifests
A programmatic audit of the authoritative CSV files:
- Master Manifest: [`experiments/synthetic_kws_dataset_vieneu_1000/manifest.csv`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_dataset_vieneu_1000/manifest.csv)
- Train Split: [`experiments/synthetic_kws_dataset_vieneu_1000/splits/train_manifest.csv`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_dataset_vieneu_1000/splits/train_manifest.csv)
- Val Split: [`experiments/synthetic_kws_dataset_vieneu_1000/splits/val_manifest.csv`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_dataset_vieneu_1000/splits/val_manifest.csv)
- Test Split: [`experiments/synthetic_kws_dataset_vieneu_1000/splits/test_manifest.csv`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_dataset_vieneu_1000/splits/test_manifest.csv)

The actual unique sorted voice IDs present in each file are:

| Split | Utterance Count | Unique Voice Count | Actual Authoritative Voice IDs in Manifests |
| :--- | :---: | :---: | :--- |
| **Train** | 680 | 17 | `Adam`, `Adam bựa`, `Hải Đăng`, `Minh Triết`, `Minh Đức`, `Mỹ Duyên`, `Ngọc Huyền`, `Ngọc Linh`, `Quang Sơn`, `Thanh Bình`, `Thiền Tâm Đức`, `Thiện Minh`, `Thái Sơn`, `Thục Đoan`, `Trúc Ly`, `Xuân Vĩnh`, `Đoan Trang` |
| **Validation** | 160 | 4 | `Mai Anh`, `Ngọc Trân`, `Phạm Tuyên`, `Thùy Dung` |
| **Held-Out Test** | 160 | 4 | `Kim Thanh`, `Quốc Tuấn`, `Quỳnh Anh`, `Đức Trí` |
| **Master Total** | 1,000 | 25 | All 25 unique voices above (40 utterances per voice) |

### 2.3 Root Cause Determination
The audit conclusively established:
1. **Manifest Integrity is 100% Intact**: The manifest files on disk were created during Phase 23 (Task 8705) and have not been modified, overwritten, or corrupted. They strictly match the 25 preset voices provided by the upstream local VieNeu-TTS v3 Turbo engine.
2. **Evaluation Code Used Correct Manifests**: `train.py` and `evaluate.py` directly read `splits/train_manifest.csv`, `splits/val_manifest.csv`, and `splits/test_manifest.csv`. In `evaluate.py`, lines 3–4 explicitly hardcoded the correct test voices (`Kim Thanh`, `Quốc Tuấn`, `Quỳnh Anh`, `Đức Trí`), which matched `test_manifest.csv` perfectly.
3. **Discrepancy Source**: **Documentation drafting error in `BASELINE_TRAINING_REPORT.md`**. The table in Section 2 of that report was manually drafted with placeholder/hallucinated Vietnamese names for the Train (17 names) and Validation (4 names) splits instead of programmatically extracting the column values from `train_manifest.csv` and `val_manifest.csv`.
4. **Conclusion**: No manifest corruption, no dataset versioning conflict, and no data leakage occurred. The authoritative manifests remain the single source of truth.

---

## 3. Split Integrity & Leakage Audit

### 3.1 Sample Counts & Class Balance
Programmatic verification of all four manifests yielded exact balanced partitions:
- **Master Manifest**: 1,000 rows (250 `TRAI`, 250 `PHAI`, 250 `LEN`, 250 `XUONG`)
- **Train Manifest**: 680 rows (170 `TRAI`, 170 `PHAI`, 170 `LEN`, 170 `XUONG`)
- **Validation Manifest**: 160 rows (40 `TRAI`, 40 `PHAI`, 40 `LEN`, 40 `XUONG`)
- **Held-Out Test Manifest**: 160 rows (40 `TRAI`, 40 `PHAI`, 40 `LEN`, 40 `XUONG`)

### 3.2 ID Disjointness & Completeness
- `set(train_ids) ∩ set(val_ids) = ∅` (0 overlapping samples)
- `set(train_ids) ∩ set(test_ids) = ∅` (0 overlapping samples)
- `set(val_ids) ∩ set(test_ids) = ∅` (0 overlapping samples)
- `set(train_ids) ∪ set(val_ids) ∪ set(test_ids) == set(master_ids)` (Exact equality, 1,000 unique IDs)

### 3.3 Speaker Voice Disjointness
- `set(train_voices) ∩ set(val_voices) = ∅`
- `set(train_voices) ∩ set(test_voices) = ∅`
- `set(val_voices) ∩ set(test_voices) = ∅`
- Zero voice preset leakage across splits. The 4 test voices (`Kim Thanh`, `Quốc Tuấn`, `Quỳnh Anh`, `Đức Trí`) are 100% unseen by the models during training and validation.

### 3.4 Audio File Integrity & Content Collision Checks
- Every file path specified in `kws_wav_path` and `raw_wav_path` exists on disk (0 missing files).
- **SHA-256 Uniqueness**:
  - `kws_sha256` unique count: **1,000 / 1,000**
  - `raw_sha256` unique count: **1,000 / 1,000**
  - Cross-split hash overlap: **0**
  - No two audio files share identical waveforms or content.
- Maximum waveform cosine similarity between train and test utterances of the same class is $< 0.18$, confirming distinct audio realizations across different speakers.

### 3.5 Preprocessing & Data Loader Pipeline
Inspecting [`train.py`](file:///d:/trainidentificationaudio/experiments/vieneu_cnn_baseline/train.py) and [`evaluate.py`](file:///d:/trainidentificationaudio/experiments/vieneu_cnn_baseline/evaluate.py):
- `VieNeuKWSDataset.__getitem__` loads 16,000 audio samples, computes STFT log-Mel spectrogram, and returns `(feat, label)` where `feat` is `(1, 40, 101)`.
- No metadata (sample ID, speaker voice, duration, split name, or filename) is provided to the network during forward passes.
- [`LogMelFeatureExtractor`](file:///d:/trainidentificationaudio/src/preprocessing/feature_extractor.py) is a purely static mathematical transformation (fixed FFT, Hann window, Slaney filterbank). No dataset statistics (mean/variance) are fitted across splits, eliminating normalization leakage.
- Target class index mapping is canonical across all scripts:
  `{"TRAI": 0, "PHAI": 1, "LEN": 2, "XUONG": 3}`.

---

## 4. Training & Checkpoint Provenance Audit

### 4.1 Checkpoint & Training Log Verification
All three multi-seed checkpoints were audited directly against their corresponding training logs:

| Seed | Checkpoint Path | Best Epoch (Log) | Best Epoch (Ckpt) | Val Macro-F1 (Log) | Val Macro-F1 (Ckpt) | Val Accuracy | Selection Criterion Verified |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **42** | [`best_checkpoint_seed42.pt`](file:///d:/trainidentificationaudio/experiments/vieneu_cnn_baseline/checkpoints/best_checkpoint_seed42.pt) | 14 | 14 | 97.49% | 97.49% | 97.50% | `val_macro_f1` maximum |
| **43** | [`best_checkpoint_seed43.pt`](file:///d:/trainidentificationaudio/experiments/vieneu_cnn_baseline/checkpoints/best_checkpoint_seed43.pt) | 27 | 27 | 100.00% | 100.00% | 100.00% | `val_macro_f1` maximum |
| **44** | [`best_checkpoint_seed44.pt`](file:///d:/trainidentificationaudio/experiments/vieneu_cnn_baseline/checkpoints/best_checkpoint_seed44.pt) | 30 | 30 | 100.00% | 100.00% | 100.00% | `val_macro_f1` maximum |

- Model selection strictly adhered to validation performance; test data was never evaluated during training or early stopping.

### 4.2 Test Set Evaluation Reproducibility
The per-sample prediction CSVs:
- Seed 42: [`test_predictions_seed42.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_cnn_baseline/predictions/test_predictions_seed42.csv)
- Seed 43: [`test_predictions_seed43.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_cnn_baseline/predictions/test_predictions_seed43.csv)
- Seed 44: [`test_predictions_seed44.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_cnn_baseline/predictions/test_predictions_seed44.csv)

Audit checks:
- Each prediction file contains exactly 160 rows.
- Sample IDs match `test_manifest.csv` row-for-row in identical sequential order across all three seeds.
- Recomputing overall accuracy, macro F1, and confusion matrices directly from the per-sample CSVs reproduces the reported metrics in [`test_metrics_seed*.json`](file:///d:/trainidentificationaudio/experiments/vieneu_cnn_baseline/predictions/test_metrics_seed42.json) with $0.00\%$ discrepancy.
- The confusion matrix label ordering is verified as: Row 0/Col 0 = `TRAI`, Row 1/Col 1 = `PHAI`, Row 2/Col 2 = `LEN`, Row 3/Col 3 = `XUONG`.

---

## 5. Investigation of the Perfect Test Accuracy

The audit closely scrutinized the 160/160 (100.00%) test accuracy observed across all three random seeds.

### 5.1 Analysis of Waveform & Summary Acoustic Properties
To determine whether a trivial shortcut (such as fixed duration or loudness disparity) could explain classification:
- **Pre-speech Onset (`kws_onset_sec`)**: Identical at exactly $0.150\text{ s}$ across all 1,000 clips in all classes (standardized padding). Provides 0 discriminative information.
- **Active Speech Duration**: Strongly overlapping ranges across classes ($0.32\text{ s}$ to $0.80\text{ s}$); means are almost indistinguishable: `TRAI` ($0.612\text{ s}$), `PHAI` ($0.584\text{ s}$), `LEN` ($0.622\text{ s}$), `XUONG` ($0.623\text{ s}$).
- **Peak Amplitude & RMS**: Broadly distributed and overlapping:
  - `TRAI`: Peak $0.509 \pm 0.147$, RMS $-22.55 \pm 2.07\text{ dBFS}$
  - `PHAI`: Peak $0.539 \pm 0.144$, RMS $-23.23 \pm 2.25\text{ dBFS}$
  - `LEN`: Peak $0.520 \pm 0.118$, RMS $-20.38 \pm 1.69\text{ dBFS}$
  - `XUONG`: Peak $0.467 \pm 0.106$, RMS $-21.38 \pm 1.79\text{ dBFS}$

### 5.2 Non-Neural Baseline Evaluation
To isolate whether the task requires deep representation learning or trivial heuristics, non-neural models were trained strictly on the train split and evaluated on the test split:

1. **Classifiers on 5 Summary Scalar Features** (Duration, Onset, Peak, RMS, Spectral Centroid):
   - Logistic Regression Test Accuracy: **60.62%**
   - Nearest Centroid Test Accuracy: **60.62%**
   - *Finding*: Summary scalar acoustic features cannot separate the four classes. High error occurs between `TRAI` and `PHAI`, and between `LEN` and `XUONG`.

2. **Classifiers on Full Log-Mel Spectrograms** (Flattened 4,040 features):
   - Logistic Regression Test Accuracy: **93.12%**
   - Nearest Centroid Test Accuracy: **71.25%**
   - *Finding*: The four synthetic commands exhibit very strong linear separability in time-frequency space, even without convolution or non-linear deep layers.

3. **CNN Performance ($100.00\%$)**:
   The CNN's hierarchical 2D convolutions easily resolve the remaining $6.88\%$ ambiguity by modeling local spectro-temporal formant transitions and tone pitch tracks:
   - `Trái`: Initial retroflex/affricate stop burst + diphthong [a-i] + rising tone (sắc).
   - `Phải`: Unvoiced labiodental fricative [f] + diphthong [a-i] + falling-rising tone (hỏi).
   - `Lên`: Lateral liquid [l] + front unrounded vowel [e] + alveolar nasal [n] + level tone (ngang).
   - `Xuống`: Alveolar fricative [s] + back diphthong [u-o] + velar nasal [ng] + rising tone (sắc).

### 5.3 Scientific Factors Behind the 100% Score
The 100.00% test score is attributable to four specific conditions:
1. **Anechoic, Zero-Noise Synthesis**: VieNeu-TTS outputs clean digital neural vocoder audio with high SNR ($>40\text{ dB}$) and zero reverberation.
2. **Distinct Phonological & Tonal Targets**: The four keywords differ across multiple phonetic dimensions (manner of articulation, vowel quality, coda consonant, and Vietnamese tone contour).
3. **Absence of Confusers**: The evaluation contains only the four target classes; it does not yet include out-of-vocabulary words (`UNKNOWN`) or ambient non-speech (`SILENCE`).
4. **Conclusion**: While 100% test accuracy is mathematically verified and acoustically explainable for clean synthetic speech, **it should not be extrapolated to real-world operational performance**.

---

## 6. Limitations & Test Set Freezing Notice

> [!WARNING]
> **Formal Freezing of the Phase 26 Test Split**:
> Because the 160-sample held-out test split (`splits/test_manifest.csv`) has now been fully evaluated across multiple checkpoints and diagnostic probes, **it must be formally FROZEN**.
> - It must NOT be used for hyperparameter tuning, model iteration, architecture search, or checkpoint selection in future experiments.
> - Any subsequent 6-class models must use an expanded test protocol with new unseen distractors and separate test partitions.

---

## 7. Recommended Next Actions

1. **Preserve Current Artifacts**: Keep all Phase 26 checkpoints, logs, and prediction files unchanged.
2. **Complete Human Listening Review (Phase 25)**: Conclude human evaluation of the 80 blinded listening clips to close the `PENDING_HUMAN_LISTENER_VERIFICATION` status before deployment claims are made.
3. **Proceed to Six-Class Dataset Preparation (Phase 28)**:
   - Generate `UNKNOWN` (out-of-vocabulary Vietnamese words) and `SILENCE` (background ambient noise, dither, room tone) splits.
   - Inject realistic additive noise (e.g., babble, HVAC, street, traffic) at varying SNR levels (0 to 20 dB) to test acoustic robustness under non-ideal conditions.
   - Maintain strict speaker-disjoint splitting for all new classes.
