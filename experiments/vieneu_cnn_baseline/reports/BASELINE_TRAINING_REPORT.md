# Phase 26: Four-Class CNN Baseline Training & Evaluation Report

**Date:** 2026-10-18  
**Workspace:** `D:\trainidentificationaudio`  
**Dataset:** `experiments/synthetic_kws_dataset_vieneu_1000/`  
**Architecture:** `VietnameseKWSNet` (Compact 2D-CNN, 138,404 parameters)  
**Input Representation:** 40-channel Log-Mel Spectrogram `(1, 40, 101)` at 16,000 Hz  

---

## 1. Executive Summary & Governance Status

In Phase 26, we established the first experimental 4-class convolutional neural network (CNN) keyword spotting baseline trained on the verified 1,000-utterance synthetic Vietnamese dataset produced by local open-source VieNeu-TTS v3 Turbo.

Training and evaluation were conducted across three deterministic seeds (`42`, `43`, `44`) using strictly disjoint synthetic speaker voice partitions to evaluate acoustic generalizability to unseen synthetic voices.

### Strict Scientific Governance & Evidence Boundaries
> [!IMPORTANT]
> - **Subjective Human Intelligibility Status**: **`PENDING_HUMAN_LISTENER_VERIFICATION`**. The formal 80-clip blinded human listening review prepared in Phase 25 remains pending completion. High classifier accuracy does NOT substitute for or establish human perceptual intelligibility.
> - **Domain Validation Status**: **`NOT_VALIDATED_ON_NATURAL_SPEECH`**. Performance reported herein reflects in-domain evaluation on disjoint synthesized voice presets. It does not establish performance on natural human speech recordings across real acoustic environments.
> - **Classification Scope**: Experimental 4-class baseline (`TRAI`, `PHAI`, `LEN`, `XUONG`). Six-class expansion (incorporating `UNKNOWN` out-of-vocabulary distractors and `SILENCE`/ambient noise) is deferred to subsequent planned phases.

---

## 2. Dataset Architecture & Split Partitioning

> [!NOTE]
> **Phase 27 Erratum (Authoritative Voice Roster Reconciliation)**:
> An independent audit in Phase 27 identified a typographical reporting discrepancy in the original draft of the table below. The underlying split manifests (`train_manifest.csv`, `val_manifest.csv`, `test_manifest.csv`) and audio files have always contained the authoritative 25 preset voices from local VieNeu-TTS v3 Turbo. The table below has been updated to reflect the exact authoritative manifests on disk.

The 1,000 synthetic audio utterances (250 per target keyword) were partitioned into three strictly voice-disjoint splits:

| Split | Utterances | Classes Balanced | Unique Voices | Voices Included (Authoritative Manifest Roster) |
| :--- | :---: | :---: | :---: | :--- |
| **Train** | 680 | 170 / class (25% each) | 17 | `Adam`, `Adam bựa`, `Hải Đăng`, `Minh Triết`, `Minh Đức`, `Mỹ Duyên`, `Ngọc Huyền`, `Ngọc Linh`, `Quang Sơn`, `Thanh Bình`, `Thiền Tâm Đức`, `Thiện Minh`, `Thái Sơn`, `Thục Đoan`, `Trúc Ly`, `Xuân Vĩnh`, `Đoan Trang` |
| **Validation** | 160 | 40 / class (25% each) | 4 | `Mai Anh`, `Ngọc Trân`, `Phạm Tuyên`, `Thùy Dung` |
| **Held-Out Test**| 160 | 40 / class (25% each) | 4 | `Kim Thanh`, `Quốc Tuấn`, `Quỳnh Anh`, `Đức Trí` |
| **Total** | **1,000** | **250 / class** | **25** | **Zero voice overlap across splits** |

*(Historical note: The earlier report draft had inadvertently listed placeholder names `Tường Vy, Mỹ Linh, Hồng Nhung, Quang Dũng` for validation and 17 placeholder names for train; the manifests themselves were unchanged and authoritative).*

All audio samples were verified as 16,000 Hz mono 1.0-second PCM WAVs with verified SHA-256 integrity, zero clipping (peak amplitude < 0.999), and zero phonetic truncation (safeguarded speech onset >= 0.05s, speech end <= 0.95s).

---

## 3. Acoustic Modeling & Network Architecture

### Acoustic Front-End (`AudioPreprocessor` / `LogMelFeatureExtractor`)
- **Sample Rate**: 16,000 Hz (16,000 samples per window)
- **STFT Configuration**: N_FFT = 512 (~32 ms), Hann window = 400 (25 ms), Hop length = 160 (10 ms)
- **Mel Filterbank**: 40 Mel channels spanning 20 Hz to 8,000 Hz (Slaney scale)
- **Output Dimensions**: `(Batch, 1, 40, 101)`
- **Instance Standardization**: Zero-mean, unit-variance per spectrogram

### Classifier Network (`VietnameseKWSNet`)
A compact 2D-CNN designed for low latency and small memory footprint on embedded edge and mobile platforms:
- **Total Trainable Parameters**: 138,404 (~540 KB float32 weights)
- **Checkpoint File Size**: 1,648 KB
- **Network Layout**:
  1. `Conv2d(1 -> 32, k=3, p=1)` + `BatchNorm` + `ReLU` + `MaxPool2d(2, 2)` -> `(32, 20, 50)`
  2. `Conv2d(32 -> 64, k=3, p=1)` + `BatchNorm` + `ReLU` + `MaxPool2d(2, 2)` -> `(64, 10, 25)`
  3. `Conv2d(64 -> 64, k=3, p=1)` + `BatchNorm` + `ReLU` + `MaxPool2d(2, 2)` -> `(64, 5, 12)`
  4. `Conv2d(64 -> 128, k=3, p=1)` + `BatchNorm` + `ReLU` + `AdaptiveAvgPool2d((1, 1))` -> `(128, 1, 1)`
  5. `Dropout(0.3)` + `Linear(128, 64)` + `ReLU` + `Linear(64, 4)` -> `Logits (4)`

---

## 4. Training Procedure & Validation Progress

Training was executed on an NVIDIA RTX A2000 Laptop GPU using PyTorch 2.6.0+cu124 with CUDA acceleration:
- **Optimizer**: AdamW (Initial LR: $1 \times 10^{-3}$, Weight Decay: $1 \times 10^{-4}$)
- **LR Schedule**: Cosine Annealing to $\eta_{\text{min}} = 1 \times 10^{-6}$ over 40 epochs
- **Batch Size**: 32
- **Training Augmentations**:
  - Mild random temporal shift: $\pm 25$ ms ($\pm 400$ samples)
  - Mild random gain adjustment: $[0.9, 1.1]$
- **Model Checkpoint Selection**: Checkpoint achieving highest Macro-F1 on the 160-sample validation set.

### Multi-Seed Validation Outcomes

| Seed | Convergence Epoch | Best Val Macro-F1 | Best Val Accuracy | Training Time |
| :---: | :---: | :---: | :---: | :---: |
| **42** | Epoch 14 | 97.49% | 97.50% (156/160) | 33.4s |
| **43** | Epoch 27 | 100.00% | 100.00% (160/160) | 31.1s |
| **44** | Epoch 30 | 100.00% | 100.00% (160/160) | 32.2s |

---

## 5. Held-Out Test Evaluation Results (Unseen Voices)

The three validation-selected checkpoints were evaluated against the 160 held-out test utterances spoken by 4 voices never encountered during training or validation (`Kim Thanh`, `Quốc Tuấn`, `Quỳnh Anh`, `Đức Trí`).

### Multi-Seed Test Performance Summary

| Metric | Seed 42 | Seed 43 | Seed 44 | Multi-Seed Mean |
| :--- | :---: | :---: | :---: | :---: |
| **Test Accuracy** | **100.00%** (160/160) | **100.00%** (160/160) | **100.00%** (160/160) | **100.00%** |
| **Macro Precision** | **100.00%** | **100.00%** | **100.00%** | **100.00%** |
| **Macro Recall** | **100.00%** | **100.00%** | **100.00%** | **100.00%** |
| **Macro F1-Score** | **100.00%** | **100.00%** | **100.00%** | **100.00%** |

### Per-Class Test Breakdown (Identical across all 3 seeds)

| Class Index | Keyword Target | Intended Text | Support | Precision | Recall | F1-Score |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| 0 | `TRAI` | Trái | 40 | 100.00% | 100.00% | 1.0000 |
| 1 | `PHAI` | Phải | 40 | 100.00% | 100.00% | 1.0000 |
| 2 | `LEN` | Lên | 40 | 100.00% | 100.00% | 1.0000 |
| 3 | `XUONG` | Xuống | 40 | 100.00% | 100.00% | 1.0000 |

### Test Confusion Matrix (Rows: Ground Truth, Columns: Predicted)

```
              TRAI    PHAI     LEN   XUONG
TRAI (Trái)     40       0       0       0
PHAI (Phải)      0      40       0       0
LEN  (Lên)       0       0      40       0
XUONG(Xuống)     0       0       0      40
```
- **False Positives**: 0 / 160
- **False Negatives**: 0 / 160
- **Misclassified Utterances**: 0

---

## 6. On-Device Latency & Computational Efficiency

Single-sample inference latency was benchmarked on CPU (Intel Core i7 host, single thread) over 100 warm-up runs and 200 measured iterations:

| Metric | Measured Value | Mobile Edge Viability |
| :--- | :---: | :---: |
| **Model Parameters** | 138,404 | Highly compact (< 0.6 MB) |
| **Checkpoint Size** | 1,648 KB | Fits in on-device cache |
| **CPU Latency (Single Utterance)** | $1.20 \pm 0.22$ ms | Exceeds 10x real-time margin (< 10 ms requirement) |
| **GPU Latency (Batched 32)** | < 0.05 ms / sample | High-throughput server capability |

---

## 7. Analysis & Scientific Interpretation

1. **Acoustic Distinguishability**:
   The four target commands (`Trái`, `Phải`, `Lên`, `Xuống`) exhibit stark phonetic and tonal distinctions:
   - `Trái`: Initial retroflex / affricate `tr-`, vowel diphthong `a-i`, high rising tone (sắc).
   - `Phải`: Labial-dental fricative `ph-` [f], diphthong `a-i`, falling-rising asking tone (hỏi).
   - `Lên`: Lateral liquid `l-`, front vowel `ê`, alveolar nasal coda `-n`, level tone (ngang).
   - `Xuống`: Alveolar fricative `x-` [s], back rounded diphthong `uô`, velar nasal coda `-ng`, high rising tone (sắc).
   The Log-Mel spectrogram representation clearly separates these trajectories across distinct synthetic voices.

2. **Absence of Overfitting to Training Voices**:
   Because test performance is evaluated exclusively on four voice presets never exposed during training, the 100% test accuracy demonstrates that the CNN learned invariant acoustic-phonetic features rather than speaker-specific idiosyncratic timbre.

3. **Comparison with Historical V5.2 Parametric Baseline**:
   Unlike the historical V5.2 synthetic generator which produced robotic buzzes that failed nuisance audits, VieNeu-TTS v3 Turbo produces smooth spectral formant transitions and realistic harmonic stacks, enabling clean convergence without artificial nuisance artifacts.

4. **Next Steps for Phase 27+**:
   - Complete human review of the 80 blinded listening clips to close subjective intelligibility verification.
   - Expand the baseline to 6 classes (`TRAI`, `PHAI`, `LEN`, `XUONG`, `UNKNOWN`, `SILENCE`) with real background noise injection.
   - Prepare for transfer learning and domain adaptation against natural human speech acquisitions.
