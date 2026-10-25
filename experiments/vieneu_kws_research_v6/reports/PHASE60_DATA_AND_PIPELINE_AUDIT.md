# Phase 60 — Comprehensive Data and Pipeline Audit Report

**Workspace:** `D:\trainidentificationaudio`  
**Experiment Directory:** `experiments/vieneu_kws_research_v6/`  
**Report Artifact:** `experiments/vieneu_kws_research_v6/reports/PHASE60_DATA_AND_PIPELINE_AUDIT.md`  
**Status Banner:** `INDEPENDENT_AUDIT_VERIFIED` | `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Executive Summary and Scope

As part of the Integrated Research Campaign (Phases 60–64), this audit provides a comprehensive, independent verification of the historical data lineage, acoustic pipelines, model architectures, and experimental governance spanning Phases 26 through 59.

The primary purpose is to establish an uncompromised scientific foundation prior to synthesizing Benchmark Dataset v6 (Phase 61) and conducting controlled bounded-utterance improvements (Phases 62–64).

### Key Audit Findings:
1. **Historical Dataset Lineage Integrity:**
   - Every historical dataset (`synthetic_kws_dataset_vieneu_1000/`, `6class/`, `6class_v2/`, `vieneu_kws_research_v2/` through `v5/`) remains byte-exact and unmodified in its dedicated frozen directory.
   - Zero historical test files or texts have been leaked into training splits.
2. **TTS Model and Voice Allocation Governance:**
   - All synthetic audio across the campaign was generated strictly using approved VieNeu-TTS pilot models within `.venv_vieneu`.
   - The canonical 25-voice speaker partition has been preserved: 17 training voices, 4 validation voices, and 4 strictly held-out test voices (`Kim Thanh`, `Quốc Tuấn`, `Quỳnh Anh`, `Đức Trí`). No held-out voice has ever appeared in training or validation splits.
3. **Pipeline Structural Distinctions:**
   - **Stage 1 CNN (`VietnameseKWSNet`):** 138,534 parameters, 6-class output (`TRAI`, `PHAI`, `LEN`, `XUONG`, `UNKNOWN`, `SILENCE`). High raw recall ($96.04\%$), but elevated non-target false alarms ($18.39\%$ FAR).
   - **Cascade v1 (`VietnameseKWSNet` + `TemporalKWSVerifier`):** 479,307 combined parameters. Stage 2 TCN applies temporal verification over multi-frame feature activations, rejecting $95.36\%$ of non-target distractors (FAR $4.64\%$) while achieving $92.29\%$ command recall with zero cross-command confusion.
4. **Strict Environmental and Ethical Separation:**
   - `data/raw/field_recordings/` and consent ledgers have remained untouched, uninspected, and isolated from the synthetic workspace.
   - All evaluations retain mandatory disclaimers: `PENDING_HUMAN_LISTENER_VERIFICATION` and `NOT_VALIDATED_ON_NATURAL_SPEECH`.

---

## 2. Historical Dataset Lineage and Evolution Audit

| Dataset Name | Phase Introduced | Total Clips | Command Distribution | Non-Target / UNKNOWN Distribution | Silence Clips | Speaker Voice Splits (Train / Val / Test) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `synthetic_kws_dataset_vieneu_1000` | Phase 26 | 1,000 | 4 classes (100 each = 400) | 400 random Wikipedia words | 200 synthetic | 15 / 5 / 5 | **FROZEN & PRESERVED** |
| `synthetic_kws_dataset_vieneu_6class` | Phase 29 | 1,200 | 4 classes (150 each = 600) | 400 generic UNKNOWN | 200 synthetic | 17 / 4 / 4 | **FROZEN & PRESERVED** |
| `synthetic_kws_dataset_vieneu_6class_v2` | Phase 33 | 1,800 | 4 classes (200 each = 800) | 600 phonetically targeted UNKNOWN | 400 synthetic | 17 / 4 / 4 | **FROZEN & PRESERVED** |
| `vieneu_kws_research_v2` | Phase 40 | 2,000 | 4 classes (200 each = 800) | 800 hard negatives (minimal pairs) | 400 synthetic | 17 / 4 / 4 | **FROZEN & PRESERVED** |
| `vieneu_kws_research_v3` | Phase 45 | 2,200 | 4 classes (220 each = 880) | 920 hard negatives + continuous | 400 synthetic | 17 / 4 / 4 | **FROZEN & PRESERVED** |
| `vieneu_kws_research_v4` | Phase 50 | 2,400 | 4 classes (240 each = 960) | 940 bounded minimal pairs | 500 synthetic | 17 / 4 / 4 | **FROZEN & PRESERVED** |
| `vieneu_kws_research_v5` | Phase 56 | 2,440 | 4 classes (250 each = 1,000) | 940 disjoint minimal pairs / phrases | 500 synthetic | 17 / 4 / 4 | **FROZEN & PRESERVED** |

### Audit Verifications:
- **No In-Place Overwrites:** Each dataset version is housed in an independent directory with immutable audio files and SHA-256 manifests.
- **Phonetic Complexity Growth:** Progressed from naive vocabulary distractors (v1) to tonal minimal pairs (v2), consonant onsets/codas (v3), phrase embeddings (v4), and fully disjoint linguistic taxonomies (v5).

---

## 3. Acoustic Preprocessing and Feature Extraction Audit

Across all baseline models and cascades, preprocessing is strictly standardized:

```
Raw Audio Input (16,000 Hz Mono PCM, 1.000s = 16,000 samples)
  │
  ├── Peak Normalization: -1.0 dBFS target
  ├── Energy Centering: Active speech centered in 1.0s window with >= 150ms border padding
  │
  ▼
STFT Spectrogram
  ├── Window Length: 400 samples (25.0 ms), Hann window
  ├── Hop Length: 160 samples (10.0 ms)
  ├── FFT Length: 512
  │
  ▼
Log-Mel Filterbank Extraction
  ├── Number of Mel Bands: 40
  ├── Frequency Range: 80 Hz to 7,600 Hz
  ├── Output Dimensions: [Batch, 1, 40 Mel Bands, 101 Time Frames]
```

### Preprocessing Determinism Audit:
- Both Stage 1 and Stage 2 feature extractors operate deterministically without randomized augmentation during validation and test phases.
- Training augmentations (where enabled) are confined strictly to training folds: time-shifting ($\pm 100\,\text{ms}$), additive Gaussian / pink noise ($\text{SNR} \in [10, 25]\,\text{dB}$), and gain variations ($\pm 3\,\text{dB}$).

---

## 4. Model Architectures and Decision Logic Audit

### Candidate 1: Standalone Stage 1 CNN (`VietnameseKWSNet`)
- **Backbone Architecture:** 4 convolutional blocks (Conv2d $3\times 3$, BatchNorm, ReLU, MaxPool2d $2\times 2$, Dropout 0.2) followed by AdaptiveAvgPool2d and 2 Fully Connected layers.
- **Parameters:** 138,534.
- **Output Classes:** 6 (`0: TRAI`, `1: PHAI`, `2: LEN`, `3: XUONG`, `4: UNKNOWN`, `5: SILENCE`).
- **Decision Logic:** Softmax probability thresholding. Command predicted if $P(c) \ge \tau$ and $c = \operatorname{argmax}(P)$; otherwise classified as `REJECT`.

### Candidate 2: Two-Stage Cascade v1
- **Stage 1 (`VietnameseKWSNet`):** Initial fast screening on 1.0s Log-Mel. If $\max_{c \in \text{commands}} P_1(c) < \tau_1$, early exit to `REJECT`.
- **Stage 2 (`TemporalKWSVerifier`):** Dilated Temporal Convolutional Network (TCN) with receptive field covering multi-frame feature sequences (340,773 parameters).
- **Combined Parameters:** 479,307 parameters.
- **Verification Logic:**
  - Evaluates temporal consistency of keyword phonemes.
  - Generates 5-class posterior: `[TRAI, PHAI, LEN, XUONG, REJECT]`.
  - Command accepted if $P_2(c) \ge \tau_2$ and $c = \operatorname{argmax}(P_2)$; otherwise rejected.
- **Acoustic Rejection Purity:** In Phase 55–59, Cascade v1 achieved **zero cross-command confusion** across all 2,160 evaluations (3 seeds $\times$ 720 clips).

---

## 5. Validation-Only Calibration Methodology Audit

To prevent data snooping and optimistic bias, the campaign enforces strict separation between calibration and evaluation:

1. **Parameter Freezing:** All neural network weights ($\theta$) are frozen after training convergence.
2. **Calibration Dataset:** Operating thresholds ($\tau_1, \tau_2$) are searched and calibrated exclusively on `data/splits/val_manifest.csv` ($N = 360$ clips).
3. **Predeclared Optimization Objective:**
   - Minimize non-target FAR subject to per-keyword target recall $\ge 95\%$ for `TRAI`, `PHAI`, `LEN`, and `XUONG`.
   - If no threshold satisfies $\ge 95\%$ recall across all 4 keywords, select the operating point that maximizes macro recall while maintaining $\text{FAR} \le 5.0\%$, and formally log `NO_FEASIBLE_VALIDATION_OPERATING_POINT`.
4. **Locked Test Evaluation:** Calibrated thresholds are saved to `calibration/selected_configs.json` and applied to `data/splits/test_manifest.csv` without adjustment.

---

## 6. Scientific Governance and Integrity Checklist

| Compliance Item | Requirement | Audit Status | Evidence / Notes |
| :--- | :--- | :--- | :--- |
| **Field Recording Separation** | Never access or modify `data/raw/field_recordings/` | **COMPLIANT** | Zero read/write handles in any script. |
| **Consent Ledger Isolation** | Never read or modify speaker consent ledgers | **COMPLIANT** | Consent infrastructure untouched. |
| **Test Set Integrity** | No recycling of test audio or texts | **COMPLIANT** | Test sets are freshly generated and locked. |
| **Voice Split Discipline** | 4 held-out voices never in train/val | **COMPLIANT** | Voice disjointness strictly verified. |
| **Git Staging Cleanliness** | No staged commits (`git diff --cached` empty) | **COMPLIANT** | Staging index is 100% clean. |
| **Status Transparency** | Disclaim synthetic origin on all reports | **COMPLIANT** | Explicit status banners across all docs. |

---

## 7. Audit Conclusion and Authorization for Phase 61

The data lineage, preprocessing pipelines, model configurations, and evaluation methodologies have been audited and verified. All preconditions are met to proceed with Phase 61: generating the fresh Benchmark Dataset v6 with 200+ brand new disjoint texts and verified acoustic partitions.
