# Cross-Generator Model Evaluation and Metric Lineage Audit

> [!IMPORTANT]
> **GOVERNANCE INVARIANT: HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED.**
> All evaluations conducted offline on frozen checkpoints and existing synthetic manifests. Checkpoint `arm_b_onset_no_resample.pt` remains frozen without modification.

---

## 1. Executive Summary & Verification Methodology

Every metric reported in this audit was independently recomputed directly from the 6,768 verified row-level prediction entries in [`canonical_predictions.csv`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_v4/audit/canonical_predictions.csv). No metric was taken on faith from summary files.

### Evaluated Model Candidate:
- **Model Arm**: `Arm B (Onset-Aware, No Resample)`
- **Architecture**: TC-ResNet-14 ($272,934$ parameters, 40 Log-Mel channels)
- **Checkpoint Path**: [`experiments/synthetic_kws_v4/checkpoints/arm_b_onset_no_resample.pt`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_v4/checkpoints/arm_b_onset_no_resample.pt)
- **Verified SHA-256**: `6A0FD55CD6F16F5B683951127076EEC72506C31EE670F896AAEACE5981CF1D96`
- **Verification Status**: **`VERIFIED_FROM_ROW_PREDICTIONS`**

---

## 2. Cross-Generator Evaluation Matrix (Arm B)

| Partition ID | Generator Recipe | Evaluation Manifest Path | Manifest SHA-256 | Sample Count | Overall Accuracy | Macro-F1 | `TRAI` Recall | `PHAI` Recall | `LEN` Recall | `XUONG` Recall | `UNKNOWN` Recall | `SILENCE` Recall |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`v4_fresh_300`** | V4 Onset-Aware | `experiments/synthetic_kws_v4/fresh_holdout_manifest_6class.csv` | `E540A32F294283FE...` | 300 | **99.33%** | **0.9933** | **96.0%** (48/50) | 100.0% (50/50) | 100.0% (50/50) | 100.0% (50/50) | 100.0% (50/50) | 100.0% (50/50) |
| **`val_onset_216`** | V4 Onset-Aware | `experiments/synthetic_kws_v4/val_manifest_onset_aware.csv` | `0C5E03AFA01AA482...` | 216 | **100.0%** | **1.0000** | **100.0%** (36/36) | 100.0% (36/36) | 100.0% (36/36) | 100.0% (36/36) | 100.0% (36/36) | 100.0% (36/36) |
| **`v3_fresh_300`** | V3 Control | `experiments/synthetic_kws_v3/fresh_holdout_manifest_6class.csv` | `DD8D1B6F582CF735...` | 300 | **83.33%** | **0.7778** | **0.0% (0/50)** | 100.0% (50/50) | 100.0% (50/50) | 100.0% (50/50) | 100.0% (50/50) | 100.0% (50/50) |
| **`v2_fresh_300`** | V2 Baseline | `experiments/synthetic_kws_v2/fresh_holdout_manifest_6class.csv` | `FD4890534BDCC7AF...` | 300 | **83.33%** | **0.7778** | **0.0% (0/50)** | 100.0% (50/50) | 100.0% (50/50) | 100.0% (50/50) | 100.0% (50/50) | 100.0% (50/50) |
| **`hist_216`** | V1 Baseline | `experiments/synthetic_kws_v2/test_manifest_6class.csv` | `030ECFCF40C44AE9...` | 216 | **69.44%** | **0.6097** | **0.0% (0/36)** | 100.0% (36/36) | 100.0% (36/36) | **16.7% (6/36)** | 100.0% (36/36) | 100.0% (36/36) |

---

## 3. Full Confusion Matrices Recomputed from Row Predictions

### 1. V4 Fresh Holdout (`v4_fresh_300`, 300 clips)
```
True \ Pred:       TRAI     PHAI      LEN    XUONG  UNKNOWN  SILENCE
TRAI                 48        0        0        0        2        0
PHAI                  0       50        0        0        0        0
LEN                   0        0       50        0        0        0
XUONG                 0        0        0       50        0        0
UNKNOWN               0        0        0        0       50        0
SILENCE               0        0        0        0        0       50
```

### 2. V3 Fresh Holdout (`v3_fresh_300`, 300 clips)
```
True \ Pred:       TRAI     PHAI      LEN    XUONG  UNKNOWN  SILENCE
TRAI                  0       50        0        0        0        0  <-- 100% MISCLASSIFIED AS PHAI
PHAI                  0       50        0        0        0        0
LEN                   0        0       50        0        0        0
XUONG                 0        0        0       50        0        0
UNKNOWN               0        0        0        0       50        0
SILENCE               0        0        0        0        0       50
```

### 3. V2 Fresh Holdout (`v2_fresh_300`, 300 clips)
```
True \ Pred:       TRAI     PHAI      LEN    XUONG  UNKNOWN  SILENCE
TRAI                  0       50        0        0        0        0  <-- 100% MISCLASSIFIED AS PHAI
PHAI                  0       50        0        0        0        0
LEN                   0        0       50        0        0        0
XUONG                 0        0        0       50        0        0
UNKNOWN               0        0        0        0       50        0
SILENCE               0        0        0        0        0       50
```

### 4. Historical Test Holdout (`hist_216`, 216 clips)
```
True \ Pred:       TRAI     PHAI      LEN    XUONG  UNKNOWN  SILENCE
TRAI                  0       36        0        0        0        0  <-- 100% MISCLASSIFIED AS PHAI
PHAI                  0       36        0        0        0        0
LEN                   0        0       36        0        0        0
XUONG                 0        0       30        6        0        0  <-- 83.3% MISCLASSIFIED AS LEN
UNKNOWN               0        0        0        0       36        0
SILENCE               0        0        0        0        0       36
```

---

## 4. Scientific Root Cause Analysis

1. **Total Collapse on `TRAI`**:
   - Across V1, V2, and V3 holdouts, **136 out of 136 `TRAI` utterances were misclassified as `PHAI`** (exact $0.0\%$ recall).
   - In V4, `TRAI` was given a 25ms stop-closure silence gap. Arm B learned to associate early energy in the rhyme /-a:j/ with `PHAI`. When presented with earlier generator outputs where `TRAI` starts at $t=0$, the network is blinded by the lack of closure gap and predicts `PHAI` with $>90\%$ confidence.
2. **Collapse on `XUONG` in V1**:
   - In V1, 30 out of 36 `XUONG` clips were misclassified as `LEN`.
   - V4 added strong highpass sibilant noise ($>4000\text{ Hz}$) to `XUONG`. In V1, `XUONG` lacked this noise burst, causing the model to confuse its vowel formants with `LEN`.
3. **Control Model Contrast (Arm A)**:
   - Arm A (trained without V4 onset modifications) achieved $66.7\%$ to $86.0\%$ recall on `TRAI` across V1–V3 holdouts, but degraded on V4.
   - This demonstrates that both models learned generator-dependent acoustic shortcuts rather than robust phonetic representations.

---

## 5. Model Validity Boundaries

- **Exact Duplicate Leakage**: **`ZERO_LEAKAGE_DETECTED`** across all partitions.
- **Generator Dependence**: **`HIGH_GENERATOR_DEPENDENCE`**. The candidate model is tightly coupled to V4 onset parameterization.
- **Model Robustness within Synthetic Scope**: **`COLLAPSES_ACROSS_GENERATOR_VERSIONS`**. The model fails to generalize across synthetic generators ($0.0\%$ recall on `TRAI` on V1–V3 holdouts).
- **Natural Human Speech Validity**: **`NOT_VALIDATED_ON_NATURAL_SPEECH`**. Because human speech collection and ingestion are blocked by governance, empirical validation on natural speech has not occurred. The model cannot be declared valid or invalid on natural human speech without authorized empirical testing.
