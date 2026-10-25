# Phase 29: Six-Class Vietnamese KWS CNN Baseline Training & Evaluation Report

**Date:** 2026-10-18  
**Workspace:** `D:\trainidentificationaudio`  
**Dataset:** `experiments/synthetic_kws_dataset_vieneu_6class/`  
**Architecture:** `VietnameseKWSNet` (Compact 2D-CNN, 138,534 parameters)  
**Input Representation:** 40-channel Log-Mel Spectrogram `(1, 40, 101)` at 16,000 Hz  
**Classification Space:** `TRAI` (0), `PHAI` (1), `LEN` (2), `XUONG` (3), `UNKNOWN` (4), `SILENCE` (5)  

---

## 1. Executive Summary & Audit Decision

### Audit Decision: **PASS WITH LIMITATIONS**

In Phase 29, the first six-class convolutional neural network (CNN) keyword-spotting baseline was trained and evaluated using the accepted balanced synthetic dataset (`experiments/synthetic_kws_dataset_vieneu_6class/`).

Training and evaluation were conducted across three fixed random seeds (`42`, `43`, `44`) with validation-driven checkpoint selection (`val_macro_f1`).

### Summary of Key Results:
- **Validation Macro F1 (3-seed mean)**: **$90.16\%$** (Seed 42: 89.60%, Seed 43: 90.21%, Seed 44: 90.66%)
- **Test Accuracy (3-seed mean)**: **$92.08\%$** (Seed 42: 92.08%, Seed 43: 92.92%, Seed 44: 91.25%)
- **Test Macro F1 (3-seed mean)**: **$91.36\%$** (Seed 42: 91.27%, Seed 43: 92.51%, Seed 44: 90.29%)
- **SILENCE Discrimination (Out-of-Condition)**: **$100.00\%$** precision, **$100.00\%$** recall across all three seeds.
- **Inference Latency**: **$1.32 \pm 0.15\text{ ms}$** per single 1.0s utterance on host CPU.

### Scientific Governance & Limitations:
> [!IMPORTANT]
> 1. **Subjective Human Intelligibility Status**: **`PENDING_HUMAN_LISTENER_VERIFICATION`**. High acoustic classification scores do not substitute for perceptual naturalness. The 80-clip blinded human listening review (Phase 25) remains pending.
> 2. **Operational Validity**: **`NOT_VALIDATED_ON_NATURAL_SPEECH`**. This experiment demonstrates in-distribution synthetic discriminability across unseen synthesized voices and disjoint out-of-vocabulary distractors. It does not validate performance on natural human speech recordings across real acoustic environments.
> 3. **SILENCE Out-of-Condition Note**: SILENCE test clips were generated from acoustic condition families (Brownian ambient tone, comb-filtered room resonance, high-pass shaped dither) disjoint from training conditions.
> 4. **Four-Class Sample Reuse**: The four target commands reuse the previously evaluated synthetic samples. This is a new 6-class task, not an independent re-sampling of those four classes.

---

## 2. Preflight Audit Verification

A rigorous preflight audit was executed prior to training:
- **Master Manifest**: 1,500 total rows; exactly 250 rows per class across all 6 classes.
- **Split Distribution**:
  - Train: 1,020 rows (170 per class, 17 distinct voices).
  - Validation: 240 rows (40 per class, 4 distinct voices).
  - Test: 240 rows (40 per class, 4 distinct voices).
- **Split Isolation**:
  - $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$ (0 sample ID overlap).
  - All 1,500 KWS WAV hashes are unique with 0 cross-split hash collisions.
- **Audio Integrity**: 100% of audio files exist, decode without error, and have exactly 16,000 samples at 16,000 Hz mono with zero clipping.

---

## 3. Acoustic Modeling & Network Architecture

### Acoustic Front-End (`LogMelFeatureExtractor`)
- **Sampling Rate**: 16,000 Hz
- **STFT Configuration**: $N_{\text{FFT}} = 512$ ($32\text{ ms}$), Hann window $= 400$ ($25\text{ ms}$), Hop length $= 160$ ($10\text{ ms}$)
- **Filterbank**: 40 Mel channels spanning $20\text{ Hz}$ to $8,000\text{ Hz}$ (Slaney scale)
- **Feature Tensor Dimensions**: $(B, 1, 40, 101)$
- **Normalization**: Static mathematical extraction; zero dataset statistics fitted across splits.

### Six-Class Network (`VietnameseKWSNet`)
- **Total Parameters**: 138,534 trainable parameters (~541 KB float32 weights, 1.65 MB checkpoint).
- **Structure**:
  1. `ConvBlock 1`: Conv2d(1 -> 32, k=3, p=1) + BatchNorm + ReLU + MaxPool(2, 2) -> (32, 20, 50)
  2. `ConvBlock 2`: Conv2d(32 -> 64, k=3, p=1) + BatchNorm + ReLU + MaxPool(2, 2) -> (64, 10, 25)
  3. `ConvBlock 3`: Conv2d(64 -> 64, k=3, p=1) + BatchNorm + ReLU + MaxPool(2, 2) -> (64, 5, 12)
  4. `ConvBlock 4`: Conv2d(64 -> 128, k=3, p=1) + BatchNorm + ReLU + AdaptiveAvgPool((1, 1)) -> (128, 1, 1)
  5. `Classifier Head`: Dropout(0.3) -> Linear(128, 64) -> ReLU -> Linear(64, 6) -> Logits (6)

---

## 4. Multi-Seed Training & Validation Results

Training was executed on an NVIDIA RTX A2000 Laptop GPU using AdamW ($\text{lr}=10^{-3}$, weight decay $=10^{-4}$) and Cosine Annealing scheduler (40 epochs) with training-only augmentations ($\pm 25\text{ ms}$ shift, $[0.9, 1.1]$ gain). Checkpoints were selected strictly by peak `val_macro_f1`.

| Seed | Best Epoch | Peak Val Macro-F1 | Peak Val Accuracy | Training Time | Checkpoint File |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **42** | 27 | **89.60%** | 90.00% (216/240) | 48.9s | [`best_checkpoint_seed42.pt`](file:///d:/trainidentificationaudio/experiments/vieneu_cnn_6class/checkpoints/best_checkpoint_seed42.pt) |
| **43** | 22 | **90.21%** | 90.83% (218/240) | 38.1s | [`best_checkpoint_seed43.pt`](file:///d:/trainidentificationaudio/experiments/vieneu_cnn_6class/checkpoints/best_checkpoint_seed43.pt) |
| **44** | 26 | **90.66%** | 91.25% (219/240) | 52.2s | [`best_checkpoint_seed44.pt`](file:///d:/trainidentificationaudio/experiments/vieneu_cnn_6class/checkpoints/best_checkpoint_seed44.pt) |
| **Mean**| — | **90.16%** | **90.69%** | **46.4s** | — |

*Training observation: Unlike the 4-class problem where validation rapidly reached 97–100%, the 6-class task with 30 disjoint UNKNOWN words and 11 disjoint SILENCE conditions converged to ~90–91%, demonstrating non-trivial acoustic complexity.*

---

## 5. Held-Out Six-Class Test Set Evaluation

The validation-selected checkpoints were evaluated against the 240 held-out test utterances (40 per class across 4 unseen voices and unseen conditions):

### 5.1 Multi-Seed Test Performance Summary

| Metric | Seed 42 | Seed 43 | Seed 44 | Multi-Seed Mean |
| :--- | :---: | :---: | :---: | :---: |
| **Test Accuracy** | **92.08%** (221/240) | **92.92%** (223/240) | **91.25%** (219/240) | **92.08%** |
| **Macro Precision** | **92.72%** | **94.06%** | **93.32%** | **93.37%** |
| **Macro Recall** | **92.08%** | **92.92%** | **91.25%** | **92.08%** |
| **Macro F1-Score** | **91.27%** | **92.51%** | **90.29%** | **91.36%** |
| **CPU Latency (single)** | $1.53 \pm 0.15\text{ ms}$ | $1.27 \pm 0.04\text{ ms}$ | $1.17 \pm 0.09\text{ ms}$ | **$1.32\text{ ms}$** |

### 5.2 Per-Class Performance Breakdown (Multi-Seed Mean)

| Class Index | Label | Intended Semantics | Support | Precision (Mean) | Recall (Mean) | F1-Score (Mean) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| `0` | `TRAI` | "Trái" | 40 | 96.7% | 97.5% | 0.971 |
| `1` | `PHAI` | "Phải" | 40 | 86.7% | 100.0% | 0.928 |
| `2` | `LEN` | "Lên" | 40 | 93.8% | 100.0% | 0.968 |
| `3` | `XUONG` | "Xuống" | 40 | 84.3% | 100.0% | 0.911 |
| `4` | `UNKNOWN`| 10 unseen test words | 40 | 98.6% | **55.0%** | **0.704** |
| `5` | `SILENCE`| 3 unseen test conditions | 40 | **100.0%** | **100.0%** | **1.000** |

### 5.3 Six-Class Confusion Matrix (Seed 43 Example, 240 Samples)

```
              TRAI    PHAI     LEN   XUONG UNKNOWN SILENCE
TRAI            38       2       0       0       0       0
PHAI             0      40       0       0       0       0
LEN              0       0      40       0       0       0
XUONG            0       0       0      40       0       0
UNKNOWN          2       9       4       0      25       0
SILENCE          0       0       0       0       0      40
```

---

## 6. Diagnostic & Error Analysis

### 6.1 SILENCE Out-of-Condition Diagnostics
The held-out test split evaluated three condition groups never seen during training or validation:
- **Condition 1: Brownian ambient noise ($1/f^2$)**: $15/15$ correct ($100.0\%$) across all seeds.
- **Condition 2: Comb-filtered room resonance ($20\text{ ms}$ delay)**: $15/15$ correct ($100.0\%$) across all seeds.
- **Condition 3: High-pass shaped dither ($2,000\text{ Hz}$ HPF)**: $10/10$ correct ($100.0\%$) across all seeds.
- **Finding**: Zero false alarms from silence into target commands (0 SILENCE misclassified as speech), and zero target commands misclassified as silence. The energy and spectral profile of non-speech ambience is cleanly decoupled from speech formants.

### 6.2 UNKNOWN Out-of-Vocabulary Confusion Analysis
The primary source of error in the 6-class system is **out-of-vocabulary UNKNOWN speech falsely triggering as target commands** (15 to 21 instances per seed):
1. **Phonetic Rhyme & Coda Overlap**:
   - `Không` (/xəwŋm/) was frequently misclassified as `XUONG` (/suəŋ˧˥/) due to identical back-diphthong and velar nasal coda `[-uông / -ông]`.
   - `Phóng` (/fawŋm/) triggered `PHAI` or `XUONG` due to initial labiodental fricative `/ph-/` and coda `[-ong]`.
   - `Lưu` (/liw/) triggered `LEN` due to initial lateral liquid `/l-/`.
   - `Hủy` (/hwi/) triggered `PHAI` due to the dipping hỏi tone.
2. **Asymmetry in Rejection**:
   - Target commands are almost never rejected as UNKNOWN (target recall: $97.5\%$ to $100.0\%$).
   - UNKNOWN recall is lower ($47.5\%$ to $62.5\%$) because a 10-word training inventory is too narrow to span all unconstrained Vietnamese phoneme combinations.
3. **Implication**: For production robustness, `UNKNOWN` training vocabulary should be expanded from 10 words to 100+ open-vocabulary words, or an explicit rejection threshold on softmax entropy should be tuned on validation data.

---

## 7. Artifacts & Deliverables

All deliverables have been created and validated:
1. **Configuration**: [`experiments/vieneu_cnn_6class/config.json`](file:///d:/trainidentificationaudio/experiments/vieneu_cnn_6class/config.json)
2. **Training Pipeline**: [`experiments/vieneu_cnn_6class/train.py`](file:///d:/trainidentificationaudio/experiments/vieneu_cnn_6class/train.py)
3. **Evaluation Pipeline**: [`experiments/vieneu_cnn_6class/evaluate.py`](file:///d:/trainidentificationaudio/experiments/vieneu_cnn_6class/evaluate.py)
4. **Documentation**: [`experiments/vieneu_cnn_6class/README.md`](file:///d:/trainidentificationaudio/experiments/vieneu_cnn_6class/README.md)
5. **Checkpoints**:
   - Seed 42: [`best_checkpoint_seed42.pt`](file:///d:/trainidentificationaudio/experiments/vieneu_cnn_6class/checkpoints/best_checkpoint_seed42.pt)
   - Seed 43: [`best_checkpoint_seed43.pt`](file:///d:/trainidentificationaudio/experiments/vieneu_cnn_6class/checkpoints/best_checkpoint_seed43.pt)
   - Seed 44: [`best_checkpoint_seed44.pt`](file:///d:/trainidentificationaudio/experiments/vieneu_cnn_6class/checkpoints/best_checkpoint_seed44.pt)
6. **Training Histories**:
   - [`history_seed42.json`](file:///d:/trainidentificationaudio/experiments/vieneu_cnn_6class/logs/history_seed42.json), [`history_seed43.json`](file:///d:/trainidentificationaudio/experiments/vieneu_cnn_6class/logs/history_seed43.json), [`history_seed44.json`](file:///d:/trainidentificationaudio/experiments/vieneu_cnn_6class/logs/history_seed44.json)
7. **Per-Sample Predictions & Metrics**:
   - [`test_predictions_seed42.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_cnn_6class/predictions/test_predictions_seed42.csv), [`test_metrics_seed42.json`](file:///d:/trainidentificationaudio/experiments/vieneu_cnn_6class/predictions/test_metrics_seed42.json)
   - [`test_predictions_seed43.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_cnn_6class/predictions/test_predictions_seed43.csv), [`test_metrics_seed43.json`](file:///d:/trainidentificationaudio/experiments/vieneu_cnn_6class/predictions/test_metrics_seed43.json)
   - [`test_predictions_seed44.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_cnn_6class/predictions/test_predictions_seed44.csv), [`test_metrics_seed44.json`](file:///d:/trainidentificationaudio/experiments/vieneu_cnn_6class/predictions/test_metrics_seed44.json)
8. **Automated Unit Tests**: [`tests/test_vieneu_cnn_6class.py`](file:///d:/trainidentificationaudio/tests/test_vieneu_cnn_6class.py) (7/7 passed).
9. **Full Test Suite Status**: 198/198 passed.

---

## 8. Recommended Next Steps

1. **Phase 30 (Controlled Noise Robustness Benchmark)**: Evaluate the clean-trained six-class checkpoints against test audio corrupted by controlled additive noise (white, pink, HVAC, babble) across $20\text{ dB}$, $15\text{ dB}$, $10\text{ dB}$, $5\text{ dB}$, and $0\text{ dB}$ SNR levels to measure acoustic degradation curves.
2. **Phase 25 Human Review**: Complete the 80-clip blinded listening review sheet to resolve the `PENDING_HUMAN_LISTENER_VERIFICATION` status before deployment.
3. **Rejection Thresholding**: Calibrate an open-set confidence threshold on validation data to suppress UNKNOWN false alarms before testing on real-world natural recordings.
