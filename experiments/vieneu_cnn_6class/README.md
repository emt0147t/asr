# VieNeu-TTS Six-Class CNN Keyword Spotting Baseline

## 1. Overview

This directory contains the first six-class convolutional neural network (CNN) keyword-spotting baseline trained on the verified 1,500-utterance synthetic Vietnamese dataset (`experiments/synthetic_kws_dataset_vieneu_6class/`).

Target classification space:
- `0`: `TRAI` (Trái) — Directional command
- `1`: `PHAI` (Phải) — Directional command
- `2`: `LEN` (Lên) — Directional command
- `3`: `XUONG` (Xuống) — Directional command
- `4`: `UNKNOWN` — Non-target Vietnamese speech (30 disjoint out-of-vocabulary words)
- `5`: `SILENCE` — Non-command audio (11 disjoint acoustic condition groups)

> [!IMPORTANT]
> **Strict Governance Mandates**:
> - **Subjective Human Intelligibility Status**: **`PENDING_HUMAN_LISTENER_VERIFICATION`** (Phase 25 blinded review pending).
> - **Operational Real-World Validity**: **`NOT_VALIDATED_ON_NATURAL_SPEECH`**.
> - **SILENCE Out-of-Condition Note**: SILENCE test clips are drawn from distinct condition templates (Brownian noise, comb-filtered room resonance, high-pass shaped dither) not seen during training.

---

## 2. Model Architecture & Acoustic Front-End

### Acoustic Front-End (`AudioPreprocessor` / `LogMelFeatureExtractor`)
- **Sampling Rate**: 16,000 Hz, mono PCM WAV (1.0 second = 16,000 samples)
- **STFT Configuration**:
  - $N_{\text{FFT}} = 512$ (~$32\text{ ms}$)
  - Window Length $= 400$ ($25\text{ ms}$ Hann window)
  - Hop Length $= 160$ ($10\text{ ms}$ hop)
  - Mel Filterbank: 40 Mel channels spanning $20\text{ Hz}$ to $8,000\text{ Hz}$ (Slaney scale)
  - Output tensor shape: $(B, 1, 40, 101)$

### Classifier Network (`VietnameseKWSNet`)
- **Trainable Parameters**: 138,534 (~541 KB float32 weights, 1.65 MB checkpoint)
- **Structure**: 4 Conv blocks with BatchNorm, ReLU, and MaxPool $\rightarrow$ Adaptive Average Pooling $\rightarrow$ Dropout ($p=0.3$) $\rightarrow$ Linear(128, 64) $\rightarrow$ Linear(64, 6)

---

## 3. Dataset Splits & Isolation

- **Train Split**: 1,020 utterances (170 per class, 17 distinct voices)
- **Validation Split**: 240 utterances (40 per class, 4 distinct voices)
- **Held-Out Test Split**: 240 utterances (40 per class, 4 distinct voices)

Total: 1,500 utterances across 25 unique voices. All audio files are verified 16 kHz mono WAVs with unique SHA-256 hashes and zero cross-split leakage.

---

## 4. Execution Commands

### Training
```bash
# Train sequentially across seeds 42, 43, 44
python experiments/vieneu_cnn_6class/train.py --all-seeds

# Train single seed
python experiments/vieneu_cnn_6class/train.py --seed 42
```

### Evaluation
```bash
# Evaluate checkpoint on held-out test split
python experiments/vieneu_cnn_6class/evaluate.py --checkpoint experiments/vieneu_cnn_6class/checkpoints/best_checkpoint_seed42.pt
```

### Unit Tests
```bash
pytest tests/test_vieneu_cnn_6class.py -v
```

---

## 5. Directory Structure

```
experiments/vieneu_cnn_6class/
├── config.json                     # Training configuration and hyperparameters
├── train.py                        # Multi-seed training pipeline
├── evaluate.py                     # Test evaluation and diagnostic script
├── README.md                       # This overview
├── checkpoints/
│   ├── best_checkpoint_seed42.pt   # Best checkpoint for seed 42
│   ├── best_checkpoint_seed43.pt   # Best checkpoint for seed 43
│   └── best_checkpoint_seed44.pt   # Best checkpoint for seed 44
├── logs/
│   ├── history_seed42.json         # Per-epoch training curves for seed 42
│   ├── history_seed43.json         # Per-epoch training curves for seed 43
│   └── history_seed44.json         # Per-epoch training curves for seed 44
├── predictions/
│   ├── test_predictions_seed42.csv # Per-sample predictions for seed 42
│   ├── test_metrics_seed42.json    # Comprehensive test metrics for seed 42
│   ├── test_predictions_seed43.csv # Per-sample predictions for seed 43
│   ├── test_metrics_seed43.json    # Comprehensive test metrics for seed 43
│   ├── test_predictions_seed44.csv # Per-sample predictions for seed 44
│   └── test_metrics_seed44.json    # Comprehensive test metrics for seed 44
└── reports/
    └── SIX_CLASS_CNN_TRAINING_REPORT.md  # Comprehensive research report
```
