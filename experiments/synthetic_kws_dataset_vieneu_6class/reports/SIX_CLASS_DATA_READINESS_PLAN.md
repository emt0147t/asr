# Six-Class Vietnamese KWS Data Readiness Plan

**Date:** 2026-10-18  
**Workspace:** `D:\trainidentificationaudio`  
**Target Dataset:** `experiments/synthetic_kws_dataset_vieneu_6class/`  
**Current Status:** Dataset assembled, validated, and ready for baseline training.  

---

## 1. Overview & Governance Invariants

This plan establishes the technical and scientific roadmap for transitioning from the 6-class dataset build (Phase 28) to 6-class CNN modeling (Phase 29), noise-robustness benchmarking, and subsequent field-speech validation.

### Strict Governance Mandates
1. **`PENDING_HUMAN_LISTENER_VERIFICATION`**: Human perceptual intelligibility evaluation of the 80 blinded clips (Phase 25) remains pending. Synthetic KWS training may proceed for acoustic separation baselines, but no clinical or production deployment claims may be made until the human listening sheet is finalized.
2. **`NOT_VALIDATED_ON_NATURAL_SPEECH`**: All training and test partitions in this phase consist of clean synthetic speech and simulated acoustic ambience. The model must not be represented as validated on natural human speech recordings.
3. **Frozen Four-Class Test Split Preservation**: The 160-sample four-class test split remains permanently frozen and was not used to select any parameters, vocabularies, or configurations in this six-class dataset build.

---

## 2. Six-Class Label Mapping & Architecture Readiness

### 2.1 Canonical Class Mapping
The classification space is formally defined in [`src/models/kws_cnn.py`](file:///d:/trainidentificationaudio/src/models/kws_cnn.py):

| Class Index | Label | Semantic Meaning | Target Vocabulary / Condition |
| :---: | :--- | :--- | :--- |
| `0` | `TRAI` | Directional command | "Trái" (/ʈaːj˧˥/) |
| `1` | `PHAI` | Directional command | "Phải" (/faːj˧˩˨/) |
| `2` | `LEN` | Directional command | "Lên" (/len˧˧/) |
| `3` | `XUONG` | Directional command | "Xuống" (/suəŋ˧˥/) |
| `4` | `UNKNOWN` | Non-target speech | 30 disjoint Vietnamese words/phrases |
| `5` | `SILENCE` | No speech / ambient background | 11 disjoint non-speech acoustic conditions |

### 2.2 Model Architecture Compatibility
The existing compact 2D-CNN architecture (`VietnameseKWSNet`) supports 6 classes natively:
- **Input Dimensions**: $(B, 1, 40, 101)$ from 16,000 samples at $16\text{ kHz}$.
- **Convolutional Backbone**: 4 Conv blocks with BatchNorm, ReLU, and MaxPool ($138,534$ parameters for 6 classes).
- **Head**: `AdaptiveAvgPool2d((1, 1))` $\rightarrow$ `Dropout(0.3)` $\rightarrow$ `Linear(128, 64)` $\rightarrow$ `Linear(64, 6)`.
- **Latency Expectation**: $< 1.5\text{ ms}$ on single-thread mobile CPU; $< 0.05\text{ ms}$ per sample on batched GPU.

---

## 3. Training Protocol for Phase 29

1. **Dataset Partitions**:
   - Train: 1,020 samples (170 per class, 17 distinct voices).
   - Validation: 240 samples (40 per class, 4 distinct voices).
   - Held-Out Test: 240 samples (40 per class, 4 distinct voices).
2. **Deterministic Multi-Seed Protocol**:
   - Seeds: `42`, `43`, `44`.
   - Optimizer: AdamW ($\text{lr}=10^{-3}$, weight decay $=10^{-4}$).
   - LR Scheduler: Cosine Annealing (40 epochs, $\eta_{\text{min}}=10^{-6}$).
   - Batch Size: 32.
3. **Training-Only Data Augmentations**:
   - Random time shift: $\pm 25\text{ ms}$ ($\pm 400$ samples within safe margins).
   - Random amplitude scaling: $[0.9, 1.1]$.
   - Augmentation applied strictly to train samples; validation and test remain strictly unaugmented.
4. **Validation-Driven Checkpointing**:
   - Save checkpoint exclusively based on peak `val_macro_f1`.
   - Test split must never be evaluated during training, early stopping, or hyperparameter selection.

---

## 4. Controlled Noise-Robustness Benchmarking Protocol

### 4.1 Separation from Core Dataset
Noise corruption must **never** be mixed indiscriminately into the clean balanced dataset. Doing so obscures whether classification errors stem from phonetic confusion or noise masking.

### 4.2 Future Noise Benchmark Architecture
In a dedicated subsequent evaluation phase:
1. **Noise Profiles**:
   - Stationary White Noise (broadband thermal noise).
   - Stationary Pink Noise ($1/f$ spectral roll-off).
   - Real-World HVAC / Office Noise (low-frequency resonance + broadband ventilation).
   - Real-World Multi-Speaker Babble (cafeteria / street background).
2. **Controlled SNR Levels**:
   - High SNR: $+20\text{ dB}$, $+15\text{ dB}$ (subtle degradation).
   - Moderate SNR: $+10\text{ dB}$ (typical indoor mobile environment).
   - Low SNR: $+5\text{ dB}$, $0\text{ dB}$ (stress-test edge cases).
3. **Evaluation Protocol**:
   - Evaluate clean-trained 6-class checkpoints against noisy copies of the held-out test split.
   - Record per-SNR and per-noise accuracy, macro F1, and confusion matrices to establish SNR breakdown curves.

---

## 5. Roadmap to Natural Human Speech Validation

1. **Step 1 (Immediate)**: Conclude Phase 28 dataset build and audit; execute focused test suite.
2. **Step 2**: Execute Phase 29 baseline training of the 6-class CNN on `synthetic_kws_dataset_vieneu_6class/`.
3. **Step 3**: Finalize Phase 25 blinded human listening evaluation sheet to confirm perceptual naturalness and intelligibility.
4. **Step 4**: Execute controlled noise-robustness benchmarking on the 6-class test split across SNRs.
5. **Step 5**: Conduct field natural human speech acquisitions under institutional consent protocols to test cross-domain transfer learning from synthetic to real human speech.
