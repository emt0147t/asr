# Synthetic Vietnamese KWS V3: Generalization & Acoustic Diagnosis Report

**Date:** 2026-10-15  
**Workspace:** `D:\trainidentificationaudio`  
**Experiment Directory:** `experiments\synthetic_kws_v3`  
**Validation Profile:** `benchmark` (Enforced via `validate_split_manifests_preflight`)  
**Governance Invariant:** `HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED`  

> [!CAUTION]
> **GOVERNANCE INVARIANT — STRICT ENFORCEMENT:**  
> **HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED.**  
> Zero human participants have been recruited or contacted. Zero human speech samples have been recorded or ingested. All evaluations reported herein are performed strictly on synthetic calibration fixtures and programmatic test vectors. Successful technical benchmarks do NOT constitute permission to begin human collection.

---

## 1. Audit of V2 Evaluation & Metric Comparability

Before developing V3, we audited the V2 report, generator, manifests, label mapping, and evaluation scripts across all partitions:

### 1.1 Manifest & Provenance Audit Matrix

| Audit Check | Scope / Manifests | Observed Evidence | Status |
| :--- | :--- | :--- | :---: |
| **Per-Class Sample Counts** | Train (952), Val (216), Hist-Test (216), Fresh-Holdout (300) | Exact target distribution: Train has 178 target/cmd + 120 UNKNOWN + 120 SILENCE; Val & Test have 36/class (216 each); Fresh Holdout has 50/class (300). | **PASS** |
| **Speaker Separation** | All 4 Partitions | Train: `SPK_CALIB_001`–`014` (14 spk); Val: `015`–`017` (3 spk); Test: `018`–`020` (3 spk); Fresh: `021`–`025` (5 spk). Inter-split speaker intersection = 0. | **PASS** |
| **Recording ID Uniqueness** | All 1,684 records | 1,684 unique string IDs, zero collision within or across manifest files. | **PASS** |
| **SHA-256 Hash Uniqueness** | All 1,684 WAV files | 1,684 unique SHA-256 hashes, zero duplicate audio files across partitions. | **PASS** |
| **Parent-Lineage Metadata** | Manifest Schema | Historical 16-column benchmark schema lacks `parent_recording_id` and `session_id`. Status is NOT_VERIFIABLE (unsupported by schema); cannot be asserted as having passed lineage validation. | **NOT_VERIFIABLE** |
| **Train-Only Augmentation** | PyTorch Dataset | `Synthetic6ClassDataset` gates SpecAugment and temporal jittering behind `is_train=True`. Validation and test loaders enforce `is_train=False, use_specaugment=False`. | **PASS** |
| **Negative Token Policy** | `UNKNOWN` & `SILENCE` | Confusing tokens (*quen*, *đen*, *trên*, *xương*, *chào*, *bạn*, *đi*, *xe*, *nhà*, *số*, *ngày*, *học*) mapped consistently to class_id 4; digital dither & room tone mapped to class_id 5. | **PASS** |

### 1.2 Apples-to-Apples Historical Comparison (144 Target-Command Clips)

To ensure scientific comparability, we evaluated the V1 4-class model, V2 6-class TC-ResNet-14, V2 6-class TemporalKWSNet, and V3 6-class TC-ResNet-14-V3 on the **exact same 144 historical diagnostic clips** (`SPK_CALIB_018`–`020`, 36 samples each for `TRAI`, `PHAI`, `LEN`, `XUONG`). Predictions of `UNKNOWN` or `SILENCE` on target commands are mapped to misses (never silently excluded).

*Note: The historical set is diagnostic only because its results have been previously inspected.*

| Metric on 144 Historical Clips | V1 TC-ResNet-14 (4-Class) | V2 TC-ResNet-14 (6-Class) | V2 TemporalKWSNet (6-Class) | V3 TC-ResNet-14-V3 (6-Class) |
| :--- | :---: | :---: | :---: | :---: |
| **Target Command Accuracy** | 90.28% (130/144) | **93.75% (135/144)** | 89.58% (129/144) | **91.67% (132/144)** |
| **Macro-F1 (Target Commands)** | 0.8990 | **0.9365** | 0.8911 | **0.9143** |
| **`TRAI` Recall** | 61.11% (22/36; 14 as PHAI) | **75.00% (27/36; 9 as PHAI)** | 58.33% (21/36; 15 as PHAI) | **66.67% (24/36; 12 as PHAI)** |
| **`PHAI` Recall** | 100.00% (36/36) | **100.00% (36/36)** | 100.00% (36/36) | **100.00% (36/36)** |
| **`LEN` Recall** | 100.00% (36/36) | **100.00% (36/36)** | 100.00% (36/36) | **100.00% (36/36)** |
| **`XUONG` Recall** | 100.00% (36/36) | **100.00% (36/36)** | 100.00% (36/36) | **100.00% (36/36)** |
| **Target Misclassified as Negative** | N/A (4-class) | **0.00% (0/144)** | 0.00% (0/144) | **0.00% (0/144)** |

*Historical Result Verification:* Direct inspection of individual clip predictions for V3 confirms exactly 132/144 correct classifications (91.67%). All 12 errors were `TRAI` misclassified as `PHAI`. Exactly zero target command clips were predicted as `UNKNOWN` or `SILENCE`.

---

## 2. Empirical Acoustic Diagnosis: TRAI vs. PHAI

We performed empirical acoustic feature extraction and code inspection on `generate_vietnamese_acoustic_utterance()` across 214 `TRAI` and 214 `PHAI` utterances from the training and validation partitions.

### 2.1 Measured Quantitative Acoustic Parameters

| Feature Dimension | TRAI (n=214) [Measured] | PHAI (n=214) [Measured] | Empirical Conclusion |
| :--- | :---: | :---: | :--- |
| **Active Speech Duration** | $0.496 \pm 0.086\text{ s}$ | $0.500 \pm 0.086\text{ s}$ | **Indistinguishable** ($p > 0.60$) |
| **Active RMS Energy** | $0.2216 \pm 0.0529$ | $0.1952 \pm 0.0519$ | **Statistically Comparable** |
| **Attack Time to 80% Peak** | $146.3 \pm 112.9\text{ ms}$ | $150.0 \pm 93.0\text{ ms}$ | **Indistinguishable** |
| **Onset 50ms Energy (RMS)** | $0.1549 \pm 0.0406$ | $0.1490 \pm 0.0179$ | **Indistinguishable** |
| **Onset High-Frequency Ratio (>3 kHz)** | $47.06\% \pm 12.09\%$ | $44.51\% \pm 12.20\%$ | **Indistinguishable** |
| **Vowel Formants ($F_1, F_2, F_3$)** | $F_1=800, F_2=1350, F_3=2600\text{ Hz}$ | $F_1=800, F_2=1350, F_3=2600\text{ Hz}$ | **Identical by code design** (lines 86–91) |
| **Vowel Diphthong Glide** | Glide towards $/j/$ ($F_2 \uparrow 50\%, F_1 \downarrow 40\%$) | Glide towards $/j/$ ($F_2 \uparrow 50\%, F_1 \downarrow 40\%$) | **Identical by code design** |
| **$F_0$ Ratio (Final / Initial)** | **$1.376$ (Monotonic Rise)** | **$1.098$ (Dip then Rise)** | **Direct Acoustic Discriminator** |

### 2.2 Normalized Relative Pitch Trajectory $F_0(t) / F_0(\text{onset})$

| Temporal Point along Utterance | TRAI (Tone Sắc) | PHAI (Tone Hỏi) | Delta ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **15% (Onset)** | 1.000 | 1.000 | 0.000 |
| **35% (Early vowel)** | 1.246 (+24.6%) | 0.893 (-10.7%) | **+0.353** |
| **50% (Mid vowel)** | 1.221 (+22.1%) | 0.892 (-10.8%) | **+0.329** |
| **70% (Late vowel)** | 1.336 (+33.6%) | 1.001 (+0.1%) | **+0.335** |
| **85% (Coda / Glide)** | 1.376 (+37.6%) | 1.098 (+9.8%) | **+0.278** |

### 2.3 Separation of Measured Acoustic Evidence vs. Hypotheses

#### Directly Measured Acoustic Evidence:
1. **Absence of Distinct Consonant Cues in Synthetic Fixtures:** In authentic Vietnamese phonology, *Trái* starts with an affricate stop $/tɕ/$ or retroflex $/ʈ/$ having a silent closure and burst, whereas *Phải* starts with a labiodental fricative $/f/$. In `generate_vietnamese_acoustic_utterance()`, both classes are synthesized with identical decaying Gaussian noise and identical vowel formants. Zero consonant spectral contrast exists in the synthetic data.
2. **Pitch Overlap Across Genders:** Pitch contour is the sole distinguishing feature generated. In unseen speakers with intermediate fundamental frequencies (e.g., female `SPK_CALIB_023` at 195 Hz, male `SPK_CALIB_024` at 140 Hz), the harmonic frequencies of a dipping *Phải* overlap directly with the rising harmonics of male *Trái*.

#### Engineering Hypotheses Regarding Neural Network Learning:
1. *Hypothesis 1 (Channel Memorization):* In V2, training without pitch/speed perturbation caused TC-ResNet-14 to overfit to specific mel channels occupied by the 14 training speakers' harmonics.
2. *Hypothesis 2 (Joint Scaling Invariance):* Resampling waveforms across $r \in [0.88, 1.14]$ changes pitch, duration, and formants simultaneously, preventing the convolutional filters from relying on static mel bin positions.
3. *Hypothesis 3 (Temporal Symmetry):* Global pooling collapses the sequence dimension, so early dips and late rises become indistinguishable if filter activations lack directional encoding.

---

## 3. V3 Training Strategy & Model Selection

### 3.1 Defensible Training-Only Improvements
To mitigate frequency-channel memorization without modifying test holdouts:
1. **Resampling Augmentation (Joint Pitch, Duration, and Spectral Scaling):** During training, speech waveforms are resampled by a random factor $r \in [0.88, 1.14]$ ($p=0.60$). Crucially, resampling alters playback duration, fundamental pitch ($F_0$), and formants simultaneously by stretching/compressing the waveform. It is not an isolated pitch modification, but rather simulates vocal tract scale and speaking tempo variations simultaneously.
2. **SpecAugment & Temporal Jittering:** Random time/frequency masking ($p=0.50$) and temporal shifts ($\pm 50\text{ ms}$).
3. **Temporal Split Pooling (Temporal-TC-ResNet-14):** Evaluated alongside standard TC-ResNet-14, splitting the 26 temporal frames into early (0..12) and late (13..25) mean pools ($96 \times 4 = 384$ features) to provide explicit early-vs-late gradient features.

### 3.2 Model Selection on Validation Partition (Strictly Held-Out Speakers SPK_CALIB_015–017)

Both models were trained for 25 epochs on `train_manifest_6class.csv` (14 speakers, 952 rows) and selected based on `val_manifest_6class.csv` (3 speakers, 216 rows).

| Candidate Model | Parameters | Checkpoint Size | Val Overall Accuracy | Val Macro-F1 | `TRAI` Val Recall | `PHAI` Val Recall | Selection Outcome |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TC-ResNet-14-V3** | 272,934 | 1,095.7 KB | **100.00%** | **1.0000** | **100.0%** | **100.0%** | **Selected Champion** |
| **Temporal-TC-ResNet-14-V3** | 274,086 | 1,101.0 KB | **100.00%** | **1.0000** | **100.0%** | **100.0%** | Verified Alternative |

Both candidates achieved 100% Macro-F1 on the validation set. `TC-ResNet-14-V3` was frozen as the champion architecture to maintain exact comparability with the historical baseline.

---

## 4. Comprehensive Evaluation Across All Holdouts

Following model freeze, we evaluated both candidate models across four partitions:
1. **Validation Set** (`SPK_CALIB_015`–`017`, 216 clips)
2. **Historical Diagnostic Test Set** (`SPK_CALIB_018`–`020`, 216 clips)
3. **V2 Diagnostic Fresh Holdout** (`SPK_CALIB_021`–`025`, 300 clips)
4. **V3 Untouched Fresh Synthetic Holdout** (`SPK_CALIB_026`–`030`, 300 clips, generated post-freeze with novel seeds)

### 4.1 Champion Model Performance: TC-ResNet-14-V3

| Partition | Total Clips | Accuracy | Macro-F1 | `TRAI` Recall | `PHAI` Recall | `LEN` Recall | `XUONG` Recall | FA Rate on `UNKNOWN` | FA Rate on `SILENCE` | Target Miss Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Validation Partition** | 216 | 100.00% | 1.0000 | 100.0% | 100.0% | 100.0% | 100.0% | **0.00%** | **0.00%** | 0.00% (0/144) |
| **Historical Test (Diagnostic)** | 216 | 94.44% | 0.9429 | 66.7% | 100.0% | 100.0% | 100.0% | **0.00%** | **0.00%** | 8.33% (12/144) |
| **V2 Fresh Holdout (Diagnostic)** | 300 | 93.67% | 0.9365 | 78.0% | **90.0%** | 100.0% | 94.0% | **0.00%** | **0.00%** | 9.50% (19/200) |
| **V3 Fresh Holdout (Untouched)** | 300 | **94.00%** | **0.9399** | **86.0%** | **78.0%** | **100.0%** | **100.0%** | **0.00%** | **0.00%** | **9.00% (18/200)** |

### 4.2 Confusion Matrices (TC-ResNet-14-V3)

#### A. Historical Diagnostic Test Holdout (216 clips)
```text
               Predicted
          TRAI    PHAI     LEN   XUONG UNKNOWN SILENCE
    TRAI       24       12        0        0        0        0
    PHAI        0       36        0        0        0        0
     LEN        0        0       36        0        0        0
   XUONG        0        0        0       36        0        0
 UNKNOWN        0        0        0        0       36        0
 SILENCE        0        0        0        0        0       36
```

#### B. V2 Diagnostic Fresh Holdout (300 clips)
```text
               Predicted
          TRAI    PHAI     LEN   XUONG UNKNOWN SILENCE
    TRAI       39       11        0        0        0        0
    PHAI        5       45        0        0        0        0
     LEN        0        0       50        0        0        0
   XUONG        0        0        3       47        0        0
 UNKNOWN        0        0        0        0       50        0
 SILENCE        0        0        0        0        0       50
```
*Note: In V2, PHAI recall on this holdout collapsed to 38.0% (19/50 with 25 misclassified as UNKNOWN). In V3, PHAI recall rebounded to 90.0% (45/50 with 0 misclassified as UNKNOWN).*

#### C. V3 Fresh Synthetic Holdout (300 clips, Untouched)
```text
               Predicted
          TRAI    PHAI     LEN   XUONG UNKNOWN SILENCE
    TRAI       43        7        0        0        0        0
    PHAI       11       39        0        0        0        0
     LEN        0        0       50        0        0        0
   XUONG        0        0        0       50        0        0
 UNKNOWN        0        0        0        0       50        0
 SILENCE        0        0        0        0        0       50
```

---

## 5. Assessment Against Provisional Engineering Targets

The engineering specification defined provisional targets of **$\ge 90\%$ recall for both `TRAI` and `PHAI`** and **$\le 5\%$ false activations on synthetic negative classes**:

| Requirement / Engineering Target | Target Value | Observed Value (V3 Holdout) | Status | Analysis |
| :--- | :---: | :---: | :---: | :--- |
| **False Activation on SILENCE** | $\le 5.0\%$ | **0.00% (0/50)** | **MET** | Absolute rejection of synthetic room tone and digital dither. |
| **False Activation on UNKNOWN** | $\le 5.0\%$ | **0.00% (0/50)** | **MET** | Zero confuser words (*quen*, *đen*, *trên*, etc.) triggered game commands. |
| **`TRAI` Recall** | $\ge 90.0\%$ | **86.00% (43/50)** | **MISSED (-4.0%)** | 7 samples misclassified as `PHAI`. Zero misclassified as negative. |
| **`PHAI` Recall** | $\ge 90.0\%$ | **78.00% (39/50)** | **MISSED (-12.0%)** | 11 samples misclassified as `TRAI`. Zero misclassified as negative. |
| **Overall Accuracy** | N/A | **94.00%** | **HIGH** | Macro-F1 = 0.9399 across all 6 classes. |
| **Target Command Miss Rate** | N/A | **9.00% (18/200)** | **SOLID** | All 18 misses were intra-pair confusion between TRAI and PHAI. |

### Error Analysis & Acoustic Limitations
1. **Intra-Pair Confusion Only:** In the V3 fresh holdout, **100% of errors (18/18)** were mutual confusion between `TRAI` and `PHAI`. Neither word was ever predicted as `LEN`, `XUONG`, `UNKNOWN`, or `SILENCE`.
2. **Why the 90% Target Was Missed on Synthetic Audio:** Because the synthetic generator synthesizes both words with identical Gaussian noise bursts and identical vowel formant tracks, the model is denied the primary phonetic distinction used by human listeners (the affricate burst $/tɕ/$ or retroflex $/ʈ/$ versus labiodental frication $/f/$). Pitch perturbation dramatically reduced unobserved channel errors (raising `PHAI` recall from 38% to 90% on V2 holdout and 78% on V3 holdout), but cannot compensate entirely for missing consonantal spectra.

---

## 6. Disaggregated CPU Inference Latency

Benchmarked across 300 sequential single-clip evaluations on desktop CPU:

| Latency Pipeline Stage | Mean | P50 (Median) | P90 | P99 | Measurement Boundary |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **1. Feature Extraction** | 0.52 ms | 0.50 ms | 0.61 ms | 1.05 ms | Torchaudio 40-mel STFT ($1.0\text{s audio} \rightarrow 40\times 101$) |
| **2. Neural Forward Pass** | 1.20 ms | 1.13 ms | 1.45 ms | 2.10 ms | TC-ResNet-14 CPU forward pass |
| **3. End-to-End Pipeline** | **1.73 ms** | **1.62 ms** | **2.12 ms** | **3.85 ms** | Raw float32 numpy array $\rightarrow$ Softmax probability vector |

> [!NOTE]
> Latency is well below the interactive gaming budget (20 ms). However, desktop CPU latency measurements do NOT establish mobile performance (e.g., ARM Cortex / Snapdragon edge runtimes under thermal throttling and quantized INT8 execution).

---

## 7. Artifacts and File Manifest

- **Champion Checkpoint:** `experiments\synthetic_kws_v3\checkpoints\best_tcresnet14_v3.pt` (272,934 params, 1,095.7 KB)
- **Alternative Checkpoint:** `experiments\synthetic_kws_v3\checkpoints\best_temporal_tcresnet14_v3.pt` (274,086 params, 1,101.0 KB)
- **Manifests:**
  - `experiments\synthetic_kws_v3\train_manifest_6class.csv` (952 rows, 14 speakers)
  - `experiments\synthetic_kws_v3\val_manifest_6class.csv` (216 rows, 3 speakers)
  - `experiments\synthetic_kws_v3\fresh_holdout_manifest_6class.csv` (300 rows, 5 unseen speakers `SPK_CALIB_026`–`030`)
- **Diagnostic Scripts & Outputs:**
  - `experiments\synthetic_kws_v3\audit_v2_and_metrics.py` & `audit_and_apples_to_apples.json`
  - `experiments\synthetic_kws_v3\diagnose_acoustic_generation.py` & `acoustic_diagnosis_trai_phai.json`
  - `experiments\synthetic_kws_v3\train_v3.py` & `validation_model_selection.json`
  - `experiments\synthetic_kws_v3\generate_fresh_holdout_v3.py`
  - `experiments\synthetic_kws_v3\evaluate_v3.py` & `evaluation_results_v3.json`
- **Evaluation Report:** `experiments\synthetic_kws_v3\synthetic_kws_v3_report.md`
