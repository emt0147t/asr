# Synthetic Vietnamese KWS V4: Controlled Onset-Acoustic Experiment Report

**Date:** 2026-10-15  
**Workspace:** `D:\trainidentificationaudio`  
**Experiment Directory:** `experiments\synthetic_kws_v4`  
**Validation Profile:** `benchmark` (Enforced via `validate_split_manifests_preflight`)  
**Governance Invariant:** `HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED`  

> [!CAUTION]
> **GOVERNANCE INVARIANT — STRICT ENFORCEMENT:**  
> **HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED.**  
> Zero human participants have been recruited or contacted. Zero human speech samples have been recorded or ingested. All evaluations reported herein are performed strictly on programmatic synthetic fixtures. Successful technical benchmarks do NOT constitute authorization or approval to collect human field data.

---

## 1. Executive Summary & Problem Formulation

In previous phases (V1–V3), empirical acoustic diagnosis proved that the baseline synthetic generator synthesized `TRAI` and `PHAI` with **identical onset noise envelopes, identical vowel formants, and identical durations**, leaving $F_0$ pitch trajectory as the sole distinguishing acoustic cue. While joint resampling augmentation (V3) mitigated pitch-channel memorization across speakers, mutual `TRAI`/`PHAI` confusion persisted because the model lacked consonantal spectral contrasts.

In **V4**, we executed a controlled experimental investigation to determine:
1. Whether introducing phonetically grounded onset parameterization (affricate stop closure + high-frequency burst for *Trái* vs. continuous diffuse frication for *Phải*) enables the model to separate both classes.
2. How onset parameterization interacts with joint pitch/speed resampling augmentation.
3. Whether synthetic onset engineering creates cross-generator fragility when evaluated against disparate acoustic assumptions.

---

## 2. Linguistic & Phonetic Foundations for Onset Modeling

### 2.1 Phonetic Contrasts in Vietnamese Initial Consonants
Before implementing synthetic onset parameters, we consulted phonetic literature on Vietnamese initial consonants:
- **Orthographic <tr-> (*Trái*):**
  - In standard Hanoi / Northern Vietnamese, <tr-> is realized as a voiceless laminal alveolo-palatal affricate stop $/tɕ/$.
  - In Central and Southern Vietnamese dialects, <tr-> is typically pronounced as a voiceless retroflex stop $/ʈ/$ or retroflex affricate $/ʈʂ/$.
  - **Acoustic Signature:** A stop closure gap (brief silence of $20\text{--}35\text{ ms}$ where vocalic energy is suppressed), followed by an abrupt, high-frequency release burst ($>3.5\text{ kHz}$) with rapid spectral transitions into the open vowel $/a/$.
- **Orthographic <ph-> (*Phải*):**
  - Across Northern, Central, and Southern dialects, <ph-> is consistently realized as a voiceless labiodental fricative $/f/$.
  - **Acoustic Signature:** Continuous voiceless frication starting immediately at $t=0$ ($40\text{--}60\text{ ms}$), with a diffuse, lower-intensity broadband spectrum ($1.8\text{--}6.0\text{ kHz}$) and no silent closure gap or impulsive transient burst.

### 2.2 Critical Dialectal Limitations
> [!WARNING]
> **DIALECT & SYNTHESIS LIMITATION:**  
> While the onset-aware generator models the general stop-affricate vs. fricative dichotomy, Vietnamese initial consonants exhibit complex regional allophonic variation (e.g. Southern retroflexion, Central tense articulation). A handcrafted mathematical waveform cannot represent authentic human dialect diversity. **No claims of regional accent generalization are made based on simulated dialect tags.**

---

## 3. Controlled Ablation Design & Experimental Arms

To isolate the specific effect of onset parameterization from confounding variables, the neural network architecture (`VietnameseTCResNet`, 272,934 parameters), optimizer (AdamW, lr=1e-3, weight_decay=1e-4), training budget (25 epochs), batch size (32), random seed (42), and split partitioning (14 train speakers, 3 validation speakers) were held strictly constant across three experimental arms:

- **Arm A (Control):** Existing synthetic generator + V3 augmentation (Resampling $r \in [0.88, 1.14]$ + SpecAugment + Temporal jitter).
- **Arm B (Onset-Aware, No Resampling):** Onset-aware generator (stop closure + burst for *Trái*; continuous frication for *Phải*) + standard SpecAugment and temporal jitter (NO resampling pitch perturbation).
- **Arm C (Onset-Aware + V3 Augmentation):** Onset-aware generator combined with both onset acoustic modeling AND joint resampling augmentation.

### 3.1 Ablation Results on Within-Distribution Validation Partition (SPK_CALIB_015–017, 216 clips)

| Experimental Arm | Generator Type | Augmentation Strategy | Validation Set Manifest | Best Val Accuracy | Best Val Macro-F1 | `TRAI` Val Recall | `PHAI` Val Recall | Val Loss |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Arm A (Control)** | Baseline Generator | Resampling + SpecAugment | `val_manifest_6class.csv` (Baseline Gen) | **100.00%** | **1.0000** | **100.0%** | **100.0%** | 0.3116 |
| **Arm B (Onset-Aware)** | Onset-Aware Generator | SpecAugment Only | `val_manifest_onset_aware.csv` (Onset Gen) | **100.00%** | **1.0000** | **100.0%** | **100.0%** | **0.2462** |
| **Arm C (Onset-Aware + Resample)**| Onset-Aware Generator | Resampling + SpecAugment | `val_manifest_onset_aware.csv` (Onset Gen) | **100.00%** | **1.0000** | **100.0%** | **100.0%** | 0.2503 |

*(Note on Arm A: At training epoch 25, Arm A achieved 96.76% accuracy and 0.9673 Macro-F1 on `val_manifest_6class.csv`, with best checkpoint achieving 100.00% at epoch 9. When evaluated on the onset-aware validation set `val_manifest_onset_aware.csv`, Arm A drops to 82.87% due to cross-generator shift, as shown in Table 4.1).*

**Selection Outcome:** Both Arm B and Arm C achieved 100% Macro-F1 on their matching validation set. Arm B demonstrated lower validation loss ($0.2462$ vs. $0.2503$) and avoided perturbation-induced confuser blurring. Both models were frozen and evaluated across all test partitions.

---

## 4. Multi-Partition Comprehensive Evaluation

Following model freeze, all three ablation arms were evaluated across all partitions:
1. **Validation Partition (Onset-Aware Gen)** (`SPK_CALIB_015`–`017`, 216 clips)
2. **Validation Partition (Baseline Gen)** (`SPK_CALIB_015`–`017`, 216 clips)
3. **Historical Diagnostic Test Holdout** (`SPK_CALIB_018`–`020`, 216 clips)
4. **V2 Diagnostic Fresh Holdout** (`SPK_CALIB_021`–`025`, 300 clips)
5. **V3 Diagnostic Fresh Holdout** (`SPK_CALIB_026`–`030`, 300 clips)
6. **V4 Untouched Fresh Holdout** (`SPK_CALIB_031`–`035`, 300 clips, generated post-freeze with novel seeds)

### 4.1 Cross-Partition Evaluation Matrix

| Experimental Arm | Metric | Val Onset Gen (216) | Val Baseline Gen (216) | Hist. Test (216) [Diag] | V2 Holdout (300) [Diag] | V3 Holdout (300) [Diag] | V4 Holdout (300) [Untouched] |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Arm A** *(Control: Baseline Gen + Resample)* | **Accuracy** | 82.87% | **100.00%** | **94.44%** | **93.67%** | **94.00%** | 86.67% |
| | **Macro-F1** | 0.7998 | **1.0000** | **0.9429** | **0.9365** | **0.9399** | 0.8543 |
| | `TRAI` Recall | 22.2% | **100.0%** | 66.7% | 78.0% | 86.0% | 34.0% |
| | `PHAI` Recall | 100.0% | **100.0%** | 100.0% | 90.0% | 78.0% | 88.0% |
| | UNKNOWN FA Rate | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 2.00% |
| | SILENCE FA Rate | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| **Arm B** *(Onset-Aware, No Resample)* | **Accuracy** | **100.00%** | 76.85% | 69.44% | 83.33% | 83.33% | **99.33%** |
| | **Macro-F1** | **1.0000** | 0.7080 | 0.6097 | 0.7778 | 0.7778 | **0.9933** |
| | `TRAI` Recall | **100.0%** | 0.0% | 0.0% | 0.0% | 0.0% | **96.0%** |
| | `PHAI` Recall | **100.0%** | 100.0% | 100.0% | 100.0% | 100.0% | **100.0%** |
| | UNKNOWN FA Rate | **0.00%** | 0.00% | **0.00%** | **0.00%** | **0.00%** | **0.00%** |
| | SILENCE FA Rate | **0.00%** | 0.00% | **0.00%** | **0.00%** | **0.00%** | **0.00%** |
| **Arm C** *(Onset-Aware + Resample)* | **Accuracy** | **100.00%** | 61.57% | 64.81% | 76.00% | 77.67% | 95.00% |
| | **Macro-F1** | **1.0000** | 0.6152 | 0.6081 | 0.7285 | 0.7346 | 0.9485 |
| | `TRAI` Recall | **100.0%** | 52.8% | 52.8% | 0.0% | 0.0% | **100.0%** |
| | `PHAI` Recall | **100.0%** | 25.0% | 75.0% | 100.0% | 100.0% | **100.0%** |
| | UNKNOWN FA Rate | **0.00%** | 47.22% | 44.44% | 42.00% | 28.00% | 30.00% |
| | SILENCE FA Rate | **0.00%** | 0.00% | **0.00%** | **0.00%** | **0.00%** | **0.00%** |

### 4.2 Apples-to-Apples Comparison on Historical 144 Target Clips [Diagnostic]

Directly evaluated on `experiments/synthetic_kws_baseline/test_manifest.csv` (144 clips, 36 per class for `TRAI`, `PHAI`, `LEN`, `XUONG`; 0 `UNKNOWN`/`SILENCE`):

| Model / Arm | Accuracy | 4-Class Macro-F1 | `TRAI` Recall | `PHAI` Recall | `LEN` Recall | `XUONG` Recall | Target Miss Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Arm A (Control)** | **91.67%** | **0.9143** | 66.7% (24/36) | **100.0% (36/36)** | **100.0% (36/36)** | **100.0% (36/36)** | **8.33% (12/144)** |
| **Arm B (Onset-Aware, No Resample)** | 54.17% | 0.4146 | 0.0% (0/36) | **100.0% (36/36)** | **100.0% (36/36)** | 16.7% (6/36) | 45.83% (66/144) |
| **Arm C (Onset-Aware + Resample)** | 58.33% | 0.5133 | 52.8% (19/36) | 75.0% (27/36) | **100.0% (36/36)** | 5.6% (2/36) | 41.67% (60/144) |
| **V3 Champion TC-ResNet-14** | **91.67%** | **0.9143** | 66.7% (24/36) | **100.0% (36/36)** | **100.0% (36/36)** | **100.0% (36/36)** | **8.33% (12/144)** |

*(All numbers verified against canonical confusion matrices in `experiments/synthetic_kws_v4/audit/metric_integrity_audit.md`).*

---

## 5. Confusion Matrix Analysis (V4 Fresh Holdout, 300 Clips)

### 5.1 Arm B: Onset-Aware Generator (Winning Arm on Target Fidelity)
```text
               Predicted
          TRAI    PHAI     LEN   XUONG UNKNOWN SILENCE
    TRAI       48        0        0        0        2        0
    PHAI        0       50        0        0        0        0
     LEN        0        0       50        0        0        0
   XUONG        0        0        0       50        0        0
 UNKNOWN        0        0        0        0       50        0
 SILENCE        0        0        0        0        0       50
```
- **Total Accuracy:** **99.33% (298/300)**
- **Macro-F1:** **0.9933**
- **TRAI Recall:** **96.00% (48/50)**; 2 samples missed as UNKNOWN; **0/50 confused with PHAI**.
- **PHAI Recall:** **100.00% (50/50)**; **0/50 confused with TRAI**.
- **False Alarm on UNKNOWN:** **0.00% (0/50)**
- **False Alarm on SILENCE:** **0.00% (0/50)**
- **Command Miss Rate:** **1.00% (2/200)**

### 5.2 Arm C: Onset-Aware + Resampling
```text
               Predicted
          TRAI    PHAI     LEN   XUONG UNKNOWN SILENCE
    TRAI       50        0        0        0        0        0
    PHAI        0       50        0        0        0        0
     LEN        0        0       50        0        0        0
   XUONG        0        0        0       50        0        0
 UNKNOWN       14        0        1        0       35        0
 SILENCE        0        0        0        0        0       50
```
- While Arm C achieves 100% recall on both `TRAI` and `PHAI`, joint resampling causes **30.0% false activations on UNKNOWN** (14 confusing tokens trigger `TRAI`). Resampling non-target words shortens their duration and alters pitch slopes into command-like profiles.

---

## 6. Scientific Analysis of Cross-Generator Fragility

The ablation exposes an essential scientific finding regarding synthetic speech evaluation:

1. **Within-Distribution Separation:** When trained and evaluated on onset-aware synthesis (Arm B on V4 Holdout), the neural network immediately utilizes the stop closure and burst features, achieving **99.33% accuracy, 96.0% TRAI recall, and 100% PHAI recall** with zero `TRAI`/`PHAI` mutual confusion.
2. **Cross-Generator Incompatibility:**
   - When Arm B (trained on closure-burst `TRAI`) is evaluated on the historical, V2, or V3 holdouts (where `TRAI` was synthesized without a closure gap), its `TRAI` recall drops to **0.0%**. The network strictly expects the 25ms silence and high-pass burst.
   - Conversely, when Arm A (trained on no-closure `TRAI`) is evaluated on the V4 holdout, its `TRAI` recall drops from 86.0% to **34.0%**.
3. **Implication for Field Data:** Manually hardcoding acoustic features into synthetic generators leads to models that memorize specific mathematical synthesis formulas. **Neither synthetic generator can replace authentic human speech recordings collected under strict governance.**

---

## 7. Assessment Against Provisional Engineering Targets

| Provisional Engineering Target | Target Value | Arm B Result (V4 Holdout) | Status | Analysis |
| :--- | :---: | :---: | :---: | :--- |
| **`TRAI` Recall** | $\ge 90.0\%$ | **96.00% (48/50)** | **MET (+6.0%)** | 48 correct, 2 as UNKNOWN, 0 as PHAI |
| **`PHAI` Recall** | $\ge 90.0\%$ | **100.00% (50/50)** | **MET (+10.0%)** | 50/50 correct, 0 confused with TRAI |
| **False Activation on UNKNOWN** | $\le 5.0\%$ | **0.00% (0/50)** | **MET** | Complete rejection of negative confusers |
| **False Activation on SILENCE** | $\le 5.0\%$ | **0.00% (0/50)** | **MET** | Complete rejection of room tone & dither |
| **Target Command Miss Rate** | $\le 10.0\%$ | **1.00% (2/200)** | **MET** | Only 2 target clips missed across all 4 commands |

---

## 8. Disaggregated CPU Latency Profiling

Measured on desktop CPU across 300 sequential inferences:

| Latency Pipeline Stage | Mean | P50 (Median) | P90 | P99 | Measurement Boundary |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **1. Feature Extraction** | 0.51 ms | 0.49 ms | 0.60 ms | 1.02 ms | Torchaudio 40-mel STFT ($1.0\text{s audio} \rightarrow 40\times 101$) |
| **2. Neural Forward Pass** | 1.13 ms | 1.07 ms | 1.35 ms | 1.85 ms | TC-ResNet-14 CPU forward pass |
| **3. End-to-End Pipeline** | **1.64 ms** | **1.56 ms** | **1.85 ms** | **2.87 ms** | Raw float32 array $\rightarrow$ Softmax probability vector |

*(Note: Desktop CPU latency confirms execution well below the 20ms interactive threshold, but edge mobile devices require specialized ARM INT8 quantization benchmarking under thermal throttling).*

---

## 9. Provenance, Lineage, and Split Audit

| Invariant / Check | Scope | Verification Output | Status |
| :--- | :--- | :--- | :---: |
| **Speaker Disjointness** | Train (14), Val (3), Fresh (5) | Inter-split speaker intersection = 0 | **PASS** |
| **Recording ID Uniqueness** | All 1,468 V4 records | 1,468 unique IDs; zero cross-split duplicates | **PASS** |
| **SHA-256 Hash Uniqueness** | All 1,468 WAV files | 1,468 unique SHA-256 hashes; zero audio collisions | **PASS** |
| **Parent Lineage Tracking** | Manifest Schema | Schema lacks `parent_recording_id` and `session_id` | **NOT_VERIFIABLE** |
| **Train-Only Augmentation** | PyTorch Dataset | Gated strictly behind `is_train=True` | **PASS** |

---

## 10. Artifacts and Checkpoint Locations

- **Arm A Checkpoint:** `experiments\synthetic_kws_v4\checkpoints\arm_a_control.pt` (272,934 params, 1,095.4 KB)
- **Arm B Checkpoint:** `experiments\synthetic_kws_v4\checkpoints\arm_b_onset_no_resample.pt` (272,934 params, 1,096.3 KB)
- **Arm C Checkpoint:** `experiments\synthetic_kws_v4\checkpoints\arm_c_onset_with_resample.pt` (272,934 params, 1,096.4 KB)
- **Manifests:**
  - `experiments\synthetic_kws_v4\train_manifest_onset_aware.csv` (952 rows, 14 speakers)
  - `experiments\synthetic_kws_v4\val_manifest_onset_aware.csv` (216 rows, 3 speakers)
  - `experiments\synthetic_kws_v4\fresh_holdout_manifest_6class.csv` (300 rows, 5 unseen speakers `SPK_CALIB_031`–`035`)
- **Scripts & Results:**
  - `experiments\synthetic_kws_v4\onset_aware_generator.py`
  - `experiments\synthetic_kws_v4\build_onset_aware_data.py`
  - `experiments\synthetic_kws_v4\run_ablation.py` & `ablation_results.json`
  - `experiments\synthetic_kws_v4\generate_fresh_holdout_v4.py`
  - `experiments\synthetic_kws_v4\evaluate_v4.py` & `evaluation_results_v4.json`
- **Evaluation Report:** `experiments\synthetic_kws_v4\synthetic_kws_v4_report.md`
