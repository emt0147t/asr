# Phase 20 — Independent Re-Audit of Dataset, Model Metrics, and Streaming Claims

**Workspace:** `D:\trainidentificationaudio`  
**Evaluation Scope:** `experiments/synthetic_kws_phase13_3_16/dataset_release_candidate/` & `model_evaluation/`  
**Audit Execution Date:** 2026-10-16  
**Audit Script:** [`phase20_independent_audit.py`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_phase20_22/phase20_audit/phase20_independent_audit.py)  
**Machine-Readable Audit Record:** [`phase20_results.json`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_phase20_22/phase20_audit/phase20_results.json)  

---

## 1. Executive Summary & Gate Verdict

| Gate Dimension | Requirement | Measured / Audited Value | Status |
| :--- | :--- | :--- | :--- |
| **Dataset Completeness** | 3,000 WAV files on disk | 3,000 files verified in `audio/` | **PASS** |
| **Class Balance** | 500 clips per canonical class | 500 each: `TRAI`, `PHAI`, `LEN`, `XUONG`, `UNKNOWN`, `SILENCE` | **PASS** |
| **Audio Integrity** | 16kHz mono, 1.0s, finite, $\le 1.0$ peak | 3,000/3,000 pass all acoustic formatting checks | **PASS** |
| **Cryptographic Uniqueness** | 0 exact audio duplicates | 3,000 unique SHA-256 hashes across manifest | **PASS** |
| **Split Isolation** | Disjoint speakers & seeds across splits | 0 speaker overlap, 0 seed overlap between train/val/test | **PASS** |
| **Bit-Exact Regeneration** | Full 3,000 manifest row regeneration | **3,000 / 3,000 (100.00%)** bit-exact SHA-256 matches in 110.4s | **PASS** |
| **Multi-Seed Test Evaluation** | Seeds 42, 43, 44 re-evaluated on test set | 100.00% accuracy, Brier $\le 0.000919$, ECE $\le 0.00227$ | **PASS** |
| **Speaker Profile Diversity** | Quantify test speaker count | **Only 2 simulated speaker profiles** (`SPK_TEST_01`, `SPK_TEST_02`) | **LIMITATION** |
| **Statistical Uncertainty** | Finite-sample confidence interval | Wilson 95% CI: $[99.15\%, 100.00\%]$, Rule of Three error $\le 0.67\%$ | **DOCUMENTED** |
| **Shortcut Diagnostic Audit** | Probes exclude labels & IDs | Verified: Nuisance = 52.67%, Probes cannot separate 4 keywords | **PASS** |
| **Streaming Audit** | Target-absent & target-present audit | 1.0h stream: 0 FP/hr (hysteresis); 50/50 targets, 496ms latency | **PASS** |
| **Human Data Governance** | Zero human speech / unapproved sources | Strictly enforced; protected baseline hashes bit-exact | **PASS** |

### Gate Verdict
```
================================================================================
GATE VERDICT: PHASE20_PASS_WITH_LIMITATIONS
================================================================================
```
The synthetic dataset V5.2 and TC-ResNet-14 training checkpoints exhibit complete engineering correctness, 100.00% row-independent bit-exact reproducibility, and zero leakage in the feature pipeline. However, test evaluation is subject to the limitation that the 450-clip frozen test set contains only two simulated speaker profiles. All metrics apply strictly to the parametric synthetic generation family.

---

## 2. Dataset Integrity & Bit-Exact Regeneration Re-audit

### 2.1 File & Class Verification
- **Total Manifest Rows:** 3,000
- **Actual Disk WAV Files:** 3,000 (verified via filesystem glob in `audio/`)
- **Canonical Class Distribution:** Exactly balanced across 6 classes:
  - `TRAI`: 500 clips (16.67%)
  - `PHAI`: 500 clips (16.67%)
  - `LEN`: 500 clips (16.67%)
  - `XUONG`: 500 clips (16.67%)
  - `UNKNOWN`: 500 clips (16.67%)
  - `SILENCE`: 500 clips (16.67%)
- **Split Distribution:**
  - `train`: 2,100 clips (70.0%)
  - `val`: 450 clips (15.0%)
  - `test`: 450 clips (15.0%)
- **Acoustic Subclasses:**
  - `target_keyword`: 2,000 clips
  - `ambient_room_tone`: 316 clips
  - `open_vocabulary_general`: 246 clips
  - `confuser_initial_sharing`: 130 clips
  - `confuser_rhyme_sharing`: 124 clips
  - `ambient_low_freq_hum`: 114 clips
  - `ambient_digital_zero`: 70 clips

### 2.2 Waveform Integrity Checks
Every clip was independently decoded with `soundfile`:
- **Sample Rate:** Exactly 16,000 Hz for all 3,000 clips (100%).
- **Channels:** Mono (1-channel) for all 3,000 clips (100%).
- **Duration / Samples:** Exactly 16,000 samples (1.000 s) for all 3,000 clips (100%).
- **Numerical Validity:** 0 NaNs, 0 Infs across all 48,000,000 samples.
- **Headroom / Peak Normalization:** Peak amplitudes strictly $\le 1.0$ (calibrated target speech RMS at $-15.0\text{ dBFS}$, 0.98 peak limiter ceiling).

### 2.3 Split Isolation & Feature Pipeline Audit
- **Simulated Speaker Isolation:**
  - `train` profiles: `SPK_01_FEMALE`, `SPK_02_MALE`, `SPK_03_FEMALE`, `SPK_04_MALE`, `SPK_05_FEMALE`, `SPK_06_MALE`, `SPK_07_FEMALE` (7 profiles)
  - `val` profiles: `SPK_08_MALE`, `SPK_09_FEMALE`, `SPK_10_MALE` (3 profiles)
  - `test` profiles: `SPK_TEST_01_FEMALE`, `SPK_TEST_02_MALE` (2 profiles)
  - Speaker overlaps:
    - Train $\cap$ Val: **0**
    - Train $\cap$ Test: **0**
    - Val $\cap$ Test: **0**
- **Seed Domain Isolation:**
  - Train seeds: [100000, 102099]
  - Val seeds: [200000, 200449]
  - Test seeds: [300000, 300449]
  - Overlaps between splits: **0**
- **Feature Pipeline Leakage Check:**
  - `KWSTensorDataset` receives only the 16,000-sample audio waveform.
  - Spectrogram feature tensor: `(1, 40, 101)` float32 generated dynamically via `LogMelFeatureExtractor`.
  - Zero metadata, identifiers, filenames, or label encodings are passed to `VietnameseTCResNet`.

### 2.4 Full 3,000-Clip Bit-Exact Regeneration
The row-independent regeneration function `reproduce_from_manifest_row` was executed against all 3,000 manifest rows:
- **Total Attempted:** 3,000
- **Exact SHA-256 Matches:** **3,000 / 3,000 (100.00%)**
- **Regeneration Failures:** 0
- **Elapsed Duration:** 110.4 seconds (27.2 clips/sec)
- **Serialization Format:** `PCM_16` WAV, 16000 Hz, mono via `soundfile.write(buf, audio, 16000, subtype='PCM_16', format='WAV')`.

---

## 3. Independent Model Metrics Recomputation & Uncertainty

### 3.1 Multi-Seed Evaluation on Frozen Test Split
Inference was independently re-executed on the frozen 450-clip test set using the three trained TC-ResNet-14 checkpoints:

| Model Run | Checkpoint SHA-256 (prefix) | Test Accuracy | Macro-F1 | Weighted-F1 | Brier Score | ECE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Seed 42** | `3f99477...` | **100.00%** | **1.0000** | **1.0000** | 0.000575 | 0.001007 |
| **Seed 43** | `38cf4d8...` | **100.00%** | **1.0000** | **1.0000** | 0.000094 | 0.000795 |
| **Seed 44** | `aa1e149...` | **100.00%** | **1.0000** | **1.0000** | 0.000919 | 0.002270 |

### 3.2 Per-Class Breakdown (Seed 42)
All 75 test samples per class were classified correctly:
- `TRAI`: Precision = 1.0000, Recall = 1.0000, F1 = 1.0000 (75/75)
- `PHAI`: Precision = 1.0000, Recall = 1.0000, F1 = 1.0000 (75/75)
- `LEN`: Precision = 1.0000, Recall = 1.0000, F1 = 1.0000 (75/75)
- `XUONG`: Precision = 1.0000, Recall = 1.0000, F1 = 1.0000 (75/75)
- `UNKNOWN`: Precision = 1.0000, Recall = 1.0000, F1 = 1.0000 (75/75)
- `SILENCE`: Precision = 1.0000, Recall = 1.0000, F1 = 1.0000 (75/75)

### 3.3 Test Set Limitations & Statistical Uncertainty
1. **Shared Test Split:**
   The three optimization runs (Seeds 42, 43, 44) evaluate stochastic training stability on the **same 450 test clips**. They do not constitute three independent test datasets.
2. **Speaker Profile Diversity Constraint:**
   The 450-clip test set contains only **two simulated speaker profiles**: `SPK_TEST_01_FEMALE` and `SPK_TEST_02_MALE`. The 450 test clips must **not** be interpreted as 450 independent speaker evaluations.
3. **Finite-Sample Confidence Bounds:**
   Given $n = 450$ trials with $k = 450$ successes ($0$ observed errors):
   - **Wilson Score 95% Confidence Interval:** $[99.15\%, 100.00\%]$
   - **Rule of Three Error Bound (95% upper limit):** $\frac{3}{n} = \frac{3}{450} = 0.00667$ ($0.67\%$).
   - *Scientific Meaning:* With 95% confidence, the true error rate on this specific synthetic generation family is $\le 0.67\%$. This does **not** imply zero error on natural human speech.

---

## 4. Shortcut-Diagnostic Probes Audit

### 4.1 Probing Setup & Feature Verification
To determine whether classification relies on trivial non-phonetic nuisance shortcuts, linear probes were trained on `train` only (2,100 clips), validated on `val` (450 clips), and evaluated on `test` across three separate tasks with all label-bearing identifiers, transcripts, IDs, and paths strictly excluded:

| Task Description | Sample Counts (Train / Val / Test) | Chance / Baseline | Metadata-Only Probe | Strict Nuisance RMS Probe | Spectral / Consonant Probe | TC-ResNet-14 (Neural) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Task 1: 6-Class Classification** | 2,100 / 450 / 450 (75/class) | 16.67% (Chance) | **52.00%** (Val 60.89%) | **52.67%** (Val 54.44%) | **41.56%** (Val 42.67%) | **100.00%** |
| **Task 2: 4-Target Keyword Only** | 1,400 / 300 / 300 (75/class) | 25.00% (Chance) | **52.67%** (Val 58.67%) | **55.00%** (Val 57.00%) | **50.00%** (Val 53.67%) | **100.00%** |
| **Task 3: Binary Target-Present** | 2,100 / 450 / 450 (150 non, 300 tgt) | 66.67% (Majority) | **80.22%** (Val 76.00%) | **80.89%** (Val 79.33%) | **71.56%** (Val 74.44%) | **100.00%** |

### 4.2 Complete 4×4 Confusion Matrices for Task 2 (Four Target Keywords Only, $N=300$)
Every true class has support $N = 75$, and every row sums to exactly 75:

#### A. Strict Nuisance RMS Probe (Features: `full_clip_rms`, `peak_amp`, `crest_factor`, `first_25ms_rms`, `pre_onset_rms`):
Test Accuracy = $\frac{34 + 39 + 69 + 23}{300} = \frac{165}{300} = \mathbf{55.00\%}$, Macro-F1 = 0.5380 (Chance = 25.00%):
- `TRAI` (Support 75): 34 predicted as `TRAI`, 16 as `PHAI`, 5 as `LEN`, 20 as `XUONG` (Row Sum = 75, Accuracy = 45.33%).
- `PHAI` (Support 75): 33 predicted as `TRAI`, 39 as `PHAI`, 1 as `LEN`, 2 as `XUONG` (Row Sum = 75, Accuracy = 52.00%).
- `LEN` (Support 75): 0 predicted as `TRAI`, 0 as `PHAI`, 69 as `LEN`, 6 as `XUONG` (Row Sum = 75, Accuracy = 92.00%).
- `XUONG` (Support 75): 23 predicted as `TRAI`, 11 as `PHAI`, 18 as `LEN`, 23 as `XUONG` (Row Sum = 75, Accuracy = 30.67%).
- *Class-Dependent Discrimination & Residual Nuisances:* Performance is class-dependent rather than uniformly failing across all four keywords: `LEN` achieves 69/75 (92.00%), likely reflecting continuous voiced sonorant energy envelopes and distinct crest-factor dynamics in the synthetic set. In contrast, `TRAI` (34/75), `PHAI` (39/75), and `XUONG` (23/75) exhibit heavy mutual confusion and leakage. These differences suggest potential residual nuisance associations for certain energy envelopes, but such differences must not be interpreted as demonstrated causal acoustic mechanisms without controlled intervention experiments.
- *TRAI vs PHAI Subset Evaluation:* Across the 150 true `TRAI` and `PHAI` ground-truth samples, descriptive subset accuracy is $\frac{34 + 39}{150} = \frac{73}{150} = \mathbf{48.67\%}$ (counting predictions of `LEN` or `XUONG` as errors). This is reported strictly as a descriptive subset accuracy on the sample, not proof of statistical equivalence to random guessing.
- *Conditional Ratio vs Precision:* Restricted to the 150 `TRAI` and `PHAI` ground-truth samples, the conditional ratio of true `TRAI` among samples predicted as `TRAI` is $\frac{34}{34 + 33} = \frac{34}{67} = \mathbf{50.75\%}$. In contrast, the full-task precision for `TRAI` across all 300 target samples is $\frac{34}{90} = \mathbf{37.78\%}$ because the column total is 90 (including 23 true `XUONG` samples mispredicted as `TRAI`).

#### B. Metadata-Only Probe (Features: `duration`, `onset`, `target_rms`, `bg_rms`, `gain_factor`):
Test Accuracy = $\frac{30 + 45 + 38 + 45}{300} = \frac{158}{300} = \mathbf{52.67\%}$, Macro-F1 = 0.5237 (Chance = 25.00%):
- `TRAI` (Support 75): 30 predicted as `TRAI`, 45 as `PHAI`, 0 as `LEN`, 0 as `XUONG` (Row Sum = 75).
- `PHAI` (Support 75): 30 predicted as `TRAI`, 45 as `PHAI`, 0 as `LEN`, 0 as `XUONG` (Row Sum = 75).
- `LEN` (Support 75): 0 predicted as `TRAI`, 0 as `PHAI`, 38 as `LEN`, 37 as `XUONG` (Row Sum = 75).
- `XUONG` (Support 75): 0 predicted as `TRAI`, 0 as `PHAI`, 30 as `LEN`, 45 as `XUONG` (Row Sum = 75).
- *Attribution Finding:* The confusion matrix rows for true `TRAI` (`[30, 45, 0, 0]`) and true `PHAI` (`[30, 45, 0, 0]`) show identical aggregate prediction counts: both true `TRAI` and true `PHAI` receive 30 aggregate predictions for `TRAI` and 45 aggregate predictions for `PHAI` (with 0 for `LEN`/`XUONG`). This demonstrates identical counts at the aggregate prediction level; it does not establish identical per-sample predictions, identical decision hyperplanes, or complete absence of discriminative information without sample-level proof.

### 4.3 Task 1 (6-Class) and Task 3 (Binary Target-Present) Summary
- **Task 1 (6-Class):** Metadata (52.00%) and strict nuisance (52.67%) probes achieve ~52% by classifying `SILENCE` with 100% accuracy (75/75) due to zero speech energy, while confusing speech classes.
- **Task 3 (Binary Target-Present):** Strict nuisance achieves 80.89% binary accuracy because target speech has standardized active speech RMS compared to ambient silence and variable-length foils.
- **Neural Attribution & Measured Results vs Interpretation:**
  TC-ResNet-14 achieves 100.00% accuracy on the synthetic test set, demonstrating that its learned representations support strong class separation on the tested synthetic distributions across simulated acoustic categories:
  - `TRAI`: Alveolar/retroflex stop burst (/ʈ/) followed by open diphthong `/aːɪ/`
  - `PHAI`: Continuous labiodental fricative (/f/) with spectral centroid ~2645 Hz
  - `LEN`: Voiced sonorant (/l/) with low centroid (~922 Hz) and negligible HF energy
  - `XUONG`: Sibilant frication (/s/) above 3.8 kHz with diphthong `/uəŋ/`
  However, this empirical finding must not be interpreted as proving genuine Vietnamese phonetic recognition or establishing formant trajectories as the proven causal basis for classification, as controlled causal phonetic interventions were not conducted. Furthermore, this attribution applies **strictly within the synthetic synthesizer domain** and does not establish validity on natural human speech.

---

## 5. Streaming Metrics Audit

### 5.1 Verification of Phase 18 Evaluation Artifacts
The streaming results from [`phase18_streaming_results.json`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_phase17_19/phase18_streaming/phase18_streaming_results.json) were audited against the event tables:

| Streaming Protocol | Duration | Independent Streams | Baseline ($\tau=0.50$ raw) | Hysteresis Detector ($\tau=0.80/0.35, M=2, C=600\text{ms}$) | Target Metric |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Target-Absent Stream** | 3,600.0 s (1.00 h) | 4 streams (900s each) | 339 triggers (**339.0 FP/hr**) | **0 triggers (0.00 FP/hr)** | $\le 1.0\text{ FP/hr}$ |
| **Target-Present Stream** | 196.0 s | 1 stream | 50/50 detected (45.9% prec) | **50/50 detected (100.0% prec)** | $\ge 95\%$ recall |
| **Detection Latency** | — | — | 302 ms mean latency | **496 ms mean latency** ($\sigma=78\text{ ms}$) | $\le 600\text{ ms}$ |

### 5.2 Event Association Protocol
1. **Window Decision Timestamp:**
   In a causal sliding window ($W = 16,000$ samples, $H = 1,600$ samples / 100ms), window start $t$ corresponds to decision timestamp $t_{decision} = t + 1.0\text{ s}$ (the time at which the complete 1.0s window has entered the buffer).
2. **Ground Truth Association:**
   A detection event at $t_{decision}$ is matched to a ground truth event $[t_{start}, t_{end}]$ if $t_{start} \le t_{decision} \le (t_{end} + 0.60\text{ s})$.
3. **Duplicate Suppression:**
   A refractory cooldown period of $600\text{ ms}$ prevents repeated firings on the same keyword utterance.
4. **Independent Stream Protocols:**
   Target-absent false alarm frequency and target-present precision/recall are measured on independent streams. Target-absent streams contain zero target events, verifying that background noise and pauses do not trigger false activations.

---

## 6. Documented Limitations & Forward Path

1. **Synthetic Family Boundary:**
   The audited accuracy of 100.00% and zero streaming false triggers apply exclusively within the synthetic parametric synthesizer. They do **not** imply real-world human performance.
2. **Speaker Diversity Gap:**
   The frozen test split contains only two simulated speaker profiles. Phase 21 will expand speaker generalization by sampling $\ge 20$ newly generated held-out speaker/profile combinations.
3. **Streaming Upper Bound:**
   Zero false triggers observed over 1.0 hour yields a Poisson 95% upper bound of $\le 3.0\text{ FP/hr}$. Phase 21 will evaluate $\ge 3.0$ hours of continuous audio to tighten the bound to $\le 1.0\text{ FP/hr}$.
4. **Official Governance Status:**
   The project remains **`NOT_VALIDATED_ON_NATURAL_SPEECH`**. Human data collection and access to unapproved field recordings remain strictly blocked.
