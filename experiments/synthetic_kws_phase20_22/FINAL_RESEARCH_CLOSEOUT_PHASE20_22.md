# Final Research Closeout: Synthetic KWS Phases 20–22
## Final Independent Verification, Extended Robustness, and Research Release Candidate

**Workspace:** `D:\trainidentificationaudio`  
**Execution Date:** 2026-10-17  
**Candidate Identifier:** `Vietnamese_Synthetic_KWS_TCResNet14_V5_2_RC`  
**Candidate Release Status:** `SYNTHETIC_RESEARCH_CANDIDATE_WITH_LIMITATIONS`  
**Official Project Governance Status:** **`NOT_VALIDATED_ON_NATURAL_SPEECH`**  

---

## 1. Executive Summary & Acceptance Gate Verdict

| Phase | Milestone | Acceptance Criteria | Measured Verification Evidence | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 20** | Independent Re-Audit | Full 3,000-clip dataset audit, bit-exact regen, multi-seed evaluation, probe audit | **3,000 / 3,000 (100.00%)** bit-exact WAV byte regeneration in 101.0s; multi-seed 100% test acc; Wilson CI $[99.15\%, 100.00\%]$; Probes recomputed across 3 tasks | **`PHASE20_PASS_WITH_LIMITATIONS`** |
| **Phase 21** | Extended Robustness & 3-Hr Streaming | $\ge 20$ held-out profiles; $\ge 3.0$ hr target-absent stream; $\ge 200$ target events; counterfactuals | 24 profiles (288 clips, **100.00%** acc); 3.00 hr stream (**0 FP, $0.00\text{ FP/hr}$, Poisson 95% bound $\le 1.00\text{ FP/hr}$**); 200/200 targets detected (**100.00% recall, 506ms latency**); 50/50 counterfactual pairs | **`PHASE21_PASS_WITH_LIMITATIONS`** |
| **Phase 22** | Research Release Candidate | Reproducible entry points, release manifest, closeout, regression tests, hash checks | All 3 entry point scripts executed and verified with exit code 0; release manifest generated; 7 baselines + 12 golden vectors verified bit-exact; 155 regression tests passed | **`SYNTHETIC_RESEARCH_CANDIDATE_WITH_LIMITATIONS`** |

### Tripartite Status Separation
1. **Engineering Correctness:** **`VERIFIED_PASS`**  
   Manifest integrity, 3,000-sample bit-exact byte serializations, feature extraction pipelines, sliding-window streaming state machines, and random seed isolation execute with 100.00% precision.
2. **Generalization within Synthetic Generator Family:** **`VERIFIED_PASS_WITH_LIMITATIONS`**  
   The frozen TC-ResNet-14 acoustic model and dual-threshold hysteresis detector demonstrate robust generalization across 24 held-out simulated speaker profiles, 3.00 hours of continuous target-absent streaming (zero false alarms), 200 continuous keyword events (100% recall), and 50 matched counterfactual pairs.
3. **Natural-Speech Scientific Validity:** **`NOT_VALIDATED_ON_NATURAL_SPEECH`**  
   Human data collection remains strictly blocked. No natural human speech was collected, processed, or evaluated. High synthetic performance does **not** establish real-world recognition accuracy.

---

## 2. End-to-End Execution of Reproducibility Entry Points

All three documented reproducibility entry points were executed end-to-end:

| Script Path | Exit Code | Elapsed Time | Execution Mode | Measured Result |
| :--- | :--- | :--- | :--- | :--- |
| [`phase22_release/reproduce_audit.py`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_phase20_22/phase22_release/reproduce_audit.py) | **0** | **101.0 s** | Full Rerun | 3,000/3,000 bit-exact WAV byte regenerations; 100% multi-seed test accuracy; probes audited across 3 tasks |
| [`phase22_release/reproduce_robustness.py`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_phase20_22/phase22_release/reproduce_robustness.py) | **0** | **20.1 s** | Full Rerun | Synthesized 288 clips across 24 held-out profiles; evaluated frozen checkpoint: 100.00% accuracy, 0 confusion |
| [`phase22_release/reproduce_streaming.py`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_phase20_22/phase22_release/reproduce_streaming.py) | **0** | **144.4 s** | Full Rerun | Synthesized and evaluated 10,800.0s (3.00h, 107,946 windows) target-absent stream (0 FP); 200 targets (100% recall); 50/50 pairs |

---

## 3. Dataset Integrity & Exact Bit-Exact Regeneration Outcome

Independent re-audit and row-independent regeneration of the entire 3,000-sample V5.2 candidate release dataset (`experiments/synthetic_kws_phase13_3_16/dataset_release_candidate/`):
- **Manifest Rows:** 3,000
- **Actual Audio Files on Disk:** 3,000 verified in [`audio/`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_phase13_3_16/dataset_release_candidate/audio)
- **Class Balance:** Exactly 500 clips per canonical class across 6 classes (`TRAI`, `PHAI`, `LEN`, `XUONG`, `UNKNOWN`, `SILENCE`).
- **Format Integrity:** 100% 16,000 Hz mono, 1.000 s (16,000 samples), 0 NaNs/Infs, peak amplitude strictly $\le 1.0$.
- **Split Isolation:** 0 speaker overlap and 0 seed domain overlap across `train` (2,100), `val` (450), and `test` (450).
- **Row-Independent Regeneration:** Full manifest regeneration executed using `reproduce_from_manifest_row`:
  $$\text{Matches: } \mathbf{3,000 / 3,000\ (100.00\%)}\quad (\text{Elapsed: } 101.0\text{ s})$$
- **Serialization Byte Contract:** `PCM_16` WAV, 16,000 Hz mono via `soundfile.write(buf, audio, 16000, subtype='PCM_16', format='WAV')`.

---

## 4. Recomputed Static Model Metrics & Finite-Sample Uncertainty

### 4.1 Multi-Seed TC-ResNet-14 Evaluation on Frozen Test Set ($N=450$)
The three trained model runs were independently evaluated on the frozen 450-clip test set:

| Model Run | Checkpoint SHA-256 | Test Accuracy | Macro-F1 | Weighted-F1 | Brier Score | ECE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Seed 42** | `4310338165b3e29de6ae95d2628cdd9bf1c3cbdfd9364ba0b57d73843f4e9a19` | **100.00%** | **1.0000** | **1.0000** | 0.000575 | 0.001007 |
| **Seed 43** | `c196ac868034d67c7824f8ec4aa915dbdcccefe99abcea4f2469dc4032af158c` | **100.00%** | **1.0000** | **1.0000** | 0.000094 | 0.000795 |
| **Seed 44** | `4a2de65520c750290a489d2458ba59ff0129524273968408fdc7821c92b21631` | **100.00%** | **1.0000** | **1.0000** | 0.000919 | 0.002270 |

### 4.2 Finite-Sample Uncertainty Bounds
- **Observed Errors on Test Set:** 0 / 450
- **Wilson Score 95% Confidence Interval:** $[99.15\%, 100.00\%]$
- **Rule of Three Error Upper Bound (95% confidence):** $\le \frac{3}{450} = \mathbf{0.67\%}$.
- **Empirical Limitation:** All 3 seeds evaluate optimization variance on the **same 450 test clips**. The frozen test set contains only **two simulated speaker profiles** (`SPK_TEST_01_FEMALE`, `SPK_TEST_02_MALE`). Zero error on 450 synthetic clips bounds the error on this synthetic distribution to $\le 0.67\%$, but does **not** imply zero error on natural speech.

---

## 5. Resolved Nuisance-Probe Metrics & Scientific Interpretation

To rigorously investigate attribution without test-guided feature selection, linear probes were fit on `train` only (2,100 clips), validated on `val` (450 clips), and evaluated on `test` across three separate tasks with all label-bearing identifiers, transcripts, IDs, and paths strictly excluded:

### 5.1 Recomputed Probing Results Across 3 Tasks

| Task Description | Sample Counts (Train / Val / Test) | Chance / Baseline | Metadata-Only Probe | Strict Nuisance RMS Probe | Spectral / Consonant Probe | TC-ResNet-14 (Neural) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Task 1: 6-Class Classification** | 2,100 / 450 / 450 (75/class) | 16.67% (Chance) | **52.00%** (Val 60.89%) | **52.67%** (Val 54.44%) | **41.56%** (Val 42.67%) | **100.00%** |
| **Task 2: 4-Target Keyword Only** | 1,400 / 300 / 300 (75/class) | 25.00% (Chance) | **52.67%** (Val 58.67%) | **55.00%** (Val 57.00%) | **50.00%** (Val 53.67%) | **100.00%** |
| **Task 3: Binary Target-Present** | 2,100 / 450 / 450 (150 non, 300 tgt) | 66.67% (Majority) | **80.22%** (Val 76.00%) | **80.89%** (Val 79.33%) | **71.56%** (Val 74.44%) | **100.00%** |

---

### 5.2 Complete 4×4 Confusion Matrices for Task 2 (Four Target Keywords Only, $N=300$)

In Task 2, all non-target samples (`UNKNOWN`, `SILENCE`) are excluded. The probe is fit strictly on the 1,400 target clips in `train` (350 per class), hyperparameter-checked on `val` (75 per class), and evaluated on the 300 target clips in `test` (75 per class). Every true class has support $N = 75$, and every row sums to exactly 75.

#### A. Strict Nuisance RMS Probe (Features: `full_clip_rms`, `peak_amp`, `crest_factor`, `first_25ms_rms`, `pre_onset_rms`)
Test Accuracy = $\frac{34 + 39 + 69 + 23}{300} = \frac{165}{300} = \mathbf{55.00\%}$, Macro-F1 = 0.5380 (Chance = 25.00%):

| True Class | Pred TRAI | Pred PHAI | Pred LEN | Pred XUONG | Row Sum | Actual Support |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TRAI** | **34** | 16 | 5 | 20 | 75 | 75 |
| **PHAI** | 33 | **39** | 1 | 2 | 75 | 75 |
| **LEN** | 0 | 0 | **69** | 6 | 75 | 75 |
| **XUONG** | 23 | 11 | 18 | **23** | 75 | 75 |
| **Total Pred** | 90 | 66 | 93 | 51 | 300 | 300 |

*Reconciliation Analysis:*
1. **Row Sums & Support:** Each row sums to exactly 75 ($34+16+5+20 = 75$, $33+39+1+2 = 75$, $0+0+69+6 = 75$, $23+11+18+23 = 75$), matching actual per-class support.
2. **Reconciliation with 55.00% Accuracy:** The diagonal sum is $34 + 39 + 69 + 23 = 165$ correct out of 300, which reconciles to exactly $\mathbf{55.00\%}$.
3. **Class-Dependent Discrimination:**
   - The probe does not uniformly fail to discriminate all four keywords; performance is strongly class-dependent:
     - `LEN`: **69 / 75 correct (92.00%)**, demonstrating that `LEN` exhibits distinct acoustic energy characteristics (likely associated with continuous voiced sonorant energy envelopes and lower crest factor in the synthetic dataset).
     - `PHAI`: **39 / 75 correct (52.00%)**.
     - `TRAI`: **34 / 75 correct (45.33%)**.
     - `XUONG`: **23 / 75 correct (30.67%)**.
   - These variations indicate possible residual nuisance associations for certain energy envelopes, but such differences must not be interpreted as demonstrated causal acoustic mechanisms without controlled intervention studies.
4. **TRAI vs PHAI Descriptive Subset Evaluation:**
   - Across the 150 ground-truth samples of `TRAI` and `PHAI` combined, exactly $34 + 39 = 73$ samples are correctly classified, yielding a descriptive subset accuracy of $\frac{73}{150} = \mathbf{48.67\%}$ (counting predictions of `LEN` or `XUONG` as errors). This is reported strictly as a descriptive subset accuracy on the sample, rather than proof of statistical equivalence to random guessing.
   - For true `TRAI`, 16 predictions misclassify into `PHAI`, while 25 leak into `LEN`/`XUONG`. For true `PHAI`, 33 misclassify into `TRAI`, while 3 leak into `LEN`/`XUONG`.
   - **Conditional Ratio vs Full-Task Precision:** Among samples where the probe predicted `TRAI` restricted to the 150 `TRAI` and `PHAI` ground-truth samples, the conditional ratio of true `TRAI` is $\frac{34}{34 + 33} = \frac{34}{67} = \mathbf{50.75\%}$. In contrast, the full-task precision for `TRAI` across all 300 target samples is $\frac{34}{90} = \mathbf{37.78\%}$, because the column total is 90 (which includes 23 true `XUONG` samples mispredicted as `TRAI`).

#### B. Metadata-Only Probe (Features: `duration`, `onset`, `target_rms`, `bg_rms`, `gain_factor`)
Test Accuracy = $\frac{30 + 45 + 38 + 45}{300} = \frac{158}{300} = \mathbf{52.67\%}$, Macro-F1 = 0.5237 (Chance = 25.00%):

| True Class | Pred TRAI | Pred PHAI | Pred LEN | Pred XUONG | Row Sum | Actual Support |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TRAI** | **30** | 45 | 0 | 0 | 75 | 75 |
| **PHAI** | 30 | **45** | 0 | 0 | 75 | 75 |
| **LEN** | 0 | 0 | **38** | 37 | 75 | 75 |
| **XUONG** | 0 | 0 | 30 | **45** | 75 | 75 |
| **Total Pred** | 60 | 90 | 68 | 82 | 300 | 300 |

*Reconciliation Analysis:*
- **Identical Aggregate Prediction Counts:** The confusion matrix rows for true `TRAI` (`[30, 45, 0, 0]`) and true `PHAI` (`[30, 45, 0, 0]`) show identical aggregate prediction counts: both true `TRAI` and true `PHAI` receive 30 aggregate predictions for `TRAI` and 45 aggregate predictions for `PHAI` (with 0 for `LEN`/`XUONG`).
- This demonstrates that at the aggregate prediction level, the linear classifier assigns identical counts to both classes. However, this observation describes aggregate confusion matrix outputs; it does not claim identical per-sample predictions, identical underlying decision hyperplanes, or complete absence of discriminative information without sample-level proof.

#### C. Spectral / Consonant Feature Probe (Features: `spectral_centroid_hz`, `hf_energy_ratio`, `zcr`)
Test Accuracy = $\frac{54 + 38 + 3 + 55}{300} = \frac{150}{300} = \mathbf{50.00\%}$, Macro-F1 = 0.4728 (Chance = 25.00%):

| True Class | Pred TRAI | Pred PHAI | Pred LEN | Pred XUONG | Row Sum | Actual Support |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TRAI** | **54** | 4 | 16 | 1 | 75 | 75 |
| **PHAI** | 5 | **38** | 9 | 23 | 75 | 75 |
| **LEN** | 60 | 5 | **3** | 7 | 75 | 75 |
| **XUONG** | 1 | 4 | 15 | **55** | 75 | 75 |
| **Total Pred** | 120 | 51 | 43 | 86 | 300 | 300 |

*Reconciliation Analysis:*
- Spectral centroid and high-frequency energy separate `XUONG` (55 correct, high sibilant frication) and `TRAI` (54 correct, burst transient), but severely confuse `LEN` with `TRAI` (60 true `LEN` predicted as `TRAI`).
- Diagonal sum: $54 + 38 + 3 + 55 = 150 / 300 = \mathbf{50.00\%}$.

---

### 5.3 Complete Confusion Matrices for Task 1 (6-Class Classification, $N=450$, Chance = 16.67%)

All 6 classes evaluated on the frozen test set (75 clips per class, row sums = 75):

#### A. Strict Nuisance RMS Probe (6-Class Accuracy = 52.67%, Macro-F1 = 0.4872):
| True Class | Pred TRAI | Pred PHAI | Pred LEN | Pred XUONG | Pred UNKNOWN | Pred SILENCE | Row Sum |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TRAI** | **35** | 15 | 3 | 17 | 5 | 0 | 75 |
| **PHAI** | 35 | **37** | 1 | 1 | 1 | 0 | 75 |
| **LEN** | 0 | 0 | **69** | 3 | 3 | 0 | 75 |
| **XUONG** | 24 | 11 | 18 | **17** | 5 | 0 | 75 |
| **UNKNOWN** | 7 | 19 | 31 | 14 | **4** | 0 | 75 |
| **SILENCE** | 0 | 0 | 0 | 0 | 0 | **75** | 75 |
| **Total** | 101 | 82 | 122 | 52 | 18 | 75 | 450 |

*Diagonal Sum:* $35 + 37 + 69 + 17 + 4 + 75 = 237 / 450 = \mathbf{52.67\%}$.

#### B. Metadata-Only Probe (6-Class Accuracy = 52.00%, Macro-F1 = 0.5117):
| True Class | Pred TRAI | Pred PHAI | Pred LEN | Pred XUONG | Pred UNKNOWN | Pred SILENCE | Row Sum |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TRAI** | **35** | 38 | 0 | 0 | 2 | 0 | 75 |
| **PHAI** | 29 | **46** | 0 | 0 | 0 | 0 | 75 |
| **LEN** | 0 | 0 | **14** | 23 | 38 | 0 | 75 |
| **XUONG** | 0 | 0 | 16 | **23** | 36 | 0 | 75 |
| **UNKNOWN** | 0 | 0 | 11 | 23 | **41** | 0 | 75 |
| **SILENCE** | 0 | 0 | 0 | 0 | 0 | **75** | 75 |
| **Total** | 64 | 84 | 41 | 69 | 117 | 75 | 450 |

*Diagonal Sum:* $35 + 46 + 14 + 23 + 41 + 75 = 234 / 450 = \mathbf{52.00\%}$.

---

### 5.4 Complete Confusion Matrices for Task 3 (Binary Target-Present vs Target-Absent, $N=450$)

- Non-Target Support = 150 (`UNKNOWN` 75, `SILENCE` 75)
- Target Support = 300 (`TRAI` 75, `PHAI` 75, `LEN` 75, `XUONG` 75)
- Majority Baseline = $\frac{300}{450} = \mathbf{66.67\%}$, Balanced Prior Baseline = $\mathbf{50.00\%}$.

#### Strict Nuisance RMS Binary Probe:
- Accuracy = $\frac{81 + 283}{450} = \frac{364}{450} = \mathbf{80.89\%}$, Precision = 0.8040, Recall = 0.9433, F1 = 0.8681:
  - Non-Target (True False, Support 150): 81 True Negative, 69 False Positive (Sum = 150)
  - Target (True True, Support 300): 17 False Negative, 283 True Positive (Sum = 300)

#### Metadata-Only Binary Probe:
- Accuracy = $\frac{96 + 265}{450} = \frac{361}{450} = \mathbf{80.22\%}$, Precision = 0.8307, Recall = 0.8833, F1 = 0.8562:
  - Non-Target (True False, Support 150): 96 True Negative, 54 False Positive (Sum = 150)
  - Target (True True, Support 300): 35 False Negative, 265 True Positive (Sum = 300)

---

### 5.5 Scientific Attribution Conclusion
1. **Why do nuisance and metadata probes score ~52% on 6-class and ~81% on binary?**  
   Because `SILENCE` has zero active speech energy and a distinct crest factor, allowing probes to separate silence from speech with 100% precision (75/75), and target speech has calibrated active duration.
2. **Behavior of Nuisance and Metadata Probes on Target Keywords:**  
   On Task 2 (4-target keyword discrimination), the strict nuisance probe achieves an overall accuracy of **55.00%**, but performance is class-dependent (`LEN` 69/75, `PHAI` 39/75, `TRAI` 34/75, `XUONG` 23/75), reflecting possible residual nuisance associations for certain energy envelopes rather than uniform failure across all classes. On the 150 ground-truth samples of `TRAI` and `PHAI`, descriptive subset accuracy is **48.67%** (73/150). When restricted to the `TRAI` and `PHAI` subset, the conditional ratio of true `TRAI` among predicted `TRAI` is $\frac{34}{67} = \mathbf{50.75\%}$, whereas the full-task `TRAI` precision is $\frac{34}{90} = \mathbf{37.78\%}$ (column total = 90). The metadata probe outputs identical aggregate prediction counts (`[30, 45, 0, 0]`) for true `TRAI` and true `PHAI`.
3. **Neural Attribution & Measured Representation vs Interpretation:**  
   In contrast, TC-ResNet-14 achieves **100.00%** on both 6-class and 4-target evaluations ($N=450$ and $N=300$). This measured result demonstrates that the learned representations support strong class separation across the tested synthetic distributions. However, this empirical finding must not be interpreted as proving genuine Vietnamese phonetic recognition or establishing formant trajectories as the proven causal basis for classification, as controlled causal phonetic interventions were not conducted. Furthermore, because all training and evaluation audio originated from parametric synthetic vocal tract models, this attribution applies **strictly within the synthetic synthesizer domain** and does not establish validity on natural human speech.

---

## 6. Extended Held-Out Synthetic Profile Performance

In Phase 21, 24 newly sampled held-out speaker profiles (`SPK_HELD_01` to `SPK_HELD_24`, spanning VTL 0.82 to 1.18 and F0 92 to 255 Hz) were evaluated without model retraining:
- **Clips Evaluated:** 288 (12 per profile across 6 classes)
- **Manifest:** [`phase21_heldout_manifest.csv`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_phase20_22/phase21_robustness/phase21_heldout_manifest.csv)
- **Accuracy:** **100.00%** (288 / 288)
- **Macro-F1:** **1.0000**
- **Wilson Score 95% Confidence Interval:** $[98.68\%, 100.00\%]$
- **Rule of Three Error Bound:** $\le \frac{3}{288} = 1.04\%$
- **Cross-Keyword Confusion:** 0 instances of `TRAI` $\leftrightarrow$ `PHAI` confusion.
- **Non-Target False Alarms:** 0 `UNKNOWN` $\rightarrow$ Target, 0 `SILENCE` $\rightarrow$ Target.

---

## 7. Extended Target-Absent Streaming Benchmark (3.00 Hours)

- **Total Evaluated Duration:** **10,800.0 seconds (3.00 Hours)** across 6 independent streams.
- **Sliding Windows Evaluated:** 107,946 windows ($W = 1.0\text{ s}$, $H = 100\text{ ms}$).
- **Observed False Triggers:** **0**
- **False-Trigger Rate:** **0.00 False Triggers / Hour**
- **CSV Artifact:** [`stream_level_target_absent.csv`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_phase20_22/phase21_robustness/stream_level_target_absent.csv)
- **Statistical Upper Bound:** Under Poisson assumptions, zero observed events over $T = 3.00\text{ h}$ yields a one-sided 95% upper bound of $\lambda_{95\%} \le \frac{3.0}{3.00} = \mathbf{1.00\text{ FP/hr}}$.

---

## 8. Continuous Target-Present Keyword Event Detection (200 Events)

- **Total Ground Truth Target Events:** **200** across 8 independent continuous streams (25 targets per stream).
- **Class Balance:** 56 `TRAI`, 48 `PHAI`, 48 `LEN`, 48 `XUONG`.
- **Target Event Recall:** **100.00%** (200 / 200 detected).
- **Target Event Precision:** **100.00%** (0 false activations on inter-target gaps).
- **Missed Targets / Duplicate Activations:** 0 (duplicate firings prevented by $600\text{ ms}$ refractory cooldown lock).
- **Detection Latency Distribution ($N=200$):**
  - **Mean:** $0.506\text{ s}$
  - **Median:** $0.503\text{ s}$
  - **90th Percentile:** $0.600\text{ s}$
  - **95th Percentile:** $0.636\text{ s}$
  - **Standard Deviation:** $0.081\text{ s}$
- **Per-Keyword Recall:** `TRAI` (100%), `PHAI` (100%), `LEN` (100%), `XUONG` (100%).
- **CSV Artifacts:**
  - Stream-level: [`stream_level_target_present.csv`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_phase20_22/phase21_robustness/stream_level_target_present.csv)
  - Event-level: [`event_level_target_present.csv`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_phase20_22/phase21_robustness/event_level_target_present.csv)

---

## 9. Factorial Counterfactual Study (TRAI vs PHAI)

- **Matched Pairs Evaluated:** 50 pairs holding speaker profile, VTL, F0, duration, onset, noise type, and gain factor strictly identical.
- **Correctly Distinguished Pairs:** **50 / 50 (100.00%)**
- **Mean Model Confidence:**
  - `TRAI` predictions: 0.9984
  - `PHAI` predictions: 0.9991
- **CSV Artifact:** [`counterfactual_trai_phai_results.csv`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_phase20_22/phase21_robustness/counterfactual_trai_phai_results.csv)

---

## 10. Frozen Streaming Detector Specification & Provenance

Established in Phase 18 and validated across all Phase 20–22 evaluations:
- **Algorithm:** Dual-threshold moving-average hysteresis state machine with refractory cooldown.
- **Activation Threshold ($\tau_{on}$):** `0.80`
- **Deactivation Threshold ($\tau_{off}$):** `0.35`
- **Moving Average Smoothing Window:** `3 frames`
- **Minimum Consecutive Frames:** `2 frames`
- **Refractory Cooldown Period:** `0.600 s` (6 frames)
- **Analysis Window ($W$):** `16,000 samples` (1.000 s)
- **Hop Size ($H$):** `1,600 samples` (0.100 s)
- **Causal Decision Timestamp:** $t_{decision} = t_{win\_start} + 1.000\text{ s}$

---

## 11. Final Regression Coverage & Integrity Verification

### 11.1 Test Suite Results
All test suites in `tests/` were executed and verified:

| Test File | Command | Test Count | Pass / Fail | Elapsed Time |
| :--- | :--- | :--- | :--- | :--- |
| `test_synthetic_kws_v5_generator.py` | `pytest tests/test_synthetic_kws_v5_generator.py` | 17 | **17 Passed, 0 Failed** | 1.51 s |
| `test_synthetic_kws_v5_1_candidate.py` | `pytest tests/test_synthetic_kws_v5_1_candidate.py` | 12 | **12 Passed, 0 Failed** | 1.60 s |
| `test_synthetic_kws_v5_2_candidate.py` | `pytest tests/test_synthetic_kws_v5_2_candidate.py` | 5 | **5 Passed, 0 Failed** | 2.05 s |
| `test_synthetic_kws_demo.py` | `pytest tests/test_synthetic_kws_demo.py` | 19 | **19 Passed, 0 Failed** | 2.54 s |
| `test_kws_export_parity.py` | `pytest tests/test_kws_export_parity.py` | 13 | **13 Passed, 0 Failed** | 3.82 s |
| `test_campaign_acceptance.py` | `pytest tests/test_campaign_acceptance.py` | 69 | **69 Passed, 0 Failed** | 28.21 s |
| `test_parity.py` | `pytest tests/test_parity.py` | 7 | **7 Passed, 0 Failed** | 2.45 s |
| `test_field_acquisition.py` | `pytest tests/test_field_acquisition.py` | 13 | **13 Passed, 0 Failed** | 5.14 s |
| **Total Test Coverage** | — | **155** | **155 Passed, 0 Failed** | **47.32 s** |

### 11.2 Protected Historical Baselines (7/7 Verified Bit-Exact)
- `data/manifests/test_manifest_v2.csv`: `9E3904C8EA427244F7280A41140D7EFD19FD9625B359290ADB7ADCD4FA2E1C13` (OK)
- `data/raw/field_recordings/field_collection_manifest.csv`: `B389282F9853A74484B39356EE85E5BD7E74C0C46481300524C7BD2894AA8C85` (OK)
- `data/raw/field_recordings/consent_ledger.json`: `BE4BFDD7CEC0C87A1DC97455E5783054EF8093C638279C44940C896F5BDA2BA0` (OK)
- `experiments/synthetic_kws_v4/checkpoints/arm_b_onset_no_resample.pt`: `6A0FD55CD6F16F5B683951127076EEC72506C31EE670F896AAEACE5981CF1D96` (OK)
- `experiments/synthetic_kws_demo/export/arm_b_tcresnet14.onnx`: `8A5AC4C105DF52F1749AE11B6BB2124B3A672486601EFF308ACC7E257FF97625` (OK)
- `experiments/synthetic_kws_demo/android_app/app/src/main/assets/models/arm_b_tcresnet14.onnx`: `8A5AC4C105DF52F1749AE11B6BB2124B3A672486601EFF308ACC7E257FF97625` (OK)
- `experiments/synthetic_kws_demo/frontend_golden/golden_manifest.json`: `5904A5503CE5528ECF88A4A670C6726D80B74285B34EE6B910CF2FA48F60E5C3` (OK)

### 11.3 Frontend Golden Test Vectors (12/12 Cases Bit-Exact)
All 12 golden WAVs and 12 golden `.npy` feature tensors in `experiments/synthetic_kws_demo/frontend_golden/` match checksums bit-exact.

### 11.4 Git Working Tree Hygiene
- `git diff --cached`: **Strictly empty** (0 staged files).
- `git diff --check`: Clean (no formatting or merge errors).
- Tracked modifications: 13 pre-existing user files preserved untouched.
- Human data governance: Zero unapproved bytes or field recordings processed.

---

## 12. Scientific Limitations & Governance Decision Gate

1. **Synthetic-to-Real Domain Gap:**  
   The observed 100.00% accuracy, zero streaming false triggers, and 100% event recall demonstrate discriminability within the parametric synthesizer. They do **not** imply that the model will achieve zero error or low false triggers on natural human speech.
2. **Dialect and Accent Variation:**  
   Synthesis implements Northern Vietnamese phonetics. Central and Southern Vietnamese phonological variations (such as merging /ʈ/ and /tʃ/ or tonal glottalization shifts) are not simulated.
3. **Conversational Ambient Acoustic Gaps:**  
   Streaming audio used synthesized ambient noise types (room tone, low-frequency hum, dither, digital zero). Real-world acoustic environments present non-stationary multi-talker babble, television speech, reverberation, and dynamic mic clipping.
4. **Mandatory Governance Stance:**  
   The release candidate is designated **`SYNTHETIC_RESEARCH_CANDIDATE_WITH_LIMITATIONS`**. The official project status remains **`NOT_VALIDATED_ON_NATURAL_SPEECH`**. Before any natural human speech may be ingested or evaluated, formal institutional approvals, protocol sign-off, and verified consent records are required.
