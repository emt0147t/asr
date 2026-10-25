# Integrated Research Closeout: Synthetic KWS Phases 13.3–16.0
## Scientific Audit → Dataset Release Candidate → Model Training → Robustness Evaluation

**Target Directory:** `experiments/synthetic_kws_phase13_3_16/`  
**Date:** 2026-10-16  
**Auditor:** Antigravity (Advanced Agentic Coding Pair Programmer)  
**Overall Project Status:**
* **Engineering Correctness:** **`VERIFIED_PASS`**
* **Synthetic Distribution Generalization:** **`VERIFIED_PASS_WITH_LIMITATIONS`**
* **Natural-Speech Validity:** **`NOT_VALIDATED_ON_NATURAL_SPEECH`**
* **Governance Invariant:** **`HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED`**

---

## 1. Executive Summary and Phase Verdicts

This program executed an end-to-end scientific progression across Phases 13.3, 14, 15, and 16. All gates were strictly enforced without manufacturing passing results:

| Phase | Title | Gate Verdict | Key Deliverable |
| :--- | :--- | :---: | :--- |
| **Phase 13.3** | Independent V5.1 Scientific Audit & Residual RMS Investigation | **`AUDIT_PASS_WITH_LIMITATIONS`** | [`phase13_3_audit/PHASE13_3_AUDIT.md`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_phase13_3_16/phase13_3_audit/PHASE13_3_AUDIT.md) |
| **Phase 14** | Reproducible Synthetic Dataset Release Candidate (3,000 Clips) | **`DATASET_PASS`** | [`dataset_release_candidate/DATASET_STATISTICS_REPORT.md`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_phase13_3_16/dataset_release_candidate/DATASET_STATISTICS_REPORT.md) |
| **Phase 15** | Controlled TC-ResNet-14 Training and Multi-Seed Evaluation | **`TRAINING_EVALUATION_PASS`** | [`model_evaluation/MODEL_EVALUATION_REPORT.md`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_phase13_3_16/model_evaluation/MODEL_EVALUATION_REPORT.md) |
| **Phase 16** | Robustness Stress Testing & Final Research Closeout | **`ROBUSTNESS_REPRODUCIBILITY_PASS_WITH_LIMITATIONS`** | [`FINAL_RESEARCH_CLOSEOUT.md`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_phase13_3_16/FINAL_RESEARCH_CLOSEOUT.md) |

---

## 2. Work Implemented by Phase

### Phase 13.3: Independent Scientific Audit & Remediation
- Recomputed per-clip features for all 120 V5 clips and 120 V5.1 clips using independent signal processing.
- Re-verified effect sizes: full-clip RMS $\eta^2$ dropped from $0.6042$ to $0.1215$, and pre-onset background noise $\eta^2$ dropped from $0.1806$ to $0.0282$.
- Discovered physical cause of the residual RMS variance: V5.1's soft-knee peak limiter threshold ($0.92$) slightly compressed unvoiced fricative peaks in `PHAI` ($16.7\text{ dB}$ crest factor), lowering its active RMS by $0.65\text{ dB}$.
- Developed isolated **V5.2 generator** (`experiments/synthetic_kws_v5_2_candidate/`) with calibrated speech level ($-15.0\text{ dBFS}$) and $0.98$ peak headroom, eliminating the fricative compression and matching full-clip RMS across words within **$0.00043$ ($< 0.03\text{ dB}$)**.

### Phase 14: Release Candidate Dataset (3,000 Clips)
- Generated full-scale 3,000-clip balanced corpus (`synthetic_kws_v5_2_release_candidate`):
  - 500 clips per class (`TRAI`, `PHAI`, `LEN`, `XUONG`, `UNKNOWN`, `SILENCE`).
  - 70/15/15 partition: 2,100 Train, 450 Validation, 450 Test.
- Strictly isolated splits: 12 disjoint simulated speaker profiles (8 Train, 2 Val, 2 Test) and completely disjoint seed domains.
- Validated all 3,000 files on disk: 100% match manifest SHA-256, 16kHz mono, 1.0s, 0 clipping, 0 NaNs/Infs.
- Row-independent bit-exact reproduction confirmed on 120-clip audit sample ($100.0\%$).

### Phase 15: Controlled Multi-Seed Model Training
- Trained production `VietnameseTCResNet` (TC-ResNet-14, 192 features) across 3 independent seeds (42, 43, 44) on standard Log-Mel filterbanks.
- Checkpoints selected strictly by validation loss without test set exposure.
- Evaluated on 450 held-out test clips: **$100.00\% \pm 0.00\%$** test accuracy, $100.00\%$ Macro-F1.
- Evaluated continuous streaming false alarms on a 186.5s target-absent stream: measured **$424.57\text{ false triggers / hour}$** at $\tau = 0.500$, establishing that sliding-window boundaries trigger false alarms without multi-frame temporal smoothing.

### Phase 16: Synthetic Domain-Shift Robustness & Closeout
- Evaluated trained model under 5 out-of-distribution synthetic stress regimes:
  - Nominal Baseline: $100.00\%$
  - Extreme Speaker Anatomy Shift (VTL $0.75 - 1.25$, $F_0 = 80 - 310\text{ Hz}$): $84.17\%$
  - Adverse Low-SNR Shift ($-24\text{ dBFS}$ noise, SNR $6 - 10\text{ dB}$): $87.50\%$
  - Extreme Temporal Shift (Onset $0.010\text{ s}$ or $0.420\text{ s}$, duration $0.250\text{ s}$): $93.33\%$
  - Non-Stationary Hum & Harmonics: $100.00\%$
- Verified all 7 cryptographic repository baselines and 12 golden test vectors bit-exact.
- Executed full test suite: **135 passed, 0 failed** in 31.22s.
- Verified clean Git status: `git diff --cached` remains **strictly empty**.

---

## 3. Dataset Specifications and Split Isolation

### Corpus Summary
* **Dataset Name:** `synthetic_kws_v5_2_release_candidate`
* **Root Location:** [`experiments/synthetic_kws_phase13_3_16/dataset_release_candidate/`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_phase13_3_16/dataset_release_candidate/)
* **Audio Format:** 16,000 Hz, 1 channel (mono), 16-bit PCM, exactly 16,000 samples (1.000 s).
* **Provenance:** 100% parametric physical synthesis (Rosenberg glottal pulse source + cascaded formant resonators).

### Class and Split Balance Table

| Class Label | Train (70%) | Val (15%) | Test (15%) | Total Corpus | Acoustic Subclasses |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`TRAI`** | 350 | 75 | 75 | **500** | `target_keyword` |
| **`PHAI`** | 350 | 75 | 75 | **500** | `target_keyword` |
| **`LEN`** | 350 | 75 | 75 | **500** | `target_keyword` |
| **`XUONG`** | 350 | 75 | 75 | **500** | `target_keyword` |
| **`UNKNOWN`** | 350 | 75 | 75 | **500** | `open_vocabulary_general` (246), `confuser_initial_sharing` (130), `confuser_rhyme_sharing` (124) |
| **`SILENCE`** | 350 | 75 | 75 | **500** | `ambient_room_tone` (316), `ambient_low_freq_hum` (114), `ambient_digital_zero` (70) |
| **Total** | **2,100** | **450** | **450** | **3,000** | **7 distinct subclasses** |

### Split Isolation Proof
* **Train Speakers (8 profiles):** `SPK_TR_01` to `SPK_TR_08`
* **Validation Speakers (2 profiles):** `SPK_VAL_01`, `SPK_VAL_02` (0% overlap with Train)
* **Test Speakers (2 profiles):** `SPK_TEST_01`, `SPK_TEST_02` (0% overlap with Train and Val)
* **Seed Domains:** Train ($100000 - 199999$), Val ($200000 - 249999$), Test ($300000 - 349999$). Zero seed leakage.

---

## 4. Nuisance Shortcut Remediation Evidence

### 4.1 Comparative ANOVA ($\eta^2$)

| Acoustic Measure | V5 Baseline $\eta^2$ | V5.1 Candidate $\eta^2$ | Delta $\eta^2$ | V5.2 Status |
| :--- | :---: | :---: | :---: | :---: |
| **Full-Clip RMS** | $0.6042$ | $0.1215$ | $-0.4827$ | **Remediated ($< 0.03\text{ dB}$ across words)** |
| **Pre-Onset Background RMS** | $0.1806$ | $0.0282$ | $-0.1523$ | **Remediated ($0.0\text{ dB}$ across words)** |
| **Crest Factor (dB)** | $0.6787$ | $0.5478$ | $-0.1309$ | **Preserved (Physical fricative peak)** |
| **Spectral Centroid (Hz)** | $0.4306$ | $0.3937$ | $-0.0369$ | **Preserved (Phonetic Formant shift)** |
| **HF Energy Ratio** | $0.4295$ | $0.4325$ | $+0.0029$ | **Preserved (Frication noise balance)** |

### 4.2 Matched Counterfactual Parity (Holding Non-Phonetic Controls Invariant)

| Word | Full-Clip RMS | Active Speech RMS | Pre-Onset RMS | Spectral Centroid | HF Energy Ratio | Speech Gain |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`TRAI`** | $0.12029$ | $0.17881$ | **$0.01163$** | $1144.9\text{ Hz}$ | $0.00398$ | $0.8114$ |
| **`PHAI`** | $0.12006$ | $0.17845$ | **$0.01163$** | $2644.5\text{ Hz}$ | $0.16266$ | $0.9197$ |
| **`LEN`** | $0.12023$ | $0.17872$ | **$0.01163$** | $922.0\text{ Hz}$ | $0.00027$ | $0.4382$ |
| **`XUONG`**| $0.12049$ | $0.17910$ | **$0.01163$** | $2930.0\text{ Hz}$ | $0.08747$ | $0.4675$ |
| **Delta** | **$0.00043$ ($< 0.03\text{ dB}$)** | **$0.00065$** | **$0.00000$ ($0.0\text{ dB}$)** | Distinct | Distinct | Decoupled |

---

## 5. TC-ResNet-14 Multi-Seed Training Results

Multi-seed evaluation on held-out test split (450 clips, 75 per class):

* **Architecture:** `VietnameseTCResNet` (TC-ResNet-14, 192 features, $296\text{k}$ parameters)
* **Frontend:** `LogMelFeatureExtractor` ($40$ mels, $16\text{ kHz}$)
* **Training Runs:** Seeds 42, 43, 44 (25 epochs, AdamW, Cosine Annealing)

| Run | Seed | Best Val Loss | Optimal Threshold $\tau^*$ | Test Accuracy | Test Macro-F1 | Brier Score | ECE |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Run 1 | 42 | $0.0001$ | $0.500$ | $100.00\%$ | $100.00\%$ | $0.0003$ | $0.0008$ |
| Run 2 | 43 | $0.0001$ | $0.500$ | $100.00\%$ | $100.00\%$ | $0.0004$ | $0.0012$ |
| Run 3 | 44 | $0.0002$ | $0.500$ | $100.00\%$ | $100.00\%$ | $0.0008$ | $0.0022$ |
| **Mean $\pm$ Std** | | **$0.00013 \pm 0.00005$** | **$0.500 \pm 0.000$** | **$100.00\% \pm 0.00\%$** | **$100.00\% \pm 0.00\%$** | **$0.0005 \pm 0.0003$** | **$0.0014 \pm 0.0007$** |

### Per-Class Test Confusion Matrix (Seed 42)
All 75 test samples per class were correctly classified:
* `TRAI`: 75/75 (0 confused as `PHAI`)
* `PHAI`: 75/75 (0 confused as `TRAI`)
* `LEN`: 75/75
* `XUONG`: 75/75
* `UNKNOWN`: 75/75
* `SILENCE`: 75/75

---

## 6. Continuous Streaming False-Trigger Analysis

* **Target-Absent Stream:** 150 non-target test clips (`UNKNOWN` + `SILENCE`) concatenated with random ambient gap pauses ($186.54\text{ seconds}$).
* **Sliding Window:** $1.0\text{ s}$ duration, $100\text{ ms}$ hop.
* **Findings:**
  - False Trigger Count: **22 activations** with confidence $\ge 0.500$.
  - **False Trigger Rate:** **$424.57\text{ false triggers / hour}$**.
* **Scientific Significance:** High holdout accuracy ($100\%$) on pre-segmented 1.0s clips does not equate to zero false triggers during continuous audio streaming. Sliding-window boundaries across speech onset/offset require multi-frame temporal confirmation or dynamic threshold calibration.

---

## 7. Synthetic Domain-Shift Robustness Results

| Synthetic Distribution Shift Regime | Perturbation Parameters | Accuracy | Drop from Baseline |
| :--- | :--- | :---: | :---: |
| **1. Nominal Baseline** | Standard V5.2 generation | **$100.00\%$** | Baseline ($0.0\%$) |
| **2. Extreme Speaker Anatomy** | Out-of-range VTL ($0.75, 1.25$), $F_0 = 80\text{ Hz}, 310\text{ Hz}$ | **$84.17\%$** | $-15.83\%$ |
| **3. Adverse Low SNR** | Background noise boosted to $-24\text{ dBFS}$ (SNR $6-10\text{ dB}$) | **$87.50\%$** | $-12.50\%$ |
| **4. Extreme Temporal Shift** | Rapid duration $0.25\text{ s}$, boundary onset $0.01\text{ s} / 0.42\text{ s}$ | **$93.33\%$** | $-6.67\%$ |
| **5. Hum & Harmonics Stress** | High-level 50Hz/60Hz mains hum + 3rd harmonic | **$100.00\%$** | $0.00\%$ |

---

## 8. Frozen Artifact & Regression Integrity

All 7 cryptographic baselines and 12 golden test vectors remain untouched:

| Baseline Path | Expected / Verified SHA-256 | Status |
| :--- | :--- | :---: |
| [`data/manifests/test_manifest_v2.csv`](file:///d:/trainidentificationaudio/data/manifests/test_manifest_v2.csv) | `9E3904C8EA427244F7280A41140D7EFD19FD9625B359290ADB7ADCD4FA2E1C13` | **MATCH** |
| [`data/raw/field_recordings/field_collection_manifest.csv`](file:///d:/trainidentificationaudio/data/raw/field_recordings/field_collection_manifest.csv) | `B389282F9853A74484B39356EE85E5BD7E74C0C46481300524C7BD2894AA8C85` | **MATCH** |
| [`data/raw/field_recordings/consent_ledger.json`](file:///d:/trainidentificationaudio/data/raw/field_recordings/consent_ledger.json) | `BE4BFDD7CEC0C87A1DC97455E5783054EF8093C638279C44940C896F5BDA2BA0` | **MATCH** |
| [`experiments/synthetic_kws_v4/checkpoints/arm_b_onset_no_resample.pt`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_v4/checkpoints/arm_b_onset_no_resample.pt) | `6A0FD55CD6F16F5B683951127076EEC72506C31EE670F896AAEACE5981CF1D96` | **MATCH** |
| [`experiments/synthetic_kws_demo/export/arm_b_tcresnet14.onnx`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_demo/export/arm_b_tcresnet14.onnx) | `8A5AC4C105DF52F1749AE11B6BB2124B3A672486601EFF308ACC7E257FF97625` | **MATCH** |
| [`experiments/synthetic_kws_demo/android_app/app/src/main/assets/models/arm_b_tcresnet14.onnx`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_demo/android_app/app/src/main/assets/models/arm_b_tcresnet14.onnx) | `8A5AC4C105DF52F1749AE11B6BB2124B3A672486601EFF308ACC7E257FF97625` | **MATCH** |
| [`experiments/synthetic_kws_demo/frontend_golden/golden_manifest.json`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_demo/frontend_golden/golden_manifest.json) | `5904A5503CE5528ECF88A4A670C6726D80B74285B34EE6B910CF2FA48F60E5C3` | **MATCH** |
| **All 12 Golden WAVs and Tensors** | 12/12 cases match manifest checksums bit-exact | **MATCH** |

### Test Suite Execution
* Command: `pytest tests/test_synthetic_kws_v5_2_candidate.py tests/test_synthetic_kws_v5_1_candidate.py tests/test_synthetic_kws_v5_generator.py tests/test_kws_export_parity.py tests/test_synthetic_kws_demo.py tests/test_campaign_acceptance.py`
* **Result:** **135 passed, 0 failed** in 31.22s.
* `git diff --check`: Clean (0 errors).
* `git diff --cached`: **Strictly empty** (0 staged files).

---

## 9. Limitations & Blocking Governance Decisions

> [!WARNING]
> **Explicit Scientific Limitations:**
> 1. **Synthetic-Only Scope:** Waveforms are generated by an articulatory vocal tract filter model. The high accuracy ($100\%$) and domain robustness are properties of the synthetic physical model.
> 2. **NOT VALIDATED ON NATURAL SPEECH:** This model must **never** be claimed as validated on human speech or deployment-ready for natural speakers.
> 3. **Lexical Constraints:** Repeated synthesis of target keywords provides acoustic and speaker diversity, but not lexical variety.
> 4. **Streaming False Triggers:** Sliding-window boundary transitions require multi-frame temporal smoothing before real-time streaming deployment.
> 5. **Human Data Collection Blocked:** `data/raw/field_recordings/` remains strictly blocked. Zero human audio was processed.

**Next Blocking Scientific Decision:**
A formal institutional review and authorization protocol must be completed before collecting, recording, or ingesting real Vietnamese human speech recordings for in-domain acoustic evaluation.

---

## 10. Reproduction Commands

To independently reproduce all results:

```powershell
# 1. Run Phase 13.3 Independent Audit
.\.venv\Scripts\python.exe experiments/synthetic_kws_phase13_3_16/phase13_3_audit/audit_v5_v5_1.py

# 2. Validate Phase 14 Dataset (3,000 clips)
.\.venv\Scripts\python.exe experiments/synthetic_kws_phase13_3_16/dataset_release_candidate/validate_dataset.py

# 3. Train and Evaluate TC-ResNet-14 across 3 seeds (Phase 15)
.\.venv\Scripts\python.exe experiments/synthetic_kws_phase13_3_16/model_evaluation/train_and_evaluate.py

# 4. Run Phase 16 Domain-Shift Stress Tests
.\.venv\Scripts\python.exe experiments/synthetic_kws_phase13_3_16/phase16_closeout/run_domain_shift_stress_tests.py

# 5. Run Complete Regression Test Suite
.\.venv\Scripts\python.exe -m pytest tests/test_synthetic_kws_v5_2_candidate.py tests/test_synthetic_kws_v5_1_candidate.py tests/test_synthetic_kws_v5_generator.py tests/test_kws_export_parity.py tests/test_synthetic_kws_demo.py tests/test_campaign_acceptance.py
```
