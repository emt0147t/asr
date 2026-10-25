# Final Research Closeout: Synthetic KWS Phases 17–19
## Independent Validity Audit, Streaming Detection, and Generalization Stress Testing

**Repository:** `D:\trainidentificationaudio`  
**Execution Date:** 2026-10-16  
**Auditor:** Antigravity (Advanced Agentic Coding Pair Programmer)  
**Governance Invariant:** **`HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED`**  
**Official Status:** **`NOT_VALIDATED_ON_NATURAL_SPEECH`**  

---

## 1. Executive Summary & Verdicts

Phases 17, 18, and 19 were executed sequentially to validate the 3,000-clip V5.2 release candidate dataset, resolve the streaming false-trigger engineering bottleneck, and stress-test the production TC-ResNet-14 architecture:

| Phase | Core Objective | Gate Verdict | Primary Report |
| :--- | :--- | :---: | :--- |
| **Phase 17** | Independent Dataset, Leakage & Performance Audit | **`PHASE17_PASS_WITH_LIMITATIONS`** | [`phase17_audit/PHASE17_INDEPENDENT_VALIDITY_AUDIT.md`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_phase17_19/phase17_audit/PHASE17_INDEPENDENT_VALIDITY_AUDIT.md) |
| **Phase 18** | Synthetic Streaming Detection & False-Trigger Reduction | **`PHASE18_PASS_WITH_LIMITATIONS`** | [`phase18_streaming/PHASE18_STREAMING_EVALUATION.md`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_phase17_19/phase18_streaming/PHASE18_STREAMING_EVALUATION.md) |
| **Phase 19** | Factorial Generalization & Research Synthesis | **`PHASE19_PASS_WITH_LIMITATIONS`** | [`phase19_generalization/PHASE19_GENERALIZATION_REPORT.md`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_phase17_19/phase19_generalization/PHASE19_GENERALIZATION_REPORT.md) |

### Status Tri-Partition
* **Engineering Correctness:** **`VERIFIED_PASS`**
* **Synthetic Distribution Generalization:** **`VERIFIED_PASS_WITH_LIMITATIONS`**
* **Natural-Speech Validity:** **`NOT_VALIDATED_ON_NATURAL_SPEECH`**

---

## 2. Artifact Paths Produced

* **Phase 17 Audit:**
  - Audit Script: [`experiments/synthetic_kws_phase17_19/phase17_audit/phase17_audit.py`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_phase17_19/phase17_audit/phase17_audit.py)
  - Results JSON: [`experiments/synthetic_kws_phase17_19/phase17_audit/phase17_results.json`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_phase17_19/phase17_audit/phase17_results.json)
  - Full Report: [`experiments/synthetic_kws_phase17_19/phase17_audit/PHASE17_INDEPENDENT_VALIDITY_AUDIT.md`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_phase17_19/phase17_audit/PHASE17_INDEPENDENT_VALIDITY_AUDIT.md)
* **Phase 18 Streaming:**
  - Streaming Engine & Detector: [`experiments/synthetic_kws_phase17_19/phase18_streaming/streaming_detector.py`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_phase17_19/phase18_streaming/streaming_detector.py)
  - Results JSON: [`experiments/synthetic_kws_phase17_19/phase18_streaming/phase18_streaming_results.json`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_phase17_19/phase18_streaming/phase18_streaming_results.json)
  - Full Report: [`experiments/synthetic_kws_phase17_19/phase18_streaming/PHASE18_STREAMING_EVALUATION.md`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_phase17_19/phase18_streaming/PHASE18_STREAMING_EVALUATION.md)
* **Phase 19 Generalization:**
  - Counterfactual Script: [`experiments/synthetic_kws_phase17_19/phase19_generalization/generalization_counterfactuals.py`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_phase17_19/phase19_generalization/generalization_counterfactuals.py)
  - Results JSON: [`experiments/synthetic_kws_phase17_19/phase19_generalization/phase19_generalization_results.json`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_phase17_19/phase19_generalization/phase19_generalization_results.json)
  - Full Report: [`experiments/synthetic_kws_phase17_19/phase19_generalization/PHASE19_GENERALIZATION_REPORT.md`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_phase17_19/phase19_generalization/PHASE19_GENERALIZATION_REPORT.md)
* **Closeout Governance Deliverables:**
  - Machine-Readable Status: [`experiments/synthetic_kws_phase17_19/FINAL_STATUS_PHASE17_19.json`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_phase17_19/FINAL_STATUS_PHASE17_19.json)
  - Comprehensive Closeout: [`experiments/synthetic_kws_phase17_19/FINAL_RESEARCH_CLOSEOUT_PHASE17_19.md`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_phase17_19/FINAL_RESEARCH_CLOSEOUT_PHASE17_19.md)

---

## 3. Full-Manifest Regeneration Audit (17.1)

Rather than evaluating a small subset, Phase 17 executed **full-manifest bit-exact regeneration of all 3,000 clips** directly from manifest parameters using [`generator.py:reproduce_from_manifest_row`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_v5_2_candidate/generator.py):
* Total Clips Attempted: **3,000**
* Exact Bit-Exact SHA-256 Matches: **3,000 / 3,000 (100.00%)**
* Unique Audio Hashes: Exactly 3,000 (0 exact duplicates)
* Split Isolation: 0 speaker profile overlaps, 0 seed domain overlaps across Train, Validation, and Test partitions.
* Label Leakage Audit: Confirmed that feature tensors contain strictly raw 40-band spectrograms. No filenames, transcripts, or sample IDs are passed to the model.

---

## 4. Recomputed Model Metrics & Statistical Uncertainty (17.2)

Evaluated the 3 trained TC-ResNet-14 checkpoints against the 450 held-out test clips (75 per class):
* **Accuracy:** **$100.00\% \pm 0.00\%$** (Seeds 42, 43, 44 all achieved $100.00\%$)
* **Macro-F1:** **$100.00\%$**
* **Calibration:** Brier score $= 0.000575$ (Seed 42), ECE $= 0.001007$ (Seed 42)
* **Finite-Sample Uncertainty Analysis:**
  - Point estimate: $100.00\%$ on $N=450$ test clips.
  - **Wilson Score 95% Confidence Interval:** **$[99.16\%, 100.00\%]$**.
  - **Rule of Three Error Bound (95% CI):** Max error rate bound on this synthetic distribution is $\le \frac{3}{450} = \mathbf{0.67\%}$.

---

## 5. Diagnostic Baselines and Nuisance Ablations (17.3)

Trained linear/logistic diagnostic probes on Train split and evaluated on held-out Test split:
* **Diagnostic A (Metadata-Only):** $52.00\%$ test accuracy (using duration, onset, target RMS, background RMS, gain factor).
* **Diagnostic B (Strict Nuisance Acoustic):** $52.67\%$ test accuracy (using whole-clip RMS, peak, crest factor, first-25ms RMS, pre-onset RMS; no spectral features).
* **Diagnostic C (Acoustic Phonetic Features):** $41.56\%$ test accuracy (spectral centroid, HF ratio, ZCR).
* **Neural Model (TC-ResNet-14):** $100.00\%$ test accuracy.
* **Random Chance Level:** $16.67\%$.

Nuisance features alone separate `SILENCE` from active speech, but cannot discriminate between target keywords. The neural network's $100\%$ accuracy relies on learned temporal-spectral representations of genuine phonetic contrasts (/ʈ/ vs /f/ vs /l/ vs /s/).

---

## 6. Streaming Detection & False-Trigger Elimination (Phase 18)

Evaluated over **$3,600.0\text{ seconds}$ (1.00 Hour) of continuous target-absent synthetic audio** (4 independent 15-minute streams) and a continuous target-present stream with 50 target keywords:

| Metric | Raw Window Baseline ($\tau=0.50$) | Hysteresis Detector (Method 4) |
| :--- | :---: | :---: |
| **Evaluated Target-Absent Audio** | $3,600.0\text{ s}$ ($1.00\text{ hr}$) | $3,600.0\text{ s}$ ($1.00\text{ hr}$) |
| **Independent Streams Evaluated** | 4 streams | 4 streams |
| **False Trigger Count (Target-Absent)** | 339 events | **0 events** |
| **False Triggers / Hour** | **$339.00\text{ FP/hr}$** | **$0.00\text{ FP/hr}$** |
| **Ground-Truth Targets Evaluated** | 50 | 50 |
| **Target Detection Recall** | $100.00\%$ ($50/50$) | **$100.00\%$ ($50/50$)** |
| **Missed Targets** | 0 | **0** |
| **Event Precision** | $45.87\%$ | **$100.00\%$** |
| **Mean Detection Latency** | $0.302\text{ s} \pm 0.079\text{ s}$ | **$0.496\text{ s} \pm 0.078\text{ s}$** |

### Selected Detector Parameters (Tuned on Validation Only)
* Activation Threshold: $\tau_{on} = 0.80$
* Deactivation Threshold: $\tau_{off} = 0.35$
* Moving Average Smoothing Window: $W = 3$ frames ($300\text{ ms}$)
* Consecutive Evidence Constraint: $M = 2$ frames ($200\text{ ms}$)
* Refractory Cooldown: $C = 600\text{ ms}$ (6 hops)

---

## 7. Factorial Generalization & Counterfactual Stress (Phase 19)

### In-Range Factorial Sweeps (Target Keywords)
* **$F_0$ Sweep ($100\text{ Hz} - 260\text{ Hz}$):** **$100.00\%$** across all 5 pitch levels.
* **$VTL$ Scale Sweep ($0.85 - 1.15$):** **$100.00\%$** across all 5 vocal tract length scales.
* **Speech Target Level ($-20\text{ dBFS} - -11\text{ dBFS}$):** **$100.00\%$** across all levels.
* **Background Noise Level ($-42\text{ dBFS} - -24\text{ dBFS}$):** **$100.00\%$** across all levels.

### Paired Counterfactual: `TRAI` vs `PHAI` Invariance
* 25 matched pairs with strictly invariant speaker profile, timing, speech target level, and room noise realization:
  - `TRAI` Accuracy: **$100.00\%$** ($25/25$)
  - `PHAI` Accuracy: **$100.00\%$** ($25/25$)
  - Cross-confusion (`TRAI` $\leftrightarrow$ `PHAI`): **0**

### Out-of-Range Stress Testing
* Extreme $VTL$ ($0.70$ child / $1.30$ giant): **$90.00\%$** ($36/40$)
* Extreme $F_0$ ($70\text{ Hz}$ sub-bass / $350\text{ Hz}$ soprano): **$90.00\%$** ($36/40$)
* Extreme Noise Floor ($-18.0\text{ dBFS}$, $SNR \approx 2\text{ dB}$): **$67.50\%$** ($27/40$)

---

## 8. Frozen Artifact & Regression Integrity

All 7 cryptographic baselines and 12 golden test vectors remain bit-exact:

| Protected Artifact Path | Verified SHA-256 Checksum | Match Status |
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

## 9. Explicit Scientific Limitations & Next Decision

> [!WARNING]
> **Core Scientific Limitations:**
> 1. **Synthetic-Only Evaluation:** All evaluated waveforms originate from parametric vocal tract synthesis. The reported $100\%$ accuracy and $0.00\text{ FP/hr}$ streaming metrics reflect performance on this physical synthetic model.
> 2. **NOT VALIDATED ON NATURAL SPEECH:** This model must **never** be represented as validated on natural human speakers, real smartphone microphones, or genuine acoustic reverberation.
> 3. **Streaming Latency Trade-Off:** Achieving $0.00\text{ FP/hr}$ required adding multi-frame persistence and smoothing, increasing mean detection latency to $496\text{ ms}$.
> 4. **Human Data Collection Remains Blocked:** `data/raw/field_recordings/` remains strictly blocked. Zero human audio was collected, loaded, or processed.

**Next Blocking Scientific Decision:**
Execution of a formal institutional review and ethics approval protocol for collecting and recording real Vietnamese human speech to establish the first authentic in-domain acoustic evaluation benchmark.

---

## 10. Exact Reproduction Commands

```powershell
# 1. Run Phase 17 Independent Audit (including full 3,000-clip regeneration)
.\.venv\Scripts\python.exe experiments/synthetic_kws_phase17_19/phase17_audit/phase17_audit.py

# 2. Run Phase 18 Streaming Detection (1.00-Hour continuous audio evaluation)
.\.venv\Scripts\python.exe experiments/synthetic_kws_phase17_19/phase18_streaming/streaming_detector.py

# 3. Run Phase 19 Factorial Generalization & Counterfactual Experiments
.\.venv\Scripts\python.exe experiments/synthetic_kws_phase17_19/phase19_generalization/generalization_counterfactuals.py

# 4. Run Complete Regression Test Suite
.\.venv\Scripts\python.exe -m pytest tests/test_synthetic_kws_v5_2_candidate.py tests/test_synthetic_kws_v5_1_candidate.py tests/test_synthetic_kws_v5_generator.py tests/test_kws_export_parity.py tests/test_synthetic_kws_demo.py tests/test_campaign_acceptance.py
```
