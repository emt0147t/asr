# Phase 22 — Research Release Reproducibility Guide

**Workspace:** `D:\trainidentificationaudio`  
**Candidate Identifier:** `Vietnamese_Synthetic_KWS_TCResNet14_V5_2_RC`  
**Official Status:** `SYNTHETIC_RESEARCH_CANDIDATE_WITH_LIMITATIONS`  
**Governance:** `NOT_VALIDATED_ON_NATURAL_SPEECH` (Human data collection strictly blocked)  

---

## 1. Environment & Pre-Requisites

### 1.1 Python Virtual Environment
- **Python Version:** 3.11.9 (Windows x64)
- **Virtual Environment Path:** `.\.venv\Scripts\python.exe`
- **Core Dependencies:**
  - `torch == 2.5.1+cu124` (NVIDIA RTX A2000 Laptop GPU detected)
  - `soundfile == 0.12.1`
  - `scipy == 1.14.1`
  - `numpy == 1.26.4`
  - `pandas == 2.2.3`
  - `scikit-learn == 1.5.2`
  - `pytest == 9.1.1`

### 1.2 Hardware Acceleration
CUDA acceleration (`cuda:0`, NVIDIA GPU) is detected and utilized automatically by the feature extractor and TC-ResNet-14 inference engine. If CUDA is unavailable, all scripts seamlessly fall back to CPU execution.

---

## 2. Reproducible Entry Points & Measured Performance

All entry points are located in [`experiments/synthetic_kws_phase20_22/phase22_release/`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_phase20_22/phase22_release/) and execute deterministically without requiring source code edits.

### Entry Point 1: Dataset Verification & Model Metric Re-Audit
```powershell
.\.venv\Scripts\python.exe experiments/synthetic_kws_phase20_22/phase22_release/reproduce_audit.py
```
- **Exit Code:** `0`
- **Measured Elapsed Time:** `101.0 seconds`
- **Operations & Measured Output:**
  - Verified 3,000 WAV files on disk with zero missing files and zero format violations.
  - Executed full 3,000-clip row-independent regeneration: **3,000 / 3,000 (100.00%)** bit-exact SHA-256 matches.
  - Evaluated Seeds 42, 43, 44 on frozen test split: **100.00%** accuracy, Brier $\le 0.000919$, ECE $\le 0.00227$.
  - Recomputed diagnostic probes across 6-class, 4-target, and binary tasks.
- **Output Artifacts:**
  - [`phase20_results.json`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_phase20_22/phase20_audit/phase20_results.json)
  - [`PHASE20_INDEPENDENT_AUDIT.md`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_phase20_22/phase20_audit/PHASE20_INDEPENDENT_AUDIT.md)

### Entry Point 2: Extended Held-Out Speaker Generalization
```powershell
.\.venv\Scripts\python.exe experiments/synthetic_kws_phase20_22/phase22_release/reproduce_robustness.py
```
- **Exit Code:** `0`
- **Measured Elapsed Time:** `20.1 seconds`
- **Operations & Measured Output:**
  - Instantiated 24 newly sampled held-out speaker profiles spanning High Female, Mid Female, Mid Male, and Deep Male vocal tracts (VTL 0.82–1.18, F0 92–255 Hz).
  - Synthesized 288 balanced evaluation clips across the 6 canonical classes.
  - Evaluated frozen TC-ResNet-14 checkpoint (`seed_42.pt`): **100.00%** accuracy (288 / 288), Macro-F1 = 1.0000.
  - Wilson 95% CI: $[98.68\%, 100.00\%]$.
  - Zero confusion between `TRAI` and `PHAI`; zero false alarms from `UNKNOWN` or `SILENCE`.
- **Output Artifacts:**
  - [`phase21_heldout_manifest.csv`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_phase20_22/phase21_robustness/phase21_heldout_manifest.csv)
  - [`phase21_heldout_eval.json`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_phase20_22/phase21_robustness/phase21_heldout_eval.json)

### Entry Point 3: Extended Streaming Benchmark (3.00 Hours & 200 Target Events)
```powershell
.\.venv\Scripts\python.exe experiments/synthetic_kws_phase20_22/phase22_release/reproduce_streaming.py
```
- **Exit Code:** `0`
- **Measured Elapsed Time:** `144.4 seconds`
- **Operations & Measured Output:**
  - Target-Absent Streaming: 10,800.0 seconds (3.00 Hours) across 6 streams (107,946 sliding windows) yielded **0 false triggers** (**$0.00\text{ FP/hr}$**, Poisson 95% upper bound $\le 1.00\text{ FP/hr}$).
  - Target-Present Detection: 200 target events across 8 continuous streams yielded **200/200 detected** (**100.00% recall, 100.00% precision**, mean latency $0.506\text{ s}$, median $0.503\text{ s}$, P95 $0.636\text{ s}$).
  - Counterfactual Study: 50 matched condition pairs for `TRAI` vs `PHAI` were **100.00% (50/50)** distinguished with high confidence margins.
- **Output Artifacts:**
  - [`stream_level_target_absent.csv`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_phase20_22/phase21_robustness/stream_level_target_absent.csv)
  - [`stream_level_target_present.csv`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_phase20_22/phase21_robustness/stream_level_target_present.csv)
  - [`event_level_target_present.csv`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_phase20_22/phase21_robustness/event_level_target_present.csv)
  - [`counterfactual_trai_phai_results.csv`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_phase20_22/phase21_robustness/counterfactual_trai_phai_results.csv)
  - [`phase21_results.json`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_phase20_22/phase21_robustness/phase21_results.json)
  - [`PHASE21_ROBUSTNESS_REPORT.md`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_phase20_22/phase21_robustness/PHASE21_ROBUSTNESS_REPORT.md)

---

## 3. Reference Checkpoint & Golden Vector Hashes

### 3.1 Model Checkpoints
| Checkpoint | Path | SHA-256 Checksum |
| :--- | :--- | :--- |
| Seed 42 | `experiments/synthetic_kws_phase13_3_16/model_evaluation/checkpoints/v5_2_tcresnet14_seed_42.pt` | `4310338165b3e29de6ae95d2628cdd9bf1c3cbdfd9364ba0b57d73843f4e9a19` |
| Seed 43 | `experiments/synthetic_kws_phase13_3_16/model_evaluation/checkpoints/v5_2_tcresnet14_seed_43.pt` | `c196ac868034d67c7824f8ec4aa915dbdcccefe99abcea4f2469dc4032af158c` |
| Seed 44 | `experiments/synthetic_kws_phase13_3_16/model_evaluation/checkpoints/v5_2_tcresnet14_seed_44.pt` | `4a2de65520c750290a489d2458ba59ff0129524273968408fdc7821c92b21631` |

### 3.2 Baseline Artifacts
All 7 protected historical repository baselines verified bit-exact:
- `data/manifests/test_manifest_v2.csv`: `9E3904C8EA427244F7280A41140D7EFD19FD9625B359290ADB7ADCD4FA2E1C13`
- `data/raw/field_recordings/field_collection_manifest.csv`: `B389282F9853A74484B39356EE85E5BD7E74C0C46481300524C7BD2894AA8C85`
- `data/raw/field_recordings/consent_ledger.json`: `BE4BFDD7CEC0C87A1DC97455E5783054EF8093C638279C44940C896F5BDA2BA0`
- `experiments/synthetic_kws_v4/checkpoints/arm_b_onset_no_resample.pt`: `6A0FD55CD6F16F5B683951127076EEC72506C31EE670F896AAEACE5981CF1D96`
- `experiments/synthetic_kws_demo/export/arm_b_tcresnet14.onnx`: `8A5AC4C105DF52F1749AE11B6BB2124B3A672486601EFF308ACC7E257FF97625`
- `experiments/synthetic_kws_demo/android_app/app/src/main/assets/models/arm_b_tcresnet14.onnx`: `8A5AC4C105DF52F1749AE11B6BB2124B3A672486601EFF308ACC7E257FF97625`
- `experiments/synthetic_kws_demo/frontend_golden/golden_manifest.json`: `5904A5503CE5528ECF88A4A670C6726D80B74285B34EE6B910CF2FA48F60E5C3`

---

## 4. Scientific Scope & Mandatory Warning

> [!CAUTION]
> **SYNTHETIC-ONLY VALIDATION NOTICE**  
> All evaluations, metrics, and benchmarks in this candidate were conducted strictly on parametric synthetic vocal tract audio.  
> - **Accuracy:** 100% synthetic accuracy and zero false triggers reflect model discrimination of clean physical acoustic models.  
> - **Natural Speech:** This candidate has **NOT** been evaluated on, nor validated against, natural human speech.  
> - **Deployment:** This artifact is a **Research Evaluation Candidate only**. It is not approved for production or natural deployment.
