# Phase 6.1C: Matched Speech-Band A/B & Negative Controls Audit Report

**Repository:** `D:\trainidentificationaudio`  
**Branch:** `feature/continuous-kws-segmentation`  
**Baseline Git Commit:** `d5af658a9afe523b306e5ca6c0159789f1c67fad`  
**Date:** 2026-10-30  
**Status:** Completed  
**Workstream:** Phase 6.1C (Workstream C)

---

## Mandatory Governance Labels

> - **`EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`**
> - **`NOT_VALIDATED_ON_NATURAL_SPEECH`**
> - **`NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`**
> - **`NOT_VALIDATED_PENDING_DATA`**

---

## 1. Frozen Artifact Integrity Verification

- **PyTorch Checkpoint:** `experiments/vieneu_kws_research_v13_4/bundle/checkpoint/variant_v11_hard_negative_weighted_seed42.pt`  
  `SHA-256: 3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2` (Verified Match)
- **ONNX Model:** `experiments/vieneu_kws_research_v13_4/bundle/onnx/vietnamese_kws_v11.onnx`  
  `SHA-256: 54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99` (Verified Match)
- **Constant Parameters:** $\theta_{\text{cmd}} = 0.20$, 138k parameter frozen v11 CNN, energy threshold -46.0 dB, silence interval 140 ms, preroll 120 ms, dip threshold 12.0 dB, rebound threshold 6.0 dB.

---

## 2. Matched Speech-Band A/B Comparison: 300 Hz vs 100 Hz Low Cutoff

A strict ceteris paribus evaluation was conducted comparing `speech_band_hz=(300.0, 3400.0)` vs `speech_band_hz=(100.0, 3400.0)` across both the Phase 6 locked manifest and the Phase 6.1A holdout manifest:

### A) Phase 6 Locked Manifest (200 Trials)

| Scenario | Metric | `speech_band_hz=(300.0, 3400.0)` | `speech_band_hz=(100.0, 3400.0)` | Delta ($\Delta$) | Target Met? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`LEN->LEN->PHAI` (0 ms gap)** | Exact Match % | 19/30 (63.3%) | **29/30 (96.7%)** | **+33.4%** | **YES ($\ge 90\%$)** |
| | Sequence WER | 20.0% | **1.1%** | **-18.9%** | **YES ($\le 10\%$)** |
| **`LEN->LEN->PHAI` (50 ms gap)** | Exact Match % | 20/30 (66.7%) | **27/30 (90.0%)** | **+23.3%** | **YES ($\ge 90\%$)** |
| | Sequence WER | 16.7% | **3.3%** | **-13.4%** | **YES ($\le 10\%$)** |
| **`LEN->LEN` (0 ms gap)** | Exact Match % | 7/10 (70.0%) | **10/10 (100.0%)** | **+30.0%** | **YES** |
| **`LEN->LEN` (50 ms gap)** | Exact Match % | 8/10 (80.0%) | **10/10 (100.0%)** | **+20.0%** | **YES** |
| **`TRAI->LEN->XUONG` (0 ms gap)** | Exact Match % | 8/10 (80.0%) | **10/10 (100.0%)** | **+20.0%** | **YES** |
| **`TRAI->LEN->XUONG` (50 ms gap)** | Exact Match % | 8/10 (80.0%) | **10/10 (100.0%)** | **+20.0%** | **YES** |

### B) Phase 6.1A Holdout Manifest (55 Trials)

| Scenario | Metric | `speech_band_hz=(300.0, 3400.0)` | `speech_band_hz=(100.0, 3400.0)` | Delta ($\Delta$) | Target Met? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Holdout `LEN->LEN->PHAI` (0 ms)** | Exact Match % | 9/15 (60.0%) | **15/15 (100.0%)** | **+40.0%** | **YES ($\ge 90\%$)** |
| | Sequence WER | 20.0% | **0.0%** | **-20.0%** | **YES ($\le 10\%$)** |
| **Holdout `LEN->LEN->PHAI` (50 ms)** | Exact Match % | 8/15 (53.3%) | **14/15 (93.3%)** | **+40.0%** | **YES ($\ge 90\%$)** |
| | Sequence WER | 15.5% | **2.2%** | **-13.3%** | **YES ($\le 10\%$)** |
| **Holdout `LEN->LEN` (0 ms)** | Exact Match % | 3/5 (60.0%) | **5/5 (100.0%)** | **+40.0%** | **YES** |
| **Holdout `LEN->LEN` (50 ms)** | Exact Match % | 4/5 (80.0%) | **5/5 (100.0%)** | **+20.0%** | **YES** |
| **Holdout `TRAI->LEN->XUONG` (0 ms)**| Exact Match % | 4/5 (80.0%) | **5/5 (100.0%)** | **+20.0%** | **YES** |
| **Holdout `TRAI->LEN->XUONG` (50 ms)**| Exact Match % | 4/5 (80.0%) | **5/5 (100.0%)** | **+20.0%** | **YES** |

---

## 3. Negative Controls Audit: Open-Gate vs Closed-Gate

A comprehensive audit was conducted across 900.0 seconds of negative audio logged in [`phase6_1c_negative_control_results.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/reports/phase6_1c_negative_control_results.csv):

| Negative Corpus | Duration | `hybrid_300hz` (Open-Gate) | `hybrid_100hz` (Open-Gate) | `game_mode_closed_gate` (Closed-Gate) |
| :--- | :---: | :---: | :---: | :---: |
| **Digital Silence** | 300.0s (5.0 min) | 0 FA (0.0 FA/min) | 0 FA (0.0 FA/min) | **0 FA (0.0 FA/min)** |
| **Environmental Noise** | 100.0s (1.7 min) | 0 FA (0.0 FA/min) | 0 FA (0.0 FA/min) | **0 FA (0.0 FA/min)** |
| **Conversational Speech** | 300.0s (5.0 min) | 379 FA (75.8 FA/min) | 669 FA (133.8 FA/min) | **0 FA (0.0 FA/min)** |
| **OOV Phonetic Foils** | 200.0s (3.3 min) | 284 FA (85.2 FA/min) | 361 FA (108.3 FA/min) | **0 FA (0.0 FA/min)** |
| **Total Audio** | **900.0s (15.0 min)** | **663 FA** | **1,030 FA** | **0 FA (100% Rejection)** |

---

## 4. Critical Governance Disclosures & Architectural Gating Behavior

1. **Acoustic Valley Gating in Open-Mic Mode:**  
   When the microphone gate is continuously open, short-time energy valley splitting triggers on natural pauses between conversational syllables, resulting in over 130 false activations per minute. Acoustic gating alone cannot substitute for a linguistic ASR model in open conversational environments.
2. **Session Gating Solves Chatter False Alarms:**  
   Under `GameModeSessionController`, the microphone is only armed during active gameplay sessions (e.g. 6.0 second timeout auto-extended upon valid command). When idle, audio is discarded, achieving **strictly 0 false activations across all 900.0 seconds of negative audio**.
3. **Disclosure of Closed-Gate Semantics:**  
   The zero false-activation result under a closed gate validates **architectural session gating behavior only**, and must NEVER be cited as evidence of open-mic acoustic classifier rejection accuracy.

---

## 5. Summary Deliverables

- Detailed Speech-Band A/B CSV: [`phase6_1c_speechband_ab_results.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/reports/phase6_1c_speechband_ab_results.csv)
- Negative Control Full Log CSV: [`phase6_1c_negative_control_results.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/reports/phase6_1c_negative_control_results.csv)
- Comprehensive Audit JSON: [`phase6_1c_speechband_audit.json`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/reports/phase6_1c_speechband_audit.json)
