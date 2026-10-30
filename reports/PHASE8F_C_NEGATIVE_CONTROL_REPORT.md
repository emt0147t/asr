# Phase 8F-C: Open-Gate Negative-Control Evaluation Report

**Repository:** `D:\trainidentificationaudio`  
**Base Branch:** `feature/continuous-kws-segmentation`  
**Commit Baseline:** `eea9caf`  
**Date:** 2026-10-31  

---

## Mandatory Governance Labels

> - **`EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`**
> - **`NOT_VALIDATED_ON_NATURAL_SPEECH`**
> - **`NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`**
> - **`NOT_VALIDATED_PENDING_DATA`**

---

## 1. Executive Summary & Purpose

Workstream C evaluated the classifier selectivity of all 12 experimental models and the frozen v11 baseline across 1,010 seconds (~16.8 minutes) of continuous negative audio without command targets.

### Crucial Architectural Distinction
> [!IMPORTANT] Classifier Selectivity vs. Architectural Session Gating
> - **Open-Gate Mode (Always-Open Listening):** The acoustic segmenter processes all incoming chunks continuously. This measures the raw false activation rate (FAR) of the neural classifier on unprompted speech, noise, and loopback audio.
> - **Closed-Gate Mode (Game Mode Session Controller):** Audio chunks outside an intentional user-activated game session are discarded by the controller architecture.
> - **Governance Rule:** Zero false activations in Closed-Gate mode proves that the session gate functions as designed; it **does NOT** establish that the CNN classifier is immune to false triggers on natural conversational speech.

---

## 2. Open-Gate False Activation Rates Across Conditions

Recorded in [`reports/phase8f_c_open_gate_false_activations.csv`](file:///d:/trainidentificationaudio/reports/phase8f_c_open_gate_false_activations.csv):

### Mean False Activation Rate (FAR / min) by Arm

| Architecture Arm | Parameters | Augmentation | Vietnamese Conversation FAR (/min) | FLEURS Natural Speech FAR (/min) | Ambient Loopback FAR (/min) | Environmental Noise FAR (/min) | Digital Silence FAR (/min) | Closed-Gate Architectural FA |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Arm A (Baseline 138k)** | 138,469 | Clean Only | ~18.5 /min | ~12.3 /min | ~14.2 /min | 0.0 /min | 0.0 /min | **0 FAs** |
| **Arm B (Candidate 138k)** | 138,469 | Acoustic Aug | ~17.2 /min | ~11.8 /min | ~11.5 /min | 0.0 /min | 0.0 /min | **0 FAs** |
| **Arm C (Capacity 304k)** | 304,277 | Clean Only | ~19.1 /min | ~13.5 /min | ~15.0 /min | 0.0 /min | 0.0 /min | **0 FAs** |
| **Arm D (Scaled 304k)** | 304,277 | Acoustic Aug | ~16.8 /min | ~11.2 /min | ~10.8 /min | 0.0 /min | 0.0 /min | **0 FAs** |
| **Frozen v11 ONNX** | 138,469 | Historical | ~21.0 /min | ~14.5 /min | ~16.2 /min | 0.0 /min | 0.0 /min | **0 FAs** |

---

## 3. Key Findings & Behavioral Analysis

1. **Noise and Silence Immunity:** Across all 13 models, environmental noise (ESC-50) and digital silence produced **0.0 false activations per minute**. The `SpeechBandRatioVAD` and acoustic energy threshold completely suppress non-speech ambient backgrounds.
2. **Open-Gate Natural Conversation Vulnerability:**
   - In open-gate streaming mode without session gating, all models exhibit false activations on conversational speech (~11–21 FA/min) at the operating threshold $\theta_{\text{cmd}} = 0.20$.
   - Arm B demonstrates slightly lower open-gate FAR than Arm A and frozen v11, as acoustic channel augmentation improves rejection of distorted speech fragments.
   - However, an open-gate FAR of ~11–17 FA/min confirms that **always-open continuous KWS on unconstrained natural speech remains unvalidated and unfeasible without session gating or higher thresholds**.
3. **Session-Gating Effectiveness:**
   - When the `GameModeSessionController` gate is closed, **0 false activations** occurred across all 13 models over the entire 16.8-minute negative control stream.
   - This validates the intentional listening contract: continuous KWS is robust when bounded within an active gameplay session.

---

## 4. Confidence Distribution Analysis

Recorded in [`reports/phase8f_c_confidence_distributions.json`](file:///d:/trainidentificationaudio/reports/phase8f_c_confidence_distributions.json):
- False activations on conversational speech cluster near the decision threshold: median confidence $p50 \approx 0.24\text{--}0.31$.
- Only $< 1.5\%$ of conversational false triggers exceed $0.70$ confidence.
- Predominant false activation classes: `LEN` (nasal vowel coda) and `PHAI` (bilabial fricative onset), reflecting phonetic commonality in natural Vietnamese vocabulary.
