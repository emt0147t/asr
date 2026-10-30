# Phase 8D: Matched Acoustic Channel Evaluation Protocol Report

**Repository:** `D:\trainidentificationaudio`  
**Base Branch:** `feature/continuous-kws-segmentation`  
**Baseline Git Commit:** `91d91dbf6ec23486cb6f95e0a62a48d590360bad`  
**Status:** Preregistered & Locked  
**Date:** 2026-10-31

---

## Mandatory Governance Labels

> - **`EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`**
> - **`NOT_VALIDATED_ON_NATURAL_SPEECH`**
> - **`NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`**
> - **`NOT_VALIDATED_PENDING_DATA`**

---

## 1. Executive Summary & Evaluation Protocol Overview

Phase 8D locks the evaluation manifest, channel simulation models, and acceptance criteria across all four experimental arms (Arm A, B, C, D) and the historical frozen v11 baseline.

### Key Governance Safeguards
1. **Quarantined Final Holdout:** The Phase 6.1A holdout manifest (13 columns, 55 continuous trials) remains completely untouched by training, validation, early stopping, or threshold tuning.
2. **Locked Matched Acoustic Channels:** Every experimental model is evaluated on exactly identical audio instances across four distinct channel conditions:
   - **Channel 1 (`direct_digital_stream`):** Clean reference bitstream.
   - **Channel 2 (`close_talk_headset`):** Proximity bass boost (+3 dB at 150 Hz), SNR ~28 dB, minimal reverb.
   - **Channel 3 (`laptop_mic_direct`):** On-axis near-field speech (~0.3m), highpass at 140 Hz, high boost at 3.2 kHz, SNR ~22 dB.
   - **Channel 4 (`ambient_loudspeaker_loopback`):** Physical/acoustic loudspeaker transmission, steep low-frequency cutoff (<280 Hz), chassis resonance, and room reverberation ($	ext{RT60} = 0.35\text{s}$).
3. **No Metric Conflation:** Every channel is reported strictly independently. Easy direct-stream scores are never averaged with difficult ambient-room scores.

---

## 2. Evaluation Manifest Composition

Recorded in [`reports/phase8d_channel_evaluation_manifest.csv`](file:///d:/trainidentificationaudio/reports/phase8d_channel_evaluation_manifest.csv):
- **Single-Command Classification Pool:** 100 test samples (20 LEN, 20 XUONG, 20 TRAI, 20 PHAI, 10 negative conversation, 10 negative noise) $\times$ 4 channels = **400 evaluation instances**.
- **Continuous Multi-Command Sequence Pool:** 55 continuous trials from Phase 6.1A holdout manifest $\times$ 4 channels = **220 sequence evaluation instances**.
- **Negative Control Stream:** 500.0s of continuous chatter and background noise evaluated for False Activations per minute.

---

## 3. Preregistered Acceptance Thresholds

Recorded in [`reports/phase8d_acceptance_metrics.json`](file:///d:/trainidentificationaudio/reports/phase8d_acceptance_metrics.json):

| Acoustic Channel Condition | Metric | Acceptance Target |
| :--- | :--- | :---: |
| **Direct Digital Stream** | Classification Accuracy (%) | $\ge 90.0\%$ |
| **Direct Digital Stream** | Holdout Sequence Exact Match (%) | $\ge 90.0\%$ |
| **Close-Talk Headset Mic** | Classification Accuracy (%) | $\ge 90.0\%$ |
| **Close-Talk Headset Mic** | Holdout Sequence Exact Match (%) | $\ge 85.0\%$ |
| **Laptop Mic Direct (0.3m)** | Classification Accuracy (%) | $\ge 80.0\%$ |
| **Ambient Loudspeaker Loopback**| Sequence Exact Match Improvement vs. Arm A | $\ge +10.0\%$ |
| **Game Mode Runtime Safety** | Duplicate Events Emitted | **0 duplicates** |
| **Game Mode Session Gating** | Idle False Activations | **0.0 FA/min** |
| **Inference Compute Latency**| p50 Latency Budget | $< 20.0\text{ ms}$ |

---

## 4. Deliverables Index

- **Channel Evaluation Manifest:** [`reports/phase8d_channel_evaluation_manifest.csv`](file:///d:/trainidentificationaudio/reports/phase8d_channel_evaluation_manifest.csv) (572 instances)
- **Acceptance Metrics JSON:** [`reports/phase8d_acceptance_metrics.json`](file:///d:/trainidentificationaudio/reports/phase8d_acceptance_metrics.json)
- **Evaluation Protocol Report:** [`reports/PHASE8D_EVALUATION_PROTOCOL.md`](file:///d:/trainidentificationaudio/reports/PHASE8D_EVALUATION_PROTOCOL.md)
