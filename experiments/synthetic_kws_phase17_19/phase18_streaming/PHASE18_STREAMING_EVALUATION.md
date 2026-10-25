# Phase 18 Streaming Detection and False-Trigger Reduction Report

**Date:** 2026-10-16  
**Auditor:** Antigravity (Advanced Agentic Coding Pair Programmer)  
**Location:** `experiments/synthetic_kws_phase17_19/phase18_streaming/`  
**Verdict:** **`PHASE18_PASS_WITH_LIMITATIONS`**  

---

## 1. Executive Summary

Phase 18 investigated and solved the streaming false-trigger problem identified in Phase 15 ($424.57\text{ false triggers/hour}$ on a 3-minute stream). 

The audit established that raw sliding-window classification with a static threshold ($\tau = 0.500$) causes boundary slicing artifacts: when non-target speech or room noise transitions enter or exit the 1.0s window, transitional frames briefly trigger top-1 confidence $> 0.500$. Without multi-frame persistence or refractory cooldown, every transition produces multiple false alarms.

To solve this, Phase 18 developed an offline event detector implementing **dual-threshold hysteresis with temporal smoothing and refractory cooldown**. Tuned strictly on validation streams, the new detector was evaluated on **1.00 hour (3,600.0 seconds) of continuous target-absent synthetic audio** across 4 independent 15-minute streams and a continuous target-present stream containing 50 target keyword instances.

### Key Streaming Results Summary

| Detection Method | Evaluated Target-Absent Audio | False Trigger Count | False Triggers / Hour | Target Recall (50 Targets) | Event Precision | Mean Detection Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline ($\tau = 0.50$, Raw Window)** | $3,600.0\text{ s}$ ($1.00\text{ hr}$) | 339 events | **$339.00\text{ FP/hr}$** | $100.00\%$ ($50/50$) | $45.87\%$ | $0.302\text{ s} \pm 0.079\text{ s}$ |
| **Method 1 (Elevated $\tau = 0.85$)** | Validation ($600.0\text{ s}$) | 2 events | $12.00\text{ FP/hr}$ | $100.00\%$ ($30/30$) | $93.75\%$ | $0.355\text{ s}$ |
| **Method 4 (Hysteresis + Cooldown)** | **$3,600.0\text{ s}$ ($1.00\text{ hr}$)** | **0 events** | **$0.00\text{ FP/hr}$** | **$100.00\%$ ($50/50$)** | **$100.00\%$** | **$0.496\text{ s} \pm 0.078\text{ s}$** |

---

## 2. Baseline Streaming Experiment Audit (18.1)

### Reconstruction of Phase 15 Baseline
* In Phase 15, evaluating a $186.54\text{ s}$ target-absent stream with a $100\text{ ms}$ hop generated $1,850$ sliding windows.
* At $\tau = 0.500$, raw window argmax fired on transitional boundary cuts between non-target phrases and pauses, producing 22 activations ($22 / 186.54 \times 3600 = 424.57\text{ FP/hr}$).
* In Phase 18, evaluating a full $1.00\text{-hour}$ stream under the identical baseline detector yielded **339 false triggers** ($339.00\text{ FP/hr}$), confirming that the Phase 15 observation was a real systemic property of raw static-window thresholding.

---

## 3. Continuous Evaluation Datasets (18.2)

All streaming audio was generated exclusively from the approved V5.2 candidate generator:
1. **Validation Corpus (10.0 minutes):**
   * $600.0\text{ s}$ target-absent stream with non-target speech, hum, room tone, and randomized pauses ($0.15\text{ s} - 0.60\text{ s}$).
   * $131.3\text{ s}$ target-present stream with 30 target keyword instances.
2. **Held-Out Test Corpus (1.00 Hour Target-Absent Audio):**
   * Four independent 15-minute streams ($900.0\text{ s}$ each = $3,600.0\text{ s}$ total) generated using test split `UNKNOWN` and `SILENCE` samples plus independent ambient noise realizations.
   * Total evaluation windows: **$35,960$ sliding windows** ($100\text{ ms}$ hop).
3. **Held-Out Test Target-Present Corpus:**
   * $196.0\text{ s}$ stream with 50 ground-truth target keyword instances (`TRAI`, `PHAI`, `LEN`, `XUONG`) placed at known randomized intervals among distractor words and pauses.

---

## 4. Detector Calibration & Grid Search (18.3)

Evaluated candidate configurations on the validation split:

| Configuration | Parameters | Val FP / Hour | Val Target Recall | Mean Latency |
| :--- | :--- | :---: | :---: | :---: |
| **Baseline** | $\tau=0.50$, $W=1$, $M=1$, $C=0\text{ms}$ | $252.00$ | $100.0\%$ | $0.285\text{ s}$ |
| **Method 1 (High Threshold)** | $\tau=0.85$, $W=1$, $M=1$, $C=300\text{ms}$ | $12.00$ | $100.0\%$ | $0.355\text{ s}$ |
| **Method 2 (Temporal Smoothing)** | $\tau=0.80$, $\tau_{off}=0.60$, $W=3$, $M=1$, $C=400\text{ms}$ | $0.00$ | $100.0\%$ | $0.425\text{ s}$ |
| **Method 3 (Consecutive Frames)** | $\tau=0.75$, $\tau_{off}=0.50$, $W=2$, $M=2$, $C=500\text{ms}$ | $0.00$ | $100.0\%$ | $0.528\text{ s}$ |
| **Method 4 (Recommended Hysteresis)** | **$\tau_{on}=0.80$, $\tau_{off}=0.35$, $W=3$, $M=2$, $C=600\text{ms}$** | **$0.00$** | **$100.0\%$** | **$0.525\text{ s}$** |

**Selection Rationale:** Method 4 was chosen because dual-threshold hysteresis ($\tau_{on}=0.80, \tau_{off}=0.35$) with 2-frame persistence and a $600\text{ ms}$ refractory period provides maximum noise-margin separation while preventing multiple window triggers for the same acoustic event.

---

## 5. Held-Out 1.00-Hour Evaluation Results (18.4)

Evaluating the frozen Method 4 detector on the held-out test streams:

### Target-Absent 1.00-Hour Stream (False Triggers)
* Stream 1 (15 min): **0** false triggers
* Stream 2 (15 min): **0** false triggers
* Stream 3 (15 min): **0** false triggers
* Stream 4 (15 min): **0** false triggers
* **Total Evaluated Audio:** $3,600.0\text{ s}$ ($1.00\text{ hr}$)
* **False Triggers / Hour:** **$0.00\text{ FP / Hour}$** (down from $339.00\text{ FP/hr}$).

### Target-Present Stream (Recall & Latency)
* Ground-Truth Target Keywords: 50
* True Positive Detections: **50 / 50 (100.00% Recall)**
* False Alarms during Stream: **0** (100.00% Event Precision)
* Missed Target Keywords: **0**
* Mean Detection Latency: **$0.496\text{ s} \pm 0.078\text{ s}$** (from keyword acoustic onset to detector trigger).

---

## 6. Scientific Limitations & Verdict

> [!WARNING]
> **Key Scientific Limitations:**
> 1. **Synthetic-Only Evaluation:** Streams consist of synthetic speech and parametric ambient noise. Real acoustic environments feature non-stationary human vocalizations, door slams, coughs, and microphonic clicks that are absent here.
> 2. **Latency Trade-Off:** Achieving $0.00\text{ FP/hr}$ required adding a 2-frame persistence constraint and 3-frame moving average, increasing average detection latency from $302\text{ ms}$ to $496\text{ ms}$.
> 3. **Human Validation Required:** The detector parameters must be re-tuned on in-domain natural speech streams once formal authorization is granted.

### Phase 18 Verdict: **`PHASE18_PASS_WITH_LIMITATIONS`**
