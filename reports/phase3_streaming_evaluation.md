# Phase 3 — Continuous Streaming Detector Evaluation Report

**Project:** Vietnamese Four-Word Keyword Spotting System (TRAI, PHAI, LEN, XUONG)  
**Evaluation Mode:** Production Streaming Inference (Sliding Window, Debounce, Re-arming)  
**Window Size:** 1.0 second ($16,000$ samples)  
**Hop Size:** 100 ms ($1,600$ samples) — 10 inferences per second  
**Debounce Window:** 600 ms lockout with $\tau_{rearm} = 0.30$  
**Temporal Confirmation:** 2 consecutive frames ($M = 2$)  
**Date:** 2026-10-12  
**Status:** **COMPLETE — COMPARATIVE CONTINUOUS AUDIT**  

---

## 1. Executive Summary & Methodology Comparison

Prior evaluations tested only isolated, pre-segmented 1.0-second files. In contrast, this benchmark evaluated the **actual real-time streaming detector** operating continuously over concatenated long-form audio streams:

- **Negative Vietnamese Speech Stream:** 6.5 minutes (390.0 s, 0.108 hours) of continuous native Vietnamese sentences from Google FLEURS test set.
- **Microphone Ambient Noise Stream:** 1.5 minutes (90.0 s, 0.025 hours) of continuous microphone handling, cable noise, and room acoustic rumble.
- **Combined Background Audio Duration:** **8.0 minutes (0.133 hours)**.
- **Total Sliding Window Inferences:** $> 4,800$ streaming inferences evaluated per threshold per model.

---

## 2. Multi-Threshold Streaming Operating Curves

### A. Champion Model: TC-ResNet-14

| Threshold ($	au$) | FA (Speech) | FA (Noise) | Combined FA/Hour | True Positives | Missed Rate | Duplicate Rate | Detection Latency |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.60** | 1 | 0 | **7.50 FA/hr** | 118/144 | **18.1%** | 0.0% | 679.5 ms |
| **0.70** | 0 | 0 | **0.00 FA/hr** | 117/144 | **18.8%** | 0.0% | 706.4 ms |
| **0.80** | 0 | 0 | **0.00 FA/hr** | 109/144 | **24.3%** | 0.0% | 745.5 ms |
| **0.85** | 0 | 0 | **0.00 FA/hr** | 93/144 | **35.4%** | 0.0% | 786.6 ms |
| **0.90** | 0 | 0 | **0.00 FA/hr** | 66/144 | **54.2%** | 0.0% | 807.2 ms |
| **0.95** | 0 | 0 | **0.00 FA/hr** | 51/144 | **64.6%** | 0.0% | 849.7 ms |

### B. Baseline Model: VietnameseKWSNet (2D-CNN)

| Threshold ($	au$) | FA (Speech) | FA (Noise) | Combined FA/Hour | True Positives | Missed Rate | Duplicate Rate | Detection Latency |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.60** | 0 | 0 | **0.00 FA/hr** | 125/144 | **13.2%** | 0.0% | 829.6 ms |
| **0.70** | 0 | 0 | **0.00 FA/hr** | 118/144 | **18.1%** | 0.0% | 880.7 ms |
| **0.80** | 0 | 0 | **0.00 FA/hr** | 88/144 | **38.9%** | 0.0% | 916.5 ms |
| **0.85** | 0 | 0 | **0.00 FA/hr** | 64/144 | **55.6%** | 0.0% | 901.8 ms |
| **0.90** | 0 | 0 | **0.00 FA/hr** | 48/144 | **66.7%** | 0.0% | 939.5 ms |
| **0.95** | 0 | 0 | **0.00 FA/hr** | 28/144 | **80.6%** | 0.0% | 955.5 ms |

---

## 3. Analysis & Key Findings

1. **Trade-Off Between Confidence Threshold and Missed Command Rate (MCR):**
   - At $\tau = 0.70$, TC-ResNet-14 achieves **117/144 detections** (18.8% Missed Rate) with **0 false alarms** observed in the 8.0-minute background audio, while slashing detection latency to **706.4 ms** (vs. 880.7 ms for Baseline).
   - For both architectures, increasing $\tau$ to $0.85$ causes missed commands to escalate (35.4% for TC-ResNet vs. 55.6% for Baseline) because probabilities fluctuate during boundary window hops.
   - Setting $\tau = 0.85$ or $0.90$ blindly in sliding window mode severely hurts user experience by dropping legitimate speech commands.

2. **False Alarm Rate Statistical Limitations:**
   - Evaluated pure negative background duration is **8.0 minutes (0.133 hours)**:
     - 6.5 minutes (390.0 s) continuous negative conversational Vietnamese speech (Google FLEURS test partition).
     - 1.5 minutes (90.0 s) continuous microphone ambient noise and handling rumble.
   - Total sliding window inferences evaluated: **4,800 inferences per threshold**.
   - Zero false alarms were observed (0 FA in 0.133 hr $\to 0.00$ FA/hr measured).
   - *Statistical Reality Check:* By the Rule of Three for Poisson zero-event observations, observing 0 events over 0.133 hours yields a 95% upper confidence bound of $\lambda_{95\%} \le \frac{3}{0.133} \approx 22.5\text{ FA/hour}$. Extrapolating a claim of "guaranteed 0.00 FA/hr in production" from an 8-minute stream is statistically unjustified. Production certification requires at least 10–50 hours of continuous negative speech.

3. **Duplicate Suppression & Debounce Efficacy:**
   - The 600 ms lockout with consecutive 2-frame confirmation ($M = 2$) completely prevented duplicate triggers across all operating thresholds (Duplicate Command Rate = 0.0%).

4. **Detection Latency:**
   - Average latency from command onset to trigger event was measured at **679.5 ms** (TC-ResNet at $\tau = 0.60$) and **829.6 ms** (Baseline at $\tau = 0.60$).
   - Since the sliding window length is 1,000 ms, detection occurs when the utterance is nearly complete in the rolling buffer.

---

## 4. Recommended Production Streaming Configuration

- **Champion Architecture:** `TC-ResNet-14`
- **Recommended Operating Threshold ($	au$):** `0.65 – 0.70`
  - High true command sensitivity
  - 0 false activations observed on test speech and noise streams
  - Low detection latency (< 900 ms)
- **Hop Size:** `100 ms` (10 inferences per second, CPU latency ~0.35 ms $	o < 4\%$ single-core CPU utilization)
- **Debounce Window:** `600 ms` with $\tau_{rearm} = 0.30$
- **Consecutive Frame Confirmation:** `2 frames` ($M = 2$, requires 100–200 ms sustained acoustic match)
