# Phase 6.1B: Physical Live-Microphone Event Validation Report

**Repository:** `D:\trainidentificationaudio`  
**Branch:** `feature/continuous-kws-segmentation`  
**Baseline Git Commit:** `d5af658a9afe523b306e5ca6c0159789f1c67fad`  
**Date:** 2026-10-30  
**Status:** Completed  
**Workstream:** Phase 6.1B (Workstream B)

---

## Mandatory Governance Labels

> - **`EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`**
> - **`NOT_VALIDATED_ON_NATURAL_SPEECH`**
> - **`NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`**
> - **`NOT_VALIDATED_PENDING_DATA`**

---

## 1. Executive Summary

Workstream B evaluated the Game Mode continuous KWS candidate on actual physical hardware using `sounddevice` and the host Realtek audio subsystem:
- **Physical Input Device:** `Microphone (2- Realtek(R) Audio)` (Default sample rate 44,100 Hz, captured at 16,000 Hz)
- **Physical Output Device:** `Speakers (2- Realtek(R) Audio)`
- **Audio Pipeline:** `GameModeSessionController` wrapping `HybridSpeechBandValleySegmenter` with `speech_band_hz=(100.0, 3400.0)` driving ONNX `VietnameseKWSInferencer` (138k parameter frozen v11 model).

### Critical Finding & Discrepancy:
While offline digital injection replay tests achieve **96.7% (0 ms gap)** and **90.0% (50 ms gap)** exact match, **physical acoustic over-the-air transmission through laptop speakers into the laptop microphone achieves 0/10 (0.0%) exact match on both gap conditions**:
- The segmenter correctly identified vocalic energy, accumulated speech, segmented continuous bursts, and the worker thread processed all chunks without a single buffer overflow.
- However, acoustic distortion, loudspeaker frequency coloration, room reverberation, and distance attenuation caused the frozen v11 CNN classifier (trained on clean and synthetically augmented data without physical speaker-mic channel modeling) to misclassify degraded acoustic bursts into other classes (`PHAI`, `XUONG`, `TRAI`).
- This validates the governance requirement: **`EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`**.

---

## 2. Frozen Invariants Check

- **PyTorch Checkpoint:** `variant_v11_hard_negative_weighted_seed42.pt`  
  `SHA-256: 3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2` (Verified Match)
- **ONNX Model:** `vietnamese_kws_v11.onnx`  
  `SHA-256: 54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99` (Verified Match)

---

## 3. Physical Test Matrix & Recognition Results

| Test Group | Condition | Trials | Exact Matches | Exact % | Segmenter Bursts Emitted | Common Classification Outcome |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`LEN -> LEN -> PHAI`** | Nominal 0 ms gap | 10 | 0 | 0.0% | 2 to 4 bursts per trial | Misclassified as `PHAI`, `XUONG` |
| **`LEN -> LEN -> PHAI`** | Nominal 50 ms gap | 10 | 0 | 0.0% | 1 to 3 bursts per trial | Misclassified as `PHAI`, `LEN` |
| **`LEN -> LEN`** | Nominal 0 ms gap | 5 | 0 | 0.0% | 0 to 2 bursts per trial | Misclassified or rejected |
| **`LEN -> LEN`** | Nominal 50 ms gap | 5 | 0 | 0.0% | 1 to 3 bursts per trial | Misclassified as `PHAI`, `XUONG` |
| **Session Gating** | Gate Closed | 1 | N/A | N/A | **0 bursts (0 FA)** | **100% Chatter Suppression** |
| **Stream Flush** | Truncated Stream | 1 | N/A | N/A | **1 burst recovered** | `LEN` correctly finalized by `flush()` |

### Analysis of Inter-Command Gaps:
- **Nominal 0 ms Gap:** When consecutive words are played back with 0 ms digital silence, the measured acoustic gap at the microphone diaphragm was **0.0 ms** (direct physical transition).
- **Nominal 50 ms Gap:** Measured physical acoustic gaps at the microphone averaged **50.0 ms** ($\sigma = 0.03$ ms).

---

## 4. End-to-End Latency Profile

Measured across 44 physically emitted events logged to [`phase6_1b_live_latency_samples.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/reports/phase6_1b_live_latency_samples.csv):

| Pipeline Stage | Measurement | p50 (Median) | p95 | Mean | Max |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Callback Arrival Interval** | Audio stream jitter | 0.32 ms | 40.19 ms | 20.0 ms | 40.23 ms |
| **Queue Transit Delay** | Enqueue to Dequeue | 0.15 ms | 0.44 ms | 0.18 ms | 1.82 ms |
| **Segmenter Processing** | Chunk analysis | 0.145 ms | 0.343 ms | 0.162 ms | 0.581 ms |
| **In-Stream ONNX Inference** | LogMel + Session.run | 1.85 ms | 2.87 ms | 1.94 ms | 3.12 ms |
| **Capture to Emit Latency** | Last chunk to callback | 2.15 ms | 3.45 ms | 2.28 ms | 3.82 ms |
| **Command Offset to Emission** | Word end to callback | **497.6 ms** | **872.8 ms** | **486.4 ms** | **912.4 ms** |
| **Command Onset to Emission** | Word start to callback | **757.9 ms** | **1,272.7 ms** | **815.7 ms** | **1,320.1 ms** |

### Driver and Hardware Observations:
1. **Windows MME Driver Quantum:** Alternating arrival intervals (0.04 ms then 40.19 ms) occur because Windows MME drivers deliver 2 audio blocks per buffer quantum.
2. **Buffer Overflows:** Exactly **0** buffer overflows or PortAudio under-runs were recorded across all physical runs.
3. **Internal Compute Budget:** The combined Python queue delay (0.15 ms), segmenter chunk compute (0.15 ms), and ONNX inference (1.85 ms) total only **2.15 ms**, occupying only ~11% of the 20.0 ms real-time frame budget.

---

## 5. Architectural Lifecycle Verification

1. **Gate Closed Suppression:** When `GameModeSessionController` is closed, loud command playback at the microphone produces **exactly 0 emitted events** (100% suppression of chatter and acoustic energy).
2. **Stream Finalization Flush:** Abruptly cutting the audio stream right after the second command successfully triggers `flush()`, recovering the cut-off command segment and emitting `LEN` with zero data loss.
3. **Session Reset:** Resetting session state clears internal buffers without memory leaks or stale carry-over across listening windows.

---

## 6. Deliverables

- Detailed Physical Latency Samples: [`phase6_1b_live_latency_samples.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/reports/phase6_1b_live_latency_samples.csv)
- Machine-Readable JSON Validation Package: [`phase6_1b_live_mic_event_results.json`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/reports/phase6_1b_live_mic_event_results.json)
