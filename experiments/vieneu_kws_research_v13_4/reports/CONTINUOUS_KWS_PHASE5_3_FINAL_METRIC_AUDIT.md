# Phase 5.3 — Final Metric Semantics Audit Report

**Repository:** `D:\trainidentificationaudio`  
**Branch:** `feature/continuous-kws-segmentation`  
**Date:** 2026-10-29  
**Audit Decision:** `METRICS_VERIFIED`  

---

## Executive Summary & Governance Attestation

> [!IMPORTANT]
> **Mandatory Governance Attestation**  
> All work in Phase 5.3 was conducted strictly as an evaluation-integrity audit resolving metric semantics and latency measurement boundaries. No models were trained, neural network parameters were not enlarged, the released v11 checkpoint (`3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2`) and ONNX model (`54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99`) remained frozen and verified, thresholds were untouched, and production defaults were preserved.
>
> The four mandatory governance labels are strictly preserved:
> 1. `EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`
> 2. `NOT_VALIDATED_ON_NATURAL_SPEECH`
> 3. `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`
> 4. `NOT_VALIDATED_PENDING_DATA`

### Summary of Audit Resolutions
1. **Explicit Separation of Sequence WER and Temporal Event Metrics:**  
   Implemented two formally decoupled evaluation pipelines:
   - **Path A (Sequence WER):** Standard Levenshtein dynamic programming alignment on ordered label strings without timestamps, reporting $S_{\text{seq}}$, $D_{\text{seq}}$, $I_{\text{seq}}$, and $\text{WER}_{\text{seq}} = (S_{\text{seq}} + D_{\text{seq}} + I_{\text{seq}}) / N_{\text{gt}}$.
   - **Path B (Temporal Event Metrics):** Monotonic One-to-One Alignment via DP with documented tolerance window $[-150\text{ ms}, +550\text{ ms}]$, reporting $\text{TP}_{\text{event}}$, $\text{FP}_{\text{event}}$, $\text{FN}_{\text{event}}$, $\text{Precision}_{\text{event}}$, $\text{Recall}_{\text{event}}$, and $\text{F1}_{\text{event}}$. A wrong-label detection inside a GT window is a Substitution, contributing to **both 1 FP and 1 FN** ($\text{FP} = I + S, \text{FN} = D + S$).
   - Legitimate discrepancies between the two paths are preserved and documented rather than forced to artificially agree.
2. **Correct Zero-Ground-Truth Reporting ($N_{\text{gt}} = 0$):**  
   When $N_{\text{gt}} = 0$ (e.g. `game_mode_idle` gate closed), Precision, Recall, F1, Sequence WER, and Exact Match are reported as **`N/A`** rather than $0.0\%$ or $100.0\%$, eliminating false claims of recognition accuracy when the classifier was never invoked.
3. **Audit of Latency Ground-Truth Association:**  
   Verified that latencies ($t_{\text{lag}}$, $t_{\text{e2e}}$, $t_{\text{onset\_to\_emit}}$) are derived **strictly from the one-to-one matched pairs $\mathcal{M} = \{(j, i)\}$** from the alignment algorithm. Independent nearest-neighbor lookups are strictly prohibited.
4. **Replay-Derived Latency vs. Live Microphone Timing:**  
   Clearly demarcated the benchmark latency as **offline replay-derived latency** ($45.0\text{ ms}$ lag p50 / $47.32\text{ ms}$ e2e p50). Executed a controlled live-microphone benchmark on Windows using `sounddevice` with monotonic timestamps (`time.perf_counter()`), measuring callback jitter ($0.44\text{ ms}$ p50 / $40.25\text{ ms}$ p95), queue transit delay ($0.25\text{ ms}$ p50), and chunk processing latency ($0.245\text{ ms}$ p50).

---

## 1. Separate Sequence WER from Temporal Event Metrics

### 1.1 Specification of the Two Evaluation Paths

```
+-------------------------------------------------------------------------------------------------+
|                                 TWO SEPARATE EVALUATION PATHS                                   |
+-------------------------------------------------------------------------------------------------+
| Path A: Sequence WER (Standard Levenshtein)                                                     |
| Input: Reference String R = [r_0, ..., r_{M-1}], Hypothesis String H = [h_0, ..., h_{K-1}]     |
| Algorithm: Dynamic Programming Minimum Edit Distance                                            |
| Metrics: S_seq, D_seq, I_seq, WER_seq = (S_seq + D_seq + I_seq) / N_gt                          |
| Exact Match: H == R (Boolean)                                                                   |
| Timing: Completely ignored; evaluates only symbolic sequence order.                             |
|-------------------------------------------------------------------------------------------------|
| Path B: Temporal Event Metrics (Monotonic 1-to-1 Alignment)                                     |
| Input: Ground Truth Intervals G = [{cmd, t_start, t_end}], Detections D = [{cmd, t_det, infer}]|
| Algorithm: Monotonic Dynamic Programming with Tolerance Window [-150 ms, +550 ms]               |
| Invariants: Strict injection (<=1 GT per det, <=1 det per GT); strict temporal monotonicity.    |
| Metrics: TP_event, FP_event = I + S, FN_event = D + S, Precision, Recall, F1                    |
| Timing: Enforces temporal causality and syllable boundary proximity.                            |
+-------------------------------------------------------------------------------------------------+
```

### 1.2 Formal Audit of the DP Reward Scheme
The temporal alignment algorithm optimizes the global objective:
$$\max_{\mathcal{M}} \sum_{(j, i) \in \mathcal{M}} \left( R(c_j, g_i) - |t_{\text{det}, j} - t_{\text{end}, i}| \right) - \sum_{j \notin \mathcal{M}} P_{\text{ins}} - \sum_{i \notin \mathcal{M}} P_{\text{del}}$$
where:
$$R(c_j, g_i) = \begin{cases} +1000.0, & \text{if } c_j = g_i \text{ (True Positive label match)} \\ +500.0, & \text{if } c_j \neq g_i \text{ (Substitution label mismatch)} \end{cases}$$
and $P_{\text{ins}} = P_{\text{del}} = 10.0$, subject to $-0.150\text{ s} \le t_{\text{det}, j} - t_{\text{end}, i} \le 0.550\text{ s}$.

#### Invariants & Behavioral Verification:
1. **Label Match Prioritization:**  
   Because $1000 - |\Delta t| \gg 500 - |\Delta t|$ (a difference of $\approx 500$ points vs max $|\Delta t| \le 0.55$), the optimizer will **always choose a True Positive label match over a Substitution** whenever both are within the tolerance window.
2. **Substitution Grouping over Spurious Insertions/Deletions:**  
   Because matching a wrong-label detection yields $+500 - |\Delta t| \approx +499.5$, while leaving both unmatched incurs $-P_{\text{ins}} - P_{\text{del}} = -20.0$, the optimizer correctly classifies a wrong-label detection inside a ground-truth window as a **Substitution** ($S_{\text{event}} = 1, \text{FP}=1, \text{FN}=1$) rather than splitting it into an arbitrary insertion and deletion ($I=1, D=1$).
3. **Strict Injection & Monotonicity:**  
   State transitions in the DP memoization `dp[(j, i)]` only allow $(j+1, i+1)$, $(j+1, i)$, or $(j, i+1)$. This mathematically guarantees that no ground-truth command can match more than one detection, no detection can match more than one ground-truth command, and index order is strictly non-decreasing.

---

## 2. Recomputed Benchmark: Sequence vs. Event Metrics

Audited data from [`phase5_3_sequence_vs_event_metrics.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/reports/phase5_3_sequence_vs_event_metrics.csv):

| Profile | $\theta_{\text{cmd}}$ | Scenario | $N_{\text{gt}}$ | $S_{\text{seq}}$ | $D_{\text{seq}}$ | $I_{\text{seq}}$ | $\text{WER}_{\text{seq}}$ | Exact Match | $\text{TP}_{\text{ev}}$ | $\text{FP}_{\text{ev}}$ | $\text{FN}_{\text{ev}}$ | $S_{\text{ev}}$ | $I_{\text{ev}}$ | $D_{\text{ev}}$ | $\text{Prec}_{\text{ev}}$ | $\text{Rec}_{\text{ev}}$ | $\text{F1}_{\text{ev}}$ | Discrepancy? |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `legacy_v11` | 0.20 | always_open | 170 | 10 | 105 | 0 | 67.6% | 5/65 (7.7%) | 55 | 10 | 115 | 10 | 0 | 105 | 84.6% | 32.4% | 46.9% | False |
| `continuous_valley` | 0.20 | always_open | 170 | 4 | 0 | 0 | 2.4% | 61/65 (93.8%) | 166 | 4 | 4 | 4 | 0 | 0 | 97.6% | 97.6% | 97.6% | False |
| `hybrid_speechband_valley` | 0.20 | always_open | 170 | 2 | 4 | 0 | 3.5% | 59/65 (90.8%) | 164 | 2 | 6 | 2 | 0 | 4 | 98.8% | 96.5% | 97.6% | False |
| `hybrid_speechband_valley` | 0.30 | always_open | 170 | 2 | 4 | 0 | 3.5% | 59/65 (90.8%) | 164 | 2 | 6 | 2 | 0 | 4 | 98.8% | 96.5% | 97.6% | False |
| `hybrid_speechband_valley` | 0.40 | always_open | 170 | 2 | 4 | 0 | 3.5% | 59/65 (90.8%) | 164 | 2 | 6 | 2 | 0 | 4 | 98.8% | 96.5% | 97.6% | False |
| `hybrid_speechband_valley` | 0.50 | always_open | 170 | 2 | 9 | 0 | 6.5% | 54/65 (83.1%) | 159 | 2 | 11 | 2 | 0 | 9 | 98.8% | 93.5% | 96.1% | False |
| `hybrid_speechband_valley` | 0.60 | always_open | 170 | 0 | 17 | 0 | 10.0% | 50/65 (76.9%) | 153 | 0 | 17 | 0 | 0 | 17 | 100.0% | 90.0% | 94.7% | False |
| `game_mode_idle` | 0.20 | gate_closed | 0 | 0 | 0 | 0 | **N/A** | **N/A** | 0 | 0 | 0 | 0 | 0 | 0 | **N/A** | **N/A** | **N/A** | False |
| `game_mode_active` (N=3, 0ms) | 0.20 | active_session | 30 | 2 | 2 | 0 | 13.3% | 6/10 (60.0%) | 26 | 2 | 4 | 2 | 0 | 2 | 92.9% | 86.7% | 89.7% | False |
| `game_mode_active` (N=3, 50ms) | 0.20 | active_session | 30 | 2 | 2 | 0 | 13.3% | 6/10 (60.0%) | 26 | 2 | 4 | 2 | 0 | 2 | 92.9% | 86.7% | 89.7% | False |
| `game_mode_active` (N=5, 0ms) | 0.20 | active_session | 50 | 1 | 8 | 0 | 18.0% | 5/10 (50.0%) | 41 | 1 | 9 | 1 | 0 | 8 | 97.6% | 82.0% | 89.1% | False |
| `game_mode_active` (N=5, 50ms) | 0.20 | active_session | 50 | 2 | 3 | 0 | 10.0% | 6/10 (60.0%) | 45 | 2 | 5 | 2 | 0 | 3 | 95.7% | 90.0% | 92.8% | False |
| `game_mode_active` (N=10, 0ms) | 0.20 | active_session | 100 | 2 | 13 | 0 | 15.0% | 1/10 (10.0%) | 85 | 2 | 15 | 2 | 0 | 13 | 97.7% | 85.0% | 90.9% | False |
| `game_mode_active` (N=10, 50ms) | 0.20 | active_session | 100 | 5 | 5 | 0 | 10.0% | 2/10 (20.0%) | 90 | 5 | 10 | 5 | 0 | 5 | 94.7% | 90.0% | 92.3% | False |
| `game_mode_active` (N=20, 0ms) | 0.20 | active_session | 200 | 3 | 27 | 0 | 15.0% | 0/10 (0.0%) | 169 | 4 | 31 | 4 | 0 | 27 | 97.7% | 84.5% | 90.6% | **True** |
| `game_mode_active` (N=20, 50ms) | 0.20 | active_session | 200 | 9 | 9 | 0 | 9.0% | 0/10 (0.0%) | 182 | 9 | 18 | 9 | 0 | 9 | 95.3% | 91.0% | 93.1% | False |

### Analysis of Preserved Discrepancy
In Row 15 (`active_session_N=20_gap=0ms`), a legitimate discrepancy exists:
- Standard Sequence Levenshtein found an optimal edit path with **$S_{\text{seq}} = 3$ substitutions** and $D_{\text{seq}} = 27$ deletions ($\text{WER}_{\text{seq}} = 15.0\%$).
- Temporal Event Matching enforced the $[-150\text{ ms}, +550\text{ ms}]$ time window, finding **$S_{\text{event}} = 4$ substitutions** and $D_{\text{event}} = 27$ deletions.
- Because one delayed event fell outside its own GT window and matched an adjacent window in time, the temporal constraint increased substitution count by 1. This discrepancy is **preserved and explicitly reported**, demonstrating the semantic separation between string edit distance and temporal event detection.

---

## 3. Zero-Ground-Truth ($N_{\text{gt}} = 0$) & Negative Control Policy

### 3.1 Policy for Closed-Gate Condition
When $N_{\text{gt}} = 0$ (such as `game_mode_idle` gate closed):
- Denominators for Recall ($N_{\text{gt}}$) and Precision ($\text{TP} + \text{FP}$) are zero.
- **Reporting Rule:** Reported as **`N/A`**.
- Zero false activations during closed gate is verified as **architectural session gating**, not classifier recognition accuracy.

### 3.2 Negative Control False Alarm Rate Quarantined
Negative control audio is strictly quarantined from positive-sequence WER calculations:

| Negative Audio Category | Total Duration (s) | Total Duration (min) | False Activations (FA) | False Alarm Rate (FAR / min) | Evaluation Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Digital Silence** | 300.0 s | 5.00 min | 0 | **0.00 / min** | Clean Silence Suppression Verified |
| **Ambient Background Noise** | 100.0 s | 1.67 min | 0 | **0.00 / min** | Acoustic Noise Gate Verified |
| **Vietnamese Conversation** | 300.0 s | 5.00 min | 379 | **75.80 / min** | Hybrid Gate Suppression (Reduced from 136.0/min) |
| **Out-of-Vocabulary Foils** | 200.0 s | 3.33 min | 284 | **85.20 / min** | Reject Class Discrimination Active |
| **Game Mode Closed Gate (Conv)**| 300.0 s | 5.00 min | 0 | **0.00 / min** | Architectural Session Gate Verified |

---

## 4. Latency Ground-Truth Association & Real-time Live Mic Timing

### 4.1 Strict One-to-One Latency Association
All latency measurements are derived **strictly from the matched pairs $\mathcal{M} = \{(j, i)\}$**:
$$t_{\text{lag}} = \max(0.0, (t_{\text{det}, j} - t_{\text{offset}, i}) \times 1000.0)$$
$$t_{\text{e2e}} = t_{\text{lag}} + t_{\text{infer}, j}$$
$$t_{\text{onset\_to\_emit}} = \max(0.0, (t_{\text{det}, j} - t_{\text{onset}, i}) \times 1000.0) + t_{\text{infer}, j}$$
Unmatched insertions ($j \notin \mathcal{M}$) and deletions ($i \notin \mathcal{M}$) are strictly excluded from latency arrays. Independent nearest-neighbor lookups were audited and confirmed eliminated.

### 4.2 Replay-Derived Latency vs. Live Microphone Timing
- **Offline Replay-Derived Latency (`hybrid_speechband_valley`, $\theta_{\text{cmd}}=0.20$, $N=166$):**
  - Acoustic Response Lag ($t_{\text{lag}}$): **$45.00\text{ ms}$ (p50)**, **$130.00\text{ ms}$ (p95)**.
  - In-Stream Inference ($t_{\text{infer}}$): **$2.21\text{ ms}$ (p50)**, **$2.44\text{ ms}$ (p95)**.
  - End-to-End Latency ($t_{\text{e2e}}$): **$47.32\text{ ms}$ (p50)**, **$132.42\text{ ms}$ (p95)**.
  - Onset-to-Emission Latency ($t_{\text{onset\_to\_emit}}$): **$382.13\text{ ms}$ (p50)**, **$517.41\text{ ms}$ (p95)**.
  - *Measurement Boundaries:* Synthetic 16 kHz stream fed chunk-by-chunk to the segmenter; latency measured from ground-truth stream markers to synchronous callback event emission.
- **Controlled Live-Microphone Timing Test (3.0s Live Capture via `sounddevice`):**
  - Chunks Captured: 148 blocks ($320\text{ samples} / 20\text{ ms}$ at 16 kHz).
  - Audio Callback Period: **$0.44\text{ ms}$ (p50)**, **$40.25\text{ ms}$ (p95)** (min: $0.03\text{ ms}$, max: $40.46\text{ ms}$).
  - Queue Transit Delay: **$0.25\text{ ms}$ (p50)**, **$0.78\text{ ms}$ (p95)**.
  - Chunk Processing Time: **$0.245\text{ ms}$ (p50)**, **$0.587\text{ ms}$ (p95)**.
  - *Measurement Limitations:* On Windows, the MME/DirectSound audio driver delivers audio blocks in 40 ms quanta (bursts of 2 chunks), introducing up to 20-40 ms of audio driver buffering jitter prior to Python callback invocation.

---

## 5. Verification Commands and Unit Test Results

### 1. Phase 5.3 Unit Tests
```powershell
.venv\Scripts\pytest tests/test_metric_integrity_phase5_3.py -v
```
Output:
```
tests/test_metric_integrity_phase5_3.py::TestPhase53MetricSemantics::test_levenshtein_wer_vs_temporal_event_matching_discrepancy PASSED [ 16%]
tests/test_metric_integrity_phase5_3.py::TestPhase53MetricSemantics::test_overlapping_timing_windows_and_repeated_commands PASSED [ 33%]
tests/test_metric_integrity_phase5_3.py::TestPhase53MetricSemantics::test_wrong_label_detection_classification PASSED [ 50%]
tests/test_metric_integrity_phase5_3.py::TestPhase53MetricSemantics::test_zero_ground_truth_policy PASSED [ 66%]
tests/test_metric_integrity_phase5_3.py::TestPhase53MetricSemantics::test_multiple_detections_competing_for_one_ground_truth PASSED [ 83%]
tests/test_metric_integrity_phase5_3.py::TestPhase53MetricSemantics::test_latency_association_strictly_via_matched_pairs PASSED [100%]
============================== 6 passed in 2.47s ==============================
```

### 2. Full Regression Suite Execution
```powershell
.venv\Scripts\pytest tests/test_continuous_kws_phase2.py tests/test_continuous_kws_phase4.py tests/test_continuous_kws_phase5.py tests/test_metric_integrity_phase5_1.py tests/test_metric_integrity_phase5_2.py tests/test_metric_integrity_phase5_3.py -v
```
Output:
```
============================= 53 passed in 3.30s ==============================
```

---

## 6. Audit Decision & Artifact Index

### Final Audit Decision: `METRICS_VERIFIED`

1. **Master Audit Report:** [CONTINUOUS_KWS_PHASE5_3_FINAL_METRIC_AUDIT.md](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/reports/CONTINUOUS_KWS_PHASE5_3_FINAL_METRIC_AUDIT.md) (and root copy [CONTINUOUS_KWS_PHASE5_3_FINAL_METRIC_AUDIT.md](file:///d:/trainidentificationaudio/CONTINUOUS_KWS_PHASE5_3_FINAL_METRIC_AUDIT.md))
2. **Sequence vs. Event Metrics CSV:** [phase5_3_sequence_vs_event_metrics.csv](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/reports/phase5_3_sequence_vs_event_metrics.csv)
3. **Master Audit JSON:** [phase5_3_metric_audit.json](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/reports/phase5_3_metric_audit.json)
4. **Unit Test Suite:** [`tests/test_metric_integrity_phase5_3.py`](file:///d:/trainidentificationaudio/tests/test_metric_integrity_phase5_3.py)
5. **Summary Artifact:** [phase5_3_final_metric_semantics_summary.md](file:///C:/Users/DELL/.gemini/antigravity/brain/d5db8723-9b87-46b0-b848-ad096e5f6344/phase5_3_final_metric_semantics_summary.md)
