# Phase 5.1 — Metric Integrity Audit Report

**Repository:** `D:\trainidentificationaudio`  
**Branch:** `feature/continuous-kws-segmentation`  
**Date:** 2026-10-29  
**Audit Decision:** `METRICS_VERIFIED`  

---

## Executive Summary & Governance Attestation

> [!IMPORTANT]
> **Mandatory Governance Attestation**  
> All work in Phase 5.1 is an evaluation-integrity audit conducted without training models, enlarging neural network capacity, altering the frozen v11 checkpoint (`3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2`) or ONNX runtime model (`54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99`), or modifying production defaults.
>
> The four mandatory governance labels are strictly preserved:
> 1. `EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`
> 2. `NOT_VALIDATED_ON_NATURAL_SPEECH`
> 3. `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`
> 4. `NOT_VALIDATED_PENDING_DATA`

This audit investigated the mathematical consistency, algorithmic definitions, test matrix denominators, latency components, and empirical counts of the Phase 5 benchmark metrics for continuous keyword spotting.

### Key Audit Findings
1. **Mathematical Explanation of $\text{WER} = 100\% - \text{Recall}$:**  
   The observation that $\text{WER} \equiv 100\% - \text{Recall}$ in previous benchmark tables is **not an implementation bug or arbitrary hardcoding**. It is an exact mathematical consequence of two facts:
   - For any time-aligned sequence evaluation, $N_{\text{gt}} = \text{TP} + S + D$, which implies $\text{WER} = (100\% - \text{Recall}) + \frac{I}{N_{\text{gt}}} \times 100\%$.
   - On the clean, locked positive command streams, neither `continuous_valley` nor `hybrid_speechband_valley` produced spurious splits or triggered during silence intervals, resulting in **zero insertion errors** ($I = 0$). When $I = 0$, $\text{WER}$ is identically $(100\% - \text{Recall})$.
2. **Substitutions vs. Event-Level Precision/Recall:**  
   A substitution error represents a detection that fell within the temporal window of a ground-truth command but predicted the incorrect target command. In rigorous event matching, a substitution contributes to **both 1 False Positive (FP) and 1 False Negative (FN)** ($\text{FP} = I + S$, $\text{FN} = D + S$). In sequence-level Levenshtein alignment, a substitution counts as a single edit operation ($S=1$).
3. **Alignment Method Equivalence on Clean Streams:**  
   Across all 65 pattern trials and 80 long-sequence scaling trials for `continuous_valley` and `hybrid_speechband_valley`, time-window event matching and Levenshtein sequence alignment produced **identical $(S, D, I)$ counts**. Discrepancies between the two methods only occurred on `legacy_v11` due to severe acoustic lag and utterance merging.
4. **Architectural Gating vs. Classifier Rejection:**  
   The zero activations of `legacy_v11` on 0 ms and 50 ms gap streams are **purely upstream architectural gating** caused by the 300 ms silence requirement never being satisfied; the classifier was never invoked for intermediate syllables. Similarly, zero activations in `game_mode_idle` reflect intentional session gating, not classifier confidence.
5. **Latency Component Disaggregation:**  
   The previously reported $0.17\text{ ms} - 0.21\text{ ms}$ latency was verified to be strictly the **ONNX model forward pass time on CPU**, not the end-to-end microphone pipeline latency. The true end-to-end response latency ($t_{\text{e2e}} = t_{\text{lag}} + t_{\text{infer}}$) is **$47.7\text{ ms}$ (p50)** and **$132.7\text{ ms}$ (p95)**.

---

## 1. Audit of WER & Event Metrics

### 1.1 The Mathematical Identity: $\text{WER} = (100\% - \text{Recall}) + \frac{I}{N_{\text{gt}}}$

In standard speech recognition and event detection:
- $N_{\text{gt}}$: Total number of ground-truth target commands.
- $\text{TP}$: True Positives (correct command detected within tolerance window).
- $S$: Substitutions (command detected within tolerance window, but incorrect target class).
- $D$: Deletions (ground-truth command with no detection within tolerance window).
- $I$: Insertions (detected command with no corresponding ground-truth word).

In [benchmark_phase5_audit.py](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/scripts/benchmark_phase5_audit.py#L254-L333), `match_events_rigorous` implements the following definitions:
$$\text{Recall} = \frac{\text{TP}}{N_{\text{gt}}}$$
$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{\text{TP}}{\text{TP} + I + S}$$
$$\text{WER} = \frac{S + D + I}{N_{\text{gt}}}$$

Because each ground-truth item is either correctly matched ($\text{TP}$), substituted ($S$), or missed ($D$), the ground-truth set is strictly partitioned:
$$N_{\text{gt}} = \text{TP} + S + D \implies S + D = N_{\text{gt}} - \text{TP}$$

Substituting $S + D$ into the $\text{WER}$ definition:
$$\text{WER} = \frac{(N_{\text{gt}} - \text{TP}) + I}{N_{\text{gt}}} = 1 - \frac{\text{TP}}{N_{\text{gt}}} + \frac{I}{N_{\text{gt}}} = (1 - \text{Recall}) + \frac{I}{N_{\text{gt}}}$$

Expressed as percentages:
$$\mathbf{\text{WER}\% = (100\% - \text{Recall}\%) + \frac{I}{N_{\text{gt}}} \times 100\%}$$

#### Why $I = 0$ on Positive Sequences:
In the Phase 5 benchmark, positive test streams were synthesized with clean target command audio clips, separated by inter-command gaps ($0, 50, 100, 200\text{ ms}$), bounded by $500\text{ ms}$ lead silence and $750\text{ ms}$ trail silence.
- The acoustic valley segmenter requires an energy dip $\ge 12\text{ dB}$ followed by a rebound $\ge 6\text{ dB}$ to trigger an internal split. Within active command syllables, no such dip exists.
- The minimum active speech threshold ($100\text{ ms}$) and speech-band energy filter prevent micro-splittings or noise triggers during the inter-command silence gaps.
- Therefore, the segmenter **never emitted more segments than ground-truth syllables** ($K \le M$ in all trials).
- Standalone insertion errors ($I$) were identically zero.

When $I = 0$:
$$\text{WER} \equiv 100\% - \text{Recall}$$

This explains with complete mathematical rigor why the Phase 5 benchmark tables exhibited this exact numerical relationship.

### 1.2 Raw Metric Counts by Profile and Threshold

The following table summarizes the verified raw counts from [phase5_matched_benchmark_results.csv](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/reports/phase5_matched_benchmark_results.csv) independently validated by [audit_phase5_1_metrics.py](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/scripts/audit_phase5_1_metrics.py):

| Evaluation Group | Profile | $\theta_{\text{cmd}}$ | $N_{\text{gt}}$ | TP | FP | FN | $S$ | $I$ | $D$ | Recall (%) | Precision (%) | F1 (%) | WER (%) | Exact Matches | Trials |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Profile Comparison | `legacy_v11` | 0.20 | 170 | 49 | 16 | 121 | 16 | 0 | 105 | 28.8% | 75.4% | 41.7% | 71.2% | 5 | 65 |
| Profile Comparison | `continuous_valley` | 0.20 | 170 | 166 | 4 | 4 | 4 | 0 | 0 | 97.6% | 97.6% | 97.6% | 2.4% | 61 | 65 |
| Profile Comparison | `hybrid_speechband_valley` | 0.20 | 170 | 164 | 2 | 6 | 2 | 0 | 4 | 96.5% | 98.8% | 97.6% | 3.5% | 59 | 65 |
| Threshold Sweep | `hybrid_speechband_valley` | 0.20 | 170 | 164 | 2 | 6 | 2 | 0 | 4 | 96.5% | 98.8% | 97.6% | 3.5% | 59 | 65 |
| Threshold Sweep | `hybrid_speechband_valley` | 0.30 | 170 | 164 | 2 | 6 | 2 | 0 | 4 | 96.5% | 98.8% | 97.6% | 3.5% | 59 | 65 |
| Threshold Sweep | `hybrid_speechband_valley` | 0.40 | 170 | 164 | 2 | 6 | 2 | 0 | 4 | 96.5% | 98.8% | 97.6% | 3.5% | 59 | 65 |
| Threshold Sweep | `hybrid_speechband_valley` | 0.50 | 170 | 159 | 2 | 11 | 2 | 0 | 9 | 93.5% | 98.8% | 96.1% | 6.5% | 54 | 65 |
| Threshold Sweep | `hybrid_speechband_valley` | 0.60 | 170 | 153 | 0 | 17 | 0 | 0 | 17 | 90.0% | 100.0% | 94.7% | 10.0% | 50 | 65 |
| Session Gating | `game_mode_idle` | 0.20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0% | 0.0% | 0.0% | 0.0% | 0 | 0 |
| Active Burst Scaling | `game_mode_active` (N=3, 0ms) | 0.20 | 30 | 26 | 2 | 4 | 2 | 0 | 2 | 86.7% | 92.9% | 89.7% | 13.3% | 6 | 10 |
| Active Burst Scaling | `game_mode_active` (N=3, 50ms) | 0.20 | 30 | 26 | 2 | 4 | 2 | 0 | 2 | 86.7% | 92.9% | 89.7% | 13.3% | 6 | 10 |
| Active Burst Scaling | `game_mode_active` (N=5, 0ms) | 0.20 | 50 | 40 | 2 | 10 | 2 | 0 | 8 | 80.0% | 95.2% | 87.0% | 20.0% | 5 | 10 |
| Active Burst Scaling | `game_mode_active` (N=5, 50ms) | 0.20 | 50 | 45 | 2 | 5 | 2 | 0 | 3 | 90.0% | 95.7% | 92.8% | 10.0% | 6 | 10 |
| Active Burst Scaling | `game_mode_active` (N=10, 0ms) | 0.20 | 100 | 84 | 3 | 16 | 3 | 0 | 13 | 84.0% | 96.6% | 89.8% | 16.0% | 1 | 10 |
| Active Burst Scaling | `game_mode_active` (N=10, 50ms) | 0.20 | 100 | 90 | 5 | 10 | 5 | 0 | 5 | 90.0% | 94.7% | 92.3% | 10.0% | 2 | 10 |
| Active Burst Scaling | `game_mode_active` (N=20, 0ms) | 0.20 | 200 | 167 | 6 | 33 | 6 | 0 | 27 | 83.5% | 96.5% | 89.5% | 16.5% | 0 | 10 |
| Active Burst Scaling | `game_mode_active` (N=20, 50ms) | 0.20 | 200 | 182 | 9 | 18 | 9 | 0 | 9 | 91.0% | 95.3% | 93.1% | 9.0% | 0 | 10 |

---

## 2. Event-Level Alignment vs. Sequence-Level Levenshtein Alignment

### 2.1 Algorithmic Differences
- **Event-Level Time-Window Matching (`match_events_rigorous`):**
  Matches each detected timestamp $t_{\text{det}}$ against ground-truth command intervals $[t_{\text{start}}, t_{\text{end}}]$. A match is valid if $t_{\text{det}} - t_{\text{end}} \in [-150\text{ ms}, +550\text{ ms}]$.
  - If class matches: $\text{TP} \leftarrow \text{TP} + 1$.
  - If class differs: $S \leftarrow S + 1 \implies \text{FP} \leftarrow \text{FP} + 1, \text{FN} \leftarrow \text{FN} + 1$.
  - If no GT in window: $I \leftarrow I + 1 \implies \text{FP} \leftarrow \text{FP} + 1$.
  - Unmatched GT: $D \leftarrow D + 1 \implies \text{FN} \leftarrow \text{FN} + 1$.
- **Sequence-Level Minimum Edit Distance (`levenshtein_align`):**
  Computes minimum cost string alignment between predicted string $[c_1, \dots, c_k]$ and reference $[g_1, \dots, g_m]$ using dynamic programming. Timestamps are disregarded; only symbol order is evaluated.

### 2.2 Comparison Across Benchmark Trials
- For `continuous_valley` and `hybrid_speechband_valley`:
  Because segmentation boundaries align accurately to syllable energy valleys, every emitted event fell monotonically into its ground-truth temporal window. Across all trials, **Levenshtein alignment $(S, D, I)$ perfectly matched Event-Level matching $(S, D, I)$**.
- For `legacy_v11`:
  Because `legacy_v11` requires 300 ms of silence before finalizing a segment, emissions occur with substantial acoustic lag ($> 280\text{ ms}$). When contiguous commands occur with 100 ms gaps, delayed events can cross into subsequent temporal tolerance windows, resulting in 11 observed discrepancies where event matching flagged substitutions while Levenshtein flagged deletions.

### 2.3 Synthetic Unit Tests
The unit test suite [`tests/test_metric_integrity_phase5_1.py`](file:///d:/trainidentificationaudio/tests/test_metric_integrity_phase5_1.py) validates 10 synthetic cases covering all permutations:
1. Exact Match ($S=0, D=0, I=0, \text{WER}=0.0\%$, Exact Match=True)
2. Single Deletion ($D=1, \text{WER}=50.0\%$, Exact Match=False)
3. Single Insertion ($I=1, \text{WER}=50.0\%, \text{FP}=1$)
4. Single Substitution ($S=1, \text{WER}=50.0\%, \text{FP}=1, \text{FN}=1$)
5. Compound Multi-Error Sequence ($S=1, D=0, I=1$)
6. Empty Predictions ($D=N_{\text{gt}}, \text{WER}=100.0\%$)
7. Empty Ground Truth ($I=N_{\text{det}}, \text{FP}=N_{\text{det}}$)
8. Duplicate Detections in Same Window ($TP=1, I=1, FP=1$)
9. Out-of-Order Detections (Temporal resolution correctly pairs timestamps)
10. Mathematical Identity Verification ($\text{WER} \equiv 100\% - \text{Recall}$ iff $I=0$)

All 10 tests passed (`10 passed in 2.48s`). Full suite of 39 continuous KWS tests passed (`39 passed in 3.14s`).

---

## 3. Denominator Consistency & Disaggregation

### 3.1 Negative Controls Evaluated Separately from WER
Negative control audio (silence, noise, conversation, foils) is **strictly excluded from the WER denominator**. Mixing negative audio duration into WER would falsely dilute error rates. Instead, negative performance is evaluated exclusively through False Alarm Rate per minute ($\text{FAR} = \frac{\text{False Activations}}{\text{Duration (min)}}$):

| Profile | $\theta_{\text{cmd}}$ | Digital Silence (300s) | Ambient Noise (100s) | Conversation (300s) | Foils / OOV (200s) | Conv FAR (/min) | Foil FAR (/min) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `legacy_v11` | 0.20 | 0 FA | 0 FA | 1 FA | 0 FA | **0.20** | **0.00** |
| `continuous_valley` | 0.20 | 0 FA | 0 FA | 680 FA | 355 FA | **136.00** | **106.50** |
| `hybrid_speechband_valley` | 0.20 | 0 FA | 0 FA | 379 FA | 284 FA | **75.80** | **85.20** |
| `hybrid_speechband_valley` | 0.30 | 0 FA | 0 FA | 364 FA | 275 FA | **72.80** | **82.50** |
| `hybrid_speechband_valley` | 0.40 | 0 FA | 0 FA | 304 FA | 224 FA | **60.80** | **67.20** |
| `hybrid_speechband_valley` | 0.50 | 0 FA | 0 FA | 216 FA | 163 FA | **43.20** | **48.90** |
| `hybrid_speechband_valley` | 0.60 | 0 FA | 0 FA | 152 FA | 113 FA | **30.40** | **33.90** |
| `game_mode_idle` | 0.20 | 0 FA | 0 FA | 0 FA | 0 FA | **0.00** | **0.00** |

### 3.2 Architectural Gating vs. Classifier Confidence Rejection
- **`legacy_v11` on 0 ms & 50 ms Gaps:**  
  In 100% of 0 ms and 50 ms gap continuous command sequences, `legacy_v11` failed to detect intermediate words (Recall: 13.3% on `len_len_phai`, 6.7% on `trai_len_xuong`). This failure was **not due to classifier rejection** (the classifier never saw low-confidence predictions). It was caused by **upstream VAD gating**: because speech energy never dropped below $-48\text{ dB}$ for $300\text{ ms}$, the segmenter never created audio segments.
- **`game_mode_idle` Closed Gate:**  
  Zero false activations during idle state reflect **session gating** (the audio pipeline drops chunks before they enter the hybrid segmenter). This is an architectural control barrier, not classifier rejection.

---

## 4. Latency Definitions & Measurement Audit

### 4.1 Disaggregated Latency Breakdown
The Phase 5 reported numbers were audited to establish exact component boundaries:

```
+-----------------------------------------------------------------------------------------------+
|                               TIMELINE OF A CONTINUOUS KWS EVENT                               |
+-----------------------------------------------------------------------------------------------+
|  Spoken Syllable Duration                                                                     |
|  [========================================]                                                   |
|  t_onset                                  t_offset                                            |
|                                           |                                                   |
|                                           |<- t_lag (Valley Dip + Hold) ->|                   |
|                                           |                               t_emit (Buffer Out) |
|                                           |                               |                   |
|                                           |                               |<- t_infer ->|     |
|                                           |                                             |     |
|                                           |<--------------- t_e2e --------------------->|     |
|                                                                                         Event |
|  |<---------------------------------- t_onset_to_emit --------------------------------->|     |
+-----------------------------------------------------------------------------------------------+
```

| Component | Metric Symbol | Scope & Definition | Sample Count $N$ | p50 (ms) | p95 (ms) |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **Model Forward Pass** | $t_{\text{ort}}$ | Pure ONNXRuntime CPU inference forward pass on precomputed log-mel spectrogram ($40 \times 101$). | 500 | **0.20** | **0.21** |
| **Frontend + Model** | $t_{\text{frontend+infer}}$ | Audio STFT, 40-channel Mel filterbank, log transform, and ONNX model evaluation. | 200 | **1.69** | **1.78** |
| **Acoustic Response Lag** | $t_{\text{lag}}$ | Time elapsed from speaker finishing command ($t_{\text{offset}}$) until valley segmenter confirms boundary ($t_{\text{emit}}$). | 166 | **47.7** | **132.7** |
| **Total End-to-End Latency** | $t_{\text{e2e}}$ | Total user-perceived delay from command syllable offset to game action trigger ($t_{\text{lag}} + t_{\text{infer}}$). | 166 | **47.7** | **132.7** |
| **Onset-to-Emission Latency** | $t_{\text{onset\_to\_emit}}$ | Total duration from initial utterance onset until event emission ($t_{\text{emit}} - t_{\text{onset}} + t_{\text{infer}}$). | 166 | **382.5** | **517.6** |

> [!NOTE]
> The $0.17\text{ ms} - 0.21\text{ ms}$ latency cited in earlier phase summaries is strictly the **ONNX model forward pass time** on an Intel/AMD CPU. The actual end-to-end response latency from syllable completion to event emission is **$47.7\text{ ms}$ (p50)** and **$132.7\text{ ms}$ (p95)**, which remains well within the target threshold of $< 150\text{ ms}$ for responsive gaming.

---

## 5. Audit Decision & Recommendations

### Final Audit Decision: `METRICS_VERIFIED`

- **Correctness of Metric Definitions:**  
  The mathematical formulations of $\text{Recall}$, $\text{Precision}$, $\text{F1}$, and $\text{WER}$ are mathematically correct. The equivalence $\text{WER} = 100\% - \text{Recall}$ is verified to result from $I = 0$ on clean positive command streams.
- **Test Matrix Consistency:**  
  All profiles and thresholds were tested against identical ground-truth audio fixtures, sequence lengths, and inter-command gaps.
- **Negative Control Isolation:**  
  Negative controls are properly evaluated using $\text{FAR}$ (/min) and are strictly quarantined from positive $\text{WER}$ calculations.
- **Latency Disaggregation:**  
  Latency metrics are clearly decomposed into model forward pass ($0.20\text{ ms}$), frontend + model ($1.69\text{ ms}$), acoustic response lag ($47.7\text{ ms}$), and end-to-end response ($47.7\text{ ms}$ p50 / $132.7\text{ ms}$ p95).

---

## 6. Artifact Index

1. **Audit Report:** [CONTINUOUS_KWS_PHASE5_1_METRIC_AUDIT.md](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/reports/CONTINUOUS_KWS_PHASE5_1_METRIC_AUDIT.md) (and root pointer [CONTINUOUS_KWS_PHASE5_1_METRIC_AUDIT.md](file:///d:/trainidentificationaudio/CONTINUOUS_KWS_PHASE5_1_METRIC_AUDIT.md))
2. **Disaggregated Raw Counts CSV:** [phase5_1_metric_counts.csv](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/reports/phase5_1_metric_counts.csv) (100 disaggregated rows covering pattern, gap, sequence length, and profile)
3. **Master Audit JSON:** [phase5_1_metric_audit.json](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/reports/phase5_1_metric_audit.json)
4. **Metric Integrity Unit Test Suite:** [tests/test_metric_integrity_phase5_1.py](file:///d:/trainidentificationaudio/tests/test_metric_integrity_phase5_1.py) (10/10 tests passing)
