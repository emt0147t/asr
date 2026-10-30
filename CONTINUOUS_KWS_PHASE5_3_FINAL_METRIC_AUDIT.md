# Phase 5.3 — Final Metric Semantics Audit Report

**Repository:** `D:\trainidentificationaudio`  
**Branch:** `feature/continuous-kws-segmentation`  
**Date:** 2026-10-29  
**Audit Decision:** `METRICS_VERIFIED`  

Full detailed report is archived at:  
[experiments/vieneu_kws_research_v13_4/reports/CONTINUOUS_KWS_PHASE5_3_FINAL_METRIC_AUDIT.md](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/reports/CONTINUOUS_KWS_PHASE5_3_FINAL_METRIC_AUDIT.md)

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

---

## Key Audit Resolutions

1. **Separation of Sequence WER and Temporal Event Metrics:**  
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

## Artifact Index
- Master Report: [CONTINUOUS_KWS_PHASE5_3_FINAL_METRIC_AUDIT.md](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/reports/CONTINUOUS_KWS_PHASE5_3_FINAL_METRIC_AUDIT.md)
- Sequence vs Event CSV: [phase5_3_sequence_vs_event_metrics.csv](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/reports/phase5_3_sequence_vs_event_metrics.csv)
- Master Audit JSON: [phase5_3_metric_audit.json](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/reports/phase5_3_metric_audit.json)
- Unit Tests: [`tests/test_metric_integrity_phase5_3.py`](file:///d:/trainidentificationaudio/tests/test_metric_integrity_phase5_3.py) (6/6 passed, 53 total continuous KWS tests passing).
