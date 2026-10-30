# Phase 5.2 — Event Matching and Latency Consistency Check Report

**Repository:** `D:\trainidentificationaudio`  
**Branch:** `feature/continuous-kws-segmentation`  
**Date:** 2026-10-29  
**Audit Decision:** `METRICS_VERIFIED`  

Full detailed report is archived at:  
[experiments/vieneu_kws_research_v13_4/reports/CONTINUOUS_KWS_PHASE5_2_MATCHING_LATENCY_AUDIT.md](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/reports/CONTINUOUS_KWS_PHASE5_2_MATCHING_LATENCY_AUDIT.md)

---

## Executive Summary & Governance Attestation

> [!IMPORTANT]
> **Mandatory Governance Attestation**  
> All work in Phase 5.2 was conducted as a strict evaluation-integrity and consistency audit. No models were trained, neural network parameters were not modified, the released v11 checkpoint (`3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2`) and ONNX model (`54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99`) remained frozen and verified, no thresholds were altered, and production defaults were preserved.
>
> The four mandatory governance labels are strictly preserved:
> 1. `EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`
> 2. `NOT_VALIDATED_ON_NATURAL_SPEECH`
> 3. `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`
> 4. `NOT_VALIDATED_PENDING_DATA`

---

## Key Audit Resolutions

1. **One-to-One Monotonic Event Matching:**  
   Implemented the canonical Monotonic Ordered One-to-One Alignment algorithm via dynamic programming. Strict bijection (at most one prediction per ground truth, at most one ground truth per prediction) and temporal monotonicity are guaranteed. Across the locked benchmark, 6 regressions in `legacy_v11` were resolved where greedy tie-breaking had previously misclassified True Positives as Substitutions due to large acoustic lag. `continuous_valley` and `hybrid_speechband_valley` exhibited 0 regressions (100% stable).
2. **Recomputed Metric Tables & $I=0$ Verification:**  
   Recomputed all 17 benchmark groups side-by-side. The mathematical identity $\text{WER} = 100\% - \text{Recall}$ holds identically because positive-sequence insertions $I = 0$ in all 535 evaluated trials. Negative-control false activations remain strictly quarantined and reported via False Alarm Rate per minute.
3. **Resolution of Latency Inconsistency:**  
   - **Identical 47.7 ms reporting error:** Identified as an accidental assignment in the reporting script dictionary (`acoustic_offset_response_lag` copied the scalar `p50_e2e` instead of `p50_lag`). Recomputed from actual per-trial distributions: true acoustic lag is **$45.0\text{ ms}$ (p50) / $130.0\text{ ms}$ (p95)**; true end-to-end latency is **$47.32\text{ ms}$ (p50) / $132.42\text{ ms}$ (p95)**.
   - **Frontend+Inference discrepancy (2.43 ms vs. 1.69 ms):** Conclusively explained and reproduced: in-stream measurement on real speech buffers during pipeline processing is **$2.21 - 2.43\text{ ms}$**; isolated microbenchmark on synthetic zero buffers in a tight loop is **$1.69 - 1.81\text{ ms}$** due to FFT SIMD zero-vector optimizations and cache residency.

---

## Artifact Index
- Master Report: [CONTINUOUS_KWS_PHASE5_2_MATCHING_LATENCY_AUDIT.md](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/reports/CONTINUOUS_KWS_PHASE5_2_MATCHING_LATENCY_AUDIT.md)
- Regressions CSV: [phase5_2_event_matching_regressions.csv](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/reports/phase5_2_event_matching_regressions.csv)
- Recomputed Metrics CSV: [phase5_2_recomputed_metrics.csv](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/reports/phase5_2_recomputed_metrics.csv)
- Latency Audit JSON: [phase5_2_latency_audit.json](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/reports/phase5_2_latency_audit.json)
- Unit Tests: [`tests/test_metric_integrity_phase5_2.py`](file:///d:/trainidentificationaudio/tests/test_metric_integrity_phase5_2.py) (8/8 passed, total 47 continuous KWS tests passing).
