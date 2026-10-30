# CONTINUOUS KWS PHASE 5 AUDIT REPORT
## Correctness Audit, Metric Alignment, and Locked-Matrix Benchmark Validation

**Full Report Location:** [`experiments/vieneu_kws_research_v13_4/reports/CONTINUOUS_KWS_PHASE5_AUDIT_REPORT.md`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/reports/CONTINUOUS_KWS_PHASE5_AUDIT_REPORT.md)  
**Repository:** `D:\trainidentificationaudio`  
**Git Branch:** `feature/continuous-kws-segmentation`  
**Phase:** Phase 5 — Continuous KWS Correctness Audit & Benchmark Validation  
**Date:** 2026-10-29  

---

### Mandatory Governance Labels
> [!WARNING]
> - `EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`
> - `NOT_VALIDATED_ON_NATURAL_SPEECH`
> - `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`
> - `NOT_VALIDATED_PENDING_DATA`

---

## Executive Summary & Decisions
- **Audited Correctness Defects:** 6 defects identified, remediated, and verified in [`phase5_correctness_regressions.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/reports/phase5_correctness_regressions.csv).
  1. *BUG-01: Preroll Duplication* (Onset frame duplicated between preroll_buf and chunk) -> Resolved.
  2. *BUG-02: Missing Duration Validation on Valley Split* -> Enforced `min_speech_samples` (100ms) on seg1.
  3. *BUG-03: Active-Speech Accounting Contamination* -> Introduced `chunk_is_speech` boolean ledger.
  4. *BUG-04: Valley Split Boundary Partitioning* -> Strict boundary range check and parallel state slicing.
  5. *BUG-05: Stream Finalization Flush* -> Implemented `flush()` on segmenters and `LiveKWSPipeline`.
  6. *BUG-06: Game Session Reset & Isolation* -> Complete buffer reset on open/close to eliminate cross-session leaks.
- **Automated Regression Suite:** 29 tests passing across Phase 2, Phase 4, and Phase 5 (`tests/test_continuous_kws_phase5.py`).
- **Locked Matrix Benchmark Results:**
  - `legacy_v11`: Recall 28.8%, Exact Match 7.7%, E2E p50 282.4ms, ConvFAR 0.2/min.
  - `continuous_valley`: Recall 97.6%, Exact Match 93.8%, E2E p50 122.4ms, ConvFAR 136.0/min.
  - `hybrid_speechband_valley`: Recall 96.5%, Precision 98.8%, Exact Match 90.8%, E2E p50 47.7ms, ConvFAR 75.8/min.
  - `game_mode_idle` (Gate Closed): Conv FAR = **0.00/min** (0 triggers across 300s conversation, 100s noise, 300s silence).
- **Latency Disaggregation:**
  - Cold-start model inference: 0.66 ms | Warmed forward pass: p50 = 0.17 ms, p95 = 0.21 ms.
  - Frontend Log-Mel + Inference: p50 = 2.43 ms.
  - Post-offset response latency ($t_{\text{e2e}}$): p50 = 47.7 ms, p95 = 132.7 ms.
  - Onset-to-emission latency ($t_{\text{onset\_to\_emit}}$): p50 = 382.5 ms, p95 = 517.6 ms.
- **20-Command Sequence Ruling:** Exact match for $N=20$ is **0.0% (0/10)**. Support for 20-command sequences is **NOT supported**.
- **Final Decision:** **`CONTINUE_EXPERIMENTAL_GAME_MODE`**. Always-open mic mode remains strictly non-deployable on natural speech.
