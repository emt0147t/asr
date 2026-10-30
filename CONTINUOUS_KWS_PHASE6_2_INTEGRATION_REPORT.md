# Phase 6.2: Integration and Final Acceptance Gate Report

**Full Report Location:** [`experiments/vieneu_kws_research_v13_4/reports/CONTINUOUS_KWS_PHASE6_2_INTEGRATION_REPORT.md`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/reports/CONTINUOUS_KWS_PHASE6_2_INTEGRATION_REPORT.md)  
**Status:** Completed  
**Final Operational Decision:** `ACCEPT_CONTROLLED_DEMO`  
**Date:** 2026-10-30

---

## Mandatory Governance Labels

> - **`EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`**
> - **`NOT_VALIDATED_ON_NATURAL_SPEECH`**
> - **`NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`**
> - **`NOT_VALIDATED_PENDING_DATA`**

---

## Summary of Final Integrated Results

### Operational Verdict: `ACCEPT_CONTROLLED_DEMO`

The parallel execution of Phase 6.1A, 6.1B, 6.1C, and 6.1D culminated in final integration and evaluation against the independent Phase 6.1A Holdout Manifest:
- **Primary Target $N=3$ (`LEN -> LEN -> PHAI`) on Holdout Manifest:**
  - **0 ms gap:** **15/15 (100.0%) Exact Match**, Sequence WER: **0.0%** (Target $\ge 90\%$).
  - **50 ms gap:** **14/15 (93.3%) Exact Match**, Sequence WER: **2.2%** (Target $\ge 90\%$).
  - **Duplicate Events Emitted:** **0** across all trials.
- **Over-Splitting on Elongated Commands ($L=750$ ms):** **0** over-splits across all trials (**100.0% exact match**).
- **Session Gating Negative Control:** **0 False Activations** across all 900.0s of negative audio under `GameModeSessionController` closed-gate suppression.
- **Physical Live-Microphone Validation:** Documented that physical over-the-air loudspeaker transmission fails (0% exact match) due to room acoustics / transducer distortion on the frozen CNN, establishing that ambient live-mic requires acoustic model retraining in a future phase.
- **Regression Test Suite:** All 68 continuous KWS and metric integrity tests pass (100% passing).
