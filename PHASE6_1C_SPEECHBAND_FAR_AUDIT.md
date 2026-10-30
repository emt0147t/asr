# Phase 6.1C: Matched Speech-Band A/B & Negative Controls Audit

**Full Report Location:** [`experiments/vieneu_kws_research_v13_4/reports/PHASE6_1C_SPEECHBAND_FAR_AUDIT.md`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/reports/PHASE6_1C_SPEECHBAND_FAR_AUDIT.md)  
**Status:** Completed  
**Workstream:** Phase 6.1C  
**Date:** 2026-10-30

---

## Mandatory Governance Labels

> - **`EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`**
> - **`NOT_VALIDATED_ON_NATURAL_SPEECH`**
> - **`NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`**
> - **`NOT_VALIDATED_PENDING_DATA`**

---

## Key Results Summary

1. **Matched A/B Evaluation (300 Hz vs 100 Hz Low Cutoff):**
   - **Phase 6 Locked Manifest:**
     - 0 ms gap: 100 Hz achieves **96.7% (29/30)** exact match vs 300 Hz at **63.3% (19/30)** (+33.4% gain).
     - 50 ms gap: 100 Hz achieves **90.0% (27/30)** exact match vs 300 Hz at **66.7% (20/30)** (+23.3% gain).
   - **Phase 6.1A Holdout Manifest:**
     - 0 ms gap: 100 Hz achieves **100.0% (15/15)** exact match vs 300 Hz at **60.0% (9/15)** (+40.0% gain).
     - 50 ms gap: 100 Hz achieves **93.3% (14/15)** exact match vs 300 Hz at **53.3% (8/15)** (+40.0% gain).
2. **Negative Controls Audit (900.0s total audio):**
   - Open-Gate Continuous Valley: produces **669 FA on conversation** (133.8 FA/min) and **361 FA on foils** (108.3 FA/min).
   - Closed-Gate Game Mode Controller: produces **strictly 0 FA across all 900.0s** of negative audio.
   - Closed-gate zero activations validate architectural session gating only, and must not be presented as proof of classifier accuracy.
