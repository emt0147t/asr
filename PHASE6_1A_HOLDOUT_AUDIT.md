# Phase 6.1A: Dataset Independence & Holdout Manifest Audit

**Full Report Location:** [`experiments/vieneu_kws_research_v13_4/reports/PHASE6_1A_HOLDOUT_AUDIT.md`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/reports/PHASE6_1A_HOLDOUT_AUDIT.md)  
**Status:** Completed  
**Workstream:** Phase 6.1A  
**Date:** 2026-10-30

---

## Mandatory Governance Labels

> - **`EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`**
> - **`NOT_VALIDATED_ON_NATURAL_SPEECH`**
> - **`NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`**
> - **`NOT_VALIDATED_PENDING_DATA`**

---

## Summary Findings

1. **Clip Inventory Ceiling:** Exactly 50 target command recordings exist per class (`len`, `phai`, `trai`, `xuong`), across 8 Northern accent speakers (3 female, 5 male).
2. **Phase 6 Locked Manifest Overlap:** Evaluating 30 trials of `LEN -> LEN -> PHAI` requires 60 `LEN` instances, resulting in mandatory clip recycling in trials 26–30. All 30 trials at 50 ms gap reused the identical clip combinations as the 0 ms trials.
3. **Tuning Contamination Disclosure:** All 50 `LEN` clips were observed when tuning the 100 Hz cutoff; the reported 96.7% and 90.0% metrics reflect confirmatory pool fitting.
4. **New Holdout Manifest:** Created [`phase6_1a_holdout_manifest.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/reports/phase6_1a_holdout_manifest.csv) (55 trials) isolating speakers `spk_north_03`, `spk_north_06`, and `spk_north_08` with zero overlap against diagnostic files.
