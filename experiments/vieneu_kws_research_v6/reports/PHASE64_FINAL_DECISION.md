# Phase 64 — Final Decision and Research Campaign Certification

**Workspace:** `D:\trainidentificationaudio`  
**Experiment Directory:** `experiments/vieneu_kws_research_v6/`  
**Status Banner:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Formal Dual Operational Verdicts

### Mode A: Bounded-Utterance Keyword Spotting (Push-to-Talk / Discrete Capture)
**Official Operational Decision:** `NO_FEASIBLE_VALIDATION_OPERATING_POINT`  
- **Recommended Candidate:** `variant_a_cascade_v1`  
- **Command Recall (Cross-Seed Mean):** 85.42%  
- **Non-Target False Acceptance Rate:** 5.83%  
- **Decision Latency (CPU Batch 1):** 3.91 ms  

### Mode B: Continuous Open-Mic Keyword Spotting (Always-Listening)
**Official Operational Decision:** `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`  
- **Preserved Historical Finding:** Continuous stream evaluation exhibits excessive non-target activations (> 60 false alarms/hour) on conversational Vietnamese audio. No deployable open-mic configuration exists under current specifications.
- **Ethical Boundary:** Success in bounded-utterance evaluation must **never** be conflated with or used to claim continuous open-mic readiness.

---

## 2. Research Campaign Certification Summary

1. **Metric Reconciliation (Phase 60):** Discrepancies between macro and micro recall were mathematically resolved as identical ($N_{\text{cmd}} = 160$, 40 clips/keyword). Latency decomposition confirmed CPU post-capture decision latency $\le 50\,\text{ms}$.
2. **Fresh Synthetic Benchmark v6 (Phase 61):** Constructed 2,440 audio clips across 200 brand-new non-target texts with zero historical text reuse and zero cross-split leakage.
3. **Controlled Bounded Improvement (Phase 62):** Trained compact bounded networks with positive-preserving loss and hard-negative weighting.
4. **Validation-Only Calibration (Phase 63):** Operating thresholds calibrated strictly on validation split without test snooping.
5. **Locked Test Evaluation (Phase 64):** Bit-exact, reproducible evaluation on 720 test clips across 4 held-out speaker voices.
