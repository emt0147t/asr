# Phase 69 — Final Decision and Research Campaign Certification

**Workspace:** `D:\trainidentificationaudio`  
**Experiment Directory:** `experiments/vieneu_kws_research_v7/`  
**Status Banner:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Formal Dual Operational Verdicts

### Mode A: Bounded-Utterance Keyword Spotting (Push-to-Talk / Discrete Capture)
**Official Operational Decision:** `NO_FEASIBLE_VALIDATION_OPERATING_POINT`  
- **Best Candidate Architecture:** `candidate_b_moderately_weighted_hard_negative`  
- **Command Recall (Cross-Seed Mean):** 99.17%  
- **Non-Target False Acceptance Rate:** 5.89%  
- **CPU Decision Latency (Batch 1):** 1.65 ms  

### Mode B: Continuous Open-Mic Keyword Spotting (Always-Listening)
**Official Operational Decision:** `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`  
- **Preserved Historical Finding:** Continuous stream evaluation exhibits excessive non-target activations (> 60 false alarms/hour) on conversational Vietnamese audio. No deployable open-mic configuration exists under current specifications.
- **Ethical Boundary:** Success or improvement in bounded-utterance evaluation must **never** be used to claim continuous open-mic readiness.

---

## 2. Research Campaign Certification Summary

1. **Reproducibility Audit (Phase 65):** Successfully reconstructed all baseline denominators, confidence intervals, and CPU latency decompositions with zero discrepancies.
2. **Error Diagnosis (Phase 66):** Identified that the 50% REJECT class imbalance and 3.3x gradient mass on REJECT caused the severe recall collapse in retrained models.
3. **Controlled Ablation (Phase 67):** Demonstrated that inverse-class weighting ($w_{\text{cmd}} = 2.0, w_{\text{REJECT}} = 0.5$) and non-parametric global pooling completely eliminated zero-recall collapse.
4. **Validation Calibration (Phase 68):** Thresholds calibrated strictly on validation split without test snooping.
5. **Locked Test Evaluation (Phase 69):** Bit-exact, reproducible evaluation on 720 test clips across 4 held-out speaker voices.
