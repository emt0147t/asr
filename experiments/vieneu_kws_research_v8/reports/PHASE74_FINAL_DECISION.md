# Phase 74 — Final Decision and Research Campaign Certification

**Workspace:** `D:\trainidentificationaudio`  
**Experiment Directory:** `experiments/vieneu_kws_research_v8/`  
**Status Banner:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Formal Dual Operational Verdicts

### Mode A: Bounded-Utterance Keyword Spotting (Push-to-Talk / Discrete Capture)
**Official Operational Decision:** `FEASIBLE_BOUNDED_CANDIDATE_ESTABLISHED`  
- **Best Candidate Architecture:** `variant_w10`  
- **Command Recall (Cross-Seed Mean):** 99.79%  
- **Non-Target False Acceptance Rate:** 2.03%  
- **CPU Decision Latency (Batch 1):** 1.73 ms  

### Mode B: Continuous Open-Mic Keyword Spotting (Always-Listening)
**Official Operational Decision:** `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`  
- **Preserved Historical Finding:** Continuous stream evaluation exhibits excessive non-target activations (> 60 false alarms/hour) on conversational Vietnamese audio. No deployable open-mic configuration exists under current specifications.
- **Ethical Boundary:** Success or improvement in bounded-utterance evaluation must **never** be used to claim continuous open-mic readiness.

---

## 2. Research Campaign Certification Summary

1. **Metric and Threshold Audit (Phase 70):** Successfully audited Candidate B across seeds 42, 43, 44, confirming bit-exact reproduction (0 mismatches), but documenting that Seed 42 fell below 95% recall on PHAI (92.5%) and all seeds exceeded 5.0% FAR.
2. **Validation Pareto Study (Phase 71):** Evaluated 50 threshold configurations on validation data, establishing the empirical Pareto frontier and recording `NO_FEASIBLE_VALIDATION_OPERATING_POINT` under strict cross-seed per-keyword criteria.
3. **Controlled Weight Ablation (Phase 72):** Trained 12 ablation models across $w_{\text{hn}} \in [1.0, 1.2, 1.4, 1.6]$, demonstrating that mild to moderate hard-negative weighting ($w_{\text{hn}}=1.0$ and $w_{\text{hn}}=1.2$) achieves optimal balance between command recall and false alarm suppression.
4. **Robust Calibration (Phase 73):** Selected top finalists under a written validation rule, freezing operating parameters before test evaluation.
5. **Fresh Locked Test (Phase 74):** Built a fresh 720-clip synthetic benchmark with 80 brand-new verified disjoint Vietnamese texts and executed independent test evaluation.
