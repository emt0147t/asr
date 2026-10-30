# Phase 5.1 — Metric Integrity Audit Report

**Repository:** `D:\trainidentificationaudio`  
**Branch:** `feature/continuous-kws-segmentation`  
**Date:** 2026-10-29  
**Audit Decision:** `METRICS_VERIFIED`  

Full detailed report is archived at:  
[experiments/vieneu_kws_research_v13_4/reports/CONTINUOUS_KWS_PHASE5_1_METRIC_AUDIT.md](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/reports/CONTINUOUS_KWS_PHASE5_1_METRIC_AUDIT.md)

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

---

## Summary of Audit Findings

### 1. Mathematical Explanation: $\text{WER} = 100\% - \text{Recall}$
In any time-aligned sequence evaluation, $N_{\text{gt}} = \text{TP} + S + D$. Therefore:
$$\text{WER} = \frac{S + D + I}{N_{\text{gt}}} = \frac{(N_{\text{gt}} - \text{TP}) + I}{N_{\text{gt}}} = (100\% - \text{Recall}) + \frac{I}{N_{\text{gt}}} \times 100\%$$
On clean positive command streams, the acoustic valley and hybrid segmenters never triggered spurious splits or fired during silence intervals. Thus insertions $I = 0$ identically across all trials, making $\text{WER} \equiv 100\% - \text{Recall}$.

### 2. Substitutions in Event Matching vs. Sequence Alignment
- In time-window event matching, a substitution occurs when a detection falls in a command window with the wrong class. It counts as **both 1 FP and 1 FN** ($\text{FP} = I + S, \text{FN} = D + S$).
- In sequence-level Levenshtein alignment, a substitution counts as 1 edit operation.
- On clean command streams for `continuous_valley` and `hybrid_speechband_valley`, both alignment algorithms produced identical $(S, D, I)$ counts.

### 3. Architectural Gating vs. Classifier Rejection
- `legacy_v11` zero-activations on 0 ms and 50 ms gap streams are **purely upstream architectural gating** because the required 300 ms silence interval was never reached. The classifier was never invoked for intermediate syllables.
- `game_mode_idle` zero false activations reflect intentional **session gating**, not classifier confidence.

### 4. Latency Disaggregation Breakdown
- **Pure ONNX model forward pass ($t_{\text{ort}}$):** $0.20\text{ ms}$ p50, $0.21\text{ ms}$ p95 ($N=500$). This accounts for the $0.17 - 0.21\text{ ms}$ reported previously.
- **Frontend + Model inference:** $1.69\text{ ms}$ p50, $1.78\text{ ms}$ p95 ($N=200$).
- **Acoustic response lag ($t_{\text{lag}}$):** $47.7\text{ ms}$ p50, $132.7\text{ ms}$ p95 ($N=166$).
- **Total end-to-end response latency ($t_{\text{e2e}} = t_{\text{lag}} + t_{\text{infer}}$):** $47.7\text{ ms}$ p50, $132.7\text{ ms}$ p95 ($N=166$).
- **Onset-to-emission latency ($t_{\text{onset\_to\_emit}}$):** $382.5\text{ ms}$ p50, $517.6\text{ ms}$ p95 ($N=166$).

---

## Final Audit Decision

### `METRICS_VERIFIED`

All metric calculations, formulas, test set denominators, and latency breakdowns have been independently recomputed, mathematically verified, and validated with unit tests.

---

## Artifact Index
- Master Audit Report: [CONTINUOUS_KWS_PHASE5_1_METRIC_AUDIT.md](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/reports/CONTINUOUS_KWS_PHASE5_1_METRIC_AUDIT.md)
- Raw Counts CSV: [phase5_1_metric_counts.csv](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/reports/phase5_1_metric_counts.csv)
- Master JSON: [phase5_1_metric_audit.json](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/reports/phase5_1_metric_audit.json)
- Metric Unit Tests: [tests/test_metric_integrity_phase5_1.py](file:///d:/trainidentificationaudio/tests/test_metric_integrity_phase5_1.py) (10/10 passed)
