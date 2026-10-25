# Phase 70 — Metric Reproduction and Verification Report

**Workspace:** `D:\trainidentificationaudio`  
**Experiment Directory:** `experiments/vieneu_kws_research_v8/`  
**Status Banner:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Reproduction Summary & Hash Integrity

Phase 70 successfully reproduced every reported metric from Phase 69 with zero discrepancies across all 2,160 test inferences:

- **Test Manifest:** `experiments/vieneu_kws_research_v6/data/splits/test_manifest.csv` ($N = 720$ clips).
- **Prediction Mismatches:** Exactly **0** mismatches against `per_clip_predictions.csv`.
- **Validation-Test Leakage Audit:** Confirmed that thresholds $\theta_{\text{cmd}} \in \{0.50, 0.35, 0.35\}$ were determined exclusively from `val_manifest.csv` grid search records without inspecting `test_manifest.csv`.

---

## 2. Detailed Confusion Matrices (Locked Test Split)

### Seed 42 Confusion Matrix (Threshold $\theta = 0.5$):

| True \ Pred | TRAI | PHAI | LEN | XUONG | REJECT | Total | Recall / Spec |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TRAI** | 40 | 0 | 0 | 0 | 0 | 40 | 100.0% |
| **PHAI** | 3 | 37 | 0 | 0 | 0 | 40 | 92.5% |
| **LEN** | 0 | 0 | 40 | 0 | 0 | 40 | 100.0% |
| **XUONG** | 0 | 0 | 0 | 40 | 0 | 40 | 100.0% |
| **REJECT** | 7 | 2 | 1 | 19 | 531 | 560 | 94.8% |

### Seed 43 Confusion Matrix (Threshold $\theta = 0.35$):

| True \ Pred | TRAI | PHAI | LEN | XUONG | REJECT | Total | Recall / Spec |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TRAI** | 40 | 0 | 0 | 0 | 0 | 40 | 100.0% |
| **PHAI** | 1 | 39 | 0 | 0 | 0 | 40 | 97.5% |
| **LEN** | 0 | 0 | 40 | 0 | 0 | 40 | 100.0% |
| **XUONG** | 0 | 0 | 0 | 40 | 0 | 40 | 100.0% |
| **REJECT** | 12 | 0 | 8 | 19 | 521 | 560 | 93.0% |

### Seed 44 Confusion Matrix (Threshold $\theta = 0.35$):

| True \ Pred | TRAI | PHAI | LEN | XUONG | REJECT | Total | Recall / Spec |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TRAI** | 40 | 0 | 0 | 0 | 0 | 40 | 100.0% |
| **PHAI** | 0 | 40 | 0 | 0 | 0 | 40 | 100.0% |
| **LEN** | 0 | 0 | 40 | 0 | 0 | 40 | 100.0% |
| **XUONG** | 0 | 0 | 0 | 40 | 0 | 40 | 100.0% |
| **REJECT** | 5 | 0 | 13 | 13 | 529 | 560 | 94.5% |

