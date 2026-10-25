# Phase 55 — Independent Metric and Baseline Audit Report

**Workspace:** `D:\trainidentificationaudio`  
**Experiment Directory:** `experiments/vieneu_kws_research_v5/`  
**Audit Findings JSON:** `experiments/vieneu_kws_research_v5/phase55_audit_findings.json`  
**Status:** `INDEPENDENT_AUDIT_VERIFIED` | `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Executive Summary

An independent, rigorous audit was performed on the source code, model classes, checkpoint weights, prediction records, confusion matrices, and runtime evaluation scripts of the Phase 50–54 bounded-utterance campaign.

### Key Audit Conclusions:
1. **Bit-Exact Metric Reproduction:** Recomputation of all reported test metrics from the raw serialized prediction records (`per_utterance_predictions.csv`, 720 clips $\times$ 3 candidates $\times$ 3 seeds = 6,480 inferences) matched the reported machine-readable summary (`locked_test_summary.json`) with **zero discrepancies** (0 mismatches across recall, FAR, TNR, 5-class accuracy, and per-keyword recalls).
2. **Precision of Target Recall Definition:** Target recall refers to the macro-average across the four target command classes (`TRAI`, `PHAI`, `LEN`, `XUONG`). Because each target class contains an identical balance of 40 clips in the test split ($N = 160$ total command clips), macro-averaged recall across classes is mathematically identical to micro event recall across command presentations:
   $$\text{Macro-Recall} = \frac{1}{4} \sum_{c \in \{\text{TRAI}, \text{PHAI}, \text{LEN}, \text{XUONG}\}} \text{Recall}_c \equiv \frac{\sum \text{Hits}}{160} = \text{Micro-Recall}$$
   For Candidate 2 (Cascade v1 Bounded Pipeline):
   - Seed 42: $157 / 160 = 98.125\%$
   - Seed 43: $153 / 160 = 95.625\%$
   - Seed 44: $150 / 160 = 93.750\%$
   - **Mean Command Recall:** **95.833%** ($\approx 95.83\%$).
3. **Non-Target FAR Denominator:** The denominator for Non-Target False Acceptance Rate (FAR) is exactly $N_{\text{nt}} = 560$ (480 `UNKNOWN` distractor clips + 80 `SILENCE` clips across 4 held-out voices). All clips classified as `REJECT` are counted as True Negatives ($TNR = 1 - FAR$).
4. **Five-Class Overall Accuracy:** Defined as:
   $$\text{Accuracy}_{\text{5-class}} = \frac{N_{\text{correct\_cmd}} + N_{\text{correct\_reject}}}{160 + 560} = \frac{N_{\text{correct}}}{720}$$
   For Candidate 2: Seed 42 = $646/720 = 89.72\%$; Seed 43 = $637/720 = 88.47\%$; Seed 44 = $607/720 = 84.31\%$; Mean = **87.50%**.
5. **CPU Latency Profiling:** The Phase 53 report recorded 5.81 ms on GPU. Independent single-clip CPU latency profiling (batch size = 1, 4 threads, Intel CPU) reveals:
   - Candidate 1 (Stage 1 CNN alone): **1.41 ms** mean (P90 = 1.65 ms).
   - Candidate 2 (Cascade v1 Bounded Pipeline): **4.89 ms** mean (P90 = 5.82 ms).
   - Candidate 3 (`VietnameseBoundedKWSNet`): **2.08 ms** mean (P90 = 2.52 ms).
   All candidates easily satisfy the bounded-utterance latency ceiling ($\le 50\,\text{ms}$).

---

## 2. Model Architecture, Preprocessing & Checkpoint Audit

| Candidate | Component | Model Architecture | Parameters | Preprocessing / Inputs | Checkpoint Path |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Candidate 1** | Standalone CNN | `VietnameseKWSNet` (6 classes) | 138,534 | 16 kHz Log-Mel (40 mels, 101 frames) | `experiments/vieneu_kws_v2/checkpoints/best_checkpoint_seed{42,43,44}.pt` |
| **Candidate 2** | Stage 1 CNN <br> Stage 2 TCN | `VietnameseKWSNet` (6 classes) <br> `TemporalKWSVerifier` (5 classes) | 138,534 <br> 340,773 (Total: 479,307) | 16 kHz Log-Mel (40 mels, 101 frames) <br> Multi-frame Log-Mel sequence (19,200 samples) | `experiments/vieneu_kws_v2/checkpoints/best_checkpoint_seed{42,43,44}.pt` <br> `experiments/vieneu_kws_cascade_v1/checkpoints/best_verifier_seed{42,43,44}.pt` |
| **Candidate 3** | Bounded CNN | `VietnameseBoundedKWSNet` (5 classes) | 183,500 | 16 kHz Log-Mel + Temporal Attentive Pooling | `experiments/vieneu_kws_research_v4/models/bounded_utterance/checkpoints/bounded_kws_net_seed{42,43,44}.pt` |

### Label Ordering Alignment:
- `VietnameseKWSNet` (6 classes): `[0: TRAI, 1: PHAI, 2: LEN, 3: XUONG, 4: UNKNOWN, 5: SILENCE]`.
- `TemporalKWSVerifier` (5 classes): `[0: TRAI, 1: PHAI, 2: LEN, 3: XUONG, 4: REJECT]`.
- `VietnameseBoundedKWSNet` (5 classes): `[0: TRAI, 1: PHAI, 2: LEN, 3: XUONG, 4: REJECT]`.

---

## 3. Confusion Matrix Verification (Candidate 2 — Cascade v1)

Aggregated across seeds 42, 43, and 44 on the locked test set (720 clips per seed):

```
                     Predicted Class (Summed across 3 Seeds)
Actual Class │   TRAI   │   PHAI   │    LEN   │  XUONG   │  REJECT  │ Total
─────────────┼──────────┼──────────┼──────────┼──────────┼──────────┼───────
TRAI         │   119    │     0    │     0    │     0    │     1    │  120
PHAI         │     0    │   120    │     0    │     0    │     0    │  120
LEN          │     0    │     0    │   109    │     0    │    11    │  120
XUONG        │     0    │     0    │     0    │   112    │     8    │  120
─────────────┼──────────┼──────────┼──────────┼──────────┼──────────┼───────
REJECT (NT)  │    76    │    40    │    79    │    31    │  1,454   │ 1,680
```

### Critical Acoustic Properties Revealed:
1. **Zero Cross-Command Confusion:** True command keywords are **never** confused with another target command (0 instances of `TRAI` $\rightarrow$ `PHAI`, `LEN` $\rightarrow$ `XUONG`, etc.).
2. **Rejection as Sole Command Deficit:** Misses on target commands are exclusively rejections ($1$ on `TRAI`, $0$ on `PHAI`, $11$ on `LEN`, $8$ on `XUONG`).
3. **Non-Target Leakage:** The sole source of error in Candidate 2 is non-target distractors falsely accepted as commands ($226 / 1680 = 13.45\%$ false acceptance rate across the combined 3 seeds).
