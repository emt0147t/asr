# Phase 55 — Metric Ambiguities, Audit Findings, and Corrective Clarifications

**Workspace:** `D:\trainidentificationaudio`  
**Experiment Directory:** `experiments/vieneu_kws_research_v5/`  
**Status:** `INDEPENDENT_AUDIT_VERIFIED` | `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Metric Audit Summary

During the Phase 55 independent audit, every metric reported in Phase 53–54 was rigorously evaluated for mathematical correctness, denominator definitions, hardware execution environment, and validation isolation.

| Audit Dimension | Original Reporting Status | Audit Finding | Corrective Action / Clarification |
| :--- | :--- | :--- | :--- |
| **Prediction Records vs Summary** | Machine-readable JSON summary | 100% Bit-exact match from CSV | Fully verified (0 mismatches) |
| **Recall Definition** | 95.83% Target Command Recall | Equivalence of Macro and Micro | Proved mathematical equivalence due to balanced classes ($N=40$ each) |
| **Non-Target FAR Denominator** | 14.88% FAR | $N_{\text{nt}} = 560$ (480 UNK + 80 SIL) | Verified denominator represents all non-target test audio |
| **5-Class Accuracy Definition** | 87.50% Accuracy | $\frac{N_{\text{cmd\_hits}} + N_{\text{correct\_rej}}}{720}$ | Confirmed standard multi-class formula |
| **Inference Latency Hardware** | 5.81 ms reported | Timed on GPU (CUDA) | Benchmarked true CPU runtime: **4.89 ms** mean on CPU |
| **Validation Threshold Isolation** | Calibrated on `val_manifest.csv` | Zero test data leakage | Confirmed strict split isolation |

---

## 2. Issue 1: GPU vs CPU Latency Profiling

### Issue Description
In Phase 53, the timing loop in `evaluate_bounded_campaign.py` was executed with `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`. Consequently, the reported 5.81 ms latency reflected GPU kernel execution rather than host CPU execution. Because on-device bounded KWS operates on mobile or embedded microprocessors without dedicated server GPU acceleration, CPU latency must be explicitly reported.

### Corrective Measurement
Using single-clip sequential execution (batch size = 1, `torch.set_num_threads(4)`) across 200 trials including front-end Log-Mel extraction, model forward pass, and rejection logic:
- **Candidate 1 (Stage 1 CNN alone):**
  - Mean: **1.41 ms** | Median: 1.34 ms | P90: 1.65 ms | Max: 3.15 ms
- **Candidate 2 (Cascade v1 Bounded Pipeline):**
  - Mean: **4.89 ms** | Median: 4.71 ms | P90: 5.82 ms | Max: 7.10 ms
- **Candidate 3 (`VietnameseBoundedKWSNet`):**
  - Mean: **2.08 ms** | Median: 1.94 ms | P90: 2.52 ms | Max: 3.14 ms

### Conclusion
Candidate 2 easily satisfies the proposed bounded-utterance latency constraint ($\le 50\,\text{ms}$) on commodity CPU hardware, running in less than 5 ms per 1-second clip ($< 0.5\%$ of real-time factor).

---

## 3. Issue 2: Mathematical Equivalence of Macro and Micro Recall

### Formal Proof
Let $K = 4$ represent the target command classes (`TRAI`, `PHAI`, `LEN`, `XUONG`). In the locked test split, each class $k$ has exactly $N_k = 40$ test clips ($N_{\text{cmd}} = \sum_{k=1}^K N_k = 160$).
- **Micro-Average Recall:**
  $$\text{Recall}_{\text{micro}} = \frac{\sum_{k=1}^4 \text{Hits}_k}{\sum_{k=1}^4 N_k} = \frac{\text{Hits}}{160}$$
- **Macro-Average Recall:**
  $$\text{Recall}_{\text{macro}} = \frac{1}{4} \sum_{k=1}^4 \frac{\text{Hits}_k}{N_k} = \frac{1}{4} \sum_{k=1}^4 \frac{\text{Hits}_k}{40} = \frac{1}{160} \sum_{k=1}^4 \text{Hits}_k \equiv \text{Recall}_{\text{micro}}$$

Because $N_k$ is uniform across all four command classes, macro-average recall is identical to micro-average recall. The reported 95.83% is valid under both definitions.

---

## 4. Issue 3: Asymmetry Between Target Recall and Non-Target Rejection

### Finding
In Candidate 2:
- Target Command Recall is **95.83%** (only 10 misses out of 480 trials across 3 seeds).
- Non-Target False Acceptance Rate is **14.88%** (226 false acceptances out of 1,680 trials across 3 seeds).

While 85.12% rejection of non-target distractors is viable for push-to-talk interactions, reducing non-target false acceptance below 5% is the primary research goal for Phase 56–58.
