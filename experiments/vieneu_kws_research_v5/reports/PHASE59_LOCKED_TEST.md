# Phase 59 — Locked Test Report on Independent Dataset v5

**Workspace:** `D:\trainidentificationaudio`  
**Experiment Directory:** `experiments/vieneu_kws_research_v5/`  
**Test Manifest:** `experiments/vieneu_kws_research_v5/data/splits/test_manifest.csv`  
**Predictions JSON:** `experiments/vieneu_kws_research_v5/predictions/locked_test_summary.json`  
**Per-Clip Predictions CSV:** `experiments/vieneu_kws_research_v5/predictions/per_clip_predictions.csv`  
**Status Banner:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Executive Summary

Phase 59 evaluated the two frozen finalists on the newly generated, strictly held-out locked test set (`test_manifest.csv`: 720 clips across 4 unseen test voices: Kim Thanh, Quốc Tuấn, Quỳnh Anh, Đức Trí). All evaluations were conducted with single-clip sequential inference on host CPU (4 threads) using operating thresholds frozen in Phase 58.

### Predeclared Proposed Acceptance Criteria:
- **Per-Keyword Target Recall:** $\ge 95.0\%$ for EACH individual target keyword (`TRAI`, `PHAI`, `LEN`, `XUONG`), assessed per seed and in aggregate.
- **Non-Target False Acceptance Rate (FAR):** $\le 5.0\%$ on 560 non-target clips.
- **Decision Latency:** $\le 50.0\,\text{ms}$ per 1-second clip on CPU.

### Locked Test Key Results:
1. **Finalist 1 (Cascade v1 Bounded Pipeline) Achieved Outstanding Non-Target Suppression:**
   - **Mean Non-Target False Acceptance Rate (FAR):** **4.64%** (Achieved the $\le 5.0\%$ ceiling!).
   - **Non-Target Rejection Accuracy (TNR):** **95.36%** (Rejection of 534/560 non-target clips).
   - **Mean Command Recall:** **92.29%** (`TRAI`: **98.33%**, `PHAI`: **90.83%**, `LEN`: **90.83%**, `XUONG`: **89.17%**).
   - **Overall 5-Class Accuracy:** **94.68%** (Seed 42 reached **96.67%**).
   - **Mean CPU Latency:** **4.15 ms** per clip (P90 = 7.05 ms).
2. **Finalist 2 (Stage 1 CNN alone) Retained High Recall with High False Acceptance:**
   - **Mean Command Recall:** **96.04%** (Exceeded 95% target).
   - **Mean False Acceptance Rate:** **18.39%** (Failed $\le 5\%$ ceiling by $3.7\times$).
   - **Overall 5-Class Accuracy:** **84.81%**.
   - **Mean CPU Latency:** **1.62 ms** per clip.
3. **Criteria Assessment:** While Finalist 1 successfully met the $\le 5.0\%$ FAR target ($4.64\%$) and the $\le 50\,\text{ms}$ latency target ($4.15\,\text{ms}$), its macro recall of $92.29\%$ did not reach the proposed $95.0\%$ threshold for each individual keyword.

---

## 2. Test Split Composition and Statistics

- **Total Clips:** 720
- **Sampling Rate / Format:** 16,000 Hz, 16-bit mono PCM, exactly 16,000 samples (1.000 s).
- **Target Command Clips (160 total, 40 per keyword):**
  - `TRAI`: 40 clips (10 per voice across Kim Thanh, Quốc Tuấn, Quỳnh Anh, Đức Trí)
  - `PHAI`: 40 clips (10 per voice)
  - `LEN`: 40 clips (10 per voice)
  - `XUONG`: 40 clips (10 per voice)
- **Non-Target Distractor Clips (480 total):**
  - 80 unique Vietnamese texts across 4 held-out voices (120 per voice) spanning tonal minimal pairs, consonant minimal pairs, vowel/coda variations, and natural conversational sentences.
- **Synthetic Silence Clips (80 total):**
  - 20 clips each of `near_silence`, `hvac_hum`, `office_ambient`, `airflow_pink`.

---

## 3. Detailed Benchmark Results Across Seeds

### Finalist 1: Cascade v1 Bounded Pipeline
- Checkpoints: `experiments/vieneu_kws_v2/checkpoints/` + `experiments/vieneu_kws_cascade_v1/checkpoints/`
- Operating Parameters: Seed 42: $(\theta_{s1}=0.30, \theta_{s2}=0.55)$; Seeds 43 & 44: $(\theta_{s1}=0.30, \theta_{s2}=0.20)$

| Seed | Command Recall (95% CI) | False Acceptance Rate (95% CI) | Rejection Rate | 5-Class Accuracy | CPU Latency (Mean / P90) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Seed 42** | 93.12% (88.11%, 96.12%) | **2.32%** (1.36%, 3.93%) | **97.68%** | **96.67%** | 4.06 ms / 7.05 ms |
| **Seed 43** | 93.12% (88.11%, 96.12%) | **5.18%** (3.63%, 7.34%) | **94.82%** | **94.44%** | 4.14 ms / 6.89 ms |
| **Seed 44** | 90.62% (85.11%, 94.24%) | **6.43%** (4.68%, 8.77%) | **93.57%** | **92.92%** | 4.24 ms / 6.99 ms |
| **Mean** | **92.29%** | **4.64%** | **95.36%** | **94.68%** | **4.15 ms / 6.98 ms** |

#### Per-Keyword Performance for Finalist 1 (Mean across seeds):
- **`TRAI`**: Recall = **98.33%** | Precision = **78.07%** | F1 = **86.91%**
- **`PHAI`**: Recall = **90.83%** | Precision = **95.06%** | F1 = **92.80%**
- **`LEN`**: Recall = **90.83%** | Precision = **78.18%** | F1 = **84.01%**
- **`XUONG`**: Recall = **89.17%** | Precision = **94.20%** | F1 = **91.54%**

---

### Finalist 2: Stage 1 CNN Alone
- Checkpoint: `experiments/vieneu_kws_v2/checkpoints/`
- Operating Parameters: Seed 42: $\theta=0.85$; Seed 43: $\theta=0.80$; Seed 44: $\theta=0.75$

| Seed | Command Recall (95% CI) | False Acceptance Rate (95% CI) | Rejection Rate | 5-Class Accuracy | CPU Latency (Mean / P90) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Seed 42** | 95.62% (91.25%, 97.86%) | 13.93% (11.31%, 17.04%) | 86.07% | 88.19% | 1.66 ms / 2.05 ms |
| **Seed 43** | 93.75% (88.88%, 96.57%) | 15.71% (12.93%, 18.96%) | 84.29% | 86.39% | 1.62 ms / 1.95 ms |
| **Seed 44** | 98.75% (95.56%, 99.66%) | 25.54% (22.10%, 29.31%) | 74.46% | 79.86% | 1.57 ms / 1.88 ms |
| **Mean** | **96.04%** | **18.39%** | **81.61%** | **84.81%** | **1.62 ms / 1.96 ms** |

---

## 4. Confusion Matrices (Aggregated Across 3 Seeds)

### Finalist 1 (Cascade v1 Bounded Pipeline):
Total predictions: $720 \times 3 = 2,160$ clips.

```
                     Predicted Class (Summed across 3 Seeds)
Actual Class │   TRAI   │   PHAI   │    LEN   │  XUONG   │  REJECT  │ Total
─────────────┼──────────┼──────────┼──────────┼──────────┼──────────┼───────
TRAI         │   118    │     0    │     0    │     0    │     2    │  120
PHAI         │     0    │   109    │     0    │     0    │    11    │  120
LEN          │     0    │     0    │   109    │     0    │    11    │  120
XUONG        │     0    │     0    │     0    │   107    │    13    │  120
─────────────┼──────────┼──────────┼──────────┼──────────┼──────────┼───────
REJECT (NT)  │    34    │     6    │    31    │     7    │  1,602   │ 1,680
```

### Acoustic Key Takeaways:
1. **Zero Cross-Command Confusion:** Commands are never confused with other commands.
2. **Rejection Selectivity:** 1,602 out of 1,680 non-target clips were correctly rejected (**95.36% true negative rate**).
3. **Rhyme Leakage Controlled:** Leaks into `TRAI` were reduced to 34 across 3 seeds (only 2.0% of non-targets), and into `PHAI` were reduced to only 6 (0.36% of non-targets).
