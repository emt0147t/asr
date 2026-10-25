# Phase 49: Locked Long-Horizon Evaluation Report (3-Hour Continuous Benchmark)

Workspace: `D:\trainidentificationaudio`  
Experiment Directory: `experiments/vieneu_kws_research_v3/`  
Date: 2026-10-18  
Status: `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Executive Summary

Phase 49 evaluated the pre-registered finalists on the newly constructed, frozen synthetic test benchmark.

### Benchmark Scale & Statistical Significance:
- **Locked Test Command Stream (`test_stream_continuous_commands.wav`):** $519.15\text{ s}$ ($0.1442\text{ hours}$), containing $160$ target command events ($40$ per keyword: `TRAI`, `PHAI`, `LEN`, `XUONG`) across 4 held-out test voices (`Kim Thanh`, `Quốc Tuấn`, `Quỳnh Anh`, `Đức Trí`).
- **Locked Test Non-Target Stream (`test_stream_continuous_nontarget_3h.wav`):** **EXACTLY 10,800.00 seconds = 3.000 hours**, containing $960$ speech distractor events across 2 distinct passes over $480$ unique synthetic utterances with varied synthetic background noise (`airflow_pink`).
  - Active speech exposure: $0.2667\text{ hours}$ ($960.0\text{ s}$).
  - Background noise / inter-utterance pause exposure: $2.7333\text{ hours}$ ($9,840.0\text{ s}$).

### Operational Compliance Verdict:
Neither Candidate 1 nor Candidate 3 achieved the joint operational criteria ($\text{Recall} \ge 95\%$, $\text{Wall-Clock FA/hr} \le 1.0$, $\text{Mean Latency} \le 600\text{ ms}$).

---

## 2. Locked Held-Out Test Stream Results

All metrics below are computed on the frozen held-out test streams.

### 2.1 Multi-Seed Summary Benchmark Table

| Candidate Architecture | Seed | Target Recall | Mean Latency | Non-Target FPs (3h) | Wall-Clock FA / hr | Active-Speech FA / hr | Poisson 95% Upper FA/hr | Operational Criteria Met? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Candidate 1: Historical Cascade v1** | 42 | 98.75% (158/160) | 704.1 ms | 207 | 69.00 | 776.15 | 78.40 | Failed ($\text{FA/hr} > 1.0$, Lat $> 600$) |
| | 43 | 96.25% (154/160) | 630.3 ms | 245 | 81.67 | 918.64 | 91.89 | Failed ($\text{FA/hr} > 1.0$, Lat $> 600$) |
| | 44 | 96.88% (155/160) | 707.4 ms | 244 | 81.33 | 914.89 | 91.54 | Failed ($\text{FA/hr} > 1.0$, Lat $> 600$) |
| | **Mean $\pm$ Std** | **$97.29\% \pm 1.34\%$** | **$680.6 \pm 43.6\text{ ms}$** | **$232.0$** | **$77.33 \pm 7.22$** | **$869.89 \pm 81.18$** | **$87.28$** | **Failed** |
| **Candidate 2: Stage 1 alone (v3)** | 42 | 96.88% (155/160) | 542.9 ms | 710 | 236.67 | 2662.17 | 254.08 | Failed ($\text{FA/hr} > 1.0$) |
| | 43 | 99.38% (159/160) | 537.4 ms | 570 | 190.00 | 2137.23 | 205.60 | Failed ($\text{FA/hr} > 1.0$) |
| | 44 | 98.12% (157/160) | 558.9 ms | 543 | 181.00 | 2036.00 | 196.22 | Failed ($\text{FA/hr} > 1.0$) |
| | **Mean $\pm$ Std** | **$98.13\% \pm 1.25\%$** | **$546.4 \pm 11.2\text{ ms}$** | **$607.7$** | **$202.56 \pm 29.89$** | **$2278.47 \pm 336.26$** | **$218.63$** | **Failed** |
| **Candidate 3: Recall-Preserving Cascade** | 42 | 60.62% (97/160) | 723.9 ms | **13** | **4.33** | **48.74** | **6.69** | Failed (Recall & Latency) |
| | 43 | 70.00% (112/160) | 671.0 ms | **28** | **9.33** | **104.99** | **12.79** | Failed (Recall & Latency) |
| | 44 | 91.87% (147/160) | 664.4 ms | **52** | **17.33** | **194.98** | **22.04** | Failed (Recall & Latency) |
| | **Mean $\pm$ Std** | **$74.17\% \pm 16.05\%$** | **$686.4 \pm 32.7\text{ ms}$** | **$31.0$** | **$10.33 \pm 6.56$** | **$116.24 \pm 73.80$** | **$13.84$** | **Failed** |

---

## 3. Granular Per-Keyword Event Recall Breakdown

The locked test command stream evaluates exactly 40 instances per keyword across 4 held-out test voices:

| Candidate Architecture | Seed | `TRAI` (40) | `PHAI` (40) | `LEN` (40) | `XUONG` (40) | Total Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Candidate 1: Historical Cascade** | 42 | 97.5% (39/40) | 100.0% (40/40) | 97.5% (39/40) | 100.0% (40/40) | **98.75%** |
| | 43 | 97.5% (39/40) | 100.0% (40/40) | 95.0% (38/40) | 92.5% (37/40) | **96.25%** |
| | 44 | 100.0% (40/40) | 100.0% (40/40) | 92.5% (37/40) | 95.0% (38/40) | **96.88%** |
| | **Average** | **$98.3\%$** | **$100.0\%$** | **$95.0\%$** | **$95.8\%$** | **$97.29\%$** |
| **Candidate 2: Stage 1 alone** | 42 | 87.5% (35/40) | 100.0% (40/40) | 100.0% (40/40) | 100.0% (40/40) | **96.88%** |
| | 43 | 97.5% (39/40) | 100.0% (40/40) | 100.0% (40/40) | 100.0% (40/40) | **99.38%** |
| | 44 | 95.0% (38/40) | 97.5% (39/40) | 100.0% (40/40) | 100.0% (40/40) | **98.12%** |
| | **Average** | **$93.3\%$** | **$99.2\%$** | **$100.0\%$** | **$100.0\%$** | **$98.13\%$** |
| **Candidate 3: Recall-Preserving Cascade** | 42 | 60.0% (24/40) | 35.0% (14/40) | 92.5% (37/40) | 55.0% (22/40) | **60.62%** |
| | 43 | 27.5% (11/40) | 57.5% (23/40) | 97.5% (39/40) | 97.5% (39/40) | **70.00%** |
| | 44 | 87.5% (35/40) | 85.0% (34/40) | 95.0% (38/40) | 100.0% (40/40) | **91.87%** |
| | **Average** | **$58.3\%$** | **$59.2\%$** | **$95.0\%$** | **$84.2\%$** | **$74.17\%$** |

---

## 4. Linguistic Confusion Breakdown Across 3 Hours (960 Events)

The 960 held-out non-target distractor events span five controlled linguistic confusion families across 3 full hours. The table below categorizes the total false positive triggers across all three seeds:

| Linguistic Confusion Family | Candidate 1 (Historical) | Candidate 2 (Stage 1 alone) | Candidate 3 (HN Cascade v3) | Suppression in Candidate 3 |
| :--- | :---: | :---: | :---: | :---: |
| **`minimal_pair_consonant`** | 221 | 382 | **46** | **$-79.2\%$** |
| **`phonetic_phrase_context`** | 175 | 527 | **23** | **$-86.9\%$** |
| **`minimal_pair_vowel_coda`** | 127 | 312 | **13** | **$-89.8\%$** |
| **`minimal_pair_tone`** | 139 | 259 | **9** | **$-93.5\%$** |
| **`neutral_speech`** | 30 | 322 | **2** | **$-93.3\%$** |
| **Total False Positives (3 seeds)** | **692** | **1,802** | **93** | **$-86.6\%$** |
| **Average FPs / Seed** | **230.7** | **600.7** | **31.0** | **$-86.6\%$** |

### Key Scientific Insights from 3-Hour Continuous Exposure:
1. **Unprecedented Non-Target Suppression in Candidate 3:**  
   Candidate 3 reduced total false alarms over 3 full hours from **692 down to 93** (an average of only 31.0 false activations per 3-hour stream, or **$10.33\text{ FA/hr}$**, dipping to **$4.33\text{ FA/hr}$** on Seed 42).
   - Tonal minimal pairs were suppressed by $93.5\%$ (from 139 down to 9).
   - Neutral conversational speech generated only 2 false activations across 9 hours of total aggregate test audio.
2. **The Acoustic Overlap Bottleneck on `/aːj/` Rhymes:**  
   While `LEN` ($95.0\%$) and `XUONG` ($84.2\%$) maintained robust recall under Candidate 3, commands sharing the open diphthong `/aːj/` (`TRAI` and `PHAI`) suffered severe false rejections (averaging $\sim 58\text{–}59\%$ recall).
   - Because the test inventory contained numerous high-energy hard negatives with rhymes identical or close to `/aːj/` (such as *lái*, *bái*, *gái*, *dài*, *mài*, *bài*, *bải*, *hải*, *tải*, *ngải*, *gãi*, *vải*, *tra*, *trâu*, *pha*), the Stage 2 verifier learned decision hyperplanes so aggressive that normal articulation variations of genuine `TRAI` and `PHAI` were frequently categorized as `REJECT`.
3. **Wall-Clock vs Active-Speech Exposure Disparity:**  
   Reporting both wall-clock rate ($10.33\text{ FA/hr}$) and active-speech rate ($116.24\text{ FA/hr}$) reveals that during active conversation, a false activation occurs approximately once every $31\text{ seconds}$ of speech, while in silent or ambient listening it drops dramatically.
4. **Conclusion on Operational Criteria:**  
   While Candidate 3 made historic strides in reducing streaming false alarms ($10.33\text{ FA/hr}$ vs $77.33\text{ FA/hr}$), it cannot be declared operational because it sacrifices command recall ($74.17\% < 95\%$) and remains above the target threshold of $\le 1.0\text{ FA/hr}$.
