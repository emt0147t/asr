# Phase 13.3: Preregistered Synthetic Confirmatory Evaluation Report

**Evaluation Status:** `CONFIRMATORY_CRITERIA_PASSED`  
**Protocol Version:** 13.2.0 (Amendment 13.1-A Ratified)  
**Execution Date:** 2026-10-19  
**Candidate Identifier:** `variant_v11_hard_negative_weighted_seed42.pt`  
**Candidate Checkpoint SHA-256 (Pre & Post):** `3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2` (Byte-identical, verified)  
**Manifest CSV SHA-256:** `b22e8f7f955fcccf2436ba1b11c61c7f02f528e9aa834f1f5e85a61d1e0f99c3`  
**Operating Decision Threshold ($\theta^*$):** `0.20`  
**Operational Assessment:** `FROZEN_CNN_CANDIDATE` | `NOT_VALIDATED_ON_NATURAL_SPEECH` | `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`

---

## 1. Executive Summary

This report documents the execution of the independent, one-shot synthetic confirmatory evaluation (Phase 13.3) for the frozen Vietnamese bounded-utterance CNN candidate `variant_v11_hard_negative_weighted_seed42.pt`.

Under user-ratified **Amendment 13.1-A**, the evaluation tested candidate generalization to unseen lexical carrier phrases under 8 novel Northern-accented synthetic voices with 0% speaker leakage and 0% lexical text overlap with prior phases.

The evaluation was executed in a single immutable forward pass across the frozen 1,000-clip confirmatory dataset using the remediated Phase 13.1 reference frontend ($N_{\text{fft}} = 512$, $W_{\text{len}} = 400$, $H_{\text{len}} = 160$, 40 Slaney bands, peak clamping if $\max(|x|) > 1.0$) and calibrated operating threshold $\theta^*_{\text{cmd}} = 0.20$.

### Key Findings
1. **Overall Five-Class Accuracy Reached 99.20% (992/1000)** (95% CI: 98.4% – 99.6%), comfortably exceeding the $\ge 95.0\%$ acceptance threshold.
2. **Aggregate Command Recall Reached 99.00% (198/200)** (95% CI: 96.4% – 99.7%), with every individual command meeting or exceeding the 95.0% threshold:
   - `TRAI` Recall: **98.00%** (49/50)
   - `PHAI` Recall: **100.00%** (50/50)
   - `LEN` Recall: **98.00%** (49/50)
   - `XUONG` Recall: **100.00%** (50/50)
3. **Non-Target False Alarm Rate (FAR) was 0.75% (6/800)** (95% CI: 0.3% – 1.6%), far below the $\le 5.0\%$ ceiling ($\le 40/800$).
   - Phonetic minimal pairs FAR: **0.00% (0/200)** (zero false alarms across all tonal/consonant foils).
   - Acoustic background and silence FAR: **0.00% (0/100)**.
   - Syntactic carrier collisions FAR: **1.50% (3/200)**.
   - General conversational sentences FAR: **1.00% (3/300)**.
4. **Single-Clip CPU Latency averaged 1.80 ms** (p95 = 2.12 ms, total dataset inference in 1.91 s), easily satisfying the $\le 50.0\text{ ms}$ latency target.
5. **Final Confirmatory Verdict:** **`CONFIRMATORY_CRITERIA_PASSED`**. All 8 preregistered acceptance criteria were satisfied on the first, untouched pass.

---

## 2. Governance, Scope, and Mandatory Reservations

In accordance with experiment governance:
1. **Scope of Claim:** Claims are strictly limited to **generalization on unseen lexical carrier phrases and novel Northern-accented synthetic voices**.
2. **Regional Accent Limitations:** Central and Southern regional dialect evaluations remain **explicitly out of scope and pending** until certified multi-accent voice engines or consented human field recordings are available.
3. **`NOT_VALIDATED_ON_NATURAL_SPEECH`:** This candidate has not undergone natural human speech validation. Physical acoustic testing under environmental noise remains pending.
4. **`NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`:** The candidate operates strictly on pre-segmented 1.0-second bounded utterances. Continuous open-mic operation requires streaming VAD and buffer management not present in this candidate.
5. **Post-Hoc Invariance:** This confirmatory dataset is now considered **inspected diagnostic data**. The model checkpoint, weights, hyperparameters, preprocessing, and decision threshold were not and will not be adjusted based on these results.

---

## 3. Dataset Architecture & Pre-Synthesis Gate Verification

The confirmatory dataset was synthesized and locked prior to running inference:

| Category | Planned Count | Generated Count | Status | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Target Commands** | 200 | 200 | LOCKED | 50 TRAI, 50 PHAI, 50 LEN, 50 XUONG |
| **Phonetic Minimal Pairs** | 200 | 200 | LOCKED | 50 foils per keyword direction (0% historical overlap) |
| **Syntactic Carrier Collisions** | 200 | 200 | LOCKED | 50 carrier sentences per keyword (0% historical overlap) |
| **General Conversational** | 300 | 300 | LOCKED | Diverse Vietnamese sentences (0% historical overlap) |
| **Acoustic Background / Silence** | 100 | 100 | LOCKED | 25 silence, 25 fan hum, 25 room tone, 25 textures |
| **Total Clips** | **1,000** | **1,000** | **LOCKED** | **Manifest SHA-256: `b22e8f7f955fcccf2436ba1b11c61c7f02f528e9aa834f1f5e85a61d1e0f99c3`** |

### 8 Novel Northern Voices Verification
All 8 voices were verified to have 0% overlap with Phase 11 Train, Validation, and Test voices:
1. `Trúc Ly` (spk_north_01, Female, Natural): 122 clips
2. `Ngọc Linh` (spk_north_02, Female, Narrative): 122 clips
3. `Đoan Trang` (spk_north_03, Female, Natural): 110 clips
4. `Thiện Minh` (spk_north_04, Male, Narrative): 110 clips
5. `Hải Đăng` (spk_north_05, Male, Natural): 109 clips
6. `Minh Đức` (spk_north_06, Male, News): 109 clips
7. `Xuân Vĩnh` (spk_north_07, Male, Natural): 109 clips
8. `Thiền Tâm Đức` (spk_north_08, Male, Narrative): 109 clips
*(Acoustic background clips: 100 clips)*

---

## 4. Preregistered Acceptance Criteria & Execution Results

| Evaluation Criterion | Preregistered Threshold | Actual Outcome | Status | 95% Wilson Confidence Interval |
| :--- | :---: | :---: | :---: | :---: |
| **TRAI Recall** | $\ge 95.0\%$ ($\ge 48/50$) | **98.00%** (49/50) | **PASSED** | [89.5%, 99.6%] |
| **PHAI Recall** | $\ge 95.0\%$ ($\ge 48/50$) | **100.00%** (50/50) | **PASSED** | [92.9%, 100.0%] |
| **LEN Recall** | $\ge 95.0\%$ ($\ge 48/50$) | **98.00%** (49/50) | **PASSED** | [89.5%, 99.6%] |
| **XUONG Recall** | $\ge 95.0\%$ ($\ge 48/50$) | **100.00%** (50/50) | **PASSED** | [92.9%, 100.0%] |
| **Aggregate Command Recall** | $\ge 95.0\%$ ($\ge 190/200$) | **99.00%** (198/200) | **PASSED** | [96.4%, 99.7%] |
| **Non-Target False Alarm Rate** | $\le 5.0\%$ ($\le 40/800$) | **0.75%** (6/800) | **PASSED** | [0.3%, 1.6%] |
| **Overall Five-Class Accuracy** | $\ge 95.0\%$ ($\ge 950/1000$) | **99.20%** (992/1000) | **PASSED** | [98.4%, 99.6%] |
| **Single-Clip CPU Latency** | Mean $\le 50$ ms, p95 $\le 50$ ms | Mean **1.80 ms**, p95 **2.12 ms** | **PASSED** | Max 58.85 ms (cold start) |
| **Overall Verdict** | **All 8 Criteria Passed** | **8 / 8 Passed** | **CONFIRMATORY_CRITERIA_PASSED** | — |

---

## 5. Detailed Breakdown of Results

### 5.1 Per-Class Performance
- **`TRAI`**: Recall = 98.00% (49/50), Precision = 100.00% (49/49), F1 = 0.9899.
- **`PHAI`**: Recall = 100.00% (50/50), Precision = 94.34% (50/53), F1 = 0.9709.
- **`LEN`**: Recall = 98.00% (49/50), Precision = 98.00% (49/50), F1 = 0.9800.
- **`XUONG`**: Recall = 100.00% (50/50), Precision = 92.59% (50/54), F1 = 0.9615.
- **`REJECT`**: Recall = 99.25% (794/800), Precision = 100.00% (794/794), F1 = 0.9962.

### 5.2 Non-Target Category FAR Breakdown
- **Phonetic Minimal Pairs ($N = 200$):** 0 / 200 false alarms (**0.00% FAR**). Zero tonal, consonant, or vowel minimal pairs triggered a command.
- **Syntactic Carrier Collisions ($N = 200$):** 3 / 200 false alarms (**1.50% FAR**).
- **General Conversational Sentences ($N = 300$):** 3 / 300 false alarms (**1.00% FAR**).
- **Acoustic Background & Silence ($N = 100$):** 0 / 100 false alarms (**0.00% FAR**).

### 5.3 5x5 Confusion Matrix

```
True \ Pred      TRAI    PHAI     LEN   XUONG  REJECT   Total
-------------------------------------------------------------
TRAI               49       1       0       0       0      50
PHAI                0      50       0       0       0      50
LEN                 0       1      49       0       0      50
XUONG               0       0       0      50       0      50
REJECT              0       1       1       4     794     800
-------------------------------------------------------------
Total Pred         49      53      50      54     794    1000
```

---

## 6. Error Audit (All 8 Misclassifications)

Across all 1,000 clips, exactly 8 misclassifications occurred (2 False Negatives, 6 False Alarms):

| Clip ID | Category | Ground Truth | Predicted | Confidence | Intended Text | Error Description |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| `v13_conf_cmd_trai_006` | target_commands | `TRAI` | `PHAI` | 0.284 | "trái" (spk_north_01) | Subtle acoustic confusion with /aːj/ rhyme off-glide |
| `v13_conf_cmd_len_007` | target_commands | `LEN` | `PHAI` | 0.219 | "lên" (spk_north_01) | Low confidence trigger above $\theta^* = 0.20$ |
| `v13_conf_carrier_phai_022` | syntactic_carrier | `REJECT` | `PHAI` | 0.412 | "bạn phải quan sát gương chiếu hậu thật kỹ" | Embedded keyword "phải" triggered bounded window |
| `v13_conf_carrier_xuong_008` | syntactic_carrier | `REJECT` | `XUONG` | 0.385 | "mực nước sông đang rút xuống rất nhanh chóng" | Embedded keyword "xuống" triggered bounded window |
| `v13_conf_carrier_xuong_041` | syntactic_carrier | `REJECT` | `XUONG` | 0.321 | "đoàn thám hiểm quyết định leo xuống hang sâu" | Embedded keyword "xuống" triggered bounded window |
| `v13_conf_conv_087` | conversational | `REJECT` | `LEN` | 0.245 | "học sinh chuẩn bị sách vở đến trường học" | Acoustic resonance in "trường" mildly triggered LEN |
| `v13_conf_conv_142` | conversational | `REJECT` | `XUONG` | 0.298 | "chúng tôi đi dạo quanh hồ tây vào buổi chiều" | Acoustic resonance in "quanh/chiều" triggered XUONG |
| `v13_conf_conv_215` | conversational | `REJECT` | `XUONG` | 0.262 | "tiếng chuông chùa ngân vang lúc hoàng hôn buông" | Acoustic resonance in "chuông/buông" triggered XUONG |

**Analytical Note on Carrier Collisions:**  
In 3 out of 200 carrier sentences, the embedded keyword segment within the 1.0s window carried sufficient standalone energy to exceed the calibrated $\theta^* = 0.20$ threshold. This confirms the necessity of streaming VAD and contextual keyword boundary modeling for full continuous open-mic deployment.

---

## 7. Machine-Readable Artifacts & Provenance Hashes

- **Candidate Checkpoint:** [`experiments/vieneu_kws_research_v13_1/bundle/checkpoint/variant_v11_hard_negative_weighted_seed42.pt`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_1/bundle/checkpoint/variant_v11_hard_negative_weighted_seed42.pt)
- **Candidate SHA-256:** `3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2`
- **Evaluation Manifest CSV:** [`experiments/vieneu_kws_research_v13_3/data/confirmatory_evaluation_manifest.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_3/data/confirmatory_evaluation_manifest.csv) (SHA-256: `b22e8f7f955fcccf2436ba1b11c61c7f02f528e9aa834f1f5e85a61d1e0f99c3`)
- **Predictions CSV:** [`experiments/vieneu_kws_research_v13_3/predictions/confirmatory_predictions.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_3/predictions/confirmatory_predictions.csv)
- **Predictions JSON:** [`experiments/vieneu_kws_research_v13_3/predictions/confirmatory_predictions.json`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_3/predictions/confirmatory_predictions.json)
- **Metrics Summary JSON:** [`experiments/vieneu_kws_research_v13_3/reports/confirmatory_evaluation_metrics.json`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_3/reports/confirmatory_evaluation_metrics.json)
- **Dataset Integrity Report:** [`experiments/vieneu_kws_research_v13_3/data/dataset_integrity_report.json`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_3/data/dataset_integrity_report.json)
