# Phase 13.2: Confirmatory Evaluation Preparation and Protocol Freeze

**Workspace:** `D:\trainidentificationaudio`  
**Date:** October 19, 2026  
**Candidate Checkpoint:** `experiments/vieneu_kws_research_v12/bundle/checkpoint/variant_v11_hard_negative_weighted_seed42.pt`  
**Verified Checkpoint SHA-256:** `3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2`  
**Operating Decision Threshold ($\theta^*$):** `0.20`  
**Protocol Status:** `AMENDMENT_13_1_A_FROZEN_READY_FOR_SYNTHESIS`  
**Governance Invariant Labels:**  
- `FROZEN_CNN_CANDIDATE` (Maintained)  
- `NOT_VALIDATED_ON_NATURAL_SPEECH` (Maintained)  
- `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION` (Maintained)  

---

## 1. Executive Summary & Protocol Amendment Ratification

Following explicit user approval of **Protocol Amendment 13.1-A**, Phase 13.2 executes the confirmatory preparation stage. Prior to generating audio data or unlocking the test, all protocol dimensions, speaker allocations, lexical splits, preprocessing implementations, checkpoint hashes, thresholds, metrics, and acceptance criteria have been frozen and verified.

### 1.1 Scope Boundaries Under Amendment 13.1-A
1. **Narrowed Confirmatory Scope:**  
   The confirmatory evaluation is formally restricted to:  
   > **Generalization of the frozen CNN candidate on unseen lexical carrier phrases under 8 novel Northern-accented synthetic speakers.**
2. **Explicit Regional Dialect Holdout:**  
   Because all Central (2/2) and Southern (8/8) synthetic voices in `Vieneu` were consumed during Phase 11 training and validation, regional generalization across Central and Southern dialects is **explicitly held out of scope and pending** until certified independent voices or consented human recordings become available.
3. **Operational Invariants:**  
   - `NOT_VALIDATED_ON_NATURAL_SPEECH`: Synthetic speech only. No natural human speech validation is claimed.
   - `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`: Operates solely as an isolated bounded-utterance classifier. Continuous open-mic operation requires streaming VAD and buffer logic not present in this candidate.
   - `ZERO_RETRAINING_POLICY`: The candidate CNN is completely frozen. No training, fine-tuning, or threshold calibration will occur on the confirmatory dataset.

---

## 2. Frozen Candidate Identity & Checkpoint Verification

| Parameter | Frozen Specification |
| :--- | :--- |
| **Model Candidate Name** | `variant_v11_hard_negative_weighted_seed42` |
| **Primary Checkpoint Path** | `experiments/vieneu_kws_research_v12/bundle/checkpoint/variant_v11_hard_negative_weighted_seed42.pt` |
| **Verified SHA-256 Checksum** | `3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2` |
| **Total Trainable Parameters** | 138,469 parameters |
| **Model Architecture** | `CorrectedVietnameseKWSNet` (4 Conv2D blocks + BatchNorm + MaxPool + Dropout + FC) |
| **Class Order** | `[0: TRAI, 1: PHAI, 2: LEN, 3: XUONG, 4: REJECT]` |
| **Operating Threshold ($\theta^*$)** | `0.20` |

---

## 3. Frozen Speaker Allocation (8 Novel Northern Voices)

To guarantee 0% speaker leakage, exactly 8 novel Northern-accented synthetic voices from `Vieneu` (`mode="v3turbo"`) have been assigned. None of these voices appeared in Phase 11 training, validation, or test sets:

| Speaker ID | Voice Name in Vieneu | Gender | Accent | Prosody Style | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `spk_north_01` | `Trúc Ly` | Nữ (Female) | Bắc (Northern) | Tự nhiên (Natural) | **VERIFIED NOVEL** |
| `spk_north_02` | `Ngọc Linh` | Nữ (Female) | Bắc (Northern) | Kể chuyện (Narrative) | **VERIFIED NOVEL** |
| `spk_north_03` | `Đoan Trang` | Nữ (Female) | Bắc (Northern) | Tự nhiên (Natural) | **VERIFIED NOVEL** |
| `spk_north_04` | `Thiện Minh` | Nam (Male) | Bắc (Northern) | Kể chuyện (Narrative) | **VERIFIED NOVEL** |
| `spk_north_05` | `Hải Đăng` | Nam (Male) | Bắc (Northern) | Tự nhiên (Natural) | **VERIFIED NOVEL** |
| `spk_north_06` | `Minh Đức` | Nam (Male) | Bắc (Northern) | Tin tức (News) | **VERIFIED NOVEL** |
| `spk_north_07` | `Xuân Vĩnh` | Nam (Male) | Bắc (Northern) | Tự nhiên (Natural) | **VERIFIED NOVEL** |
| `spk_north_08` | `Thiền Tâm Đức`| Nam (Male) | Bắc (Northern) | Kể chuyện (Narrative) | **VERIFIED NOVEL** |

*(Spare alternate: `Adam bựa` — Nam · Bắc · Tự nhiên)*

---

## 4. Frozen Confirmatory Dataset Architecture ($N = 1,000$ Clips)

The confirmatory evaluation dataset architecture is structured into 5 balanced categories:

```
Total Confirmatory Clips: 1,000
├── Target Commands: 200 clips (20.0%)
│   ├── TRAI: 50 clips (balanced across 8 speakers, 6-7 clips/spk)
│   ├── PHAI: 50 clips (balanced across 8 speakers, 6-7 clips/spk)
│   ├── LEN: 50 clips (balanced across 8 speakers, 6-7 clips/spk)
│   └── XUONG: 50 clips (balanced across 8 speakers, 6-7 clips/spk)
└── Non-Target Rejections: 800 clips (80.0%)
    ├── Phonetic Minimal Pairs: 200 clips (50 per keyword direction)
    ├── Syntactic Carrier Collisions: 200 clips (50 per keyword direction)
    ├── General Conversational Non-Targets: 300 clips (diverse vocabulary)
    └── Acoustic Background / Silence: 100 clips (ambient, noise, silence)
```

### 4.1 Lexical Independence Audit Results
The complete catalogue of 700 non-target texts was audited against all historical datasets:
- Phase 11 Training Split (`train_manifest.csv`)
- Phase 11 Validation Split (`val_manifest.csv`)
- Phase 11 Locked Test Split (`test_manifest.csv`)
- Phase 90 Review Manifest (`review_manifest.csv`)

| Metric | Audit Result | Status |
| :--- | :--- | :--- |
| **Historical Unique Non-Command Texts Audited** | 320 sentences | Verified |
| **Confirmatory Novel Non-Command Texts Authored** | 700 sentences | Verified |
| **Lexical Overlap Count** | **0 sentences** | **ZERO LEAKAGE** |
| **Lexical Overlap Ratio** | **0.00%** | **PASS** |

The authoritative lexicon catalogue is permanently saved in:  
[`experiments/vieneu_kws_research_v13_2/lexicon/confirmatory_lexicon_catalogue.json`](file:///D:/trainidentificationaudio/experiments/vieneu_kws_research_v13_2/lexicon/confirmatory_lexicon_catalogue.json)  
Audit record:  
[`experiments/vieneu_kws_research_v13_2/lexicon/lexical_independence_audit.json`](file:///D:/trainidentificationaudio/experiments/vieneu_kws_research_v13_2/lexicon/lexical_independence_audit.json)

---

## 5. Frozen Preprocessing Contract Implementation

The evaluation pipeline is bound to the remediated Phase 13.1 reference engine:  
[`experiments/vieneu_kws_research_v13_1/bundle/infer.py`](file:///D:/trainidentificationaudio/experiments/vieneu_kws_research_v13_1/bundle/infer.py)

1. **Format:** 16,000 Hz, Mono, 1.000 second ($16,000$ samples), PCM float32.
2. **Crop / Pad Policy:** Pad right with zeros if $< 16,000$ samples; truncate if $> 16,000$ samples.
3. **Clamping Normalization Policy:**
   $$\text{if } \max(|x|) > 1.0 \implies x \leftarrow \frac{x}{\max(|x|)}; \quad \text{else } x \text{ is unchanged.}$$
   *Floor amplification is strictly prohibited (0.0 dB gain on quiet noise floors).*
4. **Log-Mel Frontend:** Slaney Mel scale, 40 Mel bins, 101 frames ($N_{\text{fft}} = 512$, $W_{\text{len}} = 400$, $H_{\text{len}} = 160$, center reflection padding, $\epsilon = 1.0 \times 10^{-6}$).

---

## 6. Preregistered Acceptance Criteria

The confirmatory test will be executed as a strict one-shot evaluation scored against the following preregistered thresholds:

| Evaluation Metric | Preregistered Acceptance Threshold |
| :--- | :--- |
| **TRAI Command Recall** | $\ge 95.0\%$ ($\ge 48 / 50$ clips) |
| **PHAI Command Recall** | $\ge 95.0\%$ ($\ge 48 / 50$ clips) |
| **LEN Command Recall** | $\ge 95.0\%$ ($\ge 48 / 50$ clips) |
| **XUONG Command Recall** | $\ge 95.0\%$ ($\ge 48 / 50$ clips) |
| **Macro / Mean Command Recall** | $\ge 95.0\%$ ($\ge 190 / 200$ clips) |
| **Non-Target False Alarm Rate (FAR)** | $\le 5.0\%$ ($\le 40 / 800$ false accepts) |
| **Overall Five-Class Accuracy** | $\ge 95.0\%$ ($\ge 950 / 1000$ clips) |
| **CPU Single-Clip Decision Latency** | Mean $\le 50.0$ ms (p95 $\le 50.0$ ms) |

---

## 7. Readiness Register and Next Steps

| Preparation Dimension | Readiness Status | Evidence |
| :--- | :--- | :--- |
| **Protocol Amendment 13.1-A** | **FROZEN & RATIFIED** | Section 1 of this report & user approval |
| **Candidate Checkpoint Integrity** | **FROZEN & VERIFIED** | SHA-256: `3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2` |
| **Speaker Allocations** | **FROZEN & VERIFIED** | 8 novel Northern voices (0% Phase 11 overlap) |
| **Lexical Split** | **FROZEN & VERIFIED** | 1,000 clips catalogue (0.0% historical text overlap) |
| **Preprocessing Pipeline** | **FROZEN & VERIFIED** | Clamping only if $\max > 1.0$; bitwise verified in Phase 13.1 |
| **Operating Threshold** | **FROZEN & VERIFIED** | $\theta^* = 0.20$ |
| **Acceptance Criteria** | **FROZEN & PREREGISTERED** | Recall $\ge 95\%$, FAR $\le 5\%$, Latency $\le 50$ ms |
| **Regression Tests** | **PASSED** | [`test_vieneu_research_v13_2.py`](file:///D:/trainidentificationaudio/tests/test_vieneu_research_v13_2.py) (6/6 passed) |

**Conclusion:**  
All pre-evaluation governance gates, specifications, and manifests are completely frozen, audited, and ready. Audio data generation and one-shot evaluation may proceed upon instruction.
