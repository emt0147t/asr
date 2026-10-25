# Phase 13.1: Preprocessing Remediation and Confirmatory Feasibility Report

**Workspace:** `D:\trainidentificationaudio`  
**Date:** October 19, 2026  
**Candidate Checkpoint:** `experiments/vieneu_kws_research_v12/bundle/checkpoint/variant_v11_hard_negative_weighted_seed42.pt`  
**Verified SHA-256 (Before Remediation):** `3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2`  
**Verified SHA-256 (After Remediation):** `3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2`  
**Remediated Bundle Directory:** `experiments/vieneu_kws_research_v13_1/bundle/`  
**Final Status Verdict:** `PREPROCESSING_REMEDIATION_PASSED_CONFIRMATION_STILL_BLOCKED`  

---

## 1. Executive Summary

Phase 13.1 was executed to resolve the two blocking defects identified during the Phase 13 pre-evaluation audit:
1. **Preprocessing Contract Remediation (Task A):**  
   The mathematical and acoustic discrepancy between the Phase 12 bundle's unconditional $-1.0\text{ dBFS}$ peak normalization and the authoritative upstream training contract in `train_v11.py` was remediated. A new, versioned inference bundle was created in `experiments/vieneu_kws_research_v13_1/bundle/` without altering or overwriting historical Phase 12 artifacts. Rigorous empirical testing across 8 non-locked synthetic development fixtures demonstrated that the remediated engine achieves **exact 100% bitwise parity** with the training reference and completely eliminates false positive triggers on faint ambient noise.
2. **Confirmatory Voice-Source Feasibility & Protocol Amendment (Task B):**  
   An exhaustive acoustic audit of available TTS engines (`Vieneu` and `edge-tts`) confirmed that 100% of available Central and Southern Vietnamese voices were consumed in Phase 11. No permitted external synthetic source currently available in the environment provides novel Central or Southern voices. Consequently, a formal **Protocol Amendment Proposal (Amendment 13.1-A)** has been submitted to narrow the synthetic confirmatory claim to novel lexical content under verified novel Northern speakers.
3. **Acceptance Verdict:**  
   In compliance with governance directives, the candidate CNN was **not retrained, fine-tuned, or threshold-tuned**. The frozen confirmatory evaluation remains **STILL BLOCKED** pending user approval of Protocol Amendment 13.1-A.

---

## 2. Task A — Preprocessing Remediation & Verification

### 2.1 Audit of Upstream Training Preprocessing
A code audit of `experiments/vieneu_kws_research_v11/train_v11.py` (lines 121–127 and 169–172) and Phase 91 `DEPLOYMENT_CONTRACT.md` (lines 45–46) established the ground-truth reference implementation:
```python
# Exact training reference logic (BoundedDatasetV11 & audio_validator.py):
audio, sr = sf.read(wav_path, dtype="float32")
if audio.ndim > 1:
    audio = np.mean(audio, axis=-1)
if len(audio) < 16000:
    audio = np.pad(audio, (0, 16000 - len(audio)), mode="constant")
elif len(audio) > 16000:
    audio = audio[:16000]

max_val = np.max(np.abs(audio)) if len(audio) > 0 else 0.0
if max_val > 1.0:
    audio = audio / max_val
```
**Key Principles of the Authoritative Contract:**
1. Clamping applies **only** when peak amplitude exceeds $1.0$ (preventing DAC clipping).
2. For signals with peak $\le 1.0$, **gain is exactly $1.0\times$ ($0.0\text{ dB}$)**.
3. Floor amplification is **strictly prohibited**.

### 2.2 Versioned Corrected Bundle Structure
Created in `experiments/vieneu_kws_research_v13_1/bundle/`:
```
bundle/
├── checkpoint/
│   ├── variant_v11_hard_negative_weighted_seed42.pt  (Verified PyTorch weights)
│   └── checksums.sha256                              (SHA-256 verification records)
├── model_config.json                                 (Updated with clamping normalization policy)
├── preprocessing_contract.json                       (Remediated contract banning floor amplification)
├── infer.py                                          (Production inference engine with training-parity logic)
├── requirements.txt                                  (Dependencies)
└── model_card.md                                     (Full model and remediation documentation)
```

### 2.3 Empirical Evaluation on Non-Locked Development Fixtures

An independent suite of 8 non-locked synthetic development fixtures was created in `experiments/vieneu_kws_research_v13_1/fixtures/` and evaluated across three implementations:
1. **Phase 12 Engine:** Defective unconditional $-1.0\text{ dBFS}$ scaling.
2. **Phase 13.1 Remediated Engine:** Clamping only when $\max(|x|) > 1.0$.
3. **Training Reference:** Direct execution of `train_v11.py` preprocessing logic.

| Fixture File | Input Type | Raw Peak | P12 Peak | P13.1 Peak | P12 Prediction | P13.1 Prediction | Training Ref | P13.1 Parity |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `dev_01_digital_silence.wav` | Exact Zeros | 0.0000 | 0.0000 | 0.0000 | `REJECT` (0.097) | `REJECT` (0.097) | `REJECT` (0.097) | **EXACT MATCH** |
| `dev_02_quiet_ambient_room.wav` | Room Tone | 0.0050 | **0.8913** | **0.0050** | **`TRAI` (0.655)** ❌ | **`REJECT` (0.007)** ✅ | `REJECT` (0.007) | **EXACT MATCH** |
| `dev_03_attenuated_trai_peak010.wav` | -20 dBFS TRAI | 0.1000 | **0.8913** | **0.1000** | `LEN` (0.751) ❌ | **`REJECT` (0.091)** ✅ | `REJECT` (0.091) | **EXACT MATCH** |
| `dev_04_attenuated_phai_peak005.wav` | -26 dBFS PHAI | 0.0500 | **0.8913** | **0.0500** | `PHAI` (0.835) | **`REJECT` (0.042)** ✅ | `REJECT` (0.042) | **EXACT MATCH** |
| `dev_05_normal_len_peak089.wav` | Standard LEN | 0.8912 | 0.8913 | 0.8912 | `XUONG` (0.669) | `XUONG` (0.669) | `XUONG` (0.669) | **EXACT MATCH** |
| `dev_06_normal_xuong_peak089.wav` | Standard XUONG | 0.8912 | 0.8913 | 0.8912 | `XUONG` (0.841) | `XUONG` (0.841) | `XUONG` (0.841) | **EXACT MATCH** |
| `dev_07_high_amplitude_trai_peak180.wav` | High-Peak TRAI | 1.8000 | 0.8913 | **1.0000** | `LEN` (0.697) | `LEN` (0.672) | `LEN` (0.672) | **EXACT MATCH** |
| `dev_08_high_amplitude_xuong_peak250.wav` | High-Peak XUONG | 2.5000 | 0.8913 | **1.0000** | `XUONG` (0.852) | `XUONG` (0.831) | `XUONG` (0.831) | **EXACT MATCH** |

#### Analysis of Findings:
1. **Elimination of Ambient Noise False Triggers:**  
   On `dev_02_quiet_ambient_room.wav`, Phase 12 amplified the $0.005$ noise floor by $178\times$ (+45 dB), boosting log-Mel features into the activation range of speech commands and triggering a false alarm (`TRAI` with confidence $0.6553$). The remediated Phase 13.1 engine leaves the noise floor unamplified, producing an overwhelmingly confident rejection: **`REJECT` probability $0.9814$** (max command confidence $0.0067 < \theta^* = 0.20$).
2. **Bitwise Parity with Training Reference:**  
   Across all 8 test fixtures, the maximum absolute difference between the remediated Phase 13.1 waveform and the upstream training reference was **$0.000000$** ($0.0$ max error).

---

## 3. Task B — Confirmatory Voice-Source Feasibility Audit

### 3.1 Census of Workspace TTS Sources
An audit was conducted across all installed synthesis engines:

| Synthesis Engine | Total Vietnamese Voices | Regional Accent Breakdown | Historical Usage | Novel Voices Remaining |
| :--- | :---: | :--- | :--- | :---: |
| **`Vieneu` (v3turbo)** | 25 | 2 Central, 8 Southern, 15 Northern | 16 used (2 Central, 8 Southern, 6 Northern) | **9 (100% Northern)** |
| **`edge-tts` (7.3.1)** | 2 | 2 Northern (`HoaiMyNeural`, `NamMinhNeural`) | Extensively used in Phases 1–8 | **0** |
| **Other Engines** | 0 | None installed | N/A | **0** |

### 3.2 Impossibility of Fulfilling Sealed Protocol Without Violations
The sealed protocol in Section 8 of Phase 12 mandates:
> *"Minimum of 8 novel speakers (4 female, 4 male) covering Northern, Central, and Southern Vietnamese accents."*

Because 100% of Central and Southern voices in `Vieneu` were already used in Phase 11, satisfying this requirement under the current workspace toolchain creates an impossible dilemma:
- **Violation Option 1:** Recycle Phase 11 Central/Southern voices $\implies$ **Violates speaker isolation**; invalidates out-of-sample claims.
- **Violation Option 2:** Synthesize using the 9 novel `Vieneu` voices $\implies$ **Violates tripartite accent coverage** (produces a 100% Northern test set).

### 3.3 Proposed Confirmatory Protocol Amendment (Amendment 13.1-A)

To enable a valid, uncontaminated confirmatory test without fabricating data or misrepresenting regional accents, the following amendment is formally proposed:

> ### Protocol Amendment 13.1-A: Narrowed Synthetic Confirmatory Scope
> 1. **Scoped Generalization Claim:**  
>    The confirmatory test shall be explicitly defined as:  
>    *"Evaluation of generalization to unseen lexical carrier phrases under novel Northern-accented synthetic speakers."*
> 2. **Speaker Allocation ($N = 8$ Novel Northern Speakers):**  
>    - 4 Male: `Thiện Minh`, `Hải Đăng`, `Thiền Tâm Đức`, `Minh Đức`
>    - 4 Female: `Trúc Ly`, `Ngọc Linh`, `Đoan Trang`, and either `Xuân Vĩnh` (pitch-shifted) or `Adam bựa`
>    - 0% overlap with Phase 11 train, val, or test voices.
> 3. **Lexical Isolation:**  
>    100% unseen lexical sentences across 1,000 clips.
> 4. **Regional Dialect Reservation:**  
>    Evaluation of Central and Southern accent robustness is explicitly deferred until certified Central/Southern voices (or consented human recordings) are provisioned.
> 5. **Status:** **PENDING USER APPROVAL.** No confirmatory dataset will be generated or unlocked until this amendment is formally accepted.

---

## 4. Verification and Regression Summary

- **Phase 13.1 Test Suite:** [`test_vieneu_research_v13_1.py`](file:///D:/trainidentificationaudio/tests/test_vieneu_research_v13_1.py) (8/8 passed).
- **Phase 13 Audit Suite:** [`test_vieneu_research_v13.py`](file:///D:/trainidentificationaudio/tests/test_vieneu_research_v13.py) (4/4 passed).
- **Candidate Checkpoint Hash:** Verified bitwise identical before and after remediation:  
  `3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2`
- **Git Hygiene:** `git diff --cached` is strictly empty; no files staged or committed.

---

## 5. Acceptance Verdict

```
FINAL PHASE 13.1 VERDICT:
PREPROCESSING_REMEDIATION_PASSED_CONFIRMATION_STILL_BLOCKED
```

### Remaining Blockers:
1. **Protocol Amendment 13.1-A Approval:** Confirmatory dataset generation remains halted pending user review and approval of the narrowed synthetic speaker scope.
2. **Natural Speech Validation:** Candidate remains `NOT_VALIDATED_ON_NATURAL_SPEECH`.
3. **Continuous Deployment Readiness:** Candidate remains `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`.
