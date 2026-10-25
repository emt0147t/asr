# Phase 13: Independent Confirmatory Evaluation Audit

**Workspace:** `D:\trainidentificationaudio`  
**Date:** October 19, 2026  
**Candidate Identifier:** `variant_v11_hard_negative_weighted_seed42`  
**Candidate Checkpoint:** `experiments/vieneu_kws_research_v12/bundle/checkpoint/variant_v11_hard_negative_weighted_seed42.pt`  
**Verified Checkpoint SHA-256:** `3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2`  
**Phase 13 Verdict:** `EVALUATION_BLOCKED_PRE_EVALUATION_GATES_FAILED`  
**Governance Invariant Labels:**  
- `FROZEN_CNN_CANDIDATE` (Maintained)  
- `NOT_VALIDATED_ON_NATURAL_SPEECH` (Maintained)  
- `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION` (Maintained)  

---

## 1. Executive Summary

Phase 13 was initiated to execute an independent confirmatory evaluation of the frozen Vietnamese bounded-utterance CNN candidate finalized in Phase 12. In accordance with mandatory pre-evaluation governance gates, all prerequisites, contracts, datasets, and acoustic properties were subjected to rigorous audit prior to unlocking or generating test audio.

The pre-evaluation audit identified **two blocking defects** that legally and scientifically preclude running a valid confirmatory evaluation:

1. **Gate 2 Preprocessing Inequivalence Blocker:**  
   The Phase 12 bundle contract specifies unconditional $-1.0\text{ dBFS}$ peak normalization (`infer.py`), whereas the authoritative upstream deployment contract (`DEPLOYMENT_CONTRACT.md`), the audio validator (`audio_validator.py`), and the actual Phase 11 training script (`train_v11.py`) strictly enforce clamping only when $\max(|x|) > 1.0$ and explicitly prohibit noise-floor amplification. These two contracts are mathematically and acoustically non-equivalent (diverging by up to $45\text{ dB}$ on quiet background noise, inducing false command triggers on silence). Because equivalence cannot be proven, Gate 2 mandates stopping execution immediately.
2. **Gate 3 Acoustic Feasibility & Dialect Representation Blocker:**  
   The preregistered protocol in Section 8 of the Phase 12 report mandates a minimum of 8 novel speakers covering Northern, Central, and Southern Vietnamese accents. An exhaustive audit of the approved synthetic toolchain (`Vieneu`) revealed that 100% of available Central voices (2/2) and 100% of available Southern voices (8/8) were already consumed in Phase 11 training, validation, and test splits. The residual unused synthetic voices are exclusively Northern-accented. It is therefore impossible using the approved synthetic tools to satisfy speaker isolation without violating tripartite dialect coverage.

In strict compliance with the directive:
> *"If equivalence cannot be proven, STOP and report the blocker; do not run a potentially invalid confirmatory evaluation."*  
> *"If the protocol cannot be executed safely or its independence cannot be demonstrated, stop before unlocking/evaluating the test and document the exact blocker."*

The confirmatory test set was **not unlocked, not evaluated, and not fabricated**. The candidate model remains frozen, intact, and uncompromised.

---

## 2. Frozen Candidate Identity & Verification

The primary candidate checkpoint was verified against authoritative Phase 11 and Phase 12 records before auditing gates:

| Attribute | Authoritative Value | Verification Status |
| :--- | :--- | :--- |
| **Candidate File** | `experiments/vieneu_kws_research_v12/bundle/checkpoint/variant_v11_hard_negative_weighted_seed42.pt` | Verified on disk |
| **Checksum (SHA-256)** | `3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2` | **MATCH (Exact)** |
| **Model Architecture** | `CorrectedVietnameseKWSNet` | Verified |
| **Trainable Parameters** | 138,469 parameters | Verified |
| **Class Mapping** | `[0: TRAI, 1: PHAI, 2: LEN, 3: XUONG, 4: REJECT]` | Verified |
| **Operating Threshold ($\theta^*$)** | `0.20` | Verified against `model_config.json` |

---

## 3. Mandatory Pre-Evaluation Gate Audits

### 3.1 Gate 1: Preregistered Protocol Inspection

Section 8 of `experiments/vieneu_kws_research_v12/reports/PHASE12_CNN_FINALIZATION.md` was audited. The sealed protocol preregisters:
- **Sample Size:** $N = 1,000$ audio clips:
  - 200 Target Commands (50 per keyword: TRAI, PHAI, LEN, XUONG)
  - 200 Phonetic Minimal Pairs (tonal foils: *trại, phái, phèn, lén, xuống*)
  - 200 Syntactic Carrier Collisions (keywords embedded in longer carrier phrases)
  - 300 General Conversational Non-Targets (vocabulary disjoint from keywords)
  - 100 Acoustic Ambient / Silence (office ambient, street noise, fan hum)
- **Utterance Text Isolation:** 100% unseen lexical sentences.
- **Speaker Isolation:** Minimum 8 novel speakers (4 female, 4 male) covering Northern, Central, and Southern Vietnamese accents.
- **Acceptance Criteria:** Macro command recall $\ge 95\%$, per-command recall $\ge 95\%$, non-target FAR $\le 5\%$, CPU latency $\le 50$ ms.

---

### 3.2 Gate 2: Preprocessing Contract Discrepancy (Blocking Blocker)

#### 3.2.1 Description of the Discrepancy
Two conflicting preprocessing specifications exist within the repository:

1. **Specification A (Phase 12 Bundle / `preprocessing_contract.json` & `infer.py`):**
   ```python
   # Peak normalization to -1.0 dBFS
   peak = np.max(np.abs(audio))
   if peak > 0:
       target_peak = 10.0 ** (-1.0 / 20.0)  # 0.89125
       audio = audio * (target_peak / peak)
   ```
   Every waveform is unconditionally scaled so that its peak absolute sample equals $0.89125$ ($-1.0\text{ dBFS}$).

2. **Specification B (Phase 91/92 Deployment Contract `DEPLOYMENT_CONTRACT.md`, `audio_validator.py`, and `train_v11.py`):**
   - `DEPLOYMENT_CONTRACT.md` (lines 45-46):
     > *"Peak Normalization: Clamped only if $\max(|x|) > 1.0$: $x \leftarrow \frac{x}{\max(|x|)}$"*  
     > *"Floor Amplification: STRICTLY PROHIBITED (never amplify quiet noise floor)"*
   - `audio_validator.py` (lines 173-176):
     ```python
     max_val = torch.max(torch.abs(wav))
     if max_val > 1.0:
         wav = wav / max_val
     ```
   - `train_v11.py` (lines 169-172):
     ```python
     max_val = np.max(np.abs(audio))
     if max_val > 1.0:
         audio = audio / max_val
     ```

#### 3.2.2 Mathematical Proof of Non-Equivalence
Let $x \in \mathbb{R}^{16000}$ be an input waveform with peak amplitude $A = \max |x|$.

- Under Specification A:
  $$f_A(x) = x \cdot \frac{0.89125}{A} \quad (A > 0)$$
- Under Specification B:
  $$f_B(x) = \begin{cases} x \cdot \frac{1.0}{A} & \text{if } A > 1.0 \\ x & \text{if } A \le 1.0 \end{cases}$$

**Case Analysis:**
1. **Full-Scale Speech ($A = 0.89125$):**  
   $f_A(x) = x$ and $f_B(x) = x$. Equivalence holds *only* on this single measure-zero degenerate slice.
2. **Quiet Ambient / Near-Silence Audio ($A = 0.005$):**  
   - $f_B(x) = x$ (peak remains $0.005$; energy unchanged; $0\text{ dB}$ gain).
   - $f_A(x) = x \cdot \frac{0.89125}{0.005} = 178.25 \cdot x$ ($+45.0\text{ dB}$ amplification).
   - In the Log-Mel feature representation:
     $$\mathbf{S}_{\text{logmel}}(f_A(x)) \approx \mathbf{S}_{\text{logmel}}(x) + 2 \ln(178.25) \approx \mathbf{S}_{\text{logmel}}(x) + 10.36$$
   - This massive $+10.36$ log-spectral energy shift distorts the acoustic feature floor, transforming faint random background noise into full-scale synthetic energy patterns.

#### 3.2.3 Empirical Failure
In Phase 12 development testing, running `infer.py` (Specification A) on `parity_dev_06_silence.wav` ($A \approx 0.003$) resulted in the model predicting **`TRAI`** (false accept). Conversely, evaluating the same file under Specification B yields low log-Mel activations and correctly outputs **`REJECT`**.

Because real-world confirmatory evaluations include ambient room noise and silence (100 clips in the protocol), applying Specification A would artificially induce false alarms, whereas applying Specification B deviates from `infer.py`.

**Gate 2 Verdict:**  
**Equivalence CANNOT be proven.** The two contracts are fundamentally conflicting. Execution must STOP.

---

### 3.3 Gate 3: Speaker Isolation & Tripartite Regional Accent Audit (Blocking Blocker)

#### 3.3.1 Preregistered Requirement
Section 8.1 requires:
> *"Minimum of 8 novel speakers (4 female, 4 male) covering Northern, Central, and Southern Vietnamese accents."*

#### 3.3.2 Voice Inventory Audit
The approved synthetic TTS system available in the workspace is `Vieneu` (`mode="v3turbo"`). A complete census of all 25 preset voices provided by `tts.list_preset_voices()` was conducted and mapped against historical usage:

| Regional Accent | Voice Name | Gender | Historical Split Usage | Status for Confirmatory Phase |
| :--- | :--- | :--- | :--- | :--- |
| **Central (Trung)** | `Quang Sơn` | Nam | Phase 11 Train (`train_manifest.csv`) | **CONSUMED (Ineligible)** |
| **Central (Trung)** | `Ngọc Trân` | Nữ | Phase 11 Val (`val_manifest.csv`) | **CONSUMED (Ineligible)** |
| **Southern (Nam)** | `Adam` | Nam | Phase 11 Train (`train_manifest.csv`) | **CONSUMED (Ineligible)** |
| **Southern (Nam)** | `Minh Triết` | Nam | Phase 11 Train (`train_manifest.csv`) | **CONSUMED (Ineligible)** |
| **Southern (Nam)** | `Thái Sơn` | Nam | Phase 11 Train (`train_manifest.csv`) | **CONSUMED (Ineligible)** |
| **Southern (Nam)** | `Mỹ Duyên` | Nữ | Phase 11 Train (`train_manifest.csv`) | **CONSUMED (Ineligible)** |
| **Southern (Nam)** | `Thục Đoan` | Nữ | Phase 11 Train (`train_manifest.csv`) | **CONSUMED (Ineligible)** |
| **Southern (Nam)** | `Thùy Dung` | Nữ | Phase 11 Val (`val_manifest.csv`) | **CONSUMED (Ineligible)** |
| **Southern (Nam)** | `Đức Trí` | Nam | Phase 11 Test (`test_manifest.csv`) | **CONSUMED (Ineligible)** |
| **Southern (Nam)** | `Kim Thanh` | Nữ | Phase 11 Test (`test_manifest.csv`) | **CONSUMED (Ineligible)** |
| **Northern (Bắc)** | `Mai Anh` | Nữ | Phase 11 Val (`val_manifest.csv`) | **CONSUMED (Ineligible)** |
| **Northern (Bắc)** | `Phạm Tuyên` | Nam | Phase 11 Val (`val_manifest.csv`) | **CONSUMED (Ineligible)** |
| **Northern (Bắc)** | `Ngọc Huyền` | Nữ | Phase 11 Train (`train_manifest.csv`) | **CONSUMED (Ineligible)** |
| **Northern (Bắc)** | `Thanh Bình` | Nam | Phase 11 Train (`train_manifest.csv`) | **CONSUMED (Ineligible)** |
| **Northern (Bắc)** | `Quỳnh Anh` | Nữ | Phase 11 Test (`test_manifest.csv`) | **CONSUMED (Ineligible)** |
| **Northern (Bắc)** | `Quốc Tuấn` | Nam | Phase 11 Test (`test_manifest.csv`) | **CONSUMED (Ineligible)** |
| **Northern (Bắc)** | `Adam bựa` | Nam | None (Unused) | Available (Northern only) |
| **Northern (Bắc)** | `Trúc Ly` | Nữ | None (Unused) | Available (Northern only) |
| **Northern (Bắc)** | `Thiện Minh` | Nam | None (Unused) | Available (Northern only) |
| **Northern (Bắc)** | `Hải Đăng` | Nam | None (Unused) | Available (Northern only) |
| **Northern (Bắc)** | `Thiền Tâm Đức` | Nam | None (Unused) | Available (Northern only) |
| **Northern (Bắc)** | `Minh Đức` | Nam | None (Unused) | Available (Northern only) |
| **Northern (Bắc)** | `Xuân Vĩnh` | Nam | None (Unused) | Available (Northern only) |
| **Northern (Bắc)** | `Ngọc Linh` | Nữ | None (Unused) | Available (Northern only) |
| **Northern (Bắc)** | `Đoan Trang` | Nữ | None (Unused) | Available (Northern only) |

#### 3.3.3 Acoustic Feasibility Finding
- **Novel Central Voices Available:** **0**
- **Novel Southern Voices Available:** **0**
- **Novel Northern Voices Available:** 9 (5 male, 4 female)

It is mathematically and practically impossible to synthesize 8 novel speakers covering Northern, Central, AND Southern accents with the approved synthetic toolchain without either:
1. Re-using voices from training, validation, or test sets (violating speaker isolation and invalidating out-of-sample claims), OR
2. Eliminating Central and Southern accents entirely (violating the preregistered protocol criteria).

**Gate 3 Verdict:**  
**Speaker and dialect isolation cannot be demonstrated.** Execution must STOP.

---

### 3.4 Gate 4: Natural Human Speech Boundary

- No human subject recruitment, informed consent registry, or certified acoustic recording campaign was conducted for Phase 13.
- Unconsented or historical raw field recordings remain strictly locked and untouched under governance policy.
- The evaluation was specified as synthetic-only.

---

### 3.5 Gate 5: Candidate and Metric Freezing

The candidate checkpoint (`variant_v11_hard_negative_weighted_seed42.pt`), verified SHA-256 (`3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2`), threshold $\theta^* = 0.20$, and evaluation scripts remain frozen. No post-hoc tuning or selective re-runs were conducted.

---

## 4. Synthesis of Blockers & Required Remediations

| Gate | Requirement | Audit Result | Blocker Severity |
| :--- | :--- | :--- | :--- |
| **Gate 1** | Protocol criteria review | Section 8 criteria confirmed | Pass |
| **Gate 2** | Normalization contract resolution | Non-equivalence proven ($f_A \not\equiv f_B$; $+45\text{ dB}$ noise floor shift) | **CRITICAL BLOCKER** |
| **Gate 3** | Speaker & dialect isolation | 0 novel Central & 0 novel Southern voices available in Vieneu | **CRITICAL BLOCKER** |
| **Gate 4** | Synthetic-only explicit labeling | Confirmed synthetic-only; no natural speech claimed | Pass |
| **Gate 5** | Candidate & threshold freezing | Checkpoint and threshold verified and frozen | Pass |

### Remediations Necessary to Unblock Phase 13

Before an independent confirmatory evaluation can be executed:
1. **Contract Harmonization:**  
   Update `experiments/vieneu_kws_research_v12/bundle/infer.py` and `preprocessing_contract.json` to eliminate unconditional $-1.0\text{ dBFS}$ noise floor amplification, restoring strict alignment with the upstream `train_v11.py` and `DEPLOYMENT_CONTRACT.md` clamping policy ($\max(|x|) > 1.0$).
2. **Acoustic Voice Provisioning:**  
   Either:
   - Integrate an expanded multi-speaker TTS library containing certified, distinct Central and Southern Vietnamese voice models, OR
   - Formalize an authorized human-participant speech collection protocol with verified consent, acoustic calibration, and geographic representation across Northern, Central, and Southern Vietnam.

---

## 5. Governance Register & Final Verdict

- **Phase 13 Evaluation Verdict:** `EVALUATION_BLOCKED_PRE_EVALUATION_GATES_FAILED`
- **Candidate Label:** `FROZEN_CNN_CANDIDATE` (Retained; model remains intact and undamaged).
- **Acoustic Labels:**
  - `NOT_VALIDATED_ON_NATURAL_SPEECH` (Retained; unverified on natural speech).
  - `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION` (Retained; bounded-utterance classifier only; lacks streaming VAD and open-mic false trigger suppression).
- **Git State:** Clean; `git diff --cached` is strictly empty. No files staged or committed.
