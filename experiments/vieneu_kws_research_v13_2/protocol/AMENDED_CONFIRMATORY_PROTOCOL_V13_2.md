# Amended Confirmatory Evaluation Protocol (Amendment 13.1-A)

**Protocol Version:** 13.2.0 (Amendment 13.1-A Ratified)  
**Workspace:** `D:\trainidentificationaudio`  
**Date:** October 19, 2026  
**Candidate Identifier:** `variant_v11_hard_negative_weighted_seed42`  
**Candidate Checkpoint:** `experiments/vieneu_kws_research_v12/bundle/checkpoint/variant_v11_hard_negative_weighted_seed42.pt`  
**Authoritative Checkpoint SHA-256:** `3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2`  
**Operating Decision Threshold ($\theta^*$):** `0.20`  

---

## 1. Preregistered Scope and Operational Boundaries

Under user-approved **Protocol Amendment 13.1-A**, the scope of the confirmatory evaluation is formally defined as:
> **Generalization of the frozen Vietnamese bounded-utterance CNN candidate to unseen lexical carrier phrases and novel Northern-accented synthetic speakers.**

### Explicit Out-of-Scope Reservations:
1. **Central and Southern Accent Validation:**  
   Because 100% of available Central (2/2) and Southern (8/8) synthetic voices in the approved toolchain (`Vieneu`) were consumed in Phase 11 training and validation, regional generalization across Central and Southern dialects is **explicitly held out of scope and pending** until certified multi-accent voice engines or consented human field recordings are available.
2. **`NOT_VALIDATED_ON_NATURAL_SPEECH`:**  
   The candidate has not undergone natural human speech validation.
3. **`NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`:**  
   The candidate operates strictly on pre-segmented 1.0-second bounded utterances. Continuous open-mic operation requires streaming VAD and buffer logic not present in this candidate.

---

## 2. Speaker Assignment (8 Novel Northern Voices)

To guarantee 0% speaker leakage, exactly 8 novel Northern-accented voices from `Vieneu` (v3turbo) are allocated. None of these voices appeared in Phase 11 training, validation, or test sets:

| Speaker ID | Voice Name in Vieneu | Gender | Accent | Prosody Style | Historical Usage | Eligibility |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `spk_north_01` | `Trúc Ly` | Nữ (Female) | Bắc (Northern) | Tự nhiên (Natural) | None (Unused) | **VERIFIED NOVEL** |
| `spk_north_02` | `Ngọc Linh` | Nữ (Female) | Bắc (Northern) | Kể chuyện (Narrative) | None (Unused) | **VERIFIED NOVEL** |
| `spk_north_03` | `Đoan Trang` | Nữ (Female) | Bắc (Northern) | Tự nhiên (Natural) | None (Unused) | **VERIFIED NOVEL** |
| `spk_north_04` | `Thiện Minh` | Nam (Male) | Bắc (Northern) | Kể chuyện (Narrative) | None (Unused) | **VERIFIED NOVEL** |
| `spk_north_05` | `Hải Đăng` | Nam (Male) | Bắc (Northern) | Tự nhiên (Natural) | None (Unused) | **VERIFIED NOVEL** |
| `spk_north_06` | `Minh Đức` | Nam (Male) | Bắc (Northern) | Tin tức (News) | None (Unused) | **VERIFIED NOVEL** |
| `spk_north_07` | `Xuân Vĩnh` | Nam (Male) | Bắc (Northern) | Tự nhiên (Natural) | None (Unused) | **VERIFIED NOVEL** |
| `spk_north_08` | `Thiền Tâm Đức`| Nam (Male) | Bắc (Northern) | Kể chuyện (Narrative) | None (Unused) | **VERIFIED NOVEL** |

*(Alternate spare: `Adam bựa` — Nam · Bắc · Tự nhiên)*

---

## 3. Confirmatory Dataset Architecture ($N = 1,000$ Clips)

The confirmatory set comprises exactly 1,000 audio clips partitioned into 5 balanced categories:

```
Total Clips: 1,000
├── Target Commands: 200 clips (20.0%)
│   ├── TRAI: 50 clips
│   ├── PHAI: 50 clips
│   ├── LEN: 50 clips
│   └── XUONG: 50 clips
└── Non-Target Rejections: 800 clips (80.0%)
    ├── Phonetic Minimal Pairs: 200 clips (50 per keyword direction)
    ├── Syntactic Carrier Collisions: 200 clips (50 per keyword direction)
    ├── General Conversational Non-Targets: 300 clips (diverse vocabulary)
    └── Acoustic Background / Silence: 100 clips (ambient, noise, silence)
```

### 3.1 Category Specifications
1. **Target Commands ($N = 200$):**  
   Isolated utterances of "trái", "phải", "lên", "xuống" (50 each), distributed evenly across the 8 novel speakers (6–7 clips per speaker per keyword).
2. **Phonetic Minimal Pairs ($N = 200$):**  
   Phonetic foils testing acoustic discrimination:
   - *TRAI foils ($N = 50$):* tonal variants (*trại, trải, trãi, trài*), consonant foils (*chái, tro, trôi, trán*), vowel foils (*tra, trơ, trâu*).
   - *PHAI foils ($N = 50$):* tonal variants (*phái, phại, phài*), consonant foils (*hoài, phèn, phan, pháo, phán*), vowel foils (*phi, pho, phơ*).
   - *LEN foils ($N = 50$):* tonal variants (*lèn, lén, lẽn, lẻn*), consonant foils (*sen, men, rên, sên, bện*), coda foils (*lang, lưng, luồn*).
   - *XUONG foils ($N = 50$):* tonal variants (*xuông, xướng, xưởng*), consonant foils (*chuông, sương, xông, xuân*), coda foils (*xuôi, xua*).
3. **Syntactic Carrier Collisions ($N = 200$):**  
   Longer carrier sentences embedding target words, verifying rejection of embedded commands in compound speech:
   - 50 sentences embedding *trái* (e.g. "chúng ta chuẩn bị rẽ trái ở ngã ba phía trước").
   - 50 sentences embedding *phải* (e.g. "bạn phải quan sát gương chiếu hậu thật kỹ").
   - 50 sentences embedding *lên* (e.g. "chiếc xe tăng tốc leo lên đỉnh đèo dốc").
   - 50 sentences embedding *xuống* (e.g. "mực nước sông đang rút xuống rất nhanh chóng").
4. **General Conversational Non-Targets ($N = 300$):**  
   Diverse Vietnamese natural sentences covering science, daily life, weather, geography, and education, with 0% keyword occurrence.
5. **Acoustic Background & Silence ($N = 100$):**  
   - 25 Digital Silence (exact zero vectors)
   - 25 Office Ambient Noise (low-frequency fans, subtle room tone, peak $< 0.01$)
   - 25 Faint Gaussian Room Tone (peak $< 0.005$)
   - 25 Transient Impulses / Acoustic Textures (clatter, paper rustle, keyboard clicks)

---

## 4. Lexical Independence Contract

- **Lexical Overlap Constraint:** Exactly 0% overlap with Phase 11 Train, Validation, and Test manifests, as well as the Phase 90 Review manifest.
- **Verification Method:** Hash-set intersection check executed programmatically in `lexicon/verify_lexical_independence.py`.

---

## 5. Preprocessing Contract (Remediated Phase 13.1 Specification)

The evaluation MUST execute using the remediated Phase 13.1 reference pipeline:
1. **Target Audio Format:** 16,000 Hz, 1 channel (mono), 32-bit float PCM.
2. **Padding / Truncation:** Pad right with zeros if shorter than 16,000 samples; truncate to 16,000 samples if longer.
3. **Peak Normalization (Clamping):**
   $$\text{if } \max(|x|) > 1.0 \implies x \leftarrow \frac{x}{\max(|x|)}; \quad \text{else } x \text{ is unchanged.}$$
   *Floor amplification is strictly prohibited.*
4. **Log-Mel Frontend:** Slaney Mel scale, 40 Mel bins, 16 kHz, $N_{\text{fft}} = 512$, $W_{\text{len}} = 400$, $H_{\text{len}} = 160$, center reflection padding, $\epsilon = 1.0 \times 10^{-6}$.
5. **Output Shape:** `[1, 1, 40, 101]`.

---

## 6. Preregistered Metrics and Acceptance Criteria

The confirmatory test evaluation will be scored exactly once on the locked 1,000-clip dataset. Acceptance requires meeting all of the following criteria:

| Metric | Target Acceptance Threshold |
| :--- | :--- |
| **TRAI Recall** | $\ge 95.0\%$ ($\ge 48 / 50$ clips) |
| **PHAI Recall** | $\ge 95.0\%$ ($\ge 48 / 50$ clips) |
| **LEN Recall** | $\ge 95.0\%$ ($\ge 48 / 50$ clips) |
| **XUONG Recall** | $\ge 95.0\%$ ($\ge 48 / 50$ clips) |
| **Macro Command Recall** | $\ge 95.0\%$ ($\ge 190 / 200$ clips) |
| **Non-Target False Alarm Rate (FAR)** | $\le 5.0\%$ ($\le 40 / 800$ false accepts) |
| **Overall Five-Class Accuracy** | $\ge 95.0\%$ ($\ge 950 / 1000$ clips) |
| **Single-Clip CPU Latency** | Mean $\le 50.0$ ms (p95 $\le 50.0$ ms) |

---

## 7. Immutable Execution Rules

1. **One-Shot Evaluation:** Once generated and locked, the confirmatory set will be evaluated exactly once.
2. **No Post-Hoc Tuning:** No weights, thresholds ($\theta^* = 0.20$), hyperparameters, or preprocessing code may be adjusted after inspecting test results.
3. **Honest Failure Reporting:** If the candidate fails any acceptance criterion, the failure will be reported objectively, and the confirmatory set will be permanently marked as inspected diagnostic data.
