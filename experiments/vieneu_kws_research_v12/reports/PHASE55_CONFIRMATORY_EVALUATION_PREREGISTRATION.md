# PHASE 55: FUTURE CONFIRMATORY EVALUATION PROTOCOL & PRE-REGISTRATION

**Repository:** `D:\trainidentificationaudio`  
**Phase:** Phase 55 (Future Independent Confirmatory Protocol Registration)  
**Creation Timestamp:** `2026-10-25T14:55:00+00:00`  
**Registration Status:** **LOCKED & PRE-REGISTERED (PRIOR TO FUTURE CONFIRMATORY CORPUS ACQUISITION)**  
**Operating Threshold:** $\theta_{\text{cmd}} = 0.20$ (Strictly Locked)  
**Canonical Default Model:** Legacy $v11$ (`3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2`)  
**Target Models for Confirmation:** $v11$, $E1$, Phase 28 $D_2\_H1$, Phase 26 $D_0\_H1\_seed44$, and selected $D_3$ candidate

---

## Mandatory Governance Invariants

> [!CAUTION] MANDATORY RESEARCH STATUSES
> 1. `EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`
> 2. `NOT_VALIDATED_ON_NATURAL_SPEECH`
> 3. `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`
> 4. `NOT_VALIDATED_PENDING_DATA`
>
> Zero human speech was collected. Zero physical microphone testing was conducted. No model deployed. Checkpoints for $v11$, $E1$, Phase 28 $D_2\_H1$, and Phase 26 $D_0\_H1\_seed44$ remain strictly frozen and read-only.
> Historical evaluation corpora (Phase 21, Phase 27, Phase 28-31 `edge_tts_eval_v2`, Phase 43 `mms_tts`, Phase 51 `gTTS`) are permanently designated as observed datasets and are **STRICTLY EXCLUDED from serving as confirmatory confirmation data for D3**.

---

## 1. Objective & Generalization Doctrine

To scientifically validate whether the newly developed $D_3$ candidate resolves the out-of-domain sensitivity collapse while preserving false-alarm immunity, a future confirmatory evaluation must confront an independent, unobserved acoustic speech distribution.

### The Five-Generator Independence Rule:
To prevent circular evaluation and memorization, the confirmatory evaluation must employ a speech engine that has never been used in:
1. Model training ($v11$, $E1$, $D_0$, $D_2$, $D_3$) -> **VieNeu-TTS & Microsoft Edge-TTS EXCLUDED**.
2. Validation split -> **VieNeu-TTS EXCLUDED**.
3. Phase 43 Evaluation -> **Meta MMS-TTS EXCLUDED**.
4. Phase 51 Evaluation -> **Google Translate TTS EXCLUDED**.

Any future confirmation must utilize an independently authorized fifth engine (e.g., Amazon Polly Neural Vietnamese, OpenAI TTS, or commercial FPT.AI Speech). If no independent engine is available or authorized, the project must record `BLOCKED_CONFIRMATORY_SOURCE` rather than recycling Phase 51 data as a "blind" test.

---

## 2. Permitted Future Data Architecture

### 2.1 Environmental Noise Provenance:
- **Dataset:** Karol Piczak ESC-50 Fold 3 or Fold 4 indoor tracks.
- **Categories:** `vacuum_cleaner`, `keyboard_typing`, `washing_machine`, `clock_tick`.
- **Deduplication:** Full cross-correlation separation ($r_{\max} < 0.15$) against Fold 1, Fold 2, and all 40 historical `mic-noise` tracks.

### 2.2 Two Disaggregated Evaluation Tracks:

#### Track A: Isolated 4-Command Classification ($N = 360$ clips)
- **Target Commands ($N = 40$ clips):** 10 independent synthesis instances (5 speaking rates $\times$ 2 voice timbres) $\times$ 4 SNR tiers (`clean`, $15\text{ dB}$, $10\text{ dB}$, $5\text{ dB}$).
  - Enforces $N_{\text{inst}} = 10$ per command class, eliminating the $N=1$ base-instance bottleneck.
- **Phonetic Hard Negatives ($N = 160$ clips):** 40 clean instances across the 16 core confusers $\times$ 4 SNR tiers.
- **General Conversational OOV ($N = 100$ clips):** 25 clean instances across conversational vocabulary $\times$ 4 SNR tiers.
- **Diagnostic Tone Probes ($N = 60$ clips):** 15 clean instances of tonal syllables (`ma`, `má`, `mà`, `mả`, `mã`, `mạ`) $\times$ 4 SNR tiers.

#### Track B: Continuous Contextual Keyword Detection ($60$ Sequences, $250$ Ground-Truth Events)
- Continuous audio streams ($1.0\text{ s}$ buffer, $0.1\text{ s}$ stride, fixed VAD, $\theta_{\text{cmd}} = 0.20$).
- Contextual carrier phrases (`rẽ trái`, `sang trái`, `rẽ phải`, `sang phải`, `đi lên`, `tiến lên`, `đi xuống`, `lùi xuống`).
- Interspersed with pure ambient silence/noise sections ($\ge 2.0\text{ min}$) and pure conversational speech sections ($\ge 2.0\text{ min}$).
- Matching tolerance: $\pm 0.50\text{ s}$ window around annotated event center.

---

## 3. Sample-Size Rationale & Primary Statistical Estimands

### 3.1 Sample-Size Unit of Inference:
Past phases established that evaluating statistical significance on $N=480$ or $N=240$ acoustic clips overstates statistical power by a factor of 4 because the 4 SNR variants of each clip are correlated.
- In this future confirmatory protocol, the **independent synthesis instance ($N_{\text{inst}} = 90$)** is the primary unit of inference.
- Sample-size power calculation: With $N_{\text{inst}} = 90$ paired clusters, a two-sided paired $t$-test has $>90\%$ statistical power ($\beta < 0.10$) at $\alpha = 0.01$ to detect a true accuracy difference of $\Delta \ge 8.0\text{ pp}$ assuming cluster standard deviation $\sigma \le 20\text{ pp}$.

### 3.2 Primary Pre-Registered Estimands:
1. **Primary Paired Comparison:** Dependency-aware paired clustered $t$-test and Wilcoxon signed-rank test on the $N_{\text{inst}} = 90$ clean instance accuracy scores:
   $$s_i = \frac{1}{4} \sum_{j=1}^4 \mathbb{I}(\hat{y}_{i,j} = y_i)$$
2. **Tone-Probe Uncertainty Hierarchy:** For zero observed false alarms on tone probes, reports must report the exact Rule-of-Three bound at all three levels:
   - Clip Level ($N = 60$): $\text{UCB95}_{\text{clip}} = 1 - (0.05)^{1/60} \approx 4.87\%$
   - Synthesis Instance Level ($N = 15$): $\text{UCB95}_{\text{inst}} = 1 - (0.05)^{1/15} \approx 18.10\%$
   - Base Syllable Level ($N = 6$): $\text{UCB95}_{\text{lex}} = 1 - (0.05)^{1/6} \approx 39.30\%$

---

## 4. Pre-Registered Engineering Acceptance Criteria for D3 Confirmation

| Candidate | Target Evaluation | Specific Engineering Confirmation Targets |
|:----------|:------------------|:-------------------------------------------|
| **D3 Candidate** | Track A (Isolated) | 1. 4-Command Macro-F1 $\ge 60.0\%$ (vs $25.8\%$ for $D_2\_H1$).<br>2. Minimum per-class recall $\min_c R_c \ge 40.0\%$ (no command collapsing).<br>3. Overall OOV FAR $\le 20.0\%$.<br>4. Diagnostic Tone Probe FAR $\le 5.0\%$. |
| | Track B (Streaming) | 1. Command Event F1 $\ge 50.0\%$.<br>2. Silence False Alarm Rate $\le 1.0\text{ FA/min}$ ($\text{UCB95} \le 2.0$).<br>3. Conversational OOV False Alarm Rate $\le 5.0\text{ FA/min}$.<br>4. Exact sequence accuracy $\ge 30.0\%$. |

---

## 5. Execution Preconditions

Prior to running forward inference on any future confirmatory corpus:
1. The fifth generator must be piloted and verified in a dedicated readiness phase.
2. The complete audio files, manifests, event ledgers, and SHA-256 manifests must be generated and locked.
3. No hyperparameter tuning, threshold alteration, or cherry-picking of test instances is permitted.
4. Legacy $v11$ remains the canonical default model until a confirmed natural-speech human evaluation is authorized.
