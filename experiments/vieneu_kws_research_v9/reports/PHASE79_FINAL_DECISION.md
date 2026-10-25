# Phase 79 — Final Evidence Review and Operational Decision Report

**Workspace:** `D:\trainidentificationaudio`  
**Experiment Directory:** `experiments/vieneu_kws_research_v9/`  
**Candidate Identifier:** `variant_w10` (Frozen Compact CNN, $w_{\text{hn}}=1.0$)  
**Status Banners:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Executive Summary & Dual Operational Verdicts

Phase 75–79 completed the formal candidate freeze, independent metric reproduction audit, double-blind human listening package preparation, 13-condition synthetic robustness evaluation, and operational readiness assessment for the Vietnamese Keyword Spotting project.

### Formal Tripartite Operational Verdicts

| Operational Paradigm | Official Final Verdict | Candidate Architecture | Key Evidence & Metric Summary | Operational & Deployment Scope |
| :--- | :--- | :--- | :--- | :--- |
| **A. Synthetic Bounded-Utterance Benchmark** | **`FEASIBLE_BOUNDED_CANDIDATE_ESTABLISHED`** | **`variant_w10`** (Frozen Compact CNN, 138,469 params) | • Macro Command Recall: **99.79% $\pm$ 0.36%**<br>• Per-keyword: `TRAI` 100%, `PHAI` 99.2%, `LEN` 100%, `XUONG` 100%<br>• Non-Target FAR: **2.03% $\pm$ 0.58%** (11.3/560 FAs)<br>• 5-Class Accuracy: **98.38% $\pm$ 0.40%**<br>• CPU Decision Latency: **1.73 $\pm$ 0.03 ms** | **Qualified Synthetic Research Prototype:** Simultaneously satisfies all strict predeclared requirements ($\ge 95\%$ recall per keyword, $\le 5\%$ FAR, $\le 50$ ms latency). Qualified for human-in-the-loop pilot review. |
| **B. Controlled Acoustic Feasibility** | **`NOT_RUN`** | None | • Hardware Blocker: No physical loudspeaker or microphone hardware connected to the automated runtime.<br>• Non-Fabrication Rule: No physical audio captures simulated or synthesized. Complete protocol and template schema frozen in `acoustic_loopback/`. | **BLOCKED (Requires Laboratory Audio Setup):** Must be executed with physical speaker/mic hardware before physical device claims can be made. |
| **C. Continuous Open-Mic Operation** | **`NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`** | Historical Two-Stage Cascade v1 | • Clean Stream Event Recall: $\sim 96.25\%$<br>• Stream False Alarms: **$\sim 69.2$ events/hour** on conversational audio<br>• Mean Latency: $\sim 480$ ms | **STRICTLY BLOCKED:** Unacceptable false activation frequency (> 60× above the operational budget of $\le 1.0$ FA/hour). Performance in bounded push-to-talk mode must **never** be cited to claim open-mic readiness. |

---

## 2. Synthesis of Research Evidence Across Phases 75–78

### 1. Reproducibility & Candidate Freeze (Phase 75)
- The model architecture `CorrectedVietnameseKWSNet` (138,469 parameters) and checkpoints for seeds 42, 43, 44 were cryptographically frozen in `artifacts/checkpoints/`.
- Independent audit of 2,160 forward-pass predictions confirmed **100% bit-exact reproduction** against Phase 74 locked test outputs.
- Sampling unit audit formally noted that evaluations on 720 clips across 3 seeds are repeated clustered measurements. Uncertainty is reported via exact individual Wilson CIs and cross-seed sample standard deviations.

### 2. Blinded Human Perceptual Package (Phase 76)
- Prepared a stratified 120-clip perceptual review package drawn from the validation split (40 target commands across 4 voices, 35 confusable minimal pairs, 30 general phrase utterances, 15 background noises).
- Randomized audio files (`review_clip_001.wav` to `review_clip_120.wav`) and reviewer forms (`reviewer_form.csv`) strictly sanitize and conceal ground-truth labels, transcripts, confusion families, and model predictions.
- **Human Verification Status:** Exactly 0 human submissions are on file. In accordance with anti-fabrication guidelines, all reviewer fields remain blank, and the status remains strictly `PENDING_HUMAN_LISTENER_VERIFICATION`.

### 3. Controlled Synthetic Robustness Profile (Phase 77)
- Evaluated 13 controlled acoustic conditions across all 3 seeds (14,040 inference passes):
  - **Dynamics & Clipping:** Invariant under gain scaling ($-12\text{ dB}$ to $+6\text{ dB}$) and mild clipping ($|x| > 0.50$).
  - **Acoustic Environment:** Maintains $\ge 94.5\%$ recall under meeting room reverberation ($T_{60} = 0.20\text{ s}$).
  - **Speech Dynamics:** Preserves $\ge 93.7\%$ recall under $\pm 10\%$ speed variations and $\pm 150\text{ ms}$ boundary padding shifts.
  - **Noise Degradation:** Graceful failure mode—at high broadband noise (0 dB SNR), recall drops to $48.8\%$, while false alarms are suppressed to $0.67\%$, safely failing toward `REJECT` rather than hallucinating commands.

### 4. Acoustic Feasibility Status (Phase 78)
- Formally marked `NOT_RUN` due to lack of physical audio I/O peripherals. Zero synthetic audio was disguised or fabricated as hardware capture.

---

## 3. Explicit Scientific Boundaries & Deployment Disclaimers

> [!CAUTION]
> 1. **Synthetic Audio Boundary:** All models were trained and benchmarked strictly on synthetic audio from VieNeu-TTS v3turbo. Performance on natural human speech remains unproven (`NOT_VALIDATED_ON_NATURAL_SPEECH`).
> 2. **Fixed Voice Inventory:** Exactly 25 voices exist in the approved TTS inventory. The 4 test voices were held out from v8/v9 training splits, but appeared in historical campaigns. Generalization is established for **unseen text prompts on a fixed synthetic voice inventory**, not arbitrary unencountered human speakers.
> 3. **Non-Equivalence of Operating Modes:** Bounded push-to-talk success does **not** transfer to continuous unsegmented open-mic streaming.

---

## 4. Final Research Sign-Off

The Vietnamese Keyword Spotting candidate `variant_w10` represents a reproducible, robust, compact (138k parameter), ultra-low latency (1.7 ms CPU) synthetic bounded-utterance prototype. Production deployment, continuous open-mic usage, and natural speech claims remain firmly gated pending genuine human listener verification and physical acoustic field validation.
