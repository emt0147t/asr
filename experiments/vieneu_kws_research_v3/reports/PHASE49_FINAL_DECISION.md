# Phase 49: Final Research Decision and Campaign Synthesis

Workspace: `D:\trainidentificationaudio`  
Experiment Directory: `experiments/vieneu_kws_research_v3/`  
Date: 2026-10-18  
Status: `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Official Architectural Verdict

### Final Decision: **`NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`**

Based on rigorous, reproducible evaluation across random seeds (42, 43, 44), 25 disjoint speaker voices, 210 disjoint non-target text prompts, and **10,800.0 seconds (3.000 hours)** of locked continuous non-target streaming challenge audio:
- **No tested system meets the joint operational criteria** ($\text{Recall} \ge 95\%$, $\text{Non-Target FA} \le 1.0\text{ event/hr}$, $\text{Mean Latency} \le 600\text{ ms}$).
- Validation calibration objectively recorded **`NO_FEASIBLE_VALIDATION_OPERATING_POINT`** across all evaluated architectures and thresholds.

---

## 2. Evidence-Based Research Findings

### 1. Summary of Candidate Performance on Locked 3-Hour Benchmark

| Candidate Architecture | Target Command Recall | Mean Event Latency | Wall-Clock FA / hr (3h Test) | Active-Speech FA / hr | Primary Operational Criteria Satisfied? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Candidate 1: Historical Cascade v1** | **$97.29\% \pm 1.34\%$** | $680.6 \pm 43.6\text{ ms}$ | $77.33 \pm 7.22$ | $869.89 \pm 81.18$ | **Failed** ($\text{FA/hr} > 1.0$, Lat $> 600$) |
| **Candidate 2: Stage 1 alone (v3)** | **$98.13\% \pm 1.25\%$** | **$546.4 \pm 11.2\text{ ms}$** | $202.56 \pm 29.89$ | $2278.47 \pm 336.26$ | **Failed** ($\text{FA/hr} > 1.0$) |
| **Candidate 3: Recall-Preserving Cascade** | $74.17\% \pm 16.05\%$ | $686.4 \pm 32.7\text{ ms}$ | **$10.33 \pm 6.56$** | **$116.24 \pm 73.80$** | **Failed** (Recall & Latency) |

### 2. Generalization to Unseen Synthetic Voices and Texts
- **Non-Target Generalization:** Candidate 3 demonstrated remarkable generalization on unseen non-target texts and unseen voices:
  - Over 3 hours of continuous speech from 4 held-out test voices (`Kim Thanh`, `Quốc Tuấn`, `Quỳnh Anh`, `Đức Trí`), total false alarms dropped from 692 (Historical Cascade) to **93** (Candidate 3)—an **$86.6\%$ reduction in false activations**.
  - On Seed 42, the system sustained only 13 false activations across 3 hours ($4.33\text{ FA/hr}$).
- **The Rhyme-Level Acoustic Overlap Bottleneck:**
  - While `LEN` ($95.0\%$) and `XUONG` ($84.2\%$) generalized well, commands with open `/aːj/` rhymes (`TRAI` at $58.3\%$ and `PHAI` at $59.2\%$) suffered catastrophic false rejections.
  - In monosyllabic Vietnamese, distinguishing *trái* from *chái*, *sái*, *lái*, *bái*, *mài* or *phải* from *thải*, *thái*, *khải*, *mải*, *bải* relies almost entirely on the initial $30\text{–}60\text{ ms}$ consonant transition. On 40-channel log-Mel spectrograms, energetic speech variations cause the Stage 2 verifier to reject valid commands when trained aggressively against near-homophones.

---

## 3. Recommended Technical Roadmap

The results of Phases 45–49 prove that the speech-active false alarm bottleneck **cannot be bridged solely by discriminative fine-tuning of compact 2D-CNNs or 1D-TCNs on 40-mel filterbanks**. 

To reach $\le 1.0\text{ FA/hr}$ while guaranteeing $\ge 95\%$ target command recall:

1. **Transition from Filterbanks to Pretrained Self-Supervised Speech Representations:**
   - Small CNNs (138k–340k params) operating directly on Mel filterbanks lack the sub-phonetic resolution needed to distinguish monosyllabic minimal pairs in continuous streaming audio.
   - A distilled self-supervised Vietnamese speech foundation model (e.g. Wav2Vec2-Vietnamese, HuBERT-compact, or Conformer-CTC) provides deep contextual phonetic representations that cleanly separate minimal pair consonants and rhymes.
2. **Posterior Decoding with Duration and Transition Penalties:**
   - Replace simple sliding-window thresholding and binary consecutive-frame counting with an explicit Hidden Markov Model (HMM) or dynamic time warping (DTW) posterior decoder.
   - Modeling state durations and transition probabilities across phonetic sub-units (e.g., onset $\rightarrow$ nucleus $\rightarrow$ coda) provides strong mathematical immunity against transient energy spikes in non-target speech.
3. **Formal Natural Speech Data Governance:**
   - Complete the human listening evaluation on the 80 synthetic review clips.
   - Move beyond synthetic TTS audio to natural spoken Vietnamese recordings collected under strict informed consent protocols.

---

## 4. Status Markers and Governance Invariants

> [!IMPORTANT]
> The following status markers remain authoritative and binding across this codebase:
> - **`PENDING_HUMAN_LISTENER_VERIFICATION`**: The 80-clip human listening evaluation remains pending human review.
> - **`NOT_VALIDATED_ON_NATURAL_SPEECH`**: All training and test datasets in Phases 26–49 utilize synthetic VieNeu-TTS audio. No claim of natural speech readiness is permitted.
> - **Zero File Modifications to Historical Baselines:** Historical experiments (`synthetic_kws_dataset_vieneu_1000/`, `6class/`, `6class_v2/`, `vieneu_cnn_baseline/`, `vieneu_cnn_6class/`, `vieneu_kws_v2/`, `vieneu_kws_cascade_v1/`, `vieneu_kws_vad_ablation_v1/`, `vieneu_kws_research_v2/`) remain completely unmodified.
> - **Field Recording Isolation:** Directory `data/raw/field_recordings/` and consent ledgers were neither touched nor used in training.
> - **Git Discipline:** `git diff --cached` remains strictly empty (zero staged files).
