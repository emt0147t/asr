# Phase 12.2: Synthetic Dataset V5 Design Specification

> [!IMPORTANT]
> **GOVERNANCE INVARIANT: HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED.**
> This document is a **design specification only**. No audio generation, model training, fine-tuning, or human data processing is authorized or performed in this phase. All specifications apply strictly to a future, separately versioned synthetic-only candidate dataset (`synthetic_kws_v5_candidate`).

---

## 1. Executive Summary & Design Scope

Audits in Phase 12 and 12.1 revealed critical scientific flaws in the synthetic V4 dataset:
1. **Shortcut Learning**: The classifier learned an artificial 25ms silence closure gap for `TRAI` rather than speech phonology, causing $0.0\%$ recall on earlier synthetic versions lacking the gap.
2. **Template Matching**: Intra-class Log-Mel feature cosine similarity averaged $0.966–0.986$ across all splits due to static formant tracks and deterministic harmonics.
3. **Dead-Code Dialect**: The `dialect` parameter was never referenced in the generator code.
4. **Silence Classifier Failure**: The neural network misclassified pure zeros as `TRAI` ($30.23\%$) and relied on the action adapter threshold to suppress actions.
5. **Vocabulary Deficit**: `UNKNOWN` was restricted to 8 monosyllabic words.

The **Synthetic Dataset V5 Design Specification** establishes an acoustically grounded, mathematically diversified, and strictly partitioned framework to replace V4 without requiring human data collection.

---

## 2. Core Acoustic Modeling Specifications

### 2.1. Eliminating the Onset Shortcut & Modeling Coarticulation
To prevent the neural network from exploiting fixed temporal silence gaps or static noise bursts:

1. **Continuous VOT and Closure Modeling**:
   - **No Fixed Silence Gap**: Prohibit any hard zero-clamp (`output[:400] *= 0.05`).
   - For `trái` (voiceless stop/affricate $/t\text{ɕ}/$):
     - Closure duration must vary continuously: $T_{\text{closure}} \sim \mathcal{U}(12, 35)\text{ ms}$.
     - Ambient noise and low-level glottal leakage must remain present during the closure ($15\text{ to }25\text{ dB}$ below peak, not absolute digital silence).
     - Release burst duration must vary: $T_{\text{burst}} \sim \mathcal{U}(8, 22)\text{ ms}$.
     - Burst frequency spectrum must be shaped by a variable 2nd-order resonator with center frequency $f_{\text{burst}} \sim \mathcal{U}(3200, 4600)\text{ Hz}$ and bandwidth $B_{\text{burst}} \sim \mathcal{U}(600, 1200)\text{ Hz}$.
2. **Frication Modeling for `phải`**:
   - For `phải` (voiceless labiodental fricative $/f/$):
     - Frication duration must vary: $T_{\text{fric}} \sim \mathcal{U}(40, 85)\text{ ms}$.
     - Frication attack envelope must vary: rise time $T_{\text{rise}} \sim \mathcal{U}(15, 40)\text{ ms}$.
     - Spectral band must vary: lower cutoff $f_{\text{low}} \sim \mathcal{U}(1500, 2200)\text{ Hz}$, upper cutoff $f_{\text{high}} \sim \mathcal{U}(5500, 7000)\text{ Hz}$.
3. **Locus Equation Principle (Coarticulatory Formant Transitions)**:
   - In real speech, consonants are identified primarily by how formants transition into the following vowel:
     - For $/t\text{ɕ}/$ in `trái`: $F_2$ onset locus starts high, near $1700 - 1900\text{ Hz}$, before gliding into vowel /a:j/.
     - For $/f/$ in `phải`: $F_2$ onset locus starts low, near $1000 - 1200\text{ Hz}$, reflecting labial constriction.
   - Implementing consonant-dependent $F_2$ starting loci provides legitimate phonological contrast in the time-frequency plane, eliminating the need for artificial closure silences.

---

### 2.2. Generator Acoustic Diversity & Variance

To prevent the model from memorizing static spectral templates:

| Acoustic Parameter | V4 Baseline (Defective) | V5 Candidate Specification | Target Distribution / Range |
| :--- | :--- | :--- | :--- |
| **Vocal Tract Length (VTL)** | Fixed ($100\%$ constant) | Parameterized scaling factor $\alpha_{\text{vtl}}$ applied to all formant frequencies ($F_i \leftarrow \alpha_{\text{vtl}} \cdot F_i$) | $\alpha_{\text{vtl}} \sim \mathcal{U}(0.85, 1.15)$ per simulated speaker |
| **Formant Bandwidths** | Fixed ($Q_1=5, Q_2=7, Q_3=9$) | Independent random perturbation of $B_1, B_2, B_3$ | $B_i \sim \mathcal{N}(\mu_{B_i}, 0.2\mu_{B_i})$ |
| **Glottal Source Model** | 4 static sines ($1, 0.5, 0.25, 0.12$) | Rosenberg polynomial glottal flow pulse with open quotient $O_q$ and spectral tilt $\tau$ | $O_q \sim \mathcal{U}(0.40, 0.70)$, $\tau \sim \mathcal{U}(-6, -14)\text{ dB/oct}$ |
| **Pitch ($F_0$) Contour** | Fixed formula per class | Base pitch + micro-intonation + natural jitter | $F_{0,\text{base}} \sim \mathcal{U}(95, 250)\text{ Hz}$, Jitter $\sim \mathcal{U}(0.4, 1.2)\%$ |
| **Utterance Duration** | Narrow ($0.35 - 0.65\text{ s}$) | Broad temporal range | $T_{\text{dur}} \sim \mathcal{U}(0.28, 0.75)\text{ s}$ |
| **Temporal Placement** | Left-aligned ($t=0$ or sample 0) | Uniform random onset offset within 1.0s window | $t_{\text{onset}} \sim \mathcal{U}(0.02, 0.35)\text{ s}$ |
| **Amplitude Envelope** | Fixed attack/decay | Independent attack, sustain, and release curves | $T_{\text{decay}} \sim \mathcal{U}(0.08, 0.20)\text{ s}$ |
| **Acoustic Background** | Clean or static dither | Additive synthetic room noise, hum, and dither | $\text{SNR} \sim \mathcal{U}(12, 35)\text{ dB}$ |

---

### 2.3. Dialect Modeling Policy
- **Strict Prohibition**: Never assign different dialect labels (`"Northern"`, `"Southern"`, `"Central"`) to identical or near-identical synthesis outputs.
- **Specification**:
  - If distinct dialect variants are implemented:
    - **Northern**: Affricate $/t\text{ɕ}/$ with high-frequency burst locus ($3800 - 4500\text{ Hz}$) and standard tone contours.
    - **Southern**: Retroflex stop $/ʈ/$ with lower-frequency burst locus ($2400 - 3200\text{ Hz}$) and merged hỏi/ngã dipping tone.
  - If dialectal phonetic variants are not independently calibrated and verified, the generator must drop dialect claims entirely and catalog all samples under a single honest label: `"synthetic_parametric_standard"`.

---

### 2.4. Negative Class (`UNKNOWN`) Expansion Strategy
To provide realistic open-set rejection without using unapproved external speech:

1. **Algorithmic Vocabulary Expansion**:
   - Expand `UNKNOWN` from 8 words to a structured dictionary of **at least 200 distinct Vietnamese syllables** generated parametrically.
2. **Confuser Distribution**:
   - **Initial-Sharing Confusers ($30\%$)**: Syllables sharing initial consonants with target words (e.g. `trời`, `tròn`, `pha`, `phố`, `lớp`, `lúa`, `xinh`, `xưa`).
   - **Rhyme-Sharing Confusers ($30\%$)**: Syllables sharing rhymes with target words (e.g. `tai`, `cái`, `bài`, `mái`, `sen`, `đèn`, `buông`, `cuống`).
   - **General Open Vocabulary ($40\%$)**: Broad sample of common Vietnamese phonemes covering all monophthongs, diphthongs, and 6 lexical tones.

---

### 2.5. Ambient Class (`SILENCE`) & Edge Case Coverage
`SILENCE` in V5 must be partitioned into 5 explicit acoustic regimes:

1. **Digital Zeros ($20\%$)**: Exact zero arrays ($a[n] = 0.0$).
2. **Low-Level Digital Dither ($20\%$)**: Uniform and Gaussian noise at $-70\text{ dBFS}$ to $-50\text{ dBFS}$.
3. **Synthetic Room Acoustic Floor ($30\%$)**: Lowpass and bandpass filtered noise modeling HVAC, room resonance, and room tone ($-50\text{ to } -35\text{ dBFS}$).
4. **Mains Electrical Hum ($15\%$)**: $50\text{ Hz}$ or $60\text{ Hz}$ fundamental tone with odd harmonics ($150\text{ Hz}, 250\text{ Hz}$).
5. **Transient Non-Speech Impulses ($15\%$)**: Isolated clicks, pops, and short-duration taps ($<10\text{ ms}$) without vocal resonance.

---

## 3. Strict Split Independence Protocol

To guarantee mathematical and structural independence across dataset splits:

| Split Dimension | Training Partition (`v5_train`) | Development Partition (`v5_dev`) | Frozen Candidate Holdout (`v5_holdout`) |
| :--- | :--- | :--- | :--- |
| **Random Seed Block** | `seeds 10,000 – 19,999` | `seeds 20,000 – 24,999` | `seeds 30,000 – 34,999` |
| **Simulated Speaker Pool** | 30 synthetic speakers (`SPK_SYNTH_001..030`) | 10 synthetic speakers (`SPK_SYNTH_031..040`) | 15 synthetic speakers (`SPK_SYNTH_041..055`) |
| **VTL Scale Factors ($\alpha_{\text{vtl}}$)** | Clustered in sub-bands (e.g., $0.85-0.95, 1.05-1.15$) | Disjoint sub-bands | Stratified across entire range ($0.85-1.15$) |
| **`UNKNOWN` Vocabulary** | 120 distinct words | 40 disjoint words | 60 disjoint words (zero lexical overlap) |
| **Background Noise Seeds** | Disjoint noise generators | Disjoint noise generators | Disjoint noise generators |

---

## 4. Manifest Schema & Traceability

Every row in the V5 manifest must record full algorithmic provenance:

```csv
recording_id,split,class_id,class_label,transcript,generator_version,generator_recipe,seed,simulated_speaker_id,vtl_scale,f0_base,duration_sec,onset_offset_sec,snr_db,noise_type,sha256,processed_path,status
```

### Mandatory Rules:
1. `generator_recipe` must declare the exact parameter preset used (e.g., `parametric_coarticulated_v5`).
2. `seed` must allow bit-exact single-clip reproduction.
3. Provenance must state `synthetic_parametric_v5`.
4. No metadata field may contain placeholder or non-functional values.

---

## 5. Measurable Acceptance Quality Gates

Before any dataset generated under this specification can be approved for training:

- **Gate 1 (Acoustic Variance Acceptance)**:
  - Within-class Log-Mel cosine similarity across distinct simulated speakers must have standard deviation $\sigma \ge 0.04$ and mean similarity $\le 0.90$.
- **Gate 2 (Onset Shift Robustness)**:
  - Energy onset time must span $[20\text{ ms}, 350\text{ ms}]$ with uniform distribution. Fixed-onset clustering is rejected.
- **Gate 3 (Zero Closure Verification)**:
  - For `TRAI`, maximum absolute amplitude in the first 25ms of the utterance body must exceed $0.02$ in at least $95\%$ of active speech clips. Near-zero clamps are rejected.
- **Gate 4 (Silence Neural Logit Gate)**:
  - When evaluated on pure zeros, low-level dither, and mains hum, the baseline classifier must assign $P(\text{SILENCE}) \ge 0.95$ without relying on adapter threshold suppression.
- **Gate 5 (Lexical Disjointness)**:
  - Exact word overlap between `UNKNOWN` in training and `UNKNOWN` in holdout must be **strictly 0**.
