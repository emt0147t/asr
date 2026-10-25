# Phase 13.1: Silence Label Policy & Semantic Taxonomy Specification

> [!IMPORTANT]
> **GOVERNANCE INVARIANT: HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED.**
> This document specifies an acoustic label contract and semantic policy for non-target and ambient audio.
> Existing V5 pilot samples are **not** relabeled in-flight. This policy governs metadata annotations and future dataset versions (V5.1/V6).

---

## 1. Executive Summary & Semantic Conflict

In Keyword Spotting (KWS), the term **`SILENCE`** is an industry-standard shorthand inherited from legacy datasets (such as Google Speech Commands), but it contains a fundamental semantic contradiction:

1. **Literal Silence**: Complete absence of acoustic energy ($a[n] = 0.0$ or digital dither at $< -70\text{ dBFS}$).
2. **Ambient Non-Speech Background**: Presence of physical acoustic energy (HVAC air rush, room acoustic resonance, electrical hum, mic chassis bumps, keyboard taps) without human vocal tract excitation.
3. **Non-Target Speech (`UNKNOWN`)**: Presence of human vocal tract acoustic energy that does not articulate any of the defined target keywords (`TRAI`, `PHAI`, `LEN`, `XUONG`).

In the V5 pilot dataset, five distinct acoustic regimes were implemented under the single class label `SILENCE`. While acoustically diverse, conflating electrical hum or mechanical taps under the name `SILENCE` creates developer ambiguity, false-alarm confusion in edge applications, and evaluation opacity.

---

## 2. Formal Acoustic Taxonomy: The Five Pilot Regimes

The existing 20 V5 pilot clips for `SILENCE` are partitioned into five explicit regimes (4 clips each):

| Pilot Regime | Physical Acoustic Source | Energy Range (dBFS) | Spectral Character | Vocal Tract Presence | Current Label | Proposed Semantic Tag |
| :--- | :--- | :---: | :--- | :---: | :---: | :--- |
| **`exact_zeros`** | Digital muting, buffer zero-fill | $-\infty\text{ dBFS}$ (0.0) | None (all bins zero) | **None** | `SILENCE` | `silence_digital_zero` |
| **`low_level_dither`** | ADC quantization noise floor | $-68\text{ to } -55\text{ dBFS}$ | Flat white Gaussian spectrum | **None** | `SILENCE` | `silence_dither` |
| **`room_tone`** | Air conditioning, quiet room resonance | $-50\text{ to } -38\text{ dBFS}$ | Lowpass filtered ($f_c < 1200\text{ Hz}$) | **None** | `SILENCE` | `ambient_room_tone` |
| **`synthetic_hum`** | Mains power line electrical pickup | $-48\text{ to } -36\text{ dBFS}$ | Cyclic 50Hz/60Hz + 3rd/5th harmonics | **None** | `SILENCE` | `ambient_mains_hum` |
| **`non_speech_transient`** | Micro-clicks, taps, chassis impulses | Peaks $-26\text{ to } -17\text{ dBFS}$ | Broadband impulsive ($< 6\text{ ms}$) | **None** | `SILENCE` | `transient_non_speech` |

---

## 3. Boundary Definition: `SILENCE` vs. `UNKNOWN`

To maintain rigorous scientific validity, the boundary between `SILENCE` (ambient background) and `UNKNOWN` (non-target speech) must be strictly governed:

```
                            [ ALL INCOMING AUDIO ]
                                      |
                 +--------------------+--------------------+
                 |                                         |
       [ Human Vocalization ]                   [ Non-Vocal Acoustic ]
                 |                                         |
        +--------+--------+                       +--------+--------+
        |                 |                       |                 |
  [ Target Word ]   [ Non-Target ]          [ Ambient Floor ] [ Transients ]
  (TRAI, PHAI,      (UNKNOWN)               (Zeros, Dither,   (Clicks, Taps,
   LEN, XUONG)      (20 syllables)           Room Tone, Hum)   Bumps)
        |                 |                       |                 |
   Class 0..3          Class 4                             Class 5
   (Keywords)         (Negative)                         (SILENCE)
```

### Invariant Rules:
1. **Zero Vocalization in `SILENCE`**: No clip assigned to `SILENCE` may contain glottal pulse resonance, formants, vowel tracts, or human phonemes. Any signal containing human vocal tract excitation—even whispers, coughs, or laughter—must be cataloged under `UNKNOWN` or rejected.
2. **Literal vs. Ambient Separation**: Classifiers trained on pure zeros often produce undefined activations or false positives when deployed in rooms with electrical hum. The 5-regime partition must remain an invariant requirement for all future synthetic releases.
3. **`UNKNOWN` Vocabulary Scope**: Claims regarding `UNKNOWN` must be scientifically honest. The pilot contains exactly **20 distinct monosyllabic Vietnamese words** (e.g. `chào`, `bạn`, `núi`, `đèn`). It does not represent full open-vocabulary conversational Vietnamese.

---

## 4. Policy for Future Dataset Versions (V5.1 / V6)

To preserve backward compatibility with existing ONNX inference engines (which expect 6 output logits) while resolving semantic ambiguity:

### 4.1. Two-Tier Taxonomy Architecture
- **Tier 1 (Inference Class ID)**: Retain the canonical 6-class output contract:
  - `0: TRAI`
  - `1: PHAI`
  - `2: LEN`
  - `3: XUONG`
  - `4: UNKNOWN`
  - `5: SILENCE`
- **Tier 2 (Manifest Metadata Subclass)**: Require explicit `acoustic_subclass` serialization in the manifest:
  - For Class 5 (`SILENCE`):
    - `ambient_digital_zero`
    - `ambient_dither`
    - `ambient_room_tone`
    - `ambient_mains_hum`
    - `transient_non_speech`
  - For Class 4 (`UNKNOWN`):
    - `confuser_initial_sharing` (e.g. `trời`, `pha`)
    - `confuser_rhyme_sharing` (e.g. `mái`, `tai`, `đèn`, `cuống`)
    - `open_vocabulary_general` (e.g. `chào`, `hoa`, `sông`)

### 4.2. Migration Mapping Table

| Current Manifest Row | Current `class_label` | Current `transcript` | Proposed Future `class_label` | Proposed Future `acoustic_subclass` |
| :--- | :--- | :--- | :--- | :--- |
| `v5_pilot_silence_001..004` | `SILENCE` | `__silence_exact_zeros__` | `SILENCE` | `ambient_digital_zero` |
| `v5_pilot_silence_005..008` | `SILENCE` | `__silence_low_level_dither__` | `SILENCE` | `ambient_dither` |
| `v5_pilot_silence_009..012` | `SILENCE` | `__silence_room_tone__` | `SILENCE` | `ambient_room_tone` |
| `v5_pilot_silence_013..016` | `SILENCE` | `__silence_synthetic_hum__` | `SILENCE` | `ambient_mains_hum` |
| `v5_pilot_silence_017..020` | `SILENCE` | `__silence_non_speech_transient__` | `SILENCE` | `transient_non_speech` |
| `v5_pilot_unknown_001..020` | `UNKNOWN` | (word transcript) | `UNKNOWN` | `confuser_*` or `open_vocab_*` |

---

## 5. Deployment Recommendation

When building client action adapters (e.g. Android JNI or embedded DSP):
1. **Never rely on `SILENCE` logit alone**: Use a lightweight energy-based Voice Activity Detector (VAD) prior to neural inference.
2. **Do not use adapter thresholds to hide classifier deficiencies**: If a classifier misclassifies mains hum as `TRAI`, retraining on balanced ambient regimes is mandatory rather than arbitrarily tuning software squelch thresholds.
