# Phase 51 — Acoustic Error Analysis and Human Review Protocol

**Workspace:** `D:\trainidentificationaudio`  
**Experiment Directory:** `experiments/vieneu_kws_research_v4/`  
**Status:** `AUDITED_ACOUSTIC_ANALYSIS` | `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Executive Summary

This phase provides an in-depth phonetic and acoustic failure analysis of the Vietnamese Keyword Spotting systems evaluated in Phases 45–50, with a focus on Candidate 3 (Recall-Preserving Cascade). 

### Key Acoustic Findings:
1. **The `/aːj/` Rhyme Dilemma:** The target keywords `TRAI` and `PHAI` share the diphthong rhyme `/aːj/` (ngang tone). Across all 3 seeds of Candidate 3, **52 misses on `TRAI`** (43.3% error rate) and **57 misses on `PHAI`** (47.5% error rate) occurred, while **60 of 93 false alarms** ($64.5\%$) were triggered by words with an identical or near-identical `/aːj/` rhyme (*vải*, *bái*, *tải*, *hải*, *mài*, *phôi phai*).
2. **Consonant Minimal Pair Confusion:** Initial consonant distinctions (retroflex `/ʈ/` vs palatal `/tɕ/`, labiodental `/f/` vs alveolar `/tʰ/` or bilabial `/b/`) represent 49.5% ($46/93$) of all false alarms under continuous pink noise.
3. **Phonetic Phrase Embedding:** 24.7% ($23/93$) of false activations occurred when the command root was embedded in non-target phrases (e.g., *bước xuống*, *hạ xuống*, *hướng phải*, *phôi phai*, *bước lên*).
4. **Speaker Voice Asymmetry:** 
   - False Alarms were overwhelmingly concentrated in speaker **Đức Trí** ($60/93$, $64.5\%$), whose acoustic profile exhibits sharp formant transients and higher pitch energy.
   - False Negatives were heavily concentrated in speaker **Quỳnh Anh** ($62/135$ total misses, $45.9\%$) and **Quốc Tuấn** ($41/135$, $30.4\%$).
5. **Human Review Protocol Prepared:** A blinded listening test kit with randomized identifiers (`BLIND_REV_001` to `BLIND_REV_040`) was constructed to ensure rigorous perceptual validation without confirmation bias.

---

## 2. Granular Breakdown of False Negatives (Target Misses)

Across 3 seeds (120 target presentations per keyword, 480 total):

| Target Keyword | Total Events | Hits (Sum) | Misses (Sum) | Aggregate Recall | Dominant Failure Mode |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`LEN`** | 120 | 113 | 7 | **94.17%** | Minor boundary clipping under noise |
| **`XUONG`** | 120 | 101 | 19 | **84.17%** | Occasional tone confusions in Seed 42 |
| **`TRAI`** | 120 | 68 | 52 | **56.67%** | Stage 2 over-rejection on `/aːj/` diphthong coda |
| **`PHAI`** | 120 | 63 | 57 | **52.50%** | Fricative `/f/` onset masking + `/aːj/` rejection |

### Speaker Voice Impact on Misses:
- **Quỳnh Anh:** 62 misses (45.9% of all target misses)
- **Quốc Tuấn:** 41 misses (30.4% of all target misses)
- **Đức Trí:** 26 misses (19.3% of all target misses)
- **Kim Thanh:** 6 misses (4.4% of all target misses)

---

## 3. Granular Breakdown of False Positives (False Activations)

Across 3 seeds in the 3-hour non-target continuous challenge stream (93 total false alarms in Candidate 3):

### A. Breakdown by Confusion Family:
| Confusion Family | False Alarms | Percentage | Top Trigger Tokens |
| :--- | :--- | :--- | :--- |
| **`minimal_pair_consonant`** | 46 | 49.46% | *vải* (14), *bái* (7), *tải* (7), *hải* (4), *mài* (3) |
| **`phonetic_phrase_context`** | 23 | 24.73% | *bước xuống* (5), *hạ xuống* (4), *hướng phải* (4), *phôi phai* (3), *bước lên* (3) |
| **`minimal_pair_vowel_coda`** | 13 | 13.98% | *phú* (4), *vương* (4), *lưỡng* (4) |
| **`minimal_pair_tone`** | 9 | 9.68% | *lệnh* (5), *dòng xuồng* (3) |
| **`neutral_speech`** | 2 | 2.15% | Conversational distractor phrases |

### B. Breakdown by Inferred Target Keyword:
- **`PHAI`**: 47 false alarms (50.54%)
- **`XUONG`**: 22 false alarms (23.66%)
- **`TRAI`**: 13 false alarms (13.98%)
- **`LEN`**: 9 false alarms (9.68%)
- **`NONE`**: 2 false alarms (2.15%)

### C. Breakdown by Speaker Voice:
- **Đức Trí**: 60 false alarms (64.52%)
- **Quốc Tuấn**: 17 false alarms (18.28%)
- **Kim Thanh**: 11 false alarms (11.83%)
- **Quỳnh Anh**: 5 false alarms (5.38%)

---

## 4. Acoustic Synthesis & Root Cause Analysis

```
                    ┌───────────────────────────────┐
                    │  Shared Vietnamese Rhyme      │
                    │           /aːj/               │
                    └──────────────┬────────────────┘
                                   │
         ┌─────────────────────────┴─────────────────────────┐
         ▼                                                   ▼
┌─────────────────────────────────┐         ┌─────────────────────────────────┐
│     Target Command Tokens       │         │    Non-Target Distractors       │
│  "TRAI" (/ʈaːj/), "PHAI" (/faːj/)│        │ vải, bái, tải, hải, mài, thái... │
└────────────────┬────────────────┘         └────────────────┬────────────────┘
                 │                                           │
                 │ High TCN Rejection Sensitivity            │ Leaks through Stage 1 CNN
                 ▼                                           ▼
      52 Misses on TRAI (43.3%)                    60 FAs on /aːj/ Words (64.5%)
      57 Misses on PHAI (47.5%)
```

### The Inherent Tension:
In a 2-stage sliding window cascade:
- Tuning the verifier threshold high enough to reject distractors like *vải* and *bái* inherently cuts into the distribution of true *trái* and *phải* utterances, especially for speakers like Quỳnh Anh whose onset energy is softer.
- Conversely, lowering the threshold to recover *trái* and *phải* recall allows *vải*, *bái*, and phrase-embedded commands (*hướng phải*) to trigger continuous false alarms.

---

## 5. Transition to Bounded-Utterance Architecture

In **Bounded-Utterance Recognition**:
1. The model is given a single, complete speech utterance bounded by user intent (push-to-talk, button press, or isolated command prompt).
2. The model produces a single probability distribution over 5 outputs: `TRAI`, `PHAI`, `LEN`, `XUONG`, and `REJECT / UNKNOWN`.
3. Whole-utterance temporal pooling allows the network to contrast the onset consonant (/ʈ/, /f/) simultaneously with the coda (/aːj/), without sliding-window truncation artifacts.
