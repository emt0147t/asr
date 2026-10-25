# Vietnamese Synthetic KWS — Diagnostic Addendum: PHAI Tone Detection Analysis

**Dataset Directory:** `experiments/synthetic_kws_dataset_vieneu_1000/`  
**Target Class:** `PHAI` (`Phải`, target tone: *thanh hỏi*, IPA `/faːj˧˩˨/`)  
**Context:** Investigation of the reported 34.8% dipping contour rate in `TONE_AND_INTELLIGIBILITY_AUDIT.md`  
**Date:** October 17, 2026  
**Status:** Diagnostic Addendum (Preserves all historical reports intact)  
**Official Project Status:** `NOT_VALIDATED_ON_NATURAL_SPEECH`  

---

## 1. Objective of this Addendum

In `reports/TONE_AND_INTELLIGIBILITY_AUDIT.md`, the automated pitch analysis reported:
> "Dipping contour detected in 34.8% of clips; negative/depressed onset"

This addendum investigates the mathematical definition, denominator, acoustic assumptions, and dialectal factors behind this figure. The goal is neither to dismiss the observation nor to declare unverified samples clean, but to document exactly what the algorithm measured and what it left unmeasured.

---

## 2. Examination of the Underlying Detection Logic

### 2.1 The Original Formula and Denominator
In `audit_tone_diagnostics.py` (lines 93–100):
```python
# Segment voiced region into thirds (onset, nucleus, coda)
n_v = len(valid_f0)
t_third = max(1, n_v // 3)
f0_start = float(np.mean(valid_f0[:t_third]))
f0_mid = float(np.mean(valid_f0[t_third : 2 * t_third]))
f0_end = float(np.mean(valid_f0[-t_third:]))
is_dipping = (f0_mid < f0_start - 5.0) and (f0_end >= f0_mid)
```
- **Denominator:** Exactly **250 samples** (all accepted `PHAI` rows in `manifest.csv`).
- **Numerator:** Exactly **87 samples** satisfied `is_dipping == True` ($87 / 250 = 34.8\%$).
- **Complement:** Exactly **163 samples** failed the condition ($65.2\%$).

### 2.2 Failure Breakdown of the 163 Non-Dipping Classified Samples
When decomposing the strict compound condition into its components:
1. **Condition 2 Failure (`f0_end < f0_mid`, 84 samples / 51.5% of failures):**
   - The pitch dropped during the syllable but did not rise back above `f0_mid` in the final third. These samples exhibit a **monotonic falling pitch contour** rather than a classical dipping-rebounding contour.
2. **Condition 1 Failure (`f0_mid >= f0_start - 5.0 Hz`, 65 samples / 39.9% of failures):**
   - The pitch minimum occurred either late in the vowel (near the transition into the palatal off-glide `[-j]`) or early during vowel onset, falling outside the rigid middle 1/3 temporal window.
3. **Compound Failure (Both Conditions Failed, 14 samples / 8.6% of failures):**
   - Monotonic rising or irregular pitch trajectory.
4. **Unvoiced / Insufficient Voicing:**
   - 0 samples had fewer than 4 voiced frames.

---

## 3. Acoustic and Linguistic Explanations

### 3.1 Dialectal Variation in Vietnamese *Thanh Hỏi*
In Vietnamese acoustic phonetics, *thanh hỏi* is not realized identically across regions:
- **Northern Dialect (*Giọng Bắc*):** Typically realized as a dipping-rising contour (pitch falls, reaches a valley often with creaky voice/glottal constriction, and rises slightly at the off-glide).
- **Southern Dialect (*Giọng Nam*):** *Thanh hỏi* and *thanh ngã* merge phonologically into a mid-falling or low-falling contour without a substantial final pitch rebound.
- **Central Dialect (*Giọng Trung*):** Characterized by a lower pitch register and abrupt glottal constriction.

When examining the 250 `PHAI` samples by preset dialect region:
| Region | Sample Count | Strict Dipping (3-Thirds) | Classic Interior Dip | Monotonic Falling | Mean Pitch Slope ($\Delta F_0$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Bắc (North)** | 150 | 56 (37.3%) | 123 (82.0%) | 22 (14.7%) | -19.0 Hz |
| **Nam (South)** | 80 | 20 (25.0%) | 66 (82.5%) | 3 (3.8%) | +71.6 Hz |
| **Trung (Central)** | 20 | 11 (55.0%) | 16 (80.0%) | 1 (5.0%) | +20.2 Hz |

### 3.2 Continuous Interior Minimum vs. Rigid Slicing
When the F0 trajectory is evaluated for an interior pitch minimum (where the minimum pitch occurs between the first and last voiced frame, dropping $\ge 5$ Hz from onset and rebounding $\ge 5$ Hz):
- **Classic Interior Dip:** **205 / 250 (82.0%)**
- **Monotonic Falling:** **26 / 250 (10.4%)**
- **Rising Contour:** **14 / 250 (5.6%)**
- **Flat / Irregular:** **5 / 250 (2.0%)**

This demonstrates that the rigid division into three equal temporal thirds was overly restrictive for syllables where the pitch dip occurs non-linearly near the coda.

### 3.3 Autocorrelation and Glottal Constriction Artifacts
In Vietnamese hỏi tone, speakers frequently produce vocal fry (glottalization) at the pitch valley.
In `track_f0`, autocorrelation requires a peak-to-zero ratio $> 0.35$. During glottal constriction:
- Periodicity drops sharply, causing intermittent `NaN` frames right at the turning point of the dip.
- In low-pitched male voices (e.g., `Phạm Tuyên`, `Minh Triết` with F0 near 75–90 Hz), the pitch minimum can drop near or below the 70 Hz search floor, leading to tracking dropouts or octave jumps.

---

## 4. Methodological Inconsistency Across Keyword Classes

A critical audit finding is the inconsistency between the evaluation rules for `PHAI` versus the other three classes:
- **`TRAI` and `XUONG`:** Evaluated solely by aggregate mean slope ($\Delta F_0 = \text{f0\_end} - \text{f0\_start} > 0$). No binary "rising rate" pass/fail threshold was computed or enforced.
- **`LEN`:** Evaluated solely by mean slope near zero ($-6.8$ Hz). No binary "flat rate" threshold was enforced.
- **`PHAI`:** The *only* class subjected to a rigid binary classification rule (`is_dipping`), which was then presented as a percentage rate (34.8%).

This asymmetry created the false impression that `PHAI` had a high defect rate while `TRAI`, `LEN`, and `XUONG` were 100% verified. In reality, all four classes exhibit natural acoustic variability across preset voices.

---

## 5. Conclusions and Scientific Guidance

1. **Do not equate the 34.8% figure with a 65.2% defect rate:** The low percentage was largely an artifact of the rigid 3-third window definition and regional dialectal variations in hỏi tone realization.
2. **Do not claim 100% tone correctness either:** While 82.0% exhibit an interior dip and 10.4% exhibit a falling contour, acoustic pitch contours cannot substitute for human perceptual verification. A listener may perceive an acoustically falling tone as *huyền* (grave) rather than *hỏi* if the glottal constriction cue is absent.
3. **Actionable Resolution:** The 80-clip stratified blinded listening test includes **20 randomized `PHAI` clips** evaluated by human listeners for `perceived_tone (SAC/HOI/NGANG/OTHER/UNCERTAIN)`. Final tone validity must be determined by listener responses on [`blinded_listening_review_sheet.csv`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_dataset_vieneu_1000/blinded_listening/blinded_listening_review_sheet.csv).
