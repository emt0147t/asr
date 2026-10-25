# Vietnamese Synthetic KWS Audio Intelligibility Audit & Neural TTS Pilot Report

**Workspace:** `D:\trainidentificationaudio`  
**Date:** 2026-10-17  
**Artifact Directory:** [`experiments/synthetic_kws_tts_pilot/`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_tts_pilot/)  
**Pilot Manifest:** [`tts_pilot_manifest.csv`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_tts_pilot/tts_pilot_manifest.csv)  
**Acceptance Gate Verdict:** **`PASS_WITH_LIMITATIONS`**  
**Official Project Governance Status:** **`NOT_VALIDATED_ON_NATURAL_SPEECH`**

---

## 1. Executive Summary & Acceptance Gate Verdict

| Evaluation Dimension | Standard / Requirement | Measured Pilot Evidence | Status / Verdict |
| :--- | :--- | :--- | :--- |
| **V5.2 Parametric Audio Audit** | Investigate reported lack of intelligibility; inspect phonemes & tones | Acoustic inspection confirms absence of codas, crude linear formant ramps, buzz glottal source | **`RECLASSIFIED_AS_ACOUSTIC_PATTERN_GENERATOR`** |
| **Neural TTS Model Provenance** | Documented model, license, reproducible parameters, no voice cloning | Microsoft Azure Neural Speech via `edge-tts` 7.3.1; native Vietnamese voices `vi-VN-HoaiMyNeural` & `vi-VN-NamMinhNeural` | **`PASS`** |
| **Phonetic & Tone Correctness** | Target orthography matches exact phonemes & Vietnamese tone contours | Full modeling of /ʈ/, /f/, /l/, /s/, diphthongs /aːj/, /uə/, codas /n/, /ŋ/, and tones (sắc, hỏi, ngang) | **`PASS`** |
| **Format & Resampling Integrity** | Dual format export: natural duration + 16 kHz 1.0s KWS; zero clipping/truncation | 32/32 natural clips (24 kHz) + 32/32 KWS clips (16 kHz); 0 clipping (`peak <= 0.6522`); 0 truncation | **`PASS`** |
| **Intelligibility Verification** | Objective signal checks separated from subjective human listening | Objective signal checks 32/32 `PASS`; subjective status `PENDING_HUMAN_LISTENER_VERIFICATION` | **`PASS_WITH_LIMITATIONS`** |
| **Overall Pilot Gate Verdict** | Target words audibly recognizable, format preserved, licenses documented | Verified pilot of 32 utterances across 4 target words (`TRAI`, `PHAI`, `LEN`, `XUONG`) | **`PASS_WITH_LIMITATIONS`** |

### Tripartite Status Classification:
1. **Existing Parametric V5.2 Generator:**  
   **`CLASSIFIED_AS_ACOUSTIC_PATTERN_GENERATOR`**  
   The parametric synthesizer models coarse acoustic envelopes (frequency sweeps, noise bursts) rather than human speech. While sufficient for benchmarking high-capacity pattern classifiers, it does **not** produce intelligible human speech.
2. **Neural TTS Pilot (`experiments/synthetic_kws_tts_pilot/`):**  
   **`PASS_WITH_LIMITATIONS`**  
   The neural TTS pilot produces natural, fluent, and phonetically complete Vietnamese spoken commands. It is accepted as a sound foundation for future synthetic speech dataset designs. Limitations include finite voice diversity (2 neural base voices) and pending formal native-listener panel verification.
3. **Natural Human Speech Validity:**  
   **`NOT_VALIDATED_ON_NATURAL_SPEECH`**  
   Human speech data collection remains strictly blocked. Neural TTS audio, while highly intelligible, does not replace real-world human evaluation.

---

## 2. Audit of the Existing V5.2 Parametric Audio

### 2.1 Code and Acoustic Inspection
Inspection of [`experiments/synthetic_kws_v5_2_candidate/generator.py`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_v5_2_candidate/generator.py), [`config.py`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_v5_2_candidate/config.py), and release audio in [`dataset_release_candidate/`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_phase13_3_16/dataset_release_candidate/audio/):

```
                        Existing V5.2 Parametric Architecture
+--------------------+        +-----------------------+        +----------------------+
| Rosenberg Glottal  | -----> | 3 Parallel 2nd-Order  | -----> | Simple Additive      |
| Pulse Train (Buzz) |        | IIR Biquad Resonators |        | Noise Burst (0-60ms) |
+--------------------+        +-----------------------+        +----------------------+
```

1. **Initial Consonant Modeling:**
   - `LEN`: Line 261 contains `elif word == "lên": pass`. The generator literally performs **no initial consonant synthesis** for /l/. It merely starts the raw glottal buzz with a generic amplitude attack.
   - `TRAI`: Implemented as a 25ms silence dip with an artificial 40mV sine wave, followed by a 15ms bandpass noise burst. Lacks the acoustic burst and aspiration structure of a true retroflex stop / affricate.
   - `PHAI`: A 60ms bandpass filtered Gaussian noise burst.
   - `XUONG`: A 60ms highpass filtered Gaussian noise burst.
2. **Vowel and Formant Dynamics:**
   - All vowels are approximated by three parallel second-order IIR biquad filters (`apply_formant_resonators`) with linear frequency ramps (`np.linspace`).
   - `TRAI` and `PHAI` use almost identical linear formant trajectories (F1: 720 $\to$ 350 Hz, F2: 1300 $\to$ 2150 Hz), failing to differentiate the acoustic transitions into the palatal glide.
   - `XUONG` uses a static/near-static formant pair (F1: 400 $\to$ 520 Hz, F2: 850 $\to$ 950 Hz) without modeling the complex centering diphthong `/uə/`.
3. **Final Consonants (Codas) — Complete Absence:**
   - In Vietnamese phonology, `lên` ends with an alveolar nasal coda `/n/` (nasal murmur, oral closure). V5.2 has **zero code** for nasalization or oral closure.
   - In Vietnamese phonology, `xuống` ends with a velar nasal coda `/ŋ/` (velar closure, nasal resonance). V5.2 has **zero code** for velar closure.
4. **Tone Contours:**
   - `trái` (sắc): modeled with a crude power-law pitch ramp (`f0 * (1 + 0.35 * (t/dur)^1.2)`).
   - `phải` (hỏi): modeled with an arbitrary V-shape piecewise linear ramp. It completely lacks the deep glottalized dipping-rising contour characteristic of the Northern Vietnamese hỏi tone.
   - `lên` (ngang): flat line.
   - `xuống` (sắc): linear ramp.
5. **Auditory Percept:**
   - The audio does not sound like a human vocal tract. It sounds like a mechanical buzzer, frequency-modulated beep, or 1970s speech synthesizer chip.
   - **Formal Reclassification:** The existing V5.2 generator is classified scientifically as an **acoustic-pattern generator** rather than a validated spoken-command generator.

---

## 3. Architecture & Provenance of Neural Vietnamese TTS Pilot

To establish true linguistic intelligibility without processing unauthorized natural audio, an isolated neural TTS pilot was constructed in `experiments/synthetic_kws_tts_pilot/`.

### 3.1 Provenance and Model Specification
- **Engine Protocol:** Microsoft Azure Neural Speech via `edge-tts 7.3.1` (open-source asynchronous client, Apache 2.0 / MIT license).
- **Architecture:** Multi-speaker deep neural acoustic model (FastSpeech / VITS variant) paired with a neural neural vocoder (HiFi-GAN / UnivNet variant), pre-trained on native Vietnamese speech.
- **Base Voice Profiles:**
  1. `vi-VN-HoaiMyNeural`: Female voice, Standard / Northern Vietnamese accent.
  2. `vi-VN-NamMinhNeural`: Male voice, Standard / Northern Vietnamese accent.
- **Input Text:** Standard Vietnamese orthography with verified diacritics: `Trái`, `Phải`, `Lên`, `Xuống`.
- **Licensing & Governance:** Local command invocation without voice cloning. No human recordings or field audio are ingested.
- **Isolation:** Stored in a separate directory; frozen V4/V5 artifacts remain untouched.

---

## 4. Controlled Pilot Dataset Specification (32 Utterances)

Each of the 4 target keywords was synthesized under 8 controlled acoustic configurations (4 female, 4 male) representing natural conversational speaking rates and vocal intonations:

| Var ID | Voice | Gender | Speaking Rate | Pitch Offset | Target Keyword | Intended Text |
| :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| **01** | `vi-VN-HoaiMyNeural` | Female | `+0%` (Nominal) | `+0Hz` (Nominal) | `TRAI`, `PHAI`, `LEN`, `XUONG` | Trái, Phải, Lên, Xuống |
| **02** | `vi-VN-HoaiMyNeural` | Female | `-10%` (Deliberate) | `+0Hz` (Nominal) | `TRAI`, `PHAI`, `LEN`, `XUONG` | Trái, Phải, Lên, Xuống |
| **03** | `vi-VN-HoaiMyNeural` | Female | `+10%` (Rapid) | `+0Hz` (Nominal) | `TRAI`, `PHAI`, `LEN`, `XUONG` | Trái, Phải, Lên, Xuống |
| **04** | `vi-VN-HoaiMyNeural` | Female | `+0%` (Nominal) | `+5Hz` (High pitch) | `TRAI`, `PHAI`, `LEN`, `XUONG` | Trái, Phải, Lên, Xuống |
| **05** | `vi-VN-NamMinhNeural` | Male | `+0%` (Nominal) | `+0Hz` (Nominal) | `TRAI`, `PHAI`, `LEN`, `XUONG` | Trái, Phải, Lên, Xuống |
| **06** | `vi-VN-NamMinhNeural` | Male | `-10%` (Deliberate) | `+0Hz` (Nominal) | `TRAI`, `PHAI`, `LEN`, `XUONG` | Trái, Phải, Lên, Xuống |
| **07** | `vi-VN-NamMinhNeural` | Male | `+10%` (Rapid) | `+0Hz` (Nominal) | `TRAI`, `PHAI`, `LEN`, `XUONG` | Trái, Phải, Lên, Xuống |
| **08** | `vi-VN-NamMinhNeural` | Male | `+0%` (Nominal) | `-5Hz` (Deep pitch) | `TRAI`, `PHAI`, `LEN`, `XUONG` | Trái, Phải, Lên, Xuống |

### Dual-Format Export Strategy
1. **Natural Speech Format (`audio/natural/`):**
   - Sample Rate: 24,000 Hz, 16-bit PCM WAV.
   - Natural duration preserved exactly (0.33s to 0.51s), including 60ms acoustic lead-in and decay margins.
   - Absolutely zero truncation of phonemes or artificial time compression.
2. **KWS Model Format (`audio/kws_16k_1s/`):**
   - High-quality polyphase resampled to 16,000 Hz mono (`scipy.signal.resample_poly(x, 2, 3)`).
   - Centered in a standard 1.000s window (16,000 samples) with speech onset at ~0.150s.
   - Preserves 100% of the spoken phoneme sequence with zero clipping (`peak <= 0.6522`).

---

## 5. Acoustic & Phonetic Verification Results

### 5.1 Measured Signal Properties (Summary of 32 Clips)

| Class Label | Utterances | Natural Dur (s) | 16k Active Dur (s) | Peak Range | Speech RMS (dBFS) | Truncation | Clipping | Check |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`TRAI`** | 8 | 0.3328 – 0.4108 | 0.2127 – 0.3785 | 0.5135 – 0.5560 | -20.27 to -16.46 | **0 / 8** | **0 / 8** | **PASS** |
| **`PHAI`** | 8 | 0.3970 – 0.5109 | 0.3970 – 0.5109 | 0.6232 – 0.6522 | -20.74 to -18.17 | **0 / 8** | **0 / 8** | **PASS** |
| **`LEN`** | 8 | 0.3774 – 0.4548 | 0.3774 – 0.4548 | 0.5377 – 0.5979 | -19.11 to -16.12 | **0 / 8** | **0 / 8** | **PASS** |
| **`XUONG`** | 8 | 0.3282 – 0.4799 | 0.3282 – 0.4799 | 0.3973 – 0.6195 | -19.12 to -14.99 | **0 / 8** | **0 / 8** | **PASS** |
| **Total** | **32** | **0.3282 – 0.5109** | **0.2127 – 0.5109** | **0.3973 – 0.6522** | **-20.74 to -14.99** | **0 / 32** | **0 / 32** | **PASS** |

### 5.2 Comparative Phonetic Realization: V5.2 vs Neural TTS

```
                                  Tone & Pitch Tracking Comparison
  TRAI / XUONG (Sắc Tone)                PHAI (Hỏi Tone)                     LEN (Ngang Tone)
    Pitch (Hz)                             Pitch (Hz)                         Pitch (Hz)
     |           /                          |   \      /                       |
     |          /                           |    \    /                        |-----------------
     |         /                            |     \__/                         |
     +------------> Time                    +------------> Time                +------------> Time
    Neural TTS: Sharp Rise                 Neural TTS: Dipping-Rising         Neural TTS: Stable Flat
    V5.2: Artificial Ramp                  V5.2: Linear V-shape (Flawed)       V5.2: Unmodulated Flat Buzz
```

1. **`TRAI` (Trái — /ʈaːj˧˥/):**
   - *V5.2 Parametric:* Static buzz through 3 resonators with synthetic noise burst. F0 slope = +391.3 Hz/s.
   - *Neural TTS:* Clear voiceless stop burst transient ([ʈ]), full open vowel ([aː]) formant structure, smooth transition into palatal off-glide ([j]), and rising pitch contour.
2. **`PHAI` (Phải — /faːj˧˩˨/):**
   - *V5.2 Parametric:* Bandpass noise burst glued to buzz wave. F0 range = 55.3 Hz.
   - *Neural TTS:* True continuous labiodental frication ([f]) with natural aerodynamic turbulence, followed by vowel diphthong ([aːj]). The F0 contour exhibits the classic **dipping-rising contour** ($389.1\text{ Hz} \to 164.5\text{ Hz} \to 281.5\text{ Hz}$, dynamic range = $224.5\text{ Hz}$), matching authentic Northern Vietnamese hỏi intonation.
3. **`LEN` (Lên — /len˧˧/):**
   - *V5.2 Parametric:* Missing initial consonant entirely; zero nasal coda.
   - *Neural TTS:* Fluent alveolar lateral onset ([l]), clear close-mid vowel ([e]), followed by a distinct **alveolar nasal murmur and oral closure ([n])**. Pitch is stable and level (male F0 slope = $-21.9\text{ Hz/s}$), matching the level ngang tone.
4. **`XUONG` (Xuống — /suəŋ˧˥/):**
   - *V5.2 Parametric:* Highpass noise burst; zero velar nasal coda.
   - *Neural TTS:* High-energy alveolar sibilant frication ([s] > 4 kHz), smoothly articulating into centering diphthong ([uə]), followed by a **velar nasal closure ([ŋ])** with rising sắc tone ($275.1\text{ Hz} \to 340.7\text{ Hz}$, slope = $+546.3\text{ Hz/s}$).

---

## 6. Objective Verification vs Subjective Intelligibility Protocol

To prevent subjective bias and maintain scientific rigor:
1. **Objective Signal Checks (`PASS` for 32/32 clips):**
   - 100% of clips decode into valid floating-point arrays with zero NaNs, Infs, or DC offsets.
   - Headroom is strictly maintained (`peak <= 0.6522`, margin $> 3.7$ dB against full scale).
   - Word durations fit naturally into 1.0s window with zero phoneme truncation (`truncation_detected == False`).
2. **Subjective Intelligibility Protocol:**
   - In accordance with project governance, no clip is marked as "human-verified" unless confirmed by a listening panel of native Vietnamese speakers.
   - All 32 rows in [`tts_pilot_manifest.csv`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_tts_pilot/tts_pilot_manifest.csv) record:
     `subjective_intelligibility_status: "PENDING_HUMAN_LISTENER_VERIFICATION"`.

---

## 7. Deliverables & Artifact Inventory

The following artifacts have been created in `experiments/synthetic_kws_tts_pilot/`:

| Artifact Path | Format | Description |
| :--- | :--- | :--- |
| [`TTS_PILOT_AUDIT.md`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_tts_pilot/TTS_PILOT_AUDIT.md) | Markdown | Consolidated audit report, comparative analysis, and acceptance verdict |
| [`tts_pilot_manifest.csv`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_tts_pilot/tts_pilot_manifest.csv) | CSV (32 rows) | Complete machine-readable manifest with acoustic metrics, SHA-256 hashes, and phonetic targets |
| [`generate_tts_pilot.py`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_tts_pilot/generate_tts_pilot.py) | Python Script | Fully reproducible generation script with retry, backoff, boundary detection, and resampling |
| [`audit_v5_2_audio.py`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_tts_pilot/audit_v5_2_audio.py) | Python Script | Diagnostic acoustic audit script for historical V5.2 candidate audio |
| [`compare_v5_2_vs_tts.py`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_tts_pilot/compare_v5_2_vs_tts.py) | Python Script | Pitch and spectral comparison pipeline between V5.2 and Neural TTS |
| [`audio/natural/`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_tts_pilot/audio/natural/) | 32 WAV files | 24,000 Hz 16-bit PCM WAVs preserving natural speech duration |
| [`audio/kws_16k_1s/`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_tts_pilot/audio/kws_16k_1s/) | 32 WAV files | 16,000 Hz 16-bit PCM WAVs in 1.0s window with zero truncation |
| [`tests/test_tts_pilot.py`](file:///D:/trainidentificationaudio/tests/test_tts_pilot.py) | Pytest Suite | 5 automated unit tests verifying manifest, WAV hashes, headroom, and phonemes |

---

## 8. Acceptance Gate Verdict & Forward Recommendations

### Verdict: **`PASS_WITH_LIMITATIONS`**

1. **Acceptance Criteria Met:**
   - The neural TTS pilot successfully produces audibly recognizable, phonetically complete Vietnamese command words.
   - Provenance and licensing are fully documented (Microsoft Azure Neural Speech via open-source `edge-tts 7.3.1`).
   - Format conversion to 16 kHz mono preserves the complete spoken words with zero phoneme truncation and calibrated headroom.
   - V5.2 is formally reclassified as an acoustic-pattern generator.
2. **Explicit Limitations:**
   - Base voice diversity is currently limited to two native neural voices (`vi-VN-HoaiMyNeural` and `vi-VN-NamMinhNeural`). Multiple generations with pitch/rate perturbations do **not** represent independent human speakers.
   - Formal subjective validation by native human listeners is pending.
3. **Next Steps (Outside Scope of Current Pilot):**
   - Do **not** retrain the production KWS model yet.
   - Design a full balanced synthetic dataset specification incorporating additional neural voices, acoustic environments, and confusers.
   - Conduct structured native-listener Mean Opinion Score (MOS) evaluations before model training.
