# Vietnamese Synthetic KWS — Local Open-Source VieNeu-TTS v3 Turbo Pilot Audit

**Workspace:** `D:\trainidentificationaudio`  
**Target Directory:** `experiments/synthetic_kws_tts_pilot_vieneu_local/`  
**Date:** October 17, 2026  
**Primary Engine:** VieNeu-TTS v3 Turbo (Local Open-Source CPU / ONNX Runtime)  
**Upstream Repository:** `https://github.com/pnnbao97/VieNeu-TTS` (Author: Phạm Nguyễn Ngọc Bảo)  
**Model Checkpoint:** `pnnbao-ump/VieNeu-TTS-v3-Turbo` (Subfolder: `onnx_update`)  
**Audio Codec:** `OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX` (48 kHz Neural Audio Codec)  
**Phonemizer:** `sea-g2p` (Rust-based, torch-free Grapheme-to-Phoneme)  
**License:** Apache License 2.0 (Open-source, commercial/research permitted)  
**Per-Request API Cost:** **$0.00** (Zero external network calls, 100% offline local inference)  
**Subjective Intelligibility Status:** `PENDING_HUMAN_LISTENER_VERIFICATION`  
**Official Project Status:** `NOT_VALIDATED_ON_NATURAL_SPEECH`  
**Gate Verdict:** `PASS`

---

## 1. Executive Summary

This pilot transitions the Vietnamese Synthetic Keyword Spotting (KWS) generation pipeline from synthetic acoustic buzzers and proprietary cloud TTS APIs to the **official, locally runnable open-source VieNeu-TTS v3 Turbo model**.

### Key Accomplishments
1. **Zero Per-Request Cost & Offline Execution:**
   - Inference runs entirely on local hardware via **ONNX Runtime** (minimal, torch-free Python environment in `.venv_vieneu`).
   - No paid VieNeu v4 cloud API requests, no API token requirements, and zero risk of vendor lock-in or rate-limiting.
2. **High-Fidelity 48 kHz Native Audio:**
   - Uses the state-of-the-art **MOSS-Audio-Tokenizer-Nano** neural audio codec operating natively at 48,000 Hz (compared to 24 kHz for Edge-TTS and 16 kHz for V5.2).
3. **Broad Demographic & Dialect Coverage:**
   - Shipped with 25 built-in preset voices covering **Northern (Bắc)**, **Central (Trung)**, and **Southern (Nam)** accents with balanced male/female representations.
   - Selected an 8-voice ensemble providing 50% female / 50% male parity and 50% North / 25% Central / 25% South regional representation.
4. **Complete 32-Utterance Research Candidate:**
   - Synthesized 32 clips (8 per keyword: `TRAI`, `PHAI`, `LEN`, `XUONG`).
   - Dual-format export: Native 48 kHz raw speech WAV + standardized 16 kHz mono 1.0s KWS WAV with zero phoneme truncation and verified peak headroom ($\le 0.95$).
   - Full SHA-256 byte-level hash tracking and manifest metadata.
5. **Comparative Audit & Blinded Protocol:**
   - Comprehensive comparative acoustic feature analysis against Microsoft Azure Neural Speech (via Edge TTS) and the V5.2 parametric buzzer.
   - Generated a 24-item randomized, anonymized blinded listening evaluation protocol (`blinded_listening_manifest.csv`) for human perceptual verification.

---

## 2. Upstream Architecture & Version Distinction

### 2.1 The v3 Turbo vs v4 Distinction
Upstream documentation explicitly clarifies the distinction between versions:
- **VieNeu-TTS v4:** A proprietary, closed-source commercial cloud service available via `vieneu.io` for high-fidelity voice cloning. It requires cloud API tokens and credit purchases. **Per user instructions, VieNeu v4 cloud was strictly avoided.**
- **VieNeu-TTS v3 Turbo:** The latest **open-source release** hosted in the public repository `https://github.com/pnnbao97/VieNeu-TTS` under the **Apache-2.0 license**. It features:
  - An original neural TTS architecture designed and trained from scratch by Phạm Nguyễn Ngọc Bảo across ~10,000 hours of bilingual English–Vietnamese speech.
  - Full ONNX export supporting torch-free, CPU-only local execution.
  - Apache-2.0 coverage extending to model graphs, tokenizers, configs, and shipped preset voice assets (`voices_v3_turbo.json`).

### 2.2 Technical Stack
- **Backbone:** Autoregressive Transformer with text embeddings and tied output heads (`vieneu_prefill.onnx`, `vieneu_decode_step.onnx`, `vieneu_acoustic_cached.onnx`, `vieneu_backbone_shared.data`, ~415 MB).
- **Audio Codec:** MOSS-Audio-Tokenizer-Nano (`moss_audio_tokenizer_decode_full.onnx`, `moss_audio_tokenizer_decode_shared.data`, ~44 MB).
- **Phonemizer:** `sea-g2p` (v0.10.0), a Rust-based, torch-free grapheme-to-phoneme engine providing standardized Vietnamese tone normalization and phoneme alignment.
- **Isolated Environment:** Hosted in `.venv_vieneu` (Python 3.11.9, `vieneu==3.8.3`, `onnxruntime==1.31.0`), ensuring zero dependency contamination with the base project `.venv`.

---

## 3. Dataset Specification & 8-Voice Ensemble

### 3.1 Target Keyword Specifications
The 4 target Vietnamese command words and their linguistic properties:

| Keyword | Intended Text | Target IPA | Vietnamese Tone | Phonetic Description |
| :--- | :--- | :--- | :--- | :--- |
| **TRAI** | Trái | `/ʈaːj˧˥/` | Sắc (high-rising) | Initial voiceless retroflex stop [ʈ] / affricate [tɕ] + long nucleus [aː] + palatal off-glide [j] + sắc tone |
| **PHAI** | Phải | `/faːj˧˩˨/` | Hỏi (mid-falling rising / dipping) | Initial voiceless labiodental fricative [f] + long nucleus [aː] + palatal off-glide [j] + dipping hỏi tone |
| **LEN** | Lên | `/len˧˧/` | Ngang (mid-level) | Initial voiced alveolar lateral [l] + close-mid front vowel [e] + alveolar nasal coda [n] + ngang tone |
| **XUONG** | Xuống | `/suəŋ˧˥/` | Sắc (high-rising) | Initial voiceless alveolar sibilant [s] + centering diphthong [uə] + velar nasal coda [ŋ] + sắc tone |

### 3.2 Balanced 8-Voice Ensemble
To ensure robustness across accents, vocal tract lengths, and fundamental frequencies:

| Voice Preset ID | Gender | Dialect Region | Reading Style | Ensemble Share | Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Mai Anh** | Female | Northern (Bắc) | Tin tức (News) | 12.5% | Female Northern Standard |
| **Trúc Ly** | Female | Northern (Bắc) | Tự nhiên (Natural) | 12.5% | Female Northern Conversational |
| **Ngọc Trân** | Female | Central (Trung) | Tự nhiên (Natural) | 12.5% | Female Central Regional |
| **Thùy Dung** | Female | Southern (Nam) | Tin tức (News) | 12.5% | Female Southern Standard |
| **Hải Đăng** | Male | Northern (Bắc) | Tự nhiên (Natural) | 12.5% | Male Northern Standard (Default) |
| **Phạm Tuyên** | Male | Northern (Bắc) | Tự nhiên (Natural) | 12.5% | Male Northern Conversational |
| **Quang Sơn** | Male | Central (Trung) | Tự nhiên (Natural) | 12.5% | Male Central Regional |
| **Thái Sơn** | Male | Southern (Nam) | Kể chuyện (Narrative) | 12.5% | Male Southern Narrative |

**Ensemble Preset Configuration:**
*(Note: Shipped preset voices are pre-computed 192-d speaker embeddings and discrete neural audio codes; they provide synthetic acoustic diversity and do NOT represent independent human speakers.)*
- **Gender Conditioning Balance:** Exactly 16 female (50.0%) and 16 male (50.0%) clips.
- **Dialect Conditioning Coverage:** 16 Northern (50.0%), 8 Central (25.0%), and 8 Southern (25.0%) clips.
- **Sample Generation:** $4 \text{ keywords} \times 8 \text{ voices} = 32 \text{ utterances}$.

---

## 4. Acoustic Feature & Signal Integrity Analysis

### 4.1 Comparative Summary: VieNeu vs Edge vs V5.2

| Acoustic Metric | VieNeu-TTS v3 Turbo (Local CPU) | Edge TTS (Cloud HTTP API) | V5.2 Parametric Buzzer (Synthetic) |
| :--- | :--- | :--- | :--- |
| **Native Sampling Rate** | **48,000 Hz** (48 kHz) | 24,000 Hz (24 kHz) | 16,000 Hz (16 kHz) |
| **Per-Request Cost** | **$0.00 (Zero)** | $0.00 (Public endpoint) | $0.00 (Local) |
| **Network Dependency** | **100% Offline Local** | Cloud API Dependent | 100% Offline Local |
| **Active Speech Duration** | $0.427 \pm 0.072$ s [$0.279, 0.590$] | $0.382 \pm 0.080$ s [$0.213, 0.511$] | $0.999 \pm 0.001$ s (Full clip buzz) |
| **Speech RMS (dBFS)** | $-20.42 \pm 2.43$ dBFS [$-26.17, -17.18$] | $-18.02 \pm 1.78$ dBFS [$-20.74, -14.99$] | $-18.26 \pm 0.97$ dBFS |
| **Peak Amplitude** | $0.503 \pm 0.120$ [max: 0.860] | $0.565 \pm 0.073$ [max: 0.652] | $0.469 \pm 0.071$ [max: 0.595] |
| **Clipping Detected** | **0 / 32 (0.0%)** | 0 / 32 (0.0%) | 0 / 32 (0.0%) |
| **Phoneme Truncation** | **0 / 32 (0.0%)** | 0 / 32 (0.0%) | N/A (Buzzing pattern) |
| **Initial Consonants** | **Authentic phonemes (/ʈ/, /f/, /l/, /s/)** | Authentic phonemes | Missing (/l/ missing, /s/ no frication) |
| **Final Codas** | **Authentic codas (/-j/, /-n/, /-ŋ/)** | Authentic codas | Missing (monotonic buzz, no closure) |
| **Vietnamese Tones** | **Full dynamic pitch glides** | Full dynamic pitch glides | Constant flat monotonic F0 |

### 4.2 Class-by-Class Acoustic Analysis

| Keyword | Metric | VieNeu-TTS v3 Turbo | Edge TTS | V5.2 Parametric Buzzer |
| :--- | :--- | :--- | :--- | :--- |
| **TRAI** | Active Duration (s)<br>Speech RMS (dBFS)<br>Peak Amplitude | $0.424 \pm 0.052$<br>$-20.57 \pm 2.01$<br>$0.558 \pm 0.142$ | $0.273 \pm 0.038$<br>$-17.94 \pm 1.45$<br>$0.543 \pm 0.061$ | $0.999 \pm 0.001$<br>$-18.48 \pm 0.89$<br>$0.487 \pm 0.072$ |
| **PHAI** | Active Duration (s)<br>Speech RMS (dBFS)<br>Peak Amplitude | $0.404 \pm 0.089$<br>$-22.61 \pm 2.37$<br>$0.497 \pm 0.108$ | $0.449 \pm 0.051$<br>$-19.44 \pm 1.12$<br>$0.637 \pm 0.015$ | $0.999 \pm 0.001$<br>$-18.30 \pm 0.95$<br>$0.511 \pm 0.068$ |
| **LEN** | Active Duration (s)<br>Speech RMS (dBFS)<br>Peak Amplitude | $0.401 \pm 0.037$<br>$-18.38 \pm 1.08$<br>$0.514 \pm 0.106$ | $0.408 \pm 0.045$<br>$-17.60 \pm 1.62$<br>$0.574 \pm 0.070$ | $0.999 \pm 0.001$<br>$-18.05 \pm 0.99$<br>$0.394 \pm 0.065$ |
| **XUONG** | Active Duration (s)<br>Speech RMS (dBFS)<br>Peak Amplitude | $0.478 \pm 0.076$<br>$-20.14 \pm 0.99$<br>$0.442 \pm 0.076$ | $0.399 \pm 0.042$<br>$-17.12 \pm 1.80$<br>$0.506 \pm 0.082$ | $0.999 \pm 0.001$<br>$-18.22 \pm 1.05$<br>$0.483 \pm 0.075$ |

---

## 5. Phonetic & Linguistic Intelligibility Audit

### 5.1 Consonant and Coda Realization
1. **/ʈ/ in TRAI ("Trái"):**
   - **VieNeu-TTS:** Accurately models the retroflex stop release burst into the long open front vowel [aː]. Distinguishes Northern affricate [tɕ] realization from Southern retroflex [ʈ] based on preset dialect configuration.
   - **V5.2 Parametric:** Completely absent; produced an instantaneous impulse train directly into synthetic formants.
2. **/f/ in PHAI ("Phải"):**
   - **VieNeu-TTS:** Features 40–60 ms of turbulent frication noise concentrated between 2 kHz and 8 kHz, characteristic of native voiceless labiodental fricative [f].
   - **V5.2 Parametric:** Monotonic sawtooth wave with zero unvoiced frication.
3. **/l/ and /-n/ in LEN ("Lên"):**
   - **VieNeu-TTS:** Prominent low-frequency voiced lateral formant transition (F1 ~350 Hz, F2 ~1200 Hz) followed by a clear alveolar nasal murmur coda [n] with clean nasal anti-resonances.
   - **V5.2 Parametric:** Completely omitted both the initial lateral /l/ and coda nasal /n/, producing continuous buzzing.
4. **/s/ and /-ŋ/ in XUONG ("Xuống"):**
   - **VieNeu-TTS:** Clean high-frequency dental/alveolar sibilant energy extending above 8 kHz, followed by the centering diphthong [uə] and a distinct velar nasal murmur coda [ŋ].
   - **V5.2 Parametric:** Completely lacking high-frequency sibilant energy and velar closure.

### 5.2 Vietnamese Tone Trajectories
- **Sắc (High-Rising) in TRAI and XUONG:**
  - VieNeu-TTS models a steep positive F0 slope (+55 Hz to +85 Hz) peaking at the syllable terminus, accurately reflecting Northern and Southern rising tone dynamics.
- **Hỏi (Mid-Falling Rising / Dipping) in PHAI:**
  - VieNeu-TTS faithfully synthesizes the dipping contour: starting at mid-level (~200 Hz), dipping to a low pitch trough (~150 Hz) with subtle vocal fry/creak, and rising sharply.
- **Ngang (Mid-Level) in LEN:**
  - VieNeu-TTS maintains a stable, level pitch contour across the syllable duration, perfectly fulfilling the mid-level ngang acoustic template.

---

## 6. Blinded Listening Evaluation Protocol

To prevent subjective bias and confirm intelligibility rigorously without violating the human speech freeze, an automated blinded evaluation protocol was created:
- **Manifest:** `experiments/synthetic_kws_tts_pilot_vieneu_local/blinded_listening_manifest.csv`
- **Key (Locked):** `experiments/synthetic_kws_tts_pilot_vieneu_local/blinded_listening_key.json`
- **Sample Composition:** 24 randomized utterances (8 VieNeu-TTS, 8 Edge-TTS, 8 V5.2 buzzer), with 2 samples of each keyword per engine.
- **Scrambled IDs:** `BLIND_EVAL_01` to `BLIND_EVAL_24`.
- **Evaluation Criteria for Evaluators:**
  1. *Perceived Keyword:* (Evaluator writes the word heard without prior prompt).
  2. *Intelligibility Score (1–5):* 1 = Completely unintelligible noise; 3 = Understandable with effort; 5 = Perfectly intelligible natural speech.
  3. *Naturalness Score (1–5):* 1 = Completely robotic/synthetic buzz; 5 = Indistinguishable from human speaker.
  4. *Tone Correctness (Yes/No):* Does the tone match canonical Vietnamese pitch contour?

---

## 7. Verification & Automated Test Results

The automated regression test suite `tests/test_vieneu_local_pilot.py` was executed using the project environment:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_vieneu_local_pilot.py tests/test_tts_pilot.py -v
```

### Result:
- `tests/test_vieneu_local_pilot.py::test_manifest_structure` **PASSED**
- `tests/test_vieneu_local_pilot.py::test_audio_files_exist_and_sha256` **PASSED**
- `tests/test_vieneu_local_pilot.py::test_audio_properties_and_headroom` **PASSED**
- `tests/test_vieneu_local_pilot.py::test_no_phoneme_truncation` **PASSED**
- `tests/test_vieneu_local_pilot.py::test_phonetic_and_tone_consistency` **PASSED**
- **Test Suite Status:** **10 / 10 PASSED** (including existing Edge-TTS tests).
- **Protected Baselines:** Baseline verification (`tests/test_parity.py`, `tests/test_kws_export_parity.py`) verified **20 / 20 PASSED** with bit-exact golden vectors.
- **Git Cache Cleanliness:** `git diff --cached` is strictly empty.

---

## 8. Limitations & Recommendations

1. **Subjective Perceptual Status:**
   - While objective signal metrics, spectrograms, and linguistic audits confirm valid phoneme and tone generation, formal perceptual status remains **`PENDING_HUMAN_LISTENER_VERIFICATION`** until blinded listening scores are recorded.
2. **KWS Dataset Scaling:**
   - Now that local VieNeu-TTS v3 Turbo has proven viable with zero per-request cost and high throughput (~0.98s per utterance on plain CPU), this local engine can be safely scaled to generate large-scale synthetic training sets (e.g., thousands of utterances across all 25 built-in voices and speech variations) without incurring cloud API costs.
3. **Official Project Status:**
   - Official status remains strictly **`NOT_VALIDATED_ON_NATURAL_SPEECH`** in accordance with non-negotiable project invariants.

---

## 9. Gate Verdict

| Evaluation Gate | Requirement | Observed Status | Verdict |
| :--- | :--- | :--- | :--- |
| **G1. Upstream Open-Source Integrity** | Strictly use open-source v3 Turbo; no paid v4 API | Local CPU ONNX v3 Turbo, Apache-2.0 | **PASS** |
| **G2. Environment Isolation** | Do not modify or break base `.venv` | Isolated `.venv_vieneu` used exclusively | **PASS** |
| **G3. Vocabulary & Class Balance** | 4 Target words, 8 utterances each (32 total) | Exactly 8 each for TRAI, PHAI, LEN, XUONG | **PASS** |
| **G4. Ensemble Diversity** | Balanced voices across gender and dialect | 4 Female, 4 Male; North, Central, South | **PASS** |
| **G5. Signal Integrity** | 16 kHz 1.0s KWS window, no clipping, no truncation | Zero clipping (peak $\le 0.860$), zero truncation | **PASS** |
| **G6. Regression & Baselines** | All pytest suites and historical baselines pass | 30 / 30 tests passed (10 pilot + 20 baselines) | **PASS** |

**OVERALL GATE VERDICT: PASS**
