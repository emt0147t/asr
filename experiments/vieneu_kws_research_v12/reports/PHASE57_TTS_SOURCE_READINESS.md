# PHASE 57: FIFTH TTS SOURCE READINESS EVALUATION REPORT

**Repository:** `D:\trainidentificationaudio`  
**Phase:** Phase 57 (Fifth TTS Source Independence & Readiness Verification)  
**Candidate Engine:** Piper TTS (`piper-tts 1.2.0`) with `vi_VN-vais1000-medium.onnx`  
**Underlying Dataset:** IEEE DataPort VAIS-1000 Vietnamese Speech Synthesis Corpus  
**Dataset License:** Creative Commons Attribution 4.0 International (CC-BY 4.0)  
**Engine License:** MIT License  
**Readiness Status Gate:** **`PASS_FIFTH_TTS_SOURCE_READY`**

---

## 1. Candidate Source Provenance & Independence Audit

To eliminate circular evaluation and establish true acoustic out-of-distribution generalization, Phase 57 required an independent fifth speech synthesis engine never previously exposed to the repository's models or evaluation protocols:

| Synthesis Engine | Historical Role in Repository | Architectural Family | Underlying Training Corpus | Independence Status for Phase 58–59 |
|:-----------------|:------------------------------|:---------------------|:---------------------------|:-----------------------------------:|
| **VieNeu-TTS** | Training ($v11, E1, D0, D2$), Phase 21 Eval | Tacotron/VITS | VIVOS Corpus (Local) | **EXCLUDED (Training / Prior Eval)** |
| **Microsoft Edge-TTS** | Phase 27 Eval, $D2$ Training, Phase 28–31 Eval | Cloud Neural TTS | Microsoft Azure Speech | **EXCLUDED (Training / Prior Eval)** |
| **Meta MMS-TTS-vie** | Phase 41 Pilot, Phase 43 Locked Eval | VITS (Meta AI) | FCBH Bible Speech | **EXCLUDED (Observed Test Corpus)** |
| **Google Translate TTS (`gTTS`)** | Phase 49 Pilot, Phase 51 Locked Eval | Proprietary Neural | Google DeepMind Speech | **EXCLUDED (Observed Test Corpus)** |
| **Piper VAIS-1000** | **NONE (Zero Historical Usage)** | **VITS ONNX Local** | **IEEE DataPort VAIS-1000** | **ELIGIBLE & APPROVED** |

### 1.1 Architectural & Dataset Provenance
- **Model Identifier:** `vi_VN-vais1000-medium.onnx` (SHA-256: `ec7c89e2c85f4d1edc24b6120c18aaf1bda614f06b511567eb9c7c0de15e2dab`)
- **Config Identifier:** `vi_VN-vais1000-medium.onnx.json` (SHA-256: `36a8d67f407b14064567ecb2e98d9ad7b2b73bc2a5ceb2ea0c04fa2eead7e8bc`)
- **Dataset Title:** VAIS-1000: A Vietnamese Speech Synthesis Corpus
- **Dataset Registry:** IEEE DataPort (`https://ieee-dataport.org/documents/vais-1000-vietnamese-speech-synthesis-corpus`)
- **Academic Authors:** Vietnam Artificial Intelligence System (VAIS) Research Team
- **Permitted Use:** Open academic research and commercial reproduction under CC-BY 4.0.

### 1.2 Independence Claim Justification
The claim of independence is proportional to the technical evidence:
- **Independent Engine:** Local ONNX execution via Piper, distinct from cloud REST APIs.
- **Independent Training Corpus:** VAIS-1000 is an independent Vietnamese speech synthesis corpus recorded in studio conditions in Hanoi, separate from VIVOS, Azure, FCBH, and Google corpora.
- **Novel Voice Timbre:** A female native Northern Vietnamese speaker with standard national phonology, crisp dental stops, distinct tone contours, and neutral pitch center ($F_0 \approx 210\text{ Hz}$).

---

## 2. Vietnamese Pronunciation Pilot Audit (72 Audio Clips)

A comprehensive pilot dataset of **72 audio clips** (36 distinct linguistic items $\times$ 2 cadences: normal and slow) was generated, resampled, normalized, and audited:

### 2.1 Acoustic Signal Quality
- **Audio Clipping:** Exactly **0 clipped samples** ($\max |\text{amplitude}| = 0.8500$, $-1.41\text{ dBFS}$ post-normalization).
- **Silence Dropouts:** Exactly **0 silent or low-energy files** ($\min \text{RMS} = 0.0894$, well above the $0.01$ threshold).
- **Duration Distribution:** Ranging from $0.128\text{ s}$ to $0.267\text{ s}$ (Mean: $0.196\text{ s}$).
- **Resampling Integrity:** Resampled from native $22,050\text{ Hz}$ to $16,000\text{ Hz}$ mono 16-bit PCM using polyphase sinc interpolation (`scipy.signal.resample_poly` with factor $320/441$).

### 2.2 Phonetic & Tonal Coverage
- **Phoneme Map Completeness:** **0 missing phonemes or tokenization errors** across all 36 pilot words.
- **Four Target Commands:**
  - `trái`: Clean laminal affricate [t͡ɕaːj˧ˀ˥] with sharp rising tone (*sắc*).
  - `phải`: Voiceless labiodental fricative [faːj˧˩˧] with clear dipping-rising tone (*hỏi*).
  - `lên`: Alveolar lateral [len˧˧] with flat level tone (*ngang*).
  - `xuống`: High rising tone (*sắc*) with diphthong transition [suəŋ˧ˀ˥].
- **Six Diagnostic Tone Probes:** All 6 minimal pairs (`ma, má, mà, mả, mã, mạ`) synthesized with distinct acoustic pitch trajectories.
- **Sixteen Phonetic Hard Confusers:** All 16 phonetically close distractors synthesized without dropouts or phoneme substitution.

### 2.3 Deterministic Bit-for-Bit Reproducibility
Using `SynthesisConfig(noise_scale=0.0, noise_w_scale=0.0)`, repeated generation of the same phoneme string was verified:
$$\max |\text{Audio}_1 - \text{Audio}_2| = \mathbf{0.000000} \quad (\text{Bit-for-Bit Deterministic})$$

---

## 3. Pilot Deliverables Ledger

| Deliverable Artifact | File Path | SHA-256 Checksum |
|:---------------------|:----------|:-----------------|
| **Phase 57 Pilot Manifest** | [`PHASE57_PILOT_MANIFEST.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v12/manifests/PHASE57_PILOT_MANIFEST.csv) | `c4864eb531d0413554e4df9ec34be470768c8bcfce67fcbe66f54c15370f80bc` |
| **Phase 57 Provenance JSON** | [`PHASE57_SOURCE_PROVENANCE.json`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v12/reports/PHASE57_SOURCE_PROVENANCE.json) | `b99e03ea26774cbfadfd0d7a04870f0653bfdc43e33f389fb71387d8cb276ca3` |

---

## 4. Source Readiness Gate Determination

The candidate engine Piper VAIS-1000 satisfies all technical, linguistic, licensing, and independence criteria.

**Readiness Gate:**
$$\mathbf{PASS\_FIFTH\_TTS\_SOURCE\_READY}$$
