# Dataset Collection, Annotation, and Quality Protocol

**Document Version:** 1.0.0  
**Status:** Approved Standard  
**Project:** Vietnamese Four-Word Keyword Spotting (KWS) for Mobile (Android & iOS)

---

## 1. Project Objectives & System Scope

The goal is to engineer a compact, high-accuracy, offline keyword spotting model capable of detecting exactly four Vietnamese navigation/directional commands on resource-constrained mobile devices (Android and iOS):

1. **`TRAI`** (`trái` — turn left / left)
2. **`PHAI`** (`phải` — turn right / right)
3. **`LEN`** (`lên` — go up / ascend / forward)
4. **`XUONG`** (`xuống` — go down / descend / backward)

The system is deployed in mobile client applications and must satisfy:
- **Low latency:** On-device streaming inference under 50 ms per window on mid-tier mobile CPUs.
- **Robustness:** High rejection of background speech, phonetically similar words, conversational speech, and ambient environmental noises.
- **Cross-platform parity:** Bit-accurate or perceptually identical audio preprocessing frontends across Python training, Android (Java/Kotlin/C++ via TFLite/ONNX), and iOS (Swift/C++ via TFLite/CoreML/ONNX).

---

## 2. Six-Class Taxonomic Definition

The model is trained against a stable 6-class categorical taxonomy:

| Class ID | Label | Linguistic Target | Acoustic & Semantic Description |
| :---: | :---: | :---: | :---: |
| **0** | `TRAI` | `trái` | Isolated pronunciation of the Vietnamese word *trái* (falling tone `sắc`, onset `/tɕ/` or Southern `/ʈ/`). |
| **1** | `PHAI` | `phải` | Isolated pronunciation of *phải* (asking tone `hỏi`, labiodental onset `/f/`). |
| **2** | `LEN` | `lên` | Isolated pronunciation of *lên* (level tone `ngang`, lateral onset `/l/`). |
| **3** | `XUONG` | `xuống` | Isolated pronunciation of *xuống* (falling tone `sắc`, fricative onset `/s/`). |
| **4** | `UNKNOWN` | Non-command speech | Open-set Vietnamese speech, conversational sentences, and phonetically confusing hard negatives (see Section 2.1). |
| **5** | `SILENCE` | Background & noise | Ambient noise, HVAC hum, fan noise, road noise, mic hiss, and acoustic room tone without speech. |

### 2.1 Critical Distinction: UNKNOWN vs. SILENCE
- **`SILENCE` (Class 5):** Does NOT mean pure mathematical zero vectors ($y[n] = 0$). True acoustic silence on mobile devices contains microphone thermal noise, analog-to-digital converter (ADC) quantization noise, room reverberant decay, air conditioner rumble, and low-level mechanical vibrations. Training on non-zero environmental noise prevents false activations during silent pauses.
- **`UNKNOWN` (Class 4):** Speech of any language (primarily Vietnamese) that is not one of the isolated target commands. This explicitly includes:
  - *Confusing Near-Rhymes & Homophones:* words sharing onsets, nuclei, or codas, e.g.:
    - For `TRAI`: *trại*, *chải*, *trán*, *tra*, *trên*, *trong*, *trưa*.
    - For `PHAI`: *phai*, *phái*, *phạt*, *pháp*, *khoai*, *hoài*.
    - For `LEN`: *lén*, *lẹ*, *đen*, *men*, *sen*, *quen*, *lại*, *lúc*.
    - For `XUONG`: *xuồng*, *xương*, *xoong*, *chuông*, *luống*, *xuôi*.
  - *Embedded Target Commands in Continuous Speech:* When a speaker says *"Tôi phải đi làm"* (I have to go to work) or *"Bên trái là trường học"*, the system MUST NOT trigger an isolated command activation. Such sentences belong to class `UNKNOWN`.

---

## 3. Large Corpora Utilization Policy

In compliance with project directives:
1. **VIVOS is NOT the default or primary dataset.** VIVOS is restricted by CC BY-NC-SA 4.0 and must not be used as the core data engine for a commercial deployment.
2. **Continuous Speech Is Hard Negatives by Default:** General speech corpora (such as FLEURS Vietnamese, Common Voice Vietnamese sentences, VietSpeech) are ingested into the data pipeline primarily as `UNKNOWN` negative examples.
3. **Strict Keyword Extraction Rule:** A candidate word occurrence within continuous speech may only be extracted as a command segment if:
   - High-confidence word-level or phoneme-level forced alignment timestamps exist (e.g., via a verified acoustic aligner).
   - Audio energy boundaries show clear acoustic pauses ($\ge 100$ ms silence pre- and post-utterance) so the acoustic co-articulation does not corrupt the command envelope.
   - The candidate clip is manually or programmatically verified for isolated command intelligibility.
   - **Under no circumstances may an unsegmented full sentence containing "trái" or "phải" be labeled as class 0 or class 1.**

---

## 4. Target Data Collection Protocol

To build a high-quality corpus of isolated command utterances, a multi-stage field collection protocol is defined.

### 4.1 Target Sample Budget
- **Initial Target:** 1,500 to 3,000 verified recordings per command class (`TRAI`, `PHAI`, `LEN`, `XUONG`).
- **Total Command Pool:** 6,000 to 12,000 verified isolated command clips.
- **Negative Pool:** Minimum 10 to 30 hours of diverse Vietnamese speech (`UNKNOWN`) and 5 to 10 hours of ambient noise (`SILENCE`).

### 4.2 Speaker & Regional Diversity Requirements
Vietnamese is a tonal language with prominent dialectal variations in consonants and vowels. The collection must stratify across:
1. **Northern Dialect (Giọng Bắc - Hanoi/Northern provinces):** Distinct tones (`ngã`, `hỏi`, `nặng`), merger of `d/gi/r` and `tr/ch`.
2. **Central Dialect (Giọng Trung - Hue/Da Nang/Nghe An/Ha Tinh):** Distinct tone contours, heavy glottalization, central vowels.
3. **Southern Dialect (Giọng Nam - Ho Chi Minh City/Mekong Delta):** Distinct final consonants (`-t/-k`, `-n/-ng`), onset merger of `v/d`, tone simplification.

**Target Speaker Quotas:**
- Minimum 100 distinct speakers for the target commands.
- Gender balance: Target 45–55% female, 45–55% male.
- Age groups: 18–30 (40%), 31–50 (40%), 51+ (20%).

### 4.3 Hardware & Recording Setup Diversity
Recordings must be gathered across actual mobile form factors rather than studio-only condenser microphones:
- **iOS Devices:** iPhone (recent models with dynamic beamforming microphones, e.g., iPhone 11 through iPhone 15), iPad.
- **Android Devices:** High-tier (Samsung Galaxy S-series / Google Pixel), mid-tier (Xiaomi, Realme, Vivo, Oppo) with varying AGC (Automatic Gain Control) characteristics.
- **Peripheral Microphones:** Wired 3.5mm/USB-C headsets, Bluetooth wireless earbuds (AAC/SBC codec degradation simulation).

### 4.4 Acoustic Conditions & Speaking Styles
Each participant performs utterances across diverse acoustic scenarios:
- **Quiet Room:** Home bedroom/office ($SNR > 25$ dB).
- **Reverberant Room:** Tiled living room or corridor (RT60 $\approx 0.4 - 0.7$ s).
- **Acoustic Noise:** Room with fan noise, air conditioner hum, background TV/radio, or outdoor street traffic ($SNR \approx 10 - 15$ dB).
- **Distance & Orientation:**
  - Close-talk (15–20 cm, handheld smartphone position).
  - Desk distance (50–70 cm, speakerphone position).
  - Far-field (1.5–2.0 m, ambient voice control).
  - Angles: On-axis (0°), off-axis (45°, 90°).
- **Speaking Dynamics:** Normal conversational delivery, fast paced, slow/deliberate, quiet (whisper-adjacent), and loud.

### 4.5 Informed Consent & Ethical Guidelines
1. **Consent Protocol:** Every contributor must sign an explicit, informed consent form authorizing the use of their recorded audio for machine learning research and commercial product deployment.
2. **Data Minimization:** No Personally Identifiable Information (PII) such as full legal name, national ID, address, or phone number may be stored in the audio manifests.
3. **Anonymization:** Participants are assigned a pseudonymous UUID (e.g., `SPK_VN_N014`). Only demographic attributes essential for acoustic stratification (gender, age bracket, regional dialect) are linked.
4. **Right of Revocation:** Participants retain the right to request deletion of their contributed audio prior to final model release freeze.

---

## 5. Audio Engineering Standards & Signal Normalization

Every audio file ingested into the training and validation pipeline must strictly adhere to the following audio specification:

| Parameter | Standard Value | Description / Rationale |
| :--- | :--- | :--- |
| **Sample Rate ($f_s$)** | $16,000$ Hz (16 kHz) | Production standard for voice models; captures speech frequencies up to 8 kHz (Nyquist) while minimizing mobile compute load. |
| **Channels** | 1 (Mono) | Mobile microphone input is processed as single-channel mono. Multi-channel audio is downmixed via averaging: $y[n] = \frac{1}{C}\sum_{c=1}^C x_c[n]$. |
| **Bit Depth** | 16-bit Signed PCM (`int16`) | Standard WAV format (`audio/wav; codec=1`). Normalized to Float32 range $[-1.0, 1.0]$ in memory. |
| **Clip Duration** | Exactly $1.0$ second ($16,000$ samples) | Standard fixed receptive window for low-latency sliding-window KWS. |
| **Padding & Centering** | Centered with random/zero jitter | Short isolated words ($0.3 - 0.7$ s) are padded to 1.0 s with the target command centered and small temporal offsets ($\pm 100$ ms) for shift invariance. |
| **Peak Normalization** | $-1.0$ dBFS | Prevents digital clipping while ensuring consistent acoustic volume. |

---

## 6. Audio Preprocessing Pipeline Specification (Log-Mel Frontend)

To ensure strict parity between training (PyTorch) and on-device execution (Android TFLite / iOS CoreML):

1. **Pre-emphasis (Optional, disabled by default for mobile stability):** Filter $y[n] = x[n] - \alpha x[n-1]$ with $\alpha = 0.97$.
2. **Short-Time Fourier Transform (STFT):**
   - Frame Length: $25$ ms ($400$ samples at 16 kHz).
   - Frame Step / Hop: $10$ ms ($160$ samples at 16 kHz).
   - FFT Length: $512$ points ($N_{FFT} = 512$).
   - Window Function: Periodic Hann window: $w[n] = 0.5 - 0.5 \cos\left(\frac{2\pi n}{N}\right)$.
3. **Mel Filterbank:**
   - Number of Mel Bins: $40$.
   - Minimum Frequency: $20$ Hz.
   - Maximum Frequency: $8,000$ Hz.
   - Scale: Standard HTK or Slaney Mel formula.
4. **Logarithmic Compression:**
   - Log-Mel: $S = \log(\text{MelEnergies} + \epsilon)$, where $\epsilon = 10^{-6}$ or $1e-5$.
5. **Output Matrix Shape:**
   - For a 1.0-second audio clip ($16,000$ samples), STFT generates:
     $$\text{Frames} = 1 + \left\lfloor \frac{16000 - 400}{160} \right\rfloor = 98 \text{ or } 101 \text{ frames (with center padding)}$$
   - Feature shape: $(1, \text{Frames}, 40)$ or $(1, 40, \text{Frames})$.

---

## 7. Data Quality Control, Integrity, & Leakage Prevention

### 7.1 Quality Control Checks
Every audio clip undergoes automated gating:
1. **Corrupt File Detection:** File must parse successfully with `soundfile` without header errors.
2. **Clipping Rejection:** Any file with more than 3 consecutive samples at $\pm 1.0$ or $|x[n]| \ge 0.999$ is flagged as clipped.
3. **Abnormal Silence / Low Energy Rejection:** Signals with RMS energy below $-50$ dBFS (for speech classes 0–4) are rejected as missing speech.
4. **Duration Check:** Audio shorter than $0.25$ s is rejected as truncation; audio longer than 1.0 s is segmented or windowed.

### 7.2 Strict Speaker-Disjoint Splitting
- **Zero Leakage Rule:** No speaker may appear in both the training set and the validation or test sets.
- **Recording Session Disjointness:** Newly recorded data from the same participant in the same session must not span splits.
- **Split Proportions:**
  - **Train:** $70\%$ of speakers / audio.
  - **Validation:** $15\%$ of speakers / audio (used for early stopping and threshold tuning).
  - **Test:** $15\%$ of speakers / audio (unseen speakers, unseen devices, unseen acoustic rooms).
- **Near-Duplicate Detection:** Acoustic fingerprinting and cosine distance on pre-trained embeddings to prevent identical takes from leaking across splits.

---

## 8. Dataset Manifest & Registry Format

All audio assets are immutably cataloged in a structured registry (`data/manifests/source_registry.csv`) with the following schema:

| Column | Type | Description |
| :--- | :--- | :--- |
| `recording_id` | String | Unique hash or identifier for the recording (e.g., `fleurs_vi_vn_11154206868753468740`). |
| `source` | String | Origin corpus (e.g., `google_fleurs`, `ml_spoken_words`, `field_recording_v1`). |
| `source_version` | String | Specific release tag (e.g., `fleurs_2022`, `mswc_v1.0`). |
| `license` | String | SPDX identifier (e.g., `CC-BY-4.0`, `Apache-2.0`, `Proprietary-Consented`). |
| `speaker_id` | String | Pseudonymous speaker ID (e.g., `SPK_FLEURS_F01`). |
| `accent` | String | Dialectal region (`Northern`, `Central`, `Southern`, `Unspecified`). |
| `raw_duration_sec`| Float | Original audio duration in seconds before processing. |
| `transcript` | String | Normalized transcript of the recording. |
| `class_id` | Integer | Categorical target class (0 to 5). |
| `class_label` | String | `TRAI`, `PHAI`, `LEN`, `XUONG`, `UNKNOWN`, `SILENCE`. |
| `label_provenance`| String | `human_verified_isolated`, `continuous_speech_hard_negative`, `environmental_noise`, `synthetic_quarantine`. |
| `split` | String | `train`, `validation`, or `test`. |
| `sha256` | String | Cryptographic SHA-256 checksum of the processed 16 kHz WAV file. |
| `processed_path` | String | Relative path to normalized 16 kHz 16-bit mono WAV. |
| `status` | String | `APPROVED`, `QUARANTINED`, `REJECTED`. |
| `exclusion_reason`| String | Blank if approved; reason if quarantined/rejected (e.g., `clipping`, `unverified_license`). |
