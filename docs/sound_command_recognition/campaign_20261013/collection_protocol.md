# Vietnamese KWS — Human Speech Data Collection Protocol

> **Document Version:** 1.0.0-PROPOSAL  
> **Campaign Date:** October 13, 2026  
> **Status:** Proposed Specification — Pending Human Approval Before Recruitment  
> **Target Commands:** `TRAI` (trái), `PHAI` (phải), `LEN` (lên), `XUONG` (xuống)  
> **Auxiliary Classes:** `UNKNOWN` (non-command speech & phonetically confusing words), `SILENCE` (ambient background)

---

## 1. Executive Summary & Objective

The objective of this campaign is to transition the Vietnamese Keyword Spotting (KWS) system from a model trained on synthetic parametric speech to a high-accuracy, production-ready system validated on genuine, native Vietnamese human speech.

The Phase 3 audit and forensic reconciliation proved that 100% of the existing target-command samples in active manifests are synthetic acoustic approximations. In particular, Northern Vietnamese `TRAI` recall is bottlenecked at 25.0% due to unmodeled affricate onset burst `/tɕ/` characteristics.

This document specifies a field collection protocol that is:
1. **Ethically sound:** Enforces verified informed consent, cryptographic pseudonymity (no direct personal identifiers collected), and participant withdrawal rights.
2. **Acoustically stratified:** Balances regional dialects (Northern, Central, Southern), genders, age brackets, mobile hardware, acoustic spaces, and speaking styles.
3. **Leakage-proof:** Architected from day one for strict speaker-disjoint and session-disjoint partitioning.
4. **Reproducible:** Tracks immutable provenance, cryptographic checksums, raw audio preservation, and automated validation.

> [!IMPORTANT]
> **Proposal Notice:** All sample sizes, quotas, and participant targets in this document are engineering proposals designed for statistical coverage. No participants will be contacted or recruited until this protocol receives formal human approval.

---

## 2. Participant Recruitment & Informed Consent Protocol

### 2.1 Participant Eligibility Criteria
- **Language Proficiency:** Native speakers of Vietnamese who grew up in or primarily speak one of the three major regional dialect groups:
  - **Northern Dialect (Giọng Bắc):** Hanoi, Hai Phong, Bac Ninh, Nam Dinh, Ninh Binh, Thai Binh, Phu Tho, etc.
  - **Central Dialect (Giọng Trung):** Hue, Da Nang, Quang Nam, Quang Ngai, Nghe An, Ha Tinh, Quang Binh, etc.
  - **Southern Dialect (Giọng Nam):** Ho Chi Minh City, Can Tho, Dong Nai, Binh Duong, Tien Giang, An Giang, etc.
- **Age Requirement:** 18 years of age or older at the time of recording.
- **Vocal Health:** No self-reported severe speech impediments, acute laryngitis, or severe respiratory illness at recording time.

### 2.2 Informed Consent Form & Disclosures
Every participant must be presented with the standard Informed Consent Agreement prior to any recording:

```text
================================================================================
                    BẢN ĐỒNG THUẬN THAM GIA THU ÂM NGHIÊN CỨU
        (VIETNAMESE KEYWORD SPOTTING AUDIO DATA COLLECTION CONSENT)
================================================================================
1. Mục đích: Dữ liệu giọng nói của bạn được sử dụng cho mục đích nghiên cứu học máy,
   huấn luyện và đánh giá hệ thống nhận dạng giọng nói ngoại tuyến (offline KWS)
   trên thiết bị di động (Android và iOS).

2. Bảo vệ danh tính & Quyền riêng tư:
   - KHÔNG thu thập hoặc lưu trữ bất kỳ thông tin nhận dạng cá nhân nào (như họ tên,
     số CCCD/CMND, số điện thoại, địa chỉ email, hay địa chỉ nhà ở).
   - Bạn được gán một mã định danh giả danh ngẫu nhiên (Pseudonymous ID).
   - Chỉ lưu trữ các thuộc tính nhân khẩu học liên quan đến âm học: phương ngữ vùng miền,
     giới tính, và nhóm độ tuổi.

3. Bản quyền & Sử dụng:
   - Dữ liệu âm thanh thô và đã chuẩn hóa được sử dụng phi độc quyền cho mục đích
     nghiên cứu khoa học và phát triển sản phẩm công nghệ.
   - Bạn không chuyển giao quyền nhân thân nhưng đồng ý cấp phép sử dụng âm thanh.

4. Quyền rút lui (Right of Withdrawal):
   - Bạn có toàn quyền yêu cầu hủy bỏ và xóa toàn bộ các bản ghi giọng nói của mình
     bất kỳ lúc nào trước khi tập dữ liệu được đóng băng phát hành chính thức (Release Freeze).
   - Để rút lui, bạn chỉ cần cung cấp mã định danh giả danh đã được cấp khi đăng ký.
================================================================================
```

### 2.3 Consent Verification & Storage
- Consent must be captured digitally with a boolean flag, consent version string (`v1.0-kws-research-commercial`), and UTC timestamp.
- The consent record is stored in a dedicated, isolated ledger: `data/raw/human_recordings_v1/consent_ledger_human.json`.
- Audio ingestion will immediately abort and reject any file whose `speaker_id` does not exist in the ledger with `consent_given: true`.

### 2.4 Participant Withdrawal Procedure
If a participant requests deletion of their data:
1. Operator invokes the dedicated withdrawal tool:
   `python src/acquisition/revoke_speaker_consent.py --speaker-id <SPK_ID> --reason "Participant withdrawal request"`
2. The tool:
   - Flips `consent_given` to `false` and records `revocation_timestamp`.
   - Moves all raw and standardized files for that speaker to `data/raw/quarantined/withdrawn_speakers/<SPK_ID>/`.
   - Marks all manifest rows for that speaker as `STATUS: REVOKED_WITHDRAWN`.
   - Generates an audit trail without modifying the frozen benchmark test set.

---

## 3. Pseudonymous Speaker ID Strategy & PII Defense

To strictly guarantee that no personal identity is exposed:
1. **No Sequential Enrollment Counters:** Sequential counters (e.g., `SPK_001`, `SPK_002`) leak enrollment order, session dates, and operator logs.
2. **Structured Nonce IDs:** Speaker IDs follow a structured pseudonymous schema combining regional dialect and a cryptographic hex nonce:
   $$\text{Speaker ID} = \text{SPK\_H\_} + \{\text{N} \mid \text{C} \mid \text{S}\} + \text{\_} + \text{HexNonce}_{6}$$
   - Examples: `SPK_H_N_8f2a1b`, `SPK_H_C_4e7c09`, `SPK_H_S_a1d3f5`
3. **Allowed Demographic Metadata Fields (Strictly Audited):**
   - `dialect`: `Northern` | `Central` | `Southern`
   - `gender`: `FEMALE` | `MALE` | `OTHER`
   - `age_bracket`: `18-30` | `31-50` | `51+`
   - `native_province`: Optional general province (e.g., `Hanoi`, `Da Nang`, `Can Tho`) without sub-district or address.
   - `primary_device`: Model name (e.g., `iPhone 13`, `Samsung Galaxy S22`).
   - `primary_environment`: Acoustic room label (e.g., `quiet_room`, `living_room_reverberant`).

---

## 4. Stratification Targets & Proposed Minimum Collection Size

### 4.1 Target Cohort Sizing (Proposed)
To achieve robust statistical power and ensure reliable 70% / 15% / 15% speaker-disjoint splitting:

| Dialect Region | Target Speakers | Proportion (%) | Primary Acoustic Rationale |
| :--- | :---: | :---: | :--- |
| **Northern (Giọng Bắc)** | **50** | **41.7%** | **Top priority:** Native affricate burst `/tɕ/` (tone sắc) vs. fricative `/f/` (tone hỏi) resolution. |
| **Central (Giọng Trung)** | **30** | **25.0%** | Heavy glottal stop / creaky voice (`hỏi`/`ngã` neutralization), central vowels. |
| **Southern (Giọng Nam)** | **40** | **33.3%** | Retroflex onset `/ʈ/`, final consonant mergers (`-t/-k`, `-n/-ng`), high mobile user base. |
| **TOTAL** | **120** | **100.0%** | Minimum 100 required; 120 provides a 20% safety margin for potential quality rejections. |

> [!NOTE]
> **Cohort Arithmetic vs. Stratum Granularity Limitation:**  
> The 70% / 15% / 15% split divides the overall 120-participant cohort into clean integer subsets (84 Train, 18 Val, 18 Test). At the regional level, integer rounding is manageable (Northern: 35/7/8; Central: 21/4/5; Southern: 28/6/6). However, for fine-grained demographic bins (e.g. Central males aged 51+ with only ~2 expected contributors), it is mathematically impossible to populate all three disjoint partitions without empty bins. The project explicitly acknowledges this structural rounding limitation and does not claim demographic balance across all three partitions for small strata without empirical split diagnostic verification.

### 4.2 Contributor Diversity Breakdown
- **Gender Balance:** Target 45%–55% female, 45%–55% male across all regional tiers.
- **Age Distribution:**
  - `18-30`: ~45% (primary mobile user demographic)
  - `31-50`: ~40% (mature vocal tract characteristics)
  - `51+`: ~15% (older speakers with distinct fundamental frequency $F_0$ and articulation pacing)

---

## 5. Command Collection Script & Speaking Dynamics

Under the core keyword protocol, each participant contributes 40 target command takes across 4 sessions or blocks (10 takes per command). In the full multiclass protocol (Option B, detailed in Sections 6–9), an additional 25 `UNKNOWN` takes and 4 `SILENCE` takes are collected, totaling 69 takes per participant. Resolving whether to collect 40 takes (Option A) or 69 takes (Option B) is an open executive decision for the Project Sponsor / PI.

### 5.1 Target Commands & Phonetic Prompts
| Command Label | Vietnamese Word | Target Class ID | IPA Transcription & Dialect Details | Prompt Instruction to Speaker |
| :--- | :---: | :---: | :--- | :--- |
| **`TRAI`** | `trái` | 0 | Northern: `/tɕaːj˧˥/` (affricate onset, rising tone sắc)<br/>Southern: `/ʈaːj˧˥/` (retroflex stop onset) | Nói dứt khoát từ "TRÁI". Nhấn rõ âm đầu (tr-). |
| **`PHAI`** | `phải` | 1 | All regions: `/faːj˧˩˨/` (labiodental fricative onset, dipping-rising tone hỏi) | Nói rõ từ "PHẢI". Phát âm rõ dấu hỏi (thanh hỏi). |
| **`LEN`** | `lên` | 2 | All regions: `/leːn˧˧/` (lateral onset, mid level tone ngang) | Nói tự nhiên từ "LÊN". Âm đầu l- rõ ràng. |
| **`XUONG`** | `xuống` | 3 | All regions: `/sʰuəŋ˧˥/` (sibilant onset, rising tone sắc, labialized coda) | Nói dứt khoát từ "XUỐNG". Âm đầu x- và vần -uống. |

### 5.2 Speaking Style Stratification (10 Takes per Command per Speaker)
For each command, the 10 takes must be distributed across speaking dynamics:
1. **Normal Delivery (Takes 1–4):** Natural conversational pacing and typical vocal effort.
2. **Fast Paced (Takes 5–6):** Rapid, urgent command execution (~0.30–0.45 s duration).
3. **Deliberate / Slow (Takes 7–8):** Slower, clearly articulated delivery (~0.55–0.80 s duration).
4. **Quiet / Soft (Take 9):** Low vocal effort, soft voice (near whisper-adjacent, $RMS \approx -35$ dBFS).
5. **Loud / Projected (Take 10):** High vocal effort, authoritative command across a room ($RMS \approx -16$ dBFS).

---

## 6. Negative Speech Collection Protocol (`UNKNOWN` Class 4)

To prevent false activations in production environments, the collection campaign must gather genuine negative speech from the same speakers and recording devices:

### 6.1 Phonetically Confusing Hard Negatives (Near-Rhymes & Homophones)
Each participant records 1 take of the following 16 phonetically confusing words:
- **Confusing with `TRAI`:** *trại* (tone nặng), *chải* (onset ch-, tone hỏi), *trán* (coda -n), *tra* (open vowel).
- **Confusing with `PHAI`:** *phai* (tone ngang), *phái* (tone sắc), *phạt* (stop coda -t), *khoai* (onset kh-).
- **Confusing with `LEN`:** *lén* (tone sắc), *lẹ* (tone nặng), *đen* (onset đ-), *quen* (onset qu-).
- **Confusing with `XUONG`:** *xuồng* (tone huyền), *xương* (rhyme -ương), *xoong* (rhyme -oong), *chuông* (onset ch-).

### 6.2 Carrier Sentence Embeddings (Embedded Commands)
To train the model to reject commands embedded inside continuous conversational speech, each participant reads 4 sentences:
1. *"Bên **trái** có ngã tư."* (Command `TRAI` embedded — MUST be classified as `UNKNOWN`).
2. *"Anh **phải** kiểm tra lại."* (Command `PHAI` embedded — MUST be classified as `UNKNOWN`).
3. *"Chúng ta đi **lên** gác."* (Command `LEN` embedded — MUST be classified as `UNKNOWN`).
4. *"Bác ấy vừa đi **xuống** nhà."* (Command `XUONG` embedded — MUST be classified as `UNKNOWN`).

### 6.3 Open-Vocabulary Conversational Speech
- 5 short sentences read from a conversational transcript (e.g., weather query, time inquiry, greeting).
- Total `UNKNOWN` takes per speaker: 16 confusing words + 4 carrier sentences + 5 open sentences = **25 takes**.

---

## 7. Acoustic Environment & Background Noise Protocol (`SILENCE` Class 5)

### 7.1 True Background Noise (Non-Zero Silence)
Participants must record acoustic room tones without speech across their natural recording environments:
1. **Quiet Room Tone (Take 1):** Closed bedroom or private office with HVAC off ($SNR > 30$ dB).
2. **Reverberant Room Tone (Take 2):** Tiled living room, kitchen, or corridor (moderate reverberation).
3. **Ambient Domestic Noise (Take 3):** Room with running ceiling/desk fan or air conditioner hum.
4. **Mechanical / Handling Noise (Take 4):** Gentle phone handling and surface placement vibration.

Each silent take is recorded for 1.6 seconds. Total `SILENCE` takes per participant session: **4 takes**.

---

## 8. Hardware & Acoustic Capture Protocol

### 8.1 Device Form Factors
Recordings must span actual mobile devices used in the Vietnamese market:
1. **iOS Tier:** iPhone 11, iPhone 12/13, iPhone 14/15 (hardware beamforming array, iOS AudioUnit AGC).
2. **Android High-Tier:** Samsung Galaxy S21–S24, Google Pixel 6–8.
3. **Android Mid-Tier:** Xiaomi Redmi Note series, Oppo Reno, Vivo Y-series (hardware mic with aggressive noise suppression).
4. **Earbuds / Headset:** AirPods, Galaxy Buds, wired 3.5mm/USB-C earphone mic.

### 8.2 Microphone Distance & Angle Stratification
- **Close-Talk:** 15–20 cm (handheld smartphone position, on-axis 0°).
- **Desk Distance:** 50–70 cm (phone lying on desk during speakerphone call).
- **Far-Field:** 1.5–2.0 m (ambient smart home control position).

---

## 9. Budget Summary, Storage Footprint & Partitioning Boundary (Proposal)

### 9.1 Comparative Cohort Yield & Session Duration

| Dimension | Option A: Target Keywords Only | Option B: Full Multiclass Protocol |
| :--- | :---: | :---: |
| **Takes per Contributor** | **40 takes** (10 per command: `TRAI`, `PHAI`, `LEN`, `XUONG`) | **69 takes** (40 commands + 25 `UNKNOWN` + 4 `SILENCE`) |
| **Total Campaign Yield (120 Contributors)** | **4,800 recordings** | **8,280 recordings** |
| **Estimated Participant Session Time** | **12–15 minutes** | **25–35 minutes** |
| **Total Audio Duration (at ~1.6s capture window)** | ~2.13 hours raw capture | ~3.68 hours raw capture |
| **Partition Allocation (70% / 15% / 15%)** | - Train (84 spk): **3,360 takes**<br>- Val (18 spk): **720 takes** (180 / cmd)<br>- Test (18 spk): **720 takes** (180 / cmd) | - Train (84 spk): **5,796 takes**<br>- Val (18 spk): **1,242 takes** (180/cmd + 450 UNK + 72 SIL)<br>- Test (18 spk): **1,242 takes** (180/cmd + 450 UNK + 72 SIL) |

### 9.2 Reproducible Storage Footprint Calculations

Both the original raw capture and the standardized training audio are permanently retained on disk to ensure forensic reproducibility and future re-preprocessing capability ([`src/acquisition/ingest_field_recordings.py#L559-L568`](file:///D:/trainidentificationaudio/src/acquisition/ingest_field_recordings.py#L559-L568)):

1. **Raw Capture Audio Format:**
   - **Sample Rate:** 44.1 kHz (`44,100 Hz`)
   - **Channels:** 2 (Stereo)
   - **Bit Depth:** 16-bit PCM (`2 bytes per channel per sample`)
   - **Source Evidence:** Implemented in `AudioRecorder` ([`src/acquisition/record_command_cli.py#L49-L55`](file:///D:/trainidentificationaudio/src/acquisition/record_command_cli.py#L49-L55)) via Windows MCI (`winmm.dll`).
   - **Byte Rate:** $44,100 \times 2 \times 2 = 176,400\text{ bytes/s}$.
   - **Assumed Duration:** 1.6 seconds fixed acquisition buffer ([`record_command_cli.py#L68`](file:///D:/trainidentificationaudio/src/acquisition/record_command_cli.py#L68)).
   - **File Size per Take:** $176,400 \times 1.6 + 44\text{ (WAV header)} = 282,284\text{ bytes} \approx 275.67\text{ KB} \approx 0.282\text{ MB}$.

2. **Standardized Processed Audio Format:**
   - **Sample Rate:** 16.0 kHz (`16,000 Hz`)
   - **Channels:** 1 (Mono)
   - **Bit Depth:** 16-bit PCM (`PCM_16`, `2 bytes per sample`)
   - **Source Evidence:** Implemented in `normalize_to_target_window()` and `sf.write(..., subtype="PCM_16")` ([`src/acquisition/ingest_field_recordings.py#L564-L566`](file:///D:/trainidentificationaudio/src/acquisition/ingest_field_recordings.py#L564-L566)).
   - **Byte Rate:** $16,000 \times 1 \times 2 = 32,000\text{ bytes/s}$.
   - **Standardized Duration:** 1.0 second fixed window (`TARGET_SAMPLES = 16000`).
   - **File Size per Take:** $32,000 \times 1.0 + 44\text{ (WAV header)} = 32,044\text{ bytes} \approx 31.29\text{ KB} \approx 0.032\text{ MB}$.

3. **Storage Engineering Estimates per Collection Option:**

| Metric / Layer | Option A (4,800 Takes) | Option B (8,280 Takes) | Derivation / Notes |
| :--- | :---: | :---: | :--- |
| **Raw Capture Storage** | **1.355 GB** ($1,292.2\text{ MiB}$) | **2.337 GB** ($2,229.0\text{ MiB}$) | $N \times 282,284\text{ bytes}$. Retained permanently in `data/raw/human_recordings_v1/<SPK>/<SES>/`. |
| **Standardized Storage** | **0.154 GB** ($146.7\text{ MiB}$) | **0.265 GB** ($253.0\text{ MiB}$) | $N \times 32,044\text{ bytes}$. Retained permanently in `data/processed/human_recordings_v1/<CMD>/`. |
| **Primary Audio Subtotal** | **1.509 GB** ($1,438.9\text{ MiB}$) | **2.603 GB** ($2,482.1\text{ MiB}$) | Sum of raw capture and standardized audio files. *(Provisional engineering estimate).* |
| **Derived Feature Caches (`.npy`)** | **0.155 GB** ($148.0\text{ MiB}$) | **0.268 GB** ($255.2\text{ MiB}$) | $N \times (80 \times 101 \times 4\text{ bytes} = 32,320\text{ bytes})$. Intermediate mel-spectrogram arrays. |
| **Cold Backup Archive (Raw + Standardized Audio, `data/backups/`)** | **0.754 GB** ($719.5\text{ MiB}$) | **1.301 GB** ($1,241.0\text{ MiB}$) | Compressed `.tar.gz` archive of raw and standardized audio: $50\% \times (\text{Raw} + \text{Standardized})$ ($50\% \times 1.509\text{ GB} \approx 0.754\text{ GB}$ Option A; $50\% \times 2.603\text{ GB} \approx 1.301\text{ GB}$ Option B). 50% compression ratio is an unmeasured planning assumption. |
| **Temp Files, Quarantine & Manifests** | **0.150 GB** ($143.1\text{ MiB}$) | **0.250 GB** ($238.4\text{ MiB}$) | Provisional headroom for session temp WAVs, operator re-recordings, defect quarantine, and logs. |
| **Total Working Footprint (Sum of Components)** | **~2.57 GB** ($2,449.3\text{ MiB}$) | **~4.42 GB** ($4,216.7\text{ MiB}$) | Explicit sum: Audio ($1.509 / 2.603$) + Caches ($0.155 / 0.268$) + Backups ($0.754 / 1.301$) + Temp ($0.150 / 0.250$). |

> [!NOTE]
> **Measurement & Buffer Notice:** Storage figures are derived from software buffer parameters (1.6s nominal raw buffer, 1.0s standardized window). 1.6 seconds is a nominal capture buffer configuration in `record_command_cli.py`, not an empirical guarantee that every saved file is exactly 1.6s. Ingestion accepts any file with duration in $[0.25\text{s}, 2.50\text{s}]$, and actual physical durations will vary. Cold backup estimates assume a 50% compression ratio across raw plus standardized audio as an unmeasured planning assumption (empirical compression efficiency has not been measured on production speech). Temporary and quarantine headroom figures (~0.150 GB for Option A, ~0.250 GB for Option B) remain explicitly provisional. If alternatively budgeted using a consolidated $1.5\times$ multiplier on primary audio alone (without separate line-item breakdown for caches and backups), the working allowances are ~2.26 GB (Option A) and ~3.90 GB (Option B). When all operational components are explicitly itemized and summed, the comprehensive working footprint estimate is **~2.57 GB** for Option A and **~4.42 GB** for Option B.

### 9.3 Offline Speaker-Independent Partitioning Boundary

- **Offline Operation:** Partitioning is strictly an offline batch process executed via [`src/acquisition/campaign_validator.py::generate_stratified_disjoint_splits()`](file:///D:/trainidentificationaudio/src/acquisition/campaign_validator.py#L371-L560). Ingestion CLIs append unpartitioned records to the manifest without split tags.
- **Downstream Pipeline Mandate:** Downstream training and evaluation scripts must explicitly load and validate the disjoint partition output. Naive row-level sampling (`df.sample()` or `sklearn.model_selection.train_test_split()`) is **strictly prohibited**, as it introduces acute speaker leakage by distributing identical speaker voices across train, validation, and test sets.
- **Lineage & Augmentation Cohesion:** All augmented, pitch-shifted, or resampled variants derived from utterance $u$ must strictly inherit the split assignment of $u$. Cross-partition augmentation leakage is flagged as a fatal error by `validate_speaker_disjointness()`.
