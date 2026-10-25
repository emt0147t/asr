# PHASE 50: CONFIRMATORY EVALUATION PROTOCOL & PRE-REGISTRATION

**Repository:** `D:\trainidentificationaudio`  
**Phase:** Phase 50 (Confirmatory Protocol Registration)  
**Creation Timestamp:** `2026-10-25T14:28:57.942029+00:00`  
**Manifest Freeze Timestamp:** `2026-10-25T14:29:35.634648+00:00`  
**Status:** **LOCKED & PRE-REGISTERED (PRIOR TO CONFIRMATORY INFERENCE)**

---

## 1. Governance & Protocol Invariants

1. **Default Model Status:** Legacy $v11$ remains the canonical default model.
2. **Model Integrity:** Checkpoints for $v11$, $E1$, Phase 28 $D_2\_H1$, and Phase 26 $D_0\_H1\_seed44$ are strictly read-only and immutable.
3. **Threshold Lock:** Decision threshold $\theta_{\text{cmd}} = 0.20$ is strictly locked.
4. **Argmax Policy:** REJECT-inclusive 5-class argmax:
   $$\hat{c} = \begin{cases} \arg\max_{i \in \{0,1,2,3\}} P(c_i), & \text{if } \max_{i \in \{0,1,2,3\}} P(c_i) \ge 0.20 \\ 4 \text{ (REJECT)}, & \text{otherwise} \end{cases}$$
5. **VAD Specification:** `SpeechBandRatioVAD` with band $[300, 3400]\text{ Hz}$, energy threshold $-46.0\text{ dBFS}$, ratio threshold $0.32$.
6. **Frontend:** 40-channel Log-Mel filterbank ($16\text{ kHz}$, frame $25\text{ ms}$, hop $10\text{ ms}$, $512$ FFT).
7. **Capacity:** Fixed at $138,469$ parameters (`CorrectedVietnameseKWSNet`, `DEFER_E2_CAPACITY`).
8. **Deployment Claim Restrictions:**
   - `EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`
   - `NOT_VALIDATED_ON_NATURAL_SPEECH`
   - `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`
   - `NOT_VALIDATED_PENDING_DATA`

---

## 2. Independent Generators & Noise Assets

### 2.1 Acoustic Speech Generator
- **Engine:** Google Translate Vietnamese Text-to-Speech (`gTTS 2.5.4`).
- **Independence:** Completely independent from VieNeu-TTS, Microsoft Edge-TTS, and Meta MMS-TTS. Zero historical usage in training, validation, or candidate selection.
- **Language:** Vietnamese (`vi`). Standard Northern/National Vietnamese phonology with clear rising/falling tonal contours.

### 2.2 Environmental Noise Assets
- **Dataset:** Karol Piczak ESC-50 Fold 2 indoor recordings (`vacuum_cleaner`, `keyboard_typing`, `washing_machine`, `clock_tick`).
- **Deduplication:** Fully verified non-overlapping against all 40 historical `mic-noise` tracks and Fold 1.

---

## 3. Two Pre-Registered Evaluation Tracks

### Track A — Isolated 4-Command Classification
- **Manifest:** `PHASE50_LOCKED_MANIFEST.csv` (240 primary clips: 32 commands + 128 phonetic OOVs + 80 general OOVs).
- **Tone Probes:** `PHASE50_TONE_PROBE_MANIFEST.csv` (48 diagnostic clips).
- **Cluster Unit:** Synthesis Instance (clean waveform), grouping all 4 derivative SNR tiers.
- **Explicit Limitation:** Multiple synthesis instances sharing one command text do not establish generalization to unseen target words.

### Track B — Keyword Detection in Context / Continuous Audio
- **Manifest:** `PHASE50_STREAMING_MANIFEST.csv` (50 sequences) & `PHASE50_EVENT_LEDGER.csv` (230 events).
- **Carrier Phrases:** Natural contextual phrases (`rẽ trái`, `sang trái`, `rẽ phải`, `sang phải`, `đi lên`, `tiến lên`, `đi xuống`, `lùi xuống`).
- **Matching Rule:** $\pm 0.50\text{ s}$ window around event center.
- **Independence Policy:** Track A classification metrics and Track B event metrics are reported strictly disaggregated.

---

## 4. Manifest Cryptographic Ledger

| Manifest Artifact | File Path | Total Items | SHA-256 Checksum |
|:------------------|:----------|:-----------:|:-----------------|
| **Track A Isolated Manifest** | `experiments/vieneu_kws_research_v12/manifests/PHASE50_LOCKED_MANIFEST.csv` | 240 clips | `b644a7e71badae8394b0e693fa45e1fdafb0a346b939c456b5f33e2941c80297` |
| **Track A Tone Probe Manifest** | `experiments/vieneu_kws_research_v12/manifests/PHASE50_TONE_PROBE_MANIFEST.csv` | 48 clips | `341aec1843a99b6d59506d64f06b2d03aa971b119acceb0c61195e9a5b89a109` |
| **Track B Streaming Manifest** | `experiments/vieneu_kws_research_v12/manifests/PHASE50_STREAMING_MANIFEST.csv` | 50 sequences | `d7f712ecc1ac7c37923a1ffca95f4474b5a863642b13316038113c0d56b97aed` |
| **Track B Event Ledger (CSV)** | `experiments/vieneu_kws_research_v12/manifests/PHASE50_EVENT_LEDGER.csv` | 230 events | `b36e10c84abcc42b8d2bb050b6d00c31a22050894539fde96fffaa5fa93b7026` |
| **Track B Event Ledger (JSON)** | `experiments/vieneu_kws_research_v12/manifests/PHASE50_EVENT_LEDGER.json` | 50 sequences | `ca22dc9a100b73242b3868f20d94e4d8c3df13025d43e193f60cf7755a8cf9cc` |

---

## 5. Pre-Registered Confirmatory Candidate Profiles

| Candidate Model | Pre-Registered Profile | Specific Engineering Pass Criteria |
|:----------------|:-----------------------|:------------------------------------|
| **Phase 28 D2_H1 Candidate** | **Low False-Alarm / Standby Profile** | 1. Silence $\text{FA/min} \le 1.0\text{ FA/min}$ (UCB95 $\le 2.0$).<br>2. Conversational OOV $\text{FA/min} \le 6.0\text{ FA/min}$.<br>3. Overall Isolated OOV $\text{FAR} \le 20.0\%$.<br>4. Track A 4-Command Macro-F1 $\ge 50.0\%$. |
| **Phase 26 Candidate (seed44)** | **Balanced Sensitivity Profile** | 1. Track A 4-Command Macro-F1 $\ge 60.0\%$.<br>2. Track B Streaming Event F1 $\ge 55.0\%$.<br>3. Silence $\text{FA/min} \le 3.0\text{ FA/min}$.<br>4. Overall Isolated OOV $\text{FAR} \le 25.0\%$. |
