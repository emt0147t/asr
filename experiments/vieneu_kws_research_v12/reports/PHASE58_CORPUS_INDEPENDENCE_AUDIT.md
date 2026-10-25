# PHASE 58: CONFIRMATORY CORPUS INDEPENDENCE AND PROVENANCE AUDIT

**Repository:** `D:\trainidentificationaudio`  
**Phase:** Phase 58 (Confirmatory Corpus Independence Verification)  
**Creation Timestamp:** `2026-10-25T15:13:33+00:00`  
**Speech Generator:** Piper TTS (`vi_VN-vais1000-medium.onnx`, IEEE DataPort VAIS-1000 Corpus)  
**Environmental Noise Background:** Karol Piczak ESC-50 Fold 3 Indoor Tracks  
**Manifest Ledger:** [`PHASE58_SHA256_MANIFEST.json`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v12/reports/PHASE58_SHA256_MANIFEST.json)  
**Operating Threshold:** $\theta_{\text{cmd}} = 0.20$ (Strictly Locked)  
**Default Canonical Model:** Legacy $v11$ (`3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2`)

---

## Mandatory Governance Invariants

> [!CAUTION] MANDATORY RESEARCH STATUSES
> 1. `EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`
> 2. `NOT_VALIDATED_ON_NATURAL_SPEECH`
> 3. `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`
> 4. `NOT_VALIDATED_PENDING_DATA`
>
> Zero human speech was collected. Zero physical microphone testing was conducted. No model retrained or capacity modified. Checkpoints for $v11$, $E1$, Phase 28 $D_2\_H1$, and Phase 26 $D_0\_H1\_seed44$ remain strictly frozen and unaltered.
> Historical evaluation corpora (Phase 21, Phase 27, Phase 28 `edge_tts_eval_v2`, Phase 43 `mms_tts`, Phase 51 `gTTS`) are permanently designated as observed datasets and were NOT reused for confirmation.

---

## 1. Executive Summary & Independence Doctrine

Phase 58 establishes a fifth independent acoustic evaluation corpus, resolving the risk of circular model evaluation. The corpus guarantees complete separation from all historical training distributions, validation splits, and prior test engines:

```
Historical TTS Generators (Quarantined & Excluded):
1. VieNeu-TTS (VIVOS Dataset, Local VITS)           -> Used in Training (v11, E1, D0, D2, D3), Phase 21 Eval
2. Microsoft Edge-TTS (Azure Neural Speech)          -> Used in Training (D2, D3), Phase 27 & 28-31 Eval
3. Meta MMS-TTS-vie (FCBH New Testament Bible)       -> Used in Phase 41-43 Eval (Observed)
4. Google Translate TTS (Google DeepMind Neural)     -> Used in Phase 49-51 Eval (Observed)

Authorized Fifth Confirmatory Generator:
5. Piper VAIS-1000 (IEEE DataPort VAIS-1000 Corpus)  -> ZERO PRIOR REPOSITORY USAGE (Genuinely Independent)
```

---

## 2. Multi-Level Independence Audit

### 2.1 Acoustic Speech Source Independence:
- **Corpus Title:** VAIS-1000 Vietnamese Speech Synthesis Corpus (IEEE DataPort).
- **Acoustic Characteristics:** Native Northern Vietnamese female voice recorded in professional studio conditions in Hanoi, distinct from VIVOS (South/North mixed amateur), Azure Neural (commercial broadcast), FCBH Bible (Southern retroflex dialect), and Google Translate (standard national cloud engine).
- **Synthesis Architecture:** Local ONNX export of VITS neural acoustic model (`vi_VN-vais1000-medium.onnx`).
- **Determinism:** Evaluated with `SynthesisConfig(noise_scale=0.0, noise_w_scale=0.0)`, guaranteeing bit-for-bit mathematical reproducibility ($\max |\Delta| = 0.000000$).
- **Sample Rate Conversion:** Native $22,050\text{ Hz}$ cleanly resampled to $16,000\text{ Hz}$ mono 16-bit PCM using polyphase sinc interpolation.

### 2.2 Environmental Noise Independence (ESC-50 Fold 3):
- **Source:** Official Karol Piczak ESC-50 dataset (Fold 3).
- **Categories:** 8 indoor tracks from categories 36 (`vacuum_cleaner`), 32 (`keyboard_typing`), 35 (`washing_machine`), and 38 (`clock_tick`):
  `3-152020-A-36.wav`, `3-159346-A-36.wav`, `3-125548-A-32.wav`, `3-132830-A-32.wav`, `3-135469-A-35.wav`, `3-151269-A-35.wav`, `3-142593-A-38.wav`, `3-143933-A-38.wav`.
- **Deduplication Audit:** Cross-correlation analysis against Fold 1, Fold 2, and all historical `mic-noise` tracks confirmed:
  $$\max r = \mathbf{0.0947} < 0.15 \quad (\text{Strictly Non-Overlapping})$$

---

## 3. Disaggregated Corpus Tracks & Manifest Structure

### 3.1 Track A: Isolated 4-Command Classification ($N = 420$ primary + $60$ tone probes)
- **Manifest:** [`PHASE58_CONFIRMATORY_MANIFEST.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v12/manifests/PHASE58_CONFIRMATORY_MANIFEST.csv)
- **Structure:**
  - **160 Target Commands:** 4 command classes $\times$ 10 speaking rate variations ($0.85$ to $1.30$) $\times$ 4 SNR tiers (`clean`, $15\text{ dB}$, $10\text{ dB}$, $5\text{ dB}$).
    - Exactly $40$ clips per command class (`TRAI: 40`, `PHAI: 40`, `LEN: 40`, `XUONG: 40`).
    - Eliminates the previous $N=1$ instance limitation by deploying $10$ independent clean synthesis instances per command!
  - **160 Phonetic Hard Negatives:** 16 core phonetically close words $\times$ multiple length scales $\times$ 4 SNR tiers.
  - **100 General Conversational OOVs:** 10 common vocabulary items $\times$ multiple length scales $\times$ 4 SNR tiers.
- **Diagnostic Tone Probes:** [`PHASE58_TONE_PROBE_MANIFEST.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v12/manifests/PHASE58_TONE_PROBE_MANIFEST.csv)
  - 6 tone words (`ma, má, mà, mả, mã, mạ`) $\times$ 2.5 configs $\times$ 4 SNR tiers = **60 clips**.

### 3.2 Track B: Continuous Contextual Streaming ($50$ Sequences, $58$ Ground-Truth Events)
- **Manifest:** [`PHASE58_STREAMING_MANIFEST.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v12/manifests/PHASE58_STREAMING_MANIFEST.csv)
- **Event Ledger:** [`PHASE58_EVENT_LEDGER.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v12/manifests/PHASE58_EVENT_LEDGER.csv) / [`PHASE58_EVENT_LEDGER.json`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v12/manifests/PHASE58_EVENT_LEDGER.json)
- **Sequence Breakdown:**
  - $10$ pure silence/noise sequences ($4.5\text{ s} - 6.0\text{ s}$ each).
  - $10$ pure conversational OOV sequences ($4.5\text{ s} - 6.0\text{ s}$ each).
  - $30$ contextual command sequences with natural carrier phrases (`rẽ trái, sang trái, cua trái, bẻ lái phải, đi lên, tiến lên, đi xuống, lùi xuống`).

---

## 4. Cryptographic Checksum Ledger of Locked Manifests

| Manifest Artifact | File Path | Total Items | Verified SHA-256 Checksum |
|:------------------|:----------|:-----------:|:--------------------------|
| **Track A Isolated Manifest** | `manifests/PHASE58_CONFIRMATORY_MANIFEST.csv` | 420 clips | `d03223126f30d07559aa2b9260a927d14d232a9c379a632ea704434e3fb4b7b3` |
| **Track A Tone Probe Manifest** | `manifests/PHASE58_TONE_PROBE_MANIFEST.csv` | 60 clips | `bb621ef9a8cf6be0702c2e0c1f60f697486e1a7b453a5c5443fa7522d0590c68` |
| **Track B Streaming Manifest** | `manifests/PHASE58_STREAMING_MANIFEST.csv` | 50 sequences | `3a18a0058ec08b1a80894fe83f988fe2650085a6cfec187a55427d10c2c31e78` |
| **Track B Event Ledger (CSV)** | `manifests/PHASE58_EVENT_LEDGER.csv` | 58 events | `e31464ae3eb28ec4b9679261a8bbdd8178d46a89ca740ee4d34eb5a53828383c` |
| **Track B Event Ledger (JSON)** | `manifests/PHASE58_EVENT_LEDGER.json` | 50 sequences | `42247fc2b11e9f4007a8fe80a6b5a3eb1e095be9d02636a0487042a6be85fb82` |

---

## 5. Audit Gate Determination

All source independence, noise deduplication, manifest freezing, and cryptographic ledger verifications are complete.

**Corpus Audit Gate:**
$$\mathbf{PASS\_SOURCE\_AND\_CORPUS\_FROZEN}$$
