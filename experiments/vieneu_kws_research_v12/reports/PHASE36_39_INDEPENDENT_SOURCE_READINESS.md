# PHASE 36–39: INDEPENDENT SOURCE READINESS AUDIT REPORT

**Repository:** `D:\trainidentificationaudio`  
**Phase:** Phase 38 (Independent Source Readiness)  
**Document Status:** Source Audit & Dependency Assessment  
**Operational Status:** `BLOCKED_PENDING_INDEPENDENT_TTS_SOURCE`  
**Governing Architectural Invariant:** Fixed E1 capacity (138,469 parameters, `DEFER_E2_CAPACITY`)  
**Operating Threshold:** $\theta_{\text{cmd}} = 0.20$ (Strictly Locked)

---

## 1. Objective and Source Eligibility Criteria

The objective of Phase 38 is to investigate, verify, and document candidate independent speech synthesis sources and unexposed acoustic environments to enable a genuinely independent evaluation of frozen KWS models ($v11$, $E1$, and Phase 28 Candidate $D_2\_H1\_seed42$).

### Mandatory Eligibility Criteria:
1. **Engine Independence:** Distinct from VieNeu-TTS (V3 FastSpeech2-like) and Edge-TTS (Microsoft Azure Neural TTS).
2. **Voice Independence:** Voices and generator configurations not exposed in $D_0$, $D_1$, or $D_2$ training, nor in $v11$ or $E1$ development/validation.
3. **Lexical Independence:** Source texts and target/OOV lexemes verified against all historical training and model-selection data.
4. **Noise Provenance:** Additive noise assets whose provenance is traceable and whose underlying audio tracks have **not** been used in prior test/evaluation data.
5. **Licensing & Reproducibility:** Documented open-source license or authorized usage policy with deterministic generation settings.

---

## 2. Investigation of Candidate Independent TTS Sources

| Candidate Source | Engine Architecture | Tested Version | Verified Vietnamese Voices | Licensing / Terms | Workspace Access Status | Independence Criteria Met? | Identified Limitations / Blockers |
|:-----------------|:-------------------:|:--------------:|:--------------------------:|:------------------:|:-----------------------:|:--------------------------:|:----------------------------------|
| **Piper TTS (Local ONNX)** | VITS (End-to-End Variational Inference with Adversarial Learning) | `piper-tts` 1.8.0 | `vi_VN-vivos-x_low` (65 speakers)<br>`vi_VN-25hours_single-low`<br>`vi_VN-vais1000-medium` | MIT / VIVOS (CC BY-SA 4.0) | **INSTALLED & VERIFIED** in `.venv` | **PARTIAL** | Phonemizer (`espeak-ng`) logs tone map omission warnings (`Missing phoneme from id map: 6, \u032a, 2`), causing audible tonal distortion on Vietnamese keywords. |
| **Google Cloud Text-to-Speech** | WaveNet / Neural2 (Deep Neural Network) | Cloud API v1 | `vi-VN-Standard-A/B/C/D`<br>`vi-VN-Wavenet-A/B/C/D`<br>`vi-VN-Neural2-A/D` | Commercial Cloud Service | **NOT AVAILABLE** | **NO** | No active Google Cloud project credentials or service account key configured in workspace. |
| **FPT.AI Speech Synthesis** | FastSpeech / Tacotron2-based Neural TTS | Cloud API v4 | BanMai, ThuMinh, MyAn, GiaNhi, LeMinh | Commercial Cloud Service | **NOT AVAILABLE** | **NO** | No API key credentials configured in workspace. |
| **VieNeu-TTS (Local)** | Acoustic Backbone + ONNX Vocoder | Local V3 Nano/Turbo | 17 Vietnamese voices | Proprietary Research | Installed | **NO (Disqualified)** | Historical baseline training distribution ($D_0$). Cannot serve as an independent evaluation source. |
| **Edge-TTS (Local/API)** | Microsoft Azure Neural TTS | `edge-tts` 7.3.1 | `vi-VN-HoaiMyNeural`<br>`vi-VN-NamMinhNeural` | Microsoft Terms of Service | Installed | **NO (Disqualified)** | Exposed in $D_2$ training (544 clips, 10% dose) and Phase 28 test corpus. Cannot serve as an independent evaluation source. |

---

## 3. Investigation of Environmental Noise Provenance

### Forensic Finding on Repository Noise Assets:
A comprehensive scan of all 187 manifest files across the repository established the following:
- The workspace contains only one environmental noise collection: `data/raw/noise/extracted/noise/mic-noise` (40 audio tracks, numbered 0 to 39).
- **Tracks 0–35:** Heavily exposed in training manifests (`dataset_v12_manifest.csv`, `train_manifest.csv`, `D0_train_v1.csv`, `D2_matched_train_v1.csv`).
- **Tracks 34–39:** Heavily exposed in historical evaluation manifests:
  - `test_manifest.csv` (v12 locked test): 883 occurrences.
  - `edge_tts_eval_v2_isolated_test_manifest.csv` (Phase 28 test): 360 occurrences.
  - `synthetic_eval_test_manifest.csv` (Phase 21 test): 32 occurrences.
  - `vad_test_corpus_manifest.csv`: 48 occurrences.
  - `streaming_eval_corpus_manifest.csv`: 30 occurrences.
- **Critical Blocker:** There is **zero unexposed noise audio** in the repository. Evaluating models using tracks 0–39 would violate the mandatory requirement that underlying tracks must not have been used in prior evaluation data.

---

## 4. Formal Dependency Blocker Determination

To avoid pseudo-independent evaluation (e.g., applying superficial speed/pitch modifications to existing Edge-TTS audio or reusing previously evaluated noise tracks), the test generation and execution for Phase 39 must not proceed until the following dependencies are met.

### Missing Dependencies Required to Unblock Stage E:
1. **TTS Phonetic Realism Verification:** Resolve Piper TTS Vietnamese tone phoneme map omissions (tones 2, 6) or obtain verified offline G2P phoneme tables for Vietnamese to ensure accurate tonal synthesis on target keywords ("bật đèn", "tắt đèn", "bật quạt", "tắt quạt").
2. **Clean External Noise Corpus:** Formally acquire and ingest a verified external environmental noise library (such as ESC-50 or MUSAN) into `data/raw/noise/external/`, verify zero historical exposure, and pre-register SHA-256 digests.
3. **Pre-Registration Freeze:** Hash all input manifests and event ledgers prior to forward inference.

```
FORMAL STATUS: BLOCKED_PENDING_INDEPENDENT_TTS_SOURCE
```

---

## 5. Mandatory Governance Declarations

```
EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED
NOT_VALIDATED_ON_NATURAL_SPEECH
NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION
NOT_VALIDATED_PENDING_DATA
```
- Default model remains $v11$ across all active deployments and configurations.
- Decision threshold $\theta_{\text{cmd}} = 0.20$ strictly locked.
- Parameter count strictly locked at 138,469 (`DEFER_E2_CAPACITY`).
- Zero human speech collection, zero physical microphone testing.
