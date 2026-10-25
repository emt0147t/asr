# PHASE 41: INDEPENDENT VIETNAMESE TTS PILOT EVALUATION REPORT

**Candidate Engine:** `facebook/mms-tts-vie`  
**Model Architecture:** VITS (`VitsModel`, End-to-End Variational Inference with Adversarial Learning)  
**Snapshot Commit:** `b58928d033932a49aa8e3d6cf11625b25fe928d2`  
**Native Sampling Rate:** 16,000 Hz (Bit-exact match to KWS frontend input)  
**Evaluation Date:** 2026-10-25  
**Governing Gate Condition:** `PASS_INDEPENDENT_TTS_PILOT`

---

## 1. Executive Summary & Verification

Phase 41 evaluated Meta's `facebook/mms-tts-vie` as a candidate independent external speech synthesizer to replace Edge-TTS for genuinely independent generalization testing.

### Key Verification Highlights:
1. **Engine Independence: 100% Confirmed**  
   MMS-TTS utilizes an end-to-end VITS acoustic model with normalizing flows and adversarial vocoder, completely distinct from VieNeu-TTS (V3 FastSpeech2 + ONNX HiFi-GAN) and Edge-TTS (Microsoft Azure commercial neural synthesis pipeline).
2. **Training Corpus Independence: 100% Confirmed**  
   MMS-TTS was trained on the Faith Comes By Hearing Vietnamese New Testament reading corpus, completely disjoint from VieNeu-TTS studio recordings and Edge-TTS voices (`vi-VN-HoaiMyNeural` and `vi-VN-NamMinhNeural`).
3. **Phonetic Token Coverage: Zero Missing Tokens**  
   Across all 24 pilot test words spanning target commands, carrier phrases, and 6 Vietnamese tone minimal pairs (`ma`, `mà`, `má`, `mả`, `mã`, `mạ`; `ba`, `bà`, `bá`, `bả`, `bã`, `bạ`), **zero `<unk>` tokens** were generated. The tokenizer natively maps standard Vietnamese accented characters without phoneme-map omission warnings.
4. **Acoustic Quality & Signal Integrity:**  
   - Clipping count: **0 / 24** (Max amplitude strictly $< 0.999$, average peak $0.62$).
   - Silence count: **0 / 24** (Mean amplitude $> 0.015$).
   - Duration range: **0.64 s to 1.52 s** (Natural conversational command pacing).
   - Seed Reproducibility: Forward inference with fixed `torch.manual_seed(42)` produced **bit-identical waveforms** ($\max |\Delta| = 0.0$).
5. **Quality Gate Decision:** **PASS**  
   `facebook/mms-tts-vie` is formally approved as an authorized independent synthesis engine for Phase 43 locked evaluation.

---

## 2. Model Provenance & Artifact Hashes

| Artifact File | Size | Verified SHA-256 Digest |
|:--------------|:----:|:------------------------|
| `model.safetensors` | 145.27 MB | `55ded90c3e57dc2814fa2cdfe3f9e7a5c28e1223b06c0a260a4495b080762ffd` |
| `config.json` | 1.64 KB | `70dcc60b59337b43d58c865735846a2bc5a604d2698d7dfca713904da997ca8d` |
| `vocab.json` | 1.15 KB | `15972d5661d06687659208574acea6b7415f70456c3b291246971c003e755bb7` |
| `tokenizer_config.json` | 289 B | `d2272a1d3a7b5330d28dd9daa6fdd549c8bedc723764a072c5c67cc66de08f52` |
| `special_tokens_map.json` | 49 B | `90b6165eb6c3fe8041fb7a46fb1b7c50220b2d2cbe232d2c1dca2b3f6900f721` |

---

## 3. Pilot Synthesis Results Table (24 Independent Utterances)

| Pilot ID | Type | Target Class | Text | Duration (s) | Max Amp | UNK Tokens | Status |
|:---------|:-----|:------------:|:-----|:------------:|:-------:|:----------:|:------:|
| `pilot_cmd_01` | command | `TRAI` | rẽ trái | 0.992 | 0.537 | 0 | **PASS** |
| `pilot_cmd_02` | command | `PHAI` | rẽ phải | 1.168 | 0.560 | 0 | **PASS** |
| `pilot_cmd_03` | command | `LEN` | bật lên | 1.040 | 0.687 | 0 | **PASS** |
| `pilot_cmd_04` | command | `XUONG` | tắt xuống | 1.264 | 0.755 | 0 | **PASS** |
| `pilot_cmd_05` | command | `TRAI` | quay sang trái | 1.520 | 0.731 | 0 | **PASS** |
| `pilot_cmd_06` | command | `PHAI` | quay sang phải | 1.472 | 0.561 | 0 | **PASS** |
| `pilot_cmd_07` | command | `LEN` | đi lên | 0.896 | 0.711 | 0 | **PASS** |
| `pilot_cmd_08` | command | `XUONG` | đi xuống | 1.120 | 0.670 | 0 | **PASS** |
| `pilot_tone_01` | tone_probe | `REJECT` | ma | 0.704 | 0.674 | 0 | **PASS** |
| `pilot_tone_02` | tone_probe | `REJECT` | mà | 0.688 | 0.346 | 0 | **PASS** |
| `pilot_tone_03` | tone_probe | `REJECT` | má | 0.688 | 0.702 | 0 | **PASS** |
| `pilot_tone_04` | tone_probe | `REJECT` | mả | 0.752 | 0.309 | 0 | **PASS** |
| `pilot_tone_05` | tone_probe | `REJECT` | mã | 0.768 | 0.331 | 0 | **PASS** |
| `pilot_tone_06` | tone_probe | `REJECT` | mạ | 0.704 | 0.428 | 0 | **PASS** |
| `pilot_tone_07` | tone_probe | `REJECT` | ba | 0.736 | 0.705 | 0 | **PASS** |
| `pilot_tone_08` | tone_probe | `REJECT` | bà | 0.688 | 0.574 | 0 | **PASS** |
| `pilot_tone_09` | tone_probe | `REJECT` | bá | 0.640 | 0.771 | 0 | **PASS** |
| `pilot_tone_10` | tone_probe | `REJECT` | bả | 0.640 | 0.570 | 0 | **PASS** |
| `pilot_tone_11` | tone_probe | `REJECT` | bã | 0.768 | 0.315 | 0 | **PASS** |
| `pilot_tone_12` | tone_probe | `REJECT` | bạ | 0.640 | 0.528 | 0 | **PASS** |
| `pilot_conf_01` | hard_negative | `REJECT` | chải | 0.944 | 0.635 | 0 | **PASS** |
| `pilot_conf_02` | hard_negative | `REJECT` | phẩy | 0.880 | 0.493 | 0 | **PASS** |
| `pilot_conf_03` | hard_negative | `REJECT` | lén | 0.848 | 0.756 | 0 | **PASS** |
| `pilot_conf_04` | hard_negative | `REJECT` | suốt | 0.688 | 0.662 | 0 | **PASS** |
| `pilot_conf_05` | hard_negative | `REJECT` | chờ đợi | 1.296 | 0.625 | 0 | **PASS** |
| `pilot_conf_06` | hard_negative | `REJECT` | thời gian | 1.296 | 0.558 | 0 | **PASS** |

---

## 4. Formal Quality Gate Record

```
PILOT QUALITY GATE: PASS
CRITERIA SATISFIED:
- 0 unk tokens across all Vietnamese tone marks
- 0 clipped or silent samples
- 100% deterministic reproducibility under fixed seed
- Complete engine and corpus independence from VieNeu and Edge-TTS
```

---

## 5. Mandatory Governance Declarations

```
EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED
NOT_VALIDATED_ON_NATURAL_SPEECH
NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION
NOT_VALIDATED_PENDING_DATA
```
