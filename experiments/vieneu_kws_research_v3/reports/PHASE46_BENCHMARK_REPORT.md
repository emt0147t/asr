# Phase 46: Fresh Long-Horizon Synthetic Challenge Benchmark Report

Workspace: `D:\trainidentificationaudio`  
Experiment Directory: `experiments/vieneu_kws_research_v3/`  
Date: 2026-10-18  
Status: `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Executive Summary

Phase 46 constructed a fresh, long-horizon synthetic challenge benchmark designed to provide rigorous empirical evaluation of Vietnamese keyword spotting under realistic streaming exposure.

### Key Milestones Delivered:
1. **Long-Horizon Exposure:** Built `test_stream_continuous_nontarget_3h.wav`, providing **EXACTLY 10,800.0 seconds = 3.000 hours** of continuous non-target streaming audio. Under a 3-hour test stream, zero observed false activations provides a 95% one-sided Poisson upper bound of:
   $$\lambda_{95\%} = \frac{-\ln(0.05)}{3.0\text{ hr}} \approx \frac{2.996}{3.0} \approx 0.999\text{ FA/hour} \le 1.0\text{ FA/hour}$$
   establishing formal statistical confidence for the target operational threshold.
2. **Text Disjointness:** 210 strictly disjoint Vietnamese texts across 5 linguistic confusion families:
   - Train: 60 unique texts
   - Validation: 30 unique texts
   - Test: 120 unique texts
   - **Cross-split text overlap: 0 texts (100% disjoint)**.
3. **Voice Isolation:** 25 approved VieNeu-TTS speaker voices strictly allocated:
   - Train: 17 voices (`vie_voice_001` .. `vie_voice_017`)
   - Validation: 4 voices (`vie_voice_018` .. `vie_voice_021`)
   - Test: 4 voices (`vie_voice_022` .. `vie_voice_025`)
   - **Cross-split speech voice overlap: 0 voices (100% disjoint)**.
4. **Controlled Background Conditions:** 4 distinct synthetic acoustic noise environments (`near_silence`, `hvac_hum`, `office_ambient`, `airflow_pink`) modeled and mixed at controlled signal-to-noise ratios.
5. **No Synthetic Loop Replay:** The 3-hour stream contains 960 events comprising 2 distinct passes over 480 unique synthetic utterances with randomized interleaving and varied gap cadences ($5.0\text{–}14.0\text{ s}$), avoiding short-loop replay.

---

## 2. Dataset Inventory and Split Composition

| Split | Target Commands (`TRAI`, `PHAI`, `LEN`, `XUONG`) | UNKNOWN Hard Negatives | Background SILENCE | Total Audio Files |
| :--- | :---: | :---: | :---: | :---: |
| **Train** | 680 (17 voices $\times$ 4 commands $\times$ 10) | 340 (60 texts across 17 voices) | 340 (4 noise conditions) | **1,360** |
| **Validation** | 160 (4 voices $\times$ 4 commands $\times$ 10) | 120 (30 texts $\times$ 4 voices) | 80 (4 noise conditions) | **360** |
| **Locked Test** | 160 (4 voices $\times$ 4 commands $\times$ 10) | 480 (120 texts $\times$ 4 voices) | 80 (4 noise conditions) | **720** |
| **Total** | **1,000** | **940** | **500** | **2,440** |

---

## 3. Continuous Challenge Benchmark Streams

| Stream File Name | Stream Purpose | Total Duration (s) | Total Duration (hr) | Event Count | Active Speech (s) | Active Speech (hr) | Active Speech Fraction | Acoustic Condition |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `val_stream_continuous_commands.wav` | Validation Target Recall | 514.95 s | 0.1430 hr | 160 | 160.0 s | 0.0444 hr | 31.07% | `office_ambient` |
| `val_stream_continuous_nontarget.wav` | Validation Rejection Tuning | 478.46 s | 0.1329 hr | 120 | 120.0 s | 0.0333 hr | 25.08% | `hvac_hum` |
| `test_stream_continuous_commands.wav` | Locked Test Target Recall | 519.15 s | 0.1442 hr | 160 | 160.0 s | 0.0444 hr | 30.82% | `office_ambient` |
| `test_stream_continuous_nontarget_3h.wav` | **Locked 3-Hour Challenge** | **10,800.00 s** | **3.0000 hr** | **960** | **960.0 s** | **0.2667 hr** | **8.89%** | `airflow_pink` |

---

## 4. Frozen File Hashes (SHA-256)

Authoritative pre-registered SHA-256 checksums recorded prior to Phase 47 training:

```text
manifest.csv:
5cddf584e1a703623d5ae3f6caae774e6865890608a0251f99fb5908632f0a03

text_inventory.csv:
77165682c6bb82a7a2f9513679af9f22a2025960edc54666d0163c0e38de3292

splits/train_manifest.csv:
d1a7e0db703565741ab0fd21d4d33ec910137009c7ddcc868f2405a4c982584d

splits/val_manifest.csv:
6434b7f93835c9b33424414e3abf4b8b52805e57371831cec046867b180d795b

splits/test_manifest.csv:
45235cd417de0096798134f3aa55a57a2b7d49c7f380ef2fce85fc0722d390ab

continuous_streams/val_stream_continuous_commands.wav:
e58a7b4d270f7918a6d468e31c5739d7c876d077071d5f0c4fa47361bceebde2

continuous_streams/val_stream_continuous_commands_ground_truth.json:
ce051a093745fa93781b80ca5039a84a54da62078ac3f6ff7a755e38b9821408

continuous_streams/val_stream_continuous_nontarget.wav:
771be7e5a40baa28305aa3c201c12ced266b3255813c807b666b1bc309ab42d1

continuous_streams/val_stream_continuous_nontarget_ground_truth.json:
8155074af3b33d75d93934982c48992d60828fe44133d39d500ec706eacc7259

continuous_streams/test_stream_continuous_commands.wav:
7ee89c957b1d99d1ca30c22a80e6021cb46517c1712339fb7dee0a999e71b430

continuous_streams/test_stream_continuous_commands_ground_truth.json:
2a691210b80e05ea9d9a224320938745648a1a6ef3161be9a3df0de657435dea

continuous_streams/test_stream_continuous_nontarget_3h.wav:
1d325a33a03ba3ce97ea3884a2b064583d7c972465cffa298ac2b75762b85abe

continuous_streams/test_stream_continuous_nontarget_3h_ground_truth.json:
c0da65d736c761cda4d176843cfc5521e3ce150ea9511678759a118a3aac2686
```

All test data is locked and frozen. No training, hyperparameter tuning, or hard-negative mining will access or alter these test artifacts.
