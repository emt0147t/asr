# Phase 2 — Prototype Continuous KWS Event Detection Results Report

**Repository:** `D:\trainidentificationaudio`  
**Working Branch:** `feature/continuous-kws-segmentation`  
**Evaluation Date:** October 29, 2026  
**Reference Model Checkpoint:** `variant_v11_hard_negative_weighted_seed42.pt`  
**Checkpoint SHA-256:** `3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2`  
**ONNX Artifact:** `vietnamese_kws_v11.onnx`  
**ONNX SHA-256:** `54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99`  
**Operating Threshold:** $\theta_{\text{cmd}} = 0.20$  

---

## Mandatory Governance Disclaimers

> [!WARNING]
> This prototype report operates under the four inviolable project governance invariants:
> 1. `EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`
> 2. `NOT_VALIDATED_ON_NATURAL_SPEECH`
> 3. `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`
> 4. `NOT_VALIDATED_PENDING_DATA`
>
> All experiments reported herein utilize synthetic stream concatenation and deterministic acoustic replays. No physical microphone hardware validation or deployment readiness is claimed.

---

## 1. Executive Summary & Objective

In prior demonstration releases, live microphone streaming suffered from a critical usability failure: when speakers issued rapid, continuous command sequences such as **`LÊNLENPHAI`** without an exaggerated pause, the system failed to detect the three intended events (`LEN → LEN → PHAI`).

The objective of **Phase 2** was to design, implement, and quantitatively benchmark architectures capable of detecting contiguous and repeated keyword events **without requiring speakers to actively pause 250–300 ms between commands**.

### Key Scientific & Engineering Findings
1. **The Sliding-Window Probability Plateau Dilemma (Approach B):**
   Because the frozen v11 CNN requires a 1.0-second (16,000 samples) window, two consecutive keyword utterances (each ~250 ms) fit entirely within the model's receptive field. In continuous sliding-window inference, $P(\text{LEN})$ remains continuously elevated ($>0.80$) over a broad 700 ms span with spatial pooling ripples ($\Delta P \approx 0.05$). A probability trajectory decoder cannot reliably separate contiguous identical commands: reducing cooldown or NMS distance causes massive duplicate triggers on single words (**25.11% duplicate rate**), while increasing cooldown blocks repeated commands (**0.0% sequence match on contiguous `LEN-LEN`**).
2. **Acoustic Energy Valley Segmentation (Approach A2):**
   Vietnamese syllable phonetics naturally create an acoustic energy valley ($\Delta \text{dB} \ge 15 - 25\text{ dB}$) between the vocalic nuclei of contiguous words, even with zero silence pause (0 ms gap). By implementing `AcousticValleySegmenter`, which tracks short-time energy dips and syllable onset rebounds, contiguous commands are cleanly split and right-padded into the canonical 1.0s format.
3. **Benchmark Dominance:**
   `AcousticValleySegmenter` achieved **100.0% Event Precision**, **100.0% Event Recall**, **100.0% Exact Sequence Match**, **0.0% Duplicate Rate**, and **130.0 ms median latency** across all test gaps ($0\text{ ms}, 50\text{ ms}, 100\text{ ms}, 150\text{ ms}, 250\text{ ms}, 400\text{ ms}$), completely outperforming both the baseline segmenter (34.72% match) and the streaming cooldown decoder (25.00% match).
4. **The Natural Speech / Closed-Vocabulary Tradeoff:**
   While `AcousticValleySegmenter` solves contiguous command detection in games, fine-grained syllable segmentation exposes the closed-vocabulary 5-class classifier to out-of-vocabulary syllables in conversational speech, confirming why the governance designation `NOT_VALIDATED_ON_NATURAL_SPEECH` remains scientifically mandatory.

---

## 2. Invariants & Baseline Preservation

Throughout Phase 2, the core research integrity invariants were strictly maintained:
- **Dedicated Branch:** All development executed on branch `feature/continuous-kws-segmentation`.
- **Checkpoint Immutability:** Legacy v11 PyTorch checkpoint (`variant_v11_hard_negative_weighted_seed42.pt`) verified with SHA-256 `3c4369...`.
- **ONNX Immutability:** Authoritative ONNX model (`vietnamese_kws_v11.onnx`) verified with SHA-256 `54298c...`.
- **Frontend & Classifier Unaltered:** Standard 40-filter Log-Mel frontend, right-padding to 16,000 samples, and 5-class argmax decision rule with $\theta_{\text{cmd}}=0.20$ left strictly untouched. No models were retrained, and parameter counts were not modified.

---

## 3. Evaluated Architectures

We evaluated five distinct system architectures on the unified benchmark:

| Architecture ID | Configuration Name | Mechanism | Primary Target / Operating Principle |
| :--- | :--- | :--- | :--- |
| `baseline_segmented` | Baseline Segmented | `UtteranceSegmenter` | Fixed silence threshold: accumulates speech until $300\text{ ms}$ silence; 150ms pre-roll. |
| `baseline_streaming` | Baseline Streaming | Sliding Ring Buffer | 160ms hop sliding window over 1.0s; 800ms cooldown latch to prevent duplicates. |
| `approach_a1_speechband` | Approach A1: SpeechBandRatio | `SpeechBandRatioSegmenterAdapter` | Phase 15/18 Profile B VAD: $140\text{ ms}$ hangover, energy ratio $\ge 0.32$ in $300-3400\text{ Hz}$. |
| `approach_a2_acoustic_valley` | Approach A2: Acoustic Valley | `AcousticValleySegmenter` | Tracks vocalic energy peaks, detects dips $\ge 12\text{ dB}$, splits at syllable onset rebound. |
| `approach_b_continuous_decoder` | Approach B: Continuous Decoder | `ContinuousEventDecoder` | 40ms hop sliding window, NMS peak detection with min distance $250\text{ ms}$ and $\Delta P_{\text{dip}}=0.15$. |

---

## 4. Master Benchmark Results

The benchmark evaluated **72 test streams per configuration** (4 speech patterns $\times$ 6 inter-word gaps $\times$ 3 speaker triads), plus **15 negative control files** (foils, conversational speech, background noise).

```
========================================================================================================================
Configuration                          | Prec (%)  | Rec (%)   | F1 (%)    | SeqMatch  | DupRate  | Lat p50   | Lat p95  
------------------------------------------------------------------------------------------------------------------------
Baseline Segmented (300ms silence)     |    82.61% |    42.22% |    55.88% |    34.72% |    0.00% |  290.0 ms |  310.0 ms
Baseline Streaming (160ms hop, 800ms)  |    95.06% |    42.78% |    59.00% |    25.00% |    0.00% |    0.0 ms |  572.0 ms
Approach A1: SpeechBand (140ms hang)   |    96.90% |    69.44% |    80.91% |    62.50% |    0.00% |  130.0 ms |  150.0 ms
Approach A2: Acoustic Valley (12dB)    |   100.00% |   100.00% |   100.00% |   100.00% |    0.00% |  130.0 ms |  140.0 ms
Approach B: Continuous Decoder (40ms)  |    65.20% |    82.22% |    72.73% |    29.17% |   25.11% |  170.0 ms |  566.5 ms
========================================================================================================================
```

### Target Criteria Comparison Table

| Metric | Target Specification | Baseline Segmented | Baseline Streaming | Approach A1 | Approach A2 (Acoustic Valley) | Approach B | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Command Event Precision** | $\ge 85.0\%$ | 82.61% | 95.06% | 96.90% | **100.00%** | 65.20% | **PASS (A2)** |
| **Command Event Recall** | $\ge 85.0\%$ | 42.22% | 42.78% | 69.44% | **100.00%** | 82.22% | **PASS (A2)** |
| **Exact Sequence Match** | $\ge 70.0\%$ | 34.72% | 25.00% | 62.50% | **100.00%** | 29.17% | **PASS (A2)** |
| **Duplicate Rate** | $\le 2.0\%$ | 0.00% | 0.00% | 0.00% | **0.00%** | 25.11% | **PASS (A2)** |
| **Median Detection Latency** | $\le 250\text{ ms}$ | 290.0 ms | 0.0 ms | 130.0 ms | **130.0 ms** | 170.0 ms | **PASS (A2)** |

---

## 5. Breakdown by Inter-Word Gap

The table below reports the **Exact Sequence Match Rate (%)** across the 6 evaluated inter-word gap durations:

| Configuration | 0 ms (Contiguous) | 50 ms | 100 ms | 150 ms | 250 ms | 400 ms (Standard Pause) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Segmented (300ms)** | 25.0% | 25.0% | 25.0% | 25.0% | 25.0% | 83.3% |
| **Baseline Streaming (800ms)** | 25.0% | 25.0% | 25.0% | 25.0% | 25.0% | 25.0% |
| **Approach A1: SpeechBand (140ms)**| 25.0% | 50.0% | 50.0% | 83.3% | 83.3% | 83.3% |
| **Approach A2: Acoustic Valley** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| **Approach B: Continuous Decoder** | 0.0% | 33.3% | 33.3% | 25.0% | 50.0% | 33.3% |

### Analysis:
- **0 ms to 100 ms Gap:** Baseline Segmented fails completely on multi-command sequences (matching only Pattern 4, the single elongated word, explaining the 25.0% baseline). Approach A1 partially recovers at 50–100 ms (50.0%). Approach A2 achieves **100.0% exact match** across all gaps, successfully resolving even 0 ms gap without pauses.
- **Approach B at 0 ms:** Achieves **0.0% sequence match**. Because both commands reside within the 1.0s window simultaneously, the second `LEN` is masked and fails to trigger.

---

## 6. Breakdown by Test Pattern

The table below reports the **Exact Sequence Match Rate (%)** across the 4 core test patterns:

| Pattern Description | Ground Truth | Baseline Segmented | Baseline Streaming | Approach A1 | Approach A2 (Acoustic Valley) | Approach B |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pattern 1:** Repeated then Distinct | `LEN → LEN → PHAI` | 16.7% | 0.0% | 72.2% | **100.0%** | 66.7% |
| **Pattern 2:** Distinct Sequence | `TRAI → LEN → XUONG` | 11.1% | 0.0% | 50.0% | **100.0%** | 0.0% |
| **Pattern 3:** Triple Repeated | `LEN → LEN → LEN` | 11.1% | 0.0% | 27.8% | **100.0%** | 50.0% |
| **Pattern 4:** Elongated Command | `LEN` (750ms single utterance) | **100.0%** | **100.0%** | **100.0%** | **100.0%** | 0.0% |

### Analysis:
- **Baseline Streaming:** Scored 0.0% on Patterns 1, 2, and 3 due to the 800ms cooldown latch blocking subsequent commands.
- **Elongated Command (Pattern 4):** Approached A2 correctly recognizes the continuous vocalic energy of an elongated word, emitting strictly 1 event (zero over-splitting). In contrast, Approach B suffered multiple duplicate triggers on the elongated word (0.0% match).

---

## 7. Deep Scientific Analysis: Why Each Approach Behaves As It Does

### 7.1 Why Baseline Streaming Fails on Continuous Commands
In legacy streaming mode, the sliding window operates with an $800\text{ ms}$ debounce cooldown. When a user speaks `LÊNLENPHAI`:
1. The first `LEN` triggers at $t \approx 0.4\text{s}$.
2. The second `LEN` occurs at $t \approx 0.7\text{s}$ (300 ms later).
3. The detector's cooldown latch is active until $t = 1.2\text{s}$. The second `LEN` is completely suppressed.
4. If cooldown is removed, a single spoken word triggers 4 to 8 times as it slides through the window.

### 7.2 Why Approach B (Sliding Window NMS) Cannot Solve Repeated Commands
When evaluating a 1.0s sliding window on two contiguous identical commands (`LEN` + 0ms + `LEN`):
- Both commands span a total of $250 + 260 = 510\text{ ms}$.
- For over 400 ms, **both words are simultaneously inside the 1.0s window**.
- The CNN outputs a continuous probability plateau of $P(\text{LEN}) \approx 0.85 - 0.92$.
- The probability dip between the words is only $\Delta P \approx 0.05$, which is indistinguishable from spatial convolutional pooling ripples on a single word.
- Consequently, NMS with a dip requirement misses the second command, while NMS with a small dip requirement produces duplicate triggers on single words (**25.11% duplicate rate**).

### 7.3 Why Approach A2 (Acoustic Valley Segmentation) Succeeds
Even when commands are spoken back-to-back without a pause:
- Vietnamese syllables are phonetically structured with a vocalic nucleus (the vowel `ê`, which carries high acoustic power: $-7\text{ dB}$ to $-8\text{ dB}$) and consonantal margins (`l`, `n`, `ph`, which drop power to $-25\text{ dB}$ to $-32\text{ dB}$).
- The transition between `LÊN` and `LÊN` produces a physical acoustic dip of **17 to 25 dB**.
- By tracking this dip and the subsequent syllable onset rebound, `AcousticValleySegmenter` identifies the syllable boundary, cleanly hands off the completed word to the CNN (right-padded to 1.0s), and re-arms instantly.
- The CNN receives cleanly segmented, right-padded 1.0s waveforms matching its training distribution, resulting in **100% exact sequence match**.

---

## 8. Reproducibility & Regression Test Results

Deterministic verification of Phase 2 was conducted using PyTorch 2.6.0+cu124 and ONNXRuntime 1.31.0 on Windows.

### Test Execution 1: Phase 2 Regression Test Suite
```powershell
.venv\Scripts\pytest tests/test_continuous_kws_phase2.py -v
```
**Results:** `9 passed in 2.77s` (100% PASS)
- `test_frozen_artifact_immutability` — PASSED
- `test_repeated_command_len_len_phai[0ms]` — PASSED
- `test_repeated_command_len_len_phai[50ms]` — PASSED
- `test_repeated_command_len_len_phai[100ms]` — PASSED
- `test_repeated_command_len_len_phai[150ms]` — PASSED
- `test_three_contiguous_identical_commands` — PASSED
- `test_elongated_command_does_not_oversplit` — PASSED
- `test_distinct_continuous_sequence` — PASSED
- `test_negative_controls_and_ambient_suppression` — PASSED

### Test Execution 2: Full Live Mic Pipeline Deterministic Test Suite
```powershell
.venv\Scripts\pytest experiments/vieneu_kws_research_v13_4/demo/test_live_mic_pipeline.py -v
```
**Results:** `15 passed in 3.59s` (100% PASS)

---

## 9. Engineering Recommendations & Actionable Next Steps

Based on the quantitative benchmark and acoustic findings:

1. **Adopt Acoustic Valley Segmentation for Continuous Live Demos:**
   - Configure `profile="continuous_valley"` in `LiveKWSPipeline`.
   - Replaces the restrictive $300\text{ ms}$ silence requirement with adaptive acoustic valley detection ($12\text{ dB}$ dip, $6\text{ dB}$ rebound), enabling natural, continuous game play (`LÊNLENPHAI`).
2. **Preserve Research Governance Labels:**
   - Because fine-grained syllable segmentation exposes the 5-class model to out-of-vocabulary words in conversational speech, all four governance labels must remain active:
     `EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`
     `NOT_VALIDATED_ON_NATURAL_SPEECH`
     `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`
     `NOT_VALIDATED_PENDING_DATA`.
3. **Future Acoustic Modeling (Phase 3 / Beyond):**
   - For open-microphone natural conversational speech without false activations, the system should transition from an isolated-word CNN to a continuous CTC/CRNN or Streaming Conformer trained on continuous Vietnamese speech corpora.
