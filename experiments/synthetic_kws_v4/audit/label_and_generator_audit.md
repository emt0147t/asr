# Label Correctness and Synthetic Generator Semantics Audit

> [!IMPORTANT]
> **GOVERNANCE INVARIANT: HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED.**
> All analyses conducted strictly on existing offline synthetic generators and code artifacts. Checkpoints, models, and manifests remain frozen.

---

## 1. Executive Summary

This audit investigates the phonetic, mathematical, and algorithmic semantics of `experiments/synthetic_kws_v4/onset_aware_generator.py` and its predecessor scripts. Through static code inspection and deterministic numerical verification, we establish how the target keywords (`TRAI`, `PHAI`, `LEN`, `XUONG`), negatives (`UNKNOWN`), and ambient conditions (`SILENCE`) are synthesized.

---

## 2. Deterministic Verification of Generator Properties

### Test 1: Does the `dialect` Parameter Affect the Generated Audio?
- **Invocation**: `generate_vietnamese_onset_aware_utterance("trái", sr=16000, f0_base=130.0, duration_sec=0.5, dialect=...)`
- **Seeds**: Fixed seed `12345` for Northern, Southern, and Central calls.
- **Results**:
  - Max absolute difference (Northern vs Southern): **`0.000000`** (bit-exact match).
  - Max absolute difference (Northern vs Central): **`0.000000`** (bit-exact match).
- **Code Finding**: While the function signature declares `dialect: str = "Northern"`, the variable `dialect` is **never read or branched upon** anywhere within the 190 lines of `onset_aware_generator.py`.
- **Verdict**: **DEAD CODE**. Manifest records declaring `"Northern"`, `"Central"`, and `"Southern"` are acoustically identical in their synthesis formula.

### Test 2: The 25ms Closure & Transient Burst Asymmetry
- **Measurement**: Maximum absolute amplitude in the first 25ms (samples 0 to 400 at 16 kHz) and the subsequent 15ms burst window (samples 400 to 640):
  | Word / Command | Intended Class | First 25ms Peak ($t < 25\text{ ms}$) | Burst Window Peak ($25 \le t \le 40\text{ ms}$) | Acoustic Implementation in Code |
  | :--- | :---: | :---: | :---: | :--- |
  | **`trái`** | `TRAI` | **`0.0008`** | **`0.1459`** | `output[:closure_samples] *= 0.05` ($95\%$ suppression), followed by high-pass filtered noise burst ($>3500\text{ Hz}$). |
  | **`phải`** | `PHAI` | **`0.2935`** | **`0.3576`** | Continuous bandpass frication ($1800 - 6000\text{ Hz}$) starting at $t=0$; no closure gap. |
  | **`lên`** | `LEN` | **`0.2101`** | **`0.3136`** | Natural sinusoidal attack; no closure gap, no noise injection. |
  | **`xuống`** | `XUONG` | **`0.4052`** | **`0.6486`** | Continuous highpass noise ($>4000\text{ Hz}$) starting at $t=0$; no closure gap. |
- **Verdict**: **PROVEN NON-LINGUISTIC SHORTCUT**. The 25ms near-zero closure gap ($< 0.001$) is strictly unique to `TRAI`.

### Test 3: Shared Formant Resonator Equations for `TRAI` and `PHAI`
- **Code Inspection** (`onset_aware_generator.py` lines 75-79):
  ```python
  if word in ["trái", "phải"]:
      f1, f2, f3 = 800.0, 1350.0, 2600.0
      f1_track = f1 * (1.0 - 0.4 * (t / duration_sec))
      f2_track = f2 * (1.0 + 0.5 * (t / duration_sec))
      f3_track = np.full_like(t, f3)
  ```
- **Finding**: Because both words share the Vietnamese rhyme /-a:j/, the generator applies identical central formant tracks $F_1, F_2, F_3$.
- **Verdict**: **IDENTICAL VOWEL RESONANCE**. The steady-state vowel body provides virtually zero discriminative acoustic contrast between `TRAI` and `PHAI`, forcing any trained neural network to rely solely on the onset window.

### Test 4: Negative Vocabulary Diversity (`UNKNOWN`)
- **Code Inspection** (`onset_aware_generator.py` lines 31-40):
  ```python
  NON_TARGET_WORDS = [
      ("chào", "falling"), ("bạn", "heavy"), ("đi", "level"), ("xe", "level"),
      ("nhà", "falling"), ("số", "rising"), ("ngày", "falling"), ("học", "heavy")
  ]
  ```
- **Finding**: The entire `UNKNOWN` class is constructed by cycling through exactly **8 monosyllabic words**.
- **Verdict**: **EXTREMELY NARROW VOCABULARY**. An 8-word dictionary is insufficient to represent open-set Vietnamese speech.

### Test 5: Ambient / Background Diversity (`SILENCE`)
- **Code Inspection** (`generate_synthetic_silence_clip`):
  - `mode == "pure_silence"`: Digital zeros (`np.zeros`).
  - `mode == "dither"`: Gaussian noise (`np.random.randn * 1e-3`).
  - `mode == "room_tone"`: Lowpass-filtered Gaussian noise ($f_c = 1000\text{ Hz}$) scaled to $-50\text{ dBFS}$ RMS.
- **Dataset Ingest Inspection** (`build_onset_aware_data.py` line 153):
  ```python
  mode = "room_tone" if take % 2 == 0 else "dither"
  ```
- **Finding**: `pure_silence` was never used during training or validation generation. Only `room_tone` and `dither` were generated.
- **Verdict**: **MISSING DIGITAL ZEROS & REALISTIC ACOUSTICS**. The model never learned digital zeros during training, explaining why pure zeros receive high uncertainty and are misclassified as `TRAI`.

### Test 6: Manifest Label Alignment
- Verification of manifest generation calls in `build_onset_aware_data.py` and `generate_fresh_holdout_v4.py`:
  - Every row's `transcript` and `class_label` exactly match the function arguments passed to `generate_vietnamese_onset_aware_utterance`.
- **Verdict**: **CONSISTENT LABEL CALLS**. There is no label inversion or transposition bug between manifest rows and generation calls.
