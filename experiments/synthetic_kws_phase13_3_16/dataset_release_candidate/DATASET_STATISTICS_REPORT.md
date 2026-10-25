# Dataset Statistics and Limitations Report: V5.2 Release Candidate

**Dataset:** `synthetic_kws_v5_2_release_candidate`  
**Total Samples:** 3,000 clips  
**Audio Specs:** 16,000 Hz, 1 channel (mono), 1.000 s (16,000 samples), 16-bit PCM  
**Gate Verdict:** **`DATASET_PASS`**  

---

## 1. Class and Split Distribution

| Class Label | Train Split (70%) | Val Split (15%) | Test Split (15%) | Total Corpus | Percentage |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`TRAI`** | 350 | 75 | 75 | **500** | 16.67% |
| **`PHAI`** | 350 | 75 | 75 | **500** | 16.67% |
| **`LEN`** | 350 | 75 | 75 | **500** | 16.67% |
| **`XUONG`** | 350 | 75 | 75 | **500** | 16.67% |
| **`UNKNOWN`** | 350 | 75 | 75 | **500** | 16.67% |
| **`SILENCE`** | 350 | 75 | 75 | **500** | 16.67% |
| **Total** | **2,100** | **450** | **450** | **3,000** | **100.0%** |

---

## 2. Acoustic Subclass Breakdown

| Acoustic Subclass | Description | Total Count | Percentage |
| :--- | :--- | :---: | :---: |
| `target_keyword` | Clean synthetic target words (`trái`, `phải`, `lên`, `xuống`) | 2,000 | 66.67% |
| `ambient_room_tone` | Low-pass filtered room acoustic floor | 316 | 10.53% |
| `open_vocabulary_general` | Real Vietnamese conversational vocabulary | 246 | 8.20% |
| `confuser_initial_sharing` | Initial-consonant phonetic confusers | 130 | 4.33% |
| `confuser_rhyme_sharing` | Rhyme-vowel phonetic confusers | 124 | 4.13% |
| `ambient_low_freq_hum` | 50Hz/60Hz mains electrical hum + harmonics | 114 | 3.80% |
| `ambient_digital_zero` | Quantization dither floor ($\pm 1$ LSB) | 70 | 2.33% |
| **Total** | | **3,000** | **100.0%** |

---

## 3. Data Integrity and Verification Results

* **Files Checked:** 3,000 / 3,000 (100% full coverage)
* **WAV Header Integrity:** 3,000/3,000 valid 16kHz mono 16-bit PCM
* **SHA-256 Manifest Check:** 3,000/3,000 disk hashes match manifest exactly
* **Amplitude Limits:** Peak values strictly $\le 0.999$, 0 clipped samples
* **Numerical Validity:** 0 NaNs, 0 Infs, 0 zero-length files
* **Reproducibility Test:** 120/120 audit sample reproduced bit-exact from manifest rows (100.0%)

---

## 4. Scientific Limitations

> [!WARNING]
> 1. **Synthetic Nature:** Waveforms are generated using a physical articulatory model (Rosenberg glottal source + formant cascaded resonators). They do not reproduce real human vocal fold biomechanics or microphone non-linearities.
> 2. **Lexical Repetition:** Target keywords are repeated syntheses of the 4 specific words across 12 synthetic speakers, differing in duration, pitch, vocal tract length, and background noise. They do not constitute human speech diversity.
> 3. **Non-Human Benchmark:** This dataset is strictly an engineering training and development benchmark; it is **NOT** a substitute for real-world natural human speech evaluation.
