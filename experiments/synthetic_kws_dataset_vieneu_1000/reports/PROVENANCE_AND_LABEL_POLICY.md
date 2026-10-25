# Vietnamese Synthetic KWS — Provenance and Label Policy

**Dataset Directory:** `experiments/synthetic_kws_dataset_vieneu_1000/`  
**Dataset Version:** 1.0.0 (Research Candidate)  
**Date:** October 17, 2026  
**License:** Apache License 2.0  
**Per-Request Cost:** $0.00 (100% Offline Local Inference)  
**Official Project Status:** `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Provenance and Upstream Lineage

1. **Model Architecture & Upstream Repository:**
   * **Project Name:** VieNeu-TTS v3 Turbo
   * **Repository:** `https://github.com/pnnbao97/VieNeu-TTS`
   * **Author / Architect:** Phạm Nguyễn Ngọc Bảo
   * **Hugging Face Model Card:** `pnnbao-ump/VieNeu-TTS-v3-Turbo`
   * **License:** Apache License 2.0 (covers model weights, ONNX graphs, tokenizers, and shipped preset assets).
2. **Audio Codec:**
   * **Model:** MOSS-Audio-Tokenizer-Nano
   * **Repository:** `OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX`
   * **Native Sample Rate:** 48,000 Hz (48 kHz neural vocoder).
3. **Phonemizer:**
   * **Package:** `sea-g2p` (v0.10.0)
   * **Engine:** Rust-based Grapheme-to-Phoneme providing normalized Vietnamese tone trajectories and phonemic alignment without PyTorch.
4. **Execution Environment & Isolation:**
   * **Virtual Environment:** Dedicated `.venv_vieneu` (Python 3.11.9, `vieneu==3.8.3`, `onnxruntime==1.31.0`).
   * **Execution Mode:** 100% Local CPU via ONNX Runtime. PyTorch is never imported.
   * **Cloud Avoidance:** No calls to the proprietary VieNeu v4 cloud API or any external paid endpoints. Zero per-request financial cost.

---

## 2. Canonical Label Policy & Lexical Grounding

The dataset is strictly restricted to the four canonical Vietnamese navigation commands:

| Class Label | Canonical Orthography | Target IPA | Vietnamese Tone | Syllable Structure |
| :--- | :--- | :--- | :--- | :--- |
| **`TRAI`** | `Trái` | `/ʈaːj˧˥/` | Sắc (high-rising) | Initial retroflex stop [ʈ] + long open front vowel [aː] + palatal off-glide [j] |
| **`PHAI`** | `Phải` | `/faːj˧˩˨/` | Hỏi (mid-falling rising / dipping) | Initial labiodental fricative [f] + long open front vowel [aː] + palatal off-glide [j] |
| **`LEN`** | `Lên` | `/len˧˧/` | Ngang (mid-level) | Initial voiced lateral [l] + close-mid front vowel [e] + alveolar nasal coda [n] |
| **`XUONG`** | `Xuống` | `/suəŋ˧˥/` | Sắc (high-rising) | Initial alveolar sibilant [s] + centering diphthong [uə] + velar nasal coda [ŋ] |

* **Strict Diacritic Integrity:** Target texts are encoded strictly with standardized Unicode precomposed NFC Vietnamese characters (`Trái`, `Phải`, `Lên`, `Xuống`).
* **No Lexical Drift:** No extra words, carrier phrases, or conversational padding are permitted in this dataset.

---

## 3. Synthetic Preset Voice Policy & Human Participant Boundary

1. **Nature of Shipped Preset Voices:**
   * Each preset voice in VieNeu-TTS v3 Turbo is a pre-computed 192-dimensional speaker embedding vector $\mathbf{e}_{\text{spk}} \in \mathbb{R}^{192}$ and a discrete neural audio code matrix $\mathbf{C}_{\text{ref}} \in \mathbb{Z}^{16 \times 68}$ bundled in `vieneu/assets/voices_v3_turbo.json`.
   * **Scientific Boundary:** Preset voices are **synthetic conditioning vectors** in a neural latent manifold. They **do not represent independent human participants** in an acoustic or field speech campaign.
2. **Preset Roster Allocation (25 Voices):**
   * The complete set of 25 installed preset voices was partitioned into disjoint subsets across the dataset splits:
     - **Train Split (17 voices, 680 clips):** `Adam bựa`, `Thiện Minh`, `Hải Đăng`, `Thiền Tâm Đức`, `Minh Đức`, `Xuân Vĩnh`, `Thanh Bình`, `Trúc Ly`, `Ngọc Huyền`, `Ngọc Linh`, `Đoan Trang`, `Quang Sơn`, `Thái Sơn`, `Minh Triết`, `Adam`, `Thục Đoan`, `Mỹ Duyên`.
     - **Validation Split (4 voices, 160 clips):** `Mai Anh`, `Phạm Tuyên`, `Ngọc Trân`, `Thùy Dung`.
     - **Test Split (4 voices, 160 clips):** `Quỳnh Anh`, `Quốc Tuấn`, `Đức Trí`, `Kim Thanh`.
   * **Zero Voice Leakage:** Voices assigned to `test` never appear in `train` or `val`. Voices assigned to `val` never appear in `train`.

---

## 4. Reproducibility & RNG Control Policy

1. **Sampling Implementation:**
   * The local CPU ONNX runtime path in `OnnxV3LiteEngine._sample` uses `np.random.choice` on candidate top-$k$ logits.
   * Setting `numpy.random.seed(seed)` prior to inference provides deterministic, bit-exact autoregressive sampling.
2. **Per-Sample Deterministic Seed Assignment:**
   * Every utterance is assigned a unique, documented seed:
     $$\text{seed} = 100\,000 + 10\,000 \times \text{class\_idx} + 100 \times \text{voice\_idx} + \text{sample\_idx}$$
   * The exact seed is recorded in `manifest.csv` for every sample.
3. **Hardware Determinism Limits:**
   * Re-running synthesis on the same Python runtime (`.venv_vieneu`), OS, and ONNX Runtime version produces bit-exact identical WAV files (verified $\Delta = 0.0$ and matching SHA-256).
   * Minor floating-point variations across different CPU architectures (e.g., AVX-512 vs AVX2 FMA rounding) may alter low-significance bits if non-deterministic ONNX thread pools are invoked.

---

## 5. Intelligibility Status & Project Invariants

1. **Perceptual Status:**
   * All newly generated utterances are labeled:
     `subjective_intelligibility_status=PENDING_HUMAN_LISTENER_VERIFICATION`
   * High model confidence or objective signal health does not substitute for empirical human perceptual verification.
2. **Official Project Status:**
   * Remains strictly:
     **`NOT_VALIDATED_ON_NATURAL_SPEECH`**
