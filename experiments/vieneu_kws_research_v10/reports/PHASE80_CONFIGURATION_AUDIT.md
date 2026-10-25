# Phase 80 — Configuration Audit Report
## Point-by-Point Technical Audit Resolving Seven Governance Questions

**Workspace:** `D:\trainidentificationaudio`  
**Candidate Identifier:** `variant_w10`  
**Status Banners:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## Question 1: Dataset Splits Used in Phase 74 vs. Phase 77
> **Question:** Does Phase 74 use the new 720-clip locked test split while Phase 77 uses a validation split or another dataset?

**Audit Answer: YES.**
- **Phase 74** evaluated exclusively on `experiments/vieneu_kws_research_v8/data/splits/test_manifest.csv` ($N = 720$ clips, SHA-256: `79e9aa0b20bb921d348b55101af78157276cb56b1059fabb129a0d745d1da2e4`).
- **Phase 77** evaluated on `experiments/vieneu_kws_research_v9/robustness/robustness_clean_manifest.csv` ($N = 360$ clips, SHA-256: `0dbde3a9ef32d5889956fa53a61a4bd74f307430ac88b28208b14e74de0d7289`), which was derived from the validation partition `experiments/vieneu_kws_research_v8/data/splits/val_manifest.csv`.
- The two datasets have strictly zero clip overlap, different speaker cohorts (4 test voices vs. 4 validation voices), and different non-target denominators (560 vs. 200).

---

## Question 2: Model Checkpoints, Seeds, Class Mappings, and Thresholds
> **Question:** Are the exact same checkpoints, seeds, class mappings, and rejection threshold used in both reports?

**Audit Answer: YES, EXACTLY IDENTICAL.**
- **Checkpoints:**
  - Seed 42: `variant_w10_seed42.pt` (SHA-256: `7374af469a4bc4df558448a04bfe68c17a941fc36528d2d9b62d29e71f496ee8`)
  - Seed 43: `variant_w10_seed43.pt` (SHA-256: `77f468c68b261de520bcdaaa5c4fae85746f33d45efb00249ae5279f337f7690`)
  - Seed 44: `variant_w10_seed44.pt` (SHA-256: `bebab580e4caa693c12febe5c2759e6fbca4786d790d970e7e1c14041b375b48`)
- **Parameter Count:** Exactly **138,469** parameters in `CorrectedVietnameseKWSNet`.
- **Class Mappings:** Index 0 = `TRAI`, Index 1 = `PHAI`, Index 2 = `LEN`, Index 3 = `XUONG`, Index 4 = `REJECT`.
- **Rejection Thresholds:** $\theta_{\text{cmd}} = 0.40$ (Seed 42), $0.50$ (Seed 43), $0.35$ (Seed 44).

---

## Question 3: Preprocessing, Padding, Cropping, and Feature Extraction
> **Question:** Are the same waveform normalization, padding, cropping, feature extraction, and class-selection rules used?

**Audit Answer: YES, 100% PARITY.**
- **Feature Extractor:** `LogMelFrontend` (40 mel bins, 25 ms FFT window / 400 samples, 10 ms hop / 160 samples, slaney mel scale, log offset $10^{-6}$).
- **Padding / Cropping:** Right-padding with constant zeros to exactly 16,000 samples (1.0 s) or truncating to 16,000 samples.
- **Normalization:** Waveform amplitudes are scaled down only if $\max(|x|) > 1.0$; low-energy noise floors are preserved without artificial amplification.
- **Decision Logic:** $\text{Class} = \arg\max_{c \in [0..3]} p_c$ if $\max_{c \in [0..3]} p_c \ge \theta_{\text{cmd}}$ else `REJECT`.

---

## Question 4: Authenticity of Phase 77 "Clean Baseline"
> **Question:** Does “clean baseline” in Phase 77 actually contain unmodified clean waveforms?

**Audit Answer: YES.**
- Inspection of `run_phase77_robustness.py` confirms that for condition `clean_baseline`, the audio tensor is passed directly to inference without any additive noise, gain adjustment, filtering, clipping, or temporal perturbation.
- Waveform SHA-256 digests in `robustness_clean_manifest.csv` match the source validation `.wav` files on disk bit-for-bit.

---

## Question 5: FAR Denominators and Accepted/Rejected Definitions
> **Question:** Are FAR denominators and accepted/rejected clip definitions identical?

**Audit Answer: THE DEFINITIONS ARE IDENTICAL, BUT THE DENOMINATORS DIFFER BY DATASET DESIGN.**
- **Definition Parity:** In both phases, False Acceptance (FA) is defined as any non-target clip (`UNKNOWN` speech or `SILENCE`) that produces a predicted command class (`TRAI`, `PHAI`, `LEN`, or `XUONG`). Any non-target clip that predicts `REJECT` is scored as a True Rejection.
- **Denominator Difference:**
  - Phase 74 Test Set has **560 non-target clips** (480 UNKNOWN + 80 SILENCE).
  - Phase 77 Validation Set has **200 non-target clips** (120 UNKNOWN + 80 SILENCE).
  - Because the denominator in Phase 77 is $2.8\times$ smaller, an equal number of raw false acceptances yields a $2.8\times$ higher percentage FAR.

---

## Question 6: Statistical Units and Cross-Seed Aggregation
> **Question:** Are the cross-seed means and standard deviations calculated from the same statistical units?

**Audit Answer: YES.**
- In both reports, the evaluation units are the 3 trained models (seeds 42, 43, 44) evaluated on the fixed dataset partition.
- Cross-seed summary metrics are computed using the standard sample mean $\bar{x} = \frac{1}{3}\sum_{s} x_s$ and sample standard deviation $s = \sqrt{\frac{1}{2}\sum_s (x_s - \bar{x})^2}$ ($N-1=2\text{ degrees of freedom}$).
- Both reports refrain from pooling repeated measurements into a single inflated $N$.

---

## Question 7: CPU Latency Timing Methodology
> **Question:** Are the latency measurements based on the same timing boundaries and CPU configuration?

**Audit Answer: YES.**
- Both scripts measure single-clip sequential forward pass execution using Python's `time.perf_counter()`.
- Both enforce `torch.set_num_threads(4)` on batch size 1 on the identical Intel x86_64 CPU.
- Phase 74 reported mean latency of **1.73 $\pm$ 0.03 ms**; Phase 77 reported **1.62 $\pm$ 0.02 ms**. Both confirm that inference requires $<2.0\text{ ms}$, well within the 50 ms budget.

---

## Summary Certification
Configuration parity is verified. The numerical differences reflect legitimate dataset partition characteristics, not modeling or implementation discrepancies.
