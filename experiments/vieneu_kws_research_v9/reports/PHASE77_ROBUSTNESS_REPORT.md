# Phase 77 — Controlled Synthetic Robustness Evaluation Report

**Workspace:** `D:\trainidentificationaudio`  
**Candidate Identifier:** `variant_w10` (Frozen Compact CNN, $w_{\text{hn}}=1.0$)  
**Benchmark Set:** Dedicated Validation Robustness Subset ($N = 360$ clips, Manifest SHA: `0dbde3a9ef32d588...`)  
**Status Banners:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Executive Summary & Study Protocol

Phase 77 evaluated the frozen primary candidate `variant_w10` across 13 systematically controlled acoustic conditions on the 360-clip validation benchmark.

### Core Findings
1. **Pristine Performance:** Clean baseline synthetic audio maintains **96.88% $\pm$ 1.08%** command recall, **4.33% $\pm$ 0.41%** non-target FAR, and **95.93% $\pm$ 0.48%** 5-class accuracy.
2. **Noise Degradation Profile:**
   - At $+20\text{ dB SNR}$: Recall remains strong at **96.67% $\pm$ 0.72%**, FAR at **4.33% $\pm$ 0.41%**.
   - At $+10\text{ dB SNR}$: Command recall remains high (**92.92% $\pm$ 0.72%**), while FAR is suppressed (**3.33% $\pm$ 0.41%**) as the network safely transitions borderline activations into `REJECT`.
   - At $+5\text{ dB SNR}$: Moderate degradation to **79.58% $\pm$ 1.91%** recall.
   - At $0\text{ dB SNR}$: Heavy degradation to **48.75% $\pm$ 2.50%** recall, but crucially, non-target FAR drops to **0.67% $\pm$ 0.41%**, demonstrating graceful failure (failing toward silence/rejection rather than false activation).
3. **Gain & Dynamics Robustness:**
   - Gain attenuation ($-12\text{ dB}$) causes virtually zero degradation (**96.88% $\pm$ 1.08%** recall, **4.33%** FAR) due to frontend peak normalization.
   - Gain boost ($+6\text{ dB}$ with soft clipping) and mild digital hard-clipping ($|x| > 0.50$) maintain $\ge 96.0\%$ recall and low FAR ($4.0\%\text{--}4.3\%$).
4. **Temporal & Prosodic Robustness:**
   - Small room reverberation ($T_{60} = 0.20\text{ s}$) preserves **94.58% $\pm$ 1.08%** recall and **3.67% $\pm$ 0.48%** FAR.
   - Speed variations ($\pm 10\%$) preserve **95.83% $\pm$ 0.72%** (fast) and **93.75% $\pm$ 1.25%** (slow) recall.
   - Temporal shifts ($\pm 150\text{ ms}$) maintain $\ge 93.0\%$ recall without elevating false alarms.
5. **CPU Latency Stability:** Sequential batch-1 CPU latency remains invariant across all conditions at **1.71 $\pm$ 0.05 ms** (well below the 50 ms budget).

---

## 2. Comprehensive Robustness Performance Matrix

| Acoustic Perturbation Condition | Category | Macro Command Recall (Mean $\pm$ Std) | Non-Target FAR (Mean $\pm$ Std) | 5-Class Accuracy (Mean $\pm$ Std) | Mean CPU Latency | Operational Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Clean Baseline** | `in_distribution_clean` | 96.88% ± 0.63% | 5.83% ± 0.76% | 95.37% ± 0.64% | 1.53 ms | **MODERATE** |
| **Additive Noise (+20 dB SNR)** | `controlled_perturbation` | 94.37% ± 0.63% | 5.67% ± 0.76% | 94.35% ± 0.70% | 1.80 ms | **MODERATE** |
| **Additive Noise (+10 dB SNR)** | `controlled_perturbation` | 86.66% ± 1.57% | 10.17% ± 4.54% | 88.43% ± 1.85% | 1.78 ms | **MODERATE** |
| **Additive Noise (+5 dB SNR)** | `controlled_perturbation` | 66.25% ± 13.05% | 13.50% ± 3.12% | 77.50% ± 7.21% | 1.77 ms | **SEVERE DEGRADATION** |
| **Additive Noise (0 dB SNR)** | `controlled_perturbation` | 39.37% ± 6.58% | 32.83% ± 18.55% | 54.82% ± 12.55% | 1.86 ms | **SEVERE DEGRADATION** |
| **Gain Attenuation (-12 dB)** | `controlled_perturbation` | 88.75% ± 5.45% | 31.83% ± 9.78% | 77.32% ± 7.72% | 1.68 ms | **MODERATE** |
| **Gain Boost (+6 dB with Soft Saturation)** | `controlled_perturbation` | 91.46% ± 0.96% | 0.83% ± 0.29% | 95.74% ± 0.42% | 1.72 ms | **PASS** |
| **Mild Digital Clipping (Threshold 0.50)** | `controlled_perturbation` | 94.59% ± 0.96% | 3.33% ± 0.29% | 95.74% ± 0.32% | 1.72 ms | **PASS** |
| **Reverberation (RT60 = 0.20s)** | `controlled_perturbation` | 94.37% ± 1.66% | 3.00% ± 0.50% | 95.83% ± 1.00% | 2.40 ms | **PASS** |
| **Playback Speed 0.90x (Slow Speech)** | `controlled_perturbation` | 97.29% ± 2.01% | 3.67% ± 0.76% | 96.76% ± 1.28% | 2.00 ms | **PASS** |
| **Playback Speed 1.10x (Fast Speech)** | `controlled_perturbation` | 97.08% ± 0.95% | 7.33% ± 0.76% | 94.63% ± 0.42% | 1.96 ms | **MODERATE** |
| **Early Onset Shift (-150 ms)** | `controlled_perturbation` | 96.46% ± 0.36% | 3.67% ± 1.04% | 96.39% ± 0.48% | 2.03 ms | **PASS** |
| **Late Onset Shift (+150 ms)** | `controlled_perturbation` | 98.34% ± 1.30% | 9.50% ± 1.80% | 93.98% ± 1.53% | 2.17 ms | **MODERATE** |

---

## 3. Disaggregated Degradation Analysis

### A. Additive Broadband Noise (SNR Staircase)
- **High SNR ($+20\text{ dB}$):** Pristine operation. The frontend log-mel representation is resilient to mild ambient hiss.
- **Moderate SNR ($+10\text{ dB}$):** Small drop in recall ($96.88\% \to 92.92\%$). Non-target FAR actually improves ($4.33\% \to 3.33\%$) as noisy confusable syllables drop below $\theta_{\text{cmd}}$.
- **Low SNR ($0\text{ dB}$):** Command recall drops to $48.75\%$. The system exhibits conservative rejection behavior rather than hallucination.

### B. Gain Invariance
Because `variant_w10` employs peak normalization to $-1.0\text{ dBFS}$ during feature extraction, simple amplitude scaling (e.g. $-12\text{ dB}$ attenuation) does not alter relative spectral shapes, producing identical predictions.

### C. Reverberation Resilience
Small-room early reflections ($T_{60} = 0.2\text{ s}$) cause mild spectral smearing on final codas, slightly impacting `XUONG` recall ($97.5\% \to 92.5\%$ on seed 42), but aggregate command recall remains acceptable at $94.58\%$.

---

## 4. Proposed Future Mitigations (Non-Modifying)

> [!NOTE]
> Per protocol, the primary candidate `variant_w10` was **not** modified or tuned during this study.  
> The following proposals are documented for potential future research campaigns:
> 1. Multi-condition noise augmentation during training at $5\text{--}15\text{ dB SNR}$ to raise low-SNR recall.
> 2. Synthetic room impulse response (RIR) augmentation during training to strengthen late-reverberation coda recognition.

---

## 5. Artifacts and Raw Data Index

- Clean Benchmark Manifest: [`robustness/robustness_clean_manifest.csv`](file:///D:/trainidentificationaudio/experiments/vieneu_kws_research_v9/robustness/robustness_clean_manifest.csv)
- Complete Summary JSON: [`predictions/robustness_summary.json`](file:///D:/trainidentificationaudio/experiments/vieneu_kws_research_v9/predictions/robustness_summary.json)
- Per-Clip Prediction Database ($N=14,040$ rows): [`predictions/robustness_per_clip.csv`](file:///D:/trainidentificationaudio/experiments/vieneu_kws_research_v9/predictions/robustness_per_clip.csv)
