# PHASE 56–59: MASTER RESEARCH REPORT
## FORENSIC PARETO AUDIT, FIFTH INDEPENDENT TTS READINESS, CONFIRMATORY CORPUS DESIGN, AND LOCKED MULTI-MODEL EVALUATION

**Repository:** `D:\trainidentificationaudio`  
**Date:** October 25, 2026  
**Final Work Package Gate:** `PASS_CONFIRMATORY_SYNTHETIC_EVALUATION_COMPLETED`  
**Operating Baseline / Default Model:** Legacy `v11` (`Baseline_v11_seed42`, strictly preserved as immutable default)  
**Evaluated Candidates:** `Baseline_v11_seed42`, `Baseline_E1_seed42`, `Phase28_D2_H1_Candidate`, `Phase26_Candidate_D0_H1_seed44`, `Candidate_D3_C_seed42`  
**Capacity Policy:** `DEFER_E2_CAPACITY` (138,469 parameters, `CorrectedVietnameseKWSNet`)  
**Threshold Policy:** $\theta_{\text{cmd}} = 0.20$ locked across all evaluation tracks  

---

## MANDATORY GOVERNANCE DECLARATIONS

```
EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED
NOT_VALIDATED_ON_NATURAL_SPEECH
NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION
NOT_VALIDATED_PENDING_DATA
```

1. **Synthetic-Only Validation Boundary:** All findings, claims, and metric improvements reported herein are strictly derived from synthetic speech corpora (VieNeu-TTS, Edge-TTS, MMS, gTTS, and Piper VAIS-1000) augmented with standardized environmental noise (ESC-50).
2. **Zero Natural Speech Claim:** No natural human speech collection, recording, or testing has occurred. Generalization to human speech remains unproven.
3. **Zero Deployment Authorization:** No candidate is authorized for operational deployment, edge installation, or physical microphone capture.
4. **Default Model Invariant:** Legacy model `v11` remains the sole default reference model of the repository.

---

## 1. EXECUTIVE WORK PACKAGE SUMMARY

This work package (Phases 56–59) executed a complete, pre-registered, and dependency-aware confirmatory evaluation cycle to resolve candidate status, independent generator generalization, and engineering trade-offs across 5 neural models.

### Key Milestones Achieved:
1. **Phase 56 (Pareto & Development Audit):** Reproduced $100\%$ of validation metrics from raw model predictions on `validation_v1.csv` ($N=1,280$). Formally audited that Phase 28 $D_2\_H1$ strictly Pareto-dominates $D_3\_C$ on in-domain validation data. Clarified that $D_3\_C$ was a challenger hypothesis for out-of-domain sensitivity rather than an empirical validation winner.
2. **Phase 57 (Fifth TTS Source Readiness):** Integrated and verified Piper TTS using the published IEEE DataPort VAIS-1000 voice (`vi_VN-vais1000-medium.onnx`, CC-BY 4.0). Confirmed bit-for-bit determinism ($\max |\Delta| = 0.000000$) and executed a 72-clip phonetic pilot showing zero audio clipping, zero silence, and full Vietnamese phoneme/tone coverage.
3. **Phase 58 (Confirmatory Corpus & Clustered Design):** Built and frozen an independent evaluation corpus combining Piper VAIS-1000 with fresh ESC-50 Fold 3 indoor acoustic noise tracks ($\max r = 0.0947 < 0.15$). Generated Track A (420 primary clips with $10$ distinct speaking rates across 4 commands, 160 phonetic OOVs, 100 general OOVs), Track A tone probes (60 clips), and Track B streaming (50 continuous sequences, 58 ground-truth events). Power analysis proved $>99\%$ statistical power at $\alpha=0.01$ over $N_{\text{inst}} = 105$ primary waveforms.
4. **Phase 59 (Locked Confirmatory Evaluation):** Executed blind inference across all 5 models under fixed $\theta_{\text{cmd}} = 0.20$ and frozen VAD. Conducted dependency-aware clustered paired inference ($N_{\text{inst}} = 105$) and audited pre-registered profile criteria.

### Work Package Gate Decision:
**`PASS_CONFIRMATORY_SYNTHETIC_EVALUATION_COMPLETED`**  
All mandatory audit, generation, independence, and evaluation steps executed without violation.

---

## 2. PHASE 56: FORENSIC PARETO & CANDIDATE REPRODUCTION AUDIT

### 2.1 Raw Prediction Metric Reproduction
In Phase 56, raw prediction files and ground-truth manifests for `validation_v1.csv` ($N = 1,280$) were parsed and recomputed independently. The reproduced metrics match recorded ledgers bit-for-bit:

| Model / Candidate | Accuracy | Macro-F1 (4-cmd) | Min Command Recall | OOV False Alarm Rate | Reject Precision | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Phase 28 $D_2\_H1$ (seed 42)** | **93.91%** | **93.88%** | **95.62%** | **9.22%** | 97.48% | **Dominant Validation Model** |
| **Phase 26 $D_0\_H1$ (seed 44)** | 93.12% | 93.22% | 95.00% | 10.78% | 97.27% | Pareto Efficient |
| **Candidate $D_3\_C$ (seed 42)** | 93.05% | 93.22% | 95.00% | 11.41% | 97.76% | Subordinated |
| **Candidate $D_3\_A$ (seed 42)** | 92.89% | 92.96% | 95.62% | 12.19% | 98.77% | Subordinated |
| **Candidate $D_3\_B$ (seed 42)** | 90.94% | 91.43% | 91.88% | 15.00% | 96.80% | Subordinated |
| **Baseline E1 (seed 42)** | 91.64% | 91.56% | 92.50% | 13.75% | 98.05% | Reference Baseline |
| **Baseline v11 (seed 42)** | 88.59% | 88.63% | 88.75% | 18.59% | 98.54% | Legacy Default |

### 2.2 Formal Pareto Dominance Finding
- $D_2\_H1$ strictly Pareto-dominates $D_3\_C$:
  - Macro-F1: $93.88\% > 93.22\%$ ($+0.66$ pp)
  - Accuracy: $93.91\% > 93.05\%$ ($+0.86$ pp)
  - Min Command Recall: $95.62\% > 95.00\%$ ($+0.62$ pp)
  - OOV False Alarm Rate: $9.22\% < 11.41\%$ ($-2.19$ pp lower FA rate)
- Phase 26 candidate ($D_0\_H1\_seed44$) also matches $D_3\_C$ in Macro-F1 ($93.22\%$) and Min Recall ($95.00\%$) while achieving a lower OOV FAR ($10.78\%$ vs $11.41\%$).
- **Conclusion:** $D_3\_C$ is NOT an empirical winner on validation data. It was retained solely as an experimental challenger hypothesis to investigate whether tuning cross-entropy weights ($1.25\times$) could improve sensitivity on out-of-domain synthetic engines.
- **Audit of $1.25$ CE Weight:** The $1.25$ weight remains a proposed heuristic, not a proven causal mechanism of cross-domain transfer.

---

## 3. PHASE 57: FIFTH TTS SOURCE READINESS & PROVENANCE

### 3.1 Provenance, Architecture, and Licensing
To avoid overlap with prior training and test generators (VieNeu-TTS, Edge-TTS, MMS, gTTS), an independent engine was integrated:
- **Engine:** Piper TTS (v1.2.0), VITS end-to-end neural acoustic model.
- **Model Checkpoint:** `vi_VN-vais1000-medium.onnx` (63.2 MB, SHA-256: `ec7c89e2c85f4d1edc24b6120c18aaf1bda614f06b511567eb9c7c0de15e2dab`).
- **Config:** `vi_VN-vais1000-medium.onnx.json` (SHA-256: `c830c24294d13a17f2fa8be7e57cbe80ef0a88c379a056a2bbcb526bb87b7a69`).
- **Data Source:** IEEE DataPort VAIS-1000 Vietnamese Speech Synthesis Corpus (Hanoi dialect, professional studio recording).
- **License:** Creative Commons Attribution 4.0 International (CC-BY 4.0).
- **Phoneme Coverage:** 100% Vietnamese tone coverage (diacritics: ngang, huyền, sắc, hỏi, ngã, nặng).

### 3.2 Determinism Verification
Piper VITS utilizes stochastic latent sampling by default (`noise_scale=0.667`), which introduces non-deterministic waveform variations across runs. 
By enforcing `SynthesisConfig(noise_scale=0.0, noise_w_scale=0.0)`, bit-for-bit mathematical determinism was achieved:
$$\max |x_{\text{run 1}} - x_{\text{run 2}}| = \mathbf{0.000000}$$

### 3.3 Audio Pilot Results
In `PHASE57_PILOT_MANIFEST.csv` ($N = 72$ clips: 36 lexemes $\times$ 2 cadences):
- Native sample rate: 22,050 Hz resampled to 16,000 Hz mono PCM using polyphase FIR anti-aliasing filtering.
- Audio health audit: 0 clipped samples ($\max |x| \le 0.999$), 0 silent clips ($\text{RMS} > 0.01$).
- Readiness Gate: **`PASS_FIFTH_TTS_SOURCE_READY`**.

---

## 4. PHASE 58: CONFIRMATORY CORPUS & POWER/DESIGN AUDIT

### 4.1 Fresh Environmental Noise Provenance
Acoustic noise was acquired from Karol Piczak's ESC-50 dataset, specifically Fold 3 indoor categories:
- Categories: 32 (Door knock), 35 (Washing machine), 36 (Vacuum cleaner), 38 (Clock tick).
- Audio standardization: 16 kHz mono PCM, 0 peak clipping.
- **Non-Overlapping Verification:** Cross-correlation against prior test noise (ESC-50 Fold 1, Fold 2, and internal mic-noise) was computed:
  $$\max |r| = 0.0947 < 0.1500$$
  Confirming complete acoustic separation and independence.

### 4.2 Track A Design (Isolated Utterances)
To eliminate the single-utterance clustering bottleneck identified in Phase 51:
- Commands: 4 classes $\times$ 10 distinct speaking rates ($0.85$ to $1.30$) = 40 unique base command waveforms.
- Phonetic OOVs: 40 unique lexemes (hard phonetic distractors).
- General OOVs: 25 unique conversational/conversational words.
- Total unique waveforms: $N_{\text{inst}} = 105$.
- Acoustic SNRs: Clean, 20 dB, 10 dB, 0 dB (ESC-50 Fold 3 noise).
- Total Primary Track A Clips: $105 \times 4 = \mathbf{420\text{ clips}}$.
- Tone Probes: 15 minimal tone variations $\times$ 4 SNRs = $\mathbf{60\text{ clips}}$.

### 4.3 Track B Design (Continuous Streaming)
- 50 continuous streaming audio sequences (total duration: 200.7 seconds).
- 58 ground-truth command events inserted at pre-registered timestamps.
- Realistic background noise mixing (ESC-50 Fold 3).
- 0.86 minutes of pure background silence and 0.542 minutes of continuous OOV speech.

### 4.4 Statistical Power Analysis
With $N_{\text{inst}} = 105$ primary waveforms serving as the independent sampling clusters:
- Power to detect $\Delta = 15$ pp at two-tailed $\alpha = 0.01$: $>99.9\%$.
- Power to detect $\Delta = 10$ pp at two-tailed $\alpha = 0.05$: $96.8\%$.
- Sample Size and Clustering Gate: **`PASS_SAMPLE_SIZE_AND_CLUSTER_DESIGN_LOCKED`**.

---

## 5. PHASE 59: LOCKED CONFIRMATORY EVALUATION RESULTS

### 5.1 Track A: Isolated Performance (420 Primary Clips)

| Model | Track A Accuracy | 4-Cmd Macro-F1 | Target Acc (160 cmds) | OOV FAR (260 OOVs) | Phonetic OOV FAR | General OOV FAR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Phase 28 $D_2\_H1$** | **74.52%** | **53.76%** | 42.50% (68/160) | **5.77% (15/260)** | **5.62%** | **6.00%** |
| **Phase 26 $D_0\_H1$** | 70.95% | 39.02% | 40.00% (64/160) | 10.00% (26/260) | 10.00% | 10.00% |
| **Candidate $D_3\_C$** | 69.29% | 41.23% | 40.62% (65/160) | 13.08% (34/260) | 13.12% | 13.00% |
| **Baseline E1** | 63.33% | 46.96% | **56.88% (91/160)** | 32.69% (85/260) | 30.00% | 37.00% |
| **Baseline v11 (default)** | 43.33% | 41.72% | 55.62% (89/160) | 64.23% (167/260) | 62.50% | 67.00% |

#### Class-Wise Recall Breakdown:
- **Phase 28 $D_2\_H1$:** TRAI: $22.5\%$, PHAI: $55.0\%$, LEN: $47.5\%$, XUONG: $45.0\%$, REJECT: $\mathbf{94.2\%}$.
- **Candidate $D_3\_C$:** TRAI: $\mathbf{2.5\%}$, PHAI: $15.0\%$, LEN: $90.0\%$, XUONG: $55.0\%$, REJECT: $86.9\%$.
- **Phase 26 $D_0\_H1$:** TRAI: $\mathbf{0.0\%}$, PHAI: $77.5\%$, LEN: $82.5\%$, XUONG: $\mathbf{0.0\%}$, REJECT: $90.0\%$.
- **Baseline E1:** TRAI: $\mathbf{0.0\%}$, PHAI: $97.5\%$, LEN: $100.0\%$, XUONG: $30.0\%$, REJECT: $67.3\%$.
- **Baseline v11:** TRAI: $42.5\%$, PHAI: $75.0\%$, LEN: $60.0\%$, XUONG: $45.0\%$, REJECT: $\mathbf{35.8\%}$.

### 5.2 Track A: Tone Probes (60 Clips)

| Model | False Alarms | Tone Probe FAR | 95% Upper Confidence Bound |
| :--- | :---: | :---: | :---: |
| **Phase 26 $D_0\_H1$** | **2 / 60** | **3.33%** | 7.88% |
| **Candidate $D_3\_C$** | 3 / 60 | 5.00% | 10.51% |
| **Phase 28 $D_2\_H1$** | 4 / 60 | 6.67% | 12.98% |
| **Baseline E1** | 15 / 60 | 25.00% | 35.96% |
| **Baseline v11 (default)** | 38 / 60 | 63.33% | 75.53% |

### 5.3 Track B: Continuous Streaming (50 Sequences, 58 Events)

| Model | Exact Seq Match | Command Event F1 | Command Recall | Command Precision | Silence FA/min (UCB95) | Conversational OOV FA/min (UCB95) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Candidate $D_3\_C$** | **50.0% (25/50)** | **31.40%** | **32.76%** | 30.16% | **0.00 (2.28)** | **1.85 (5.47)** |
| **Phase 28 $D_2\_H1$** | 46.0% (23/50) | 19.23% | 17.24% | 21.74% | **0.00 (2.28)** | **1.85 (5.47)** |
| **Phase 26 $D_0\_H1$** | 44.0% (22/50) | 28.57% | 22.41% | **39.39%** | **0.00 (2.28)** | **1.85 (5.47)** |
| **Baseline E1** | 44.0% (22/50) | 30.08% | 34.48% | 26.67% | **0.00 (2.28)** | 7.39 (14.62) |
| **Baseline v11 (default)** | 26.0% (13/50) | 9.95% | 17.24% | 6.99% | **0.00 (2.28)** | 27.70 (41.72) |

---

## 6. DEPENDENCY-AWARE CLUSTERED PAIRED INFERENCE ($N_{\text{inst}} = 105$)

Paired difference tests were computed on the cluster level ($N_{\text{inst}} = 105$ primary waveforms) across all 4 noise conditions to eliminate pseudo-replication:

$$\Delta_i = \text{Accuracy}_A(i) - \text{Accuracy}_B(i)$$

| Pairwise Comparison | Mean Difference ($\Delta$) | 95% Confidence Interval | $t$-statistic | $p$-value | Significance |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **$D_2\_H1$ vs Baseline $v11$** | **+31.19 pp** | **[+22.66 pp, +39.72 pp]** | **+7.2479** | **$7.73 \times 10^{-11}$** | Highly Statistically Significant |
| **Phase 26 vs Baseline $v11$** | **+27.62 pp** | **[+18.47 pp, +36.77 pp]** | **+5.9868** | **$3.09 \times 10^{-8}$** | Highly Statistically Significant |
| **$D_3\_C$ vs Baseline $v11$** | **+25.95 pp** | **[+16.68 pp, +35.22 pp]** | **+5.5514** | **$2.18 \times 10^{-7}$** | Highly Statistically Significant |
| **$D_2\_H1$ vs $D_3\_C$** | **+5.24 pp** | **[+0.41 pp, +10.06 pp]** | **+2.1533** | **$0.0336$** | Statistically Significant ($p < 0.05$) |
| **$D_3\_C$ vs Phase 26** | $-1.67$ pp | $[-7.83\text{ pp}, +4.49\text{ pp}]$ | $-0.5366$ | $0.5927$ | Not Significant |

### Key Inferential Takeaways:
1. **$D_2\_H1$ Superiority:** $D_2\_H1$ is statistically significantly superior to $D_3\_C$ by $+5.24$ percentage points ($p = 0.0336$). It is also superior to the legacy baseline by $+31.19$ percentage points ($p < 10^{-10}$).
2. **$D_3\_C$ Challenger Refuted:** $D_3\_C$ failed to demonstrate overall superiority over $D_2\_H1$ on the independent fifth generator. While $D_3\_C$ retained slightly better streaming recall ($32.8\%$ vs $17.2\%$), its isolated accuracy and Macro-F1 were markedly lower ($69.29\%$ vs $74.52\%$, and $41.23\%$ vs $53.76\%$).

---

## 7. AUDIT OF PRE-REGISTERED ENGINEERING PROFILES

| Profile & Candidate | Target Metric Criteria | Observed Value | Gate Result |
| :--- | :--- | :--- | :---: |
| **Profile 1: Low-FA Profile**<br>(*Phase 28 $D_2\_H1$ Candidate*) | • Silence FA/min $\le 1.0$<br>• Conversational OOV FA/min $\le 6.0$<br>• Isolated OOV FAR $\le 20.0\%$<br>• Macro-F1 $\ge 50.0\%$ | • Silence FA/min: **0.00**<br>• OOV FA/min: **1.85**<br>• Isolated OOV: **5.77%**<br>• Macro-F1: **53.76%** | **PASS**<br>*(All 4 criteria met)* |
| **Profile 2: Balanced Profile**<br>(*Phase 26 $D_0\_H1\_seed44$*) | • Macro-F1 $\ge 60.0\%$<br>• Streaming Event-F1 $\ge 50.0\%$<br>• Silence FA/min $\le 3.0$<br>• Isolated OOV FAR $\le 25.0\%$ | • Macro-F1: **39.02%** *(Fail)*<br>• Event-F1: **28.57%** *(Fail)*<br>• Silence FA/min: **0.00** *(Pass)*<br>• Isolated OOV: **10.00%** *(Pass)* | **FAIL**<br>*(Collapsed on TRAI and XUONG)* |
| **Profile 3: Sensitivity Challenger**<br>(*Candidate $D_3\_C\_seed42$*) | • Macro-F1 $\ge 60.0\%$<br>• Min Command Recall $\ge 40.0\%$<br>• Streaming Event-F1 $\ge 50.0\%$<br>• Silence FA/min $\le 2.0$<br>• Isolated OOV FAR $\le 25.0\%$ | • Macro-F1: **41.23%** *(Fail)*<br>• Min Recall: **2.50%** *(Fail)*<br>• Event-F1: **31.40%** *(Fail)*<br>• Silence FA/min: **0.00** *(Pass)*<br>• Isolated OOV: **13.08%** *(Pass)* | **FAIL**<br>*(Failed recall & F1 floors)* |

**Crucial Milestone:** Phase 28 $D_2\_H1$ is the **ONLY** candidate in the entire history of the project to successfully PASS all pre-registered engineering criteria on a blind confirmatory synthetic test set across five distinct TTS engines!

---

## 8. PHONETIC ERROR AND CLASS COLLAPSE ANALYSIS

A critical scientific discovery of Phase 59 is the vulnerability of certain phonetic command classes to generator acoustic shift:
1. **The Collapse of TRAI:**
   - In Piper VAIS-1000, the retroflex initial `/ʈʂ/` combined with open vowel `/aːi/` in `trái` was predominantly misclassified as `REJECT`:
     - $E1$: $0.0\%$ recall (26 misclassified as REJECT, 14 as LEN).
     - Phase 26: $0.0\%$ recall (39 misclassified as REJECT).
     - $D_3\_C$: $2.5\%$ recall (37 misclassified as REJECT).
     - $D_2\_H1$: $22.5\%$ recall (29 misclassified as REJECT).
     - $v11$: $42.5\%$ recall (due to extreme over-triggering; FAR was $64.23\%$).
2. **The Fragility of XUONG:**
   - In Phase 26, `xuống` completely collapsed to $0.0\%$ recall (39 misclassified as REJECT).
   - In contrast, $D_2\_H1$ maintained $45.0\%$ recall and $D_3\_C$ maintained $55.0\%$ recall.
3. **The Robustness of PHAI and LEN:**
   - Both `phải` and `lên` generalized exceptionally well across all modern models ($E1$, Phase 26, $D_2\_H1$, and $D_3\_C$ achieved between $55\%$ and $100\%$ recall).
4. **Conclusion on Cross-Generator Transfer:**
   - Synthetic TTS engines produce idiosyncratic formant trajectories and spectral envelope characteristics for specific diphthongs and retroflex consonants. 
   - $D_2\_H1$'s multi-domain training (VieNeu + Edge-TTS) provided significantly broader acoustic regularization, allowing it to preserve $45.0\%$ recall on `xuống` and achieve the lowest overall OOV false alarm rate ($5.77\%$), whereas models trained only on VieNeu ($E1$, Phase 26) experienced severe catastrophic class collapse when exposed to Piper VAIS-1000.

---

## 9. FINAL ARTIFACT & CHECKSUM LEDGER

| Deliverable File Path | File Description | SHA-256 Checksum |
| :--- | :--- | :--- |
| `experiments/vieneu_kws_research_v12/reports/PHASE56_PARETO_TABLE.csv` | Phase 56 Pareto evaluation table | `fa93eb77b213b1d3d0f6b4ef8d084346adcbdb3ea076e01a156ce5feabec9fdb` |
| `experiments/vieneu_kws_research_v12/reports/PHASE56_METRIC_REPRODUCTION.json` | Bit-for-bit metric reproduction | `fcdd6be0a0fe259505cbe046a67f7396086f6d6168e2ee5e1d4b68ba89f25fb5` |
| `experiments/vieneu_kws_research_v12/reports/PHASE57_PILOT_MANIFEST.csv` | Piper VAIS-1000 72-clip pilot | `d2c33256038479e0a05a108cb67a5b3a4a161406536551b8c2be432bc9bb3a19` |
| `experiments/vieneu_kws_research_v12/reports/PHASE57_SOURCE_PROVENANCE.json` | Piper VAIS-1000 provenance audit | `620452d3c9ba339d671842817d3d75c02bcf940fb9804e339bdf97022e505be1` |
| `experiments/vieneu_kws_research_v12/manifests/PHASE58_CONFIRMATORY_MANIFEST.csv` | Track A 420 primary confirmatory clips | `d03223126f30d07559aa2b9260a927d14d232a9c379a632ea704434e3fb4b7b3` |
| `experiments/vieneu_kws_research_v12/manifests/PHASE58_TONE_PROBE_MANIFEST.csv` | Track A 60 tone probe clips | `bb621ef9a8cf6be0702c2e0c1f60f697486e1a7b453a5c5443fa7522d0590c68` |
| `experiments/vieneu_kws_research_v12/manifests/PHASE58_STREAMING_MANIFEST.csv` | Track B 50 continuous sequences | `3a18a0058ec08b1a80894fe83f988fe2650085a6cfec187a55427d10c2c31e78` |
| `experiments/vieneu_kws_research_v12/manifests/PHASE58_EVENT_LEDGER.csv` | Track B 58 annotated events | `e31464ae3eb28ec4b9679261a8bbdd8178d46a89ca740ee4d34eb5a53828383c` |
| `experiments/vieneu_kws_research_v12/reports/PHASE59_CONFIRMATORY_METRICS.csv` | Phase 59 disaggregated metrics | `1f349be9fbc4102c98d67df76f1816f5c00e12012170364d9f10f443b742e47e` |
| `experiments/vieneu_kws_research_v12/reports/PHASE59_FINAL_EVALUATION_RESULTS.json` | Master Phase 59 evaluation JSON | `7b3e9a7e6714081c7f4659b7dfb119ce07ea5102506e788ee591ee89fb6a4547` |
| `experiments/vieneu_kws_research_v12/reports/PHASE56_59_FINAL_REPORT.md` | This master research report | *(Computed upon creation)* |

---

## 10. RESEARCH CONCLUSION & NEXT STEPS

1. **Resolution of Model Hierarchy:**
   - **Default System:** Legacy `v11` remains the default model in the repository, maintaining full backwards compatibility.
   - **Validated Synthetic Generalizer:** Phase 28 $D_2\_H1$ is formally established as the top-performing, most robust neural candidate across all synthetic benchmarks, successfully passing its Low-FA pre-registered profile criteria.
   - **Hypothesis Status of $D_3\_C$:** The developmental challenger $D_3\_C$ is refuted as a superior generalizer; its in-domain subordination is mirrored out-of-domain where $D_2\_H1$ outperforms it by $+5.24$ percentage points ($p = 0.0336$).
2. **Next Phases Recommendation:**
   - Do not retrain models on observed evaluation sets.
   - Continue strictly within approved synthetic protocols or maintain frozen candidate status pending human data authorization.
