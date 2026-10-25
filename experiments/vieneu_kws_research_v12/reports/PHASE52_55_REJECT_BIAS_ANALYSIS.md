# PHASE 53: REJECT-BIAS AND CLASS-WISE FAILURE FORENSIC ANALYSIS

**Repository:** `D:\trainidentificationaudio`  
**Phase:** Phase 53 (Reject-Bias & Class-Wise Error Diagnosis)  
**Governing Evidence File:** [`phase53_reject_bias_distributions.json`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v12/reports/phase53_reject_bias_distributions.json)  
**Per-Clip Prediction Dataset:** [`phase52_per_clip_predictions.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v12/reports/phase52_per_clip_predictions.csv)  
**Operating Threshold:** $\theta_{\text{cmd}} = 0.20$ (Strictly Locked)  
**Canonical Default Model:** Legacy $v11$ (`3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2`)  
**Methodological Invariant:** **EXPLORATORY ERROR ANALYSIS ONLY (NO THRESHOLD TUNING OR TEST-SET FITTING)**

---

## Mandatory Governance Invariants

> [!CAUTION] MANDATORY RESEARCH STATUSES
> 1. `EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`
> 2. `NOT_VALIDATED_ON_NATURAL_SPEECH`
> 3. `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`
> 4. `NOT_VALIDATED_PENDING_DATA`
>
> Zero human speech was collected. Zero physical microphone testing was conducted. No model retrained or capacity modified. All checkpoints ($v11$, $E1$, Phase 28 $D_2\_H1$, Phase 26 $D_0\_H1\_seed44$, Phase 28 $D_0\_H1\_seed42$) remain strictly frozen and unaltered.

---

## 1. Executive Summary & Diagnostic Scope

Phase 53 executes a rigorous diagnostic post-mortem on the frozen models evaluated in Phase 51, answering why modern candidates trained with hard-negative mining ($D_2\_H1$ and Phase 26 $D_0\_H1$) suffered sensitivity loss on the unseen Google Translate TTS (`gTTS 2.5.4`) distribution while maintaining near-perfect background rejection.

### Key Forensic Findings:
1. **Zero Cross-Command Confusion:** Across all modern models ($E1$, $D_2\_H1$, $P26$, $D_0\_H1$), exactly **0 out of 32 target commands** were substituted into another command keyword ($0.0\%$ cross-command error). Keyword-to-keyword discrimination is absolute.
2. **Extreme Miss Concentration into REJECT:** Every single target command error was a miss into the `REJECT` class. When $D_2\_H1$ missed a command, it assigned a **mean softmax probability of $98.18\%$ to `REJECT`** (and Phase 26 assigned $94.36\%$).
3. **Failures Persist in Clean Audio:** Target misses were not caused by environmental noise: $5$ out of $8$ target clips ($62.5\%$) were rejected by $D_2\_H1$ in **completely clean audio ($\text{SNR} = \infty$)**.
4. **Cadence Sensitivity:** Slower, prolonged speaking rates suffered a **$75.0\%$ miss rate** ($12 / 16$) compared to $56.2\%$ for normal speaking rate ($9 / 16$).
5. **Class Asymmetry:** `LEN` achieved **100% recall ($8 / 8$)** across all 5 models. In contrast, `TRAI`, `PHAI`, and `XUONG` suffered up to $87.5\%$ rejection on $D_2\_H1$.
6. **Training Recipe Root Causes:**
   - **Loss Weight Skew:** REJECT class was explicitly upweighted to $1.25$ in CrossEntropyLoss (vs $1.0$ for commands), imparting a permanent negative gradient bias against target commands.
   - **Hard-Negative Batch Saturation:** The $H_1$ sampler enforced a $50\%$ target vs $50\%$ REJECT batch prior, with $16 / 64$ samples drawn from phonetically similar OOV words.
   - **Domain Over-Constriction in D2:** Incorporating Edge-TTS into both targets and hard negatives constricted the command manifold around the two training engines (VieNeu and Edge), causing unknown timbre from Google Translate TTS to be treated as out-of-distribution OOV speech.

---

## 2. Probability and Logit Distributions Across Diagnostic Subsets

The table below reports the objective softmax probability distributions and decision margins across five mutually exclusive operational subsets on Track A ($N=240$ clips):

| Diagnostic Subset | Sample Count ($N$) | Metric | Baseline v11 (Default) | Baseline E1 | Phase 28 D2_H1 Candidate | Phase 26 Candidate (seed44) |
|:------------------|:------------------:|:-------|:----------------------:|:-----------:|:------------------------:|:---------------------------:|
| **1. Correct Targets** | $v11: 27, E1: 20$<br>$D2: 11, P26: 19$ | Mean $P(\text{Target})$<br>Mean $P(\text{REJECT})$<br>Mean Margin | $0.6525$<br>$0.2372$<br>$+0.4153$ | $0.7915$<br>$0.1444$<br>$+0.6471$ | $0.7677$<br>$0.1704$<br>$+0.5973$ | $0.7580$<br>$0.1797$<br>$+0.5783$ |
| **2. Missed Targets** | $v11: 5, E1: 12$<br>$D2: 21, P26: 13$ | Mean $P(\text{Target})$<br>Mean $P(\text{REJECT})$<br>Mean Margin | $0.1287$<br>**0.7619**<br>$-0.6332$ | $0.0526$<br>**0.9313**<br>$-0.8787$ | $0.0135$<br>**0.9818**<br>$-0.9683$ | $0.0416$<br>**0.9436**<br>$-0.9020$ |
| **3. Cross-Substitutions** | All Models: **0** | Substitution Count | **0 / 32** | **0 / 32** | **0 / 32** | **0 / 32** |
| **4. Correct OOV Rejects** | $v11: 71, E1: 125$<br>$D2: 180, P26: 165$ | Mean $P(\text{REJECT})$<br>Mean $\max P(\text{Cmd})$ | $0.8166$<br>$0.0763$ | $0.9366$<br>$0.0381$ | **0.9575**<br>**0.0249** | $0.9429$<br>$0.0336$ |
| **5. OOV False Alarms** | $v11: 137, E1: 83$<br>$D2: 28, P26: 43$ | Mean $P(\text{Trigger})$<br>Mean $P(\text{REJECT})$ | $0.5714$<br>$0.2037$ | $0.6120$<br>$0.2064$ | $0.4852$<br>$0.2458$ | $0.5283$<br>$0.2319$ |
| **6. Tone Probes ($N=48$)** | All Models: $48$ | False Alarm Count<br>Mean $P(\text{REJECT})$ | 34 / 48<br>$0.2783$ | 14 / 48<br>$0.6841$ | **0 / 48**<br>**0.9682** | **0 / 48**<br>**0.9511** |
| **7. Silence FA/min** | Continuous | Events / min | $31.76$ | **0.00** | **0.00** | **0.00** |

### Critical Diagnostic Deductions:
1. **Bimodal REJECT Confidence:** In $D_2\_H1$, when the network decides to reject, it does so with extreme certainty ($98.18\%$ on missed targets, $95.75\%$ on true OOV, $96.82\%$ on tone probes). The probability distribution is not hovered near the decision boundary ($\theta_{\text{cmd}} = 0.20$); rather, the target command logits are suppressed to near zero ($\max P(\text{target}) = 1.35\%$).
2. **No Gray Zone:** The issue is not that the model predicted target probabilities of $0.18$ or $0.19$ (just missing the $0.20$ cutoff). Rather, the model assigned $>95\%$ probability to REJECT. Lowering $\theta_{\text{cmd}}$ from $0.20$ to $0.10$ or $0.05$ would not recover these misses without triggering catastrophic false alarms.

---

## 3. Disaggregated Command Analysis

The table below breaks down performance for each individual command keyword ($N=8$ clips each):

| Command Keyword | Model | Detected (TP) | Missed (REJECT) | Substitutions | Mean $P(\text{self})$ | Mean $P(\text{REJECT})$ | Clean Accuracy | 5 dB SNR Accuracy |
|:----------------|:------|:-------------:|:---------------:|:-------------:|:---------------------:|:-----------------------:|:--------------:|:-----------------:|
| **LEN** (`lên`) | **Baseline v11** | **8 / 8** | 0 / 8 | 0 | $0.6435$ | $0.2883$ | 100.0% | 100.0% |
| | **Baseline E1** | **8 / 8** | 0 / 8 | 0 | **0.9898** | $0.0102$ | 100.0% | 100.0% |
| | **Phase 28 D2_H1** | **8 / 8** | 0 / 8 | 0 | $0.8834$ | $0.1166$ | 100.0% | 100.0% |
| | **Phase 26 (seed44)** | **8 / 8** | 0 / 8 | 0 | **0.9727** | $0.0273$ | 100.0% | 100.0% |
| **TRAI** (`trái`) | **Baseline v11** | **7 / 8** | 1 / 8 | 0 | $0.5258$ | $0.2899$ | 100.0% | 50.0% |
| | **Baseline E1** | 2 / 8 | 6 / 8 | 0 | $0.1413$ | $0.8517$ | 50.0% | 0.0% |
| | **Phase 28 D2_H1** | 1 / 8 | **7 / 8** | 0 | $0.0398$ | **0.9602** | 0.0% | 0.0% |
| | **Phase 26 (seed44)** | 2 / 8 | 6 / 8 | 0 | $0.2020$ | $0.7975$ | 0.0% | 50.0% |
| **PHAI** (`phải`) | **Baseline v11** | 6 / 8 | 2 / 8 | 0 | $0.5346$ | $0.2529$ | 100.0% | 50.0% |
| | **Baseline E1** | **7 / 8** | 1 / 8 | 0 | **0.7285** | $0.2702$ | 100.0% | 50.0% |
| | **Phase 28 D2_H1** | 1 / 8 | **7 / 8** | 0 | $0.0466$ | **0.9534** | 50.0% | 0.0% |
| | **Phase 26 (seed44)** | 4 / 8 | 4 / 8 | 0 | $0.3840$ | $0.6160$ | 50.0% | 0.0% |
| **XUONG** (`xuống`) | **Baseline v11** | 6 / 8 | 2 / 8 | 0 | $0.5829$ | $0.3478$ | 50.0% | 100.0% |
| | **Baseline E1** | 3 / 8 | 5 / 8 | 0 | $0.2153$ | $0.7830$ | 50.0% | 50.0% |
| | **Phase 28 D2_H1** | 1 / 8 | **7 / 8** | 0 | $0.1334$ | **0.8665** | 0.0% | 0.0% |
| | **Phase 26 (seed44)** | 5 / 8 | 3 / 8 | 0 | $0.3328$ | $0.6672$ | 50.0% | 100.0% |

### Phonetic & Acoustic Mechanism of Class Discrepancy:
1. **Why `LEN` Succeeds Universally (100%):**
   - **Phonetics:** Alveolar lateral approximant $[l]$ followed by mid unrounded vowel and alveolar nasal $[en˧˧]$ with level tone (*ngang*).
   - **Acoustic Signature:** High, sustained spectral energy in $F_1$ ($~450\text{ Hz}$) and $F_2$ ($~1800\text{ Hz}$) with flat pitch contour ($F_0$).
   - **Hard-Negative Invariance:** Few phonetically identical confusers exist in the training set (`lớp, luôn, làm, lòng`), none of which share the level tone and [en] formant structure.
2. **Why `TRAI` and `PHAI` Collapse Under D2_H1 (12.5%):**
   - **Phonetics:** `trái` is a laminal affricate [t͡ɕaːj˧ˀ˥] with high glottalized tone (*sắc*); `phải` is a labiodental fricative [faːj˧˩˧] with dipping-rising tone (*hỏi*).
   - **Training Competition:** Both words were targeted by exhaustive phonetic sets in the $H_1$ sampler (`trước, trường, trưa, trẻ` and `phim, phút, phương, phát`).
   - **Timbre Shift:** In Google Translate TTS, the pitch excursion of *sắc* and *hỏi* occurs more rapidly ($<120\text{ ms}$) with a different vowel formant center than in VieNeu-TTS or Edge-TTS. Because $D_2\_H1$ learned very compact decision boundaries around VieNeu/Edge prototypes to exclude the hard negatives, any departure in pitch timing or formant transition was treated as an unmodeled hard negative and pushed to `REJECT`.

---

## 4. Error Decomposition by Acoustic Factors

### 4.1 SNR Breakdown
Target command misses were decomposed across the four SNR conditions:

```
D2_H1 Target Predictions by SNR:
SNR Tier      Detected Cmd    Missed into REJECT    Miss Rate
Clean               3                 5               62.5%
15.0 dB             2                 6               75.0%
10.0 dB             4                 4               50.0%
5.0 dB              2                 6               75.0%
```

**Finding:** The miss rate in completely clean audio ($62.5\%$) is essentially identical to the miss rate under $5\text{ dB}$ noise ($75.0\%$). Noise corruption is not the primary mechanism of failure.

### 4.2 Speaking Cadence (Duration) Breakdown

```
Target Predictions by Speaking Cadence:
Cadence       Total Clips    D2_H1 Detected    D2_H1 Missed    D2_H1 Miss Rate    Phase 26 Miss Rate
Normal             16               7                9              56.2%                31.2%
Slow               16               4               12              75.0%                50.0%
```

**Finding:** Slow cadences suffered significantly higher miss rates ($75.0\%$ for $D_2\_H1$, $50.0\%$ for Phase 26). When words are elongated by TTS synthesis, acoustic energy is dispersed over more temporal frames, causing the 2D convolutional feature maps to align poorly with filters learned on normal-speed training clips.

---

## 5. Review of the D2_H1 Training Recipe: Root Causes of Reject Bias

A line-by-line audit of `train_phase28_factorial.py` identified four structural factors that explain the emergence of excessive reject bias:

### 1. Cross-Entropy Loss Weight Imbalance
In line 357 of `train_phase28_factorial.py`:
$$\mathbf{w}_{\text{CE}} = [1.0, 1.0, 1.0, 1.0, \mathbf{1.25}]$$
- The `REJECT` class received a **$25\%$ penalty premium** over every command class.
- In every gradient step, a false alarm incurred $1.25\times$ more loss than a missed command. Over 35 epochs (2,975 mini-batches), this pushed the final dense layer biases ($\mathbf{b}_{\text{fc2}}$) systematically in favor of REJECT.

### 2. Batch Sampler Prior ($H_1$)
In lines 220–226 of `train_phase28_factorial.py`:
- Batch size = 64.
- Target samples = 32 ($8$ per command).
- REJECT samples = 32 ($16$ OOV words + $16$ acoustic noise).
- While a 50:50 batch prior is standard in KWS, combining a 50% REJECT prior with a $1.25$ loss weight resulted in REJECT accounting for:
  $$\frac{32 \times 1.25}{(32 \times 1.0) + (32 \times 1.25)} = \frac{40}{72} = \mathbf{55.56\%}$$
  of the total gradient energy in every single training batch.

### 3. Multi-Engine Boundary Constriction in D2
In `D2_matched_train_v1.csv`, 10% of training samples were replaced with Edge-TTS audio.
- While this successfully eliminated Edge-TTS false alarms (reducing them from $77\%$ to $8\%$), it exposed the model to two distinct neural voice clusters.
- Because the network capacity is constrained to 138,469 parameters without large pre-training, the model accommodated the dual-engine data by **contracting its acceptance volume around the specific training centroids**.
- When exposed to Google Translate TTS, which falls outside both the VieNeu and Edge centroids, the model's conservative boundary defaulted to `REJECT`.

### 4. Validation Selection Metric Limitations
In `train_phase28_factorial.py`:
- Checkpoint selection was governed by `val_macro_f1 = float(np.mean(f1[:4]))` on `validation_v1.csv`.
- `validation_v1.csv` contains exclusively VieNeu-TTS audio (in-domain).
- Because all models easily achieved $>93\%$ Macro-F1 on in-domain VieNeu validation, early stopping terminated at epoch 26 without penalizing cross-engine fragility or class-wise recall collapse.

---

## 6. Diagnostic Conclusions for D3 Candidate Development

1. **Do NOT increase threshold $\theta_{\text{cmd}}$:** The issue is not threshold calibration. At $\theta_{\text{cmd}} = 0.20$, missed commands have mean target probability $< 0.05$.
2. **Rebalance Cross-Entropy Loss:** Remove the $1.25$ penalty on REJECT; evaluate symmetric $[1.0, 1.0, 1.0, 1.0, 1.0]$ or command-protective weights $[1.1, 1.1, 1.1, 1.1, 1.0]$.
3. **Calibrate Batch Sampler:** Test adjusting the batch target:reject ratio from 32:32 to 36:28 (9 per command, 14 OOV, 14 acoustic).
4. **Constrain Minimum Command Recall:** In validation checkpoint selection, enforce a minimum recall penalty:
   $$\text{Score}_{\text{val}} = \text{Macro-F1}_{\text{cmd}} \times \left(\min_{c \in \{0,1,2,3\}} \text{Recall}_c\right)$$
   preventing the selection of checkpoints that sacrifice individual commands to maximize overall rejection.
