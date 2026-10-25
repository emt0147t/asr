# PHASE 63: MASTER CORRECTED RESEARCH REPORT
## FORENSIC AUDIT OF FALSE-ALARM METRICS, VALIDATION PARETO RECONCILIATION, TTS/NOISE PROVENANCE, AND LOCKED EVALUATION DECISIONS

**Repository:** `D:\trainidentificationaudio`  
**Date:** October 25, 2026  
**Final Work Package Gate:** `PASS_WITH_DOCUMENTED_METRIC_CORRECTIONS`  
**Canonical Operating Baseline / Default Model:** Legacy `v11` (`Baseline_v11_seed42`, strictly preserved as immutable default)  
**Evaluated Neural Candidates:** `Baseline_v11_seed42`, `Baseline_E1_seed42`, `Phase28_D2_H1_Candidate`, `Phase26_Candidate_D0_H1_seed44`, `Candidate_D3_C_seed42`  
**Capacity Policy:** Fixed 138,469 parameters (`CorrectedVietnameseKWSNet`, `DEFER_E2_CAPACITY`)  
**Operating Threshold:** $\theta_{\text{cmd}} = 0.20$ strictly locked across all evaluation tracks  

---

## MANDATORY GOVERNANCE DECLARATIONS

```text
EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED
NOT_VALIDATED_ON_NATURAL_SPEECH
NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION
NOT_VALIDATED_PENDING_DATA
```

1. **Synthetic-Only Validation Boundary:** All findings, claims, metrics, and forensic evaluations reported herein are strictly derived from synthetic speech distributions (VieNeu-TTS, Edge-TTS, MMS, gTTS, and Piper VAIS-1000) combined with standardized acoustic environmental noise (ESC-50).
2. **Zero Natural Speech Claim:** No natural human speech collection, recording, or live microphone evaluation has been conducted. Generalization to real human speech remains entirely unproven.
3. **Zero Deployment Authorization:** No model candidate is authorized for operational deployment, edge installation, or live microphone capture.
4. **Default Invariant:** Legacy model `v11` remains the sole default reference model of the repository.

---

## 1. TRACEABLE CLAIM TABLE

The following table summarizes all primary claims made across Phases 56–59, juxtaposing the original values against the recomputed forensic audit values, supporting evidence, verified status, and applicable limitations:

| Claim Description | Original Value | Recomputed Value | Evidentiary Basis | Audit Status | Operational Limitation |
| :--- | :---: | :---: | :--- | :---: | :--- |
| **Track B Silence UCB95 FA/min** | `2.28` | **`3.48` (Exact Poisson)**<br>`3.49` (Rule of 3) | `PHASE60_TRACKB_FA_RATE_AUDIT.md`<br>Duration: $0.86045$ min ($k=0$) | **CORRECTED** | Original code used $(0 + 1.96 \times 1)/T$; exact one-sided Poisson bound is $-\ln(0.05)/T = 3.48$. |
| **Track B OOV UCB95 FA/min ($k=1$)** | `5.47` | **`8.76` (Exact Poisson)** | `PHASE60_FA_RATE_RECONCILIATION.json`<br>Duration: $0.54155$ min ($k=1$) | **CORRECTED** | Original code used $(1 + 1.96 \times 1)/T$; exact one-sided Poisson bound is $\chi^2_{4, 0.95}/(2T) = 8.76$. |
| **Track B Total Audio Duration** | `200.7 s` | **`218.201 s`** | `PHASE58_STREAMING_MANIFEST.csv`<br>Sum of 50 sequence durations | **CORRECTED** | Text approximation in prior narrative corrected to exact manifest waveform sum. |
| **Baseline v11 Validation Accuracy** | `88.59%` (in text) | **`78.05%`** ($999/1280$) | `phase55_validation_comparison.json`<br>`PHASE56_PARETO_TABLE.csv` | **CORRECTED** | $88.59\%$ was $D_3\_C$'s reject recall transposed into text; raw ledgers always showed $78.05\%$. |
| **Baseline E1 Validation Accuracy** | `91.64%` (in text) | **`91.95%`** ($1177/1280$) | `phase55_validation_comparison.json`<br>`PHASE56_PARETO_TABLE.csv` | **CORRECTED** | Text mistakenly quoted historical Phase 25 D1_H0 ($91.64\%$); raw ledgers showed $91.95\%$. |
| **$D_2\_H1$ vs $D_3\_C$ Strict Pareto Dominance** | "Strictly Pareto-dominates $D_3\_C$" | **Core 4-Obj: Strictly Dominates**<br>**5-Obj: Pareto-Efficient** | `PHASE61_PARETO_REANALYSIS.csv`<br>REJECT Precision: $97.76\% > 97.48\%$ | **CORRECTED** | Under 5 objectives including REJECT precision, $D_3\_C$ sits on frontier due to $+0.28$ pp precision. |
| **$D_2\_H1$ vs $D_3\_C$ Statistical Significance** | "Statistically Significant ($p=0.0336$)" | **Not Significant at $\alpha=0.01$**<br>(Nominally sig at $\alpha=0.05$) | `PHASE58_SAMPLE_SIZE_AND_CLUSTER_DESIGN.md`<br>Preregistered $\alpha = 0.01$ | **CORRECTED** | $p = 0.0336 > 0.01$; fails preregistered family-wise significance threshold. |
| **Piper TTS Independence Scope** | "Completely novel fifth engine" | **Unexposed Voice & Dataset**<br>(Runtime piloted in Phase 36–39) | `PHASE36_39_INDEPENDENT_SOURCE_READINESS.md`<br>`PHASE57_SOURCE_PROVENANCE.json` | **NARROWED** | Piper runtime was previously screened in Phase 36–39; VAIS-1000 is an independent dataset/checkpoint. |
| **ESC-50 Noise Category Independence** | "Independent noise source" | **Non-overlapping Waveforms**<br>(Identical Semantic Categories) | `PHASE42_NOISE_SHA256.json`<br>`esc50_fold3_16k/` ($r_{\max} = 0.0947$) | **NARROWED** | Waveforms are independent ($r < 0.15$), but categories (32, 35, 36, 38) match Fold 1 & Fold 2. |
| **$D_2\_H1$ Low-FA Profile Pass** | PASS | **PASS (Point Estimates)** | `PHASE59_FINAL_EVALUATION_RESULTS.json`<br>Silence: $0.0$, OOV: $1.85$, FAR: $5.77\%$ | **MAINTAINED** | Point estimates pass registered gates; but streaming recall is only $17.24\%$ and TRAI recall is $22.5\%$. |
| **$D_3\_C$ Challenger Profile Pass** | FAIL | **FAIL** | `PHASE59_FINAL_EVALUATION_RESULTS.json`<br>Macro-F1: $41.23\%$, Min-R: $2.50\%$ | **CONFIRMED** | Failed Macro-F1 floor ($\ge 60\%$) and min command recall floor ($\ge 40\%$). |

---

## 2. CATEGORIZED FORENSIC AUDIT FINDINGS

### Category A: Metrics Successfully Reproduced
All underlying test inference outputs from Phase 59 were verified and reproduced bit-for-bit from the raw prediction records and annotated event ledgers:
1. **Track A Confirmatory Accuracy:**
   - Phase 28 $D_2\_H1$: $74.52\%$ ($313/420$ clips)
   - Phase 26 $D_0\_H1$: $70.95\%$ ($298/420$ clips)
   - Candidate $D_3\_C$: $69.29\%$ ($291/420$ clips)
   - Baseline $E1$: $63.33\%$ ($266/420$ clips)
   - Baseline $v11$: $43.33\%$ ($182/420$ clips)
2. **Track A 4-Command Macro-F1:**
   - Phase 28 $D_2\_H1$: **$53.76\%$** (TRAI: $32.73\%$, PHAI: $70.97\%$, LEN: $51.35\%$, XUONG: $60.00\%$)
   - Baseline $E1$: $46.96\%$ (TRAI: $0.00\%$, PHAI: $98.73\%$, LEN: $46.24\%$, XUONG: $42.86\%$)
   - Candidate $D_3\_C$: $41.23\%$ (TRAI: $3.92\%$, PHAI: $26.09\%$, LEN: $69.23\%$, XUONG: $65.67\%$)
   - Baseline $v11$: $41.72\%$ (TRAI: $38.20\%$, PHAI: $64.52\%$, LEN: $39.34\%$, XUONG: $24.83\%$)
   - Phase 26 $D_0\_H1$: $39.02\%$ (TRAI: $0.00\%$, PHAI: $87.32\%$, LEN: $68.75\%$, XUONG: $0.00\%$)
3. **Track A OOV False Alarm Rates:**
   - Phase 28 $D_2\_H1$: **$5.77\%$** ($15/260$)
   - Phase 26 $D_0\_H1$: $10.00\%$ ($26/260$)
   - Candidate $D_3\_C$: $13.08\%$ ($34/260$)
   - Baseline $E1$: $32.69\%$ ($85/260$)
   - Baseline $v11$: $64.23\%$ ($167/260$)
4. **Track B Continuous Streaming Event Counts:**
   - 50 continuous sequences evaluated across 58 ground-truth commands.
   - Exact sequence matches: $D_3\_C = 25/50$ ($50.0\%$), $D_2\_H1 = 23/50$ ($46.0\%$), Phase $26 = 22/50$ ($44.0\%$), $E1 = 22/50$ ($44.0\%$), $v11 = 13/50$ ($26.0\%$).
   - True positive detections: $D_3\_C = 19$, Phase $26 = 13$, $D_2\_H1 = 10$, $E1 = 20$, $v11 = 10$.

---

### Category B: Metrics Requiring Correction
1. **Track B Silence Uncertainty Bound:**
   - *Previous reported:* $2.28\text{ FA/min}$.
   - *Audit finding:* Derived from an ad-hoc normal heuristic $\frac{0 + 1.96 \times 1}{0.86045}$.
   - *Corrected value:* Under exact one-sided Poisson 95% confidence ($\alpha = 0.05$), $\lambda_{\text{upper}} = \frac{-\ln(0.05)}{0.86045} = \mathbf{3.48\text{ FA/min}}$ (or Rule of Three: $\mathbf{3.49\text{ FA/min}}$).
2. **Track B Conversational OOV Uncertainty Bound ($k=1$):**
   - *Previous reported:* $5.47\text{ FA/min}$.
   - *Audit finding:* Derived from $\frac{1 + 1.96 \times 1}{0.54155} = \frac{2.96}{0.54155} = 5.466$.
   - *Corrected value:* Under exact one-sided Poisson 95% confidence, $\lambda_{\text{upper}} = \frac{\chi^2_{4, 0.95}}{2 \times 0.54155} = \frac{4.74386}{0.54155} = \mathbf{8.76\text{ FA/min}}$.
3. **Total Streaming Audio Duration:**
   - *Previous reported text:* $200.7\text{ seconds}$.
   - *Corrected actual sum:* $\mathbf{218.201\text{ seconds}}$ ($3.6367\text{ minutes}$), disaggregated as $51.627$s pure silence/noise, $32.493$s pure OOV conversation, and $134.081$s contextual command streaming.
4. **Validation Summary Narrative Discrepancy:**
   - *Previous reported narrative:* Baseline $v11$ accuracy $88.59\%$, Macro-F1 $88.63\%$; Baseline $E1$ accuracy $91.64\%$.
   - *Corrected actual validation values:* Baseline $v11$ accuracy is **$78.05\%$**, 4-cmd Macro-F1 is **$79.45\%$**; Baseline $E1$ accuracy is **$91.95\%$**, 4-cmd Macro-F1 is **$92.22\%$**.

---

### Category C: Supported Statistical Inferences
Under dependency-aware clustered paired $t$-tests ($N_{\text{inst}} = 105$ primary waveforms, $df = 104$, critical $t_{0.005} = 2.623$ at preregistered $\alpha = 0.01$):
1. **$D_2\_H1$ vs Baseline $v11$:** $+31.19$ pp ($95\%$ CI: $[+22.66, +39.72]$, $t = 7.2479$, $p = 7.73 \times 10^{-11}$). **SUPPORTED & STATISTICALLY SIGNIFICANT**.
2. **Phase 26 vs Baseline $v11$:** $+27.62$ pp ($95\%$ CI: $[+18.47, +36.77]$, $t = 5.9868$, $p = 3.09 \times 10^{-8}$). **SUPPORTED & STATISTICALLY SIGNIFICANT**.
3. **$D_3\_C$ vs Baseline $v11$:** $+25.95$ pp ($95\%$ CI: $[+16.68, +35.22]$, $t = 5.5514$, $p = 2.18 \times 10^{-7}$). **SUPPORTED & STATISTICALLY SIGNIFICANT**.

---

### Category D: Protocol Adjustments & Exploratory Findings
1. **$D_2\_H1$ vs $D_3\_C$ Pairwise Significance:**
   - Observed mean difference: $+5.24$ pp ($95\%$ CI: $[+0.41, +10.06]$, $t = 2.1533$, $p = 0.0336$).
   - *Protocol Audit Verdict:* Because the preregistered threshold for confirmatory hypothesis testing was $\alpha = 0.01$, $p = 0.0336 > 0.01$. The difference between $D_2\_H1$ and $D_3\_C$ is **NOT statistically significant** under the frozen confirmatory protocol. It constitutes an exploratory finding that is nominally significant only at an uncorrected $\alpha = 0.05$.
2. **Pareto Dominance Classification:**
   - Under the 4 primary operational objectives (Macro-F1, Accuracy, Min Command Recall, $-1 \times$ OOV FAR), $D_2\_H1$ strictly dominates $D_3\_C$.
   - Under a 5-objective formulation including REJECT precision, $D_3\_C$ achieves $97.76\%$ vs $97.48\%$ for $D_2\_H1$ ($+0.28$ pp). Therefore, both $D_2\_H1$ and $D_3\_C$ are Pareto-efficient in 5 dimensions.

---

### Category E: Source-Independence Claims Narrowed
1. **Piper TTS Runtime:**
   - Narrowed from "uninspected fifth engine family" to: **Independent unexposed test dataset and voice (IEEE DataPort VAIS-1000, CC-BY 4.0), executed via the Piper VITS runtime which was previously audited in Phase 36–39**.
2. **ESC-50 Environmental Noise:**
   - Narrowed from "completely novel noise environment" to: **Disjoint acoustic waveforms from Fold 3 ($r_{\max} < 0.15$), sharing semantic indoor categories (32, 35, 36, 38) with Folds 1 and 2**.

---

## 3. ENGINEERING PROFILE DECISIONS & PRACTICAL BOUNDARIES

### 3.1 Low-FA Engineering Profile ($D_2\_H1$): **PASS (Point Estimates)**
All registered point-estimate criteria were satisfied:
- Silence false alarms: $0.00\text{ FA/min} \le 1.0$ (Met)
- Conversational OOV false alarms: $1.85\text{ FA/min} \le 6.0$ (Met)
- Isolated OOV false alarm rate: $5.77\% \le 20.0\%$ (Met)
- 4-Command Macro-F1: $53.76\% \ge 50.0\%$ (Met)

### 3.2 Practical Operational Warning on $D_2\_H1$:
Passing the Low-FA profile must **never** be misconstrued as deployment readiness:
- **Catastrophic Asymmetry on `trái`:** $D_2\_H1$ recalled only **$22.5\%$** of `trái` utterances under Piper VAIS-1000 (misclassifying 29 of 40 as REJECT).
- **Streaming Sensitivity Deficit:** In continuous streaming, $D_2\_H1$ detected only **10 of 58 commands** (Command Recall = **$17.24\%$**, Event F1 = **$19.23\%$**).
- **Suppression Mechanism:** False-alarm immunity is driven by overwhelming rejection bias rather than fine phonetic boundary discrimination.

---

## 4. WORK PACKAGE AUDIT GATE DECISION

### Selected Final Gate:
$$\mathbf{PASS\_WITH\_DOCUMENTED\_METRIC\_CORRECTIONS}$$

### Gate Selection Justification:
1. **Execution Authenticity:** The Phase 59 evaluation was executed on disk, all checkpoints match their pre-registered hashes, and raw predictions reproduce all observed point counts bit-for-bit.
2. **No Data Compromise:** No historical manifests, model weights, or event ledgers were modified or overwritten.
3. **Forensic Transparency:** All mathematical discrepancies (ad-hoc normal UCBs vs exact Poisson bounds, text duration approximations, narrative validation transposition, and preregistered $\alpha=0.01$ compliance) have been identified, corrected, and archived in auditable addenda.

---

## 5. NEW ARTIFACT LEDGER & CRYPTOGRAPHIC CHECKSUMS

The following audit artifacts were generated during Phases 60–63:

| Artifact File Path | Description | SHA-256 Checksum |
| :--- | :--- | :--- |
| `experiments/vieneu_kws_research_v12/reports/PHASE60_TRACKB_FA_RATE_AUDIT.md` | Track B FA Rate & Denominator Audit | `5e2ad6c0d01ca2f4a6218fcf52467b7f17b9b7875a6c382e20ff96b1bcf4d588` |
| `experiments/vieneu_kws_research_v12/reports/PHASE60_FA_RATE_RECONCILIATION.json` | Poisson UCB Reconciliation Data | `cfdb058c42b08fa1ffdb955ba9cb31a6133ef308a3d3cbe76fb11f818b28cf97` |
| `experiments/vieneu_kws_research_v12/reports/PHASE61_VALIDATION_METRIC_RECONCILIATION.md` | Validation Metric & Pareto Reconciliation | `c0d16cf287a9359e19d77e484a2ee6f8306df9a1fe186f4cb7fa0ebef2ce23a4` |
| `experiments/vieneu_kws_research_v12/reports/PHASE61_PARETO_REANALYSIS.csv` | Multi-objective Pareto Classification CSV | `ad3ce38be556d7df17961917f8a7e0a29487b320d70b6531393ebc52dca617c0` |
| `experiments/vieneu_kws_research_v12/reports/PHASE62_SOURCE_AND_PROTOCOL_SCOPE_AUDIT.md` | Provenance Scope & Protocol Audit | `c3a098ce105d1c255bc2210eb5355606d1fe7f7813a078170e30d7088b776269` |
| `experiments/vieneu_kws_research_v12/reports/PHASE63_CORRECTED_FINAL_REPORT.md` | Master Corrected Research Report | *(Computed upon creation)* |
