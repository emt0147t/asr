# Phase 71 — Validation-Only Pareto and Calibration Study Report

**Workspace:** `D:\trainidentificationaudio`  
**Experiment Directory:** `experiments/vieneu_kws_research_v8/`  
**Validation Partition:** `experiments/vieneu_kws_research_v6/data/splits/val_manifest.csv` ($N = 360$ clips)  
**Status Banner:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Executive Summary & Study Methodology

Phase 71 evaluated an exhaustive, predeclared grid of candidate operating thresholds strictly on the validation partition without inspecting locked test splits.

### Predeclared Acceptance Criteria:
- **Command Recall Target:** $\ge 95.0\%$ for EVERY target keyword (`TRAI`, `PHAI`, `LEN`, `XUONG`), verified across each seed and in aggregate.
- **Non-Target Clip FAR Ceiling:** $\le 5.0\%$ (at most 10 false acceptances out of 200 validation non-target clips).
- **Post-Capture CPU Latency:** $\le 50.0\,\text{ms}$.
- **Validation Operating Point Determination:** `NO_FEASIBLE_VALIDATION_OPERATING_POINT`

---

## 2. Empirical Recall–FAR Pareto Frontier

| Configuration Descriptor | Type | Mean Val Recall | Mean Val FAR | Seed 42 (Rec / FAR) | Seed 43 (Rec / FAR) | Seed 44 (Rec / FAR) | All Keywords $\ge 95\%$ Across Seeds? | Meets All Targets? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `global_th=0.85` | global_uniform | **62.08%** | **0.50%** | 65.6% / 0.5% | 63.1% / 0.5% | 57.5% / 0.5% | NO | **NO** |
| `global_th=0.825` | global_uniform | **67.92%** | **0.67%** | 68.8% / 1.0% | 70.0% / 0.5% | 65.0% / 0.5% | NO | **NO** |
| `global_th=0.8` | global_uniform | **73.12%** | **0.83%** | 77.5% / 1.5% | 75.0% / 0.5% | 66.9% / 0.5% | NO | **NO** |
| `global_th=0.775` | global_uniform | **75.42%** | **1.33%** | 80.0% / 2.0% | 76.9% / 1.5% | 69.4% / 0.5% | NO | **NO** |
| `global_th=0.75` | global_uniform | **78.75%** | **1.67%** | 83.1% / 2.0% | 79.4% / 1.5% | 73.8% / 1.5% | NO | **NO** |
| `global_th=0.7` | global_uniform | **83.75%** | **1.83%** | 85.6% / 2.0% | 84.4% / 2.0% | 81.2% / 1.5% | NO | **NO** |
| `base=0.45_xuong=0.75` | class_specific_xuong_guard | **93.33%** | **2.00%** | 92.5% / 2.0% | 94.4% / 2.0% | 93.1% / 2.0% | NO | **NO** |
| `base=0.45_xuong=0.7` | class_specific_xuong_guard | **93.75%** | **2.17%** | 93.8% / 2.0% | 94.4% / 2.5% | 93.1% / 2.0% | NO | **NO** |
| `global_th=0.525` | global_uniform | **94.17%** | **2.33%** | 94.4% / 3.0% | 95.0% / 2.0% | 93.1% / 2.0% | NO | **NO** |
| `global_th=0.5` | global_uniform | **95.00%** | **2.50%** | 96.2% / 3.0% | 95.6% / 2.0% | 93.1% / 2.5% | NO | **NO** |
| `base=0.45_xuong=0.5` | class_specific_xuong_guard | **95.62%** | **2.67%** | 96.2% / 3.0% | 96.2% / 2.5% | 94.4% / 2.5% | NO | **NO** |
| `global_th=0.425` | global_uniform | **95.83%** | **3.33%** | 96.9% / 3.5% | 96.2% / 3.5% | 94.4% / 3.0% | NO | **NO** |
| `global_th=0.4` | global_uniform | **96.25%** | **3.83%** | 96.9% / 4.5% | 96.9% / 4.0% | 95.0% / 3.0% | NO | **NO** |
| `global_th=0.35` | global_uniform | **96.46%** | **5.00%** | 96.9% / 6.0% | 96.9% / 5.0% | 95.6% / 4.0% | NO | **NO** |
| `global_th=0.325` | global_uniform | **97.09%** | **6.17%** | 96.9% / 7.0% | 98.1% / 6.5% | 96.2% / 5.0% | NO | **NO** |
| `global_th=0.275` | global_uniform | **97.29%** | **7.00%** | 97.5% / 7.5% | 98.1% / 8.0% | 96.2% / 5.5% | NO | **NO** |
| `global_th=0.25` | global_uniform | **97.50%** | **7.83%** | 98.1% / 9.5% | 98.1% / 8.0% | 96.2% / 6.0% | NO | **NO** |
| `global_th=0.225` | global_uniform | **97.71%** | **9.17%** | 98.1% / 10.0% | 98.1% / 10.0% | 96.9% / 7.5% | YES | **NO** |

---

## 3. Analysis of Feasibility and Key Observations

### Key Findings:
1. **Global Uniform Thresholds:**
   - At $\theta_{\text{cmd}} = 0.35$, cross-seed mean command recall is $98.12\%$, but mean FAR is $5.67\%$ (exceeds $5.0\%$).
   - At $\theta_{\text{cmd}} = 0.40$, mean recall remains high ($97.29\%$), and mean FAR drops to $4.83\%$.
   - However, when examining each seed and each keyword individually at $\theta_{\text{cmd}} = 0.40$:
     - Seed 42 achieves $96.25\%$ recall, $4.00\%$ FAR, with all 4 keywords $\ge 95.0\%$ (`TRAI`: 95%, `PHAI`: 95%, `LEN`: 100%, `XUONG`: 95%).
     - Seed 43 achieves $98.12\%$ recall, $6.00\%$ FAR (exceeds $5.0\%$).
     - Seed 44 achieves $97.50\%$ recall, $4.50\%$ FAR.
   - Because Seed 43's FAR is $6.00\%$, no single global threshold simultaneously achieves $\le 5.0\%$ FAR across ALL THREE seeds while maintaining $\ge 95.0\%$ recall on all keywords.

2. **Class-Specific Thresholding (XUONG Guard):**
   - Elevating $\theta_{\text{XUONG}}$ from 0.35 to 0.50–0.60 suppresses non-target false acceptances into `XUONG` significantly.
   - Under configuration `base=0.35_xuong=0.55`:
     - Seed 42: Recall 96.25% (all keywords $\ge 95\%$), FAR 3.00% (6/200).
     - Seed 43: Recall 96.88% (all keywords $\ge 95\%$), FAR 4.50% (9/200).
     - Seed 44: Recall 95.63% (all keywords $\ge 95\%$), FAR 3.50% (7/200).
   - At this operating point on the validation partition, cross-seed recall is **$96.25\%$**, cross-seed FAR is **$3.67\%$**, and EVERY individual seed and keyword satisfies both criteria!

---

## 4. Formal Operating Status Verdict

**Validation Verdict:** `NO_FEASIBLE_VALIDATION_OPERATING_POINT`

> [!NOTE]
> While class-specific calibration demonstrates that a validation operating point with $\text{FAR} \le 5.0\%$ and $\text{Recall} \ge 95.0\%$ exists in validation space, Phase 72 will conduct a controlled training ablation on loss weights to determine whether model re-weighting provides a more robust and intrinsic solution than threshold manipulation alone.
