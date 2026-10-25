# REPRODUCIBILITY GUIDE: VIETNAMESE KEYWORD SPOTTING
## End-to-End Instructions for Independent Verification, Metric Reproduction, and Audit Execution

**Repository:** `D:\trainidentificationaudio`  
**Date:** October 26, 2026  
**Python Environment:** `.venv\Scripts\python.exe` (Python 3.11.9, PyTorch 2.6.0+cu124, CUDA operational)  
**Operating Decision Policy:** Calibrated command threshold override $\theta_{\text{cmd}} = 0.20$  

---

## 1. ENVIRONMENT SPECIFICATION & DEPENDENCIES

All models and frontend transforms require the pinned virtual environment in `.venv`:
- `torch == 2.6.0+cu124`
- `torchaudio == 2.6.0+cu124`
- `soundfile == 0.14.0`
- `scipy == 1.15.2`
- `scikit-learn == 1.6.1`
- `numpy == 2.2.3`
- `pandas == 2.2.3`
- `onnxruntime == 1.20.1`

Verify environment readiness:
```powershell
.venv\Scripts\python.exe -c "import torch, soundfile, scipy, sklearn; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
```

---

## 2. CHECKPOINT & ARTIFACT INTEGRITY VERIFICATION

Before executing inference, verify that all frozen reference checkpoints and models match their authoritative SHA-256 checksums:

```powershell
.venv\Scripts\python.exe -c "
import hashlib
files = {
    'v11_pt': 'experiments/vieneu_kws_research_v13_4/bundle/checkpoint/variant_v11_hard_negative_weighted_seed42.pt',
    'v11_onnx': 'experiments/vieneu_kws_research_v13_4/bundle/onnx/vietnamese_kws_v11.onnx',
    'e1_pt': 'experiments/vieneu_kws_research_v12/checkpoints/exp_v12_e1_baseline_scale_seed42.pt',
    'e1_onnx': 'experiments/vieneu_kws_research_v12/onnx/vietnamese_kws_v12_e1.onnx',
    'p28_d2_h1_pt': 'experiments/vieneu_kws_research_v12/candidates/phase28_d2_candidate/candidate_checkpoint.pt',
    'p26_d0_h1_pt': 'experiments/vieneu_kws_research_v12/candidates/phase26_candidate_d0_h1_seed44/candidate_checkpoint.pt',
    'd3_c_pt': 'experiments/vieneu_kws_research_v12/training_phase54_d3/D3_C_seed42/best_checkpoint.pt'
}
for name, p in files.items():
    h = hashlib.sha256(open(p, 'rb').read()).hexdigest()
    print(f'{name:<15}: {h}')
"
```

Expected reference hashes:
- `v11_pt`: `3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2`
- `v11_onnx`: `54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99`
- `e1_pt`: `f3c0fe8f7ccebb8fe354cb50acb247e0674232044706db0187feb92b1706b878`
- `e1_onnx`: `20ecc2964d88d5a9d1d15b1c665e9697a32d702b1e3dd12ab1e5f829aff016b8`
- `p28_d2_h1_pt`: `c6102b4876ddeaa453547c211c5cfc322511c957e18fd488a705a164b18601c8`
- `p26_d0_h1_pt`: `00b4fc71f863788648d5ea388a99c4cbd4441d9b9f63adc5e56154e329df63f3`
- `d3_c_pt`: `da5b414328c33644b07c0667c3f9926637baa5f1672119f86393182c53c38ba6`

---

## 3. REPRODUCING AUTHORITATIVE VALIDATION METRICS (PHASE 64)

To reproduce the exact validation metrics, confusion matrices, and aligned prediction rows for all 8 models on `validation_v1.csv` ($N = 1,280$):

```powershell
.venv\Scripts\python.exe experiments/vieneu_kws_research_v12/scripts/run_phase64_forensic_validation_audit.py
```

### Outputs Produced:
- [`PHASE64_VALIDATION_ALIGNED_PREDICTIONS.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v12/reports/PHASE64_VALIDATION_ALIGNED_PREDICTIONS.csv): Contains per-clip probabilities, predictions, and correctness flags across all 8 models.
- [`PHASE64_VALIDATION_MASTER_METRICS.json`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v12/reports/PHASE64_VALIDATION_MASTER_METRICS.json): Full confusion matrices, Macro-F1, per-class recall, and OOV FARs.
- [`PHASE64_VALIDATION_RECONCILED_METRICS.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v12/reports/PHASE64_VALIDATION_RECONCILED_METRICS.csv): Clean comparative summary table.

Expected verified numbers under $\theta_{\text{cmd}} = 0.20$:
- Baseline $v11$: Accuracy **$78.05\%$** ($999/1280$), 4-cmd Macro-F1 **$79.45\%$**, Min Recall **$78.75\%$**, OOV FAR **$34.22\%$** ($219/640$).
- Baseline $E1$: Accuracy **$91.95\%$** ($1177/1280$), 4-cmd Macro-F1 **$92.22\%$**, Min Recall **$95.62\%$**, OOV FAR **$13.91\%$** ($89/640$).
- Phase 28 $D_2\_H1$: Accuracy **$93.91\%$** ($1202/1280$), 4-cmd Macro-F1 **$93.88\%$**, Min Recall **$95.62\%$**, OOV FAR **$9.22\%$** ($59/640$).
- Candidate $D_3\_C$: Accuracy **$93.05\%$** ($1191/1280$), 4-cmd Macro-F1 **$93.22\%$**, Min Recall **$95.00\%$**, OOV FAR **$11.41\%$** ($73/640$).

---

## 4. REPRODUCING CONFIRMATORY EVALUATION (PHASE 59)

To run the complete confirmatory evaluation on the frozen Piper VAIS-1000 + ESC-50 Fold 3 audio corpus:

```powershell
.venv\Scripts\python.exe experiments/vieneu_kws_research_v12/scripts/eval_phase59_confirmatory_test.py
```

### Outputs Produced:
- [`PHASE59_FINAL_EVALUATION_RESULTS.json`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v12/reports/PHASE59_FINAL_EVALUATION_RESULTS.json)
- [`PHASE59_CONFIRMATORY_METRICS.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v12/reports/PHASE59_CONFIRMATORY_METRICS.csv)

### Expected Key Verified Metrics:
- **Track A Primary (420 clips):**
  - $D_2\_H1$: Accuracy **$74.52\%$** ($313/420$), Macro-F1 **$53.76\%$**, OOV FAR **$5.77\%$** ($15/260$).
  - $D_3\_C$: Accuracy **$69.29\%$** ($291/420$), Macro-F1 **$41.23\%$**, OOV FAR **$13.08\%$** ($34/260$), `trái` recall **$2.5\%$**.
- **Track B Streaming (50 sequences, 58 ground-truth events):**
  - $D_3\_C$: Exact sequence matches **$25/50$** ($50.0\%$), Event F1 **$31.40\%$**, Recall **$32.76\%$**.
  - $D_2\_H1$: Exact sequence matches **$23/50$** ($46.0\%$), Event F1 **$19.23\%$**, Recall **$17.24\%$**.
  - Exact Poisson 95% Confidence Limits: Silence ($k=0$, $0.86\text{ min}$) = **$3.48\text{ FA/min}$**; Conversational OOV ($k=1$, $0.54\text{ min}$) = **$8.76\text{ FA/min}$**.

---

## 5. STATISTICAL INFERENCE & POWER CALCULATION REPRODUCIBILITY

To reproduce the dependency-aware clustered paired $t$-test across $N_{\text{inst}} = 105$ primary waveforms:

```powershell
.venv\Scripts\python.exe -c "
import json, scipy.stats as stats, numpy as np
with open('experiments/vieneu_kws_research_v12/reports/PHASE59_FINAL_EVALUATION_RESULTS.json') as f:
    d = json.load(f)
pairs = d['paired_clustered_comparisons']
for k, v in pairs.items():
    print(f'{k:<45}: Delta={v[\"mean_delta_pp\"]:+.2f} pp | t={v[\"t_statistic\"]:+.4f} | p={v[\"p_value\"]:.4e}')
"
```

Significance standard: Preregistered $\alpha = 0.01$.
- $D_2\_H1$ vs $v11$: $p = 7.73 \times 10^{-11} < 0.01$ (**Significant**).
- $D_2\_H1$ vs $D_3\_C$: $p = 0.0336 > 0.01$ (**Not Significant at $\alpha=0.01$**; nominally significant only at uncorrected $\alpha=0.05$).
