# PHASE 54: D3 BALANCED-SENSITIVITY CANDIDATE TRAINING PRE-REGISTRATION

**Repository:** `D:\trainidentificationaudio`  
**Phase:** Phase 54 (D3 Training Pre-Registration)  
**Creation Timestamp:** `2026-10-25T14:52:00+00:00`  
**Registration Status:** **LOCKED & PRE-REGISTERED (PRIOR TO INITIATING D3 TRAINING)**  
**Target Model Capacity:** 138,469 parameters (`CorrectedVietnameseKWSNet`, strictly preserved, `DEFER_E2_CAPACITY`)  
**Operating Threshold:** $\theta_{\text{cmd}} = 0.20$ (Strictly Locked)  
**Canonical Default Model:** Legacy $v11$ (`3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2`)

---

## Mandatory Governance Invariants

> [!CAUTION] MANDATORY RESEARCH STATUSES
> 1. `EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`
> 2. `NOT_VALIDATED_ON_NATURAL_SPEECH`
> 3. `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`
> 4. `NOT_VALIDATED_PENDING_DATA`
>
> Zero human speech was collected. Zero physical microphone testing was conducted. No historical checkpoints ($v11$, $E1$, Phase 28 $D_2\_H1$, Phase 26 $D_0\_H1\_seed44$, Phase 28 $D_0\_H1\_seed42$) will be overwritten or altered.
> Historical test corpora (Phase 21, Phase 27, Phase 28 `edge_tts_eval_v2`, Phase 43 MMS-TTS, Phase 51 gTTS) are strictly quarantined and EXCLUDED from training, validation, and candidate selection.

---

## 1. Experimental Question & Hypothesis

### Core Controlled Question:
Can keyphrase detection sensitivity on out-of-domain synthetic speech be improved by removing the artificial loss penalty on command classes and enforcing a minimum class-wise recall objective during training, **without triggering an unacceptable regression in background OOV rejection**?

### Specific Hypotheses:
1. **Hypothesis 1 (Loss Weight Neutralization):** In Phase 28 $D_2\_H1$, the cross-entropy loss weight vector was $\mathbf{w}_{\text{CE}} = [1.0, 1.0, 1.0, 1.0, 1.25]$, penalizing false alarms $25\%$ more heavily than misses and biasing the dense layer logits towards REJECT. Setting symmetric weights $[1.0, 1.0, 1.0, 1.0, 1.0]$ will remove this systematic negative logit offset.
2. **Hypothesis 2 (Command-Protective Sampling & Weights):** Setting $\mathbf{w}_{\text{CE}} = [1.10, 1.10, 1.10, 1.10, 1.00]$ combined with a $36:28$ target:reject batch prior (9 per command, 14 OOV, 14 acoustic) will provide the network with sufficient gradient support to resist out-of-domain command suppression.
3. **Hypothesis 3 (Constrained Checkpoint Selection):** Checkpoint selection based purely on aggregate Macro-F1 allows models to achieve high scores while sacrificing individual fragile commands (e.g., `TRAI` or `PHAI`). Optimizing a composite objective that penalizes the minimum per-class recall:
   $$\text{Score}_{\text{val}} = \text{Macro-F1}_{\text{cmd}} \times \left(\min_{c \in \{0,1,2,3\}} \text{Recall}_c\right)^{0.25}$$
   ensures that no individual command is sacrificed during checkpoint freeze.

---

## 2. Permitted Data Manifests & Strict Quarantines

### 2.1 Permitted Training Manifest:
- **Path:** `experiments/vieneu_kws_research_v12/manifests/D2_matched_train_v1.csv`
- **Total Samples:** $5,440$
- **SHA-256 Checksum:** `e7529810a08e1e72e1e07b469b61d3345dbe7264857b29a008c2a472c67dc91b`
- **Composition:** $90\%$ VieNeu-TTS ($4,896$ clips) + $10\%$ Edge-TTS matched non-test lexemes ($544$ clips across `vi-VN-HoaiMyNeural` and `vi-VN-NamMinhNeural`).
- **Target / REJECT Distribution:** Exactly $680$ per command class ($2,720$ total commands) and $2,720$ REJECT samples ($1,360$ OOV words + $1,360$ acoustic noise/silence).

### 2.2 Permitted Validation Manifest (Development Selection Only):
- **Path:** `experiments/vieneu_kws_research_v12/manifests/validation_v1.csv`
- **Total Samples:** $1,280$
- **SHA-256 Checksum:** `061d4bdf882eb7bc4803964ff0bb50ebbe6c5c7ce1a5cc6a00cb74431e5bb0ee`
- **Composition:** $640$ REJECT ($50\%$) and $160$ per command class ($12.5\%$ each).

### 2.3 Quarantined & Strictly Excluded Manifests:
- `PHASE50_LOCKED_MANIFEST.csv` (Phase 51 Track A, gTTS): **EXCLUDED**
- `PHASE50_STREAMING_MANIFEST.csv` (Phase 51 Track B, gTTS): **EXCLUDED**
- `PHASE43_LOCKED_MANIFEST.csv` (Phase 43 Track A, MMS-TTS): **EXCLUDED**
- `PHASE43_STREAMING_MANIFEST.csv` (Phase 43 Track B, MMS-TTS): **EXCLUDED**
- `edge_tts_eval_v2` / `edge_tts_streaming_v2`: **EXCLUDED**
- Any human speech recording or field recording: **EXCLUDED**

---

## 3. Pre-Registered Controlled Comparison Matrix

We pre-register three controlled experimental configurations across seed 42 (the primary comparative seed against $D_2\_H1\_seed42$, $E1\_seed42$, and $v11\_seed42$):

| Condition ID | Architecture Scale | Cross-Entropy Weights ($\mathbf{w}_{\text{CE}}$) | Batch Sampler Prior (Target : OOV : Acoustic) | Validation Checkpoint Objective | Maximum Epochs & Patience |
|:-------------|:------------------:|:-----------------------------------------------:|:---------------------------------------------:|:-------------------------------:|:-------------------------:|
| **D3_A** | 138,469 params | $[1.0, 1.0, 1.0, 1.0, 1.0]$ (Symmetric) | 32 : 16 : 16 (Standard H1) | Standard Macro-F1 ($F_{1,\text{cmd}}$) | 35 epochs (Patience: 8) |
| **D3_B** | 138,469 params | $[1.10, 1.10, 1.10, 1.10, 1.0]$ (Cmd-Boost) | 36 : 14 : 14 (Target-Boosted) | Standard Macro-F1 ($F_{1,\text{cmd}}$) | 35 epochs (Patience: 8) |
| **D3_C** | 138,469 params | $[1.05, 1.05, 1.05, 1.05, 1.0]$ (Moderate) | 34 : 15 : 15 (Balanced) | $F_{1,\text{cmd}} \times (\min_c R_c)^{0.25}$ | 35 epochs (Patience: 8) |

### Invariant Fixed Hyperparameters Across All Runs:
- **Architecture:** `CorrectedVietnameseKWSNet` (conv1: 32, conv2: 64, conv3: 64, conv4: 128, fc1: 64, fc2: 5). Trainable parameters: exactly **138,469**.
- **Frontend:** 40-channel Log-Mel spectrogram, $16\text{ kHz}$, $25\text{ ms}$ window, $10\text{ ms}$ hop, $512$ FFT.
- **Optimizer:** AdamW, learning rate $\eta = 1.0 \times 10^{-3}$, weight decay $\lambda = 0.01$.
- **Scheduler:** CosineAnnealingLR, $T_{\max} = 35$, $\eta_{\min} = 1.0 \times 10^{-5}$.
- **Data Augmentation:** Time shift $\pm 800$ samples ($\pm 50\text{ ms}$), random gain $\pm 3\text{ dB}$, SpecAugment (freq mask max: 4, time mask max: 10).
- **Runtime Inference Rule:** Decision threshold strictly locked at $\theta_{\text{cmd}} = 0.20$.

---

## 4. Pre-Registered Validation-Only Acceptance Gate

Candidate selection will occur strictly on `validation_v1.csv`. A D3 training run is eligible for candidate selection only if it satisfies all of the following criteria:

```
Pre-Registered Validation Acceptance Gate (on validation_v1.csv at theta_cmd = 0.20):
  1. Validation 4-Command Macro-F1 >= 92.0%
  2. Per-Class Recall Floor: Recall(c) >= 90.0% for EVERY command c in {TRAI, PHAI, LEN, XUONG}
  3. Validation OOV False Alarm Rate <= 10.0% (REJECT Recall >= 90.0%)
  4. Validation Rejection Precision >= 95.0%
  5. Training convergence without NaN, inf, or loss divergence
```

If multiple runs pass this gate, the candidate with the highest balanced objective score:
$$\text{Score}_{\text{sel}} = \text{Macro-F1}_{\text{cmd}} \times \left(\min_{c} \text{Recall}_c\right)$$
will be frozen as the `D3_Development_Candidate`.

If no run satisfies all criteria, the outcome will be recorded as `BLOCKED_DEVELOPMENT_DATA_INTEGRITY`.
