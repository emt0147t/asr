# Model Card: Vietnamese Bounded KWS CNN (Phase 13.1 Remediated Bundle)

## Model Details
- **Model Name:** `CorrectedVietnameseKWSNet`
- **Candidate Identifier:** `variant_v11_hard_negative_weighted_seed42`
- **Architecture:** 4-stage Conv2D + BatchNorm + MaxPool + Dropout(0.3) + FC layers
- **Total Parameters:** 138,469 trainable parameters
- **Input Feature Representation:** Log-Mel Spectrogram (40 filterbanks, 101 timeframes, 16 kHz mono)
- **Target Keywords:** `TRAI`, `PHAI`, `LEN`, `XUONG`
- **Rejection Class:** `REJECT`
- **Operating Threshold ($\theta^*$):** `0.20`

## Preprocessing Contract Remediation
In Phase 13.1, the inference preprocessing path was remediated to restore strict bitwise and mathematical alignment with the training implementation in `train_v11.py` and the Phase 91 deployment contract (`DEPLOYMENT_CONTRACT.md`):
- **Clamping Policy:** Peak amplitude normalization is applied **only** when $\max(|x|) > 1.0$ (preventing digital clipping).
- **Floor Amplification Prohibition:** Quiet signals, room tone, and digital silence are **never** scaled or amplified upward. This corrects the defect in the historical Phase 12 bundle that unconditionally normalized all audio to $-1.0\text{ dBFS}$, which artificially induced false positive triggers on faint ambient noise.

## Performance Characteristics (on 640-clip synthetic locked test)
- Command Recall: 100.0% (160 / 160)
- Non-Target False Accept Rate: 0.63% (3 / 480)
- Accuracy: 99.53% (637 / 640)
- Single-Clip CPU Latency: $2.04 \pm 0.25$ ms

## Scope and Operational Boundaries
1. `FROZEN_CNN_CANDIDATE`: Candidate weights and threshold are permanently frozen.
2. `NOT_VALIDATED_ON_NATURAL_SPEECH`: Trained and evaluated strictly on synthetic speech. Performance on natural human speech remains unproven.
3. `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`: Operates only on bounded 1.0-second pre-cut utterances. Continuous open-mic operation requires streaming VAD and ring-buffer trigger logic.
