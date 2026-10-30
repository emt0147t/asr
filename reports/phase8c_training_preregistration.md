# Phase 8C: Training Reproducibility & Preregistration Document

**Repository:** `D:\trainidentificationaudio`  
**Base Branch:** `feature/continuous-kws-segmentation`  
**Baseline Git Commit:** `91d91dbf6ec23486cb6f95e0a62a48d590360bad`  
**Status:** Preregistered & Locked  
**Date:** 2026-10-31

---

## Mandatory Governance Labels

> - **`EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`**
> - **`NOT_VALIDATED_ON_NATURAL_SPEECH`**
> - **`NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`**
> - **`NOT_VALIDATED_PENDING_DATA`**

---

## 1. Experimental Design: 2 × 2 Factorial Matrix

To isolate the independent and interactive contributions of **acoustic channel augmentation** and **neural network capacity**, Phase 8 specifies a controlled 2 × 2 experimental design:

| Experimental Arm | Architecture Class | Trainable Parameters | Acoustic Channel Augmentation | Seeds |
| :--- | :--- | :---: | :---: | :---: |
| **Arm A (Baseline Control)** | `CorrectedVietnameseKWSNet` | 138,469 | **OFF** (Clean) | `[42, 43, 44]` |
| **Arm B (Augmented Candidate)**| `CorrectedVietnameseKWSNet` | 138,469 | **ON** (Acoustic Pipeline) | `[42, 43, 44]` |
| **Arm C (Capacity Control)** | `ScaledVietnameseKWSNet_E2` | 304,277 | **OFF** (Clean) | `[42, 43, 44]` |
| **Arm D (Augmented Scaled)** | `ScaledVietnameseKWSNet_E2` | 304,277 | **ON** (Acoustic Pipeline) | `[42, 43, 44]` |
| **External Reference** | `Frozen v11 ONNX` | 138,469 | Historical Baseline | Historical |

---

## 2. Invariants & Non-Confounding Controls

All four experimental arms are locked to identical training conditions:
1. **Identical Dataset Splits:** All arms train on `train_manifest.csv` (5,440 samples) and evaluate on `val_manifest.csv` (1,280 samples) and quarantined `phase6_1a_holdout_manifest.csv` (55 continuous trials).
2. **Identical Audio Frontend:** Standard 16 kHz Log-Mel Spectrogram (40 Slaney bands, FFT=512, window=400, hop=160).
3. **Identical Optimization Budget:** AdamW optimizer ($\text{lr} = 10^{-3}$, weight decay = $0.01$), CosineAnnealingLR scheduler ($\eta_{\min} = 10^{-5}$), batch size 64, maximum 40 epochs with early stopping patience 10.
4. **Identical Model Selection Rule:** Checkpoint selection is strictly based on the highest validation macro-F1 on the clean validation set.
5. **Acoustic Augmentation Engine:** Arms B and D apply the Phase 7B validated pipeline (simulated RIR convolution, microphone frequency response EQ, low-frequency shelf, and ambient noise injection at SNR 10–24 dB).

---

## 3. Preregistered Hypotheses

1. **$H_1$ (Augmentation Effect on 138k):** Arm B will achieve higher macro-F1 and sequence exact match than Arm A on ambient loudspeaker loopback without regressing direct stream accuracy by more than 3.0%.
2. **$H_2$ (Capacity vs. Augmentation Interaction):** Arm B (138k + Aug) will outperform Arm C (304k + NoAug) on ambient acoustic loopback, confirming that channel augmentation is more critical than raw model capacity for physical acoustic mismatch.
3. **$H_3$ (Additive Scaling Effect):** Arm D (304k + Aug) will demonstrate whether capacity scaling provides additive benefits once matched acoustic training is provided.

---

## 4. Deliverables Index

- **Configuration File:** [`configs/phase8_training_matrix.yaml`](file:///d:/trainidentificationaudio/configs/phase8_training_matrix.yaml)
- **Parameter Audit JSON:** [`reports/phase8c_parameter_audit.json`](file:///d:/trainidentificationaudio/reports/phase8c_parameter_audit.json)
- **Preregistration Document:** [`reports/phase8c_training_preregistration.md`](file:///d:/trainidentificationaudio/reports/phase8c_training_preregistration.md)
"""

    for p in [report_md_v13, report_md_root, report_md_ptr]:
        with open(p, "w", encoding="utf-8") as f:
            f.write(md_content)

    print(f"[Report Generated] {report_md_v13}")


if __name__ == "__main__":
    main()
