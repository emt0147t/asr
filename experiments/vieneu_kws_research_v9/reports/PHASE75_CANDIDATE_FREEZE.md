# Phase 75 — Candidate Freeze Report

**Workspace:** `D:\trainidentificationaudio`  
**Candidate Identifier:** `variant_w10`  
**Research Campaign:** Phase 75–79 (`experiments/vieneu_kws_research_v9/`)  
**Status Banners:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Executive Summary & Freeze Objectives

The synthetic bounded-utterance candidate `variant_w10` was selected in Phase 73 and independently evaluated in Phase 74. Phase 75 formally freezes this model architecture, weights, feature extractor, threshold logic, and evaluation pipeline into an immutable, versioned research package under [`experiments/vieneu_kws_research_v9/artifacts/`](file:///D:/trainidentificationaudio/experiments/vieneu_kws_research_v9/artifacts/).

### Key Freeze Specifications
- **Model Architecture:** `CorrectedVietnameseKWSNet` (3 Conv2D blocks, BatchNorm, MaxPool2d, Dropout 0.25, FC layer).
- **Total Parameters:** **138,469** parameters (identical across all seeds).
- **Feature Extractor:** `LogMelFrontend` (40 mel bins, 25 ms FFT window, 10 ms hop size, $f_{\min}=20\text{ Hz}$, $f_{\max}=8000\text{ Hz}$, log offset $10^{-6}$).
- **Audio Preprocessing:** 16 kHz mono, peak-normalized to $-1.0\text{ dBFS}$, centered zero-padding / truncation to exactly 1.0 s (16,000 samples).
- **Class Mapping:**
  - Index 0: `TRAI`
  - Index 1: `PHAI`
  - Index 2: `LEN`
  - Index 3: `XUONG`
  - Index 4: `REJECT` (handles both `UNKNOWN` non-targets and `SILENCE`)
- **Threshold Decision Logic:**
  $$\text{Class} = \begin{cases} \arg\max_{c \in [0..3]} p_c, & \text{if } \max_{c \in [0..3]} p_c \ge \theta_{\text{cmd}} \\ \text{REJECT}, & \text{otherwise} \end{cases}$$

---

## 2. Checkpoint Integrity and Cryptographic Hashing

All three seed checkpoints were inspected from Phase 72, verified against model parameters, copied into the frozen artifact directory, and confirmed for exact SHA-256 integrity:

| Seed | Checkpoint File | Checkpoint SHA-256 Hash | Epoch | Best Val Loss | Parameter Count | Calibrated $\theta_{\text{cmd}}$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | `variant_w10_seed42.pt` | `7374af469a4bc4df5e124aae27720a8c11e688413f045eec3ee4ffd79c836bed` | 22 | 0.6991 | 138,469 | 0.40 |
| **43** | `variant_w10_seed43.pt` | `77f468c68b261de506a3fcb022c8d7cc68f8b0c2ad650852b7c300f274dda251` | 24 | 0.6471 | 138,469 | 0.50 |
| **44** | `variant_w10_seed44.pt` | `bebab580e4caa6934c98a62b095611e4eef9221a7f4282d36f9a4b6928abde64` | 21 | 0.6051 | 138,469 | 0.35 |

Frozen checkpoint location: [`experiments/vieneu_kws_research_v9/artifacts/checkpoints/`](file:///D:/trainidentificationaudio/experiments/vieneu_kws_research_v9/artifacts/checkpoints/)

---

## 3. Data Splits and Manifest Integrity Hashes

The candidate evaluation relies on historical splits whose cryptographic digests remain strictly frozen:

| Dataset / Split Asset | Relative Path | SHA-256 Digest |
| :--- | :--- | :--- |
| **Test Manifest ($N=720$)** | `experiments/vieneu_kws_research_v8/data/splits/test_manifest.csv` | `79e9aa0b20bb921d348b55101af78157276cb56b1059fabb129a0d745d1da2e4` |
| **Val Manifest ($N=360$)** | `experiments/vieneu_kws_research_v8/data/splits/val_manifest.csv` | `cb66a769e6101709e707a092cebd851aa7f7d53ea4850d8a73a3a2f7460a8378` |
| **Train Manifest ($N=1,360$)** | `experiments/vieneu_kws_research_v8/data/splits/train_manifest.csv` | `701a6cc772b2f4d6378e30e2ce2b1745c4716efcd9ab12d649abc015270dee80` |
| **Selected Configs** | `experiments/vieneu_kws_research_v8/calibration/selected_configs.json` | `330fa3c2f4d616ddb065b9da90cb3080e2fcdf65eb2309a64d10e2a3e7d3de72` |

---

## 4. Voice Provenance & Plain Limitation Disclosure

An exhaustive audit of the approved VieNeu-TTS inventory confirmed that exactly 25 voices exist in total:
- **17 Training Voices:** `Bình Dương`, `Cẩm Phả`, `Cần Thơ`, `Cao Lãnh`, `Duy Khánh`, `Hà Tĩnh`, `Hải Dương`, `Hương Ly`, `Mai Phương`, `Mỹ Tho`, `Nam Định`, `Nha Trang`, `Phương Trang`, `Rạch Giá`, `Tây Ninh`, `Thanh Hóa`, `Việt Trì`.
- **4 Validation Voices:** `Bảo Ngọc`, `Diễm My`, `Hoàng Kiên`, `Tuấn Tú`.
- **4 Test Voices:** `Kim Thanh`, `Quốc Tuấn`, `Quỳnh Anh`, `Đức Trí`.

### Explicit Voice Generalization Boundary
The 4 test voices are strictly held out from the v8 training partition and evaluated on 80 brand-new, unseen textual prompts. However, they were utilized in historical project campaigns (Phases 56–69). There are **no unencountered approved TTS voices** available in the project inventory. Therefore:
- The candidate demonstrates generalization to **brand-new unseen text prompts on a fixed synthetic voice inventory**.
- This must **NOT** be claimed as zero-shot generalization to arbitrary human speakers.

---

## 5. Deployment Boundary Notice

This freeze packages an **immutable synthetic research prototype**. It does not constitute a mobile app, production binary, or edge deployment package.
