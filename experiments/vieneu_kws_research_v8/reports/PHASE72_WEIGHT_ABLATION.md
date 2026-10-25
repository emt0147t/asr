# Phase 72 — Controlled Hard-Negative Weight Ablation Report

**Workspace:** `D:\trainidentificationaudio`  
**Experiment Directory:** `experiments/vieneu_kws_research_v8/`  
**Model Architecture:** `CorrectedVietnameseKWSNet` (138,469 parameters, global average pooling)  
**Status Banner:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Executive Summary & Predeclared Rationale

Phase 72 executes a controlled ablation on the hard-negative loss weight $w_{\text{hn}}$ across seeds 42, 43, and 44, holding all other components strictly fixed (command weight $w_{\text{cmd}}=2.0$, silence weight $w_{\text{silence}}=0.5$, architecture, AdamW optimizer, Cosine Annealing scheduler, 25 epochs).

### Predeclared Ablation Grid & Mathematical Rationale:
- **`variant_w10` ($w_{\text{hn}} = 1.0$):** Mild hard-negative weight. Tests whether reducing hard-negative pressure further boosts recall on difficult keywords like `PHAI`.
- **`variant_w12` ($w_{\text{hn}} = 1.2$):** Candidate B reference re-implementation. Reconstructs baseline behavior under fresh v8 pipeline.
- **`variant_w14` ($w_{\text{hn}} = 1.4$):** Balanced class-mass parity. On 244 hard-negative training samples, $244 \times 1.4 = 341.6$, perfectly balancing the mass of an individual command class ($170 \times 2.0 = 340.0$).
- **`variant_w16` ($w_{\text{hn}} = 1.6$):** Stronger hard-negative suppression. Evaluates whether stronger gradient pressure on confusable minimal pairs reduces FAR towards the $\le 5.0\%$ target while preserving total REJECT mass ($608.4$) well below total command mass ($1,360.0$).

---

## 2. Dataset Composition Audit (Training Split)

| Partition / Class | Samples | Loss Weight $w_i$ | Total Loss Mass | Mass Ratio vs Command Class (340.0) |
| :--- | :--- | :--- | :--- | :--- |
| **TRAI** | 170 | 2.0 | 340.0 | $1.00\times$ |
| **PHAI** | 170 | 2.0 | 340.0 | $1.00\times$ |
| **LEN** | 170 | 2.0 | 340.0 | $1.00\times$ |
| **XUONG** | 170 | 2.0 | 340.0 | $1.00\times$ |
| **Hard Negatives (Minimal Pairs)** | 244 | $w_{\text{hn}} \in [1.0, 1.6]$ | 244.0 to 390.4 | $0.72\times$ to $1.15\times$ |
| **Plain Silence & Ambient** | 436 | 0.5 | 218.0 | $0.64\times$ |
| **Total Commands** | 680 | 2.0 | 1,360.0 | $4.00\times$ |
| **Total REJECT** | 680 | Mixed | 462.0 to 608.4 | $1.36\times$ to $1.79\times$ |

---

## 3. Training Convergence & Best Validation Epoch Performance

| Variant | $w_{\text{hn}}$ | Seed | Best Epoch | Train Loss | Val Acc | Val Cmd Recall | Val Non-Target FAR | TRAI | PHAI | LEN | XUONG | Checkpoint SHA-256 (prefix) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **variant_w10** | 1.0 | 42 | 22 | 0.6991 | 96.11% | **95.63%** | **3.50%** | 90.0% | 95.0% | 100.0% | 97.5% | `7374af469a4b` |
| **variant_w10** | 1.0 | 43 | 24 | 0.6471 | 96.67% | **98.12%** | **4.50%** | 97.5% | 100.0% | 100.0% | 95.0% | `77f468c68b26` |
| **variant_w10** | 1.0 | 44 | 21 | 0.6051 | 97.22% | **96.88%** | **2.50%** | 95.0% | 97.5% | 100.0% | 95.0% | `bebab580e4ca` |
| **variant_w12** | 1.2 | 42 | 22 | 0.6930 | 95.56% | **95.63%** | **4.50%** | 92.5% | 95.0% | 100.0% | 95.0% | `a242b3f374a9` |
| **variant_w12** | 1.2 | 43 | 22 | 0.6302 | 97.50% | **98.12%** | **3.00%** | 97.5% | 100.0% | 100.0% | 95.0% | `ea62a1519560` |
| **variant_w12** | 1.2 | 44 | 24 | 0.6587 | 95.28% | **95.00%** | **4.50%** | 95.0% | 92.5% | 100.0% | 92.5% | `9fed5cccef13` |
| **variant_w14** | 1.4 | 42 | 24 | 0.7103 | 95.00% | **92.50%** | **3.00%** | 90.0% | 95.0% | 100.0% | 85.0% | `588066356f28` |
| **variant_w14** | 1.4 | 43 | 24 | 0.6820 | 97.22% | **98.13%** | **3.50%** | 100.0% | 95.0% | 100.0% | 97.5% | `2f178666f60d` |
| **variant_w14** | 1.4 | 44 | 22 | 0.6822 | 95.83% | **93.75%** | **2.50%** | 90.0% | 100.0% | 100.0% | 85.0% | `0a06d419a123` |
| **variant_w16** | 1.6 | 42 | 20 | 0.7943 | 95.00% | **93.12%** | **3.50%** | 92.5% | 87.5% | 100.0% | 92.5% | `d1f89a17a79e` |
| **variant_w16** | 1.6 | 43 | 20 | 0.6681 | 95.56% | **95.63%** | **4.50%** | 87.5% | 100.0% | 100.0% | 95.0% | `a2649a64def8` |
| **variant_w16** | 1.6 | 44 | 25 | 0.7132 | 96.39% | **95.00%** | **2.50%** | 95.0% | 92.5% | 100.0% | 92.5% | `4fff76d11912` |

---

## 4. Cross-Seed Validation Comparison

| Variant Descriptor | Weight $w_{\text{hn}}$ | Mean Val Recall $\pm$ Std | Mean Val FAR $\pm$ Std | Mean Val 5-Class Accuracy | Best Operating Candidate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Mild Hard-Negative (w_hn=1.0)** | 1.0 | **96.88% $\pm$ 1.25%** | **3.50% $\pm$ 1.00%** | 96.67% | `variant_w10` |
| **Baseline Candidate B Re-implementation (w_hn=1.2)** | 1.2 | **96.25% $\pm$ 1.65%** | **4.00% $\pm$ 0.87%** | 96.11% | `variant_w12` |
| **Balanced Hard-Negative Parity (w_hn=1.4)** | 1.4 | **94.79% $\pm$ 2.96%** | **3.00% $\pm$ 0.50%** | 96.02% | `variant_w14` |
| **Stronger Hard-Negative Suppression (w_hn=1.6)** | 1.6 | **94.58% $\pm$ 1.31%** | **3.50% $\pm$ 1.00%** | 95.65% | `variant_w16` |

---

## 5. Next Steps for Phase 73

- All 12 checkpoints and metadata hashes are permanently archived in `experiments/vieneu_kws_research_v8/models/checkpoints/`.
- In Phase 73, these variants alongside historical Cascade v1 and Candidate B will undergo robust validation calibration across clean and synthetic noise/gain conditions to select the top finalists for locked test benchmarking.
