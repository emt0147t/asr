# Báo Cáo Đối Chiếu Thực Nghiệm Dung Lượng Mô Hình E2
## (Phase E2 Controlled Capacity Experiment & Validation Comparison Report)

> **Mã tài liệu:** `PHASE_E2_VALIDATION_COMPARISON`  
> **Repository:** `D:\trainidentificationaudio`  
> **Thời điểm thẩm định:** 2026-10-28  
> **Cổng quyết định nghiên cứu:** **STOP_CAPACITY_EXPANSION (DỪNG MỞ RỘNG DUNG LƯỢNG — DUY TRÌ v11 MẶC ĐỊNH)**  
> **Trạng thái:** FORMAL AUDIT COMPLETE  

---

### 1. Ranh Giới Quản Trị Khoa Học Bắt Buộc

Báo cáo này tuân thủ nghiêm ngặt 4 nhãn quản trị bắt buộc:
- `EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`
- `NOT_VALIDATED_ON_NATURAL_SPEECH`
- `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`
- `NOT_VALIDATED_PENDING_DATA`

**Các cam kết quản trị bất biến:**
1. **Mô hình mặc định v11:** Mô hình gốc v11 (`variant_v11_hard_negative_weighted_seed42.pt` và `vietnamese_kws_v11.onnx`) được bảo toàn nguyên vẹn 100%, không bị sửa đổi, ghi đè hay thay thế.
2. **Không xâm phạm các tập test lịch sử:** Toàn bộ các tập kiểm thử đã quan sát gồm Phase 43, Phase 51, và Phase 58/59 tuyệt đối không bị truy cập hoặc sử dụng trong việc huấn luyện, chọn mô hình hay tinh chỉnh ngưỡng.
3. **Phân lập biến số:** Thử nghiệm giữ nguyên hoàn toàn frontend Log-Mel, kích thước phổ `[1, 1, 40, 101]`, thuật toán tối ưu (AdamW, CosineAnnealing), tăng cường SpecAugment, tập dữ liệu huấn luyện và ngưỡng quyết định vận hành $\theta_{\text{cmd}} = 0.20$. Biến số duy nhất được so sánh là **dung lượng tham số (138.469 vs 304.277 tham số)**.
4. **Không tuyên bố triển khai:** Thử nghiệm chỉ đánh giá năng lực đại diện trên tập kiểm định nội bộ `validation_v1.csv` ($N=1.280$).

---

### 2. Tổng Quan Thiết Kế Kiến Trúc & Dung Lượng

| Đặc Tính Kỹ Thuật | Baseline Scale (E1 Architecture) | Scaled Capacity (E2 Architecture) | Tỷ Lệ Biến Đổi ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Tên lớp mô hình** | `CorrectedVietnameseKWSNet` | `ScaledVietnameseKWSNet_E2` | Topology tương đương |
| **Kênh tích chập (Conv Channels)** | `[32, 64, 64, 128]` | `[48, 96, 96, 192]` | Width multiplier $\alpha = 1.5\times$ |
| **Phân loại Fully-Connected** | `128 -> 64 -> 5` | `192 -> 64 -> 5` | Bảo toàn bottleneck 64 neurons |
| **Số tham số huấn luyện (Trainable)**| **138.469** | **304.277** | **$+119.74\%$ ($2.20\times$)** |
| **Phần tử đệm BatchNorm (Buffers)** | 580 | 868 | $+49.66\%$ |
| **Kích thước checkpoint PyTorch** | 1.648,5 KB (~1.61 MB) | 3.592,8 KB (~3.51 MB) | $+117.94\%$ ($2.18\times$) |
| **Độ trễ suy luận CPU (1 luồng)** | $2.09 \pm 0.75\text{ ms}$ | $3.06 \pm 0.46\text{ ms}$ | $+46.37\%$ |
| **Độ trễ suy luận GPU (RTX A2000)** | $2.23 \pm 0.90\text{ ms}$ | $2.59 \pm 0.18\text{ ms}$ | $+16.14\%$ |
| **Mức chiếm dụng VRAM tối đa** | 258,0 MB | 343,6 MB | $+33.18\%$ |

---

### 3. Bảng Tổng Hợp Kết Quả Thực Nghiệm Đa Hạt Giống (N=1.280 trên `validation_v1.csv`)

Mỗi mô hình được huấn luyện độc lập qua 3 hạt giống chuẩn: **seed 42, 43, 44** trên 5.440 mẫu `train_manifest.csv` và kiểm định đối chứng trên 1.280 mẫu `validation_v1.csv`.

#### 3.1 So Sánh Thống Kê Tổng Hợp (Mean $\pm$ Std) Tại Ngưỡng Vận Hành $\theta_{\text{cmd}} = 0.20$

| Chỉ Số Đánh Giá (Metric) | Baseline 138k (3 seeds) | E2 Scaled 304k (3 seeds) | Chênh Lệch ($\Delta = \text{E2} - \text{Base}$) | Kết Luận Thống Kê |
| :--- | :---: | :---: | :---: | :--- |
| **Train Loss** | $0.1008 \pm 0.0095$ | $0.0865 \pm 0.0139$ | $-0.0143$ | Khớp tốt hơn trên tập train |
| **Train Accuracy** | $96.60\% \pm 0.48\%$ | $97.12\% \pm 0.55\%$ | $+0.52\%$ | Tăng nhẹ độ khớp dữ liệu train |
| **Validation Loss** | $0.2060 \pm 0.0057$ | $0.2116 \pm 0.0285$ | $+0.0056$ | Val loss của E2 không giảm (thậm chí tăng) |
| **Độ chính xác (Accuracy $\theta_{0.20}$)** | **$92.71\% \pm 0.75\%$** | **$92.92\% \pm 0.91\%$** | **$+0.21\%$** | **Không có ý nghĩa thống kê** ($< 1\text{ std}$) |
| **4-Command Macro-F1** | **$92.85\% \pm 0.67\%$** | **$93.14\% \pm 0.66\%$** | **$+0.29\%$** | Cải thiện không đáng kể |
| **5-Class Macro-F1** | **$92.81\% \pm 0.69\%$** | **$93.06\% \pm 0.76\%$** | **$+0.25\%$** | Nằm trong biên độ biến thiên ngẫu nhiên |
| **Độ nhạy lệnh tối thiểu (Min-R)** | **$96.04\% \pm 0.36\%$** | **$95.63\% \pm 0.63\%$** | **$-0.41\%$** | **E2 bị suy giảm độ nhạy lệnh tối thiểu** |
| **Tỷ lệ báo động giả OOV (OOV FAR)** | **$12.13\% \pm 1.66\%$** | **$12.65\% \pm 1.49\%$** | **$+0.52\%$** | **E2 làm tăng tỷ lệ kích hoạt sai trên REJECT** |
| **Độ chuẩn xác REJECT (Precision)** | $98.03\% \pm 0.32\%$ | $98.82\% \pm 0.73\%$ | $+0.79\%$ | Tăng nhẹ độ chính xác khi dự đoán REJECT |

---

#### 3.2 Bảng Đối Chiếu Từng Hạt Giống Độc Lập (Individual Run Breakdown)

| Run ID | Params | Best Epoch | Train Loss | Val Loss | Acc ($\theta_{0.20}$) | 4-Cmd F1 | 5-Class F1 | Min-R | OOV FAR | Rej Prec | CPU Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `baseline_138k_seed42` | 138.469 | 35 | 0.1086 | 0.2120 | 91.95% | 92.22% | 92.14% | 95.63% | 13.91% | 98.39% | 1.29 ms |
| `baseline_138k_seed43` | 138.469 | 34 | 0.0902 | 0.2051 | 93.44% | 93.55% | 93.51% | 96.25% | 10.62% | 97.78% | 2.23 ms |
| `baseline_138k_seed44` | 138.469 | 32 | 0.1036 | 0.2008 | 92.73% | 92.77% | 92.77% | 96.25% | 11.87% | 97.92% | 2.76 ms |
| `e2_scaled_304k_seed42` | 304.277 | 36 | 0.0788 | 0.1999 | 93.36% | 93.44% | 93.42% | 95.63% | 11.72% | 98.95% | 2.64 ms |
| `e2_scaled_304k_seed43` | 304.277 | 39 | 0.0781 | 0.1909 | 93.52% | 93.60% | 93.57% | 96.25% | 11.87% | 99.47% | 3.55 ms |
| `e2_scaled_304k_seed44` | 304.277 | 28 | 0.1025 | 0.2441 | 91.87% | 92.38% | 92.19% | 95.00% | 14.37% | 98.03% | 3.00 ms |

---

### 4. Phân Tích Hiệu Suất Từng Lớp (Per-Class Performance)

Đánh giá chi tiết Precision (P), Recall (R), và F1-Score (F1) trung bình qua 3 hạt giống tại $\theta_{\text{cmd}} = 0.20$:

| Lớp (Class) | Hỗ Trợ (Support) | Baseline 138k: P | Baseline 138k: R | Baseline 138k: F1 | E2 304k: P | E2 304k: R | E2 304k: F1 | Nhận Xét Chênh Lệch |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`TRAI`** | 160 | 92.43% | **96.04%** | **94.19%** | 91.08% | 95.63% | 93.29% | **Baseline tốt hơn E2 ở cả P, R, F1** |
| **`PHAI`** | 160 | 80.36% | 97.50% | 88.08% | **82.17%** | **99.38%** | **89.94%** | E2 tăng nhẹ độ nhạy cho `PHAI` (+1.86% F1) |
| **`LEN`** | 160 | 88.82% | 97.08% | 92.74% | 87.83% | **99.17%** | **93.15%** | E2 tăng độ nhạy nhưng giảm độ chính xác |
| **`XUONG`** | 160 | 93.37% | 99.59% | **96.37%** | 92.83% | **99.79%** | 96.19% | Tương đương (Baseline F1 cao hơn 0.18%) |
| **`REJECT`** | 640 | 98.03% | **87.86%** | 92.66% | **98.82%** | 87.34% | **92.72%** | E2 từ chối ít hơn (FAR cao hơn 0.52%) |

---

### 5. Phân Tích Ma Trận Nhầm Lẫn Tổng Hợp (Tổng 3 Seeds, $N=3.840$)

```
Baseline 138k (Sum 3 Seeds, N=3,840):
                 Pred: TRAI  PHAI   LEN XUONG REJECT
Actual: TRAI         [ 461     3     0     0     16 ]  (Recall: 96.04%)
        PHAI         [  10   468     0     0      2 ]  (Recall: 97.50%)
        LEN          [   0     0   466     0     14 ]  (Recall: 97.08%)
        XUONG        [   0     0     0   478      2 ]  (Recall: 99.58%)
        REJECT       [  28   112    59    34   1687 ]  (FAR: 233/1920 = 12.14%)

Scaled E2 304k (Sum 3 Seeds, N=3,840):
                 Pred: TRAI  PHAI   LEN XUONG REJECT
Actual: TRAI         [ 459     6     0     0     15 ]  (Recall: 95.63%)
        PHAI         [   3   477     0     0      0 ]  (Recall: 99.38%)
        LEN          [   0     0   476     0      4 ]  (Recall: 99.17%)
        XUONG        [   0     0     0   479      1 ]  (Recall: 99.79%)
        REJECT       [  42    98    66    37   1677 ]  (FAR: 243/1920 = 12.66%)
```

#### Bóc Tách Cơ Chế Lỗi:
1. **Lỗi báo động giả trên REJECT tăng ở E2:** E2 ghi nhận tổng cộng **243 lần kích hoạt nhầm trên REJECT**, so với **233 lần của Baseline**. Cụ thể, số lần REJECT bị nhận nhầm thành `TRAI` tăng vọt từ 28 lên 42 (+50%), thành `LEN` tăng từ 59 lên 66, và thành `XUONG` tăng từ 34 lên 37.
2. **Hiện tượng đánh đổi (Trade-off):** E2 bắt được thêm 9 lệnh `PHAI` và 10 lệnh `LEN` hợp lệ, nhưng cái giá phải trả là mất 2 lệnh `TRAI` và **thêm 10 lần báo động giả nghiêm trọng từ các mẫu OOV / hội thoại**.
3. **Độ nhạy lệnh tối thiểu (Minimum Command Recall):** Tại seed 44, E2 tụt Min-R xuống mức $95.00\%$, trong khi Baseline duy trì mức đáy $95.63\%$ và trung bình $96.04\%$.

---

### 6. Phân Tích Khoa Học: Mở Rộng Dung Lượng Có Đem Lại Lợi Ích Đo Lường Được?

Thực nghiệm cho thấy:
1. **Bão hòa dung lượng biểu diễn (Capacity Saturation):**  
   Việc tăng hơn gấp đôi số lượng tham số ($138\text{k} \to 304\text{k}$) chỉ giúp cải thiện độ chính xác chung thêm $+0.21\%$ (từ $92.71\%$ lên $92.92\%$) và Macro-F1 thêm $+0.25\%$. Khoảng chênh lệch này hoàn toàn nằm dưới độ lệch chuẩn của quá trình khởi tạo ngẫu nhiên ($0.75\% - 0.91\%$).
2. **Không giải quyết được bài toán False Alarm:**  
   Vấn đề cốt lõi của KWS tiếng Việt trên môi trường nhiễu âm học là **sự tương đồng ngữ âm giữa các từ phủ định (hard negatives) và từ khóa mục tiêu**. Mở rộng dung lượng nơ-ron không giúp mô hình học cách phân biệt tốt hơn nếu không có sự bổ sung về dữ liệu biên; ngược lại, mạng lớn hơn có xu hướng tăng số lượng false triggers trên các âm tiết OOV tương đồng.
3. **Chi phí tính toán vượt trội:**  
   Mô hình E2 tăng $119.7\%$ số tham số, tăng $117.9\%$ kích thước tệp, và tăng $46.4\%$ độ trễ suy luận trên CPU. Đây là sự đánh đổi không tương xứng khi hiệu năng phân loại thực tế không có cải thiện có ý nghĩa thống kê.

---

### 7. Quyết Định Nghiên Cứu Chính Thức (Formal Research Decision)

> [!IMPORTANT]
> **QUYẾT ĐỊNH NGHIÊN CỨU:** **`STOP_CAPACITY_EXPANSION` (CHẤM DỨT NHÁNH MỞ RỘNG DUNG LƯỢNG E2)**  
> **MÔ HÌNH MẶC ĐỊNH DUY TRÌ:** **`Legacy v11` (`CorrectedVietnameseKWSNet`, 138.469 tham số)**

**Căn Cứ Khoa Học:**
- Tuân thủ nghiêm ngặt Điều 10 của yêu cầu nghiên cứu: *"If E2 is not better on development metrics or adds cost without measurable benefit, report that finding and stop."*
- Thực nghiệm đối chứng đa hạt giống xác nhận kiến trúc E2 304k không đem lại ưu thế phân biệt vượt trội so với Baseline 138k, đồng thời làm tăng OOV FAR và gia tăng gánh nặng tài nguyên tính toán.
- Do đó, chính thức khép lại nhánh nghiên cứu mở rộng dung lượng kiến trúc E2; không khuyến nghị tiếp tục mở rộng tham số hay xuất mô hình E2 sang môi trường thử nghiệm tiếp theo.

---

### 8. Bảng Kê Mã Băm Cổ Vật Thẩm Định (Cryptographic SHA-256 Catalog)

| Tên Cổ Vật | Đường Dẫn Tệp | SHA-256 |
| :--- | :--- | :---: |
| **Thiết Kế Kiến Trúc E2** | `PHASE_E2_ARCHITECTURE_DESIGN.md` | `b58b92d141c3819d6267e97502ed97101c0c30b163b993f76f4dd338f3202aa5` |
| **Bản Tiền Đăng Ký E2** | `PHASE_E2_PREREGISTRATION.md` | `e8c7d998846daf280e803696ab99e141c0dc98142a906b9f130e775df48adf54` |
| **Sổ Bộ Chạy E2** | `PHASE_E2_RUN_LEDGER.csv` | `ca12da4448e77ae30bac4e9c6062f6ca49d9ddd66e5643daa3df571466c78983` |
| **Script Huấn Luyện & Đo Đạc** | `experiments/vieneu_kws_research_v12/scripts/train_eval_phase_e2.py` | `b512c216813d9319ced83031628aabaf23de682e6a25d515a0e5b0781f27cd53` |
| **Dữ Liệu Đo Đạc Chi Tiết** | `experiments/vieneu_kws_research_v12/reports/phase_e2_evaluation_results.json` | `e84efd8fa4c798e2c1d85350b644058a71c9f017f28415e867540965ad45bf06` |
| **Baseline Seed 42 Checkpoint** | `experiments/vieneu_kws_research_v12/checkpoints/baseline_138k_seed42.pt` | `44f7e223974169d37e4a62e0668f1f8effcf70637f2a0cd9a8396046de61c24f` |
| **Baseline Seed 43 Checkpoint** | `experiments/vieneu_kws_research_v12/checkpoints/baseline_138k_seed43.pt` | `02171e45c2a123c7819a2b55fbdf5ebdc38aea7b253ce7d793c36bc68ef8e06d` |
| **Baseline Seed 44 Checkpoint** | `experiments/vieneu_kws_research_v12/checkpoints/baseline_138k_seed44.pt` | `71628501c51ad01bfbc36901cd6e9f564d6522204b79a1ef8473340e7cf58798` |
| **E2 Scaled Seed 42 Checkpoint**| `experiments/vieneu_kws_research_v12/checkpoints/e2_scaled_304k_seed42.pt` | `0870b6222d50a93d2689870288727f56862cee305b64b4808edd69f8640afbbc` |
| **E2 Scaled Seed 43 Checkpoint**| `experiments/vieneu_kws_research_v12/checkpoints/e2_scaled_304k_seed43.pt` | `a1969f6e38a2d08d92f9d59965922897195d7d4acdc02d2d6e43893e7ad74668` |
| **E2 Scaled Seed 44 Checkpoint**| `experiments/vieneu_kws_research_v12/checkpoints/e2_scaled_304k_seed44.pt` | `71f609e54f85f7ac24e9a1aed58e261fdb6c5b7185758b0c7b6b5c015714d2d1` |
