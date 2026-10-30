# Bản Tiền Đăng Ký Thử Nghiệm Dung Lượng Mô Hình E2
## (Phase E2 Formal Experiment Preregistration & Evaluation Protocol)

> **Mã tài liệu:** `PHASE_E2_PREREGISTRATION`  
> **Repository:** `D:\trainidentificationaudio`  
> **Thời điểm khóa đăng ký:** 2026-10-28  
> **Trạng thái:** LOCKED BEFORE TRAINING EXECUTION  

---

### 1. Ranh Giới Quản Trị Khoa Học Bắt Buộc

Báo cáo này tuân thủ nghiêm ngặt 4 nhãn quản trị bắt buộc:
- `EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`
- `NOT_VALIDATED_ON_NATURAL_SPEECH`
- `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`
- `NOT_VALIDATED_PENDING_DATA`

**Nguyên Tắc Bất Biến Tuyệt Đối:**
1. **Mô hình v11 là mặc định duy nhất:** Checkpoint v11 đóng băng (`variant_v11_hard_negative_weighted_seed42.pt`) và ONNX v11 (`vietnamese_kws_v11.onnx`) được giữ nguyên, không thay đổi cấu hình mặc định.
2. **Cách ly hoàn toàn các tập kiểm thử đã quan sát:**
   - Tuyệt đối KHÔNG sử dụng các tập dữ liệu Phase 43, Phase 51, và Phase 58/59 trong quá trình huấn luyện, chọn lọc mô hình hay tinh chỉnh ngưỡng. Toàn bộ các tập này được bảo tồn vĩnh viễn là các tập benchmark đã quan sát (observed test corpora).
3. **Phân lập biến số thực nghiệm:**
   - Không thay đổi đồng thời VAD, frontend, ngưỡng quyết định, hay ánh xạ nhãn. Biến số thử nghiệm duy nhất được can thiệp là **dung lượng tham số (model parameter capacity)** giữa Baseline 138k và E2 304k.
4. **Không suy diễn triển khai:**
   - Kết quả đánh giá trên tập kiểm định nội bộ `validation_v1.csv` chỉ phục vụ mục đích so sánh năng lực biểu diễn lý thuyết. Không tuyên bố E2 là mô hình sẵn sàng triển khai.

---

### 2. Thiết Kế Thử Nghiệm Đối Chứng (Controlled Experimental Setup)

| Thành Phần Thực Nghiệm | Đặc Tả Cố Định | Cơ Sở / Ràng Buộc |
| :--- | :--- | :--- |
| **Frontend Âm Học** | `LogMelFrontend` (TorchAudio MelSpectrogram) | 16 kHz, n_fft=512, win=400, hop=160, 40 Mel filterbanks, eps=1e-6 |
| **Kích Thước Đầu Vào** | `[B, 1, 40, 101]` | Đúng 16.000 mẫu âm thanh mono 1.0 giây |
| **Không Gian Nhãn (5 lớp)** | 0: `TRAI`, 1: `PHAI`, 2: `LEN`, 3: `XUONG`, 4: `REJECT` | 4 lớp lệnh mục tiêu + 1 lớp từ chối đa nguồn |
| **Chính Sách Ra Quyết Định** | $\theta_{\text{cmd}} = 0.20$ | $\hat{y} = \operatorname{argmax}_{c<4} p_c$ nếu $\max_{c<4} p_c \ge 0.20$, ngược lại $\hat{y} = 4$ |
| **Tập Huấn Luyện (Approved)**| `manifests/train_manifest.csv` | 5.440 mẫu v12 (2.720 lệnh + 2.720 REJECT) |
| **Tập Kiểm Định (Approved)** | `manifests/validation_v1.csv` | 1.280 mẫu v12 (640 lệnh + 640 REJECT) |
| **Hàm Mất Mát (Loss)** | `WeightedCrossEntropyLoss` | Trọng số: `[1.0, 1.0, 1.0, 1.0, 1.25]` cho REJECT |
| **Thuật Toán Tối Ưu** | `AdamW` | lr = $1.0 \times 10^{-3}$, weight_decay = $0.01$ |
| **Lịch Biểu Learning Rate** | `CosineAnnealingLR` | $T_{\max} = 40$, $\eta_{\min} = 1.0 \times 10^{-5}$ |
| **Kích Thước Batch** | 64 | drop_last=True trong huấn luyện |
| **Số Epoch Tối Đa** | 40 epochs | Early stopping: patience = 8 epochs theo `val_macro_f1` (5-class) |
| **Tăng Cường Dữ Liệu** | SpecAugment + TimeShift + Gain | FreqMask $\le 4$, TimeMask $\le 10$, Shift $\pm 800$ mẫu, Gain $\pm 3\text{ dB}$, clamp $>1.0$ |
| **Bộ Hạt Giống Ngẫu Nhiên**| 3 Hạt giống: **42, 43, 44** | Thực hiện độc lập trên cả Baseline (138k) và E2 (304k) |

---

### 3. Hai Họ Mô Hình So Sánh

1. **Họ Baseline (138,469 tham số huấn luyện):**
   - Kiến trúc: `CorrectedVietnameseKWSNet`
   - Kênh Conv: `[32, 64, 64, 128]`
   - Head FC: `128 -> 64 -> 5`
   - Chạy 3 seed: 42, 43, 44.
2. **Họ E2 Scaled Capacity (304,277 tham số huấn luyện):**
   - Kiến trúc: `ScaledVietnameseKWSNet_E2`
   - Kênh Conv: `[48, 96, 96, 192]` (hệ số $\alpha = 1.5$)
   - Head FC: `192 -> 64 -> 5`
   - Chạy 3 seed: 42, 43, 44.

---

### 4. Hệ Thống Chỉ Số Báo Cáo Định Lượng (Evaluation Ledger)

Mỗi mô hình trong 6 lần chạy huấn luyện độc lập sẽ được đánh giá trên toàn bộ 1.280 mẫu của `validation_v1.csv` tại cả 2 chế độ quyết định:
1. Chế độ vận hành chuẩn: **$\theta_{\text{cmd}} = 0.20$**
2. Chế độ tham chiếu: **Standard 5-class Argmax**

Các chỉ số thống kê bắt buộc:
- **Độ chính xác tổng thể (Overall Accuracy):** $\text{Acc} = \frac{N_{\text{correct}}}{1280}$
- **Macro-F1 4 lệnh mục tiêu:** Trung bình F1 của `TRAI`, `PHAI`, `LEN`, `XUONG`.
- **Macro-F1 5 lớp:** Trung bình F1 của cả 5 lớp gồm `REJECT`.
- **Độ nhạy lệnh thấp nhất (Minimum Command Recall):** $\min(R_{\text{TRAI}}, R_{\text{PHAI}}, R_{\text{LEN}}, R_{\text{XUONG}})$.
- **Tỷ lệ báo động giả trên tập OOV (OOV False Alarm Rate - FAR):**
  $$\text{FAR}_{\text{oov}} = \frac{\sum_{i \in \text{REJECT}} \mathbb{I}(\hat{y}_i \ne \text{REJECT})}{N_{\text{REJECT}}} \quad (N_{\text{REJECT}} = 640)$$
- **Độ chuẩn xác của lớp REJECT (REJECT Precision):** $P_{\text{REJECT}} = \frac{\text{TP}_{\text{REJECT}}}{\text{TP}_{\text{REJECT}} + \text{FP}_{\text{REJECT}}}$.
- **Bảng ma trận nhầm lẫn (Confusion Matrix $5 \times 5$).**
- **Độ trễ suy luận (Latency) và Mức sử dụng bộ nhớ (RAM / VRAM).**
- **Thống kê biến thiên giữa các seed (Mean $\pm$ Std).**

---

### 5. Tiêu Chí Quyết Định Khoa Học (Scientific Decision Protocol)

- **Giả thuyết kiểm định:** Việc tăng gấp đôi dung lượng mô hình ($138\text{k} \to 304\text{k}$) có giúp cải thiện độ nhạy phân biệt âm học (Minimum Command Recall, 4-Cmd F1) mà **không làm gia tăng tỷ lệ kích hoạt sai trên từ ngoài tập từ vựng (OOV FAR)** hay không?
- **Rào cản dừng sớm (Stopping Rule):**
  - Nếu mô hình E2 **không vượt trội hơn Baseline trên các chỉ số kiểm định phát triển** (Accuracy, 4-Cmd Macro-F1, Minimum Command Recall), hoặc nếu E2 đạt độ nhạy cao hơn nhưng đi kèm **tăng vọt OOV FAR một cách không thể chấp nhận được**, thì nhóm nghiên cứu sẽ ghi nhận kết luận: **Mở rộng dung lượng không mang lại lợi ích tương xứng với chi phí tính toán (unjustified capacity expansion)**.
  - Ngừng nghiên cứu nhánh E2 và duy trì v11 làm mô hình mặc định.
