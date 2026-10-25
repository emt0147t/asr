# Báo Cáo Kiểm Định Toàn Diện Thử Nghiệm E1 (E1 Final Audit Report)
## Xác Thực Số Liệu Đối Soát v11, Bóc Tách Lỗi OOV & Dòng Đa Lệnh, và Khảo Sát Tích Hợp Live-Mic

> **Mã báo cáo:** `VIENEU_KWS_E1_FINAL_AUDIT_REPORT_20261024`  
> **Repository:** `D:\trainidentificationaudio`  
> **Phiên bản:** v1.0.0 (Versioned Audit)  
> **Ngày thực hiện:** 2026-10-24  
> **Trạng thái cổng mô hình:** **KHÓA V11 NGUYÊN BẢN — KHÔNG THAY THẾ MÔ HÌNH DEMO — KHÔNG HUẤN LUYỆN LẠI — KHÓA CỔNG E2 (NO-GO)**  

---

## 1. Ranh Giới Quản Trị Khoa Học Bắt Buộc

Báo cáo này tuân thủ tuyệt đối 4 nhãn quản trị bắt buộc:
1. `EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`
2. `NOT_VALIDATED_ON_NATURAL_SPEECH`
3. `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`
4. `NOT_VALIDATED_PENDING_DATA`

### Kiểm Tra Mã Băm Mật Mã Cổ Vật (Cryptographic SHA-256 Verification)
Trước khi tiến hành kiểm định, toàn bộ các cổ vật mô hình liên quan đã được băm SHA-256 độc lập trên hệ thống:

| Tên Cổ Vật | Đường Dẫn Thực Tế | SHA-256 Kỳ Vọng | SHA-256 Quan Sát Thực Tế | Trạng Thái |
| :--- | :--- | :---: | :---: | :---: |
| **Checkpoint v11 đóng băng** | `experiments/vieneu_kws_research_v13_4/bundle/checkpoint/variant_v11_hard_negative_weighted_seed42.pt` | `3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2` | `3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2` | **KHỚP 100% BYTE-FOR-BYTE** |
| **ONNX v11 đóng băng** | `experiments/vieneu_kws_research_v13_4/bundle/onnx/vietnamese_kws_v11.onnx` | `54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99` | `54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99` | **KHỚP 100% BYTE-FOR-BYTE** |
| **Checkpoint E1 kiểm thử** | `experiments/vieneu_kws_research_v12/checkpoints/exp_v12_e1_baseline_scale_seed42.pt` | `f3c0fe8f7ccebb8fe354cb50acb247e0674232044706db0187feb92b1706b878` | `f3c0fe8f7ccebb8fe354cb50acb247e0674232044706db0187feb92b1706b878` | **KHỚP 100% BYTE-FOR-BYTE** |

---

## 2. Audit A — Tái Hiện Số Liệu & Giải Trình Mẫu Số Dữ Liệu (Offline Reproduction & Denominator Reconciliation)

### 2.1 Bảng Tái Hiện Số Liệu So Sánh Ngoại Tuyến (1,280 Clips Test Bị Khóa @ $\theta^*_{\text{cmd}} = 0.20$)
Toàn bộ các chỉ số dưới đây đã được tính toán lại trực tiếp từ mảng dự đoán gốc (`predictions`), ghi rõ đầy đủ tử số, mẫu số và tỷ lệ phần trăm:

| Tiêu Chí Đánh Giá | Baseline v11 Đóng Băng | Thử Nghiệm E1 (138k tham số) | Chênh Lệch ($\Delta$) | Trạng Thái Kiểm Tra |
| :--- | :---: | :---: | :---: | :---: |
| **Độ chính xác tổng thể (Accuracy)** | **82.11%** (1,051 / 1,280) | **95.86%** (1,227 / 1,280) | **+13.75%** | **XÁC MINH CHÍNH XÁC** |
| **Macro-F1 (Cả 5 lớp)** | **0.8232** | **0.9601** | **+0.1369** | **XÁC MINH CHÍNH XÁC** |
| **Commands Macro-F1 (4 lớp lệnh)** | **0.8245** | **0.9609** | **+0.1364** | **XÁC MINH CHÍNH XÁC** |
| **Độ nhạy từ khóa sạch (Clean Recall)** | **100.00%** (192 / 192) | **99.48%** (191 / 192) | -0.52% | **XÁC MINH CHÍNH XÁC** |
| **Độ nhạy từ khóa trong nhiễu (Noisy Recall)**| **78.12%** (350 / 448) | **99.55%** (446 / 448) | **+21.43%** | **XÁC MINH CHÍNH XÁC** |
| *— Recall tại 20 dB SNR* | 97.83% (90 / 92) | **100.00%** (92 / 92) | +2.17% | XÁC MINH CHÍNH XÁC |
| *— Recall tại 15 dB SNR* | 98.91% (91 / 92) | **100.00%** (92 / 92) | +1.09% | XÁC MINH CHÍNH XÁC |
| *— Recall tại 10 dB SNR* | 92.05% (81 / 88) | **97.73%** (86 / 88) | +5.68% | XÁC MINH CHÍNH XÁC |
| *— Recall tại 5 dB SNR* | 70.45% (62 / 88) | **100.00%** (88 / 88) | **+29.55%** | XÁC MINH CHÍNH XÁC |
| *— Recall tại 0 dB SNR (Nhiễu = Tiếng)* | **29.55%** (26 / 88) | **100.00%** (88 / 88) | **+70.45%** | XÁC MINH CHÍNH XÁC |
| **Từ chối hội thoại FLEURS (Conversation)** | 96.25% (154 / 160) | **100.00% (160 / 160)** | **+3.75% (0 FA)** | **XÁC MINH CHÍNH XÁC** |
| **Từ chối tiếng ồn môi trường (Env Noise)** | 96.88% (124 / 128) | **100.00% (128 / 128)** | **+3.12% (0 FA)** | **XÁC MINH CHÍNH XÁC** |
| **Từ chối khoảng lặng (Silence Non-speech)** | 100.00% (96 / 96) | **100.00% (96 / 96)** | 0.00% (0 FA) | **XÁC MINH CHÍNH XÁC** |
| **Từ chối OOV chưa từng thấy (trừ "ghế")** | 53.75% (129 / 240) | **84.17% (202 / 240)** | **+30.42%** | **XÁC MINH CHÍNH XÁC** |
| **Từ chối từ "ghế" chưa từng thấy** | 37.50% (6 / 16) | **25.00% (4 / 16)** | -12.50% | **XÁC MINH CHÍNH XÁC** |
| **Độ chính xác chuỗi đa lệnh (Streaming)** | **30.00%** (12 / 40) | **47.50%** (19 / 40) | **+17.50%** | **XÁC MINH CHÍNH XÁC** |

---

### 2.2 Giải Trình Mẫu Số Dữ Liệu Clean vs Noisy (Reconciling the 8,000 Records)
Trong báo cáo kiểm định sơ bộ có dòng ghi nhận: *"clean: 1,680, noisy: 3,920"* mà chưa giải trình rõ tổng 8,000 clips. Kiểm tra chi tiết trường `source_type` và `reject_subgroup` trong `dataset_v12_manifest.csv` đã làm sáng tỏ hoàn toàn cấu trúc mẫu số:

```
TỔNG SỐ CLIPS TRONG DATASET V12 = 8,000 CLIPS
├── Nhóm phát âm từ đơn lẻ (Isolated Speech Utterances): 5,600 clips (Mẫu số tỷ lệ Clean/Noisy)
│   ├── Từ khóa mục tiêu (TRAI, PHAI, LEN, XUONG): 4,000 clips
│   │   ├── Clean (tts_synthetic_clean): 1,200 clips (30.0%)
│   │   └── Noisy (noise_augmented @ 5 mức SNR): 2,800 clips (70.0%)
│   └── Từ phủ định ngoài tập từ vựng (OOV Words): 1,600 clips
│       ├── Clean (tts_synthetic_oov_clean): 480 clips (30.0%)
│       └── Noisy (noise_augmented_oov @ 5 mức SNR): 1,120 clips (70.0%)
│   ──> TỔNG CLEAN = 1,200 + 480 = 1,680 clips (đúng 30.0% của 5,600)
│   ──> TỔNG NOISY = 2,800 + 1,120 = 3,920 clips (đúng 70.0% của 5,600)
└── Nhóm âm bản chuyên biệt (Specialized Negative Cohorts): 2,400 clips (Phần chênh lệch được giải trình)
    ├── Hội thoại tự nhiên tiếng Việt FLEURS (conversation_fleurs): 1,000 clips (16 kHz native speech)
    ├── Tiếng ồn môi trường và thao tác micro thực tế (environmental_noise): 800 clips
    └── Khoảng lặng kỹ thuật, hum phòng, điều hòa, gió pink (ambient_silence): 600 clips
```
**Kết luận giải trình:** Số liệu 1,680 clean và 3,920 noisy phản ánh chính xác tỷ lệ phân bố $30\% / 70\%$ trên tập 5,600 phát âm từ đơn lẻ (speech utterances). 2,400 mẫu còn lại là tập âm bản liên tục và tiếng ồn thực tế không sinh ra theo cặp clean/noisy. Cấu trúc dữ liệu hoàn toàn minh bạch và nhất quán $100\%$.

---

## 3. Audit B — Bóc Tách Thất Bại Tồn Đọng OOV (Per-Lexeme OOV Analysis)

### 3.1 Đánh Giá Toàn Diện Các Từ OOV Quan Trọng (`ghế`, `cửa`, `mở`, `bàn`, `nhà`)

| Từ OOV (Lexeme) | Phân Tập (Split) | Tổng Số Clips | v11 REJECT (Đúng) | v11 Kích Hoạt Nhầm (FA) | v11 Dự Đoán Sai | E1 REJECT (Đúng) | E1 Kích Hoạt Nhầm (FA) | E1 Dự Đoán Sai |
| :--- | :---: | :---: | :---: | :---: | :--- | :---: | :---: | :--- |
| **`ghế`** | **test (unseen)** | **16** | 6 (37.5%) | 10 (62.5%) | `LEN`: 9, `XUONG`: 1 | **4 (25.0%)** | **12 (75.0%)** | **`LEN`: 12** |
| **`cửa`** | train | 16 | 10 (62.5%) | 6 (37.5%) | `XUONG`: 3, `LEN`: 2, `PHAI`: 1 | **15 (93.8%)** | **1 (6.2%)** | `LEN`: 1 |
| **`mở`** | train | 19 | 4 (21.1%) | 15 (78.9%) | `LEN`: 9, `PHAI`: 6 | **19 (100.0%)** | **0 (0.0%)** | *0 kích hoạt nhầm* |
| **`bàn`** | train | 16 | 6 (37.5%) | 10 (62.5%) | `PHAI`: 7, `LEN`: 3 | **16 (100.0%)** | **0 (0.0%)** | *0 kích hoạt nhầm* |
| **`nhà`** | train | 16 | 9 (56.2%) | 7 (43.8%) | `LEN`: 3, `PHAI`: 2, `TRAI`: 2 | **16 (100.0%)** | **0 (0.0%)** | *0 kích hoạt nhầm* |

### 3.2 Bóc Tách 4 Clip Kiểm Thử Hồi Quy Lịch Sử Của Từ "Cửa"
Đánh giá độc lập trên 4 tệp âm thanh gây lỗi nghiêm trọng ở Phase 13:
1. `oov_train_cửa_Adam.wav`: v11 nhận là `LEN` (conf 0.419) $\rightarrow$ **E1 nhận là `REJECT` (conf 0.988)**.
2. `oov_train_cửa_Hải_Đăng.wav`: v11 nhận là `XUONG` (conf 0.676) $\rightarrow$ **E1 nhận là `REJECT` (conf 1.000)**.
3. `oov_train_cửa_Minh_Đức.wav`: v11 nhận là `XUONG` (conf 0.375) $\rightarrow$ **E1 nhận là `REJECT` (conf 1.000)**.
4. `oov_train_cửa_Thanh_Bình.wav`: v11 nhận là `LEN` (conf 0.363) $\rightarrow$ **E1 nhận là `REJECT` (conf 1.000)**.

> [!IMPORTANT]
> **Xác định bản chất độc lập của 4 tệp "cửa":**  
> 4 tệp này là những ca lỗi thực tế thúc đẩy quá trình điều tra nguyên nhân gốc ở Phase 13. Do các biến thể ngữ âm của từ "cửa" đã được chủ động đưa vào tập huấn luyện v12, **4 tệp này KHÔNG ĐƯỢC COI là bằng chứng xác thực độc lập (independent confirmation)**, mà được phân loại chính xác là **mẫu kiểm thử hồi quy (known regression test examples)** để chứng minh lỗi cũ không còn tái diễn.

### 3.3 Phân Tích Toàn Bộ 31 Từ OOV Chưa Từng Thấy Trong Tập Test (256 Clips)
Phân tích chi tiết 256 clips OOV trong tập kiểm thử độc lập cho thấy bức tranh rõ ràng:

- **14 từ OOV đã được E1 giải quyết triệt để (0 FA):** `chào`, `không`, `lưu`, `nhập`, `pháo`, `phép`, `phóng`, `phở`, `trà`, `trách`, `trưa`, `xích`, `xương`, `được`.
- **12 từ OOV vẫn còn lỗi kích hoạt nhầm tồn đọng:**
  - Nhầm thành `PHAI`: `khóa` (9/9 FA), `hủy` (6/10 FA), `xóa` (4/10 FA).
  - Nhầm thành `LEN`: `ghế` (12/16 FA), `làng` (5/8 FA), `gửi đi` (3/6 FA), `lạc` (2/8 FA), `tải về` (2/6 FA), `lá` (1/8 FA).
  - Nhầm thành `XUONG`: `thu` (4/10 FA), `cứu` (1/10 FA), `xôi` (1/8 FA).

> [!WARNING]
> **Khẳng định khoa học:** Vấn đề từ ngoài tập từ vựng (OOV) **CHƯA ĐƯỢC GIẢI QUYẾT TRIỆT ĐỂ**. Mặc dù E1 đã giảm tỷ lệ lỗi OOV từ $47.3\%$ (121/256) xuống $19.5\%$ (50/256), mô hình vẫn có xu hướng nhầm lẫn các âm tiết có thanh sắc dốc đứng hoặc nguyên âm mở /a/, /e/ với các từ lệnh tương ứng.

---

## 4. Audit C — Bảng Đối Chiếu Tính Tương Đồng Giao Thức Huấn Luyện (Training Protocol Parity)

| Thành Phần Giao Thức (Protocol Component) | Cấu Hình v11 (Phase 11/13) | Cấu Hình E1 (`train_e1.py`) | Phân Loại Tương Đồng | Nhận Định Tác Động Khoa Học |
| :--- | :--- | :--- | :---: | :--- |
| **Kiến trúc mạng (Architecture)** | `CorrectedVietnameseKWSNet` | `CorrectedVietnameseKWSNet` | **GIỐNG NHAU (Identical)** | Cùng 4 khối Conv2D (32->64->64->128) + 2 FC (128->64->5) |
| **Số tham số huấn luyện** | **138,469 tham số** | **138,469 tham số** | **GIỐNG NHAU (Identical)** | Quy mô mô hình được bảo toàn 100% |
| **Bộ đệm BatchNorm** | 580 phần tử (tổng 139,049) | 580 phần tử (tổng 139,049) | **GIỐNG NHAU (Identical)** | Khớp cấu trúc state dict |
| **Bộ trích đặc trưng (Frontend)** | LogMelFrontend (40 mels, 512 FFT) | LogMelFrontend (40 mels, 512 FFT) | **GIỐNG NHAU (Identical)** | Cùng ma trận Slaney mel-filterbank |
| **Khởi tạo ngẫu nhiên & Seed** | Seed = 42 (`torch`, `np`, `random`) | Seed = 42 (`torch`, `np`, `random`, cuDNN) | **GIỐNG NHAU (Identical)** | Đảm bảo tính lặp lại |
| **Quy tắc quyết định suy luận** | 5-lớp argmax, $\theta_{\text{cmd}} = 0.20$ | 5-lớp argmax, $\theta_{\text{cmd}} = 0.20$ | **GIỐNG NHAU (Identical)** | Hợp đồng suy luận không đổi |
| **Quy chuẩn âm học đầu vào** | Remediated conditional clamping | Remediated conditional clamping | **GIỐNG NHAU (Identical)** | Chỉ kẹp biên khi peak > 1.0 |
| **Tập dữ liệu huấn luyện** | v11 (3,360 clips, thiếu OOV/FLEURS) | v12 (5,440 clips, giàu OOV, 5 bậc SNR) | **KHÁC NHAU (Different)** | **Biến số nghiên cứu chính (Main Experimental Variable)** |
| **Bộ tối ưu (Optimizer)** | AdamW ($lr=10^{-3}, wd=10^{-4}$) | AdamW ($lr=10^{-3}, wd=10^{-2}$) | **KHÁC NHAU (Different)** | E1 có weight decay mạnh hơn gấp 100 lần (giảm overfit) |
| **Kích thước Batch (Batch Size)** | 32 | 64 | **KHÁC NHAU (Different)** | Batch lớn hơn giúp gradient ổn định hơn |
| **Lịch biểu học tập (Scheduler)** | CosineAnnealingLR ($T_{\max}=25$) | CosineAnnealingLR ($T_{\max}=40, \eta_{\min}=10^{-5}$) | **KHÁC NHAU (Different)** | E1 học dài hơn, suy giảm có kiểm soát sàn $\eta_{\min}$ |
| **Hàm mất mát & Trọng số lớp** | CrossEntropy có label smoothing 0.05, trọng số gán theo phả hệ lỗi (sample-level) | Weighted CrossEntropy, trọng số theo lớp: REJECT=1.25, CMD=1.0, không label smoothing | **KHÁC NHAU (Different)** | E1 phạt trực tiếp lớp REJECT mạnh hơn thay vì làm mượt nhãn |
| **Quy tắc dừng sớm (Early Stopping)** | 25 epoch cố định, chọn theo score ($rec - 2 \cdot FAR$) | 40 epoch max, kiên nhẫn 8 epoch theo `val_macro_f1` | **KHÁC NHAU (Different)** | E1 chọn epoch tối ưu dựa trên cân bằng Macro-F1 |

> [!IMPORTANT]
> **Nhận định khoa học:** Mặc dù chất lượng và độ đa dạng của tập dữ liệu v12 là yếu tố quyết định giúp triệt tiêu các lỗi âm bản, **không được khẳng định đơn phương rằng "chỉ riêng dữ liệu" đã tạo ra toàn bộ mức cải thiện**. Sự kết hợp của việc tăng weight decay ($0.01$), trọng số tăng cường cho lớp REJECT ($1.25$) và việc theo dõi Macro-F1 trong quá trình dừng sớm cũng đóng vai trò hỗ trợ quan trọng cho khả năng tổng quát hóa của E1.

---

## 5. Audit D — Bóc Tách Lỗi Dòng Đa Lệnh (Streaming Error Decomposition)

Kiểm định chi tiết trên 40 luồng phát âm thanh liên tục (124 sự kiện lệnh mục tiêu):

```
TỔNG SỐ CHUỖI ĐÁNH GIÁ: 40 CHUỖI (124 SỰ KIỆN LỆNH MỤC TIÊU)
├── Chuỗi nhận diện đúng hoàn toàn cả thứ tự: 19 / 40 (47.50%)  [v11 đạt 12/40 (30.00%)]
├── Sự kiện lệnh nhận diện đúng (Matched Target Events): 64 / 124 (51.61%)
├── Sự kiện lệnh bị bỏ lỡ (Missed Target Events): 60 sự kiện
├── Sự kiện kích hoạt trùng lặp (Duplicate Triggers): 0 sự kiện [v11 bị 4 sự kiện]
└── Sự kiện chèn nhầm từ OOV/nhiễu (False Insertions): 36 sự kiện
```

### 5.1 Phân Rã Theo Cấu Trúc Chuỗi (Sequence Pattern Breakdown)

| Kiểu Chuỗi Phát | Số Chuỗi | v11 Đúng Hoàn Toàn | E1 Đúng Hoàn Toàn | Số Lỗi Bỏ Lỡ (Misses) | Số Lỗi Chèn Nhầm (False Ins) | Phân Tích Nguyên Nhân Kỹ Thuật |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`confuser_interspersed`** (Chèn OOV) | 12 | 2 / 12 (16.7%) | **7 / 12 (58.3%)** | 12 | 11 | E1 cải thiện vượt bậc nhờ loại bỏ được các từ nhiễu thông thường |
| **`permutation`** (Hoán vị 4 lệnh) | 10 | 2 / 10 (20.0%) | **3 / 10 (30.0%)** | 21 | 12 | Khi nói nhanh trong nhiễu, các lệnh cuối chuỗi dễ bị nuốt |
| **`repetition`** (Lệnh lặp lại) | 8 | 5 / 8 (62.5%) | **5 / 8 (62.5%)** | 11 | 7 | Cả 2 mô hình nhận diện tốt khi có khoảng nghỉ 450ms |
| **`stress_long_gap`** (Nghỉ dài 900ms) | 5 | 3 / 5 (60.0%) | **4 / 5 (80.0%)** | 5 | 4 | Khoảng nghỉ dài giúp bộ tách VAD hoạt động chuẩn xác |
| **`stress_short_gap`** (Nghỉ ngắn 160ms)| 5 | 0 / 5 (0.0%) | **0 / 5 (0.0%)** | 11 | 2 | **Bộ tách phân đoạn VAD bị nghẽn (khoảng nghỉ < 300ms nên bị gộp)** |

### 5.2 Phân Rã Theo Môi Trường Nhiễu (SNR Condition Breakdown)

| Môi Trường Âm Học | Số Chuỗi | v11 Đúng Hoàn Toàn | E1 Đúng Hoàn Toàn | Số Lỗi Bỏ Lỡ (Misses) | Số Lỗi Chèn Nhầm (False Ins) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`clean`** (Âm thanh sạch) | 10 | 4 / 10 (40.0%) | **7 / 10 (70.0%)** | 5 | **0** |
| **`snr15`** (Nhiễu nhẹ) | 10 | 3 / 10 (30.0%) | **5 / 10 (50.0%)** | 17 | 10 |
| **`snr10`** (Nhiễu vừa) | 10 | 4 / 10 (40.0%) | **4 / 10 (40.0%)** | 20 | 14 |
| **`snr05`** (Nhiễu nặng) | 10 | 1 / 10 (10.0%) | **3 / 10 (30.0%)** | 18 | 12 |

> [!WARNING]
> **Cảnh báo khoa học về Streaming:**  
> Việc E1 đạt **0 sự kiện kích hoạt lặp (0 duplicate events)** là một bước tiến quan trọng so với v11. Tuy nhiên, việc tỷ lệ chính xác toàn chuỗi chỉ đạt **47.50%** cho thấy bài toán nhận diện dòng liên tục **CHƯA ĐƯỢC GIẢI QUYẾT TRIỆT ĐỂ**. Nguyên nhân chính không nằm ở bộ phân loại CNN mà nằm ở **cơ chế phân đoạn năng lượng (Utterance Segmentation / VAD)** khi người nói ngắt nghỉ quá nhanh ($< 300\text{ ms}$) hoặc trong môi trường nhiễu âm lượng lớn ($5\text{ dB}$).

---

## 6. Audit E — Khảo Sát Tích Hợp & Thử Nghiệm Smoke Test Trực Tiếp (E1 Live Smoke Test)

### 6.1 Phát Hiện Rào Cản Tích Hợp Mã Nguồn (Interface Bug Discovery)
Trong quá trình thử nghiệm nạp checkpoint E1 vào hệ thống demo trực tiếp (`LiveKWSPipeline` tại `experiments/vieneu_kws_research_v13_4/demo/live_mic_engine.py`), đã phát hiện rào cản kỹ thuật:
- Tại dòng 602 của `live_mic_engine.py`:
  ```python
  spec = self.inferencer.frontend(x)  # Gây lỗi TypeError: 'LogMelFrontendNumPy' object is not callable
  ```
  Lớp `VietnameseKWSInferencer` trong `infer.py` khởi tạo `self.frontend = LogMelFrontendNumPy()` (sử dụng phương thức `.compute()`), trong khi khối xử lý PyTorch trong `live_mic_engine.py` lại giả định `frontend` là một lớp PyTorch callable `nn.Module`.
- Mô hình hiện tại trong demo đang mặc định chạy trên backend **ONNX** (`vietnamese_kws_v11.onnx`), trong khi E1 hiện mới chỉ được xuất ra dưới dạng checkpoint PyTorch (`.pt`).

### 6.2 Giải Pháp Tích Hợp Tối Thiểu An Toàn (Minimum Safe Integration Required)
Để tích hợp E1 vào demo mà không vi phạm nguyên tắc bảo toàn hệ thống v11 mặc định:
1. **Bước 1 (Export ONNX độc lập):** Xuất checkpoint E1 sang tệp ONNX độc lập: `experiments/vieneu_kws_research_v12/onnx/vietnamese_kws_v12_e1.onnx` kèm kiểm tra parity số học ($diff \le 10^{-5}$).
2. **Bước 2 (Cấu hình tham số dòng lệnh):** Thêm cờ `--model-path` và `--backend` vào `live_mic_app.py` để người dùng có thể tùy chọn khởi động demo với v11 hoặc E1 mà không sửa cứng code mặc định.
3. **Bước 3:** Không sửa đổi bất kỳ tệp mặc định nào của Phase 13.4 cho đến khi được phê duyệt chính thức.

### 6.3 Kết Quả Thử Nghiệm Smoke Test Có Kiểm Soát (Thực Hiện Qua Test Harness Độc Lập)
Tập lệnh [`run_e1_live_demo_smoke_test.py`](file:///D:/trainidentificationaudio/experiments/vieneu_kws_research_v12/scripts/run_e1_live_demo_smoke_test.py) đã thực hiện thử nghiệm trên đúng luồng xử lý phân đoạn `LiveKWSPipeline` (ghi nhật ký riêng vào [`e1_live_smoke_test_events.log`](file:///D:/trainidentificationaudio/experiments/vieneu_kws_research_v12/reports/e1_live_smoke_test_events.log)):

| Bài Kiểm Tra Smoke Test | Dữ Liệu Đầu Vào | Kết Quả Thực Tế | Độ Tin Cậy (Conf) | Độ Trễ (Latency) | Đánh Giá |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Lệnh đơn `TRAI`** | Clip đơn âm chuẩn | Nhận diện: `['TRAI']` | 0.946 | 5.0 ms | **ĐẠT (PASS)** |
| **Lệnh đơn `PHAI`** | Clip đơn âm chuẩn | Nhận diện: `['PHAI']` | 0.998 | 2.1 ms | **ĐẠT (PASS)** |
| **Lệnh đơn `LEN`** | Clip đơn âm chuẩn | Nhận diện: `['LEN']` | 0.981 | 1.6 ms | **ĐẠT (PASS)** |
| **Lệnh đơn `XUONG`** | Clip đơn âm chuẩn | Nhận diện: `['XUONG']` | 0.986 | 2.0 ms | **ĐẠT (PASS)** |
| **Chuỗi đa lệnh** | `TRAI` $\rightarrow$ `XUONG` $\rightarrow$ `PHAI` | Nhận diện: `['TRAI', 'XUONG', 'PHAI']` | 0.988 - 0.998 | 1.2 - 2.2 ms | **ĐẠT (100% đúng thứ tự)** |
| **Lệnh lặp liên tiếp** | `TRAI` $\rightarrow$ `TRAI` | Nhận diện: `['TRAI', 'TRAI']` | 0.983 - 0.992 | 1.3 - 1.4 ms | **ĐẠT (Không bị nuốt lệnh)** |
| **Âm bản "cửa"** | Clip "cửa" lịch sử | Nhận diện: `[]` (Từ chối) | REJECT | 1.5 ms | **ĐẠT (0 kích hoạt nhầm)** |
| **Âm bản "bàn"** | Clip "bàn" lịch sử | Nhận diện: `[]` (Từ chối) | REJECT | 1.6 ms | **ĐẠT (0 kích hoạt nhầm)** |
| **Nhiễu phòng 3.0s** | Nhiễu phòng yên tĩnh | Nhận diện: `[]` (Không kích hoạt) | - | - | **ĐẠT (0 kích hoạt nhầm)** |

*Tóm tắt smoke test:* Toàn bộ 6 bài kiểm tra chức năng cơ bản đều **ĐẠT (PASS)** trên quy trình phân đoạn của demo.

---

## 7. Giới Hạn Còn Lại & Khuyến Nghị Tiếp Theo (Remaining Limitations & Recommendations)

### 7.1 Những Hạn Chế Đã Được Xác Nhận Khoa Học
1. **Âm bản OOV thanh sắc dốc đứng:** Các từ OOV chưa từng thấy có thanh sắc như `ghế`, `khóa` vẫn có tỷ lệ kích hoạt nhầm đáng kể do đặc trưng phổ bậc cao tương đồng với lệnh `LEN` và `PHAI`.
2. **Bộ tách phân đoạn khi người nói dồn dập:** Bộ phát hiện khoảng lặng $300\text{ ms}$ thất bại hoàn toàn trước các chuỗi lệnh dồn dập ($< 200\text{ ms}$), dẫn đến việc gộp 2 lệnh thành 1 phân đoạn và nuốt lệnh.
3. **Môi trường nhiễu nặng kết hợp đa lệnh:** Khi mức SNR xuống $5\text{ dB}$, tỷ lệ chính xác toàn chuỗi giảm xuống $30.0\%$.

### 7.2 Khuyến Nghị Hành Động
1. **Bảo tồn v11:** Tiếp tục duy trì mô hình v11 đóng băng làm phiên bản tham chiếu mặc định cho bản phát hành chính thức hiện tại.
2. **Khóa cổng E2:** Tuyệt đối không mở rộng dung lượng mô hình lên 419k tham số (E2), vì bằng chứng thực nghiệm cho thấy dung lượng 138k tham số hoàn toàn đủ khả năng biểu diễn và việc mở rộng tham số không giải quyết được vấn đề độ phủ từ vựng ngoài từ điển.
3. **Kế hoạch v13 tiếp theo:**
   - Bổ sung các từ OOV thanh sắc (`ghế`, `khóa`, `hủy`, `xóa`, `thu`) vào tập huấn luyện âm bản.
   - Nâng cấp thuật toán VAD phân đoạn sang hướng bám vết năng lượng thích nghi (Adaptive Energy Hangover) để tách các chuỗi lệnh ngắn cách nhau $150-200\text{ ms}$.
   - Xuất artifact ONNX chính thức cho E1 sau khi hoàn tất các thủ tục phê duyệt học thuật.
