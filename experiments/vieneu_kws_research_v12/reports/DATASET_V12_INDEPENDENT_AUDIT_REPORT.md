# Báo Cáo Kiểm Định Độc Lập Tập Dữ Liệu Vietnamese KWS v12
## (Independent Audit Report — Vietnamese KWS Dataset v12 & Pre-E1 Gate Verification)

> **Mã tài liệu:** `DATASET_V12_INDEPENDENT_AUDIT_REPORT`  
> **Repository:** `D:\trainidentificationaudio`  
> **Thời điểm thẩm định:** 2026-10-23  
> **Mã băm tập tin phân chia:** `24702bfd64b19457e75c7c7d08b60ec8a10321dc73e0e2496ae591b3d5dac1bb`  
> **Mã băm bảng kê dữ liệu (Manifest SHA-256):** `bca9cb604d6b21668a70d4690297bffac25be935bd3843de3c3d3435dfb77806`  
> **Cổng kỹ thuật tiến hành thử nghiệm E1:** **PASSED (ĐỦ ĐIỀU KIỆN TIẾN HÀNH E1)**  
> **Trạng thái thẩm định người nghe:** **PENDING / NOT_PERFORMED (`NOT_VALIDATED_PENDING_DATA`)**  

---

## 1. Ranh Giới Quản Trị Khoa Học Bắt Buộc

Báo cáo này tuân thủ nghiêm ngặt 4 nhãn quản trị bắt buộc:
1. `EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`
2. `NOT_VALIDATED_ON_NATURAL_SPEECH`
3. `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`
4. `NOT_VALIDATED_PENDING_DATA`

### Nguyên Tắc Bất Biến Về Cổ Vật (Immutable Artifacts)
- **Checkpoint v11 (`variant_v11_hard_negative_weighted_seed42.pt`):**
  - SHA-256 Kỳ vọng: `3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2`
  - SHA-256 Thực tế: `3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2`
  - Trạng thái: **KHỚP 100% BYTE-FOR-BYTE — NGUYÊN BẢN TUYỆT ĐỐI**
- **ONNX v11 (`vietnamese_kws_v11.onnx`):**
  - SHA-256 Kỳ vọng: `54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99`
  - SHA-256 Thực tế: `54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99`
  - Trạng thái: **KHỚP 100% BYTE-FOR-BYTE — NGUYÊN BẢN TUYỆT ĐỐI**

---

## 2. Kết Quả Kiểm Định Chi Tiết 10 Hạng Mục (Phase A Audit)

### A1. Khối Lượng và Tính Toàn Vẹn Số Liệu (Counts & Integrity)
- **Tổng số clips:** 8,000 (Mục tiêu: 8,000) — **ĐẠT**
- **Clips từ khóa mục tiêu (Positive Keywords):** 4,000 clips
  - `TRAI`: 1,000 clips
  - `PHAI`: 1,000 clips
  - `LEN`: 1,000 clips
  - `XUONG`: 1,000 clips
- **Clips từ chối (REJECT):** 4,000 clips
  - `oov_word`: 1,600 clips (từ ngoài tập từ vựng, âm bản thanh hỏi "cửa", "mở", "bàn", "nhà", test giữ lại "ghế")
  - `conversation`: 1,000 clips (trích từ FLEURS tiếng Việt 16 kHz không chứa từ khóa)
  - `environmental_noise`: 800 clips (tiếng ồn mic, văn phòng, phòng hội nghị)
  - `silence_non_speech`: 600 clips (khoảng lặng kỹ thuật, hum điều hòa, quạt gió)
- **Định dạng âm thanh:** 100% tệp là WAV Mono, tần số lấy mẫu 16,000 Hz, đúng 16,000 mẫu (1.0 giây). 0 tệp mất, 0 tệp lỗi I/O.
- **Biên độ & SNR:** 0 tệp xảy ra clipping (>0.999), 0 tệp từ khóa bị im lặng. Sai lệch giữa SNR danh định và thực tế $\le 1.5	ext{ dB}$.

### A2. Kiểm Định Nhiễm Bẩn Nhãn REJECT (Blocking Audit — REJECT Label Correctness)
- **Vấn đề đã xử lý:** Đã tiến hành quét toàn bộ 1,218 tệp nguồn hội thoại FLEURS tiếng Việt để loại bỏ triệt để 146 tệp nguồn có chứa các từ khóa `trái`, `phải`, `lên`, `xuống`.
- **Kết quả tái kiểm định:**
  - Tổng số tệp hội thoại REJECT trong manifest: **1000**
  - Số tệp có transcript nguồn giao cắt với bất kỳ từ khóa nào (`trái`, `phải`, `lên`, `xuống`): **0 / 1000 (0.00%)**
  - Số tệp OOV bị nhiễm từ khóa: **0 / 1600 (0.00%)**
  - **Kết luận:** Nguy cơ rò rỉ từ khóa mục tiêu vào nhãn REJECT đã được **triệt tiêu hoàn toàn (PASS)**.

### A3. Cách Ly Phân Chia Tập & Không Rò Rỉ OOV (Split Isolation & Leakage)
- **Phân chia mẫu:**
  - `train`: 5,440 clips (68.0%)
  - `val`: 1,280 clips (16.0%)
  - `test`: 1,280 clips (16.0%)
- **Cách ly giọng đọc (Voice Disjointness):**
  - Train (700 giọng), Val (164 giọng), Test (168 giọng).
  - Số lượng rò rỉ giọng đọc giữa các tập: **0**
- **Cách ly nguồn phát sinh (Source Utterance Isolation):**
  - Toàn bộ các biến thể tăng cường (noise, time-shift) từ cùng một mẫu gốc đều nằm trọn vẹn trong một tập phân chia duy nhất.
  - Số lượng rò rỉ mẫu gốc: **0**
- **Cách ly từ OOV kiểm thử (Unseen OOV Isolation):**
  - Tập `test` chứa 31 loại từ OOV chưa từng thấy (bao gồm "ghế" và 30 âm tiết OOV độc lập).
  - Độ giao nhau giữa OOV của test với train/val: $\emptyset$ (0 từ rò rỉ).
- **Cách ly tệp nhiễu (Noise Disjointness):**
  - Bộ tệp nhiễu nền được phân bổ tách biệt hoàn toàn giữa Train (tracks 0..27), Val (tracks 28..33), và Test (tracks 34..39). Rò rỉ: **0**.

### A4. Giải Trình Chênh Lệch Tiêu Chí 10 (Subjective Listening QA Addendum)
- **Kiểm định kỹ thuật tự động:** ĐẠT 100% (Checks 1-9).
- **Thẩm định nghe chủ quan của con người:** Ghi nhận trung thực `PENDING / NOT_PERFORMED` (`NOT_VALIDATED_PENDING_DATA`). Nghiêm cấm ngụy tạo kết quả của người nghe.

---

## 3. Khắc Phục Bất Đồng Kiến Trúc & Số Lượng Tham Số (Phase B Resolution)

### Phát Hiện Bất Đồng Tài Liệu
- Trong tài liệu cũ `experiment_e1_e2_plan.json` và `EXPERIMENT_E1_E2_DESIGN.md`, kiến trúc v11 được ghi chú là ~148,000 tham số với 3 khối Conv2D (32, 64, 128).
- Tuy nhiên, kiểm tra trực tiếp state_dict của checkpoint có thẩm quyền `variant_v11_hard_negative_weighted_seed42.pt` xác nhận:
  - **Tên lớp kiến trúc:** `CorrectedVietnameseKWSNet`
  - **Số lớp Conv2D:** 4 lớp (`conv1`: 1->32, `conv2`: 32->64, `conv3`: 64->64, `conv4`: 64->128)
  - **Pooling:** 3 x `MaxPool2d(2, 2)` + 1 x `AdaptiveAvgPool2d((1, 1))`
  - **Lớp kết nối đầy đủ:** `fc1` (128->64) + `fc2` (64->5)
  - **Tổng số tham số có thể huấn luyện (Trainable Parameters):** **Chính xác 138,469 tham số**
  - **Bộ đệm BatchNorm (Non-trainable Buffers):** 580 phần tử (tổng state_dict: 139,049 phần tử).

### Quyết Định Cho Thử Nghiệm E1 (`exp_v12_e1_baseline_scale`)
- Thử nghiệm E1 **BẮT BUỘC sử dụng chính xác kiến trúc `CorrectedVietnameseKWSNet` với 138,469 tham số huấn luyện**.
- Việc giữ nguyên chính xác 100% quy mô và cấu trúc của mô hình v11 là điều kiện tiên quyết để cô lập biến số nghiên cứu: **chỉ có chất lượng và phân phối dữ liệu v12 là biến số thay đổi**, từ đó chứng minh một cách khoa học liệu cải thiện dữ liệu có giải quyết triệt để lỗi false acceptance hay không.

---

## 4. Quyết Định Cổng Quản Trị (Governance Gate Decision)

| Tiêu Chí Quyết Định | Trạng Thái | Ghi Chú |
| :--- | :---: | :--- |
| **Cổng thẩm định dữ liệu v12 (Data Integrity Gate)** | **PASSED** | Đạt toàn bộ 9 kiểm tra kỹ thuật tự động |
| **Nhiễm bẩn nhãn REJECT (Target Contamination)** | **RESOLVED** | 0/1,000 tệp FLEURS chứa từ khóa |
| **Bất đồng kiến trúc mô hình (Model Architecture)** | **RESOLVED** | Xác định đúng 138,469 tham số (`CorrectedVietnameseKWSNet`) |
| **Cổng khởi động huấn luyện E1 (E1 Execution Gate)** | **APPROVED** | Được phép tiến hành thử nghiệm E1 |
| **Cổng mở rộng tham số E2 (E2 Scaled Capacity)** | **LOCKED** | Bị khóa cho đến khi hoàn tất đánh giá độc lập E1 |
| **Sẵn sàng triển khai thực tế (Production Readiness)** | **BLOCKED** | `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION` |
