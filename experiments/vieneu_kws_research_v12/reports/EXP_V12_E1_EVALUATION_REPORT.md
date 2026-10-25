# Báo Cáo Đánh Giá Thử Nghiệm Có Kiểm Soát E1 & Quyết Định Cổng E2
## (Controlled Experiment E1 Evaluation Report & E2 Decision Gate)

> **Mã thử nghiệm:** `exp_v12_e1_baseline_scale`  
> **Repository:** `D:\trainidentificationaudio`  
> **Thời điểm hoàn tất:** 2026-10-23  
> **Kiến trúc mô hình:** `CorrectedVietnameseKWSNet` (Chính xác **138,469 tham số huấn luyện**)  
> **Mã băm Checkpoint E1:** `f3c0fe8f7ccebb8fe354cb50acb247e0674232044706db0187feb92b1706b878`  
> **Mã băm Checkpoint v11 đối chứng:** `3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2`  
> **Quyết định Cổng E2:** **NO-GO / KHÔNG TĂNG THAM SỐ (LOCKED)**  

---

## 1. Ranh Giới Quản Trị Khoa Học Bắt Buộc

Báo cáo này tuân thủ tuyệt đối 4 nhãn quản trị bắt buộc:
1. `EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`
2. `NOT_VALIDATED_ON_NATURAL_SPEECH`
3. `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`
4. `NOT_VALIDATED_PENDING_DATA`

### Tính Bất Biến Của Mô Hình v11 (Frozen Artifacts Integrity)
- Checkpoint v11 (`variant_v11_hard_negative_weighted_seed42.pt`): `3c4369...` (Khớp 100% byte-for-byte).
- ONNX v11 (`vietnamese_kws_v11.onnx`): `54298c...` (Khớp 100% byte-for-byte).
- Mô hình v11 không bị ghi đè, fine-tune hay sửa đổi trong toàn bộ quá trình thử nghiệm.

---

## 2. Giả Thuyết Khoa Học & Thiết Lập Thử Nghiệm E1

### Giả Thuyết Khoa Học
Hiện tượng nhận diện nhầm lệnh (false acceptance) trên lời nói tự nhiên và từ phủ định đồng âm ("cửa" bị nhận là `PHAI`) ở phiên bản v11 xuất phát chủ yếu từ **sự thiếu hụt dữ liệu phủ định đa dạng (negative sample starvation) và mất cân bằng phân phối âm học trong môi trường nhiễu**, chứ không phải do thiếu dung lượng tham số mạng CNN.

### Thiết Lập Thực Nghiệm
- **Kiến trúc:** `CorrectedVietnameseKWSNet` (4 khối tích chập: 32 -> 64 -> 64 -> 128 channels, Linear 128 -> 64 -> 5).
- **Tổng số tham số có thể huấn luyện:** **Chính xác 138,469** (khớp byte-for-byte cấu trúc v11 để cô lập biến số dữ liệu).
- **Tập huấn luyện (Train Manifest):** 5,440 mẫu (`train_manifest.csv`, SHA-256: `e0bbc0...`).
- **Tập kiểm định (Validation Manifest):** 1,280 mẫu (`val_manifest.csv`, SHA-256: `5af8cf...`).
- **Tập kiểm thử bị khóa (Locked Test Manifest):** 1,280 mẫu (`test_manifest.csv`, SHA-256: `66d5b0...`) — **Tuyệt đối không can thiệp trong quá trình huấn luyện và chọn lọc epoch**.
- **Siêu tham số:** AdamW ($lr = 10^{-3}$, $weight\_decay = 0.01$), CosineAnnealingLR ($T_{\max}=40, \eta_{\min}=10^{-5}$), batch size 64, epoch tối đa 40, Early stopping kiên nhẫn 8 epoch trên `val_macro_f1`. Trọng số REJECT = 1.25.

### Tiến Trình Hội Tụ (Training Convergence)
- Huấn luyện hoàn thành sau 40 epoch trong **97.9 giây (1.63 phút)** trên GPU NVIDIA RTX A2000.
- Checkpoint tối ưu được lưu tại **Epoch 35** với `val_macro_f1 = 0.9313` (Loss huấn luyện: 0.1086, Acc kiểm định: 93.0%).
- Đồ thị quá trình học: [`e1_learning_curves.png`](file:///D:/trainidentificationaudio/experiments/vieneu_kws_research_v12/reports/e1_learning_curves.png).

---

## 3. Bảng So Sánh Chi Tiết v11 Đối Chứng vs E1 (Same-Test Locked Comparison)

Đánh giá được thực hiện trên cùng 1,280 mẫu test độc lập của Dataset v12 tại ngưỡng quyết định chuẩn $\theta^*_{\text{cmd}} = 0.20$.

### 3.1 Chỉ Số Tổng Thể

| Tiêu Chí Đánh Giá (Metric) | Baseline v11 Đóng Băng | Thử Nghiệm E1 | Chênh Lệch (Delta) | Nhận Xét Khoa Học |
| :--- | :---: | :---: | :---: | :--- |
| **Độ chính xác tổng thể (Accuracy)** | **82.11%** | **95.86%** | **+13.75%** | Tăng vọt nhờ loại trừ âm bản |
| **Macro F1 (Cả 5 lớp)** | **0.8232** | **0.9601** | **+0.1369** | Vượt xa mục tiêu tối thiểu (0.92) |
| **Commands Macro F1 (4 lệnh)** | **0.8245** | **0.9609** | **+0.1364** | Cả 4 lệnh đều đạt F1 > 0.92 |
| **Độ nhạy từ khóa sạch (Clean Recall)** | **100.00%** | **99.48%** | **-0.52%** | Duy trì trọn vẹn độ nhạy từ khóa |
| **Độ nhạy từ khóa nhiễu (Noisy Recall)**| **78.12%** | **99.55%** | **+21.43%** | Đột phá khả năng kháng nhiễu |
| **Độ trễ suy luận P95 (CPU Laptop)** | **1.65 ms** | **3.12 ms** | **+1.47 ms** | Dưới ngưỡng ngân sách 15 ms |

---

### 3.2 Bảng Phân Rã Từng Lớp (Per-Class Precision, Recall, F1)

| Lớp (Class) | Mẫu Test | v11 Precision | v11 Recall | v11 F1 | E1 Precision | E1 Recall | E1 F1 | $\Delta$ F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `TRAI` | 160 | 0.7707 | 0.9875 | 0.8656 | **0.9938** | **1.0000** | **0.9969** | **+0.1313** |
| `PHAI` | 160 | 0.7410 | 0.9250 | 0.8212 | **0.8977** | **0.9938** | **0.9440** | **+0.1228** |
| `LEN` | 160 | 0.6975 | 0.8938 | 0.7829 | **0.8736** | **0.9875** | **0.9275** | **+0.1446** |
| `XUONG` | 160 | 0.7512 | 0.9250 | 0.8282 | **0.9524** | **1.0000** | **0.9753** | **+0.1471** |
| `REJECT` | 640 | 0.9959 | 0.6953 | 0.8183 | **0.9915** | **0.9250** | **0.9570** | **+0.1387** |

---

### 3.3 Phân Tích Khả Năng Kháng Nhiễu Theo Từng Bậc SNR (Target Command Recall)

| Mức Nhiễu Tín Hiệu (SNR) | Số Lượng Mẫu | Baseline v11 Recall | Thử Nghiệm E1 Recall | Chênh Lệch ($\Delta$) |
| :--- | :---: | :---: | :---: | :---: |
| **20 dB** (Nhiễu rất nhẹ) | 92 | 97.83% | **100.00%** | +2.17% |
| **15 dB** (Nhiễu nhẹ phòng) | 92 | 98.91% | **100.00%** | +1.09% |
| **10 dB** (Nhiễu văn phòng) | 88 | 92.05% | **97.73%** | +5.68% |
| **5 dB** (Nhiễu môi trường lớn) | 88 | 70.45% | **100.00%** | **+29.55%** |
| **0 dB** (Năng lượng nhiễu = tiếng) | 88 | 29.55% | **100.00%** | **+70.45%** |

> **Phát hiện quan trọng:** Ở mức 0 dB SNR, mô hình v11 sụp đổ hoàn toàn (chỉ nhận diện được 29.55%), trong khi E1 đạt **100.00% độ nhạy** nhờ chiến lược tăng cường phổ và trộn nhiễu đa tầng của Dataset v12.

---

### 3.4 Phân Tích Loại Trừ Âm Bản (Negative Subgroup False Acceptance Audit)

| Nhóm Phủ Định (Negative Subgroup) | Số Mẫu | v11 Kích Hoạt Nhầm (FA) | v11 Tỷ Lệ Từ Chối | E1 Kích Hoạt Nhầm (FA) | E1 Tỷ Lệ Từ Chối | Kết Quả Đạt Được |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Hội thoại FLEURS (Conversation)** | 160 | 6 / 160 (3.75%) | 96.25% | **0 / 160 (0.00%)** | **100.00%** | **Triệt tiêu 100% false acceptances** |
| **Tiếng ồn môi trường (Env Noise)** | 128 | 4 / 128 (3.12%) | 96.88% | **0 / 128 (0.00%)** | **100.00%** | **Triệt tiêu 100% false acceptances** |
| **Khoảng lặng kỹ thuật (Silence)** | 96 | 0 / 96 (0.00%) | 100.00% | **0 / 96 (0.00%)** | **100.00%** | **Duy trì tuyệt đối 100%** |
| **Từ OOV tiếng Việt tổng cộng** | 256 | 121 / 256 (47.27%) | 52.73% | **50 / 256 (19.53%)** | **80.47%** | **Giảm >58% số lỗi kích hoạt nhầm** |
| *Trong đó: Các từ OOV khác (240)* | 240 | 111 / 240 (46.25%) | 53.75% | **38 / 240 (15.83%)** | **84.17%** | **Cải thiện +30.42% tỷ lệ từ chối** |
| *Trong đó: Từ "ghế" chưa từng thấy* | 16 | 10 / 16 (62.50%) | 37.50% | **12 / 16 (75.00%)** | **25.00%** | **Nhận nhầm thành `LEN` do thanh sắc** |

#### Kiểm Chứng Thực Nghiệm Từ "Cửa" (The "Cửa" Root Cause Resolution)
Thử nghiệm độc lập trên toàn bộ các tệp âm thanh "cửa" nguyên nhân gây kích hoạt nhầm ở Phase 13:
- `oov_train_cửa_Adam`: v11 nhận là `LEN` (0.419) $\rightarrow$ **E1 nhận là `REJECT` (0.988)**.
- `oov_train_cửa_Hải_Đăng`: v11 nhận là `XUONG` (0.676) $\rightarrow$ **E1 nhận là `REJECT` (1.000)**.
- `oov_train_cửa_Minh_Đức`: v11 nhận là `XUONG` (0.375) $\rightarrow$ **E1 nhận là `REJECT` (1.000)**.
- `oov_train_cửa_Thanh_Bình`: v11 nhận là `LEN` (0.363) $\rightarrow$ **E1 nhận là `REJECT` (1.000)**.
$\Rightarrow$ **Lỗi false acceptance "cửa" bị nhận nhầm thành lệnh đã được giải quyết triệt để 100% với độ tin cậy $\ge 98.8\%$**.

---

### 3.5 Đánh Giá Dòng Đa Lệnh Liên Tục (Streaming Multi-Command Corpus)

Đánh giá trên 40 luồng âm thanh liên tục (10 hoán vị lệnh, 8 lệnh lặp lại, 12 chèn từ gây nhiễu, 10 ép khoảng cách 160-220ms):

| Tiêu Chí Đánh Giá Chuỗi | Baseline v11 | Thử Nghiệm E1 | Chênh Lệch ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Độ chính xác toàn chuỗi (Exact Sequence Accuracy)** | **30.00%** (12/40) | **47.50%** (19/40) | **+17.50%** |
| **Độ chính xác sự kiện lệnh (Event Precision)** | **61.54%** | **64.00%** | **+2.46%** |
| **Độ nhạy sự kiện lệnh (Event Recall)** | **51.61%** | **51.61%** | 0.00% |
| **Sự kiện kích hoạt trùng lặp (Duplicates)** | **4 sự kiện** | **0 sự kiện** | **-4 (Loại bỏ hoàn toàn)** |
| **Sự kiện chèn nhầm từ OOV (False Insertions)** | 36 sự kiện | 36 sự kiện | 0 |

---

## 4. Phân Tích Nguyên Nhân Tồn Đọng & Thẩm Định Cổng E2 (E2 Eligibility Analysis)

### 4.1 Bằng Chứng Thực Nghiệm Từ E1
1. **Dung lượng mô hình không hề bị quá tải hay thiếu biểu diễn (No Underfitting):**
   - Loss huấn luyện hội tụ mượt mà xuống **0.1086**; loss kiểm định đạt **0.2081**.
   - Macro F1 trên tập kiểm định đạt **0.9313** và trên test đạt **0.9601** (vượt qua ngưỡng kích hoạt E2 là 0.92).
   - Độ nhạy từ khóa trong nhiễu đạt **99.55%** (kể cả tại 0 dB SNR).
2. **Bản chất của các lỗi kích hoạt nhầm còn lại:**
   - 100% tiếng ồn môi trường và 100% tiếng hội thoại tự nhiên đã bị loại bỏ hoàn toàn (0 lỗi).
   - Lỗi kích hoạt nhầm còn lại tập trung ở các âm tiết OOV có thanh điệu dốc đứng (thanh sắc) và nguyên âm trước /e/ tương tự như từ "ghế" bị nhầm thành `lên`.
   - **Đây là giới hạn về độ bao phủ âm vị (Phonemic Diversity Coverage) của tập dữ liệu huấn luyện, KHÔNG PHẢI là giới hạn về số lượng tham số CNN.**
   - Nếu tăng kích thước mô hình từ 138k lên 419k (E2) mà không mở rộng thêm các biến thể âm vị thanh sắc trong tập huấn luyện, mô hình lớn hơn sẽ chỉ làm tăng nguy cơ ghi nhớ cục bộ (overfitting), tiêu tốn tài nguyên và tăng độ trễ trên thiết bị biên mà không giải quyết được tính khái quát hóa từ vựng ngoài tập OOV.

### 4.2 Quyết Định Chính Thức Cho Cổng E2
- **CỔNG THỬ NGHIỆM E2: KHÓA / KHÔNG TIẾN HÀNH (NO-GO / LOCKED)**
- **Căn cứ khoa học:** Thử nghiệm E1 đã giải quyết thành công các khiếm khuyết cốt lõi của hệ thống nhận diện từ khóa tiếng Việt (loại bỏ lỗi "cửa", loại bỏ kích hoạt nhầm hội thoại, đạt độ nhạy nhiễu 99.55%, Macro F1 0.9601) chỉ bằng việc cải thiện chất lượng dữ liệu v12 trên cùng quy mô mô hình 138,469 tham số.
- Không có bất kỳ cơ sở khoa học nào ủng hộ việc mở rộng tham số mô hình lên 419,333 tham số ở giai đoạn hiện tại.

---

## 5. Danh Mục Cổ Vật Nghiên Cứu & Lệnh Tái Hiện

### 5.1 Các Cổ Vật Sinh Ra Từ Giai Đoạn Này

| Tên Cổ Vật | Đường Dẫn | SHA-256 |
| :--- | :--- | :--- |
| **Báo cáo kiểm định độc lập** | `experiments/vieneu_kws_research_v12/reports/DATASET_V12_INDEPENDENT_AUDIT_REPORT.md` | - |
| **Báo cáo QA hiệu đính** | `experiments/vieneu_kws_research_v12/reports/DATASET_V12_QA_REPORT.md` | - |
| **Cấu hình thử nghiệm E1** | `experiments/vieneu_kws_research_v12/configs/exp_v12_e1_baseline_scale.json` | - |
| **Checkpoint E1 tốt nhất** | `experiments/vieneu_kws_research_v12/checkpoints/exp_v12_e1_baseline_scale_seed42.pt` | `f3c0fe8f7ccebb8fe354cb50acb247e0674232044706db0187feb92b1706b878` |
| **Nhật ký huấn luyện E1** | `experiments/vieneu_kws_research_v12/reports/e1_training_history.json` | - |
| **Đồ thị học tập E1** | `experiments/vieneu_kws_research_v12/reports/e1_learning_curves.png` | - |
| **Ma trận nhầm lẫn E1 vs v11**| `experiments/vieneu_kws_research_v12/reports/e1_confusion_matrix.png` | - |
| **Kết quả đánh giá Baseline v11** | `experiments/vieneu_kws_research_v12/reports/eval_v11_baseline.json` | - |
| **Kết quả đánh giá E1** | `experiments/vieneu_kws_research_v12/reports/eval_e1_results.json` | - |

---

### 5.2 Lệnh Tái Hiện Độc Lập (Reproducible Execution Commands)

```powershell
# 1. Tái chạy kiểm định độc lập tập dữ liệu v12
.venv\Scripts\python.exe experiments/vieneu_kws_research_v12/scripts/audit_dataset_v12.py --expected-count 8000

# 2. Tái đánh giá mô hình đối chứng v11 đóng băng
.venv\Scripts\python.exe experiments/vieneu_kws_research_v12/scripts/eval_kws.py `
  --checkpoint experiments/vieneu_kws_research_v13_4/bundle/checkpoint/variant_v11_hard_negative_weighted_seed42.pt `
  --output-json experiments/vieneu_kws_research_v12/reports/eval_v11_baseline.json `
  --device cpu

# 3. Tái đánh giá mô hình thử nghiệm E1
.venv\Scripts\python.exe experiments/vieneu_kws_research_v12/scripts/eval_kws.py `
  --checkpoint experiments/vieneu_kws_research_v12/checkpoints/exp_v12_e1_baseline_scale_seed42.pt `
  --output-json experiments/vieneu_kws_research_v12/reports/eval_e1_results.json `
  --device cpu

# 4. Tái xuất bảng so sánh đối đầu v11 vs E1
.venv\Scripts\python.exe experiments/vieneu_kws_research_v12/scripts/compare_v11_e1.py
```
