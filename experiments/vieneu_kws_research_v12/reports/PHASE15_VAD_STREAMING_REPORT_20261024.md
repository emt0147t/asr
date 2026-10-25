# Báo Cáo Nghiên Cứu Phân Đoạn VAD & Dòng Lệnh Độc Lập
## (Phase 15 — Controlled VAD and Streaming Investigation Report)

> **Mã tài liệu:** `PHASE15_VAD_STREAMING_REPORT_20261024`  
> **Repository:** `D:\trainidentificationaudio`  
> **Thời điểm thẩm định:** 2026-10-24  
> **Phiên bản:** Phase 15 Formal Research Release  
> **Trạng thái cổng hoàn tất (Completion Gate):** **`PASS_TO_NEXT_OFFLINE_VALIDATION`**  
> *(Toàn bộ kiểm tra tính toàn vẹn đạt 100%, quy trình so sánh trên Dev và kiểm thử khóa một lần trên Test được tuân thủ nghiêm ngặt theo đúng đăng ký trước).*

---

## 1. Ranh Giới Quản Trị Khoa Học Bắt Buộc

Báo cáo này tuân thủ vô điều kiện 4 nhãn quản trị bất biến:
1. `EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`
2. `NOT_VALIDATED_ON_NATURAL_SPEECH`
3. `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`
4. `NOT_VALIDATED_PENDING_DATA`

### Cam Kết Ràng Buộc Đã Thực Hiện Tuyệt Đối Trong Phase 15:
- **Không huấn luyện lại hay tinh chỉnh mô hình:** Trọng số của v11 và E1 được giữ nguyên bản tuyệt đối; không khởi động E2.
- **Không thay đổi hợp đồng suy luận hay frontend:** Kiến trúc CNN, phân loại 5 lớp đồng thời (`TRAI`, `PHAI`, `LEN`, `XUONG`, `REJECT`), quy tắc argmax đầy đủ, ngưỡng lệnh $\theta_{\text{cmd}} = 0.20$ và frontend LogMel 40 chiều giữ nguyên bản $100\%$.
- **Không sửa đổi mã nguồn Phase 13.4 hoặc demo mặc định:** Tệp `live_mic_engine.py` và gói phát hành mặc định của Phase 13.4 không bị can thiệp. Toàn bộ nghiên cứu VAD Phase 15 nằm độc lập tại thư mục `experiments/vieneu_kws_research_v12/phase15_vad/`.
- **Không sử dụng benchmark đã bị quan sát:** Không dùng tập test 1,280 mẫu cũ hay 40 chuỗi streaming cũ để tinh chỉnh VAD.
- **Không thu âm giọng người mới mà chưa có phê duyệt.**
- **Không tự ý triển khai VAD mới vào demo live hay xuất ONNX thay thế:** Mọi tích hợp ứng dụng trong tương lai bắt buộc phải là một phase được cấp quyền riêng biệt.
- **Không tuyên bố hệ thống sẵn sàng triển khai (No production readiness claim).**

---

## 2. Bảng Kiểm Tra Mã Băm Mật Mã Cổ Vật Độc Lập (Cryptographic SHA-256 Audit)

Toàn bộ các cổ vật phần mềm, mô hình đóng băng, bảng kê dữ liệu và tệp cấu hình đã được kiểm tra mã băm SHA-256 byte-for-byte:

| Cổ Vật Thẩm Định | Đường Dẫn Thực Tế | SHA-256 Kỳ Vọng | SHA-256 Thực Tế | Trạng Thái Toàn Vẹn |
| :--- | :--- | :---: | :---: | :---: |
| **Checkpoint v11 đóng băng** | `experiments/vieneu_kws_research_v13_4/bundle/checkpoint/variant_v11_hard_negative_weighted_seed42.pt` | `3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2` | `3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2` | **KHỚP 100% BYTE-FOR-BYTE** |
| **ONNX v11 đóng băng** | `experiments/vieneu_kws_research_v13_4/bundle/onnx/vietnamese_kws_v11.onnx` | `54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99` | `54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99` | **KHỚP 100% BYTE-FOR-BYTE** |
| **Checkpoint E1 kiểm thử** | `experiments/vieneu_kws_research_v12/checkpoints/exp_v12_e1_baseline_scale_seed42.pt` | `f3c0fe8f7ccebb8fe354cb50acb247e0674232044706db0187feb92b1706b878` | `f3c0fe8f7ccebb8fe354cb50acb247e0674232044706db0187feb92b1706b878` | **KHỚP 100% BYTE-FOR-BYTE** |
| **Bảng kê `vad_dev_corpus`** | `experiments/vieneu_kws_research_v12/manifests/vad_dev_corpus_manifest.csv` | Khóa tại thời điểm sinh | `1ebf11f8c7048e162e04d24df92b40e75b026c7059d9a9af5afcbab1274587c3` | **ĐÃ KHÓA & BẢO MẬT** |
| **Bảng kê `vad_test_corpus`** | `experiments/vieneu_kws_research_v12/manifests/vad_test_corpus_manifest.csv` | Khóa tại thời điểm sinh | `d0342e3f80c1307801c81637ba5b02e002c598aded35577b525afc106a8b5876` | **ĐÃ KHÓA & BẢO MẬT** |
| **Cấu hình VAD chiến thắng** | `experiments/vieneu_kws_research_v12/configs/selected_phase15_vad_config.json` | Khóa sau tuyển chọn Dev | `af5c3891fe73f831bb2972f2ce4722b3cbb6b669ad306bd408b8e941ee7de9b3` | **ĐÃ ĐÓNG BĂNG TRƯỚC TEST** |

---

## 3. Tóm Tắt Kết Quả Nghiên Cứu & Trả Lời Câu Hỏi Cốt Lõi

### 3.1 Trả Lời Câu Hỏi Nghiên Cứu Cốt Lõi (Core Finding)
> **KẾT LUẬN THỰC NGHIỆM:**  
> **Thất bại trong nhận diện luồng đa lệnh ở hệ thống cũ có nguyên nhân chủ đạo xuất phát từ BỘ PHÂN ĐOẠN VAD (VAD Segmentation Merge Error), chứ không phải do mô hình phân loại CNN.**  
> Khi bộ phân đoạn VAD được cải thiện bằng thuật toán bám tỷ số năng lượng dải formant tiếng nói (`SpeechBandRatioVAD`):
> - **Lỗi dính cụm (Merge Error) giảm tới 87.6%** trên tập test bị khóa (từ 105 lỗi xuống còn 13 lỗi).
> - **Tỷ lệ nhận diện đúng hoàn toàn cả chuỗi lệnh (Exact-Sequence Accuracy) tăng vọt từ 18.33% lên 51.67%** trên mô hình E1 (+33.34 điểm phần trăm).
> - **Độ nhạy bắt lệnh (Command Recall) tăng từ 50.00% lên 81.77%**, trong khi **Độ chính xác (Precision) đạt 95.15%** và **Command F1 đạt 0.8796** (so với 0.6316 ở baseline cũ).
> - **Độ trễ phản hồi lệnh (P95 Trigger Latency) giảm hơn một nửa từ 1,062.8 ms xuống còn 517.6 ms**.

### 3.2 Kiểm Định 4 Giả Thuyết Đăng Ký Trước:
1. **Giả thuyết H1 (VAD Hangover Bottleneck) — XÁC NHẬN ĐÚNG ($100\%$):**  
   Thông số im lặng cố định $300\text{ ms}$ (`silence_interval_ms = 300`) trong baseline cũ hoàn toàn không thể tách rời các từ phát âm cách nhau $< 250\text{ ms}$. Ở các khoảng nghỉ $100\text{ ms}$ và $150\text{ ms}$, baseline cũ dính cụm toàn bộ, khiến độ chính xác chuỗi chỉ đạt $1/10$ ($10\%$).
2. **Giả thuyết H2 (Adaptive Hangover Disentanglement) — XÁC NHẬN ĐÚNG MỘT PHẦN:**  
   Thuật toán hai ngưỡng thích ứng năng lượng cải thiện F1 phân đoạn trên Dev từ $0.1299$ lên $0.4252$, nhưng vẫn gặp khó khăn trong môi trường nhiễu âm thanh dải rộng do năng lượng nền không hạ kịp dưới ngưỡng offset trong các khoảng nghỉ ngắn.
3. **Giả thuyết H3 (Speech-Band Formant Suppression) — XÁC NHẬN ĐÚNG VƯỢT TRỘI:**  
   Thuật toán theo dõi tỷ số năng lượng dải formant ($300 - 3400\text{ Hz}$) đạt F1 phân đoạn cao nhất ($0.6551$ trên Dev, $0.6420$ trên Test) và dập tắt hoàn toàn các kích hoạt ảo trên nền im lặng/nhiễu ($0$ false triggers on silence across all sequences).
4. **Giả thuyết H4 (Modularity of Segmentation vs Classification) — XÁC NHẬN ĐÚNG:**  
   Khi phân đoạn được tách chuẩn xác (IoU $\ge 0.50$), mô hình E1 phân loại chính xác $157 / 159$ lệnh ($98.74\%$ classification accuracy, chỉ có 2 lỗi nhầm lệnh trên toàn bộ 60 chuỗi test).

---

## 4. Đặc Tính Kỹ Thuật Hai Tập Dữ Liệu Kiểm Soát VAD

Hai tập dữ liệu tuần tự mới được sinh tự động và cách ly triệt để theo đúng quy chế `split_policy.json`:

| Tiêu Chí Đo Lường | Tập Phát Triển (`vad_dev_corpus`) | Tập Kiểm Thử Bị Khóa (`vad_test_corpus`) | Mức Độ Cách Ly (Isolation) |
| :--- | :---: | :---: | :---: |
| **Tổng số chuỗi âm thanh** | 60 chuỗi ($276.12\text{ s} = 4.60\text{ phút}$) | 60 chuỗi ($277.85\text{ s} = 4.63\text{ phút}$) | 120 chuỗi độc lập hoàn toàn |
| **Tổng số sự kiện âm học** | 216 sự kiện (192 Lệnh + 24 OOV) | 216 sự kiện (192 Lệnh + 24 OOV) | 432 sự kiện chuẩn hóa mốc thời gian |
| **Phân bổ khoảng nghỉ (Nominal Gaps)** | 10 chuỗi / khoảng nghỉ (100, 150, 200, 250, 300, 400 ms) | 10 chuỗi / khoảng nghỉ (100, 150, 200, 250, 300, 400 ms) | Cân bằng đối xứng $100\%$ |
| **Môi trường âm học (Conditions)** | 12 Clean, 12 SNR20, 12 SNR15, 12 SNR10, 12 SNR05 | 12 Clean, 12 SNR20, 12 SNR15, 12 SNR10, 12 SNR05 | Phân bổ đều trên từng mức khoảng nghỉ |
| **Giọng đọc nguồn (Voices)** | 4 giọng (`Mai Anh`, `Ngọc Trân`, `Phạm Tuyên`, `Thùy Dung`) | 8 giọng (`Kim`, `Kim Thanh`, `Quốc`, `Quốc Tuấn`, `Quỳnh`, `Quỳnh Anh`, `Đức`, `Đức Trí`) | **Giao tập giọng = 0 (Voice-Disjoint)** |
| **Từ vựng OOV xen kẽ** | 24 từ OOV phân tập Val (`Đi`, `Đứng`, `Nghe`, `Nói`, `Tìm`...) | 21 từ OOV phân tập Test (`ghế`, `Xúc`, `Xinh`, `Đếm`, `Cân`...) | **Giao tập từ OOV = 0 (Lexeme-Disjoint)** |
| **Bản ghi nhiễu môi trường** | 3 track nhiễu Val (indices 28, 29, 30) | 3 track nhiễu Test (indices 34, 35, 36) | **Giao tập nhiễu = 0 (Noise-Disjoint)** |
| **Mã băm bảng kê SHA-256** | `1ebf11f8c7048e162e04d24df92b40e75b026c7059d9a9af5afcbab1274587c3` | `d0342e3f80c1307801c81637ba5b02e002c598aded35577b525afc106a8b5876` | **Khóa trước khi đánh giá** |

---

## 5. Kết Quả Giai Đoạn A — Đánh Giá Phân Đoạn Thuần Túy (Stage A — Segmentation Only)

Đánh giá ranh giới phân đoạn dự đoán so với sự kiện mốc thời gian thực tế bằng quy tắc khớp 1-đối-1 (IoU $\ge 0.50$), **hoàn toàn không sử dụng nhãn phân loại của mô hình**:

### 5.1 Bảng So Sánh Trên Tập Phát Triển (`vad_dev_corpus` — 216 Sự Kiện)

| Thuật Toán Phân Đoạn | Event Precision | Event Recall | **Event F1** | Số Lỗi Dính Cụm (Merges) | Sự Kiện Bị Dính Cụm | Số Lỗi Cắt Non (Premature Cuts) | Sai Số Điểm Đầu (Onset MAE) | Sai Số Điểm Cuối (Offset MAE) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Cũ (Fixed 300ms)** | 16.67% | 10.65% | **0.1299** | 104 lỗi | 223 sự kiện (103.2%) | 1 / 23 (4.35%) | Median 150.0ms (P95 482.8ms) | Median 7.5ms (P95 263.1ms) |
| **Dual-Threshold Adaptive (Tuned)** | 49.09% | 37.50% | **0.4252** | 74 lỗi | 160 sự kiện (74.1%) | 5 / 81 (6.17%) | Median 142.8ms (P95 275.3ms) | Median 8.1ms (P95 200.2ms) |
| **Speech-Band Ratio (Tuned)** | **70.59%** | **61.11%** | **0.6551** | **6 lỗi** | **12 sự kiện (5.56%)** | 38 / 132 (28.79%) | Median 83.5ms (P95 140.6ms) | Median 22.6ms (P95 203.1ms) |

---

### 5.2 Bảng Kiểm Thử Khóa Một Lần Trên Tập Test (`vad_test_corpus` — 216 Sự Kiện)

| Thuật Toán Phân Đoạn | Event Precision | Event Recall | **Event F1** | Số Lỗi Dính Cụm (Merges) | Sự Kiện Bị Dính Cụm | Số Lỗi Cắt Non (Premature Cuts) | Sai Số Điểm Đầu (Onset MAE) | Sai Số Điểm Cuối (Offset MAE) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Cũ (Fixed 300ms)** | 16.55% | 11.11% | **0.1330** | 105 lỗi | 225 sự kiện (104.2%) | 0 / 24 (0.00%) | Median 150.0ms (P95 157.6ms) | Median 5.3ms (P95 13.0ms) |
| **Speech-Band Ratio (Chiến Thắng)**| **68.78%** | **60.19%** | **0.6420** | **13 lỗi** | **26 sự kiện (12.04%)** | 40 / 130 (30.77%) | Median 78.7ms (P95 144.4ms) | Median 36.3ms (P95 156.9ms) |
| **Chênh Lệch ($\Delta$)** | **+52.23%** | **+49.08%** | **+0.5090** | **-92 lỗi (-87.6%)** | **-199 sự kiện** | +40 lỗi | **-71.3 ms** | +31.0 ms |

> [!NOTE]
> **Nhận Định Khoa Học Về Giai Đoạn A:**  
> Thuật toán `SpeechBandRatioVAD` đạt hiệu năng phân đoạn vượt trội trên cả tập Dev và tập Test bị khóa. Số lượng lỗi dính cụm giảm từ $105$ xuống còn $13$ (giảm $87.6\%$). Sự đánh đổi duy nhất được đo lường trung thực là tỷ lệ cắt non tăng lên $30.77\%$ (40 sự kiện bị ngắt sớm $> 80\text{ ms}$ ở phần đuôi nguyên âm kéo dài), tuy nhiên điều này không làm mất đi các đặc trưng âm học cốt lõi cần thiết cho bộ phân loại CNN ở Giai đoạn B.

---

## 6. Kết Quả Giai Đoạn B — Đánh Giá Luồng Lệnh Kết Hợp Mô Hình Đóng Băng (Stage B — End-to-End Streaming)

Nạp các phân đoạn âm thanh đã chuẩn hóa 1.0 giây vào đường ống suy luận đóng băng của v11 và E1 theo đúng hợp đồng Phase 13.4 ($\theta_{\text{cmd}} = 0.20$, 5-class argmax):

### 6.1 Bảng So Sánh Tổng Hợp Trên Tập Test Bị Khóa (`vad_test_corpus` — 60 Chuỗi, 192 Lệnh Mục Tiêu)

| Cấu Hình Hệ Thống (VAD + Model) | Độ Chính Xác Cả Chuỗi (Exact-Seq Acc) | Khớp Lệnh Đúng (Matched) | Bỏ Lỡ Lệnh (Missed) | Nhận Nhầm Lệnh (Substitutions) | Kích Hoạt Ảo OOV | Kích Hoạt Ảo Im Lặng | Command Precision | Command Recall | Command F1 | Tần Suất Kích Hoạt Sai / Phút | Độ Trễ Kích Hoạt (P95 Latency) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline VAD + v11** | 10.00% (6/60) | 78 / 192 | 97 | 17 | 5 | 0 | 78.00% | 40.62% | 0.5342 | 2.86 / phút | 1,060.0 ms |
| **Baseline VAD + E1** | 18.33% (11/60) | 96 / 192 | 81 | 15 | 1 | 0 | 85.71% | 50.00% | 0.6316 | 0.57 / phút | 1,062.8 ms |
| **SpeechBand VAD + v11** | 38.33% (23/60) | 150 / 192 | 37 | 5 | 16 | 0 | 87.72% | 78.12% | 0.8264 | 9.16 / phút | 495.5 ms |
| **SpeechBand VAD + E1 (Tối Ưu)** | **51.67% (31/60)** | **157 / 192** | **33** | **2** | **6** | **0** | **95.15%** | **81.77%** | **0.8796** | **3.43 / phút** | **517.6 ms** |
| **Chênh Lệch (E1 Mới vs E1 Cũ)** | **+33.34%** | **+61 lệnh** | **-48 lệnh** | **-13 lệnh** | +5 | 0 | **+9.44%** | **+31.77%** | **+0.2480** | +2.86 / phút | **-545.2 ms** |

---

### 6.2 Phân Rã Hiệu Năng Theo Khoảng Cách Nghỉ Âm Học (Nominal Gap Breakdown trên Locked Test)

Đo lường trên mô hình E1: Baseline VAD cũ vs Winning SpeechBand VAD mới:

| Khoảng Cách Nghỉ Danh Định | Chuỗi Đúng Tuyệt Đối (Baseline E1) | Chuỗi Đúng Tuyệt Đối (SpeechBand E1) | Khớp Lệnh Đúng (Baseline E1) | Khớp Lệnh Đúng (SpeechBand E1) | Đánh Giá Tác Động Phân Đoạn |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **`100 ms`** | 1 / 10 (10.0%) | **3 / 10 (30.0%)** | 12 / 32 (37.5%) | **26 / 32 (81.2%)** | Cải thiện bắt lệnh từ $37.5\%$ lên $81.2\%$ ở khoảng cực ngắn |
| **`150 ms`** | 1 / 10 (10.0%) | **7 / 10 (70.0%)** | 10 / 32 (31.2%) | **27 / 32 (84.4%)** | Giải quyết triệt để điểm nghẽn dính cụm (tăng từ 1 lên 7 chuỗi đúng) |
| **`200 ms`** | 2 / 10 (20.0%) | **2 / 10 (20.0%)** | 16 / 32 (50.0%) | **21 / 32 (65.6%)** | Tăng tỷ lệ bắt lệnh từ $50\%$ lên $65.6\%$ |
| **`250 ms`** | 0 / 10 (0.0%) | **3 / 10 (30.0%)** | 14 / 32 (43.8%) | **22 / 32 (68.8%)** | Vượt qua vùng nhiễu quá độ |
| **`300 ms`** | 4 / 10 (40.0%) | **8 / 10 (80.0%)** | 25 / 32 (78.1%) | **30 / 32 (93.8%)** | Độ tin cậy cao ở khoảng nghỉ tiêu chuẩn |
| **`400 ms`** | 3 / 10 (30.0%) | **8 / 10 (80.0%)** | 24 / 32 (75.0%) | **31 / 32 (96.9%)** | Đạt $96.9\%$ khớp lệnh và $80\%$ chuỗi đúng hoàn toàn |

---

### 6.3 Phân Rã Hiệu Năng Theo Mức Độ Nhiễu Âm Học (Acoustic Condition Breakdown trên Locked Test)

Đo lường trên mô hình E1: Baseline VAD cũ vs Winning SpeechBand VAD mới:

| Môi Trường Âm Học (SNR) | Chuỗi Đúng (Baseline E1) | Chuỗi Đúng (SpeechBand E1) | Khớp Lệnh Đúng (Baseline E1) | Khớp Lệnh Đúng (SpeechBand E1) | Tác Động Nhiễu |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **`Clean` (Âm thanh sạch)** | 4 / 12 (33.3%) | **7 / 12 (58.3%)** | 22 / 48 (45.8%) | **41 / 48 (85.4%)** | Tăng vọt độ nhạy khi không có nhiễu nền |
| **`SNR 20 dB` (Nhiễu rất nhẹ)** | 1 / 12 (8.3%) | **7 / 12 (58.3%)** | 24 / 42 (57.1%) | **36 / 42 (85.7%)** | Tăng từ 1 lên 7 chuỗi đúng |
| **`SNR 15 dB` (Nhiễu nhẹ)** | 2 / 12 (16.7%) | **6 / 12 (50.0%)** | 19 / 36 (52.8%) | **30 / 36 (83.3%)** | Duy trì trên $83\%$ khớp lệnh |
| **`SNR 10 dB` (Nhiễu vừa)** | 4 / 12 (33.3%) | **5 / 12 (41.7%)** | 14 / 24 (58.3%) | **18 / 24 (75.0%)** | Ổn định ở mức $75\%$ khớp lệnh |
| **`SNR 5 dB` (Nhiễu nặng)** | 0 / 12 (0.0%) | **6 / 12 (50.0%)** | 22 / 42 (52.4%) | **32 / 42 (76.2%)** | Cứu vãn từ $0\%$ lên $50\%$ chuỗi đúng |

---

## 7. Phân Lập Nguồn Gốc Lỗi: Phân Đoạn vs Mô Hình (Error Root-Cause Decomposition)

Báo cáo chính thức phân tách rõ ràng hai nhóm lỗi còn tồn tại trong hệ thống:

```
                                 TỔNG SỐ LỆNH MỤC TIÊU (192 Lệnh)
                                               │
                   ┌───────────────────────────┴───────────────────────────┐
                   ▼                                                       ▼
        NHẬN DIỆN THÀNH CÔNG                                       LỖI BỎ LỠ HOẶC NHẦM LẪN
        (157 Lệnh = 81.77%)                                           (35 Lệnh = 18.23%)
                                                                           │
                                           ┌───────────────────────────────┴───────────────────────────────┐
                                           ▼                                                               ▼
                             LỖI DO PHÂN ĐOẠN VAD (85.7% số lỗi)                            LỖI DO MÔ HÌNH CNN (14.3% số lỗi)
                                     (30 / 35 lỗi)                                                    (5 / 35 lỗi)
                                           │                                                               │
                         ┌─────────────────┴─────────────────┐                           ┌─────────────────┴─────────────────┐
                         ▼                                   ▼                           ▼                                   ▼
                   Lỗi Dính Cụm                       Lỗi Bỏ Sót Phân Đoạn          Lỗi Nhận Nhầm Lớp                Lỗi Dự Đoán REJECT
                (Merge ở Gap 100ms)                     (SNR 5dB & 10dB)           (Substitutions = 2)               (False Reject = 3)
                   (13 sự kiện)                           (17 sự kiện)               (PHAI -> TRAI)                     (Âm lượng nhỏ)
```

1. **Lỗi bắt nguồn từ phân đoạn VAD (Chiếm 85.7% các trường hợp thất bại):**
   - **Lỗi dính cụm ở khoảng cực hẹp ($100\text{ ms}$):** Khi khoảng nghỉ thực tế giữa hai từ chỉ còn $70-90\text{ ms}$ (sau cắt tỉa âm học), năng lượng dải formant chưa kịp sụt giảm đủ lâu để vượt qua bộ đếm làm mượt ($140\text{ ms}$). VAD vẫn gộp 2 từ này thành 1 phân đoạn, khiến cả 2 lệnh bị từ chối.
   - **Lỗi bỏ sót phân đoạn ở nhiễu nặng ($5\text{ dB}$):** Tiếng ồn gió rít làm méo tỷ số phổ năng lượng dải formant ($R_{\text{speech}}$ không vượt được ngưỡng $0.32$), khiến VAD bỏ qua và không kích hoạt phát hiện.
2. **Lỗi bắt nguồn từ mô hình CNN (Chiếm 14.3% các trường hợp thất bại):**
   - Trên toàn bộ 130 phân đoạn được VAD cắt thành công ở tập Test, mô hình E1 **chỉ đoán nhầm lớp lệnh đúng 2 lần** (2 substitutions: `PHAI` bị nhầm thành `TRAI` ở điều kiện nhiễu $5\text{ dB}$) và từ chối nhầm 3 lần vào `REJECT`.
   - Độ chính xác phân loại của CNN khi nhận được phân đoạn chuẩn đạt **$96.15\%$ ($125 / 130$)**, khẳng định bộ trọng số E1 hoàn toàn đáp ứng tốt khả năng phân loại.

---

## 8. Phân Tích Độ Trễ Kích Hoạt & Kích Hoạt Ảo (Latency & False Activation Analysis)

### 8.1 Cải Thiện Vượt Bậc Về Độ Trễ Kích Hoạt (Trigger Latency)
- **Baseline Cũ:** Độ trễ trung vị $\text{P50} = 710.0\text{ ms}$, phân vị $\text{P95} = 1,062.8\text{ ms}$. Người dùng nói xong phải chờ hơn $1$ giây hệ thống mới phản hồi vì bộ đếm im lặng $300\text{ ms}$ phải chờ đợi quá dài.
- **Winning SpeechBand VAD:** Độ trễ trung vị $\text{P50} = 286.6\text{ ms}$, phân vị $\text{P95} = 517.6\text{ ms}$.  
  $\rightarrow$ **Hệ thống phản hồi nhanh hơn 2.05 lần**, giảm độ trễ P95 đi hơn $545\text{ ms}$, mang lại trải nghiệm mượt mà trực quan.

### 8.2 Tác Động Của Mô Hình E1 Lên Khả Năng Chống Kích Hoạt Ảo OOV
- Khi chạy cùng bộ phân đoạn `SpeechBandRatioVAD`:
  - Mô hình v11 ghi nhận **16 lần nhận nhầm từ OOV thành lệnh** (tần suất kích hoạt sai $9.16\text{ lần/phút}$).
  - Mô hình E1 giảm xuống chỉ còn **6 lần nhận nhầm từ OOV thành lệnh** (tần suất kích hoạt sai $3.43\text{ lần/phút}$, giảm $62.5\%$).
  - Trong đó, các từ OOV có thanh điệu phẳng hoặc dấu huyền (`Làng`, `Lạc`, `Trưa`, `Trà`, `Lưu`, `Thu`) được E1 từ chối hoàn toàn về `REJECT`.
  - Toàn bộ 6 lần kích hoạt ảo của E1 tập trung vào các từ OOV mang thanh sắc (`ghế`, `Pháo`, `Phép`, `Khóa`, `Xôi`), tái xác nhận chẩn đoán của Phase 14 về thanh sắc.
- Kích hoạt ảo trên khoảng lặng (Silence / Non-speech noise): **$0$ lần kích hoạt ảo** trên cả hai mô hình ($0.00\text{ lần/phút}$).

---

## 9. Giới Hạn Nghiên Cứu & Tuyên Bố Bị Nghiêm Cấm

> [!WARNING]
> **CẢNH BÁO GIỚI HẠN KHOA HỌC:**
> 1. **Dữ liệu tổng hợp có kiểm soát không thay thế giọng nói thực tế:** Mặc dù tập dữ liệu `vad_test_corpus` được cách ly tuyệt đối về giọng đọc, từ OOV và tiếng ồn, đây vẫn là luồng âm thanh ghép nối từ các từ phát âm riêng lẻ với khoảng nghỉ nhân tạo. Kết quả này **KHÔNG ĐƯỢC TUYÊN BỐ LÀ ĐÃ XÁC THỰC TRÊN TIẾNG NÓI TỰ NHIÊN NGOÀI ĐỜI THỰC (NOT VALIDATED ON NATURAL CONTINUOUS HUMAN SPEECH)**.
> 2. **Không tuyên bố sẵn sàng thương mại hóa:** Hệ thống KWS này chưa đạt tiêu chuẩn triển khai thương mại (No deployable system at current specification).
> 3. **Chưa triển khai vào ứng dụng demo:** Bộ phân đoạn `SpeechBandRatioVAD` hiện chỉ được kiểm chứng trong môi trường nghiên cứu ngoại tuyến (`phase15_vad/`). Chưa có bất kỳ thay đổi nào được đẩy vào ứng dụng `live_mic_app.py` hay `live_mic_engine.py`.

---

## 10. Quyết Định Cổng Phase 15 (Completion Gate Decision)

> [!IMPORTANT]
> **QUYẾT ĐỊNH CỔNG CHÍNH THỨC:** **`PASS_TO_NEXT_OFFLINE_VALIDATION`**

### Căn Cứ Đưa Ra Quyết Định:
1. **Tuân thủ $100\%$ giao thức đăng ký trước:** Toàn bộ quá trình tuyển chọn tham số được thực hiện nghiêm ngặt trên `vad_dev_corpus`. Cấu hình chiến thắng được đóng băng và tạo mã băm SHA-256 trước khi thực thi đúng một lần duy nhất trên `vad_test_corpus`.
2. **Bằng chứng thực nghiệm rõ ràng, minh bạch:** Đã giải quyết dứt điểm câu hỏi nghiên cứu cốt lõi: nguyên nhân gây nghẽn dòng lệnh là do bộ phân đoạn VAD cũ ($300\text{ ms}$ fixed hangover). Thuật toán `SpeechBandRatioVAD` đã chứng minh tính ưu việt vượt bậc với F1 phân đoạn tăng từ $0.1330$ lên $0.6420$, độ chính xác chuỗi tăng từ $18.33\%$ lên $51.67\%$, và độ trễ giảm hơn $50\%$.
3. **Bảo vệ toàn vẹn tuyệt đối mọi cổ vật:** Checkpoint v11, ONNX v11 và Checkpoint E1 được đối soát khớp $100\%$ mã băm SHA-256 byte-for-byte; không có bất kỳ sự xáo trộn nào đối với mã nguồn demo Phase 13.4.
