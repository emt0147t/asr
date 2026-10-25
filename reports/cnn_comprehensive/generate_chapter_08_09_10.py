import os

BASE_DIR = r"d:\trainidentificationaudio"
DOCS_DIR = os.path.join(BASE_DIR, "reports", "cnn_comprehensive")

ch08_content = """# Chương 08: Đánh giá Lắng nghe của Con người & Thẩm định (Human Listening & Validation)

## 1. Bản chất và Mục tiêu của Nghiên cứu Lắng nghe Mù Phase 90
Trong giai đoạn Phase 90, dự án đã thiết kế một thử nghiệm nghe thẩm định mù (blinded human listening study) nhằm xác minh tính tự nhiên và mức độ dễ nhận diện của các mẫu âm thanh do bộ tổng hợp Vieneu TTS sinh ra trước khi đưa vào đánh giá mô hình KWS.

### Thiết kế Thử nghiệm:
- **Quy mô tập mẫu:** 120 tệp âm thanh WAV được đánh mã định danh mù ngẫu nhiên (`clip_001.wav` đến `clip_120.wav`).
- **Cơ cấu nội dung:**
  - 40 clips chứa 4 khẩu lệnh hợp lệ (10 clips `TRAI`, 10 clips `PHAI`, 10 clips `LEN`, 10 clips `XUONG`).
  - 80 clips ngoài mục tiêu: 40 clips từ ngữ gây nhiễu âm vị học (phonetic confusions), 20 clips câu hội thoại tiếng Việt thông thường (general speech), và 20 clips tạp âm môi trường/nhiễu phòng (background noise).
- **Thành phần tham gia dự kiến:** 6 người đánh giá độc lập (`reviewer_01` đến `reviewer_06`).
- **Nhiệm vụ của người đánh giá:**
  1. Ghi lại từ ngữ nghe được (Transcript).
  2. Phân loại vào 1 trong các nhãn danh mục.
  3. Cho điểm độ rõ ràng (Intelligibility Score) theo thang điểm Likert 5 mức (từ 1: Hoàn toàn không hiểu, đến 5: Rất rõ ràng).

---

## 2. Thẩm Tra Thực Tế: Sự Thật Về Dữ Liệu Thu Thập Được
Khi tiến hành kiểm toán lại toàn bộ kho lưu trữ dữ liệu tại [`data/raw/field_recordings/phase90_listening/`](file:///d:/trainidentificationaudio/data/raw/field_recordings/phase90_listening/) và [`experiments/vieneu_kws_research_v9/human_review/`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v9/human_review/), kết quả thực chứng ghi nhận:

```mermaid
flowchart TD
    Plan["Kế hoạch Phase 90: 6 Người Đánh giá"] --> R1["Reviewer 01: Hoàn thành 120/120 clips<br/>Điểm rõ ràng TB: 4.88 / 5.0"]
    Plan --> R2["Reviewer 02: Không nộp bài (0 clips)"]
    Plan --> R3["Reviewer 03: Không nộp bài (0 clips)"]
    Plan --> R4["Reviewer 04: Không nộp bài (0 clips)"]
    Plan --> R5["Reviewer 05: Không nộp bài (0 clips)"]
    Plan --> R6["Reviewer 06: Không nộp bài (0 clips)"]
    
    R1 --> Summary["File tổng hợp validated_responses.csv: RỖNG (0 bytes)"]
    R2 & R3 & R4 & R5 & R6 --> Summary
```

### Chi tiết Hồ sơ Thực chứng:
1. **Duy nhất 01 Người Đánh giá Hoàn thành:** Chỉ có tệp [`reviewer_01_responses.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v9/human_review/reviewer_01_responses.csv) chứa đủ 120 dòng phản hồi hợp lệ.
   - Điểm đánh giá độ rõ ràng trung bình: $4.88 / 5.0$.
   - Tỷ lệ nhận diện đúng từ khóa của con người trên 40 clips lệnh: $100.00\\%$ (40/40 clips).
2. **5 Người Đánh giá Bỏ cuộc:** Các tệp phản hồi của `reviewer_02` đến `reviewer_06` hoàn toàn không tồn tại trong kho mã nguồn.
3. **Tệp Tổng hợp Bị Trống:** Tệp [`validated_responses.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v9/human_review/validated_responses.csv) có dung lượng 0 bytes hoặc chỉ có tiêu đề rỗng, do quy trình tổng hợp tự động bị dừng lại khi thiếu dữ liệu của 5 người còn lại.
4. **Không Thể Tính Hệ Số Đồng Thuận Liên Đánh Giá (Inter-Rater Reliability):** Do chỉ có một người đánh giá duy nhất ($N=1$), các chỉ số thống kê như Fleiss' Kappa hay Krippendorff's Alpha toán học không thể xác định được.

---

## 3. Hiệu Năng Mô Hình Baseline `variant_w10` Trên Bộ 120 Clips Phase 90
Mô hình được thử nghiệm đối chứng trong Phase 90 là baseline `variant_w10_seed42.pt` tại ngưỡng $\\theta_{\\text{cmd}} = 0.40$:
- **Nhận diện Khẩu lệnh (Target Commands):** 39/40 clips chính xác ($97.50\\%$ Recall). Bỏ sót 01 clip lệnh `TRAI` (dự đoán xác suất dưới ngưỡng).
- **Từ chối Ngoại lai (Non-Target Rejection):** 71/80 clips bị từ chối chính xác. Tuy nhiên, có tới **9/80 clips bị kích hoạt nhầm** (False Alarm Rate = $11.25\\%$).
- **Phân tích Báo động giả:** Toàn bộ 9 trường hợp báo động giả đều rơi vào nhóm các từ ngữ gây nhiễu âm vị học (ví dụ: *"tai"*, *"lái"* bị nhầm thành `TRAI`; *"phấn"* bị nhầm thành `PHAI`). Đây chính là động lực trực tiếp dẫn đến chiến dịch hoàn thiện CNN Phase 11.

---

## 4. Kết Luận Pháp Lý & Quản Trị Nghiên Cứu (Strict Governance Verdict)
Tuân thủ nguyên tắc liêm chính học thuật và quản trị dữ liệu nghiêm ngặt của dự án:
1. **Tuyệt đối không suy diễn dữ liệu:** Không được phép tự động điền (impute) dữ liệu đánh giá của 5 người còn thiếu hoặc tạo lập các bảng điểm giả định.
2. **Nhãn Trạng thái Chính thức:** Toàn bộ hệ thống duy trì bắt buộc nhãn:
   $$\\mathbf{NOT\\_VALIDATED\\_ON\\_NATURAL\\_SPEECH}$$
3. Thử nghiệm Phase 90 chỉ chứng minh được rằng âm thanh do Vieneu TTS sinh ra có độ trong trẻo và dễ hiểu cao đối với tai người nghe ($4.88/5.0$), nhưng **hoàn toàn chưa chứng minh được** mô hình KWS có khả năng hoạt động tốt trên giọng nói người thật tự nhiên trong đời sống hàng ngày.
"""

ch09_content = """# Chương 09: Đánh giá Xác nhận Độc lập Phase 13.3 (Confirmatory Evaluation Phase 13.3)

## 1. Tổng quan Giao thức Đăng ký Trước (Preregistered Protocol)
Sau khi khắc phục triệt để lỗi khuếch đại tín hiệu im lặng tại Phase 13.1 và giải quyết bất đồng tham số Frontend tại Phase 13.2.1, giai đoạn [Phase 13.3](file:///d:/trainidentificationaudio/reports/PHASE13_3_CONFIRMATORY_EVALUATION.md) đã thực thi quy trình **Đánh giá Xác nhận Độc lập 1 Lần Duy Nhất (Preregistered One-Shot Confirmatory Evaluation)**.

Quy trình tuân thủ nghiêm ngặt Giao thức sửa đổi [Amendment 13.1-A](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_2/protocol/AMENDED_CONFIRMATORY_PROTOCOL_V13_2.md):
- **Checkpoint Đánh giá:** [`experiments/vieneu_kws_research_v13_1/bundle/checkpoint/variant_v11_hard_negative_weighted_seed42.pt`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_1/bundle/checkpoint/variant_v11_hard_negative_weighted_seed42.pt)
- **Mã băm SHA-256 Xác thực:** `3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2`
- **Ngưỡng Quyết định Bất biến:** $\\theta_{\\text{cmd}} = 0.20$
- **Quy mô Tập dữ liệu:** 1.000 clips âm thanh tổng hợp hoàn toàn mới, được sinh ra từ **8 giọng đọc Vieneu phương ngữ Miền Bắc** chưa từng xuất hiện trong bất kỳ tập huấn luyện hay kiểm thử nào trước đây.

---

## 2. Kết Quả Định Lượng Toàn Diện 8 Tiêu Chí Nghiệm Thu
Cuộc đánh giá đã kiểm tra mô hình qua 8 cổng nghiệm thu kỹ thuật (Acceptance Gates). Dưới đây là bảng đối soát chi tiết:

| Tiêu chí Nghiệm thu (Gate) | Yêu cầu Kỹ thuật Đăng ký trước | Kết quả Thực chứng Đạt được | Trạng thái (Verdict) |
| :--- | :--- | :--- | :--- |
| **Gate 1: Command Recall Tổng** | $\\ge 95.00\\%$ | **$99.00\\%$** (198/200 clips) | **ĐẠT (PASSED)** |
| **Gate 2: Recall Từng Lớp Độc lập** | $\\ge 90.00\\%$ cho mỗi lệnh | `TRAI`: $98.0\\%$ (49/50)<br/>`PHAI`: $100.0\\%$ (50/50)<br/>`LEN`: $98.0\\%$ (49/50)<br/>`XUONG`: $100.0\\%$ (50/50) | **ĐẠT (PASSED)** |
| **Gate 3: Tỷ lệ Báo động Giả (FAR)** | $\\le 2.00\\%$ trên 800 non-targets | **$0.75\\%$** (6/800 clips) | **ĐẠT (PASSED)** |
| **Gate 4: Độ chính xác Toàn diện** | $\\ge 95.00\\%$ trên 1.000 clips | **$99.20\\%$** (992/1.000 clips) | **ĐẠT (PASSED)** |
| **Gate 5: Lẫn lộn giữa các Lệnh** | $\\le 1.00\\%$ (Command Confusion) | **$0.00\\%$** (0/200 clips) | **ĐẠT (PASSED)** |
| **Gate 6: Độ trễ Suy luận Trung bình** | Mean Latency $\\le 20.0\\text{ ms}$ | **$1.80\\text{ ms}$** (trên CPU Intel x86) | **ĐẠT (PASSED)** |
| **Gate 7: Phân vị Độ trễ P95** | P95 Latency $\\le 50.0\\text{ ms}$ | **$2.12\\text{ ms}$** | **ĐẠT (PASSED)** |
| **Gate 8: Kiểm soát Hạt & Băm File** | SHA-256 khớp tuyệt đối | Khớp $100\\%$ mã băm gốc | **ĐẠT (PASSED)** |

---

## 3. Ma Trận Nhầm Lẫn Chi Tiết (Confusion Matrix)
Ma trận phân loại trên toàn bộ 1.000 clips kiểm thử độc lập được thể hiện như sau:

```
                  DỰ ĐOÁN (PREDICTED)
THỰC TẾ      TRAI    PHAI     LEN   XUONG  REJECT   TỔNG
TRAI           49       0       0       0       1     50
PHAI            0      50       0       0       0     50
LEN             0       0      49       0       1     50
XUONG           0       0       0      50       0     50
NON-TARGET      1       3       0       2     794    800
--------------------------------------------------------
TỔNG           50      53      49      52     796   1000
```

### Phân rã 800 Mẫu Ngoại lai (Non-Target Categories):
1. **Tạp âm Môi trường Thuần túy (Pure Noise - 200 clips):** $0 / 200$ báo động giả ($0.00\\%$ FAR). Lớp từ chối hấp thụ hoàn hảo.
2. **Tiếng nói Thông thường Ngoài Danh mục (General Speech - 200 clips):** $1 / 200$ báo động giả ($0.50\\%$ FAR) vào nhãn `PHAI`.
3. **Từ ngữ Gây nhiễu Âm vị Học (Hard Negatives - 200 clips):** $2 / 200$ báo động giả ($1.00\\%$ FAR): 1 clip nhầm `TRAI`, 1 clip nhầm `XUONG`.
4. **Câu Hội thoại Chứa Từ Gây nhiễu (Carrier Phrases - 200 clips):** $3 / 200$ báo động giả ($1.50\\%$ FAR): 2 clips nhầm `PHAI`, 1 clip nhầm `XUONG`.

---

## 4. Ranh Giới và Phạm Vi Giá Trị Thống Kê
Mặc dù vượt qua xuất sắc cả 8 tiêu chí nghiệm thu, kết quả của Phase 13.3 bắt buộc phải được hiểu trong giới hạn đã đăng ký trước:
1. **Phạm vi Phương ngữ:** Kết quả chỉ có giá trị xác nhận khả năng tổng quát hóa trên **8 giọng đọc tổng hợp mới thuộc phương ngữ Miền Bắc**.
2. **Giọng Miền Trung & Miền Nam:** Hoàn toàn chưa được xác nhận độc lập do không có giọng đọc tổng hợp đạt chuẩn và chưa có dữ liệu thu âm người thật có bản quyền.
3. **Tuyên bố Hợp lệ:** Checkpoint `variant_v11_hard_negative_weighted_seed42.pt` được xác nhận là một **Ứng viên Mô hình Bounded-Utterance Xuất sắc cho Giọng Bắc**, sẵn sàng bước vào giai đoạn đóng gói phát hành (Phase 13.4).
"""

ch10_content = """# Chương 10: Phả hệ Checkpoint & Đóng gói Phát hành (Checkpoint Lineage & Release Artifacts)

## 1. Biểu Đồ Phả Hệ Checkpoint Cốt Lõi (Lineage Tree)
Toàn bộ lịch sử tiến hóa của các trọng số mô hình từ giai đoạn đầu đến bản phát hành cuối cùng được phác họa trong biểu đồ phả hệ dưới đây:

```mermaid
flowchart TD
    M_P27["Phase 27: baseline_cnn_4class.pt<br/>(4 classes, không reject)"] --> M_P35["Phase 35: baseline_cnn_5class.pt<br/>(5 classes, thêm noise)"]
    M_P35 --> M_P40["Phase 40: tcresnet_6class.pt<br/>(6 classes, speech vs noise)"]
    M_P40 --> M_P65["Phase 65: corrected_kws_net.pt<br/>(TC-ResNet cải tiến)"]
    M_P65 --> M_W10["Phase 85: variant_w10_seed42.pt<br/>(Frozen Baseline, N_fft=400, theta=0.40)<br/>SHA-256: 602b9f35..."]
    
    M_W10 -.->|"Đánh giá mù 120 clips"| Eval_P90["Phase 90 Human Review<br/>(FAR = 11.25%, 9/80 lỗi)"]
    M_W10 -.->|"Thử nghiệm di động"| Dep_P91["Phase 91-92 Mobile Scaffolds"]
    
    M_W10 -->|"Nâng cấp N_fft=512 + Hard-Negative Loss"| M_V11["Phase 11: variant_v11_weighted_seed42.pt<br/>(Champion Candidate, theta=0.20)<br/>SHA-256: 3c4369d2..."]
    
    M_V11 -->|"Đóng gói bundle v12"| B_V12["Phase 12 Bundle<br/>(Phát hiện lỗi khuếch đại infer.py)"]
    B_V12 -->|"Khắc phục tiền xử lý"| B_V13_1["Phase 13.1 Bundle<br/>(Conditional Clamping)"]
    B_V13_1 -->|"Xác nhận độc lập 1.000 clips"| Conf_P13_3["Phase 13.3 Confirmatory<br/>(Pass 8/8 Gates, Accuracy 99.2%)"]
    Conf_P13_3 -->|"Đóng gói phát hành chính thức"| B_V13_4["Phase 13.4 Final Release Bundle<br/>(FROZEN_CNN_RELEASE_CANDIDATE)"]
```

---

## 2. Thẩm Tra Mã Băm SHA-256 Tuyệt Đối (Cryptographic Integrity)
Bảng dưới đây liệt kê các checkpoint bản lề trong lịch sử nghiên cứu cùng mã băm mã hóa xác thực (trích xuất từ [`tables/checkpoint_lineage.csv`](file:///d:/trainidentificationaudio/reports/cnn_comprehensive/tables/checkpoint_lineage.csv)):

| Tên File Checkpoint | Giai đoạn Phát hành | Kích thước (Bytes) | Mã Băm SHA-256 Chính thức | Vai trò Kỹ thuật |
| :--- | :--- | :--- | :--- | :--- |
| `variant_w10_seed42.pt` | Phase 85 / Phase 9 / 10 | 479.524 | `602b9f359147d337f7422f25608cb926ec03794d0705f1eb02c9a927a44f9547` | Frozen Baseline lịch sử. |
| `variant_v11_hard_negative_weighted_seed42.pt` | Phase 11 Checkpoints | 479.524 | `3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2` | Trọng số gốc từ huấn luyện Phase 11. |
| `variant_v11_hard_negative_weighted_seed42.pt` | Phase 12 Bundle | 479.524 | `3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2` | Bản sao nguyên vẹn trong bundle v12. |
| `variant_v11_hard_negative_weighted_seed42.pt` | Phase 13.1 Bundle | 479.524 | `3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2` | Bản sao nguyên vẹn dùng cho khắc phục 13.1. |
| `variant_v11_hard_negative_weighted_seed42.pt` | **Phase 13.4 Final Release** | 479.524 | `3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2` | **Bản phát hành ứng viên đông kết cuối cùng.** |
| `vietnamese_kws_v11.onnx` | **Phase 13.4 Final Release** | 485.498 | `54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99` | **Mô hình ONNX Runtime chuẩn hóa.** |

> **Bằng chứng Thực chứng:** Mã băm SHA-256 của tệp trọng số `variant_v11_hard_negative_weighted_seed42.pt` hoàn toàn trùng khớp từng byte xuyên suốt từ Phase 11, Phase 12, Phase 13.1, Phase 13.3 đến Phase 13.4. Không có bất kỳ hiện tượng huấn luyện lại (retraining), tinh chỉnh lén lút (fine-tuning) hay biến đổi trọng số nào xảy ra.

---

## 3. Cấu Trúc Đóng Gói Bundle Phát Hành Phase 13.4 (Release Bundle Inventory)
Thư mục phát hành chính thức được đặt cô lập tại [`experiments/vieneu_kws_research_v13_4/bundle/`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/bundle/) với đầy đủ các thành phần phục vụ triển khai:

```
experiments/vieneu_kws_research_v13_4/bundle/
├── checkpoint/
│   └── variant_v11_hard_negative_weighted_seed42.pt   # File trọng số PyTorch gốc
├── onnx/
│   └── vietnamese_kws_v11.onnx                        # Mô hình ONNX đã kiểm định đẳng cấu
├── src/
│   ├── model.py                                       # Định nghĩa mạng CorrectedVietnameseKWSNet
│   ├── frontend.py                                    # Trích xuất Mel Spectrogram chuẩn N_fft=512
│   └── infer.py                                       # Mã suy luận hoàn thiện với Conditional Clamping
├── config/
│   ├── model_config.json                              # Tham số cấu trúc mạng và ánh xạ 5 nhãn
│   └── preprocessing_contract.json                    # Hợp đồng tiền xử lý âm thanh chính thức
├── RELEASE_METADATA.json                              # Siêu dữ liệu phát hành, mã băm, kết quả 13.3
└── checksums.sha256                                   # Bản kiểm toán mã băm SHA-256 toàn bộ bundle
```

### Lệnh Xác Minh Mã Băm Trên Môi Trường Thực Tế:
- **Trên Windows PowerShell:**
  ```powershell
  Get-FileHash -Algorithm SHA256 experiments/vieneu_kws_research_v13_4/bundle/checkpoint/variant_v11_hard_negative_weighted_seed42.pt
  ```
- **Trên Linux / macOS Bash:**
  ```bash
  sha256sum experiments/vieneu_kws_research_v13_4/bundle/checkpoint/variant_v11_hard_negative_weighted_seed42.pt
  ```
Kết quả trả về bắt buộc phải là: `3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2`.
"""

with open(os.path.join(DOCS_DIR, "08_HUMAN_LISTENING_AND_VALIDATION.md"), "w", encoding="utf-8") as f:
    f.write(ch08_content.strip() + "\n")

with open(os.path.join(DOCS_DIR, "09_CONFIRMATORY_EVALUATION_V13_3.md"), "w", encoding="utf-8") as f:
    f.write(ch09_content.strip() + "\n")

with open(os.path.join(DOCS_DIR, "10_CHECKPOINT_LINEAGE_AND_RELEASE_ARTIFACTS.md"), "w", encoding="utf-8") as f:
    f.write(ch10_content.strip() + "\n")

print("[OK] Generated 08_HUMAN_LISTENING_AND_VALIDATION.md, 09_CONFIRMATORY_EVALUATION_V13_3.md, and 10_CHECKPOINT_LINEAGE_AND_RELEASE_ARTIFACTS.md")
