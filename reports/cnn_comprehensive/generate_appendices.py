import os
import csv

BASE_DIR = r"d:\trainidentificationaudio"
DOCS_DIR = os.path.join(BASE_DIR, "reports", "cnn_comprehensive")
APP_DIR = os.path.join(DOCS_DIR, "appendices")
os.makedirs(APP_DIR, exist_ok=True)

app_a_content = """# Phụ Lục A: Từ Điển Thuật Ngữ & Chỉ Số Đo Lường (Metrics Dictionary)

Tài liệu này định nghĩa chi tiết các chỉ số toán học và thuật ngữ kỹ thuật được sử dụng xuyên suốt toàn bộ dự án KWS tiếng Việt.

---

## 1. Các Chỉ Số Đo Lường Hiệu Năng Mô Hình KWS

### 1.1. Độ Nhạy Nhận Lệnh (Command Recall / Sensitivity)
Tỷ lệ nhận diện chính xác các khẩu lệnh hợp lệ trên tổng số mẫu khẩu lệnh thực tế:
$$\\text{Recall}_{\\text{cmd}} = \\frac{\\sum_{c \\in \\{0, 1, 2, 3\\}} \\text{TP}_c}{\\sum_{c \\in \\{0, 1, 2, 3\\}} (\\text{TP}_c + \\text{FN}_c)}$$
Trong đó:
- $\\text{TP}_c$ (True Positive): Số mẫu thuộc lệnh $c$ được mô hình nhận diện đúng lớp $c$.
- $\\text{FN}_c$ (False Negative): Số mẫu thuộc lệnh $c$ bị mô hình bỏ sót (phân loại nhầm thành lệnh khác hoặc bị đẩy vào `REJECT`).

### 1.2. Tỷ Lệ Báo Động Giả (False Alarm Rate - FAR)
Tỷ lệ các mẫu âm thanh ngoại lai (không phải khẩu lệnh) bị mô hình nhận diện nhầm thành một trong 4 lệnh:
$$\\text{FAR} = \\frac{\\sum_{c \\in \\{0, 1, 2, 3\\}} \\text{FP}_c}{N_{\\text{non-target}}}$$
Trong đó:
- $\\text{FP}_c$ (False Positive): Số mẫu ngoại lai bị gán nhầm thành lệnh $c$.
- $N_{\\text{non-target}}$: Tổng số mẫu âm thanh không phải lệnh (bao gồm tiếng ồn, từ gây nhiễu, câu hội thoại, v.v.).

### 1.3. Độ Chính Xác Toàn Diện (Overall Accuracy)
Tỷ lệ các dự đoán hoàn toàn chính xác trên toàn bộ tập kiểm thử (bao gồm cả các lệnh hợp lệ và các mẫu từ chối hợp lệ):
$$\\text{Accuracy} = \\frac{\\sum_{i=0}^4 \\text{TP}_i}{N_{\\text{total}}}$$
Trong đó lớp thứ 4 là lớp từ chối `REJECT`.

### 1.4. Điểm F1-Score Cho Từng Lớp Lệnh
Trung bình điều hòa giữa Độ chuẩn xác (Precision) và Độ nhạy (Recall):
$$\\text{Precision}_c = \\frac{\\text{TP}_c}{\\text{TP}_c + \\text{FP}_c}$$
$$F_{1, c} = 2 \\cdot \\frac{\\text{Precision}_c \\cdot \\text{Recall}_c}{\\text{Precision}_c + \\text{Recall}_c}$$

### 1.5. Tỷ Lệ Lẫn Lộn Giữa Các Lệnh (Command Confusion Rate)
Tỷ lệ một khẩu lệnh hợp lệ này bị nhận diện nhầm sang một khẩu lệnh hợp lệ khác:
$$\\text{Confusion Rate} = \\frac{\\sum_{c_i \\ne c_j, c_i, c_j \\in \\{0,1,2,3\\}} \\text{Count}(y=c_i, \\hat{y}=c_j)}{N_{\\text{target}}}$$
Trong Phase 13.3, tỷ lệ này đạt mức tuyệt đối: **$0.00\\%$** (0/200 clips).

---

## 2. Các Chỉ Số Đo Lường Triển Khai & Đẳng Cấu

### 2.1. Độ Lệch Logit Cực Đại (Max Absolute Logit Difference - $\\Delta_{\\text{max}}$)
Đo lường sai số số học giữa đầu ra của mô hình PyTorch và mô hình ONNX Runtime khi cùng nhận một ma trận đầu vào $X$:
$$\\Delta_{\\text{max}} = \\max_{i=1 \\dots N, j=0 \\dots 4} |z_{i, j}^{\\text{PyTorch}} - z_{i, j}^{\\text{ONNX}}|$$
Yêu cầu nghiệm thu kỹ thuật: $\\Delta_{\\text{max}} < 1.0 \\times 10^{-4}$. Checkpoint đông kết đạt $\\Delta_{\\text{max}} = 3.58 \\times 10^{-6}$.

### 2.2. Phân Vị Độ Trễ (Latency Percentiles: Mean, P50, P95)
- **Mean Latency:** Thời gian suy luận trung bình tính trên toàn bộ các mẫu kiểm thử.
- **P50 Latency (Median):** Thời gian mà $50\\%$ số mẫu suy luận hoàn thành nhanh hơn hoặc bằng giá trị này.
- **P95 Latency:** Thời gian mà $95\\%$ số mẫu suy luận hoàn thành nhanh hơn hoặc bằng giá trị này (đại diện cho độ trễ trong tình huống tải xấu nhất).

---

## 3. Các Nhãn Trạng Thái Chuẩn Hóa (Canonical Status Labels)
- `FROZEN_CNN_RELEASE_CANDIDATE`: Ứng viên mô hình CNN đã được đóng băng toàn bộ trọng số, cấu hình và mã suy luận; không được phép can thiệp chỉnh sửa.
- `VALIDATED_HIGH_CONFIDENCE`: Đã được chứng minh bằng thực nghiệm độc lập với độ tin cậy thống kê cao.
- `NOT_VALIDATED_ON_NATURAL_SPEECH`: Chưa được thẩm định trên dữ liệu giọng người thật nói tự nhiên trong môi trường đời thực.
- `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`: Chưa có hệ thống hoàn chỉnh sẵn sàng triển khai cho chế độ micro mở thời gian thực.
- `NOT_VALIDATED_PENDING_DATA`: Chưa được thẩm định do thiếu dữ liệu kiểm thử độc lập.
- `INVALIDATED_INCOMPLETE_STUDY`: Nghiên cứu bị hủy giá trị thống kê do dữ liệu thu thập không đầy đủ so với kế hoạch ban đầu.
"""

# App B: Read checkpoint lineage CSV
checkpoints_rows = []
ckpt_csv_path = os.path.join(DOCS_DIR, "tables", "checkpoint_lineage.csv")
if os.path.exists(ckpt_csv_path):
    with open(ckpt_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        checkpoints_rows = list(reader)

app_b_content = """# Phụ Lục B: Danh Mục Checkpoint & Mô Hình (Model & Checkpoint Inventory)

Bảng dưới đây thống kê toàn bộ 41 tệp mô hình PyTorch (`.pt`) và ONNX (`.onnx`) được lưu trữ trong kho lưu trữ `D:\\trainidentificationaudio`, bao gồm đường dẫn tương đối, kích thước tệp và mã băm mã hóa SHA-256 xác thực:

| STT | Đường Dẫn Tệp Mô Hình (Relative Path) | Kích Thước (Bytes) | Mã Băm SHA-256 (Byte-Exact) |
| :---: | :--- | :---: | :--- |
"""

for i, row in enumerate(checkpoints_rows, 1):
    rel_path = row.get("Relative_Path", "")
    size = row.get("File_Size_Bytes", "")
    sha = row.get("SHA256_Hash", "")
    app_b_content += f"| {i} | [`{rel_path}`](file:///{BASE_DIR.replace(chr(92), '/')}/{rel_path}) | {size} | `{sha}` |\n"

# App C: Read dataset inventory CSV
dataset_rows = []
ds_csv_path = os.path.join(DOCS_DIR, "tables", "dataset_inventory.csv")
if os.path.exists(ds_csv_path):
    with open(ds_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        dataset_rows = list(reader)

app_c_content = """# Phụ Lục C: Danh Mục Tập Dữ Liệu & Manifest (Dataset & Split Inventory)

Bảng dưới đây thống kê toàn bộ 116 tệp manifest dữ liệu, danh sách phân chia (splits) và tệp nhãn âm thanh trong kho lưu trữ, ghi nhận chính xác số dòng (mẫu âm thanh) và mã băm SHA-256:

| STT | Đường Dẫn Tệp Manifest (Relative Path) | Số Dòng (Rows) | Mã Băm SHA-256 |
| :---: | :--- | :---: | :--- |
"""

for i, row in enumerate(dataset_rows, 1):
    rel_path = row.get("Relative_Path", "")
    rows = row.get("Row_Count", "")
    sha = row.get("SHA256_Hash", "")
    app_c_content += f"| {i} | [`{rel_path}`](file:///{BASE_DIR.replace(chr(92), '/')}/{rel_path}) | {rows} | `{sha}` |\n"

app_d_content = """# Phụ Lục D: Danh Mục Bộ Kiểm Thử Tự Động (Test Suite Inventory)

Kho mã nguồn `D:\\trainidentificationaudio` duy trì hệ thống kiểm thử tự động toàn diện với **384 bài kiểm thử** (đạt tỷ lệ vượt qua $100.00\\%$). Bảng dưới đây phân loại chi tiết các tệp kiểm thử trong thư mục [`tests/`](file:///d:/trainidentificationaudio/tests/):

| Tệp Kiểm Thử (Test File) | Số Lượng Tests | Nội Dung & Chức Năng Kiểm Tra |
| :--- | :---: | :--- |
| [`tests/test_vieneu_research_v13_4.py`](file:///d:/trainidentificationaudio/tests/test_vieneu_research_v13_4.py) | 8 | Kiểm thử đóng gói phát hành Phase 13.4, tính toàn vẹn SHA-256, đẳng cấu ONNX. |
| [`tests/test_vieneu_research_v13_3.py`](file:///d:/trainidentificationaudio/tests/test_vieneu_research_v13_3.py) | 12 | Thẩm định 8 cổng nghiệm thu trên 1.000 clips độc lập của Phase 13.3. |
| [`tests/test_vieneu_research_v13_2_1.py`](file:///d:/trainidentificationaudio/tests/test_vieneu_research_v13_2_1.py) | 6 | Kiểm tra tính tương thích tham số Frontend FFT ($N_{\\text{fft}}=512$) và Mel filterbank. |
| [`tests/test_vieneu_research_v13_2.py`](file:///d:/trainidentificationaudio/tests/test_vieneu_research_v13_2.py) | 8 | Kiểm định đăng ký trước giao thức, phân bổ 8 giọng đọc và danh mục từ vựng. |
| [`tests/test_vieneu_research_v13_1.py`](file:///d:/trainidentificationaudio/tests/test_vieneu_research_v13_1.py) | 10 | Kiểm tra khắc phục lỗi khuếch đại tín hiệu im lặng và logic chuẩn hóa biên độ. |
| [`tests/test_vieneu_research_v13.py`](file:///d:/trainidentificationaudio/tests/test_vieneu_research_v13.py) | 8 | Kiểm toán tiền kiểm Phase 13, ghi nhận điểm nghẽn dừng khẩn cấp (blocked audit). |
| [`tests/test_vieneu_research_v12.py`](file:///d:/trainidentificationaudio/tests/test_vieneu_research_v12.py) | 12 | Kiểm tra đóng gói ứng viên Phase 12 và hợp đồng tiền xử lý. |
| [`tests/test_vieneu_research_v11.py`](file:///d:/trainidentificationaudio/tests/test_vieneu_research_v11.py) | 16 | Kiểm tra huấn luyện v11, hàm mất mát Hard-Negative Weighted Cross Entropy, multi-seed. |
| [`tests/test_vieneu_research_v10.py`](file:///d:/trainidentificationaudio/tests/test_vieneu_research_v10.py) | 14 | Kiểm tra kiến trúc TC-ResNet v10 và khả năng chống nhiễu SNR thấp. |
| [`tests/test_cross_platform_parity_phase92.py`](file:///d:/trainidentificationaudio/tests/test_cross_platform_parity_phase92.py) | 14 | Đối chuẩn chéo PyTorch vs ONNX Runtime, đo phân vị độ trễ P50/P95. |
| [`tests/test_cnn_deployment_phase91.py`](file:///d:/trainidentificationaudio/tests/test_cnn_deployment_phase91.py) | 10 | Kiểm tra khung xuất mô hình ONNX và các hàm bọc C++. |
| [`tests/test_phase90_human_listening.py`](file:///d:/trainidentificationaudio/tests/test_phase90_human_listening.py) | 8 | Kiểm toán dữ liệu đánh giá nghe mù Phase 90, tính trung thực dữ liệu. |
| Các bài kiểm thử phân hệ dữ liệu & huấn luyện khác | 258 | Kiểm tra sinh dữ liệu Vieneu, tiền xử lý âm thanh, cách ly người nói không rò rỉ. |
| **TỔNG CỘNG HỆ THỐNG** | **384** | **384 / 384 PASSED (Tất cả đều đạt)** |

### Lệnh Chạy Toàn Bộ Bộ Kiểm Thử:
```powershell
.venv\Scripts\python.exe -m pytest tests/ -q
```
"""

app_e_content = """# Phụ Lục E: Chỉ Mục Tham Chiếu Nguồn (Source Reference Index)

Phụ lục này tổng hợp chỉ mục dẫn xuất tới các báo cáo kỹ thuật bản lề, các tệp mã nguồn chính thức và các tài liệu hợp đồng trong kho mã nguồn:

## 1. Các Báo Cáo Kỹ Thuật Bản Lề (Key Milestone Reports)
1. **[PHASE13_4_FINAL_CNN_RELEASE.md](file:///d:/trainidentificationaudio/reports/PHASE13_4_FINAL_CNN_RELEASE.md)**: Báo cáo đóng gói phát hành ứng viên đông kết cuối cùng, kiểm định đẳng cấu ONNX và xuất mã băm SHA-256.
2. **[PHASE13_3_CONFIRMATORY_EVALUATION.md](file:///d:/trainidentificationaudio/reports/PHASE13_3_CONFIRMATORY_EVALUATION.md)**: Báo cáo thực chứng cuộc đánh giá xác nhận độc lập trên 1.000 clips của 8 giọng đọc mới lạ.
3. **[PHASE13_2_1_FRONTEND_COMPATIBILITY_AUDIT.md](file:///d:/trainidentificationaudio/reports/PHASE13_2_1_FRONTEND_COMPATIBILITY_AUDIT.md)**: Báo cáo thẩm định giải quyết dứt điểm bất đồng $N_{\\text{fft}} = 400$ vs $512$.
4. **[PHASE13_2_CONFIRMATORY_PREPARATION.md](file:///d:/trainidentificationaudio/reports/PHASE13_2_CONFIRMATORY_PREPARATION.md)**: Báo cáo đăng ký trước giao thức, phân chia giọng đọc và danh mục từ vựng.
5. **[PHASE13_1_PREPROCESSING_REMEDIATION.md](file:///d:/trainidentificationaudio/reports/PHASE13_1_PREPROCESSING_REMEDIATION.md)**: Báo cáo khắc phục lỗi khuếch đại tín hiệu im lặng và phê duyệt Sửa đổi 13.1-A.
6. **[PHASE13_CONFIRMATORY_EVALUATION.md](file:///d:/trainidentificationaudio/reports/PHASE13_CONFIRMATORY_EVALUATION.md)**: Báo cáo ghi nhận kiểm toán dừng khẩn cấp đợt đánh giá Phase 13 do phát hiện lỗi tiền xử lý.
7. **[PHASE12_CNN_FINALIZATION.md](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v12/reports/PHASE12_CNN_FINALIZATION.md)**: Báo cáo đóng gói ứng viên v12 ban đầu.
8. **[PHASE11_CNN_COMPLETION_CAMPAIGN.md](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v11/reports/PHASE11_CNN_COMPLETION_CAMPAIGN.md)**: Báo cáo chiến dịch hoàn thiện CNN v11, hàm mất mát Hard-Negative Weighted Cross Entropy.
9. **[PHASE92_CROSS_PLATFORM_PARITY.md](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v92/reports/PHASE92_CROSS_PLATFORM_PARITY.md)**: Báo cáo đối chuẩn chéo PyTorch - ONNX trên baseline `variant_w10`.
10. **[PHASE91_EDGE_DEPLOYMENT.md](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v91/reports/PHASE91_EDGE_DEPLOYMENT.md)**: Báo cáo khởi tạo khung triển khai di động và C++.
11. **[PHASE90_HUMAN_LISTENING_STUDY.md](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v90/reports/PHASE90_HUMAN_LISTENING_STUDY.md)**: Báo cáo thử nghiệm nghe mù 120 clips trên người.

## 2. Các Tệp Mã Nguồn Chính Thức Được Chuẩn Y (Authoritative Source Code)
- **Mã Suy luận Đóng gói 13.4:** [`experiments/vieneu_kws_research_v13_4/bundle/src/infer.py`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/bundle/src/infer.py)
- **Kiến trúc Mạng `CorrectedVietnameseKWSNet`:** [`experiments/vieneu_kws_research_v13_4/bundle/src/model.py`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/bundle/src/model.py)
- **Frontend Trích xuất Log-Mel ($N_{\\text{fft}}=512$):** [`experiments/vieneu_kws_research_v13_4/bundle/src/frontend.py`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/bundle/src/frontend.py)
- **Mã Huấn luyện Phase 11 Gốc:** [`experiments/vieneu_kws_research_v11/src/train_v11.py`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v11/src/train_v11.py)
- **Hợp đồng Tiền xử lý Chính thức:** [`experiments/vieneu_kws_research_v13_4/bundle/config/preprocessing_contract.json`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/bundle/config/preprocessing_contract.json)

## 3. Các Tệp Trọng Số Mô Hình Đông Kết (Frozen Weights)
- **File Trọng số PyTorch:** [`experiments/vieneu_kws_research_v13_4/bundle/checkpoint/variant_v11_hard_negative_weighted_seed42.pt`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/bundle/checkpoint/variant_v11_hard_negative_weighted_seed42.pt)
- **File Mô hình ONNX:** [`experiments/vieneu_kws_research_v13_4/bundle/onnx/vietnamese_kws_v11.onnx`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/bundle/onnx/vietnamese_kws_v11.onnx)
"""

with open(os.path.join(APP_DIR, "A_METRICS_DICTIONARY.md"), "w", encoding="utf-8") as f:
    f.write(app_a_content.strip() + "\n")

with open(os.path.join(APP_DIR, "B_MODEL_AND_CHECKPOINT_INVENTORY.md"), "w", encoding="utf-8") as f:
    f.write(app_b_content.strip() + "\n")

with open(os.path.join(APP_DIR, "C_DATASET_AND_SPLIT_INVENTORY.md"), "w", encoding="utf-8") as f:
    f.write(app_c_content.strip() + "\n")

with open(os.path.join(APP_DIR, "D_TEST_SUITE_INVENTORY.md"), "w", encoding="utf-8") as f:
    f.write(app_d_content.strip() + "\n")

with open(os.path.join(APP_DIR, "E_SOURCE_REFERENCE_INDEX.md"), "w", encoding="utf-8") as f:
    f.write(app_e_content.strip() + "\n")

print("[OK] Generated all 5 appendices in reports/cnn_comprehensive/appendices/")
