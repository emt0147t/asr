# Chương 12: Khả năng Tái lập & Độ phủ Kiểm thử (Reproducibility & Test Coverage)

## 1. Môi trường Thực thi Chuẩn (Authoritative Execution Environment)
Để tái lập chính xác $100\%$ các kết quả nghiên cứu và kiểm định, môi trường chạy thử nghiệm cần đáp ứng các thông số kỹ thuật sau:
- **Hệ điều hành:** Microsoft Windows 10/11 Pro (hoặc Linux Ubuntu 22.04 LTS).
- **Trình thông dịch:** Python 3.10.x hoặc Python 3.11.x (môi trường ảo `.venv`).
- **Các thư viện phụ thuộc cốt lõi:**
  - `torch == 2.6.0+cpu`
  - `torchaudio == 2.6.0+cpu`
  - `onnx == 1.17.0`
  - `onnxruntime == 1.20.1`
  - `numpy == 1.26.4`
  - `scipy == 1.14.1`
  - `pytest == 8.3.4`

---

## 2. Hướng Dẫn Các Lệnh Tái Lập Từng Bước (Step-by-Step Reproduction)

### Bước 1: Xác thực Toàn vẹn Mã Băm Checkpoint
Kiểm tra mã băm SHA-256 của file trọng số đông kết:
```powershell
# PowerShell trên Windows
Get-FileHash -Algorithm SHA256 experiments/vieneu_kws_research_v13_4/bundle/checkpoint/variant_v11_hard_negative_weighted_seed42.pt
```
*Kết quả bắt buộc:* `3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2`

### Bước 2: Chạy Toàn Bộ Bộ Kiểm Thử Tự Động (Workspace Test Suite)
Kho mã nguồn hiện có **384 bài kiểm thử tự động**, bao phủ toàn diện từ toán học Mel Spectrogram, kiểm tra chống rò rỉ dữ liệu, kiểm tra đẳng cấu ONNX, đến quy trình đánh giá mù:
```powershell
.venv\Scripts\python.exe -m pytest tests/ -q
```
*Kết quả kỳ vọng:* **384 passed in ~15-25s** (Tỷ lệ đạt $100.00\%$).

### Bước 3: Tái lập Bài Đánh giá Xác nhận Độc lập Phase 13.3
Chạy trực tiếp tệp kiểm thử chuyên dụng của Phase 13.3:
```powershell
.venv\Scripts\python.exe -m pytest tests/test_vieneu_research_v13_3.py -v
```
Kiểm thử này sẽ:
1. Đọc lại 1.000 clips âm thanh tổng hợp từ `experiments/vieneu_kws_research_v13_3/audio/`.
2. Nạp checkpoint ứng viên đông kết với ngưỡng $\theta_{\text{cmd}} = 0.20$.
3. Kiểm tra tính trùng khớp của 8 cổng nghiệm thu (Command Recall $99.00\%$, FAR $0.75\%$, Accuracy $99.20\%$).

### Bước 4: Kiểm Tra Tính Đẳng Cấu PyTorch vs ONNX Runtime
Chạy bài kiểm thử xuất mô hình và kiểm tra sai số số học:
```powershell
.venv\Scripts\python.exe -m pytest tests/test_vieneu_research_v13_4.py -v
```
Kiểm thử này đảm bảo độ lệch logit cực đại $\Delta_{\text{max}} < 1.0 \times 10^{-4}$ và nhãn dự đoán trùng khớp $100\%$.

---

## 3. Tổng Quan Danh Mục Bộ Kiểm Thử Tự Động (Test Suite Inventory)
Bảng dưới đây thống kê các mô-đun kiểm thử quan trọng trong thư mục [`tests/`](file:///d:/trainidentificationaudio/tests/):

| Tệp Kiểm Thử (Test Module) | Số Lượng Tests | Phạm Vi và Mục Tiêu Kiểm Tra |
| :--- | :--- | :--- |
| [`tests/test_vieneu_research_v13_4.py`](file:///d:/trainidentificationaudio/tests/test_vieneu_research_v13_4.py) | 8 | Đóng gói phát hành Phase 13.4, kiểm tra đẳng cấu ONNX, tính bất biến của SHA-256. |
| [`tests/test_vieneu_research_v13_3.py`](file:///d:/trainidentificationaudio/tests/test_vieneu_research_v13_3.py) | 12 | Thẩm định 8 cổng nghiệm thu Phase 13.3 trên 1.000 clips âm thanh độc lập. |
| [`tests/test_vieneu_research_v13_2_1.py`](file:///d:/trainidentificationaudio/tests/test_vieneu_research_v13_2_1.py) | 6 | Kiểm định giải quyết bất đồng $N_{\text{fft}} = 400$ vs $512$ và Mel Filterbank. |
| [`tests/test_vieneu_research_v13_1.py`](file:///d:/trainidentificationaudio/tests/test_vieneu_research_v13_1.py) | 10 | Khắc phục lỗi khuếch đại tín hiệu im lặng, xác minh Conditional Clamping. |
| [`tests/test_vieneu_research_v11.py`](file:///d:/trainidentificationaudio/tests/test_vieneu_research_v11.py) | 16 | Huấn luyện Phase 11, hàm mất mát Hard-Negative Weighted Cross Entropy, multi-seed. |
| [`tests/test_cross_platform_parity_phase92.py`](file:///d:/trainidentificationaudio/tests/test_cross_platform_parity_phase92.py) | 14 | Đối chuẩn chéo PyTorch - ONNX, đo đạc độ trễ phân vị P50/P95. |
| [`tests/test_phase90_human_listening.py`](file:///d:/trainidentificationaudio/tests/test_phase90_human_listening.py) | 8 | Kiểm toán dữ liệu đánh giá con người Phase 90, phát hiện thiếu 5 người chấm. |
| Các bài kiểm thử phân hệ khác | 310 | Kiểm tra tiền xử lý âm thanh, sinh dữ liệu Vieneu, phân chia tập dữ liệu không rò rỉ. |
| **TỔNG CỘNG WORKSPACE** | **384** | **100% PASS RATE (Tất cả đều vượt qua).** |
