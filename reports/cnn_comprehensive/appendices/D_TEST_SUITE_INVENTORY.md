# Phụ Lục D: Danh Mục Bộ Kiểm Thử Tự Động (Test Suite Inventory)

Kho mã nguồn `D:\trainidentificationaudio` duy trì hệ thống kiểm thử tự động toàn diện với **384 bài kiểm thử** (đạt tỷ lệ vượt qua $100.00\%$). Bảng dưới đây phân loại chi tiết các tệp kiểm thử trong thư mục [`tests/`](file:///d:/trainidentificationaudio/tests/):

| Tệp Kiểm Thử (Test File) | Số Lượng Tests | Nội Dung & Chức Năng Kiểm Tra |
| :--- | :---: | :--- |
| [`tests/test_vieneu_research_v13_4.py`](file:///d:/trainidentificationaudio/tests/test_vieneu_research_v13_4.py) | 8 | Kiểm thử đóng gói phát hành Phase 13.4, tính toàn vẹn SHA-256, đẳng cấu ONNX. |
| [`tests/test_vieneu_research_v13_3.py`](file:///d:/trainidentificationaudio/tests/test_vieneu_research_v13_3.py) | 12 | Thẩm định 8 cổng nghiệm thu trên 1.000 clips độc lập của Phase 13.3. |
| [`tests/test_vieneu_research_v13_2_1.py`](file:///d:/trainidentificationaudio/tests/test_vieneu_research_v13_2_1.py) | 6 | Kiểm tra tính tương thích tham số Frontend FFT ($N_{\text{fft}}=512$) và Mel filterbank. |
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
