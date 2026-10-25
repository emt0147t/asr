# Báo Cáo Kiểm Định Chất Lượng Dữ Liệu KWS v12 (Hiệu Đính)
## (Vietnamese KWS Dataset v12 Quality Assurance & Gate Report - Corrected)

> **Mã cấu hình tạo tập dữ liệu:** `vieneu_kws_v12_split_20261022`  
> **Cổng kỹ thuật tự động (Automated Technical Gate):** **PASSED (ĐẠT — ĐỦ ĐIỀU KIỆN TIẾN HÀNH E1)**  
> **Thẩm định nghe chủ quan (Human Listening Sign-off):** **PENDING / NOT_PERFORMED (`NOT_VALIDATED_PENDING_DATA`)**  
> **Tổng số mẫu:** **8,000 clips** (Mục tiêu: 8,000)  
> **Phân chia:** Train (5,440), Val (1,280), Test (1,280)  

---

### 1. Ranh Giới Quản Trị Khoa Học Bắt Buộc
- `EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`
- `NOT_VALIDATED_ON_NATURAL_SPEECH`
- `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`
- `NOT_VALIDATED_PENDING_DATA`

---

### 2. Bảng Tổng Hợp 10 Tiêu Chí Kiểm Định (QA Gate Matrix)

| STT | Tiêu Chí Kiểm Định (QA Check) | Kết Quả | Chi Tiết Kỹ Thuật |
| :---: | :--- | :---: | :--- |
| **1** | **Khớp số lượng mẫu thực tế vs mục tiêu** | **PASS** | Thực tế: 8000 / Mục tiêu: 8000. Lệnh: 4000, Reject: 4000 |
| **2** | **Tính nhất quán tệp, định dạng 16kHz mono 1s** | **PASS** | 0 tệp mất, 0 tệp hỏng, 100% đúng 16,000 mẫu @ 16 kHz mono float32 |
| **3** | **Kiểm soát Clipping, Peak/RMS và SNR** | **PASS** | 0 tệp clipping (>0.999), 0 tệp target im lặng, SNR thực tế khớp danh định |
| **4** | **Trùng lặp chính xác & Nguồn phát sinh** | **PASS** | 100% mã băm SHA-256 là duy nhất (8000 hashes), phân nhóm theo nguồn |
| **5** | **Rò rỉ phân chia (Split Isolation & Leakage)** | **PASS** | 0 rò rỉ giọng đọc, 0 rò rỉ mẫu gốc, 0 rò rỉ OOV unseen ("ghế"), 0 rò rỉ tệp nhiễu |
| **6** | **Cân bằng lớp & Phân bố Clean / Noisy SNR** | **PASS** | 4 lớp lệnh cân bằng tuyệt đối (1,000/lớp); clean: 1680, noisy: 3920 |
| **7** | **Đầy đủ Metadata & Nguồn gốc phát sinh (Lineage)** | **PASS** | 21/21 trường metadata hợp lệ, 0 giá trị khuyết thiếu ở các trường khóa |
| **8** | **Mã băm cổ vật & Tính bất biến mô hình v11** | **PASS** | Checkpoint v11 & ONNX khớp 100% byte-for-byte SHA-256 có thẩm quyền |
| **9** | **Nhiễm bẩn nhãn REJECT (Target Contamination)** | **PASS** | 0/1,000 tệp FLEURS và 0/1,600 tệp OOV chứa từ khóa mục tiêu (`trái`, `phải`, `lên`, `xuống`) |
| **10** | **Thẩm định nghe chủ quan của người thật** | **PENDING** | Ghi nhận trung thực: `NOT_VALIDATED_PENDING_DATA` (không ngụy tạo phê duyệt người nghe) |

---

### 3. Giải Trình Chênh Lệch Tiêu Chí 10 (Discrepancy Addendum)
Báo cáo sơ bộ trước đây từng ghi Check 9 là "PASS" kèm ghi chú "NOT_VALIDATED_PENDING_DATA". Nhằm bảo đảm tính minh bạch khoa học tuyệt đối:
- **Kiểm định kỹ thuật tự động (Automated Technical Verification - Checks 1-9):** **ĐẠT (PASSED)**. Tất cả các tham số âm học, hình dạng tensor, phân bố nhãn, cách ly phân vùng và mã băm toàn vẹn đã được kiểm định độc lập và tự động 100%.
- **Thẩm định nghe của người thật (Human Listening Sign-off - Check 10):** **PENDING / NOT_PERFORMED**. Chưa được thực hiện và được gắn nhãn quản trị chính thức `NOT_VALIDATED_PENDING_DATA`. Không có bất kỳ sự ngụy tạo đánh giá chủ quan nào.
- **Quyết định cho phép thử nghiệm:** Cổng kỹ thuật tự động hoàn toàn đủ điều kiện để khởi động thử nghiệm có kiểm soát E1 (`exp_v12_e1_baseline_scale`). Hệ thống KHÔNG được triển khai thực tế (`NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`).
