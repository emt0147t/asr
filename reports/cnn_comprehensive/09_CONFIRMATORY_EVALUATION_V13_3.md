# Chương 09: Đánh giá Xác nhận Độc lập Phase 13.3 (Confirmatory Evaluation Phase 13.3)

## 1. Tổng quan Giao thức Đăng ký Trước (Preregistered Protocol)
Sau khi khắc phục triệt để lỗi khuếch đại tín hiệu im lặng tại Phase 13.1 và giải quyết bất đồng tham số Frontend tại Phase 13.2.1, giai đoạn [Phase 13.3](file:///d:/trainidentificationaudio/reports/PHASE13_3_CONFIRMATORY_EVALUATION.md) đã thực thi quy trình **Đánh giá Xác nhận Độc lập 1 Lần Duy Nhất (Preregistered One-Shot Confirmatory Evaluation)**.

Quy trình tuân thủ nghiêm ngặt Giao thức sửa đổi [Amendment 13.1-A](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_2/protocol/AMENDED_CONFIRMATORY_PROTOCOL_V13_2.md):
- **Checkpoint Đánh giá:** [`experiments/vieneu_kws_research_v13_1/bundle/checkpoint/variant_v11_hard_negative_weighted_seed42.pt`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_1/bundle/checkpoint/variant_v11_hard_negative_weighted_seed42.pt)
- **Mã băm SHA-256 Xác thực:** `3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2`
- **Ngưỡng Quyết định Bất biến:** $\theta_{\text{cmd}} = 0.20$
- **Quy mô Tập dữ liệu:** 1.000 clips âm thanh tổng hợp hoàn toàn mới, được sinh ra từ **8 giọng đọc Vieneu phương ngữ Miền Bắc** chưa từng xuất hiện trong bất kỳ tập huấn luyện hay kiểm thử nào trước đây.

---

## 2. Kết Quả Định Lượng Toàn Diện 8 Tiêu Chí Nghiệm Thu
Cuộc đánh giá đã kiểm tra mô hình qua 8 cổng nghiệm thu kỹ thuật (Acceptance Gates). Dưới đây là bảng đối soát chi tiết:

| Tiêu chí Nghiệm thu (Gate) | Yêu cầu Kỹ thuật Đăng ký trước | Kết quả Thực chứng Đạt được | Trạng thái (Verdict) |
| :--- | :--- | :--- | :--- |
| **Gate 1: Command Recall Tổng** | $\ge 95.00\%$ | **$99.00\%$** (198/200 clips) | **ĐẠT (PASSED)** |
| **Gate 2: Recall Từng Lớp Độc lập** | $\ge 90.00\%$ cho mỗi lệnh | `TRAI`: $98.0\%$ (49/50)<br/>`PHAI`: $100.0\%$ (50/50)<br/>`LEN`: $98.0\%$ (49/50)<br/>`XUONG`: $100.0\%$ (50/50) | **ĐẠT (PASSED)** |
| **Gate 3: Tỷ lệ Báo động Giả (FAR)** | $\le 2.00\%$ trên 800 non-targets | **$0.75\%$** (6/800 clips) | **ĐẠT (PASSED)** |
| **Gate 4: Độ chính xác Toàn diện** | $\ge 95.00\%$ trên 1.000 clips | **$99.20\%$** (992/1.000 clips) | **ĐẠT (PASSED)** |
| **Gate 5: Lẫn lộn giữa các Lệnh** | $\le 1.00\%$ (Command Confusion) | **$0.00\%$** (0/200 clips) | **ĐẠT (PASSED)** |
| **Gate 6: Độ trễ Suy luận Trung bình** | Mean Latency $\le 20.0\text{ ms}$ | **$1.80\text{ ms}$** (trên CPU Intel x86) | **ĐẠT (PASSED)** |
| **Gate 7: Phân vị Độ trễ P95** | P95 Latency $\le 50.0\text{ ms}$ | **$2.12\text{ ms}$** | **ĐẠT (PASSED)** |
| **Gate 8: Kiểm soát Hạt & Băm File** | SHA-256 khớp tuyệt đối | Khớp $100\%$ mã băm gốc | **ĐẠT (PASSED)** |

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
1. **Tạp âm Môi trường Thuần túy (Pure Noise - 200 clips):** $0 / 200$ báo động giả ($0.00\%$ FAR). Lớp từ chối hấp thụ hoàn hảo.
2. **Tiếng nói Thông thường Ngoài Danh mục (General Speech - 200 clips):** $1 / 200$ báo động giả ($0.50\%$ FAR) vào nhãn `PHAI`.
3. **Từ ngữ Gây nhiễu Âm vị Học (Hard Negatives - 200 clips):** $2 / 200$ báo động giả ($1.00\%$ FAR): 1 clip nhầm `TRAI`, 1 clip nhầm `XUONG`.
4. **Câu Hội thoại Chứa Từ Gây nhiễu (Carrier Phrases - 200 clips):** $3 / 200$ báo động giả ($1.50\%$ FAR): 2 clips nhầm `PHAI`, 1 clip nhầm `XUONG`.

---

## 4. Ranh Giới và Phạm Vi Giá Trị Thống Kê
Mặc dù vượt qua xuất sắc cả 8 tiêu chí nghiệm thu, kết quả của Phase 13.3 bắt buộc phải được hiểu trong giới hạn đã đăng ký trước:
1. **Phạm vi Phương ngữ:** Kết quả chỉ có giá trị xác nhận khả năng tổng quát hóa trên **8 giọng đọc tổng hợp mới thuộc phương ngữ Miền Bắc**.
2. **Giọng Miền Trung & Miền Nam:** Hoàn toàn chưa được xác nhận độc lập do không có giọng đọc tổng hợp đạt chuẩn và chưa có dữ liệu thu âm người thật có bản quyền.
3. **Tuyên bố Hợp lệ:** Checkpoint `variant_v11_hard_negative_weighted_seed42.pt` được xác nhận là một **Ứng viên Mô hình Bounded-Utterance Xuất sắc cho Giọng Bắc**, sẵn sàng bước vào giai đoạn đóng gói phát hành (Phase 13.4).
