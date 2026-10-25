# Chương 14: Khoảng trống Bằng chứng & Bất đồng (Evidence Gaps & Discrepancies)

## 1. Sổ Đăng Ký 6 Điểm Bất Đồng Kỹ Thuật Trong Lịch Sử Dự Án
Trong quá trình kiểm toán toàn bộ kho mã nguồn và 114 báo cáo kỹ thuật, 6 sai lệch/bất đồng kỹ thuật lớn đã được phát hiện, phân tích nguyên nhân gốc rễ và xử lý dứt điểm (trích xuất từ [`tables/unresolved_discrepancies.csv`](file:///d:/trainidentificationaudio/reports/cnn_comprehensive/tables/unresolved_discrepancies.csv)):

---

### Bất đồng 1: Sai lệch Kích thước FFT ($N_{\text{fft}} = 400$ vs $512$) Giữa Phase 92 và Phase 11
- **Vấn đề ghi nhận:** Báo cáo [Phase 92](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v10/reports/PHASE92_CROSS_PLATFORM_PARITY.md) mô tả frontend trích xuất Log-Mel sử dụng $N_{\text{fft}} = 400$ và bộ lọc Slaney kích thước $40 \times 201$. Tuy nhiên, quy trình chuẩn bị Phase 13.2 lại ghi nhận $N_{\text{fft}} = 512$, $W_{\text{len}} = 400$, $H_{\text{len}} = 160$ và kích thước ma trận Mel là $40 \times 257$.
- **Nguyên nhân gốc rễ (Root Cause):** Phase 92 thực hiện đo đạc đối chuẩn cho mô hình baseline lịch sử `variant_w10` (huấn luyện từ Phase 9/10 với $N_{\text{fft}} = 400$). Khi bước sang chiến dịch hoàn thiện Phase 11, pipeline huấn luyện `train_v11.py` đã chủ động nâng cấp lên $N_{\text{fft}} = 512$ (lũy thừa 2) để tương thích tối ưu với thuật toán FFT trên vi xử lý di động. Báo cáo Phase 92 mang tính lịch sử phản ánh thế hệ cũ, không áp dụng cho thế hệ v11.
- **Biện pháp giải quyết (Resolution):** Thẩm tra mã nguồn thực thi [Phase 13.2.1](file:///d:/trainidentificationaudio/reports/PHASE13_2_1_FRONTEND_COMPATIBILITY_AUDIT.md) xác nhận $100\%$ mã nguồn từ `train_v11.py`, `infer.py` đến file cấu hình `preprocessing_contract.json` của checkpoint đông kết đều sử dụng **$N_{\text{fft}} = 512$**.

---

### Bất đồng 2: Lỗi Chuẩn Hóa Khuếch Đại Tín Hiệu Im Lặng Trong Phase 12
- **Vấn đề ghi nhận:** Tệp `infer.py` trong bundle Phase 12 tự ý áp dụng chuẩn hóa đỉnh biên độ $-1.0\text{ dBFS}$ vô điều kiện. Khi gặp đoạn âm thanh im lặng (nhiễu phòng với biên độ cực đại $0.005$), hàm này khuếch đại tín hiệu lên $+45\text{ dB}$, biến nhiễu nền thành xung âm thanh lớn và kích hoạt nhầm thành lệnh `TRAI`.
- **Nguyên nhân gốc rễ:** Lập trình viên viết mã suy luận Phase 12 đã hiểu sai yêu cầu "chuẩn hóa âm thanh" và tự ý đưa logic căn đỉnh vào mà không đối soát với hàm tiền xử lý của tập huấn luyện `train_v11.py`.
- **Biện pháp giải quyết:** Đợt đánh giá Phase 13 đã kích hoạt dừng khẩn cấp (**BLOCKED**). Giai đoạn Phase 13.1 đã khôi phục đúng logic chuẩn của `train_v11.py`: **Chỉ co tỷ lệ khi $\max(|x|) > 1.0$**, giữ nguyên năng lượng tự nhiên của tín hiệu im lặng.

---

### Bất đồng 3: Sự Cố Nghiên Cứu Lắng Nghe Mù Con Người Phase 90
- **Vấn đề ghi nhận:** Báo cáo Phase 90 lên kế hoạch thử nghiệm nghe mù với 6 người đánh giá độc lập trên 120 clips. Tuy nhiên, tệp kết quả tổng hợp `validated_responses.csv` hoàn toàn bị rỗng và không thể tính toán chỉ số đồng thuận liên đánh giá.
- **Nguyên nhân gốc rễ:** Chỉ có duy nhất `reviewer_01` hoàn thành 120 câu hỏi; 5 người còn lại bỏ dở và không nộp tệp kết quả. Quy trình tổng hợp tự động bị treo do thiếu dữ liệu đầu vào.
- **Biện pháp giải quyết:** Ghi nhận trung thực bằng chứng: Phase 90 chỉ có $N=1$ người hoàn thành (điểm rõ ràng $4.88/5.0$). Không thực hiện điền khống dữ liệu. Duy trì nhãn chính thức `NOT_VALIDATED_ON_NATURAL_SPEECH`.

---

### Bất đồng 4: Thiên Kiến Chọn Lọc Mô Hình Champion Trong Phase 11
- **Vấn đề ghi nhận:** Trong chiến dịch Phase 11, mô hình `seed 42` được chọn làm Champion sau khi người nghiên cứu quan sát thấy nó đạt FAR thấp nhất trên tập Locked Test Split ($0.63\%$ so với $1.04\%$ của seed 43 và $1.67\%$ của seed 44). Điều này vi phạm nguyên tắc cách ly tập kiểm thử khóa (Test Set Peeking Bias).
- **Nguyên nhân gốc rễ:** Áp lực muốn tìm kiếm mô hình có tỷ lệ báo động giả thấp nhất cho giai đoạn tiếp theo đã dẫn đến việc kiểm tra kết quả test set trước khi chốt checkpoint.
- **Biện pháp giải quyết:** Thừa nhận minh bạch thiên kiến chọn lựa trong báo cáo Phase 11 và Phase 12. Triệt tiêu hoàn toàn thiên kiến này bằng cách tổ chức **Đánh giá Xác nhận Độc lập Phase 13.3** trên 1.000 clips âm thanh tổng hợp mới tinh từ 8 giọng đọc chưa từng tiếp xúc. Checkpoint Seed 42 đã chứng minh năng lực thực sự khi đạt $99.20\%$ Accuracy và $0.75\%$ FAR.

---

### Bất đồng 5: Thiên Kiến Phương Ngữ Trong Tập Dữ Liệu Tổng Hợp (Accent Scope Bias)
- **Vấn đề ghi nhận:** Bộ dữ liệu tổng hợp Vieneu TTS v3-Turbo chủ yếu bao gồm các giọng đọc phát âm chuẩn theo phương ngữ Miền Bắc. Các biến thể phương ngữ Miền Trung (ngữ điệu nặng, biến đổi thanh điệu hỏi/ngã) và Miền Nam (biến đổi phụ âm cuối /n/ thành /ŋ/, /t/ thành /k/) chưa được đại diện đầy đủ.
- **Nguyên nhân gốc rễ:** Thiếu hụt các mô hình giọng đọc tổng hợp chất lượng cao cho phương ngữ Trung và Nam tại thời điểm nghiên cứu.
- **Biện pháp giải quyết:** Ban hành Giao thức Sửa đổi 13.1-A (Amendment 13.1-A), thu hẹp phạm vi tuyên bố: mô hình đông kết chỉ được xác nhận khả năng tổng quát hóa trên phương ngữ Miền Bắc; trạng thái của phương ngữ Trung và Nam được gắn nhãn `NOT_VALIDATED_PENDING_DATA`.

---

### Bất đồng 6: Nhầm Lẫn Giữa Kiểm Thử Đóng Khung (Bounded) và Luồng Micro Thực Tế (Streaming)
- **Vấn đề ghi nhận:** Một số tài liệu sơ bộ ban đầu mô tả mô hình "sẵn sàng nhận diện giọng nói liên tục", gây hiểu lầm rằng hệ thống có thể thả vào micro mở để nhận lệnh ngay lập tức.
- **Nguyên nhân gốc rễ:** Không phân định rõ ràng giữa bài toán phân loại clip âm thanh cắt sẵn ($1.0\text{ giây}$) và bài toán nhận diện luồng âm thanh liên tục có tiếng ồn mép cửa sổ.
- **Biện pháp giải quyết:** Chuẩn hóa toàn bộ hệ thống tài liệu với nhãn chính thức `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION` cho chế độ micro mở, đồng thời vạch rõ lộ trình tích hợp VAD và bộ đệm vòng trong tương lai.
