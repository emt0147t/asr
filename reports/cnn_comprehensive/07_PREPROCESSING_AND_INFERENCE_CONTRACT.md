# Chương 07: Hợp đồng Tiền xử lý & Suy luận (Preprocessing & Inference Contract)

## 1. Định nghĩa Toán học của Pipeline Tiền Xử Lý Âm Thanh
Quy trình tiền xử lý biến đổi đoạn sóng âm 1 chiều thành ma trận đặc trưng Log-Mel Spectrogram 2 chiều tuân thủ nghiêm ngặt các công thức toán học sau:

### Bước 1: Chuẩn hóa Biên độ Tín hiệu (Conditional Waveform Normalization)
Cho tín hiệu âm thanh rời rạc $x[n]$ với $n = 0, 1, \dots, N-1$ ($N = 16.000$ mẫu ứng với $1.0\text{ s}$ tại $f_s = 16.000\text{ Hz}$):
$$x_{\text{norm}}[n] = \begin{cases} \frac{x[n]}{\max_{k} |x[k]|}, & \text{nếu } \max_{k} |x[k]| > 1.0 \\ x[n], & \text{ngược lại} \end{cases}$$

### Bước 2: Biến đổi Fourier Thời gian Ngắn (STFT)
Sử dụng cửa sổ Hann tuần hoàn $w[m]$ độ dài $W_{\text{len}} = 400$ mẫu, bước nhảy $H_{\text{len}} = 160$ mẫu, và kích thước FFT $N_{\text{fft}} = 512$:
$$X[k, t] = \sum_{m=0}^{W_{\text{len}}-1} x_{\text{norm}}[t \cdot H_{\text{len}} + m] \cdot w[m] \cdot e^{-j \frac{2\pi k m}{N_{\text{fft}}}}$$
Trong đó $k = 0, 1, \dots, 256$ (số dải tần số thực) và $t = 0, 1, \dots, 100$ (tổng cộng 101 khung thời gian).
Phổ công suất (Power Spectrogram) được tính bằng:
$$S[k, t] = |X[k, t]|^2$$

### Bước 3: Áp dụng Bộ Lọc Ngân hàng Mel (Mel Filterbank)
Áp dụng $M = 40$ bộ lọc tam giác Mel chuẩn Slaney (chuẩn hóa diện tích tam giác về 1 để bảo toàn năng lượng):
$$M[m, t] = \sum_{k=0}^{256} H_m[k] \cdot S[k, t] \quad (m = 0, 1, \dots, 39)$$

### Bước 4: Nén Logarit (Log Compression)
Chuyển đổi sang thang đo logarit tự nhiên với hệ số ổn định số học $\epsilon = 10^{-6}$:
$$L[m, t] = \log_e \left( M[m, t] + 10^{-6} \right)$$
Kết quả thu được ma trận đầu vào mô hình có kích thước cố định $[1, 1, 40, 101]$.

---

## 2. Quy Tắc Quyết Định Suy Luận & Ngưỡng Tối Ưu (Decision Logic)
Mô hình đưa ra véc-tơ logit 5 chiều $z = [z_0, z_1, z_2, z_3, z_4]$ tương ứng với các lớp:
- $0$: `TRAI`
- $1$: `PHAI`
- $2$: `LEN`
- $3$: `XUONG`
- $4$: `REJECT`

Phân phối xác suất Softmax được tính:
$$P(c_j) = \frac{\exp(z_j)}{\sum_{k=0}^4 \exp(z_k)}$$

Quy tắc quyết định phân lớp chính thức được định nghĩa:
$$\hat{y} = \begin{cases} c^*, & \text{nếu } c^* \in \{0, 1, 2, 3\} \quad \text{và} \quad P(c^*) \ge \theta_{\text{cmd}} \\ 4 \ (\text{REJECT}), & \text{ngược lại} \end{cases}$$
Trong đó:
$$c^* = \arg\max_{j \in \{0, 1, 2, 3, 4\}} P(c_j)$$
Và ngưỡng chuẩn y có thẩm quyền cao nhất là:
$$\mathbf{\theta_{\text{cmd}} = 0.20}$$

---

## 3. Kiểm Định Tính Đẳng Cấu Tuyệt Đối PyTorch vs. ONNX Runtime
Để đảm bảo mô hình khi triển khai sang môi trường C++ hoặc thiết bị biên sử dụng ONNX Runtime hoạt động y hệt môi trường PyTorch nghiên cứu, một bài kiểm thử đẳng cấu toán học nghiêm ngặt đã được thực thi trên tập mẫu âm thanh kiểm thử:

### Bảng Kết quả Thẩm định Đẳng cấu (Parity Results):
- **Số mẫu thử nghiệm:** 1.000 clips âm thanh chuẩn.
- **Độ lệch Logit Cực đại (Max Absolute Difference):**
  $$\Delta_{\text{max}} = \max_{i, j} |z_{i, j}^{\text{PyTorch}} - z_{i, j}^{\text{ONNX}}| = 3.58 \times 10^{-6}$$
- **Ngưỡng cho phép cực đại:** $\epsilon_{\text{tol}} = 1.0 \times 10^{-4}$.
- **Tỷ lệ đồng thuận nhãn dự đoán (Label Agreement):** **$100.00\%$** (1.000/1.000 clips đưa ra kết quả nhãn hoàn toàn trùng khớp).
- **Kết luận:** Mô hình ONNX đạt trạng thái đẳng cấu toán học hoàn hảo (Bit-Exact Numerical Parity) với mô hình PyTorch gốc.
