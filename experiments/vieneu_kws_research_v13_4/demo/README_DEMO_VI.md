# Hướng Dẫn Sử Dụng Demo Nhận Diện Khẩu Lệnh Micro Trực Tiếp (Laptop Live-Mic KWS Demo)
> **Dự án:** Nhận diện khẩu lệnh tiếng Việt (Vietnamese Keyword Spotting - KWS)  
> **Ứng viên mô hình:** `variant_v11_hard_negative_weighted_seed42` (Phase 13.4 Release Candidate)  
> **Phiên bản:** Thử nghiệm tương tác phục vụ trình bày học thuật  

---

## 1. Tuyên Bố Ranh Giới Quản Trị & Khoa Học Bắt Buộc (Mandatory Disclaimers)

> [!CAUTION]
> **TÍNH CHẤT NGUYÊN MẪU THỬ NGHIỆM (EXPERIMENTAL PROTOTYPE ONLY):**  
> Ứng dụng này là một **bản demo thử nghiệm kỹ thuật tương tác** phục vụ minh họa trong buổi báo cáo học thuật. Đây **KHÔNG PHẢI** là một hệ thống thương mại hoàn chỉnh và **CHƯA ĐƯỢC KIỂM ĐỊNH KHOA HỌC** cho bài toán streaming micro mở liên tục ngoài thực địa.

Mọi tài liệu, màn hình demo và báo cáo phải bảo lưu chính xác 4 nhãn định danh quản trị:
- `EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`
- `NOT_VALIDATED_ON_NATURAL_SPEECH` (Mô hình mới kiểm định trên dữ liệu tổng hợp Vieneu TTS, chưa thẩm định trên giọng người thật tự nhiên).
- `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION` (Mô hình huấn luyện trên khung 1.0 giây đóng, chưa tối ưu cho continuous open-mic).
- `NOT_VALIDATED_PENDING_DATA` (Chưa kiểm định độc lập phương ngữ Miền Trung và Miền Nam do thiếu tập dữ liệu).

---

## 2. Hướng Dẫn Khởi Chạy Nhanh Trên Windows (Quickstart Commands)

### Cách 1: Khởi chạy nhanh bằng tệp Batch (One-Click Launch)
Nhấp đúp chuột vào tệp [`run_live_demo.bat`](file:///d:/trainidentificationaudio/run_live_demo.bat) tại thư mục gốc của dự án. Ứng dụng sẽ tự động khởi chạy ở chế độ **Segmented Command Mode** (chế độ nhận diện đa khẩu lệnh tối ưu).

### Cách 2: Khởi chạy bằng dòng lệnh PowerShell / Command Prompt

```powershell
# 1. Di chuyển vào thư mục gốc dự án
cd D:\trainidentificationaudio

# 2. Liệt kê danh sách các thiết bị micro khả dụng trên máy tính
.venv\Scripts\python.exe run_live_demo.py --list-devices

# 3. Khởi chạy giao diện Terminal Dashboard (Mặc định: Segmented Mode, ONNX)
.venv\Scripts\python.exe run_live_demo.py

# 4. Chỉ định thiết bị micro cụ thể (ví dụ ID 1: Microphone Realtek)
.venv\Scripts\python.exe run_live_demo.py --device 1

# 5. Khởi chạy giao diện đồ họa cửa sổ (Tkinter GUI Window)
.venv\Scripts\python.exe run_live_demo.py --gui

# 6. Chạy với backend PyTorch (thay vì ONNX)
.venv\Scripts\python.exe run_live_demo.py --backend pytorch

# 7. (Tùy chọn) Chạy chế độ streaming cũ (Sliding Window 160ms)
.venv\Scripts\python.exe run_live_demo.py --mode streaming
```

---

## 3. Kiến Trúc Hai Chế Độ Hoạt Động (Operating Modes & Signal Pipeline)

### 3.1. Chế Độ Mặc Định Khuyến Nghị: Phân Đoạn Khẩu Lệnh (Segmented Command Mode)
Chế độ này được thiết kế để giải quyết triệt để vấn đề nhận diện chuỗi nhiều khẩu lệnh liên tiếp (ví dụ: `TRÁI` &rarr; `XUỐNG` &rarr; `PHẢI` hoặc lặp lại `TRÁI` &rarr; `TRÁI`):
1. **Dò điểm bắt đầu lời nói (Speech Onset Detection):** Kích hoạt khi năng lượng âm thanh vượt ngưỡng `-48.0 dB`.
2. **Đệm đón đầu (Pre-roll Buffer 150 ms):** Duy trì 2.400 mẫu trong đệm vòng FIFO để không làm mất phụ âm xát đầu từ (/tr/, /ph/, /l/, /x/).
3. **Tích lũy & Bắt kết thúc khẩu lệnh (Trailing Silence 300 ms):** Tích lũy chuỗi âm thanh và chốt phân đoạn khi phát hiện khoảng lặng nghỉ giữa các từ kéo dài 300 ms (tương ứng với nhịp ngắt tự nhiên khi nói).
4. **Chuẩn hóa khung 1.0 giây chuẩn:** Đệm số 0 phía phải (right-zero pad) để đạt đúng 16.000 mẫu float32 theo đúng hợp đồng tiền xử lý Phase 13.1.
5. **Tái kích hoạt tức thì (Zero-Latch Instant Re-arm):** Mỗi phân đoạn chỉ phát sinh đúng 1 sự kiện kích hoạt và ngay lập tức giải phóng bộ tích lũy để đón nhận lệnh tiếp theo, không bị khóa chặn bởi cooldown toàn cục 0.8s.

### 3.2. Chế Độ Cũ: Cửa Sổ Trượt (Streaming Sliding Window Mode)
- Trượt cửa sổ 1.0 giây (16.000 mẫu) với bước nhảy 160 ms.
- Sử dụng bộ lọc VAD và cơ chế cooldown 0.8s để chống kích hoạt trùng lặp. Phù hợp cho việc thử nghiệm phản hồi liên tục nhưng sẽ gặp khó khăn khi nói nhiều lệnh liên tiếp trong thời gian ngắn.

---

## 4. Hướng Dẫn Phát Âm Khi Thuyết Trình Học Thuật (Presentation Speaking Tips)

Để demo đạt độ nhạy và chính xác cao nhất trong buổi báo cáo:
1. **Nói dứt khoát với khoảng ngắt tự nhiên:** Khi thử nghiệm chuỗi lệnh liên tiếp (ví dụ: `TRÁI` &rarr; `XUỐNG` &rarr; `PHẢI`), hãy giữ khoảng nghỉ tự nhiên khoảng **300–500 ms** giữa mỗi từ (ví dụ: *"Trái"* ... dừng một nhịp ngắn ... *"Xuống"* ... dừng một nhịp ngắn ... *"Phải"*).
2. **Tránh nói dính liền không ngắt nghỉ:** Không phát âm dính liền *"tráixuốngphải"* như một từ đơn không có khoảng lặng; mô hình CNN v11 là bộ phân loại đơn nhãn 1.0s, cần ranh giới âm học giữa các từ.
3. **Phát âm chuẩn âm vị:** Phát âm rõ ràng âm đầu và thanh điệu tiếng Việt theo 4 nhãn chuẩn: `TRÁI`, `PHẢI`, `LÊN`, `XUỐNG`. Mọi từ khác hoặc tiếng nói chuyện phiếm sẽ rơi vào nhãn `REJECT`.

---

## 5. Hợp Đồng Tiền Xử Lý & Logic Quyết Định (Frozen Inference Contract)

1. **Hợp đồng tiền xử lý đông kết (Frozen Preprocessing Contract):**
   - Co biên độ có điều kiện: chỉ co lại khi $\max(|x|) > 1.0$; **tuyệt đối không khuếch đại nhiễu nền**.
   - STFT: $N_{\text{fft}} = 512$, Hann window $W_{\text{len}} = 400$, $H_{\text{len}} = 160$.
   - 40 dải lọc Slaney Mel trong dải $20.0 - 8000.0\text{ Hz}$. Nén $\ln(\text{Mel} + 10^{-6})$.
   - Kích thước đặc trưng: đúng `[1, 1, 40, 101]`.
2. **Logic Quyết Định & Ngưỡng Kích Hoạt:**
   - 5 lớp: `TRAI`, `PHAI`, `LEN`, `XUONG`, `REJECT`.
   - Ngưỡng kích hoạt: **$\theta^*_{\text{cmd}} = 0.20$**.
   - Phân loại: $\text{argmax}$ trên 4 lệnh mục tiêu nếu $\max(p) \ge 0.20$; ngược lại trả về `REJECT`.

---

## 6. Xử Lý Sự Cố Thường Gặp (Troubleshooting Guide)

### Sự cố 1: Không mở được Micro ("PortAudioError: Invalid number of channels" hoặc "Permission Denied")
- **Nguyên nhân:** Windows chặn quyền truy cập micro đối với ứng dụng máy tính hoặc driver Realtek đang bị khóa độc quyền (Exclusive Mode).
- **Cách khắc phục:**
  1. Mở Windows **Settings &rarr; Privacy & Security &rarr; Microphone**.
  2. Bật công tắc **"Microphone access"** và **"Let desktop apps access your microphone"**.
  3. Mở Control Panel &rarr; Sound &rarr; Recording &rarr; Nhấp đúp vào Microphone &rarr; Advanced &rarr; Bỏ chọn *"Allow applications to take exclusive control of this device"*.

### Sự cố 2: Danh sách thiết bị báo nhiều micro, không biết chọn cái nào
- **Cách khắc phục:** Chạy lệnh:
  ```powershell
  .venv\Scripts\python.exe run_live_demo.py --list-devices
  ```
  Thông thường:
  - Thiết bị có tên `Microphone (2- Realtek(R) Audio)` với API `MME` hoặc `DirectSound` là micro tích hợp sẵn trên thân laptop.
  - Thiết bị `Jack Mic` là cổng tai nghe 3.5mm.
  - Sử dụng cờ `--device <ID>` để chỉ định đúng micro mong muốn.

### Sự cố 3: Nói nhưng mô hình không phản hồi (Level dB quá thấp)
- **Cách khắc phục:**
  - Nhìn thanh VU Meter trên giao diện demo: nếu nói bình thường mà thanh mức âm chỉ dao động dưới `-55 dB`, hãy tăng âm lượng thu của Micro trong Windows Sound Settings (kéo slider lên 80-100% hoặc bật Microphone Boost +10dB).
  - Nói khẩu lệnh dứt khoát, to, rõ ràng theo đúng âm vị tiếng Việt: "TRÁI", "PHẢI", "LÊN", "XUỐNG".

---

## 7. Nhật Ký & Quyền Riêng Tư (Privacy & Logging Policy)

- **Không ghi âm thô:** Hệ thống **TUYỆT ĐỐI KHÔNG** lưu các tệp âm thanh `.wav` giọng nói người thật từ microphone vào ổ cứng.
- **Nhật ký sự kiện:** Chỉ ghi nhận siêu dữ liệu kích hoạt (thời điểm, tên lệnh, độ tin cậy, độ trễ và mức dB) vào tệp [`live_demo_events.log`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/demo/live_demo_events.log).
- **Dừng an toàn:** Nhấn tổ hợp phím **Ctrl+C** trên bàn phím (hoặc nhấn nút "DỪNG MICRO" trên cửa sổ GUI) để đóng luồng thu âm ngay lập tức.
