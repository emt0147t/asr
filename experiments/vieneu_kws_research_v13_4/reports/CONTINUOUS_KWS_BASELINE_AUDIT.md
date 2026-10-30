# Báo Cáo Kiểm Toán Kỹ Thuật Pipeline Continuous KWS & Chẩn Đoán Lỗi Lệnh Liền Mạch
## (Technical Audit of Continuous KWS Live-Mic Pipeline & Root-Cause Analysis for Rapid Command Sequences)

> **Mã tài liệu:** `CONTINUOUS_KWS_BASELINE_AUDIT`  
> **Repository:** `D:\trainidentificationaudio`  
> **Thời điểm thẩm định:** 2026-10-29  
> **Phạm vi:** Kiểm toán trước khi sửa lỗi (Pre-Fix Forensic Audit) — Chưa triển khai giải pháp  
> **Trạng thái:** AUDIT COMPLETE — ROOT CAUSE ISOLATED  

---

### 1. Ranh Giới Quản Trị Khoa Học Bắt Buộc

Báo cáo này tuân thủ nghiêm ngặt 4 nhãn quản trị bất biến của dự án:
- `EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`
- `NOT_VALIDATED_ON_NATURAL_SPEECH`
- `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`
- `NOT_VALIDATED_PENDING_DATA`

**Nguyên tắc kỷ luật kỹ thuật tuyệt đối trong bước kiểm toán:**
1. **Không sửa model v11:** Checkpoint và ONNX v11 giữ nguyên bản byte-for-byte.
2. **Không huấn luyện lại:** Không thay đổi tham số, không chạy huấn luyện E1/E2.
3. **Không chỉnh ngưỡng giả lập:** Không hạ thấp ngưỡng khoảng lặng nhân tạo chỉ để làm demo vượt qua bài kiểm tra một cách hình thức.
4. **Không ngụy tạo kết quả kiểm thử:** Toàn bộ dữ liệu trong báo cáo được trích xuất từ mã nguồn thực tế và các thực nghiệm phát lại (replay) có thể tái lập 100%.

---

### 2. Kiểm Tra Trạng Thái Ban Đầu (Initial State Verification)

#### 2.1 Trạng thái Git (Git Status)
- Nhánh hiện tại: `master`
- Trạng thái tệp được theo dõi: Sạch sẽ, không có tệp mã nguồn nào bị sửa đổi ngoại trừ tệp nhật ký sự kiện runtime [`live_demo_events.log`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/demo/live_demo_events.log).
- Các tệp chưa theo dõi (untracked): Là các tài liệu và sổ bộ nghiên cứu từ Phase E2 trước đó.

#### 2.2 Xác nhận sự tồn tại của các tệp mã nguồn trọng tâm:
1. [`experiments/vieneu_kws_research_v13_4/demo/live_mic_engine.py`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/demo/live_mic_engine.py): **TỒN TẠI** (35.920 bytes, 899 dòng).
2. [`run_live_demo.py`](file:///d:/trainidentificationaudio/run_live_demo.py): **TỒN TẠI** (1.097 bytes, 37 dòng).
3. [`tests/test_vieneu_research_v13_4.py`](file:///d:/trainidentificationaudio/tests/test_vieneu_research_v13_4.py): **TỒN TẠI** (5.471 bytes, 126 dòng).
4. [`experiments/vieneu_kws_research_v13_4/demo/test_live_mic_pipeline.py`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/demo/test_live_mic_pipeline.py): **TỒN TẠI** (14.847 bytes, 380 dòng).

#### 2.3 Kiểm tra toàn vẹn mã băm cổ vật v11 (Cryptographic Hashes):
- **Checkpoint v11 (`variant_v11_hard_negative_weighted_seed42.pt`):**  
  `3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2` (**Khớp 100%**)
- **Mô hình ONNX v11 (`vietnamese_kws_v11.onnx`):**  
  `54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99` (**Khớp 100%**)

#### 2.4 Cấu hình Model, Frontend, VAD và Quyết Định 5 Lớp:
- **Kiến trúc mạng:** `CorrectedVietnameseKWSNet` (138.469 tham số huấn luyện, 580 phần tử đệm BatchNorm).
- **Frontend âm học:** Log-Mel Spectrogram (`sample_rate=16000`, `n_fft=512`, `win_length=400`, `hop_length=160`, `n_mels=40`, `eps=1e-6`), kích thước đầu vào tensor: `[1, 1, 40, 101]`.
- **Không gian 5 lớp:** `0: TRAI`, `1: PHAI`, `2: LEN`, `3: XUONG`, `4: REJECT`.
- **Quy tắc quyết định 5 lớp:**  
  $$\text{top\_idx} = \operatorname{argmax}_{c \in \{0, 1, 2, 3, 4\}} P(c)$$
  Lệnh được kích hoạt khi và chỉ khi: $\text{top\_idx} < 4$ **VÀ** $P(\text{top\_idx}) \ge \theta_{\text{cmd}}$ ($\theta_{\text{cmd}} = 0.20$). Nếu $\text{top\_idx} = 4$ (`REJECT`) hoặc xác suất $< 0.20$, hệ thống dự đoán `REJECT`.

---

### 3. Truy Vết Luồng Xử Lý Microphone (Pipeline Trace & Architecture)

#### 3.1 Sơ đồ luồng dữ liệu thực tế trong mã nguồn:

```
[ Microphone Input ] (44.1 kHz / 48 kHz / 16 kHz)
        │
        ▼ (PortAudio Callback via sounddevice)
[ AudioStreamManager._audio_callback ] -> raw_queue (Queue maxsize=100)
        │
        ▼ (Worker Thread: AudioStreamManager._processing_worker)
[ Mono Downmix ] -> [ Soxr / Scipy Resampling to 16,000 Hz ] -> [ Acoustic Injection Buffer Mix ]
        │
        ▼ (callback: LiveKWSPipeline._on_new_audio_chunk)
[ AudioRingBuffer.append ] (16,000 samples FIFO for visual VU / legacy streaming)
        │
        ├────────────────────────────────────────────────────────┐
        ▼ (Chế độ mặc định: mode="segmented")                    ▼ (Chế độ cũ: mode="streaming")
[ UtteranceSegmenter.process_chunk ] (Chunk 20-30ms)     [ Sliding Window ] (Hop: 160ms = 2,560 samples)
  • Tính RMS, dB = 20*log10(RMS)                         • VAD: SpeechBandFormantVAD
  • Dynamic Noise Floor Adaptation                       • Cửa sổ phân tích: 16,000 mẫu
  • State: WAITING -> ACCUMULATING (dB >= Threshold)             │
  • Pre-roll Buffer FIFO (150ms)                                 ▼
  • Accumulate chunks until Silence >= 300ms OR Dur >= 1.2s [ _run_in_memory_inference ] (Log-Mel [1, 1, 40, 101])
        │                                                        │
        ▼ (Segment Finalized: audio_1s, seg_dur_ms)              ▼
[ Right-Pad or Truncate to exact 16,000 samples ]        [ Debounce & Cooldown Logic ]
        │                                                 • consecutive_frames = 2 (320ms)
        ▼                                                 • cooldown_sec = 0.80s (800ms LATCH)
[ _run_in_memory_inference ]                              • re-arm: max_p < 0.15 & t >= 600ms
  • Amplitude Clamping (nếu peak > 1.0)                          │
  • LogMelFrontend (Slaney Mel, shape [1, 1, 40, 101])           ▼
  • Model Forward (ONNX Runtime / PyTorch)               [ Emit DetectionEvent ]
  • 5-Class Softmax & Argmax Rule (theta_cmd = 0.20)
        │
        ▼
[ _evaluate_segmented_utterance ]
  • Phát hiện: đúng 1 kết quả phân loại
  • Nếu pred_class in TARGET_CLASSES: Emit DetectionEvent
  • Reset Segmenter state về WAITING (0ms cooldown)
```

---

#### 3.2 Phân tích chi tiết 7 yếu tố kỹ thuật theo mã nguồn thực tế:

| Câu hỏi kỹ thuật | Hiện trạng trong mã nguồn | Vị trí file & dòng code |
| :--- | :--- | :--- |
| **1. Điều kiện bắt đầu & kết thúc đoạn** | **Bắt đầu:** Năng lượng $dB \ge \text{effective\_thresh}$ với $dB = 20 \log_{10}(\text{RMS})$. Chuyển trạng thái `WAITING -> ACCUMULATING`, chèn 150ms pre-roll audio.<br>**Kết thúc:** Im lặng liên tiếp đạt $\ge 300\text{ ms}$ (`silence_interval_ms = 300`) **HOẶC** tổng độ dài vượt quá $1.2\text{ s}$ (`max_segment_duration_ms = 1200`). | [`live_mic_engine.py:217-222`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/demo/live_mic_engine.py#L217-L222)<br>[`live_mic_engine.py:236-240`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/demo/live_mic_engine.py#L236-L240) |
| **2. Ngưỡng năng lượng & khoảng lặng thực tế** | **Ngưỡng năng lượng tĩnh:** $-48.0\text{ dB}$ (`energy_threshold_db`).<br>**Ngưỡng thích ứng:** $\text{effective\_thresh} = \max(-48.0, \min(\text{noise\_floor} + 5.0, -32.0))$.<br>**Khoảng lặng kết thúc:** Đúng $300\text{ ms}$ (4.800 mẫu ở 16kHz).<br>**Lọc âm ngắn:** Bắt buộc thời gian nói có ích $\ge 120\text{ ms}$. | [`live_mic_engine.py:135-147`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/demo/live_mic_engine.py#L135-L147)<br>[`live_mic_engine.py:198-200`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/demo/live_mic_engine.py#L198-L200) |
| **3. Cửa sổ, bước trượt & ghép đoạn** | **Mode Segmented:** Không dùng bước trượt. Tích lũy toàn bộ các chunk từ onset đến offset. Nếu tổng độ dài $< 16.000$ mẫu, đệm số 0 về bên phải (`right-pad zero`). Nếu $\ge 16.000$ mẫu, cắt cụt lấy đúng 16.000 mẫu đầu.<br>**Mode Streaming:** Cửa sổ 16.000 mẫu (1.0s), bước trượt $160\text{ ms}$ (2.560 mẫu). | [`live_mic_engine.py:246-250`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/demo/live_mic_engine.py#L246-L250)<br>[`live_mic_engine.py:746-757`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/demo/live_mic_engine.py#L746-L757) |
| **4. Cơ chế Cooldown, Debounce & Re-arm** | **Mode Segmented:** **Không có cooldown thời gian (0ms cooldown)!** Sau khi phát sự kiện, segmenter lập tức chuyển về `"WAITING"`, sẵn sàng nhận từ tiếp theo.<br>**Mode Streaming:** Debounce 2 khung ($320\text{ ms}$); **Cooldown cứng 800ms** (`cooldown_sec = 0.80s`). Khóa chốt `is_armed = False` và chỉ re-arm khi $\max P(\text{cmd}) < 0.15$ và thời gian trôi qua $\ge 600\text{ ms}$. | [`live_mic_engine.py:254-258`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/demo/live_mic_engine.py#L254-L258)<br>[`live_mic_engine.py:786-805`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/demo/live_mic_engine.py#L786-L805) |
| **5. Số sự kiện 1 đoạn có thể phát ra** | **Chính xác tối đa 1 sự kiện (hoặc 0 nếu bị REJECT).** Hàm `_evaluate_segmented_utterance()` chỉ thực thi suy luận 1 lần trên 1 tệp âm thanh 1.0s đầu vào. Hoàn toàn không có cơ chế bóc tách đa sự kiện bên trong một đoạn âm thanh. | [`live_mic_engine.py:646-695`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/demo/live_mic_engine.py#L646-L695) |
| **6. Giữ xác suất theo thời gian vs Phân loại đoạn** | **Chỉ phân loại từng đoạn độc lập (Isolated Segment Classification).** Không có mô hình chuỗi thời gian (không có CTC, HMM, hay Beam Search). Toàn bộ phổ Log-Mel của đoạn được xử lý như một "bức ảnh tĩnh" qua mạng CNN 2D. | [`live_mic_engine.py:696-742`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/demo/live_mic_engine.py#L696-L742) |
| **7. Nguyên nhân lệnh `LEN` thứ 2 bị mất** | **Xem phân tích chi tiết tại Mục 4.** | — |

---

### 4. Nguyên Nhân Đã Xác Minh & Giả Thuyết Kỹ Thuật (Root-Cause Decomposition)

Khi người dùng nói chuỗi khẩu lệnh dồn dập: `LÊNLENPHAI` $\to$ yêu cầu nhận diện 3 sự kiện: `LEN`, `LEN`, `PHAI`. Thực nghiệm phân tích mã nguồn và kiểm thử phát lại âm thanh xác nhận **3 nguyên nhân kỹ thuật cốt lõi**:

#### 4.1 Nguyên Nhân Đã Xác Minh 1: Lỗi Gộp Phân Đoạn VAD (Segmentation Merge Error)
- **Cơ chế:** Một từ khẩu lệnh tiếng Việt như `LÊN` có độ dài phát âm tự nhiên khoảng $350 - 450\text{ ms}$. Khi người dùng nói liền mạch `LÊN LÊN PHẢI`, khoảng ngắt âm học giữa hai từ `LÊN` và `LÊN` chỉ kéo dài từ $50\text{ ms} - 180\text{ ms}$.
- **Điểm nghẽn mã nguồn:** Tại [`live_mic_engine.py:135, 236`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/demo/live_mic_engine.py#L135), `UtteranceSegmenter` đặt thông số cứng:
  $$\text{silence\_threshold\_samples} = \frac{300}{1000} \times 16000 = 4.800\text{ mẫu} \quad (300\text{ ms})$$
- **Hệ quả:** Vì khoảng ngắt âm giữa 2 từ ($50-180\text{ ms}$) nhỏ hơn ngưỡng $300\text{ ms}$, bộ phân đoạn **không phát hiện điểm kết thúc của từ thứ nhất**. Bộ tích lũy (`self.accumulated_chunks`) tiếp tục giữ trạng thái `ACCUMULATING` và nuốt luôn từ `LÊN` thứ hai vào cùng một khối âm thanh.
- Khi người dùng nói tiếp từ `PHẢI`, tổng thời lượng vượt quá trần $1.2\text{ s}$ (`max_segment_duration_ms = 1200`). Bộ phân đoạn bị ép ngắt khẩn cấp, cắt lấy 1.0s đầu tiên (`full_segment[:16000]`).
- Đoạn âm thanh 1.0s này chứa cả 2 từ `LÊN` và `LÊN` dồn nén. Do mô hình chỉ xuất tối đa 1 nhãn, lệnh `LEN` thứ hai bị nuốt hoàn toàn!

#### 4.2 Nguyên Nhân Đã Xác Minh 2: Khóa Chốt Cooldown 800ms Trong Chế Độ Streaming (Debounce Lockout)
Nếu hệ thống được cấu hình chạy ở chế độ trượt thời gian thực (`mode="streaming"`):
- Tại [`live_mic_engine.py:537, 803`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/demo/live_mic_engine.py#L537), cơ chế chống kích hoạt trùng lặp áp đặt:
  $$\text{cooldown\_sec} = 0.80\text{ s} \quad (800\text{ ms})$$
- Sau khi lệnh `LEN` đầu tiên kích hoạt, hệ thống gán `is_armed = False`.
- Lệnh `LEN` thứ hai xuất hiện sau đó $350 - 500\text{ ms}$, hoàn toàn nằm trong thời gian cấm 800ms.
- Điều kiện re-arm ([`live_mic_engine.py:787`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/demo/live_mic_engine.py#L787)) đòi hỏi $P(\text{cmd}) < 0.15$. Nhưng vì từ `LEN` thứ hai đang phát âm, xác suất $P(\text{LEN})$ vẫn duy trì $> 0.20$, tạo ra trạng thái **khóa chốt bế tắc (deadlock)** khiến `is_armed` không thể bật lại.

#### 4.3 Nguyên Nhân Đã Xác Minh 3: Bản Chất Mô Hình Là Bộ Phân Loại Đơn Từ (Isolated Keyword Classifier)
- Mô hình mạng tích chập `CorrectedVietnameseKWSNet` được huấn luyện trên các mẫu đơn từ 1.0s đã căn chỉnh.
- Đầu ra của mạng là một vector softmax 5 phần tử: $[P_{\text{TRAI}}, P_{\text{PHAI}}, P_{\text{LEN}}, P_{\text{XUONG}}, P_{\text{REJECT}}]$.
- Mô hình **không có cấu trúc CTC (Connectionist Temporal Classification) hay Transducer**, không thể giải mã chuỗi ký tự hay chuỗi nhãn $[y_1, y_2, \dots]$ theo thời gian từ một đoạn âm thanh chứa nhiều từ.
- Khi nhận một đoạn âm thanh chứa 2 từ nói dính nhau, mô hình hoặc bị quá tải đặc trưng dẫn đến dự đoán `REJECT`, hoặc chỉ đưa ra nhãn cho từ có năng lượng phổ chiếm ưu thế nhất.

---

### 5. Kiểm Tra Thực Nghiệm Baseline & Replay Âm Thanh

#### 5.1 Kết quả chạy bộ test hiện có:
Thực thi lệnh:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_vieneu_research_v13_4.py experiments/vieneu_kws_research_v13_4/demo/test_live_mic_pipeline.py -v
```

**Kết quả tổng hợp:** `19 PASSED, 1 FAILED (thời gian: 4.29s)`
- **15 / 15 tests của `test_live_mic_pipeline.py` PASSED:**
  - `test_three_separate_speech_segments_in_order`: **PASSED** (vì test dùng `silence_gap = 350ms` $> 300\text{ ms}$).
  - `test_repeated_identical_commands_in_separate_segments`: **PASSED** (cũng dùng `silence_gap = 350ms`).
  - Toàn bộ test hiện có đều chủ động chèn khoảng lặng 350ms, do đó **chưa từng kiểm thử trường hợp nói liền mạch thực tế**.
- **1 test FAILED trong `tests/test_vieneu_research_v13_4.py`:**
  - `test_infer_cli_parity_and_latency` (dòng 93):  
    `AssertionError: PyTorch latency 102.11 ms exceeded 50ms SLA`.  
    *Nguyên nhân:* Lần chạy đầu tiên (cold start) của PyTorch trên CPU mất 102.11ms do nạp thư viện và khởi tạo luồng; các lần chạy sau đạt 12-15ms. Độ tương đồng suy luận giữa PyTorch và ONNX đạt 100% (Parity PASS). Tuân thủ đúng nguyên tắc: **Không sửa mã nguồn chỉ để làm test PASS giả tạo**.

#### 5.2 Thực nghiệm phát lại âm thanh (Audio Replay Experiment) với chuỗi `LÊNLENPHAI`:
Sử dụng các đoạn phát âm chuẩn từ tập dữ liệu thẩm định độc lập (`v13_conf_cmd_len` và `v13_conf_cmd_phai`), ghép thành chuỗi `LEN -> gap -> LEN -> gap -> PHAI` với các khoảng ngắt $400\text{ ms}$, $250\text{ ms}$, $150\text{ ms}$, $50\text{ ms}$, $0\text{ ms}$.

| Khoảng ngắt (Inter-word Gap) | Mode Segmented (v11 Default: Silence=300ms) | Profile B (SpeechBandRatioVAD: Hangover=140ms) | Mode Streaming (Sliding Window: Cooldown=800ms) | Đánh Giá Kỹ Thuật |
| :---: | :---: | :---: | :---: | :--- |
| **$400\text{ ms}$** | **3 sự kiện:** `['LEN', 'LEN', 'PHAI']` | **3 sự kiện:** `['LEN', 'LEN', 'PHAI']` | **1 sự kiện:** `['LEN']` | Segmented đạt; Streaming nuốt do cooldown 800ms |
| **$250\text{ ms}$** | **3 sự kiện:** `['LEN', 'LEN', 'PHAI']` | **3 sự kiện:** `['LEN', 'LEN', 'PHAI']` | **1 sự kiện:** `['LEN']` | Segmented đạt (năng lượng rơi chạm sàn); Streaming nuốt |
| **$150\text{ ms}$** | **2 sự kiện:** `['LEN', 'PHAI']` <br>*(LEN thứ nhất dài 1210ms)* | **3 sự kiện:** `['LEN', 'LEN', 'PHAI']` | **1 sự kiện:** `['LEN']` | **LỆNH `LEN` THỨ HAI BỊ NUỐT HOÀN TOÀN TRÊN V11** |
| **$50\text{ ms}$** | **1 sự kiện:** `['LEN']` <br>*(LEN dài 1210ms)* | **3 sự kiện:** `['LEN', 'LEN', 'PHAI']` | **1 sự kiện:** `['LEN']` | **V11 MẤT CẢ `LEN` THỨ HAI LẪN `PHAI`** |
| **$0\text{ ms}$ (Nói liền mạch)**| **0 sự kiện:** `[]` *(Bị REJECT toàn bộ)* | **0 sự kiện:** `[]` *(Bị REJECT toàn bộ)* | **1 sự kiện:** `['LEN']` | **TOÀN BỘ CÁC BỘ PHÂN ĐOẠN ĐỀU THẤT BẠI HOÀN TOÀN** |

> **Bằng chứng thực nghiệm không thể chối cãi:**  
> Tại khoảng ngắt tự nhiên $\le 150\text{ ms}$, pipeline mặc định v11 nuốt mất lệnh `LEN` thứ hai. Khi nói liền không ngắt ($0\text{ ms}$), hệ thống không bắt được bất kỳ lệnh nào.

---

### 6. Thiết Kế Benchmark Cho Yêu Cầu Giảng Viên (Evaluation Battery Design)

Để đánh giá khoa học và khách quan năng lực nhận diện chuỗi lệnh dồn dập trước và sau khi cải tiến, đề xuất bộ test chuẩn hóa gồm 5 ca kiểm thử:

#### 6.1 Bảng Đặc Tả 5 Ca Kiểm Thử Tối Thiểu:

| STT | Mã Ca Kiểm Thử | Chuỗi Khẩu Lệnh | Cấu Trúc Khoảng Ngắt (Gap) | Ground Truth Events (Kỳ Vọng) | Hành Vi Cần Kiểm Định |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **1** | `TC_CONT_01` | `LEN → LEN → PHAI` | Ngắt ngắn: $100\text{ ms}$ và $0\text{ ms}$ | $E_1$: `LEN`, $E_2$: `LEN`, $E_3$: `PHAI` | Tách biệt hai lệnh trùng lặp đứng cạnh nhau không bị nuốt lệnh. |
| **2** | `TC_CONT_02` | `TRAI → LEN → XUONG` | Ngắt ngắn: $100\text{ ms}$ và $0\text{ ms}$ | $E_1$: `TRAI`, $E_2$: `LEN`, $E_3$: `XUONG` | Phân đoạn chuỗi 3 lệnh khác nhau không bị gộp thành từ sai (không bị thành `PHAI`). |
| **3** | `TC_CONT_03` | `LEN` kéo dài | Phát âm ngân dài $800 - 900\text{ ms}$ | Đúng $1$ sự kiện: `LEN` | Chống kích hoạt lặp; không được tách 1 từ kéo dài thành 2 lệnh. |
| **4** | `TC_CONT_04` | `LEN → LEN → LEN` | Ba lần `LEN` liên tiếp (gap $150\text{ ms}$) | Đúng $3$ sự kiện: `LEN`, `LEN`, `LEN` | Đếm chính xác số lượng kích hoạt lặp cùng nhãn liên tiếp. |
| **5** | `TC_CONT_05` | Đa khoảng ngắt & Đa mức nhiễu | Gaps: $0, 50, 100, 150, 250, 400\text{ ms}$<br>SNR: Clean, $15\text{ dB}$, $10\text{ dB}$, $5\text{ dB}$ | Khớp chính xác nhãn và mốc thời gian | Đánh giá độ bền âm học và ranh giới suy giảm năng lực. |

#### 6.2 Định Dạng Bảng Ghi Nhãn Tham Chiếu (Ground Truth Event Format):
Mỗi sự kiện chuẩn trong tệp âm thanh kiểm thử phải được định nghĩa tường minh với mốc thời gian tuyệt đối:
```json
[
  {"event_id": "ev_01", "label": "LEN", "start_sec": 0.350, "end_sec": 0.720, "duration_sec": 0.370},
  {"event_id": "ev_02", "label": "LEN", "start_sec": 0.850, "end_sec": 1.230, "duration_sec": 0.380},
  {"event_id": "ev_03", "label": "PHAI", "start_sec": 1.350, "end_sec": 1.710, "duration_sec": 0.360}
]
```

#### 6.3 Quy Tắc Đối Khớp Sự Kiện Dự Đoán (Bipartite Event Matching Protocol):
1. **Cửa sổ dung sai thời gian (Temporal Tolerance Window):** $\Delta t = \pm 0.50\text{ s}$ tính từ tâm sự kiện ground truth:
   $$|t_{\text{pred}} - t_{\text{gt\_center}}| \le 0.50\text{ s}$$
2. **Nguyên tắc đối khớp 1-1 (No Duplicate Reuse):** Mỗi sự kiện ground truth chỉ được ghép với tối đa 1 sự kiện dự đoán đúng nhãn đầu tiên rơi vào cửa sổ dung sai. Các lần kích hoạt sau trong cùng cửa sổ bị tính là **Duplicate Triggers (Lỗi lặp)**.
3. **Chèn sai (False Insertion):** Sự kiện dự đoán không ghép được với bất kỳ ground truth nào.
4. **Bỏ lỡ (Missed Event):** Ground truth không có dự đoán nào ghép trúng.

#### 6.4 Hệ Thống Chỉ Số Nghiệm Thu (Acceptance Metrics):
- **Command Event Recall:** $\frac{\text{Matched Events}}{\text{Total Ground Truth Events}} \ge 85\%$
- **Command Event Precision:** $\frac{\text{Matched Events}}{\text{Total Detections}} \ge 85\%$
- **Exact Sequence Match Rate:** Tỷ lệ chuỗi nhận diện đúng 100% cả về số lượng và thứ tự $\ge 70\%$.
- **Duplicate Trigger Rate:** $\le 2\%$.

---

### 7. Đề Xuất Thay Đổi Nhỏ Nhất Cần Thử Nghiệm Ở Nhánh Riêng

Để giải quyết triệt để nút thắt mà vẫn **bảo toàn 100% mô hình v11 và tag phát hành hiện tại**, đề xuất tạo nhánh phát triển độc lập: `feature/continuous-kws-segmentation`.

#### Hướng Giải Pháp A: Cải Tiến Bộ Phân Đoạn (Energy Valley & Dip Segmentation)
- **Bản chất:** Giữ nguyên mô hình v11, chỉ thay đổi thuật toán tách phân đoạn trước khi đưa vào mô hình.
- **Cơ chế:** Thay vì chờ $300\text{ ms}$ im lặng tuyệt đối (vốn không tồn tại khi nói liền), thuật toán theo dõi **độ sụt giảm năng lượng cục bộ (Energy Dip / Valley Detector)** kết hợp tỷ số dải tần tiếng nói (như trong Phase 15).
- Trong tiếng Việt, giữa hai âm tiết liền nhau luôn có một điểm trũng âm học tự nhiên kéo dài $40 - 80\text{ ms}$ (do sự khép khẩu hình của phụ âm đầu hoặc thanh hầu đóng). Bắt được điểm trũng này sẽ cho phép ngắt phân đoạn ngay cả khi không có khoảng lặng dài.

#### Hướng Giải Pháp B: Chuyển Sang Streaming với Bộ Giải Mã Phi Cực Đại (Continuous NMS Decoder)
- **Bản chất:** Bỏ cơ chế `cooldown_sec = 0.80s` kiểu chốt cứng trong chế độ streaming.
- **Cơ chế:** Cho cửa sổ trượt $1.0\text{ s}$ bước trượt $80\text{ ms}$ xuất liên tục xác suất $P_t(\text{LEN})$. Khi hai từ `LEN` nói liền nhau, đường cong xác suất $P_t(\text{LEN})$ sẽ tạo ra **hai đỉnh cục bộ riêng biệt (Two Local Peaks)** vượt ngưỡng $0.20$.
- Áp dụng thuật toán **Non-Maximum Suppression (NMS)** trên trục thời gian với bán kính triệt tiêu $250\text{ ms}$ để bắt đúng 2 đỉnh tương ứng với 2 sự kiện riêng biệt.

---

### 8. Kết Luận Ngắn Gọn (Executive Audit Conclusion)

> **NÚT THẮT CHÍNH NẰM Ở ĐÂU?**  
> Nút thắt khiến hệ thống bỏ sót lệnh `LEN` thứ hai trong chuỗi `LÊNLENPHAI` nằm ở **SỰ KẾT HỢP GIỮA BỘ PHÂN ĐOẠN (SEGMENTATION) VÀ BỘ GIẢI MÃ SỰ KIỆN (EVENT DECODING)**:
> 1. **Ở chế độ phân đoạn (`mode="segmented"`):** Nút thắt nằm ở **Segmentation (VAD)**. Ngưỡng ngắt khoảng lặng $300\text{ ms}$ quá lớn so với khoảng nghỉ tự nhiên ($50-180\text{ ms}$) của khẩu lệnh nói liền, dẫn đến lỗi gộp 2 từ thành 1 phân đoạn 1.2s kịch trần; sau đó bộ phân loại chỉ có thể xuất tối đa 1 nhãn duy nhất.
> 2. **Ở chế độ dòng (`mode="streaming"`):** Nút thắt nằm ở **Event Decoding**. Cơ chế khóa chốt Cooldown cứng $800\text{ ms}$ (`cooldown_sec = 0.80s`) lập tức triệt tiêu và vô hiệu hóa mọi lệnh tiếp theo xuất hiện trong vòng 0.8 giây.
> 3. **Giới hạn của Model Output:** Mô hình CNN v11 là bộ phân loại đơn từ (Isolated Classifier), không có cơ chế CTC giải mã chuỗi, do đó không thể "cứu" được một phân đoạn âm thanh khi nó đã bị VAD gộp lỗi ngay từ đầu vào.
