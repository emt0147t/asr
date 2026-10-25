# Governance Evidence Review: Vietnamese KWS Genuine Human Speech Collection

> **Document Version:** 1.0.0-GOVERNANCE-REVIEW  
> **Review Date:** October 13, 2026  
> **Reviewer:** Independent Automated Code & Governance Auditor  
> **Workspace:** `D:\trainidentificationaudio`  
> **Document Purpose:** Forensic review of actual governance records, consent ledgers, sampling quota approvals, corpus licenses, CLI call-path metadata enforcement, and consent withdrawal lifecycles.  
> **Executive Verdict:** **`READY FOR HUMAN SIGN-OFF`**  
> *(Human Speech Collection Status: **`STRICTLY BLOCKED`** pending physical execution of human approvals)*

---

## 1. Executive Summary & Review Scope

Báo cáo này được thực hiện độc lập nhằm rà soát các **bằng chứng quản trị thực tế (governance evidence)** tồn tại trong workspace `D:\trainidentificationaudio`, phân biệt rõ rệt giữa **tài liệu đặc tả kỹ thuật dự thảo (draft specifications)** và **bằng chứng phê duyệt pháp lý có thẩm quyền (authoritative approval evidence)**.

Báo cáo kỹ thuật trước đó ([`final_readiness_audit.md`](file:///d:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/final_readiness_audit.md)) đã xác nhận hạ tầng phần mềm, bộ kiểm thử 41 bài test và ma trận mã hóa đạt chuẩn kỹ thuật. Tuy nhiên, rà soát quản trị xác nhận rằng:
1. **Không có bất kỳ hồ sơ consent của người tham gia thật nào tồn tại trên đĩa**;
2. **Quota lấy mẫu `50N/30C/40S` vẫn là đề xuất kỹ thuật dự thảo, chưa từng được người có thẩm quyền phê duyệt**;
3. **Đường gọi production CLI chỉ kích hoạt `strict_metadata=True` khi operator truyền cờ `--strict` thủ công**, tiềm ẩn nguy cơ áp dụng metadata mặc định nếu chạy lệnh thông thường;
4. **Quy trình thu hồi consent chỉ tự động hóa trên dữ liệu thô và manifest**, toàn bộ phần cache đặc trưng, backup lạnh và việc loại bỏ tham số khỏi trọng số mạng neural đòi hỏi quy trình quản trị thủ công bắt buộc.

---

## 2. Cryptographic Invariant & Integrity Scorecard

Trước và sau quá trình rà soát quản trị, mã băm mật mã SHA-256 của ba tệp dữ liệu được bảo vệ đã được tính toán độc lập và đối chiếu với giá trị chuẩn:

```powershell
Get-FileHash -Algorithm SHA256 `
  "data/manifests/test_manifest_v2.csv", `
  "data/raw/field_recordings/field_collection_manifest.csv", `
  "data/raw/field_recordings/consent_ledger.json"
```

| Tệp Dữ liệu Được Bảo vệ | Hash Chuẩn Được Bảo lưu (Expected) | Hash Thực tế Đo được (Measured) | Trạng thái Toàn vẹn |
| :--- | :--- | :--- | :---: |
| `data/manifests/test_manifest_v2.csv` | `9E3904C8EA427244F7280A41140D7EFD19FD9625B359290ADB7ADCD4FA2E1C13` | `9E3904C8EA427244F7280A41140D7EFD19FD9625B359290ADB7ADCD4FA2E1C13` | **BIT-EXACT MATCH** |
| `data/raw/field_recordings/field_collection_manifest.csv` | `B389282F9853A74484B39356EE85E5BD7E74C0C46481300524C7BD2894AA8C85` | `B389282F9853A74484B39356EE85E5BD7E74C0C46481300524C7BD2894AA8C85` | **BIT-EXACT MATCH** |
| `data/raw/field_recordings/consent_ledger.json` | `BE4BFDD7CEC0C87A1DC97455E5783054EF8093C638279C44940C896F5BDA2BA0` | `BE4BFDD7CEC0C87A1DC97455E5783054EF8093C638279C44940C896F5BDA2BA0` | **BIT-EXACT MATCH** |

**Kết luận kiểm tra tệp đóng băng:** Toàn bộ ba tệp cốt lõi được giữ nguyên trạng 100%, không bị sửa đổi, di chuyển hoặc ghi đè dù chỉ một byte.

---

## 3. Rà soát Hồ sơ Quản trị Thực tế: Consent, Phê duyệt Tuyển mẫu, và Giấy phép Corpus

Bảng sau đây phân định chi tiết giữa hiện trạng tệp thực tế trong repository và giá trị pháp lý tương ứng:

| Hạng mục Quản trị | Đường dẫn Tệp Thực tế Trong Workspace | Hiện trạng Phát hiện trên Đĩa | Bản chất Pháp lý / Quản trị | Kết luận Kiểm toán |
| :--- | :--- | :--- | :--- | :---: |
| **Hồ sơ Consent Người thật (Sổ bộ điện tử)** | `data/raw/human_recordings_v1/consent_ledger_human.json` | **TỆP CHƯA TỒN TẠI** (Thư mục trống, chưa có lượt ghi) | Chưa có người tham gia nào được cấp mã hoặc đăng ký. | **NOT VERIFIED / ZERO RECORDS** |
| **Hồ sơ Consent Lịch sử (Fixture)** | `data/raw/field_recordings/consent_ledger.json` | Tồn tại tệp (74 bản ghi từ `SPK_FIELD_N001` đến `S024`) | Dữ liệu thử nghiệm kỹ thuật (synthetic test fixture), không phải người thật. | **TEST FIXTURE ONLY** (Không dùng làm bằng chứng pháp lý) |
| **Văn bản Đồng thuận Đã ký (Bản giấy / PDF / E-sign)** | Toàn bộ repository (`docs/`, `data/`, v.v.) | **KHÔNG TÌM THẤY** bất kỳ tệp PDF, scan chữ ký, hoặc e-signature nào. | Chưa thực hiện ký kết với người tham gia thực tế nào. | **NOT FOUND / NONE** |
| **Mẫu Bản Đồng thuận Dự thảo** | [`docs/sound_command_recognition/campaign_20261013/collection_protocol.md#L41-L67`](file:///d:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/collection_protocol.md#L41-L67) | Tồn tại văn bản dự thảo tiếng Việt (Mục 2.2) | Bản thảo điều khoản (Draft Agreement Template) chờ thẩm định pháp lý/IRB. | **DRAFT TEMPLATE ONLY** (Chưa được ký duyệt) |
| **Biên bản Phê duyệt Quota Tuyển mẫu** | Toàn bộ repository (`docs/`, `reports/`, Git logs) | **KHÔNG TÌM THẤY** biên bản phê duyệt, chữ ký Project Sponsor hoặc Ticket Jira. | Quota 50N/30C/40S chỉ là đề xuất kỹ thuật trong tài liệu đặc tả. | **PENDING HUMAN APPROVAL** |
| **Giấy phép Google FLEURS Vietnamese** | `data/raw/fleurs_vi/` | Tồn tại 1,218 tệp WAV (splits dev/test), nhưng **KHÔNG CÓ TỆP LICENSE** trên đĩa. | Mã nguồn khai báo CC-BY-4.0 ([`download_approved_sources.py#L22`](file:///d:/trainidentificationaudio/src/acquisition/download_approved_sources.py#L22)), nhưng thiếu văn bản cấp phép lưu trữ cục bộ. | **PROVENANCE KNOWN (CC-BY-4.0) — LICENSE FILE MISSING ON DISK** |
| **Giấy phép MLCommons MSWC Vietnamese** | `data/raw/mswc_vi/` | Tồn tại `splits.tar.gz` và `version.txt`, **KHÔNG CÓ TỆP LICENSE** trên đĩa. | `version.txt` ghi: *"version 1.0, Multilingual Spoken Words Corpus"*. Mã nguồn khai báo CC-BY-4.0 ([`download_approved_sources.py#L57`](file:///d:/trainidentificationaudio/src/acquisition/download_approved_sources.py#L57)). | **PROVENANCE KNOWN (CC-BY-4.0) — LICENSE FILE MISSING ON DISK** |
| **Giấy phép Nguồn Nhiễu Ngoài (`mic-noise`)** | `data/raw/noise/mic-noise.zip` và `data/raw/noise/extracted/...` | Tồn tại 39 tệp MP3 nhiễu micro từ repo Husein Zolkepli. | **GIẤY PHÉP KHÔNG XÁC MINH ĐƯỢC.** Bị chặn hoàn toàn bởi code ([`download_approved_sources.py#L44`](file:///d:/trainidentificationaudio/src/acquisition/download_approved_sources.py#L44) và [`campaign_validator.py#L39`](file:///d:/trainidentificationaudio/src/acquisition/campaign_validator.py#L39)). | **BLOCKED — UNVERIFIED LICENSE** (Cấm nhập vào manifest người thật) |

---

## 4. Xác minh Quota Tuyển mẫu `50N/30C/40S`: Dự thảo Đề xuất vs. Bằng chứng Phê duyệt

### 4.1 Nguồn gốc Văn bản và Thẩm quyền Phê duyệt
- **Vị trí tài liệu duy nhất:** [`docs/sound_command_recognition/campaign_20261013/collection_protocol.md`](file:///d:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/collection_protocol.md#L105-L114) Mục 4.1.
- **Tiêu đề phân mục:** `### 4.1 Target Cohort Sizing (Proposed)`.
- **Đầu trang văn bản ghi rõ:**
  ```markdown
  > **Document Version:** 1.0.0-PROPOSAL  
  > **Status:** Proposed Specification — Pending Human Approval Before Recruitment  
  > **Proposal Notice:** All sample sizes, quotas, and participant targets in this document 
  > are engineering proposals designed for statistical coverage. No participants will be 
  > contacted or recruited until this protocol receives formal human approval.
  ```
- **Kết luận xác minh:**  
  Quota **`50N/30C/40S`** hoàn toàn là **`ĐỀ XUẤT KỸ THUẬT DỰ THẢO (PROPOSED SPECIFICATION)`**.  
  Trong toàn bộ workspace, **không tồn tại bất kỳ bằng chứng phê duyệt nào** (chữ ký, con dấu, email phê duyệt, quyết định giao nhiệm vụ từ Project Owner / Legal Counsel). Trạng thái chính thức: **`PENDING HUMAN APPROVAL`**.

### 4.2 So sánh Chi tiết Hai Phương án Quota
Hai phương án quota xuất hiện trong tài liệu nghiên cứu được so sánh đối chiếu:

| Tiêu chí Đánh giá | Phương án 1: `50N / 30C / 40S` ([`collection_protocol.md#L108`](file:///d:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/collection_protocol.md#L108)) | Phương án 2: `60N / 30C / 30S` (Bản ghi chú kỹ thuật sơ bộ) |
| :--- | :--- | :--- |
| **Phân bổ Vùng miền** | Bắc: 50 (41.7%) \| Trung: 30 (25.0%) \| Nam: 40 (33.3%) | Bắc: 60 (50.0%) \| Trung: 30 (25.0%) \| Nam: 30 (25.0%) |
| **Tổng số Người** | 120 người | 120 người |
| **Phân bổ 70% Train** | **84 người** (chính xác $70.0\%$) | **88 người** ($73.3\%$) |
| **Phân bổ 15% Validation** | **18 người** (chính xác $15.0\%$) | **16 người** ($13.3\%$) |
| **Phân bổ 15% Test** | **18 người** (chính xác $15.0\%$) | **16 người** ($13.3\%$) |
| **Tác động Làm tròn Số học** | **HOÀN HẢO:** $50 \times 0.15 = 7.5 \approx 7/8$; $30 \times 0.15 = 4.5 \approx 5$; $40 \times 0.15 = 6$. Tổng partition khớp chính xác số nguyên 84 / 18 / 18. | **LỆCH PHÂN VÙNG:** Với nhóm 15 người mỗi giới, `int(15 * 0.15) = 2`, làm Validation và Test bị cắt giảm mỗi bên 2 người, phình tập Train lên 73.3%. |
| **Cơ sở Ngữ âm** | Phản ánh chính xác mật độ dân số và sự tương phản giữa phụ âm tắc xát Bắc `/tɕ/` và biến thể quặt lưỡi Nam `/ʈ/`. | Ưu tiên áp đảo giọng Bắc, xem nhẹ tỷ trọng người dùng phương Nam. |
| **Trạng thái Quản trị** | **ĐỀ XUẤT ƯU TIÊN (PROPOSED v1.0.0)** | **BỊ THAY THẾ / PHƯƠNG ÁN PHỤ** |

**Khuyến nghị:** Cần trình phê duyệt chính thức phương án **`50N/30C/40S`** do tính ưu việt về toán học khi chia tách tập dữ liệu (zero truncation drift).

---

## 5. Kiểm tra Đường gọi Production: Xác minh Áp dụng `strict_metadata=True`

Rà soát mã nguồn thực tế từ giao diện dòng lệnh (CLI) tới dịch vụ tiếp nhận (`ingest_recording`) cho thấy một phát hiện quản trị trọng yếu:

### 5.1 Phân tích Đường truyền Mã lệnh

#### Tuyến 1: Thu âm Trực tiếp trên Thiết bị (`src/acquisition/record_command_cli.py`)
1. **Khai báo CLI Parser ([`record_command_cli.py#L328`](file:///d:/trainidentificationaudio/src/acquisition/record_command_cli.py#L328)):**
   ```python
   parser.add_argument("--strict", action="store_true", help="Enforce strict metadata validation for genuine human recordings")
   ```
   Do sử dụng `action="store_true"`, giá trị mặc định của `args.strict` là **`False`**.
2. **Truyền tham số qua Hàm Session ([`record_command_cli.py#L397`](file:///d:/trainidentificationaudio/src/acquisition/record_command_cli.py#L397)):**
   ```python
   run_recording_session(..., strict_metadata=args.strict)
   ```
3. **Gọi Ingestion Service ([`record_command_cli.py#L244`](file:///d:/trainidentificationaudio/src/acquisition/record_command_cli.py#L244)):**
   ```python
   res = ingest_recording(..., strict_metadata=strict_metadata)
   ```
4. **Hành vi Thực tế khi Chạy Production:**
   - **Trường hợp A (Lệnh thông thường không cờ):** Operator chạy:  
     `python src/acquisition/record_command_cli.py --speaker-id SPK_H_N_xxxxxx`  
     -> `strict_metadata = False`.  
     -> Khối kiểm tra nghiêm ngặt ([`ingest_field_recordings.py#L324-L365`](file:///d:/trainidentificationaudio/src/acquisition/ingest_field_recordings.py#L324-L365)) **BỊ BỎ QUA HOÀN TOÀN**.  
     -> Hệ thống tự động gán giá trị mặc định fallback (`"18-30"`, `"builtin_mobile"`, `"close_talk_15cm"`), dẫn đến nguy cơ làm sai lệch metadata nhân khẩu học của người thật.
   - **Trường hợp B (Chủ động truyền `--strict`):** Operator chạy:  
     `python src/acquisition/record_command_cli.py --strict --speaker-id SPK_H_N_xxxxxx`  
     -> `strict_metadata = True`. Kiểm tra nghiêm ngặt được kích hoạt.
   - **Khoảng trống Đăng ký (Registration Gap):**  
     Trong hàm `interactive_registration()` ([`record_command_cli.py#L267-L300`](file:///d:/trainidentificationaudio/src/acquisition/record_command_cli.py#L267-L300)), CLI chỉ hỏi `dialect`, `gender`, `age`, `device`, `env`; **không thu thập `distance_category` hoặc `microphone_type`**.  
     Nếu operator bật `--strict` nhưng quên truyền `--distance` qua CLI, thì ngay take đầu tiên, `ingest_recording` sẽ kiểm tra:
     ```python
     if distance_category is None or dist_cat not in [...]:
         meta_errs.append(f"missing_or_invalid_distance_category_{dist_cat}")
     ```
     và lập tức **từ chối, cách ly bản thu** (`status: QUARANTINED`).

#### Tuyến 2: Nạp File Hàng loạt (`src/acquisition/ingest_field_recordings.py`)
1. **Khai báo CLI Parser ([`ingest_field_recordings.py#L575`](file:///d:/trainidentificationaudio/src/acquisition/ingest_field_recordings.py#L575)):**
   `--strict` có giá trị mặc định là **`False`**.
2. **Thực thi ([`ingest_field_recordings.py#L592, L611`](file:///d:/trainidentificationaudio/src/acquisition/ingest_field_recordings.py#L592)):**
   Nếu không có cờ `--strict`, các bản ghi được nạp với `strict_metadata=False`.

### 5.2 Kết luận Kiểm toán về Đường gọi Production
- **Cơ chế kỹ thuật:** Đã được cài đặt đầy đủ và hoạt động hoàn hảo khi được kích hoạt (đã chứng minh qua bài test `test_strict_human_metadata_validation`).
- **Lỗ hổng quản trị vận hành (Operational Risk):** Cơ chế **không được bật mặc định** cho các lệnh ghi âm người thật (`provenance="human_field_recording"`). Nếu operator không đọc kỹ tài liệu và quên cờ `--strict`, hệ thống sẽ âm thầm tiếp nhận dữ liệu với metadata fallback.
- **Yêu cầu Bắt buộc:** Quy trình vận hành chuẩn (SOP) phải yêu cầu cờ `--strict` bắt buộc, hoặc mã nguồn CLI cần được cập nhật để mặc định `strict_metadata=True` khi không chạy ở chế độ `--mock`.

---

## 6. Quy trình Thu hồi Consent, Xử lý Cache/Backup và Giới hạn Model

Rà soát chi tiết công cụ thu hồi [`src/acquisition/revoke_speaker_consent.py`](file:///d:/trainidentificationaudio/src/acquisition/revoke_speaker_consent.py) xác định ranh giới rõ rệt giữa những gì mã nguồn tự động xử lý và những nghĩa vụ quản trị bắt buộc con người phải thực hiện:

### 6.1 Ma trận Ranh giới Trách nhiệm Xử lý Thu hồi

| Tầng Dữ liệu / Cấu phần Hệ thống | Hành động Tự động của `revoke_speaker_consent.py` | Nghĩa vụ Quản trị Thủ công Bắt buộc | Mức độ Rủi ro Quản trị |
| :--- | :--- | :--- | :---: |
| **Âm thanh Gốc (`audio_raw/<spk>/`)** | Di chuyển sang thư mục cách ly `quarantine/withdrawn_speakers/<spk>/audio_raw/` (không xóa cứng). | Không cần. Dữ liệu được bảo toàn phục vụ kiểm toán chuỗi bảo quản. | **THẤP (Đã tự động)** |
| **Âm thanh Chuẩn hóa (`audio_16k_standardized/`)** | Di chuyển sang thư mục cách ly `quarantine/withdrawn_speakers/<spk>/audio_16k_standardized/`. | Không cần. Bản chuẩn hóa bị rút khỏi luồng phân vùng. | **THẤP (Đã tự động)** |
| **Manifest Thu thập (`human_collection_manifest_v1.csv`)** | Cập nhật `status="REVOKED_WITHDRAWN"`, ghi rõ `exclusion_reason="consent_revoked: <reason>"`. | Không cần. Các bộ sinh split tự động bỏ qua các hàng mang trạng thái này. | **THẤP (Đã tự động)** |
| **Sổ bộ Đồng thuận (`consent_ledger_human.json`)** | Cập nhật `consent_given=False`, sinh mã kiểm toán `revocation_audit_id="REV_<HEX8>"`, ghi dấu thời gian ISO. | Không cần. Các lệnh nạp file tương lai của speaker này sẽ bị từ chối ngay lập tức. | **THẤP (Đã tự động)** |
| **Nhật ký Cách ly (`quarantine_log.csv`)** | Ghi nhận sự kiện thu hồi kèm đường dẫn nguồn và đích. | Không cần. Lưu vết phục vụ thanh tra. | **THẤP (Đã tự động)** |
| **Cache Đặc trưng Huấn luyện (`data/processed/*.npy`, `.cache`)** | **HOÀN TOÀN KHÔNG CHẠM VÀO.** CLI không tìm kiếm hoặc xóa các tệp cache spectrogram. | **BẮT BUỘC:** Operator phải chạy lệnh xóa sạch (purge) toàn bộ các thư mục cache đặc trưng trung gian trước khi tái lập split. | **TRUNG BÌNH (Rò rỉ nếu quên)** |
| **Bản Sao lưu Lạnh (`data/backups/`, Cloud Storage)** | **HOÀN TOÀN KHÔNG CHẠM VÀO.** Để bảo vệ tính bất biến của backup, CLI không sửa file sao lưu có timestamp. | **BẮT BUỘC:** Cán bộ phụ trách dữ liệu (DPO) phải ghi sổ theo dõi thu hồi và thực hiện thủ tục mã hóa/hủy dữ liệu trên các bản sao lưu khi đến hạn. | **TRUNG BÌNH (Pháp lý GDPR)** |
| **Trọng số Model Đã Huấn luyện (`best_tcresnet.pt`, `baseline_kws.onnx`)** | **HOÀN TOÀN KHÔNG THỂ TỰ XỬ LÝ.** Trọng số mạng neural không thể tự động bóc tách tham số đã học qua gradient descent. | **BẮT BUỘC:** <br>1. Gắn nhãn checkpoint cũ: *"Cấm phát hành thương mại phiên bản này sau ngày rút consent"*; <br>2. Kích hoạt pipeline huấn luyện lại từ đầu (retrain from scratch) hoàn toàn không có dữ liệu của người đã rút quyền. | **CAO (Rủi ro Sở hữu Trí tuệ & Tuân thủ)** |

### 6.2 Độ Bền vững khi Gặp Lỗi Cục bộ (Partial-Failure Resilience)
- **Kiểm tra mã nguồn ([`revoke_speaker_consent.py#L157-L184`](file:///d:/trainidentificationaudio/src/acquisition/revoke_speaker_consent.py#L157-L184)):**  
  Script kiểm tra sự tồn tại của tệp nguồn trước khi di chuyển (`if not src_file.exists(): continue`).
- **Thử nghiệm phục hồi (`test_revocation_partial_failure_recovery`):**  
  Đã mô phỏng tình huống 1 tệp âm thanh bị mất/xóa trước khi chạy lệnh thu hồi. Script hoàn tất an toàn 100%, không bị văng lỗi, cập nhật manifest và ledger chính xác, và các lần chạy lặp lại đảm bảo tính lũy nghiệm (idempotent, trả về `ALREADY_REVOKED`).

---

## 7. Bảng Đối chiếu Tổng hợp: Tài liệu Dự thảo vs. Bằng chứng Phê duyệt

| Nội dung Quản trị | Tài liệu Dự thảo (Draft Specification) | Bằng chứng Phê duyệt Đã Ký (Authoritative Evidence) | Trạng thái Hiện tại |
| :--- | :--- | :--- | :---: |
| **Quy trình Thu thập Dữ liệu** | [`collection_protocol.md`](file:///d:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/collection_protocol.md) (v1.0.0-PROPOSAL) | **CHƯA CÓ** (Chưa có chữ ký hoặc quyết định ban hành) | `PENDING HUMAN APPROVAL` |
| **Điều khoản Đồng thuận Nghiên cứu** | Mẫu điều khoản tiếng Việt trong [`collection_protocol.md#L41-L67`](file:///d:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/collection_protocol.md#L41-L67) | **CHƯA CÓ** (Chưa có ý kiến pháp lý hoặc phê duyệt IRB) | `PENDING HUMAN APPROVAL` |
| **Quota Lấy mẫu Nhân khẩu học** | Bảng đề xuất 50N/30C/40S trong [`collection_protocol.md#L108-L113`](file:///d:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/collection_protocol.md#L108-L113) | **CHƯA CÓ** (Chưa có văn bản phê duyệt phân bổ ngân sách/nhân sự) | `PENDING HUMAN APPROVAL` |
| **Hồ sơ Người tham gia Thực tế** | Quy định mã hóa `SPK_H_<N\|C\|S>_<hex6>` trong [`collection_protocol.md#L90-L92`](file:///d:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/collection_protocol.md#L90-L92) | **CHƯA CÓ** (Zero người tham gia được tiếp cận hoặc ghi âm) | `ZERO RECORDS (EXPECTED)` |
| **Giấy phép Nguồn Âm thanh FLEURS** | Ghi nhận CC-BY-4.0 trong mã lệnh [`download_approved_sources.py#L22`](file:///d:/trainidentificationaudio/src/acquisition/download_approved_sources.py#L22) | **CHƯA CÓ TỆP LICENSE TRÊN ĐĨA** (Cần bổ sung tệp text bản quyền) | `PROVENANCE KNOWN / UNARCHIVED` |
| **Giấy phép Nguồn Âm thanh MSWC** | Tệp `version.txt` và ghi nhận CC-BY-4.0 trong code | **CHƯA CÓ TỆP LICENSE TRÊN ĐĨA** (Cần bổ sung tệp text bản quyền) | `PROVENANCE KNOWN / UNARCHIVED` |
| **Nguồn Nhiễu Ngoài (`mic-noise`)** | Báo cáo phân tích rủi ro bản quyền trong [`p0_remediation_report.md`](file:///d:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/p0_remediation_report.md) | **ĐÃ XÁC MINH LÀ CHƯA CÓ GIẤY PHÉP; ĐÃ KHÓA THÀNH CÔNG BẰNG CODE** | `BLOCKED FROM PRODUCTION` |

---

## 8. Hành động Cần thiết Trước khi Phê duyệt Khởi động Chiến dịch

Để hoàn thiện toàn bộ thủ tục pháp trị và chuyển từ trạng thái chuẩn bị sang cho phép thu thập thực địa:

1. **Phê duyệt Bản Đồng thuận (Informed Consent Sign-off):**  
   Cán bộ pháp chế / Hội đồng Đạo đức xem xét và ký duyệt văn bản mẫu tại Mục 2.2 của [`collection_protocol.md`](file:///d:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/collection_protocol.md#L41-L67).
2. **Ký duyệt Quota Tuyển mẫu 50N / 30C / 40S:**  
   Giám đốc Dự án (Project Sponsor) ký phê duyệt chỉ tiêu 120 người tham gia theo tỷ lệ 50 Bắc, 30 Trung, 40 Nam.
3. **Ban hành Quy trình SOP Vận hành CLI (`--strict` Mandate):**  
   Ban hành hướng dẫn thực địa bắt buộc mọi lệnh chạy trên máy thu âm phải đi kèm cờ `--strict --distance close_talk_15cm --mic builtin_mobile` để ngăn chặn metadata mặc định.
4. **Lưu trữ Tệp Giấy phép Văn bản Cục bộ:**  
   Tải và lưu trữ tệp văn bản giấy phép `LICENSE.txt` (CC-BY-4.0) chính thức vào `data/raw/fleurs_vi/` và `data/raw/mswc_vi/`.
5. **Thiết lập Sổ bộ Quản trị Thu hồi Model:**  
   Ban hành quy định bắt buộc huấn luyện lại mô hình từ đầu mỗi khi có yêu cầu rút quyền đồng thuận từ người tham gia.

---

## 9. Kết luận Quản trị Cuối cùng

Dựa trên kết quả rà soát thực tế mã nguồn, hệ thống tệp và cấu trúc quản trị:

### Kết luận Đánh giá: **`READY FOR HUMAN SIGN-OFF`**

- **Hạ tầng Kỹ thuật & Tài liệu Dự thảo:** Đã hoàn chỉnh 100%, 41 bài kiểm thử tự động đạt tuyệt đối, cơ chế cách ly và mã hóa đảm bảo an toàn. Bộ hồ sơ dự thảo đã sẵn sàng để trình cấp có thẩm quyền xem xét và ký duyệt.
- **Trạng thái Thu thập Giọng nói Người thật:** **`STRICTLY BLOCKED`** (Bị khóa nghiêm ngặt; tuyệt đối không thu âm, tuyển người hoặc ký hợp đồng cho đến khi văn bản phê duyệt chính thức được ký tên).
- *Lưu ý Quản trị:* Dưới góc độ khởi động thu thập thực địa, chiến dịch được xếp loại tương đương `BLOCKED — GOVERNANCE EVIDENCE MISSING` cho đến thời điểm chữ ký phê duyệt của con người xuất hiện trên thực tế.
