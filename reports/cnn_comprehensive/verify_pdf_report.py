import os
import sys
import io
import pypdf

# Set UTF-8 encoding for standard output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE_DIR = r"d:\trainidentificationaudio"
DOCS_DIR = os.path.join(BASE_DIR, "reports", "cnn_comprehensive")
PDF_PATH = os.path.join(DOCS_DIR, "CNN_PROJECT_OVERVIEW_VI.pdf")
MD_PATH = os.path.join(DOCS_DIR, "CNN_PROJECT_OVERVIEW_VI.md")

print("=" * 60)
print("KIỂM TRA CHẤT LƯỢNG FILE PDF VÀ TÍNH ĐỒNG BỘ NỘI DUNG")
print("=" * 60)

# 1. Check file existence & size
if not os.path.exists(PDF_PATH):
    print("[ERROR] File PDF không tồn tại!")
    sys.exit(1)

size_bytes = os.path.getsize(PDF_PATH)
print(f"[1] Kích thước file PDF: {size_bytes:,} bytes ({size_bytes / 1024:.2f} KB)")

# 2. Inspect PDF structure
reader = pypdf.PdfReader(PDF_PATH)
num_pages = len(reader.pages)
print(f"[2] Tổng số trang A4: {num_pages} trang (Mục tiêu thiết kế: 8-14 trang)")

# 3. Extract text from each page and verify Vietnamese characters
print("\n[3] Trích xuất văn bản từng trang:")
all_text = ""
for idx, page in enumerate(reader.pages):
    text = page.extract_text()
    all_text += text + "\n"
    first_line = text.splitlines()[0] if text.splitlines() else "EMPTY"
    print(f"  - Trang {idx+1:02d}: {len(text):4d} ký tự | Mở đầu: {first_line[:70]}")

# 4. Verify critical numbers & hashes
critical_items = [
    ("Mã băm Checkpoint SHA-256", "3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2"),
    ("Mã băm ONNX SHA-256", "54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99"),
    ("Độ chính xác Phase 13.3", "99.20%"),
    ("Độ nhạy lệnh tổng hợp (Command Recall)", "99.00%"),
    ("Tỷ lệ báo động giả (Non-Target FAR)", "0.75%"),
    ("Độ trễ suy luận trung bình CPU", "1.80 ms"),
    ("Ngưỡng quyết định lệnh tối ưu", "0.20"),
    ("Kích thước FFT", "512"),
    ("Số lượng tham số mô hình", "119.333"),
    ("Số bài kiểm thử tự động workspace", "384"),
    ("Số clip nghe mù Phase 90", "120"),
    ("Người đánh giá duy nhất Phase 90", "reviewer_01"),
    ("Nhãn Canonical Frozen Candidate", "FROZEN_CNN_RELEASE_CANDIDATE"),
    ("Nhãn Canonical Natural Speech", "NOT_VALIDATED_ON_NATURAL_SPEECH"),
    ("Nhãn Canonical Streaming", "NO_DEPLOYABLE_SYSTEM"),
]

print("\n[4] Kiểm tra các số liệu & định danh trọng yếu trong PDF:")
all_passed = True
for label, token in critical_items:
    if token in all_text:
        print(f"  [PASSED] {label:42s}: Tìm thấy '{token}'")
    else:
        print(f"  [FAILED] {label:42s}: KHÔNG tìm thấy '{token}'")
        all_passed = False

# 5. Check consistency between Markdown and PDF
print("\n[5] Kiểm tra tính thống nhất giữa Markdown và PDF:")
with open(MD_PATH, "r", encoding="utf-8") as f:
    md_text = f.read()

print(f"  - Kích thước Markdown: {len(md_text):,} ký tự, {len(md_text.splitlines())} dòng")
for label, token in critical_items:
    in_md = token in md_text
    in_pdf = token in all_text
    if in_md and in_pdf:
        print(f"  [SYNC OK] '{token[:30]}' có mặt ở cả Markdown và PDF")
    else:
        print(f"  [ASYNC ERR] '{token[:30]}' MD={in_md}, PDF={in_pdf}")
        all_passed = False

if all_passed:
    print("\n>>> TẤT CẢ CÁC BƯỚC KIỂM TRA ĐỀU THÀNH CÔNG VƯỢT TRỘI! <<<")
else:
    print("\n>>> CÓ MỘT SỐ ĐIỂM CHƯA ĐỒNG BỘ CẦN XỬ LÝ! <<<")
