"""
Distributed Blinded Human-Listening Workflow Generator.
Workspace: D:\trainidentificationaudio
Experiment: experiments/vieneu_kws_research_v10/human_review/distributed_batches/

Tasks:
1. Verify source manifest integrity and all 120 audio SHA-256 hashes.
2. Stratified reproducible allocation into 6 non-overlapping groups of 20 clips.
3. Create 6 blinded ZIP packages (one per reviewer) with neutral filenames (clip_01.wav to clip_20.wav).
4. Verify audio byte hashes within ZIP packages.
5. Create confidential organizer-only assignment mapping.
"""

import os
import sys
import csv
import json
import zipfile
import hashlib
import random
from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
HUMAN_REVIEW_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
MANIFEST_PATH = os.path.join(HUMAN_REVIEW_DIR, "review_manifest.csv")
SOURCE_AUDIO_DIR = os.path.join(HUMAN_REVIEW_DIR, "audio")
INSTRUCTIONS_PATH = os.path.join(HUMAN_REVIEW_DIR, "reviewer_instructions.md")

DISTRIBUTED_DIR = SCRIPT_DIR
os.makedirs(DISTRIBUTED_DIR, exist_ok=True)

RANDOM_SEED = 20261018
NUM_REVIEWERS = 6
CLIPS_PER_REVIEWER = 20


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_source_assets(manifest_path: str, audio_dir: str) -> List[Dict[str, str]]:
    """Verify source manifest integrity and all 120 audio SHA-256 hashes before packaging."""
    print("[Step 1] Verifying source manifest and audio SHA-256 hashes...")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest missing: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if len(rows) != 120:
        raise ValueError(f"Expected exactly 120 rows in manifest, found {len(rows)}")

    mismatches = []
    missing = []
    for r in rows:
        audio_path = os.path.join(audio_dir, r["audio_file"])
        if not os.path.exists(audio_path):
            missing.append(r["audio_file"])
            continue
        h = sha256_file(audio_path)
        if h != r["sha256"]:
            mismatches.append((r["audio_file"], h, r["sha256"]))

    if missing:
        raise FileNotFoundError(f"Missing audio files ({len(missing)}): {missing}")
    if mismatches:
        raise ValueError(f"SHA-256 mismatches ({len(mismatches)}): {mismatches}")

    print(f"  --> Source integrity verified: 120/120 clips matched with 0 errors.")
    return rows


def allocate_stratified_batches(rows: List[Dict[str, str]], seed: int = RANDOM_SEED) -> List[List[Dict[str, str]]]:
    """
    Allocate the 120 clips into 6 non-overlapping groups of 20 using reproducible seed
    and stratified balancing based on confusion family, canonical class, and voice metadata.
    """
    print(f"[Step 2] Allocating 120 clips across {NUM_REVIEWERS} reviewers (seed={seed})...")
    rng = random.Random(seed)

    # Group into strata
    strata = defaultdict(list)
    for r in rows:
        cf = r["confusion_family"]
        if cf == "target_command":
            strata[("target", r["canonical_class"])].append(r)
        elif cf in ("office_ambient", "near_silence", "hvac_hum", "airflow_pink"):
            strata["silence"].append(r)
        else:
            strata[cf].append(r)

    reviewers: List[List[Dict[str, str]]] = [[] for _ in range(NUM_REVIEWERS)]

    # Distribute items stratum by stratum
    for k in sorted(strata.keys(), key=lambda x: str(x)):
        items = list(strata[k])
        rng.shuffle(items)
        for it in items:
            idx = min(range(NUM_REVIEWERS), key=lambda i: len(reviewers[i]))
            reviewers[idx].append(it)

    # Check that each reviewer has exactly 20 clips
    for i, r in enumerate(reviewers):
        if len(r) != CLIPS_PER_REVIEWER:
            raise ValueError(f"Reviewer {i+1} allocated {len(r)} clips instead of {CLIPS_PER_REVIEWER}")

    # Verify no overlaps
    assigned_ids = [x["review_clip_id"] for r in reviewers for x in r]
    if len(assigned_ids) != 120 or len(set(assigned_ids)) != 120:
        raise ValueError("Allocation overlap or incomplete coverage detected!")

    # Randomize presentation order within each reviewer package
    for r in reviewers:
        rng.shuffle(r)

    print("  --> Stratified allocation complete. Summary per reviewer:")
    for i, r in enumerate(reviewers):
        classes = Counter(x["canonical_class"] for x in r)
        voices = Counter(x["voice_id"] for x in r if not x["voice_id"].startswith("office") and not x["voice_id"].startswith("near"))
        print(f"      Reviewer {i+1}: 20 clips | Commands: {sum(cnt for k, cnt in classes.items() if k in ('TRAI', 'PHAI', 'LEN', 'XUONG'))} | Non-targets: {classes.get('UNKNOWN', 0) + classes.get('SILENCE', 0)}")

    return reviewers


def build_package_instructions(reviewer_id: str) -> str:
    """Generate tailored reviewer instructions in Vietnamese for a 20-clip distributed package."""
    return f"""# Hướng Dẫn Dành Cho Người Đánh Giá Nghe Giấu Tên (Blinded Reviewer Instructions)
## Gói đánh giá: {reviewer_id}

**Loại đánh giá:** Đánh giá cảm thụ âm thanh người nghe hai chiều ẩn danh (Double-Blind Human Perceptual Listening Test)  
**Số lượng mẫu âm thanh:** 20 đoạn (`clip_01.wav` đến `clip_20.wav`, 1.0 giây mỗi đoạn, 16 kHz mono)  
**Trạng thái quản trị:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Mục Đích Nghiên Cứu & Cơ Chế Giấu Tên (Blinding)
Bạn đang đánh giá độ rõ ràng, độ tự nhiên và độ chuẩn xác phát âm của các đoạn âm thanh tiếng Việt ngắn (1.0 giây) được tổng hợp tự động.
Để loại bỏ hoàn toàn định kiến xác nhận (confirmation bias), thử nghiệm này được thiết kế **hoàn toàn giấu tên hai chiều (strictly double-blind)**:
- Bạn chỉ nhận được mã số thứ tự ngẫu nhiên của đoạn âm thanh (`clip_01` đến `clip_20`).
- Bạn **KHÔNG** biết từ mục tiêu dự kiến ban đầu là gì, không biết danh mục nào, và không biết máy tính hay mô hình AI nhận diện ra từ nào.
- Vui lòng đánh giá khách quan dựa hoàn toàn trên những gì tai bạn nghe được thực tế.

---

## 2. Quy Trình Thực Hiện Đánh Giá
1. Giải nén gói zip này trên máy tính cá nhân của bạn.
2. Đeo tai nghe trong phòng yên tĩnh, nghe từng file âm thanh trong thư mục `audio/` (từ `clip_01.wav` đến `clip_20.wav`). Bạn có thể nghe lại mỗi đoạn nhiều lần tùy thích.
3. Mở file `response_template.csv` bằng Microsoft Excel, LibreOffice Calc hoặc trình chỉnh sửa văn bản.
4. Với mỗi đoạn âm thanh, bạn phải hoàn thành đầy đủ **5 tiêu chí [BẮT BUỘC]** và có thể điền thêm **1 tiêu chí [TÙY CHỌN]**:

### Tiêu Chí Đánh Giá Chi Tiết:
1. **Từ/Âm tiết nghe được (`perceived_text`) [BẮT BUỘC ĐÁNH GIÁ]:**
   - Gõ chính xác từ hoặc các âm tiết tiếng Việt có đầy đủ dấu thanh mà bạn nghe thấy (Ví dụ: *trái*, *phải*, *lên*, *xuống*, *trại*, *bánh rán*).
   - **Giữ nguyên mã tiếng Anh nếu gặp:**
     - Nếu đoạn âm thanh chỉ có tiếng ồn phòng/rè nền hoặc hoàn toàn yên lặng: Nhập chính xác `[SILENCE]`
     - Nếu nghe thấy có tiếng nói nhưng bị méo mó, biến dạng không nhận ra từ nào: Nhập chính xác `[UNINTELLIGIBLE]`
2. **Độ rõ ràng/Dễ hiểu (`intelligibility_1to5`) [BẮT BUỘC ĐÁNH GIÁ]:**
   - `5` = Rất rõ ràng; nghe hiểu ngay lập tức không cần gắng sức.
   - `4` = Khá rõ ràng; nghe hiểu dễ dàng, tạp âm rất nhỏ.
   - `3` = Tương đối rõ; hiểu được từ nhưng có biến dạng nhận thấy rõ.
   - `2` = Khó nghe; âm thanh nghẹt/méo nhiều, phần lớn phải đoán.
   - `1` = Hoàn toàn không thể hiểu được; không nhận diện được âm tiết nào.
3. **Độ tự nhiên (`naturalness_1to5`) [BẮT BUỘC ĐÁNH GIÁ]:**
   - `5` = Hoàn toàn tự nhiên như giọng người thật nói.
   - `4` = Tự nhiên, chỉ có cảm giác nhân tạo rất nhẹ.
   - `3` = Giọng máy chấp nhận được, ngữ điệu nhận ra được.
   - `2` = Giọng máy cứng nhắc/robot, cao độ hoặc tiết tấu bất thường.
   - `1` = Rất rè, méo mó cơ học, âm thanh điện tử khó chịu.
4. **Tính chuẩn xác phát âm & Thanh điệu (`pronunciation_correct_yn`) [BẮT BUỘC ĐÁNH GIÁ]:**
   - Nhập chính xác một trong 3 mã tiếng Anh:
     - `Y` (Yes) = Phát âm đúng ngữ âm tiếng Việt, thanh điệu chuẩn xác, không nuốt âm đuôi.
     - `N` (No) = Phát âm sai thanh điệu, sai nguyên âm/phụ âm, hoặc bị cắt cụt đuôi âm tiết.
     - `Unsure` = Không chắc chắn, phát âm mơ hồ giữa các thanh điệu.
5. **Độ tự tin của người nghe (`confidence_1to5`) [BẮT BUỘC ĐÁNH GIÁ]:**
   - `5` = Tuyệt đối chắc chắn về những gì mình nghe được.
   - `4` = Khá tự tin.
   - `3` = Tự tin ở mức trung bình.
   - `2` = Ít tự tin, có sự phân vân.
   - `1` = Đoán mò hoàn toàn.
6. **Ghi chú quan sát (`comments`) [TÙY CHỌN - KHÔNG BẮT BUỘC]:**
   - Ghi chú các hiện tượng âm thanh đặc biệt (Ví dụ: "tiếng kim loại rè", "nuốt âm ng cuối", "tiếng quạt gió"). Có thể để trống nếu không có nhận xét.

---

## 3. Lưu & Gửi Kết Quả
1. Lưu file kết quả của bạn với tên `{reviewer_id}_responses.csv` (định dạng CSV UTF-8 để giữ đúng dấu tiếng Việt).
2. Kiểm tra đảm bảo cả 20 dòng đều có đủ dữ liệu ở các cột bắt buộc.
3. Gửi lại file `{reviewer_id}_responses.csv` cho ban tổ chức.
"""


def build_response_template(package_clip_ids: List[str]) -> str:
    """Generate blank response template matching existing 7-column schema."""
    lines = ["review_clip_id,perceived_text,intelligibility_1to5,naturalness_1to5,pronunciation_correct_yn,confidence_1to5,comments"]
    for cid in package_clip_ids:
        lines.append(f"{cid},,,,,,")
    return "\n".join(lines) + "\n"


def create_packages_and_manifests(reviewers: List[List[Dict[str, str]]]):
    """Creates the 6 separate blinded ZIP packages and the organizer-only mapping."""
    print("[Step 3] Building blinded ZIP packages and confidential assignment map...")

    organizer_records = []
    package_paths = []

    for r_idx, clip_list in enumerate(reviewers):
        rev_num = r_idx + 1
        rev_id = f"reviewer_{rev_num:02d}"
        zip_filename = f"{rev_id}_package.zip"
        zip_path = os.path.join(DISTRIBUTED_DIR, zip_filename)
        package_paths.append(zip_path)

        package_clip_ids = [f"clip_{i:02d}" for i in range(1, CLIPS_PER_REVIEWER + 1)]
        instructions_text = build_package_instructions(rev_id)
        template_text = build_response_template(package_clip_ids)

        # Create ZIP
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            # Add instructions
            zf.writestr(f"{rev_id}/reviewer_instructions.md", instructions_text)
            # Add blank response template
            zf.writestr(f"{rev_id}/response_template.csv", template_text)

            # Add audio files with neutral names
            for local_id, orig_item in zip(package_clip_ids, clip_list):
                src_audio_path = os.path.join(SOURCE_AUDIO_DIR, orig_item["audio_file"])
                with open(src_audio_path, "rb") as af:
                    audio_bytes = af.read()

                # Neutral zip entry path: reviewer_XX/audio/clip_YY.wav
                entry_name = f"{rev_id}/audio/{local_id}.wav"
                zf.writestr(entry_name, audio_bytes)

                # Record in confidential organizer map
                organizer_records.append({
                    "reviewer_id": rev_id,
                    "package_clip_id": local_id,
                    "package_audio_entry": f"audio/{local_id}.wav",
                    "original_review_clip_id": orig_item["review_clip_id"],
                    "original_sample_id": orig_item["original_sample_id"],
                    "canonical_class": orig_item["canonical_class"],
                    "bounded_label": orig_item["bounded_label"],
                    "confusion_family": orig_item["confusion_family"],
                    "voice_id": orig_item["voice_id"],
                    "transcription": orig_item["transcription"],
                    "source_audio_file": orig_item["audio_file"],
                    "audio_sha256": orig_item["sha256"]
                })

        print(f"  --> Built package: {zip_filename} ({len(clip_list)} clips)")

    # Step 4: Verify byte hashes inside created ZIP packages
    print("[Step 4] Verifying audio byte hashes inside ZIP packages...")
    for zip_path in package_paths:
        with zipfile.ZipFile(zip_path, "r") as zf:
            infolist = zf.infolist()
            wav_entries = [info for info in infolist if info.filename.endswith(".wav")]
            if len(wav_entries) != CLIPS_PER_REVIEWER:
                raise ValueError(f"ZIP {zip_path} has {len(wav_entries)} WAVs, expected {CLIPS_PER_REVIEWER}")

            for info in wav_entries:
                local_id = os.path.basename(info.filename).replace(".wav", "")
                rev_id = info.filename.split("/")[0]
                data = zf.read(info)
                h = sha256_bytes(data)

                # Find corresponding record in organizer map
                matching = [rec for rec in organizer_records if rec["reviewer_id"] == rev_id and rec["package_clip_id"] == local_id]
                assert len(matching) == 1, f"Record not found for {rev_id}/{local_id}"
                if h != matching[0]["audio_sha256"]:
                    raise ValueError(f"Hash mismatch inside {zip_path} for {info.filename}")

    print("  --> All 120 packaged clips verified with 100% SHA-256 byte parity.")

    # Step 5: Save organizer-only assignment map
    map_csv_path = os.path.join(DISTRIBUTED_DIR, "organizer_assignment_map.csv")
    map_json_path = os.path.join(DISTRIBUTED_DIR, "organizer_assignment_map.json")

    fieldnames = [
        "reviewer_id", "package_clip_id", "package_audio_entry",
        "original_review_clip_id", "original_sample_id", "canonical_class",
        "bounded_label", "confusion_family", "voice_id", "transcription",
        "source_audio_file", "audio_sha256"
    ]

    with open(map_csv_path, "w", encoding="utf-8", newline="") as f:
        # Write confidential header comment
        f.write("# CONFIDENTIAL_ORGANIZER_KEY_NOT_FOR_REVIEWERS\n")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(organizer_records)

    with open(map_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "notice": "CONFIDENTIAL - ORGANIZER KEY ONLY - DO NOT DISTRIBUTE TO REVIEWERS",
            "seed": RANDOM_SEED,
            "total_reviewers": NUM_REVIEWERS,
            "clips_per_reviewer": CLIPS_PER_REVIEWER,
            "total_clips": len(organizer_records),
            "assignments": organizer_records
        }, f, indent=2, ensure_ascii=False)

    print(f"  --> Saved organizer assignment map: {map_csv_path}")
    print(f"  --> Saved organizer JSON metadata: {map_json_path}")
    return package_paths, organizer_records


def main():
    print("=" * 80)
    print("Distributed Blinded Review Workflow Packaging")
    print("=" * 80)
    rows = verify_source_assets(MANIFEST_PATH, SOURCE_AUDIO_DIR)
    reviewers = allocate_stratified_batches(rows)
    package_paths, records = create_packages_and_manifests(reviewers)
    print("=" * 80)
    print("All 6 reviewer packages and organizer map successfully created!")
    print("=" * 80)


if __name__ == "__main__":
    main()
