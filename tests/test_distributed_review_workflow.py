"""
Automated Tests for Distributed Blinded Human-Listening Workflow.
Workspace: D:\trainidentificationaudio
Experiment: experiments/vieneu_kws_research_v10/human_review/distributed_batches/

Verifies:
1. Complete 120-clip coverage across 6 reviewers.
2. Zero duplicates and pairwise disjoint subsets across all reviewers.
3. Exactly 20 unique clips per package.
4. Absolute blinding: zero answer-key leakage, voice names, or labels in ZIP packages.
5. Exact response schema matching existing validation engine.
6. Byte-for-byte SHA-256 audio integrity inside ZIP packages.
7. Reproducibility of stratified allocation.
8. Anti-fabrication check: zero prefilled responses in packages.
"""

import os
import sys
import csv
import json
import zipfile
import hashlib
import pytest

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
V10_DIR = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v10")
HUMAN_DIR = os.path.join(V10_DIR, "human_review")
DIST_DIR = os.path.join(HUMAN_DIR, "distributed_batches")
MANIFEST_PATH = os.path.join(HUMAN_DIR, "review_manifest.csv")
MAP_CSV_PATH = os.path.join(DIST_DIR, "organizer_assignment_map.csv")
MAP_JSON_PATH = os.path.join(DIST_DIR, "organizer_assignment_map.json")

sys.path.insert(0, HUMAN_DIR)
sys.path.insert(0, DIST_DIR)
from response_validation import REQUIRED_FIELDS
from create_distributed_batches import allocate_stratified_batches, RANDOM_SEED


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def test_complete_120_clip_coverage_and_no_duplicates():
    assert os.path.exists(MAP_CSV_PATH), f"Organizer assignment map missing: {MAP_CSV_PATH}"
    with open(MAP_CSV_PATH, "r", encoding="utf-8") as f:
        lines = [line for line in f if not line.startswith("#")]
    rows = list(csv.DictReader(lines))

    assert len(rows) == 120, f"Expected 120 assigned records, found {len(rows)}"

    reviewers = [f"reviewer_{i:02d}" for i in range(1, 7)]
    assigned_per_rev = {r: [] for r in reviewers}

    for row in rows:
        rev_id = row["reviewer_id"]
        assert rev_id in assigned_per_rev, f"Unknown reviewer ID: {rev_id}"
        assigned_per_rev[rev_id].append(row["original_review_clip_id"])

    # Check exactly 20 clips per reviewer
    for rev_id, clips in assigned_per_rev.items():
        assert len(clips) == 20, f"{rev_id} has {len(clips)} clips, expected 20"
        assert len(set(clips)) == 20, f"{rev_id} contains duplicate clips internally"

    # Check pairwise disjointness (zero overlap between any two reviewers)
    for i in range(len(reviewers)):
        for j in range(i + 1, len(reviewers)):
            r_a = reviewers[i]
            r_b = reviewers[j]
            overlap = set(assigned_per_rev[r_a]).intersection(set(assigned_per_rev[r_b]))
            assert len(overlap) == 0, f"Overlap detected between {r_a} and {r_b}: {overlap}"

    # Check complete union coverage of all 120 clips
    all_assigned = [cid for clips in assigned_per_rev.values() for cid in clips]
    expected_120 = [f"review_clip_{i:03d}" for i in range(1, 121)]
    assert sorted(all_assigned) == sorted(expected_120), "Union of assignments does not match all 120 clips"


def test_packages_exist_and_contain_required_files():
    for i in range(1, 7):
        rev_id = f"reviewer_{i:02d}"
        zip_path = os.path.join(DIST_DIR, f"{rev_id}_package.zip")
        assert os.path.exists(zip_path), f"Package ZIP missing: {zip_path}"

        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()

            # Check instructions
            inst_name = f"{rev_id}/reviewer_instructions.md"
            assert inst_name in names, f"Instructions missing in {zip_path}"

            # Check template
            tmpl_name = f"{rev_id}/response_template.csv"
            assert tmpl_name in names, f"Response template missing in {zip_path}"

            # Check exactly 20 audio files
            wav_names = [n for n in names if n.endswith(".wav")]
            assert len(wav_names) == 20, f"Expected 20 WAV files in {zip_path}, found {len(wav_names)}"

            expected_wavs = [f"{rev_id}/audio/clip_{j:02d}.wav" for j in range(1, 21)]
            assert sorted(wav_names) == sorted(expected_wavs), f"Audio naming mismatch in {zip_path}"


def test_packages_strict_blinding_and_zero_leakage():
    forbidden_terms = [
        "review_manifest.csv",
        "organizer_assignment_map",
        "confusion_family",
        "minimal_pair",
        "phonetic_phrase_context",
        "v6_cmd_",
        "v6_unk_",
        "v6_sil_",
        "canonical_class",
        "bounded_label",
    ]

    for i in range(1, 7):
        rev_id = f"reviewer_{i:02d}"
        zip_path = os.path.join(DIST_DIR, f"{rev_id}_package.zip")

        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                # Verify no confidential filenames
                for term in forbidden_terms:
                    assert term not in name, f"Confidential term '{term}' found in ZIP entry name: {name}"

                # Inspect text assets for leaked answer keys
                if name.endswith((".md", ".csv", ".txt")):
                    content = zf.read(name).decode("utf-8")
                    for term in forbidden_terms:
                        assert term not in content, f"Confidential term '{term}' leaked in {name}"


def test_response_template_schema_and_anti_fabrication():
    for i in range(1, 7):
        rev_id = f"reviewer_{i:02d}"
        zip_path = os.path.join(DIST_DIR, f"{rev_id}_package.zip")

        with zipfile.ZipFile(zip_path, "r") as zf:
            tmpl_bytes = zf.read(f"{rev_id}/response_template.csv")
            lines = tmpl_bytes.decode("utf-8").strip().splitlines()
            reader = list(csv.DictReader(lines))

            assert len(reader) == 20, f"Expected 20 rows in template for {rev_id}, got {len(reader)}"

            # Verify schema
            header = lines[0].split(",")
            for field in REQUIRED_FIELDS:
                assert field in header, f"Required field '{field}' missing from template in {rev_id}"

            # Verify clip IDs and empty ratings (zero fabrication)
            expected_cids = [f"clip_{j:02d}" for j in range(1, 21)]
            for idx, row in enumerate(reader):
                assert row["review_clip_id"] == expected_cids[idx]
                assert row["perceived_text"] == "", f"Fabricated perceived_text in {rev_id}"
                assert row["intelligibility_1to5"] == "", f"Fabricated intelligibility in {rev_id}"
                assert row["naturalness_1to5"] == "", f"Fabricated naturalness in {rev_id}"
                assert row["pronunciation_correct_yn"] == "", f"Fabricated pronunciation in {rev_id}"
                assert row["confidence_1to5"] == "", f"Fabricated confidence in {rev_id}"


def test_audio_byte_sha256_hash_integrity():
    with open(MAP_CSV_PATH, "r", encoding="utf-8") as f:
        lines = [line for line in f if not line.startswith("#")]
    map_rows = list(csv.DictReader(lines))

    # Index by (reviewer_id, package_clip_id)
    key_map = {(r["reviewer_id"], r["package_clip_id"]): r["audio_sha256"] for r in map_rows}

    for i in range(1, 7):
        rev_id = f"reviewer_{i:02d}"
        zip_path = os.path.join(DIST_DIR, f"{rev_id}_package.zip")

        with zipfile.ZipFile(zip_path, "r") as zf:
            for j in range(1, 21):
                cid = f"clip_{j:02d}"
                entry_name = f"{rev_id}/audio/{cid}.wav"
                data = zf.read(entry_name)
                h = sha256_bytes(data)

                expected_hash = key_map[(rev_id, cid)]
                assert h == expected_hash, f"Byte hash mismatch for {entry_name} in {zip_path}"


def test_allocation_reproducibility():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        src_rows = list(csv.DictReader(f))

    # Run allocator with same seed
    batches_run1 = allocate_stratified_batches(src_rows, seed=RANDOM_SEED)
    batches_run2 = allocate_stratified_batches(src_rows, seed=RANDOM_SEED)

    for r_idx in range(6):
        ids1 = [x["review_clip_id"] for x in batches_run1[r_idx]]
        ids2 = [x["review_clip_id"] for x in batches_run2[r_idx]]
        assert ids1 == ids2, f"Allocation not reproducible for reviewer {r_idx + 1}"
