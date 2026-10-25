"""
Combine Distributed Reviewer Responses.
Workspace: D:\trainidentificationaudio
Experiment: experiments/vieneu_kws_research_v10/human_review/distributed_batches/

Ingests the 6 completed 20-clip reviewer response CSV files, validates each
against schema and constraints, maps package-local clip IDs (clip_01..clip_20)
back to original manifest review IDs (review_clip_001..review_clip_120) using
organizer_assignment_map.csv, and produces a combined master response CSV
without modifying or overwriting raw returned files.
"""

import os
import sys
import csv
import json
import argparse
from typing import Dict, List, Any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
HUMAN_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
MAP_PATH = os.path.join(SCRIPT_DIR, "organizer_assignment_map.csv")

sys.path.insert(0, HUMAN_DIR)
from response_validation import validate_response_file, REQUIRED_FIELDS


def combine_responses(responses_dir: str, output_csv: str) -> Dict[str, Any]:
    """
    Validates and combines 6 reviewer response CSVs into a single 120-row master CSV.
    """
    if not os.path.exists(MAP_PATH):
        raise FileNotFoundError(f"Assignment map missing: {MAP_PATH}")

    # Load assignment map (skip header comment line if present)
    with open(MAP_PATH, "r", encoding="utf-8") as f:
        lines = [line for line in f if not line.startswith("#")]
    map_rows = list(csv.DictReader(lines))

    # Index map by (reviewer_id, package_clip_id)
    key_map = {}
    for r in map_rows:
        key_map[(r["reviewer_id"], r["package_clip_id"])] = r

    expected_reviewers = [f"reviewer_{i:02d}" for i in range(1, 7)]
    found_files = {}

    for rev_id in expected_reviewers:
        candidate_names = [
            f"{rev_id}_responses.csv",
            f"{rev_id}.csv",
            f"response_{rev_id}.csv"
        ]
        matched_path = None
        for name in candidate_names:
            p = os.path.join(responses_dir, name)
            if os.path.exists(p):
                matched_path = p
                break
        if matched_path:
            found_files[rev_id] = matched_path

    status = {
        "expected_reviewers": expected_reviewers,
        "found_reviewers": list(found_files.keys()),
        "missing_reviewers": [r for r in expected_reviewers if r not in found_files],
        "validation_results": {},
        "is_ready_to_combine": False,
        "combined_rows_count": 0,
        "output_csv": output_csv
    }

    if status["missing_reviewers"]:
        print(f"[Combine] Missing response files for reviewers: {status['missing_reviewers']}")
        return status

    # Validate each submission
    all_valid = True
    validated_rows_by_reviewer = {}
    for rev_id, file_path in found_files.items():
        v_res = validate_response_file(file_path, expected_count=20)
        status["validation_results"][rev_id] = {
            "is_complete": v_res["is_complete"],
            "valid_rows": v_res["valid_rows_count"],
            "invalid_rows": v_res["invalid_rows_count"],
            "errors": v_res["errors"]
        }
        if not v_res["is_complete"]:
            all_valid = False
            print(f"[Combine] Validation failed for {rev_id} ({file_path}): {v_res['errors']}")
        else:
            validated_rows_by_reviewer[rev_id] = v_res["valid_rows"]

    if not all_valid:
        print("[Combine] Cannot combine responses: One or more reviewer submissions failed validation.")
        return status

    # Combine into 120-clip master response list
    combined_rows = []
    seen_master_ids = set()

    for rev_id in expected_reviewers:
        rows = validated_rows_by_reviewer[rev_id]
        for row in rows:
            pkg_cid = row["review_clip_id"]
            mapping_key = (rev_id, pkg_cid)
            if mapping_key not in key_map:
                raise KeyError(f"No mapping found for reviewer={rev_id}, clip={pkg_cid}")

            meta = key_map[mapping_key]
            orig_cid = meta["original_review_clip_id"]

            if orig_cid in seen_master_ids:
                raise ValueError(f"Duplicate master clip ID mapped: {orig_cid}")
            seen_master_ids.add(orig_cid)

            combined_rows.append({
                "review_clip_id": orig_cid,
                "reviewer_id": rev_id,
                "package_clip_id": pkg_cid,
                "perceived_text": row["perceived_text"],
                "intelligibility_1to5": row["intelligibility_1to5"],
                "naturalness_1to5": row["naturalness_1to5"],
                "pronunciation_correct_yn": row["pronunciation_correct_yn"],
                "confidence_1to5": row["confidence_1to5"],
                "comments": row.get("comments", "")
            })

    # Sort by original clip ID (review_clip_001 to review_clip_120)
    combined_rows.sort(key=lambda x: x["review_clip_id"])

    # Write output
    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fieldnames = [
        "review_clip_id", "reviewer_id", "package_clip_id", "perceived_text",
        "intelligibility_1to5", "naturalness_1to5", "pronunciation_correct_yn",
        "confidence_1to5", "comments"
    ]

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined_rows)

    status["is_ready_to_combine"] = True
    status["combined_rows_count"] = len(combined_rows)
    print(f"[Combine] Successfully combined {len(combined_rows)} responses into {output_csv}")
    return status


def main():
    parser = argparse.ArgumentParser(description="Combine 6 distributed reviewer response CSVs")
    parser.add_argument("--responses_dir", required=True, help="Directory containing returned reviewer CSV files")
    parser.add_argument("--output_csv", required=True, help="Destination path for combined master CSV")
    args = parser.parse_args()

    res = combine_responses(args.responses_dir, args.output_csv)
    print(json.dumps(res, indent=2))
    sys.exit(0 if res.get("is_ready_to_combine") else 1)


if __name__ == "__main__":
    main()
