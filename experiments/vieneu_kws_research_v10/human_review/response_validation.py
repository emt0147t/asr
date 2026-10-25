"""
Response Validation Module for Blinded Vietnamese Audio Review.
Experiments: experiments/vieneu_kws_research_v10/human_review/

Validates reviewer responses against schema, valid ranges, completeness, and integrity:
- Checks exactly 120 review clip IDs (review_clip_001 to review_clip_120)
- Checks required fields: perceived_text, intelligibility_1to5, naturalness_1to5, pronunciation_correct_yn, confidence_1to5
- Checks range constraints: 1..5 for ratings, {Y, N, Unsure} for pronunciation
- Separates valid rows and malformed / incomplete rows into validation reports
- Never fabricates or alters responses
"""

import os
import sys
import csv
import json
import argparse
from typing import Dict, List, Any, Tuple, Optional

EXPECTED_CLIP_IDS = [f"review_clip_{i:03d}" for i in range(1, 121)]
BATCH_20_CLIP_IDS = [f"clip_{i:02d}" for i in range(1, 21)]
REQUIRED_FIELDS = [
    "review_clip_id",
    "perceived_text",
    "intelligibility_1to5",
    "naturalness_1to5",
    "pronunciation_correct_yn",
    "confidence_1to5",
]
VALID_RATINGS = {"1", "2", "3", "4", "5", 1, 2, 3, 4, 5}
VALID_PRONUNCIATION = {"Y", "N", "Unsure", "y", "n", "unsure"}


def validate_response_file(
    csv_path: str,
    expected_clip_ids: Optional[List[str]] = None,
    expected_count: Optional[int] = None
) -> Dict[str, Any]:
    """
    Validates a single reviewer response CSV file.
    Returns a dictionary detailing validation status, errors, completed count, and row records.
    Supports either the 120-clip master package or 20-clip distributed batch packages.
    """
    if expected_clip_ids is not None:
        target_ids = list(expected_clip_ids)
    elif expected_count == 20:
        target_ids = BATCH_20_CLIP_IDS
    elif expected_count is not None:
        target_ids = [f"clip_{i:02d}" for i in range(1, expected_count + 1)]
    else:
        target_ids = EXPECTED_CLIP_IDS
    result = {
        "file_path": csv_path,
        "exists": os.path.exists(csv_path),
        "total_rows_read": 0,
        "valid_rows_count": 0,
        "invalid_rows_count": 0,
        "is_complete": False,
        "errors": [],
        "valid_rows": [],
        "invalid_rows": [],
    }

    if not result["exists"]:
        result["errors"].append(f"File not found: {csv_path}")
        return result

    try:
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fieldnames = [c.strip() for c in (reader.fieldnames or [])]
            missing_fields = [rf for rf in REQUIRED_FIELDS if rf not in fieldnames]
            if missing_fields:
                result["errors"].append(f"Missing required columns in header: {missing_fields}")
                return result

            seen_clip_ids = set()
            for row_idx, raw_row in enumerate(reader, start=2):
                row = {k.strip(): (v.strip() if v is not None else "") for k, v in raw_row.items()}
                cid = row.get("review_clip_id", "")
                result["total_rows_read"] += 1

                row_errors = []
                if not cid:
                    row_errors.append("Empty review_clip_id")
                elif cid not in target_ids:
                    row_errors.append(f"Unrecognized review_clip_id: {cid}")
                elif cid in seen_clip_ids:
                    row_errors.append(f"Duplicate review_clip_id: {cid}")
                else:
                    seen_clip_ids.add(cid)

                # Required non-empty check
                perceived_text = row.get("perceived_text", "")
                if not perceived_text:
                    row_errors.append("Empty perceived_text")

                # Rating checks
                for col in ["intelligibility_1to5", "naturalness_1to5", "confidence_1to5"]:
                    val = row.get(col, "")
                    if val not in VALID_RATINGS:
                        row_errors.append(f"Invalid {col}: '{val}' (expected 1-5)")

                # Pronunciation check
                pron = row.get("pronunciation_correct_yn", "")
                if pron not in VALID_PRONUNCIATION:
                    row_errors.append(f"Invalid pronunciation_correct_yn: '{pron}' (expected Y, N, or Unsure)")

                if row_errors:
                    result["invalid_rows"].append({
                        "row_number": row_idx,
                        "review_clip_id": cid,
                        "errors": row_errors,
                        "raw_row": row
                    })
                else:
                    # Normalize pronunciation casing
                    normalized_row = dict(row)
                    if normalized_row["pronunciation_correct_yn"].upper() in {"Y", "N"}:
                        normalized_row["pronunciation_correct_yn"] = normalized_row["pronunciation_correct_yn"].upper()
                    elif normalized_row["pronunciation_correct_yn"].title() == "Unsure":
                        normalized_row["pronunciation_correct_yn"] = "Unsure"
                    result["valid_rows"].append(normalized_row)

        result["valid_rows_count"] = len(result["valid_rows"])
        result["invalid_rows_count"] = len(result["invalid_rows"])
        result["missing_clips_count"] = len(target_ids) - len(seen_clip_ids)
        result["is_complete"] = (result["valid_rows_count"] == len(target_ids) and result["invalid_rows_count"] == 0)

        if result["missing_clips_count"] > 0:
            missing_ids = [cid for cid in target_ids if cid not in seen_clip_ids]
            result["errors"].append(f"Missing {len(missing_ids)} clips: {missing_ids[:5]}...")

    except Exception as e:
        result["errors"].append(f"Exception during parsing: {str(e)}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Validate blinded listening response CSV")
    parser.add_argument("csv_path", help="Path to reviewer response CSV")
    parser.add_argument("--issues_output", default=None, help="Optional path to write invalid rows")
    parser.add_argument("--batch", action="store_true", help="Validate a 20-clip distributed batch (clip_01 to clip_20)")
    parser.add_argument("--expected_count", type=int, default=None, help="Expected number of unique clips (e.g. 20)")
    args = parser.parse_args()

    expected_ids = BATCH_20_CLIP_IDS if args.batch else None
    res = validate_response_file(args.csv_path, expected_clip_ids=expected_ids, expected_count=args.expected_count)
    print(json.dumps({
        "file": args.csv_path,
        "is_complete": res["is_complete"],
        "valid_rows": res["valid_rows_count"],
        "invalid_rows": res["invalid_rows_count"],
        "missing_clips": res.get("missing_clips_count", 0),
        "errors": res["errors"][:5]
    }, indent=2))

    if args.issues_output and res["invalid_rows"]:
        with open(args.issues_output, "w", encoding="utf-8") as f:
            json.dump(res["invalid_rows"], f, indent=2, ensure_ascii=False)
        print(f"Wrote invalid rows to {args.issues_output}")

    sys.exit(0 if res["is_complete"] else 1)


if __name__ == "__main__":
    main()
