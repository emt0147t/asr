#!/usr/bin/env python3
"""
Comprehensive Repository Audit Script for Vietnamese KWS CNN History.
Scans all checkpoints, reports, manifests, predictions, and human review records.
Outputs machine-readable inventories and evidence tables.
"""

import os
import sys
import json
import csv
import glob
import hashlib
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = PROJECT_ROOT / "reports" / "cnn_comprehensive" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def audit_checkpoints():
    print("--- Auditing All Checkpoints and ONNX Models ---")
    pt_files = sorted(PROJECT_ROOT.glob("experiments/**/*.pt"))
    onnx_files = sorted(PROJECT_ROOT.glob("experiments/**/*.onnx"))
    all_models = pt_files + onnx_files

    records = []
    for p in all_models:
        rel = p.relative_to(PROJECT_ROOT).as_posix()
        size = p.stat().st_size
        sha = compute_sha256(p)
        records.append({
            "relative_path": rel,
            "filename": p.name,
            "format": p.suffix.upper()[1:],
            "size_bytes": size,
            "sha256": sha
        })
        print(f"  {rel:<75} | {size:>10} B | {sha[:16]}...")

    csv_out = TABLES_DIR / "checkpoint_lineage.csv"
    with open(csv_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["relative_path", "filename", "format", "size_bytes", "sha256"])
        writer.writeheader()
        writer.writerows(records)
    print(f"[OK] Saved {len(records)} model records to {csv_out.name}\n")
    return records


def audit_datasets():
    print("--- Auditing Datasets and Splits ---")
    manifests = sorted(list(PROJECT_ROOT.glob("experiments/**/splits/*.csv")) +
                       list(PROJECT_ROOT.glob("experiments/**/data/*.csv")) +
                       list(PROJECT_ROOT.glob("experiments/**/*manifest*.csv")))

    records = []
    for m in manifests:
        rel = m.relative_to(PROJECT_ROOT).as_posix()
        if "human_review" in rel or "field_recording" in rel:
            continue
        try:
            with open(m, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                rows = list(reader)
                row_count = len(rows)
            sha = compute_sha256(m)
            records.append({
                "relative_path": rel,
                "row_count": row_count,
                "header_columns": len(header) if header else 0,
                "sha256": sha
            })
            print(f"  {rel:<75} | Rows: {row_count:>6} | SHA: {sha[:16]}...")
        except Exception as e:
            print(f"  Error reading {rel}: {e}")

    csv_out = TABLES_DIR / "dataset_inventory.csv"
    with open(csv_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["relative_path", "row_count", "header_columns", "sha256"])
        writer.writeheader()
        writer.writerows(records)
    print(f"[OK] Saved {len(records)} dataset records to {csv_out.name}\n")
    return records


def audit_human_review():
    print("--- Auditing Phase 90 Human Review Responses ---")
    review_dir = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v10" / "human_review"
    if not review_dir.exists():
        print("  Human review directory missing!")
        return

    csv_files = list(review_dir.glob("*.csv"))
    print(f"  Found {len(csv_files)} review CSV files in {review_dir.as_posix()}")
    for cf in csv_files:
        with open(cf, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            rows = list(reader)
            print(f"    {cf.name}: {len(rows)} rows, header: {header}")

    # Check responses subdirectory
    resp_dir = review_dir / "responses"
    if resp_dir.exists():
        resp_files = list(resp_dir.glob("*.csv"))
        print(f"  Found {len(resp_files)} response submissions in {resp_dir.as_posix()}:")
        for rf in resp_files:
            with open(rf, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                rows = list(reader)
                print(f"    {rf.name}: {len(rows)} responses, header: {header}")


if __name__ == "__main__":
    audit_checkpoints()
    audit_datasets()
    audit_human_review()
