#!/usr/bin/env python3
"""
Deep inspection of historical experiment reports, configurations, and metrics.
Extracts timeline, model configurations, datasets, accuracy, recall, FAR, latency, and conclusions.
"""

import os
import sys
import json
import csv
import glob
import re
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = PROJECT_ROOT / "reports" / "cnn_comprehensive" / "tables"


def parse_phase_reports():
    reports_to_scan = [
        # Baseline & Early phases
        ("Phase 27 Baseline", "experiments/vieneu_cnn_baseline/reports/PHASE27_INDEPENDENT_AUDIT.md"),
        ("Cascade v1", "experiments/vieneu_kws_cascade_v1/reports/CASCADE_TRAINING_AND_EVALUATION_REPORT.md"),
        ("VAD Ablation v1", "experiments/vieneu_kws_vad_ablation_v1/reports/VAD_ABLATION_REPORT.md"),
        ("Robustness Campaign", "experiments/vieneu_kws_robustness_campaign/reports/ROBUSTNESS_AND_CALIBRATION_REPORT.md"),
        # Phase 40-44 (v2)
        ("Phase 44 (v2 Decision)", "experiments/vieneu_kws_research_v2/reports/PHASE44_FINAL_RESEARCH_DECISION.md"),
        # Phase 45-49 (v3)
        ("Phase 49 (v3 Decision)", "experiments/vieneu_kws_research_v3/reports/PHASE49_FINAL_DECISION.md"),
        # Phase 50-54 (v4)
        ("Phase 54 (v4 Decision)", "experiments/vieneu_kws_research_v4/reports/PHASE54_FINAL_DECISION.md"),
        # Phase 55-59 (v5)
        ("Phase 59 (v5 Decision)", "experiments/vieneu_kws_research_v5/reports/PHASE59_FINAL_DECISION.md"),
        # Phase 60-64 (v6)
        ("Phase 64 (v6 Decision)", "experiments/vieneu_kws_research_v6/reports/PHASE64_FINAL_DECISION.md"),
        # Phase 65-69 (v7)
        ("Phase 69 (v7 Decision)", "experiments/vieneu_kws_research_v7/reports/PHASE69_FINAL_DECISION.md"),
        # Phase 70-74 (v8)
        ("Phase 74 (v8 Decision)", "experiments/vieneu_kws_research_v8/reports/PHASE74_FINAL_DECISION.md"),
        # Phase 75-79 (v9)
        ("Phase 79 (v9 Decision)", "experiments/vieneu_kws_research_v9/reports/PHASE79_FINAL_DECISION.md"),
        # Phase 80-84 (v10)
        ("Phase 84 (v10 Decision)", "experiments/vieneu_kws_research_v10/reports/PHASE84_FINAL_DECISION.md"),
        # Phase 85-89 (v10)
        ("Phase 89 (v10 Decision)", "experiments/vieneu_kws_research_v10/reports/PHASE89_FINAL_DECISION.md"),
        # Phase 90
        ("Phase 90 (Human Listening)", "experiments/vieneu_kws_research_v10/reports/PHASE90_HUMAN_LISTENING_VALIDATION.md"),
        # Phase 91-92 Deployment
        ("Phase 91 (CNN Deployment)", "experiments/vieneu_kws_research_v10/reports/PHASE91_CNN_DEPLOYMENT.md"),
        ("Phase 92 (Portability Parity)", "experiments/vieneu_kws_research_v10/reports/PHASE92_CROSS_PLATFORM_PARITY.md"),
        # Phase 11-13.4
        ("Phase 11 (Completion Campaign)", "experiments/vieneu_kws_research_v11/reports/PHASE11_CNN_COMPLETION_CAMPAIGN.md"),
        ("Phase 12 (Finalization)", "experiments/vieneu_kws_research_v12/reports/PHASE12_CNN_FINALIZATION.md"),
        ("Phase 13 (Audit)", "experiments/vieneu_kws_research_v13/reports/PHASE13_CONFIRMATORY_EVALUATION.md"),
        ("Phase 13.1 (Remediation)", "experiments/vieneu_kws_research_v13_1/reports/PHASE13_1_PREPROCESSING_REMEDIATION.md"),
        ("Phase 13.2 (Protocol Prep)", "experiments/vieneu_kws_research_v13_2/reports/PHASE13_2_CONFIRMATORY_PREPARATION.md"),
        ("Phase 13.2.1 (Frontend Audit)", "experiments/vieneu_kws_research_v13_2/reports/PHASE13_2_1_FRONTEND_COMPATIBILITY_AUDIT.md"),
        ("Phase 13.3 (Confirmatory Eval)", "experiments/vieneu_kws_research_v13_3/reports/PHASE13_3_CONFIRMATORY_EVALUATION.md"),
        ("Phase 13.4 (Release Candidate)", "experiments/vieneu_kws_research_v13_4/reports/PHASE13_4_FINAL_CNN_RELEASE.md"),
    ]

    timeline_records = []
    for phase_name, rel_path in reports_to_scan:
        full_path = PROJECT_ROOT / rel_path
        exists = full_path.exists()
        summary = ""
        if exists:
            content = full_path.read_text(encoding="utf-8", errors="ignore")
            # Extract first 500 chars or heading
            lines = [l.strip() for l in content.splitlines() if l.strip() and not l.startswith("#")]
            summary = lines[0] if lines else ""
        print(f"{phase_name:<30} | Exists: {exists!s:<5} | Path: {rel_path}")
        timeline_records.append({
            "stage_or_phase": phase_name,
            "report_path": rel_path,
            "exists": exists,
            "brief_description": summary[:120]
        })

    csv_out = TABLES_DIR / "experiment_timeline.csv"
    with open(csv_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["stage_or_phase", "report_path", "exists", "brief_description"])
        writer.writeheader()
        writer.writerows(timeline_records)
    print(f"\n[OK] Wrote timeline records to {csv_out.name}")


if __name__ == "__main__":
    parse_phase_reports()
