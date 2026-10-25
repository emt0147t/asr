#!/usr/bin/env python3
"""
Extracts detailed metrics across all historical phases.
"""

import os
import sys
import json
import csv
import re
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = PROJECT_ROOT / "reports" / "cnn_comprehensive" / "tables"

def inspect_metrics():
    # Let's inspect Phase 27, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v13.3, v13.4
    reports_map = [
        ("Phase 27 Baseline", "experiments/vieneu_cnn_baseline/reports/PHASE27_INDEPENDENT_AUDIT.md"),
        ("Cascade v1", "experiments/vieneu_kws_cascade_v1/reports/CASCADE_TRAINING_AND_EVALUATION_REPORT.md"),
        ("VAD Ablation v1", "experiments/vieneu_kws_vad_ablation_v1/reports/VAD_ABLATION_REPORT.md"),
        ("Robustness Campaign", "experiments/vieneu_kws_robustness_campaign/reports/ROBUSTNESS_AND_CALIBRATION_REPORT.md"),
        ("v2 (Phase 44)", "experiments/vieneu_kws_research_v2/reports/PHASE44_FINAL_RESEARCH_DECISION.md"),
        ("v3 (Phase 49)", "experiments/vieneu_kws_research_v3/reports/PHASE49_LOCKED_EVALUATION.md"),
        ("v4 (Phase 54)", "experiments/vieneu_kws_research_v4/reports/PHASE54_FINAL_DECISION.md"),
        ("v5 (Phase 59)", "experiments/vieneu_kws_research_v5/reports/PHASE59_LOCKED_TEST.md"),
        ("v6 (Phase 64)", "experiments/vieneu_kws_research_v6/reports/PHASE64_LOCKED_TEST.md"),
        ("v7 (Phase 69)", "experiments/vieneu_kws_research_v7/reports/PHASE69_LOCKED_EVALUATION.md"),
        ("v8 (Phase 74)", "experiments/vieneu_kws_research_v8/reports/PHASE74_LOCKED_TEST.md"),
        ("v9 (Phase 79)", "experiments/vieneu_kws_research_v9/reports/PHASE79_FINAL_DECISION.md"),
        ("v10 (Phase 84)", "experiments/vieneu_kws_research_v10/reports/PHASE84_FINAL_DECISION.md"),
        ("v10 (Phase 89)", "experiments/vieneu_kws_research_v10/reports/PHASE89_FINAL_DECISION.md"),
        ("Phase 90 Human Review", "experiments/vieneu_kws_research_v10/reports/PHASE90_HUMAN_LISTENING_VALIDATION.md"),
        ("v11 (Phase 11)", "experiments/vieneu_kws_research_v11/reports/PHASE11_CNN_COMPLETION_CAMPAIGN.md"),
        ("v12 (Phase 12)", "experiments/vieneu_kws_research_v12/reports/PHASE12_CNN_FINALIZATION.md"),
        ("v13 (Phase 13)", "experiments/vieneu_kws_research_v13/reports/PHASE13_CONFIRMATORY_EVALUATION.md"),
        ("v13.1 (Phase 13.1)", "experiments/vieneu_kws_research_v13_1/reports/PHASE13_1_PREPROCESSING_REMEDIATION.md"),
        ("v13.2.1 (Phase 13.2.1)", "experiments/vieneu_kws_research_v13_2/reports/PHASE13_2_1_FRONTEND_COMPATIBILITY_AUDIT.md"),
        ("v13.3 (Phase 13.3)", "experiments/vieneu_kws_research_v13_3/reports/PHASE13_3_CONFIRMATORY_EVALUATION.md"),
        ("v13.4 (Phase 13.4)", "experiments/vieneu_kws_research_v13_4/reports/PHASE13_4_FINAL_CNN_RELEASE.md"),
    ]

    for label, rpath in reports_map:
        f = PROJECT_ROOT / rpath
        if f.exists():
            text = f.read_text(encoding="utf-8", errors="ignore")
            # print summary findings
            print(f"=== {label} ===")
            # Look for lines with Accuracy, Recall, FAR, Threshold, FA/hour
            matches = [line.strip() for line in text.splitlines() if any(k in line.lower() for k in ["accuracy", "recall", "far", "threshold", "theta", "false alarm", "latency", "verdict"])]
            for m in matches[:10]:
                print("  ", m)
            print()

if __name__ == "__main__":
    inspect_metrics()
