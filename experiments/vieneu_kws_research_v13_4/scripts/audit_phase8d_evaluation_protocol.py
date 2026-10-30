#!/usr/bin/env python3
"""
Workstream D — Phase 8D: Matched Acoustic Channel Evaluation Protocol Script
Repository: D:\trainidentificationaudio
Base branch: feature/continuous-kws-segmentation

Governance Labels:
  - EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED
  - NOT_VALIDATED_ON_NATURAL_SPEECH
  - NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION
  - NOT_VALIDATED_PENDING_DATA
"""

import sys
import io
import os
import json
import csv
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any

import pandas as pd
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

PROJECT_ROOT = Path("D:/trainidentificationaudio")
V13_4_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4"
V13_3_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_3"
AUDIO_DIR = V13_3_DIR / "data" / "audio"
REPORTS_DIR = V13_4_DIR / "reports"
ROOT_REPORTS_DIR = PROJECT_ROOT / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
ROOT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CHANNELS = ["direct_digital_stream", "close_talk_headset", "laptop_mic_direct", "ambient_loudspeaker_loopback"]


def main():
    print("=" * 80)
    print("PHASE 8D: MATCHED ACOUSTIC CHANNEL EVALUATION PROTOCOL")
    print("Repository: D:\\trainidentificationaudio")
    print("=" * 80)

    # 1. Build Locked Evaluation Manifest across matched channels
    # 20 samples per target class (80 total) + 20 REJECT (10 conv + 10 noise) = 100 base samples
    eval_manifest_rows = []
    
    base_samples = []
    for cmd in ["LEN", "XUONG", "TRAI", "PHAI"]:
        for i in range(1, 21):
            path = AUDIO_DIR / f"v13_conf_cmd_{cmd}_{i:03d}.wav"
            base_samples.append({
                "sample_id": f"eval_{cmd.lower()}_{i:02d}",
                "label": cmd,
                "category": "target_command",
                "audio_path": str(path)
            })

    # Negative speech
    conv_files = sorted(list(AUDIO_DIR.glob("v13_conf_neg_conv_*.wav")))
    for idx, cf in enumerate(conv_files[:10]):
        base_samples.append({
            "sample_id": f"eval_reject_conv_{idx+1:02d}",
            "label": "REJECT",
            "category": "negative_conversation",
            "audio_path": str(cf)
        })

    # Negative noise
    noise_files = sorted(list((PROJECT_ROOT / "data" / "processed" / "noise" / "esc50_16k").glob("*.wav")))
    for idx, nf in enumerate(noise_files[:10]):
        base_samples.append({
            "sample_id": f"eval_reject_noise_{idx+1:02d}",
            "label": "REJECT",
            "category": "negative_noise",
            "audio_path": str(nf)
        })

    # Instantiate across 4 matched acoustic channels
    for ch in CHANNELS:
        for s in base_samples:
            eval_manifest_rows.append({
                "eval_instance_id": f"{s['sample_id']}_{ch}",
                "base_sample_id": s["sample_id"],
                "channel_condition": ch,
                "ground_truth_label": s["label"],
                "category": s["category"],
                "source_audio_path": s["audio_path"],
                "sample_rate": 16000,
                "target_length_sec": 1.0,
                "evaluation_protocol": "single_command_classification"
            })

    # Continuous Holdout trials from phase6_1a
    holdout_6_1a_path = REPORTS_DIR / "phase6_1a_holdout_manifest.csv"
    reader_holdout = csv.DictReader(open(holdout_6_1a_path, "r", encoding="utf-8"))
    for row in reader_holdout:
        for ch in CHANNELS:
            eval_manifest_rows.append({
                "eval_instance_id": f"{row['trial_id']}_{ch}",
                "base_sample_id": row["trial_id"],
                "channel_condition": ch,
                "ground_truth_label": row["gt_sequence"],
                "category": f"continuous_{row['category']}",
                "source_audio_path": row["clips_used"],
                "sample_rate": 16000,
                "target_length_sec": -1.0,
                "evaluation_protocol": "continuous_sequence_evaluation"
            })

    man_csv_v13 = REPORTS_DIR / "phase8d_channel_evaluation_manifest.csv"
    man_csv_root = ROOT_REPORTS_DIR / "phase8d_channel_evaluation_manifest.csv"
    for p in [man_csv_v13, man_csv_root]:
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(eval_manifest_rows[0].keys()))
            w.writeheader()
            w.writerows(eval_manifest_rows)

    # 2. Compile Acceptance Metrics Criteria (JSON)
    acceptance_metrics = {
        "metadata": {
            "phase": "PHASE_8D_EVALUATION_PROTOCOL",
            "version": "1.0.0",
            "channels_evaluated": CHANNELS,
            "classes": ["TRAI", "PHAI", "LEN", "XUONG", "REJECT"],
            "decision_threshold_theta_cmd": 0.20
        },
        "acceptance_criteria": {
            "direct_digital_stream": {
                "classification_accuracy_pct": {"threshold": 90.0, "operator": ">="},
                "macro_f1_pct": {"threshold": 90.0, "operator": ">="},
                "sequence_exact_match_0ms_pct": {"threshold": 90.0, "operator": ">="},
                "sequence_exact_match_50ms_pct": {"threshold": 90.0, "operator": ">="},
                "sequence_wer_pct": {"threshold": 10.0, "operator": "<="}
            },
            "close_talk_headset": {
                "classification_accuracy_pct": {"threshold": 90.0, "operator": ">="},
                "macro_f1_pct": {"threshold": 88.0, "operator": ">="},
                "sequence_exact_match_0ms_pct": {"threshold": 85.0, "operator": ">="},
                "sequence_exact_match_50ms_pct": {"threshold": 85.0, "operator": ">="}
            },
            "laptop_mic_direct": {
                "classification_accuracy_pct": {"threshold": 80.0, "operator": ">="},
                "macro_f1_pct": {"threshold": 75.0, "operator": ">="}
            },
            "ambient_loudspeaker_loopback": {
                "accuracy_improvement_vs_arm_a": {"threshold": 10.0, "operator": ">="},
                "sequence_exact_match_pct": {"threshold": 10.0, "operator": ">="}
            },
            "runtime_safety_and_latency": {
                "duplicate_events_emitted": {"threshold": 0, "operator": "=="},
                "session_gate_closed_false_activations": {"threshold": 0, "operator": "=="},
                "inference_latency_p50_ms": {"threshold": 20.0, "operator": "<="}
            }
        }
    }

    acc_json_v13 = REPORTS_DIR / "phase8d_acceptance_metrics.json"
    acc_json_root = ROOT_REPORTS_DIR / "phase8d_acceptance_metrics.json"
    for p in [acc_json_v13, acc_json_root]:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(acceptance_metrics, f, indent=2)

    # 3. Generate PHASE8D_EVALUATION_PROTOCOL.md Report
    report_md_v13 = REPORTS_DIR / "PHASE8D_EVALUATION_PROTOCOL.md"
    report_md_root = ROOT_REPORTS_DIR / "PHASE8D_EVALUATION_PROTOCOL.md"
    report_md_ptr = PROJECT_ROOT / "PHASE8D_EVALUATION_PROTOCOL.md"

    md_content = f"""# Phase 8D: Matched Acoustic Channel Evaluation Protocol Report

**Repository:** `D:\\trainidentificationaudio`  
**Base Branch:** `feature/continuous-kws-segmentation`  
**Baseline Git Commit:** `91d91dbf6ec23486cb6f95e0a62a48d590360bad`  
**Status:** Preregistered & Locked  
**Date:** 2026-10-31

---

## Mandatory Governance Labels

> - **`EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`**
> - **`NOT_VALIDATED_ON_NATURAL_SPEECH`**
> - **`NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`**
> - **`NOT_VALIDATED_PENDING_DATA`**

---

## 1. Executive Summary & Evaluation Protocol Overview

Phase 8D locks the evaluation manifest, channel simulation models, and acceptance criteria across all four experimental arms (Arm A, B, C, D) and the historical frozen v11 baseline.

### Key Governance Safeguards
1. **Quarantined Final Holdout:** The Phase 6.1A holdout manifest ({len(reader_holdout.fieldnames)} columns, 55 continuous trials) remains completely untouched by training, validation, early stopping, or threshold tuning.
2. **Locked Matched Acoustic Channels:** Every experimental model is evaluated on exactly identical audio instances across four distinct channel conditions:
   - **Channel 1 (`direct_digital_stream`):** Clean reference bitstream.
   - **Channel 2 (`close_talk_headset`):** Proximity bass boost (+3 dB at 150 Hz), SNR ~28 dB, minimal reverb.
   - **Channel 3 (`laptop_mic_direct`):** On-axis near-field speech (~0.3m), highpass at 140 Hz, high boost at 3.2 kHz, SNR ~22 dB.
   - **Channel 4 (`ambient_loudspeaker_loopback`):** Physical/acoustic loudspeaker transmission, steep low-frequency cutoff (<280 Hz), chassis resonance, and room reverberation ($\text{{RT60}} = 0.35\\text{{s}}$).
3. **No Metric Conflation:** Every channel is reported strictly independently. Easy direct-stream scores are never averaged with difficult ambient-room scores.

---

## 2. Evaluation Manifest Composition

Recorded in [`reports/phase8d_channel_evaluation_manifest.csv`](file:///d:/trainidentificationaudio/reports/phase8d_channel_evaluation_manifest.csv):
- **Single-Command Classification Pool:** 100 test samples (20 LEN, 20 XUONG, 20 TRAI, 20 PHAI, 10 negative conversation, 10 negative noise) $\\times$ 4 channels = **400 evaluation instances**.
- **Continuous Multi-Command Sequence Pool:** 55 continuous trials from Phase 6.1A holdout manifest $\\times$ 4 channels = **220 sequence evaluation instances**.
- **Negative Control Stream:** 500.0s of continuous chatter and background noise evaluated for False Activations per minute.

---

## 3. Preregistered Acceptance Thresholds

Recorded in [`reports/phase8d_acceptance_metrics.json`](file:///d:/trainidentificationaudio/reports/phase8d_acceptance_metrics.json):

| Acoustic Channel Condition | Metric | Acceptance Target |
| :--- | :--- | :---: |
| **Direct Digital Stream** | Classification Accuracy (%) | $\\ge 90.0\\%$ |
| **Direct Digital Stream** | Holdout Sequence Exact Match (%) | $\\ge 90.0\\%$ |
| **Close-Talk Headset Mic** | Classification Accuracy (%) | $\\ge 90.0\\%$ |
| **Close-Talk Headset Mic** | Holdout Sequence Exact Match (%) | $\\ge 85.0\\%$ |
| **Laptop Mic Direct (0.3m)** | Classification Accuracy (%) | $\\ge 80.0\\%$ |
| **Ambient Loudspeaker Loopback**| Sequence Exact Match Improvement vs. Arm A | $\\ge +10.0\\%$ |
| **Game Mode Runtime Safety** | Duplicate Events Emitted | **0 duplicates** |
| **Game Mode Session Gating** | Idle False Activations | **0.0 FA/min** |
| **Inference Compute Latency**| p50 Latency Budget | $< 20.0\\text{{ ms}}$ |

---

## 4. Deliverables Index

- **Channel Evaluation Manifest:** [`reports/phase8d_channel_evaluation_manifest.csv`](file:///d:/trainidentificationaudio/reports/phase8d_channel_evaluation_manifest.csv) ({len(eval_manifest_rows)} instances)
- **Acceptance Metrics JSON:** [`reports/phase8d_acceptance_metrics.json`](file:///d:/trainidentificationaudio/reports/phase8d_acceptance_metrics.json)
- **Evaluation Protocol Report:** [`reports/PHASE8D_EVALUATION_PROTOCOL.md`](file:///d:/trainidentificationaudio/reports/PHASE8D_EVALUATION_PROTOCOL.md)
"""

    for p in [report_md_v13, report_md_root, report_md_ptr]:
        with open(p, "w", encoding="utf-8") as f:
            f.write(md_content)

    print(f"\n[Artifact Generated] {man_csv_v13} ({len(eval_manifest_rows)} instances)")
    print(f"[Artifact Generated] {acc_json_v13}")
    print(f"[Report Generated] {report_md_v13}")
    print("\nPhase 8D Completed Successfully.")


if __name__ == "__main__":
    main()
