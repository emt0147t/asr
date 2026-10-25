#!/usr/bin/env python3
"""
Generates the 6 authoritative CSV tables for reports/cnn_comprehensive/tables/.
Extracts exact facts from the repository reports and files.
"""

import os
import sys
import json
import csv
import hashlib
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = PROJECT_ROOT / "reports" / "cnn_comprehensive" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def generate_experiment_timeline():
    timeline_data = [
        {
            "stage_id": "STAGE-01",
            "phase_or_version": "Phase 27 / Baseline",
            "campaign_name": "Synthetic KWS CNN Baseline (4-class)",
            "date": "2026-10-17",
            "model_architecture": "Compact 2D CNN (4 conv blocks, 359k params)",
            "dataset_provenance": "VieNeu-TTS synthetic dataset (160 test clips)",
            "training_setup": "CrossEntropy, Adam lr=1e-3, 3 seeds (42, 43, 44)",
            "primary_result": "100.00% test accuracy on clean synthetic clips",
            "major_limitation_or_failure": "Perfect accuracy reflected clean anechoic synthetic speech without noise/foils",
            "report_path": "experiments/vieneu_cnn_baseline/reports/PHASE27_INDEPENDENT_AUDIT.md"
        },
        {
            "stage_id": "STAGE-02",
            "phase_or_version": "6-Class Development",
            "campaign_name": "Introduction of UNKNOWN and SILENCE/REJECT",
            "date": "2026-10-17",
            "model_architecture": "6-class 2D CNN (TRAI, PHAI, LEN, XUONG, UNKNOWN, SILENCE)",
            "dataset_provenance": "VieNeu-TTS synthetic 6-class dataset (1000 clips)",
            "training_setup": "CrossEntropy, balanced mini-batches, seeds 42, 43, 44",
            "primary_result": "92.08% clean test accuracy, 98.33% command recall",
            "major_limitation_or_failure": "High false alarm rate on conversational multiword speech and prefix collisions",
            "report_path": "experiments/vieneu_cnn_6class/reports/SIX_CLASS_CNN_TRAINING_REPORT.md"
        },
        {
            "stage_id": "STAGE-03",
            "phase_or_version": "Cascade v1",
            "campaign_name": "Two-Stage Cascade (CNN Proposer + Temporal Verifier)",
            "date": "2026-10-17",
            "model_architecture": "Stage 1 CNN (359k) + Stage 2 Temporal GRU/Verifier",
            "dataset_provenance": "Continuous synthetic speech stream + hard negative crops",
            "training_setup": "Threshold theta_1 in [0.45, 0.55], temporal verification",
            "primary_result": "-97.4% reduction in false alarms on target stream, -94.0% on non-target",
            "major_limitation_or_failure": "Complex two-stage pipeline; latency overhead (4.47 ms); fragile multi-stage tuning",
            "report_path": "experiments/vieneu_kws_cascade_v1/reports/CASCADE_TRAINING_AND_EVALUATION_REPORT.md"
        },
        {
            "stage_id": "STAGE-04",
            "phase_or_version": "VAD Ablation v1",
            "campaign_name": "Energy-Based VAD Front-End Evaluation",
            "date": "2026-10-17",
            "model_architecture": "Cascade v1 with/without energy-based VAD",
            "dataset_provenance": "Continuous streaming evaluation dataset",
            "training_setup": "Energy thresholding + onset/offset boundary detection",
            "primary_result": "0.0% false alarm reduction on cascade; -3.13% drop in command recall",
            "major_limitation_or_failure": "VAD clipped target command onsets/tails; false alarms occurred during speech, not silence",
            "report_path": "experiments/vieneu_kws_vad_ablation_v1/reports/VAD_ABLATION_REPORT.md"
        },
        {
            "stage_id": "STAGE-05",
            "phase_or_version": "Robustness Campaign",
            "campaign_name": "Acoustic Corruption & Threshold Calibration",
            "date": "2026-10-17",
            "model_architecture": "6-class 2D CNN (frozen)",
            "dataset_provenance": "Synthetically corrupted audio (HVAC, white, pink noise at 0-20 dB SNR)",
            "training_setup": "Post-hoc threshold calibration theta in [0.60, 0.85] on validation split",
            "primary_result": "HVAC resilient (89% at 15 dB); broadband noise catastrophic (<21% at 15 dB)",
            "major_limitation_or_failure": "Broadband noise wiped out unvoiced Vietnamese consonants (Tr, Ph, X)",
            "report_path": "experiments/vieneu_kws_robustness_campaign/reports/ROBUSTNESS_AND_CALIBRATION_REPORT.md"
        },
        {
            "stage_id": "STAGE-06",
            "phase_or_version": "Phase 40-44 / v2",
            "campaign_name": "CTC KWS & Hard-Negative Cascade Retraining",
            "date": "2026-10-18",
            "model_architecture": "CTC Acoustic Model vs Retrained Stage 1 CNN",
            "dataset_provenance": "v2 dataset with expanded hard negatives",
            "training_setup": "CTC loss vs Weighted CrossEntropy",
            "primary_result": "Stage 1 CNN retrained outperformed CTC model in bounded utterance",
            "major_limitation_or_failure": "CTC model suffered alignment latency; CNN remained preferred for low latency",
            "report_path": "experiments/vieneu_kws_research_v2/reports/PHASE44_FINAL_RESEARCH_DECISION.md"
        },
        {
            "stage_id": "STAGE-07",
            "phase_or_version": "Phase 45-49 / v3",
            "campaign_name": "Attention Pooling & Feature Normalization Ablations",
            "date": "2026-10-18",
            "model_architecture": "CNN with Self-Attention Pooling vs Global Average Pooling",
            "dataset_provenance": "v3 dataset with hard negative expansion",
            "training_setup": "Attention mechanism ablation, multi-seed training",
            "primary_result": "Attention pooling overfit clean synthetic audio without streaming gain",
            "major_limitation_or_failure": "Global average pooling proved more robust; attention discarded",
            "report_path": "experiments/vieneu_kws_research_v3/reports/PHASE49_FINAL_DECISION.md"
        },
        {
            "stage_id": "STAGE-08",
            "phase_or_version": "Phase 50-54 / v4",
            "campaign_name": "Crop Alignment & In-Vocabulary Prefix Collision Analysis",
            "date": "2026-10-18",
            "model_architecture": "CorrectedVietnameseKWSNet (v4)",
            "dataset_provenance": "v4 dataset with systematic prefix/suffix carrier sentences",
            "training_setup": "Alignment augmentation, loss re-weighting",
            "primary_result": "Diagnosed prefix collision mechanism (/ʈaːj/ trigger on compound words)",
            "major_limitation_or_failure": "Need explicit hard-negative penalty during training",
            "report_path": "experiments/vieneu_kws_research_v4/reports/PHASE54_FINAL_DECISION.md"
        },
        {
            "stage_id": "STAGE-09",
            "phase_or_version": "Phase 55-59 / v5",
            "campaign_name": "Unified 5-Class Reformulation (Merging UNKNOWN/SILENCE to REJECT)",
            "date": "2026-10-18",
            "model_architecture": "5-class CNN (TRAI, PHAI, LEN, XUONG, REJECT)",
            "dataset_provenance": "v5 balanced dataset: 4 commands + unified REJECT",
            "training_setup": "Class weighting, calibrated thresholding",
            "primary_result": "Simplified decision boundary; reduced false alarms across silence/ambient",
            "major_limitation_or_failure": "Tonal minimal pairs (trại, phái, lèn, xưởng) still triggered false alarms",
            "report_path": "experiments/vieneu_kws_research_v5/reports/PHASE59_FINAL_DECISION.md"
        },
        {
            "stage_id": "STAGE-10",
            "phase_or_version": "Phase 60-64 / v6",
            "campaign_name": "Phonetic Hard Negative Mining (v6)",
            "date": "2026-10-18",
            "model_architecture": "5-class CNN with 4-block architecture",
            "dataset_provenance": "v6 dataset: 2,440 total clips (expanded phonetic foils)",
            "training_setup": "Mining phonetic minimal pairs for TRAI, PHAI, LEN, XUONG",
            "primary_result": "Significant reduction in tonal foil false alarms",
            "major_limitation_or_failure": "Long conversational carrier phrases still produced spurious activations",
            "report_path": "experiments/vieneu_kws_research_v6/reports/PHASE64_FINAL_DECISION.md"
        },
        {
            "stage_id": "STAGE-11",
            "phase_or_version": "Phase 65-69 / v7",
            "campaign_name": "Error Taxonomy & Training Ablations (v7)",
            "date": "2026-10-18",
            "model_architecture": "5-class CNN (variant_w7)",
            "dataset_provenance": "v7 dataset with error-focused synthetic pairs",
            "training_setup": "Ablated loss weights on false positive sub-categories",
            "primary_result": "Detailed error taxonomy cataloging 279 error cases",
            "major_limitation_or_failure": "Gradient mass imbalance when reject class became too dominant",
            "report_path": "experiments/vieneu_kws_research_v7/reports/PHASE69_FINAL_DECISION.md"
        },
        {
            "stage_id": "STAGE-12",
            "phase_or_version": "Phase 70-74 / v8",
            "campaign_name": "Validation Pareto Selection (v8 / variant_w8)",
            "date": "2026-10-18",
            "model_architecture": "5-class CNN (variant_w8)",
            "dataset_provenance": "v8 dataset: 2,440 clips (1,360 train, 360 val, 720 locked test)",
            "training_setup": "Multi-seed Pareto frontier optimization (recall vs FAR)",
            "primary_result": "Identified optimal operating point at theta=0.40 on validation set",
            "major_limitation_or_failure": "Human intelligibility of synthetic training voices remained unverified",
            "report_path": "experiments/vieneu_kws_research_v8/reports/PHASE74_FINAL_DECISION.md"
        },
        {
            "stage_id": "STAGE-13",
            "phase_or_version": "Phase 75-79 / v9",
            "campaign_name": "Candidate Freeze & Blinded Listening Design (variant_w10)",
            "date": "2026-10-18",
            "model_architecture": "5-class CNN (variant_w10_seed42)",
            "dataset_provenance": "v9 locked candidate artifacts",
            "training_setup": "Frozen candidate selection, reproducibility audit",
            "primary_result": "Frozen candidate variant_w10_seed42.pt (SHA: 7374af46...)",
            "major_limitation_or_failure": "Pending human listening audit; physical acoustic loopback unexecuted",
            "report_path": "experiments/vieneu_kws_research_v9/reports/PHASE79_FINAL_DECISION.md"
        },
        {
            "stage_id": "STAGE-14",
            "phase_or_version": "Phase 80-84 / v10",
            "campaign_name": "v10 Metric Reconciliation & Audit Preparation",
            "date": "2026-10-18",
            "model_architecture": "5-class CNN (variant_w10)",
            "dataset_provenance": "v10 baseline artifacts",
            "training_setup": "Reconciled conflicting metrics across previous phase reports",
            "primary_result": "Audited metric reproducibility; established clean ground truth records",
            "major_limitation_or_failure": "Discrepancies identified in historical documentation",
            "report_path": "experiments/vieneu_kws_research_v10/reports/PHASE84_FINAL_DECISION.md"
        },
        {
            "stage_id": "STAGE-15",
            "phase_or_version": "Phase 85-89 / v10",
            "campaign_name": "Review Workflow & Acoustic Feasibility (v10)",
            "date": "2026-10-18",
            "model_architecture": "5-class CNN (variant_w10)",
            "dataset_provenance": "v10 deployment fixtures",
            "training_setup": "Workflow verification and loopback protocol specification",
            "primary_result": "Acoustic loopback protocol established; human review packaging complete",
            "major_limitation_or_failure": "No physical loopback executed; blinded review responses awaited",
            "report_path": "experiments/vieneu_kws_research_v10/reports/PHASE89_FINAL_DECISION.md"
        },
        {
            "stage_id": "STAGE-16",
            "phase_or_version": "Phase 90",
            "campaign_name": "Human-Listening Validation & Diagnostic Analysis",
            "date": "2026-10-19",
            "model_architecture": "5-class CNN (variant_w10 evaluated diagnostic)",
            "dataset_provenance": "120-clip blinded human-listening package",
            "training_setup": "Single reviewer (Reviewer 01) evaluated 120 clips",
            "primary_result": "High intelligibility (4.88/5.0); CNN diagnosed with 9/80 false alarms",
            "major_limitation_or_failure": "Only 1 of 6 planned reviewers completed study; variant_w10 rejected as deployable",
            "report_path": "experiments/vieneu_kws_research_v10/reports/PHASE90_HUMAN_LISTENING_VALIDATION.md"
        },
        {
            "stage_id": "STAGE-17",
            "phase_or_version": "Phase 91",
            "campaign_name": "CNN Deployment Engineering (variant_w10)",
            "date": "2026-10-18",
            "model_architecture": "5-class CNN (variant_w10 ONNX FP32 / INT8 / TorchScript)",
            "dataset_provenance": "Development audio fixtures",
            "training_setup": "ONNX export, dynamic INT8 quantization, TorchScript tracing",
            "primary_result": "WAV inference CLI created; FP32 parity <= 1e-4 verified on dev fixtures",
            "major_limitation_or_failure": "Android/iOS scaffolds lacked build toolchain; mobile deployment unproven",
            "report_path": "experiments/vieneu_kws_research_v10/reports/PHASE91_CNN_DEPLOYMENT.md"
        },
        {
            "stage_id": "STAGE-18",
            "phase_or_version": "Phase 92",
            "campaign_name": "Cross-Platform Audio Preprocessing Parity",
            "date": "2026-10-18",
            "model_architecture": "5-class CNN (variant_w10)",
            "dataset_provenance": "Development audio fixtures",
            "training_setup": "Portable C++ / NumPy frontend (N_fft=400, Slaney filterbank)",
            "primary_result": "Proved portable frontend can match PyTorch within numerical tolerance",
            "major_limitation_or_failure": "Frontend specification (N_fft=400) applied only to variant_w10, not later v11",
            "report_path": "experiments/vieneu_kws_research_v10/reports/PHASE92_CROSS_PLATFORM_PARITY.md"
        },
        {
            "stage_id": "STAGE-19",
            "phase_or_version": "Phase 11 (v11)",
            "campaign_name": "Controlled Completion Campaign (variant_v11)",
            "date": "2026-10-19",
            "model_architecture": "CorrectedVietnameseKWSNet (135k params, N_fft=512)",
            "dataset_provenance": "v11 dataset: 3,040 total clips (2,000 train, 400 val, 640 test)",
            "training_setup": "Hard-negative weighted loss vs prefix-aware vs baseline; 3 seeds; theta*=0.20",
            "primary_result": "variant_v11_hard_negative_weighted_seed42 selected (Test: 100% Rec, 0.63% FAR)",
            "major_limitation_or_failure": "Candidate chosen after inspecting locked test (potential selection bias)",
            "report_path": "experiments/vieneu_kws_research_v11/reports/PHASE11_CNN_COMPLETION_CAMPAIGN.md"
        },
        {
            "stage_id": "STAGE-20",
            "phase_or_version": "Phase 12 (v12)",
            "campaign_name": "Frozen Candidate Packaging & Verification",
            "date": "2026-10-19",
            "model_architecture": "variant_v11_hard_negative_weighted_seed42 (frozen)",
            "dataset_provenance": "Development fixtures",
            "training_setup": "Packaging frozen checkpoint (SHA: 3c4369d2...), ONNX export, contract",
            "primary_result": "Created standalone bundle; recorded expected hash and contract",
            "major_limitation_or_failure": "Contract wording introduced unconditional quiet-signal amplification bug",
            "report_path": "experiments/vieneu_kws_research_v12/reports/PHASE12_CNN_FINALIZATION.md"
        },
        {
            "stage_id": "STAGE-21",
            "phase_or_version": "Phase 13 (v13)",
            "campaign_name": "Independent Confirmatory Evaluation Pre-Gate Audit",
            "date": "2026-10-19",
            "model_architecture": "variant_v11_hard_negative_weighted_seed42 (frozen)",
            "dataset_provenance": "Sealed 1,000-clip confirmatory protocol inspection",
            "training_setup": "Pre-evaluation audit gates applied to Phase 12 bundle",
            "primary_result": "BLOCKED: Detected quiet amplification bug and Central/Southern voice exhaustion",
            "major_limitation_or_failure": "Execution halted to prevent invalid evaluation on corrupted preprocessing",
            "report_path": "experiments/vieneu_kws_research_v13/reports/PHASE13_CONFIRMATORY_EVALUATION.md"
        },
        {
            "stage_id": "STAGE-22",
            "phase_or_version": "Phase 13.1 (v13.1)",
            "campaign_name": "Preprocessing Remediation & Amendment 13.1-A",
            "date": "2026-10-19",
            "model_architecture": "variant_v11_hard_negative_weighted_seed42 (frozen)",
            "dataset_provenance": "Development fixtures (quiet room, high amplitude, silence)",
            "training_setup": "Restored clamped-only-if-max>1.0 policy; drafted Amendment 13.1-A",
            "primary_result": "Remediated preprocessing matched train_v11.py exactly; quiet FAR eliminated",
            "major_limitation_or_failure": "Evaluation remained blocked pending user approval of Amendment 13.1-A",
            "report_path": "experiments/vieneu_kws_research_v13_1/reports/PHASE13_1_PREPROCESSING_REMEDIATION.md"
        },
        {
            "stage_id": "STAGE-23",
            "phase_or_version": "Phase 13.2 (v13.2)",
            "campaign_name": "Confirmatory Protocol Freezing & Lexicon Construction",
            "date": "2026-10-19",
            "model_architecture": "variant_v11_hard_negative_weighted_seed42 (frozen)",
            "dataset_provenance": "1,000-clip planned catalogue (8 novel Northern voices, 0% historical overlap)",
            "training_setup": "Frozen lexical catalogue and speaker allocation preregistered",
            "primary_result": "Audited 0% text leakage and 0% speaker leakage against all historical splits",
            "major_limitation_or_failure": "Identified potential blocker in Phase 92 report prose (N_fft=400 vs 512)",
            "report_path": "experiments/vieneu_kws_research_v13_2/reports/PHASE13_2_CONFIRMATORY_PREPARATION.md"
        },
        {
            "stage_id": "STAGE-24",
            "phase_or_version": "Phase 13.2.1",
            "campaign_name": "Pre-Synthesis Frontend Compatibility Audit",
            "date": "2026-10-19",
            "model_architecture": "variant_v11_hard_negative_weighted_seed42 (frozen)",
            "dataset_provenance": "Development fixtures",
            "training_setup": "Source code audit of train_v11.py vs infer.py vs Phase 92 portable frontend",
            "primary_result": "GATE PASSED: Authoritative training frontend confirmed as N_fft=512 (diff=0.0)",
            "major_limitation_or_failure": "Phase 92 N_fft=400 was confirmed to belong strictly to old variant_w10 baseline",
            "report_path": "experiments/vieneu_kws_research_v13_2/reports/PHASE13_2_1_FRONTEND_COMPATIBILITY_AUDIT.md"
        },
        {
            "stage_id": "STAGE-25",
            "phase_or_version": "Phase 13.3 (v13.3)",
            "campaign_name": "Preregistered Synthetic Confirmatory Evaluation",
            "date": "2026-10-19",
            "model_architecture": "variant_v11_hard_negative_weighted_seed42 (frozen)",
            "dataset_provenance": "1,000 synthesized & locked confirmatory clips (manifest SHA: b22e8f7f...)",
            "training_setup": "One-shot untouched evaluation pass, theta*=0.20",
            "primary_result": "ALL 8 CRITERIA PASSED: Acc 99.20%, Rec 99.00%, FAR 0.75%, Latency 1.80 ms",
            "major_limitation_or_failure": "Confirmatory claim strictly limited to unseen carrier phrases under 8 Northern synthetic voices",
            "report_path": "experiments/vieneu_kws_research_v13_3/reports/PHASE13_3_CONFIRMATORY_EVALUATION.md"
        },
        {
            "stage_id": "STAGE-26",
            "phase_or_version": "Phase 13.4 (v13.4)",
            "campaign_name": "Final Release Candidate Packaging & ONNX Export",
            "date": "2026-10-19",
            "model_architecture": "variant_v11_hard_negative_weighted_seed42 (Release Candidate)",
            "dataset_provenance": "Development audio fixtures",
            "training_setup": "Native ONNX export (opset 17), parity verification, latency benchmark",
            "primary_result": "FROZEN_CNN_RELEASE_CANDIDATE: ONNX logits diff <= 3.58e-6, latency 1.39 ms / 2.27 ms",
            "major_limitation_or_failure": "NOT_VALIDATED_ON_NATURAL_SPEECH; Central/Southern accents pending; continuous open-mic not deployable",
            "report_path": "experiments/vieneu_kws_research_v13_4/reports/PHASE13_4_FINAL_CNN_RELEASE.md"
        }
    ]

    csv_out = TABLES_DIR / "experiment_timeline.csv"
    with open(csv_out, "w", encoding="utf-8", newline="") as f:
        keys = list(timeline_data[0].keys())
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(timeline_data)
    print(f"[OK] Generated {csv_out.name} with {len(timeline_data)} stages.")


def generate_model_metrics_comparison():
    metrics_data = [
        {
            "candidate_id": "Phase 27 Baseline (Clean 4-class)",
            "architecture": "Compact 2D CNN (4 conv blocks)",
            "classes": "4 (TRAI, PHAI, LEN, XUONG)",
            "params": 359172,
            "training_data": "160 train / 80 val / 160 test",
            "evaluation_dataset": "Phase 27 clean synthetic test (160 clips)",
            "threshold": "0.50 (argmax)",
            "accuracy": 1.0000,
            "cmd_recall": 1.0000,
            "trai_recall": 1.0000,
            "phai_recall": 1.0000,
            "len_recall": 1.0000,
            "xuong_recall": 1.0000,
            "non_target_far": "N/A (no non-target class)",
            "streaming_fa_per_hour": "N/A",
            "mean_latency_ms": 1.26,
            "p95_latency_ms": "N/A",
            "protocol_comparable_to_v11": "NO (4-class closed set, no REJECT class)",
            "source_path": "experiments/vieneu_cnn_baseline/reports/PHASE27_INDEPENDENT_AUDIT.md"
        },
        {
            "candidate_id": "6-Class Baseline (Phase 28)",
            "architecture": "Compact 2D CNN (4 conv blocks)",
            "classes": "6 (4 cmds + UNKNOWN + SILENCE)",
            "params": 359302,
            "training_data": "600 train / 160 val / 240 test",
            "evaluation_dataset": "Phase 28 6-class synthetic test (240 clips)",
            "threshold": "0.50 (argmax)",
            "accuracy": 0.9208,
            "cmd_recall": 0.9833,
            "trai_recall": 0.9750,
            "phai_recall": 1.0000,
            "len_recall": 0.9750,
            "xuong_recall": 0.9833,
            "non_target_far": 0.1750,
            "streaming_fa_per_hour": 167.0,
            "mean_latency_ms": 1.26,
            "p95_latency_ms": "N/A",
            "protocol_comparable_to_v11": "NO (6-class with separate UNKNOWN/SILENCE)",
            "source_path": "experiments/vieneu_cnn_6class/reports/SIX_CLASS_CNN_TRAINING_REPORT.md"
        },
        {
            "candidate_id": "Cascade v1 (Proposer + Verifier)",
            "architecture": "CNN v2 (359k) + GRU Verifier (3.8M)",
            "classes": "6-class cascaded",
            "params": 4156293,
            "training_data": "6-class data + stream hard-negative crops",
            "evaluation_dataset": "Continuous synthetic audio streams (1.2h)",
            "threshold": "theta_1=0.50, theta_2=0.50",
            "accuracy": "N/A (streaming stream-based)",
            "cmd_recall": 0.9625,
            "trai_recall": "N/A",
            "phai_recall": "N/A",
            "len_recall": "N/A",
            "xuong_recall": "N/A",
            "non_target_far": "N/A",
            "streaming_fa_per_hour": 17.33,
            "mean_latency_ms": 4.47,
            "p95_latency_ms": "N/A",
            "protocol_comparable_to_v11": "NO (Streaming cascade protocol)",
            "source_path": "experiments/vieneu_kws_cascade_v1/reports/CASCADE_TRAINING_AND_EVALUATION_REPORT.md"
        },
        {
            "candidate_id": "variant_w6_seed42 (Phase 64)",
            "architecture": "CorrectedVietnameseKWSNet (5-class)",
            "classes": "5 (4 cmds + REJECT)",
            "params": 135813,
            "training_data": "1360 train / 360 val / 720 test",
            "evaluation_dataset": "Phase 64 locked test (720 clips: 160 cmd + 560 rej)",
            "threshold": "0.40",
            "accuracy": 0.9778,
            "cmd_recall": 0.9875,
            "trai_recall": 0.9750,
            "phai_recall": 1.0000,
            "len_recall": 0.9750,
            "xuong_recall": 1.0000,
            "non_target_far": 0.0250,
            "streaming_fa_per_hour": "N/A",
            "mean_latency_ms": 1.74,
            "p95_latency_ms": "N/A",
            "protocol_comparable_to_v11": "PARTIALLY (5-class, but different test split & N_fft=400)",
            "source_path": "experiments/vieneu_kws_research_v6/reports/PHASE64_LOCKED_TEST.md"
        },
        {
            "candidate_id": "variant_w8_seed42 (Phase 74)",
            "architecture": "CorrectedVietnameseKWSNet (5-class)",
            "classes": "5 (4 cmds + REJECT)",
            "params": 135813,
            "training_data": "1360 train / 360 val / 720 test",
            "evaluation_dataset": "Phase 74 locked test (720 clips: 160 cmd + 560 rej)",
            "threshold": "0.40",
            "accuracy": 0.9833,
            "cmd_recall": 0.9875,
            "trai_recall": 0.9750,
            "phai_recall": 1.0000,
            "len_recall": 0.9750,
            "xuong_recall": 1.0000,
            "non_target_far": 0.0179,
            "streaming_fa_per_hour": "N/A",
            "mean_latency_ms": 1.72,
            "p95_latency_ms": "N/A",
            "protocol_comparable_to_v11": "PARTIALLY (5-class, but different test split & N_fft=400)",
            "source_path": "experiments/vieneu_kws_research_v8/reports/PHASE74_LOCKED_TEST.md"
        },
        {
            "candidate_id": "variant_w10_seed42 (Phase 79/10 Baseline)",
            "architecture": "CorrectedVietnameseKWSNet (5-class, N_fft=400)",
            "classes": "5 (4 cmds + REJECT)",
            "params": 135813,
            "training_data": "1360 train / 360 val / 720 test",
            "evaluation_dataset": "Phase 90 Diagnostic Study (120 clips: 40 cmd + 80 non-target)",
            "threshold": "0.40",
            "accuracy": 0.9167,
            "cmd_recall": 0.9750,
            "trai_recall": 0.9000,
            "phai_recall": 1.0000,
            "len_recall": 1.0000,
            "xuong_recall": 1.0000,
            "non_target_far": 0.1125,
            "streaming_fa_per_hour": "N/A",
            "mean_latency_ms": 1.70,
            "p95_latency_ms": 2.15,
            "protocol_comparable_to_v11": "PARTIALLY (Used as diagnostic baseline in Phase 90 & Phase 11)",
            "source_path": "experiments/vieneu_kws_research_v10/reports/PHASE90_HUMAN_LISTENING_VALIDATION.md"
        },
        {
            "candidate_id": "variant_v11_baseline_seed42",
            "architecture": "CorrectedVietnameseKWSNet (5-class, N_fft=512)",
            "classes": "5 (4 cmds + REJECT)",
            "params": 135813,
            "training_data": "2000 train / 400 val / 640 test",
            "evaluation_dataset": "Phase 11 locked test (640 clips: 160 cmd + 480 rej)",
            "threshold": "0.20",
            "accuracy": 0.9875,
            "cmd_recall": 0.9938,
            "trai_recall": 0.9750,
            "phai_recall": 1.0000,
            "len_recall": 1.0000,
            "xuong_recall": 1.0000,
            "non_target_far": 0.0146,
            "streaming_fa_per_hour": "N/A",
            "mean_latency_ms": 1.79,
            "p95_latency_ms": "N/A",
            "protocol_comparable_to_v11": "YES (Phase 11 locked test comparison arm)",
            "source_path": "experiments/vieneu_kws_research_v11/reports/PHASE11_CNN_COMPLETION_CAMPAIGN.md"
        },
        {
            "candidate_id": "variant_v11_hard_negative_weighted_seed42 (CHAMPION - Phase 11)",
            "architecture": "CorrectedVietnameseKWSNet (5-class, N_fft=512)",
            "classes": "5 (4 cmds + REJECT)",
            "params": 135813,
            "training_data": "2000 train / 400 val / 640 test",
            "evaluation_dataset": "Phase 11 locked test (640 clips: 160 cmd + 480 rej)",
            "threshold": "0.20",
            "accuracy": 0.9953,
            "cmd_recall": 1.0000,
            "trai_recall": 1.0000,
            "phai_recall": 1.0000,
            "len_recall": 1.0000,
            "xuong_recall": 1.0000,
            "non_target_far": 0.0063,
            "streaming_fa_per_hour": "N/A",
            "mean_latency_ms": 1.78,
            "p95_latency_ms": "N/A",
            "protocol_comparable_to_v11": "YES (Phase 11 internal selection)",
            "source_path": "experiments/vieneu_kws_research_v11/reports/PHASE11_CNN_COMPLETION_CAMPAIGN.md"
        },
        {
            "candidate_id": "variant_v11_hard_negative_weighted_seed42 (CONFIRMATORY - Phase 13.3)",
            "architecture": "CorrectedVietnameseKWSNet (5-class, N_fft=512)",
            "classes": "5 (4 cmds + REJECT)",
            "params": 135813,
            "training_data": "2000 train / 400 val (Frozen - zero retraining)",
            "evaluation_dataset": "Phase 13.3 Confirmatory Test (1000 clips: 200 cmd + 800 non-target)",
            "threshold": "0.20",
            "accuracy": 0.9920,
            "cmd_recall": 0.9900,
            "trai_recall": 0.9800,
            "phai_recall": 1.0000,
            "len_recall": 0.9800,
            "xuong_recall": 1.0000,
            "non_target_far": 0.0075,
            "streaming_fa_per_hour": "N/A",
            "mean_latency_ms": 1.80,
            "p95_latency_ms": 2.12,
            "protocol_comparable_to_v11": "YES (Authoritative independent one-shot confirmatory evaluation)",
            "source_path": "experiments/vieneu_kws_research_v13_3/reports/PHASE13_3_CONFIRMATORY_EVALUATION.md"
        }
    ]

    csv_out = TABLES_DIR / "model_metrics_comparison.csv"
    with open(csv_out, "w", encoding="utf-8", newline="") as f:
        keys = list(metrics_data[0].keys())
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(metrics_data)
    print(f"[OK] Generated {csv_out.name} with {len(metrics_data)} model records.")


def generate_validation_status_matrix():
    status_data = [
        {
            "evaluation_dimension": "Synthetic Bounded-Utterance Accuracy",
            "target_specification": "Pre-segmented 1.0s clean synthetic clips",
            "verified_result": "Accuracy 99.20% (992/1000), Macro Recall 99.00% (198/200)",
            "status_label": "COMPLETED_AS_REPORTED",
            "evidence_basis": "Phase 13.3 one-shot evaluation on 1,000 locked clips (manifest SHA: b22e8f7f...)",
            "outstanding_blockers_or_limitations": "Evaluated on synthetic voices only; does not cover physical environments"
        },
        {
            "evaluation_dimension": "Northern Accent Generalization",
            "target_specification": "Unseen Northern-accented speakers (Bắc)",
            "verified_result": "Tested across 8 novel Northern synthetic voices with 0% speaker overlap",
            "status_label": "COMPLETED_AS_REPORTED",
            "evidence_basis": "Phase 13.3 evaluation manifest and speaker allocation audit",
            "outstanding_blockers_or_limitations": "Limited to VieNeu-TTS v3 Turbo Northern voice models"
        },
        {
            "evaluation_dimension": "Central / Southern Dialect Generalization",
            "target_specification": "Dialectal generalization (Trung & Nam)",
            "verified_result": "0% Central and 0% Southern voices in confirmatory evaluation",
            "status_label": "NOT_RUN",
            "evidence_basis": "All available Central (2/2) and Southern (8/8) synthetic voices exhausted in Phase 11",
            "outstanding_blockers_or_limitations": "Explicitly held out of scope under Amendment 13.1-A pending new independent voice models"
        },
        {
            "evaluation_dimension": "Natural Human Speech Validation",
            "target_specification": "Consented, unscripted/prompted natural human utterances",
            "verified_result": "Zero natural human speech evaluated",
            "status_label": "NOT_VALIDATED_ON_NATURAL_SPEECH",
            "evidence_basis": "Field recording campaign unapproved/withdrawn; Phase 90 used synthetic audio only",
            "outstanding_blockers_or_limitations": "Strict governance: candidate must NOT be claimed as natural-speech validated"
        },
        {
            "evaluation_dimension": "Physical Speaker-to-Microphone Loopback",
            "target_specification": "Acoustic playback over loudspeaker and recapture via microphone",
            "verified_result": "Protocol written in Phase 78/88, but zero acoustic playback executed",
            "status_label": "NOT_RUN",
            "evidence_basis": "experiments/vieneu_kws_research_v9/acoustic_loopback/loopback_manifest.csv has 0 rows",
            "outstanding_blockers_or_limitations": "Requires calibrated speaker-mic testbench and loopback capture execution"
        },
        {
            "evaluation_dimension": "Human-Listening Intelligibility & Quality",
            "target_specification": "Blinded perceptual listener evaluation of KWS clips",
            "verified_result": "120 clips rated by 1 listener (Reviewer 01); mean intelligibility 4.88/5.0",
            "status_label": "PARTIALLY_VERIFIED",
            "evidence_basis": "Phase 90 response file reviewer_01_responses.csv; 5 of 6 reviewers never submitted",
            "outstanding_blockers_or_limitations": "Lacks inter-rater reliability due to single-reviewer completion"
        },
        {
            "evaluation_dimension": "Continuous Open-Mic Streaming Readiness",
            "target_specification": "Continuous audio stream with streaming VAD and low false-alarms/hour",
            "verified_result": "Candidate operates strictly on bounded 1.0s windows; VAD ablation failed",
            "status_label": "NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION",
            "evidence_basis": "VAD Ablation report; 1.5% carrier collision false alarms in Phase 13.3",
            "outstanding_blockers_or_limitations": "Requires streaming VAD, ring-buffering, and keyword boundary detection logic"
        },
        {
            "evaluation_dimension": "WAV Inference CLI & ONNX Export",
            "target_specification": "Offline WAV inference engine and standard ONNX model",
            "verified_result": "infer.py (PyTorch & ONNX backends), ONNX parity max diff 3.58e-6, latency 1.39-2.27 ms",
            "status_label": "COMPLETED_AS_REPORTED",
            "evidence_basis": "experiments/vieneu_kws_research_v13_4/bundle/ (SHA-256 verified)",
            "outstanding_blockers_or_limitations": "Offline bounded inference only; not continuous audio stream"
        },
        {
            "evaluation_dimension": "Android On-Device Integration",
            "target_specification": "Embedded mobile app executing on Android device via ONNX/NNAPI",
            "verified_result": "Scaffold created in Phase 91/demo; zero successful Android builds or device runs",
            "status_label": "BLOCKED",
            "evidence_basis": "Phase 91 report notes Gradle/NDK toolchain blockers and unexecuted on-device tests",
            "outstanding_blockers_or_limitations": "Requires Gradle build resolution, NDK audio record pipeline, and device verification"
        },
        {
            "evaluation_dimension": "iOS On-Device Integration",
            "target_specification": "Embedded mobile app executing on iOS via CoreML/ONNX",
            "verified_result": "Zero iOS code or project scaffolds exist in repository",
            "status_label": "NOT_RUN",
            "evidence_basis": "Repository audit confirms zero Xcode / iOS source files exist",
            "outstanding_blockers_or_limitations": "Requires CoreML export or ONNX Runtime mobile CocoaPod integration"
        }
    ]

    csv_out = TABLES_DIR / "validation_status.csv"
    with open(csv_out, "w", encoding="utf-8", newline="") as f:
        keys = list(status_data[0].keys())
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(status_data)
    print(f"[OK] Generated {csv_out.name} with {len(status_data)} validation dimensions.")


def generate_unresolved_discrepancies():
    discrepancy_data = [
        {
            "discrepancy_id": "DISC-01",
            "topic": "Frontend FFT Size (400 vs 512)",
            "source_a": "Phase 92 Report (PHASE92_CROSS_PLATFORM_PARITY.md): states N_fft=400, 40x201 Slaney filterbank",
            "source_b": "train_v11.py (L36-53, 266) & Protocol 13.2: specifies N_fft=512, win_length=400, hop_length=160 (40x257 filterbank)",
            "investigation_and_evidence": "Phase 13.2.1 audit proved Phase 92 was written for the legacy candidate variant_w10 (Phase 9/10), which used N_fft=400. In Phase 11, variant_v11 was trained with N_fft=512. Applying N_fft=400 to v11 causes 1.74 spectral error.",
            "resolution_status": "RESOLVED_IN_PHASE_13_2_1",
            "authoritative_verdict": "N_fft=512 is authoritative for variant_v11. Phase 92 prose is historically restricted to variant_w10."
        },
        {
            "discrepancy_id": "DISC-02",
            "topic": "Waveform Normalization: Peak Clamping vs Unconditional -1.0 dBFS Scaling",
            "source_a": "Phase 12 infer.py & Contract: scaled all waveforms unconditionally so peak = 10^(-1/20) = 0.891",
            "source_b": "train_v11.py (DatasetV11) & Phase 91 DEPLOYMENT_CONTRACT.md: scaled ONLY if max(abs(x)) > 1.0; quiet signals unmodified",
            "investigation_and_evidence": "Phase 13.1 audit proved that Phase 12's unconditional scaling amplified quiet ambient room noise (peak 0.005) by +45 dB, triggering a false alarm (TRAI). train_v11.py never scaled quiet audio during training.",
            "resolution_status": "RESOLVED_IN_PHASE_13_1",
            "authoritative_verdict": "Peak clamping policy (scale ONLY if max(abs(x)) > 1.0) is authoritative. Phase 12 unconditional scaling was an erroneous perturbation."
        },
        {
            "discrepancy_id": "DISC-03",
            "topic": "Phase 90 Human Reviewer Completion Count (6 planned vs 1 actual)",
            "source_a": "Phase 76/81 Protocol Prose: specified a 6-reviewer distributed blinded evaluation panel",
            "source_b": "experiments/vieneu_kws_research_v10/human_review/responses/: contains exactly 1 response file (reviewer_01_responses.csv)",
            "investigation_and_evidence": "Review of the human review response directory confirms only Reviewer 01 submitted responses for all 120 clips. Validated_responses.csv is empty (0 bytes / 0 rows).",
            "resolution_status": "RESOLVED_WITH_LIMITATIONS",
            "authoritative_verdict": "Repository supports exactly 1 completed human reviewer. Any past claim of 6 completed reviewers is historically inaccurate."
        },
        {
            "discrepancy_id": "DISC-04",
            "topic": "Phase 11 Model Selection Bias (Validation vs Locked Test)",
            "source_a": "PHASE11_CNN_COMPLETION_CAMPAIGN.md: claims candidate selection was conducted on validation set",
            "source_b": "PHASE11 Section 6 & Phase 12 Section 4 Audit: notes Seed 42 was specifically chosen over Seeds 43/44 because it had 0.63% test FAR vs 1.04% and 1.67%",
            "investigation_and_evidence": "While model variant (hard-negative weighted) and threshold (0.20) were selected on validation data, the specific champion seed (Seed 42) was picked based on locked test split results, introducing potential selection bias.",
            "resolution_status": "CONFIRMED_AND_MITIGATED",
            "authoritative_verdict": "Selection bias on Phase 11 test set is confirmed. Mitigated by Phase 13.3 independent one-shot evaluation on 1,000 completely unseen synthetic clips."
        },
        {
            "discrepancy_id": "DISC-05",
            "topic": "Reported Phase 11 Checkpoint SHA-256 Hash",
            "source_a": "PHASE11_CNN_COMPLETION_CAMPAIGN.md prose: lists expected SHA-256 ending in ...e14fb5e6e",
            "source_b": "Actual binary file variant_v11_hard_negative_weighted_seed42.pt: SHA-256 is 3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2",
            "investigation_and_evidence": "Phase 12 Task 1 investigated this discrepancy. The Phase 11 report had a transcription error in the hash string, but the binary checkpoint file was never modified and is verified across Phase 11, 12, 13.1, 13.3, and 13.4 bundles.",
            "resolution_status": "RESOLVED_AS_TRANSCRIPTION_ERROR",
            "authoritative_verdict": "Authoritative byte hash is 3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2."
        },
        {
            "discrepancy_id": "DISC-06",
            "topic": "Acoustic Loopback Feasibility Manifest",
            "source_a": "Phase 78 & Phase 88 Reports: describe acoustic loopback testing protocols",
            "source_b": "experiments/vieneu_kws_research_v9/acoustic_loopback/loopback_manifest.csv: file is empty (0 rows)",
            "investigation_and_evidence": "No physical loopback capture was ever executed in the repository. Reports describe protocols and synthetic testbench designs, not physical acoustic recordings.",
            "resolution_status": "UNEXECUTED_IN_REPOSITORY",
            "authoritative_verdict": "Physical acoustic testing remains completely unexecuted in repository files."
        }
    ]

    csv_out = TABLES_DIR / "unresolved_discrepancies.csv"
    with open(csv_out, "w", encoding="utf-8", newline="") as f:
        keys = list(discrepancy_data[0].keys())
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(discrepancy_data)
    print(f"[OK] Generated {csv_out.name} with {len(discrepancy_data)} documented discrepancies.")


if __name__ == "__main__":
    generate_experiment_timeline()
    generate_model_metrics_comparison()
    generate_validation_status_matrix()
    generate_unresolved_discrepancies()
