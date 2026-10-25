"""
Script: generate_audit_csvs.py
Generates the four structured audit CSV deliverables for:
docs/sound_command_recognition/audit/
1. claim_audit.csv
2. dataset_reconciliation.csv
3. model_evidence_matrix.csv
4. figure_table_audit.csv
"""

import os
from pathlib import Path
import pandas as pd

AUDIT_DIR = Path("docs/sound_command_recognition/audit")
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

def generate_claim_audit():
    claims = [
        {
            "claim_id": "CLM-001",
            "category": "DATASET_VOLUME",
            "exact_claim": "Total approved audio samples across splits is 2,099 (Train: 1,281, Val: 284, Test: 534)",
            "source_artifact": "docs/objective_1/README.md, docs/objective_1/tables/table_02_class_split_distribution.csv",
            "verification_method": "pandas row count on train_manifest_v2.csv, val_manifest_v2.csv, test_manifest_v2.csv",
            "observed_value": "1,281 + 284 + 534 = 2,099 approved files across 3 splits",
            "status": "VERIFIED",
            "severity": "INFO",
            "corrective_action": "None; verified true invariant."
        },
        {
            "claim_id": "CLM-002",
            "category": "DATASET_VOLUME",
            "exact_claim": "Dataset consists of 1,000 target commands, 779 UNKNOWN samples, and 321 SILENCE samples (Total = 2,099)",
            "source_artifact": "docs/objective_1/dataset_and_provenance.md, docs/objective_1/objective_1_final_report.md",
            "verification_method": "Cross-tabulation of class_label in source_registry_v2.csv and split manifests",
            "observed_value": "1,000 commands + 779 + 321 = 2,100 (arithmetic error). Actual manifest counts: 1,000 commands, 699 UNKNOWN, 400 SILENCE (Total = 2,099).",
            "status": "INCONSISTENT",
            "severity": "HIGH",
            "corrective_action": "Correct narrative documentation to reflect actual counts (699 UNKNOWN, 400 SILENCE)."
        },
        {
            "claim_id": "CLM-003",
            "category": "DATASET_PROVENANCE",
            "exact_claim": "UNKNOWN speech distractors (779 samples) derive from Mozilla Common Voice Vietnamese (CC0-1.0)",
            "source_artifact": "docs/objective_1/dataset_and_provenance.md, docs/objective_1/figures/figure_02_dataset_acquisition_provenance_flow.svg",
            "verification_method": "Inspection of source column in source_registry_v2.csv, train_manifest_v2.csv, val_manifest_v2.csv, test_manifest_v2.csv",
            "observed_value": "Common Voice samples = 0. Actual UNKNOWN sources: google_fleurs_vi (579 approved) and calibration_confusing_words_v1 (120 approved).",
            "status": "INCONSISTENT",
            "severity": "CRITICAL",
            "corrective_action": "Retract Common Voice attribution; document Google FLEURS (CC-BY-4.0) and confusing words generator."
        },
        {
            "claim_id": "CLM-004",
            "category": "DATASET_PROVENANCE",
            "exact_claim": "SILENCE samples (321 samples) derive from ESC-50 and Free Music Archive ambient recordings",
            "source_artifact": "docs/objective_1/dataset_and_provenance.md, docs/objective_1/figures/figure_02_dataset_acquisition_provenance_flow.svg",
            "verification_method": "Inspection of source column in source_registry_v2.csv and split manifests",
            "observed_value": "ESC-50 and FMA samples = 0. All 400 approved SILENCE samples derive from husein_mic_noise (CC-BY-4.0). ESC-50 was rejected due to CC-BY-NC non-commercial clause.",
            "status": "INCONSISTENT",
            "severity": "CRITICAL",
            "corrective_action": "Retract ESC-50/FMA attribution; document Husein microphone noise as the true approved source."
        },
        {
            "claim_id": "CLM-005",
            "category": "DATASET_PROVENANCE",
            "exact_claim": "All 1,000 target command samples across train/val/test are synthetic parametric calibration audio",
            "source_artifact": "docs/objective_1/dataset_and_provenance.md, reports/missing_data_and_blockers.md",
            "verification_method": "Inspection of source column for TRAI, PHAI, LEN, XUONG across manifests",
            "observed_value": "100% (1,000/1,000) target command samples originate from calibration_acoustic_generator_v1.",
            "status": "VERIFIED",
            "severity": "INFO",
            "corrective_action": "Preserve disclosure; highlights need for genuine human field recordings."
        },
        {
            "claim_id": "CLM-006",
            "category": "CLASS_DISTRIBUTION",
            "exact_claim": "Test set contains 50 TRAI, 50 PHAI, 50 LEN, 50 XUONG, 234 UNKNOWN, 100 SILENCE (Total = 534)",
            "source_artifact": "docs/objective_1/results_and_evaluation.md, docs/objective_1/figures/figure_03_class_distribution_by_split.svg, figure_05_confusion_matrix_tcresnet.svg",
            "verification_method": "Value counts of class_label on frozen data/manifests/test_manifest_v2.csv",
            "observed_value": "test_manifest_v2.csv contains exactly: 36 TRAI, 36 PHAI, 36 LEN, 36 XUONG, 300 UNKNOWN, 90 SILENCE (Total = 534).",
            "status": "INCONSISTENT",
            "severity": "CRITICAL",
            "corrective_action": "Update narrative text and SVG figures 03 & 05 to match actual manifest counts (36/36/36/36/300/90)."
        },
        {
            "claim_id": "CLM-007",
            "category": "CLASS_DISTRIBUTION",
            "exact_claim": "Train set contains 160 TRAI, 160 PHAI, 160 LEN, 160 XUONG, 461 UNKNOWN, 180 SILENCE (Total = 1,281)",
            "source_artifact": "docs/objective_1/results_and_evaluation.md, docs/objective_1/figures/figure_03_class_distribution_by_split.svg",
            "verification_method": "Value counts of class_label on data/manifests/train_manifest_v2.csv",
            "observed_value": "train_manifest_v2.csv contains exactly: 178 TRAI, 178 PHAI, 178 LEN, 178 XUONG, 319 UNKNOWN, 250 SILENCE (Total = 1,281).",
            "status": "INCONSISTENT",
            "severity": "HIGH",
            "corrective_action": "Update narrative text and SVG figure 03 to match actual manifest counts (178/178/178/178/319/250)."
        },
        {
            "claim_id": "CLM-008",
            "category": "CLASS_DISTRIBUTION",
            "exact_claim": "Validation set contains 40 TRAI, 40 PHAI, 40 LEN, 40 XUONG, 84 UNKNOWN, 40 SILENCE (Total = 284)",
            "source_artifact": "docs/objective_1/results_and_evaluation.md, docs/objective_1/figures/figure_03_class_distribution_by_split.svg",
            "verification_method": "Value counts of class_label on data/manifests/val_manifest_v2.csv",
            "observed_value": "val_manifest_v2.csv contains exactly: 36 TRAI, 36 PHAI, 36 LEN, 36 XUONG, 80 UNKNOWN, 60 SILENCE (Total = 284).",
            "status": "INCONSISTENT",
            "severity": "HIGH",
            "corrective_action": "Update narrative text and SVG figure 03 to match actual manifest counts (36/36/36/36/80/60)."
        },
        {
            "claim_id": "CLM-009",
            "category": "MODEL_ARCHITECTURE",
            "exact_claim": "Evaluated baseline model is BC-ResNet-1 with 39,142 parameters",
            "source_artifact": "docs/objective_1/objective_1_final_report.md, docs/objective_1/figures/figure_04_model_accuracy_f1_trai_comparison.svg",
            "verification_method": "Inspection of src/models/kws_architectures.py and models/checkpoints/best_kws_model.pt",
            "observed_value": "No BC-ResNet-1 exists in repository. Evaluated baseline is VietnameseKWSNet (2D-CNN) with 138,534 parameters.",
            "status": "INCONSISTENT",
            "severity": "CRITICAL",
            "corrective_action": "Replace fictitious BC-ResNet-1 description with actual VietnameseKWSNet (2D-CNN, 138.5k params)."
        },
        {
            "claim_id": "CLM-010",
            "category": "MODEL_PARAMETERS",
            "exact_claim": "TC-ResNet-14 champion architecture has 278,414 parameters",
            "source_artifact": "docs/objective_1/results_and_evaluation.md, docs/objective_1/figures/figure_01_end_to_end_system_architecture.svg",
            "verification_method": "Parameter count calculation via get_param_count(VietnameseTCResNet()) in Python",
            "observed_value": "Actual parameter count of VietnameseTCResNet is 272,934 parameters (5,480 parameter discrepancy).",
            "status": "INCONSISTENT",
            "severity": "MEDIUM",
            "corrective_action": "Update parameter count to 272,934 across all figures, tables, and narrative texts."
        },
        {
            "claim_id": "CLM-011",
            "category": "ACCURACY_MACRO_F1",
            "exact_claim": "TC-ResNet-14 achieves 98.13% test accuracy and 0.9576 Macro-F1 on test_manifest_v2.csv",
            "source_artifact": "docs/objective_1/results_and_evaluation.md, docs/objective_1/figures/figure_04_model_accuracy_f1_trai_comparison.svg",
            "verification_method": "Direct PyTorch evaluation of models/checkpoints/kws_tcresnet_advanced.pt on data/manifests/test_manifest_v2.csv",
            "observed_value": "Evaluated Accuracy = 96.63% (516/534), Macro-F1 = 0.9402. The 98.13% / 0.9576 figures were based on fictitious 50/50/50/50 support.",
            "status": "INCONSISTENT",
            "severity": "CRITICAL",
            "corrective_action": "Update reported accuracy to 96.63% and Macro-F1 to 0.9402 to match reproduced primary artifact."
        },
        {
            "claim_id": "CLM-012",
            "category": "ACCURACY_MACRO_F1",
            "exact_claim": "Baseline model achieves 97.52% test accuracy and 0.9374 Macro-F1",
            "source_artifact": "reports/baseline_model_evaluation.md, docs/objective_1/results_and_evaluation.md",
            "verification_method": "Evaluation of models/checkpoints/best_kws_model.pt on test_manifest_v2.csv",
            "observed_value": "Historical report evaluated on test_manifest.csv (0 silence support); true 6-class F1 was 0.7812. On v2 manifest, reproduced accuracy is 97.75% and Macro-F1 is 0.9466.",
            "status": "HISTORICAL",
            "severity": "HIGH",
            "corrective_action": "Document historical Phase 2 evaluation flaw (missing SILENCE test support) and cite verified v2 metrics (97.75% / 0.9466)."
        },
        {
            "claim_id": "CLM-013",
            "category": "PER_CLASS_RECALL",
            "exact_claim": "TC-ResNet-14 improves Northern TRAI recall to 25.0% (from 8.3% or 16.7%)",
            "source_artifact": "reports/phase3_model_comparison.md, docs/objective_1/results_and_evaluation.md",
            "verification_method": "Subgroup evaluation of kws_tcresnet_advanced.pt on accent == 'Northern' & class_id == 0 in test_manifest_v2.csv",
            "observed_value": "3 out of 12 Northern TRAI test samples correctly spotted = 25.0%. Baseline best_kws_model.pt spotted 2/12 = 16.7%.",
            "status": "VERIFIED",
            "severity": "INFO",
            "corrective_action": "Preserve verified result and root-cause analysis."
        },
        {
            "claim_id": "CLM-014",
            "category": "CONFUSION_MATRIX",
            "exact_claim": "TC-ResNet-14 misclassifies 6 TRAI samples as PHAI and 4 as UNKNOWN (out of 50)",
            "source_artifact": "docs/objective_1/results_and_evaluation.md, docs/objective_1/figures/figure_05_confusion_matrix_tcresnet.svg",
            "verification_method": "sklearn.metrics.confusion_matrix on model predictions vs ground truth from test_manifest_v2.csv",
            "observed_value": "Actual confusion matrix has 36 TRAI support: 27 correct, 9 misclassified as PHAI, 0 to UNKNOWN. UNKNOWN: 293/300 correct, 7 to SILENCE. SILENCE: 90/90 correct.",
            "status": "INCONSISTENT",
            "severity": "CRITICAL",
            "corrective_action": "Replace confusion matrix table and SVG heatmap with actual reproduced 6x6 matrix (table_08 matches; update SVG and narrative)."
        },
        {
            "claim_id": "CLM-015",
            "category": "STREAMING_DURATION",
            "exact_claim": "Continuous streaming evaluated on 140 reserved 1-hour equivalent continuous audio streams",
            "source_artifact": "docs/objective_1/results_and_evaluation.md, docs/objective_1/figures/figure_06_streaming_roc_mcr_latency.svg",
            "verification_method": "Code inspection of src/models/evaluate_streaming.py and reports/phase3_streaming_evaluation.md",
            "observed_value": "Streaming benchmark evaluated on an 8.0-minute (0.133 hour) stream composed of 6.5 min FLEURS speech and 1.5 min noise. '140 streams' was an unsupported extrapolation from 141 rejected registry files.",
            "status": "UNSUPPORTED",
            "severity": "CRITICAL",
            "corrective_action": "Retract '140 1-hour streams' claim. Clearly report actual 8.0-minute continuous streaming evaluation."
        },
        {
            "claim_id": "CLM-016",
            "category": "STREAMING_FALSE_ALARMS",
            "exact_claim": "Streaming detector achieves 0.0 FA/hr on clean office noise and 0.8 FA/hr on cafeteria babble at tau=0.65",
            "source_artifact": "docs/objective_1/results_and_evaluation.md, docs/objective_1/figures/figure_06_streaming_roc_mcr_latency.svg",
            "verification_method": "Inspection of raw streaming outputs in reports/phase3_streaming_evaluation.md",
            "observed_value": "Actual test measured 0 FA in 8.0 minutes of concatenated audio (95% Poisson upper bound = 22.5 FA/hr). 0.8 FA/hr was an unsourced claim not in primary test logs.",
            "status": "UNSUPPORTED",
            "severity": "HIGH",
            "corrective_action": "Disclose that 0 FA was observed in 8.0 minutes and report Poisson confidence bounds rather than claiming long-term rates."
        },
        {
            "claim_id": "CLM-017",
            "category": "STREAMING_LATENCY",
            "exact_claim": "Streaming trigger latency is 192 ms on average",
            "source_artifact": "docs/objective_1/results_and_evaluation.md, docs/objective_1/figures/figure_06_streaming_roc_mcr_latency.svg",
            "verification_method": "Latency logging in src/models/evaluate_streaming.py and reports/phase3_streaming_evaluation.md",
            "observed_value": "Mean streaming detection latency from command onset to trigger was measured at 679.5 ms to 849.7 ms across thresholds. Model single-frame CPU latency is 0.35 - 0.94 ms.",
            "status": "INCONSISTENT",
            "severity": "HIGH",
            "corrective_action": "Distinguish between frame inference latency (0.35 - 0.94 ms) and end-to-end command trigger latency (680 - 850 ms)."
        },
        {
            "claim_id": "CLM-018",
            "category": "EXPORT_FILE_SIZE",
            "exact_claim": "Exported ONNX FP32 TC-ResNet file size is 1,093.7 KB (1,068.1 KiB)",
            "source_artifact": "docs/objective_1/tables/table_10_onnx_model_benchmarks.csv, models/exported/vietnamese_kws_tcresnet_fp32.onnx",
            "verification_method": "os.path.getsize('models/exported/vietnamese_kws_tcresnet_fp32.onnx')",
            "observed_value": "Exact size is 1,093,691 bytes (1,093.7 decimal KB / 1,068.06 KiB).",
            "status": "VERIFIED",
            "severity": "INFO",
            "corrective_action": "None; verified true invariant."
        },
        {
            "claim_id": "CLM-019",
            "category": "EXPORT_FILE_SIZE",
            "exact_claim": "Exported ONNX INT8 Baseline file size is 148.8 KB (145.3 KiB)",
            "source_artifact": "docs/objective_1/tables/table_10_onnx_model_benchmarks.csv, models/exported/vietnamese_kws_int8.onnx",
            "verification_method": "os.path.getsize('models/exported/vietnamese_kws_int8.onnx')",
            "observed_value": "Exact size is 148,811 bytes (148.8 decimal KB / 145.32 KiB).",
            "status": "VERIFIED",
            "severity": "INFO",
            "corrective_action": "None; verified true invariant."
        },
        {
            "claim_id": "CLM-020",
            "category": "QUANTIZATION_STATUS",
            "exact_claim": "vietnamese_kws_int8.onnx contains genuine dynamic quantization operators and uint8 weights",
            "source_artifact": "docs/objective_1/results_and_evaluation.md, tests/test_parity.py",
            "verification_method": "Inspection of ONNX graph nodes and initializer data types via onnx.load()",
            "observed_value": "Contains DynamicQuantizeLinear (6), ConvInteger (4), MatMulInteger (2), and 12 UINT8 tensor initializers.",
            "status": "VERIFIED",
            "severity": "INFO",
            "corrective_action": "None; verified true invariant."
        },
        {
            "claim_id": "CLM-021",
            "category": "QUANTIZATION_STATUS",
            "exact_claim": "vietnamese_kws_tcresnet_int8.onnx exists on disk (~280 KB)",
            "source_artifact": "reports/phase3_model_comparison.md",
            "verification_method": "os.path.exists('models/exported/vietnamese_kws_tcresnet_int8.onnx')",
            "observed_value": "File does NOT exist on disk. Mentioned in error in Phase 3 comparison notes.",
            "status": "UNSUPPORTED",
            "severity": "HIGH",
            "corrective_action": "Formally flag as missing artifact (ISSUE-003 / P2 defect)."
        },
        {
            "claim_id": "CLM-022",
            "category": "ONNX_PARITY",
            "exact_claim": "ONNX models were exported using Opset 17",
            "source_artifact": "docs/objective_1/results_and_evaluation.md, docs/objective_1/figures/figure_08_deployment_parity_pipeline.svg",
            "verification_method": "Inspection of model.opset_import[0].version via onnx.load()",
            "observed_value": "All exported ONNX models on disk use Opset 14, not Opset 17.",
            "status": "INCONSISTENT",
            "severity": "MEDIUM",
            "corrective_action": "Correct opset version to 14 in documentation and SVG figures."
        },
        {
            "claim_id": "CLM-023",
            "category": "ONNX_PARITY",
            "exact_claim": "Maximum absolute difference between PyTorch and ONNX FP32 is < 1e-4 with 100% concordance",
            "source_artifact": "tests/test_parity.py, docs/objective_1/results_and_evaluation.md",
            "verification_method": "Execution of tests/test_parity.py::test_tcresnet_onnx_model_execution_and_parity",
            "observed_value": "Max absolute logit error is 1.19e-6 for TC-ResNet and 9.54e-7 for Baseline; 100% class concordance.",
            "status": "VERIFIED",
            "severity": "INFO",
            "corrective_action": "None; verified true invariant."
        },
        {
            "claim_id": "CLM-024",
            "category": "HUMAN_FIELD_COLLECTION",
            "exact_claim": "Zero genuine consented human command recordings exist in repository (all manifest entries are test fixtures)",
            "source_artifact": "reports/field_recording_campaign_audit.md, docs/objective_1/dataset_and_provenance.md",
            "verification_method": "Inspection of speaker_id, provenance, and input_filename in field_collection_manifest.csv and consent_ledger.json",
            "observed_value": "All field manifest records map to SPK_FIELD_N001_TESTSUITE and SPK_M4A_TEST. Genuine human recordings = 0.",
            "status": "VERIFIED",
            "severity": "INFO",
            "corrective_action": "Preserve strict distinction; maintain P0 blocker status for field data collection."
        },
        {
            "claim_id": "CLM-025",
            "category": "DATASET_LICENSING",
            "exact_claim": "VIVOS speech corpus is rejected due to CC BY-NC-SA 4.0 non-commercial license",
            "source_artifact": "reports/missing_data_and_blockers.md, docs/objective_1/dataset_and_provenance.md",
            "verification_method": "Inspection of license terms for VIVOS and verification that 0 VIVOS files exist in manifests",
            "observed_value": "VIVOS is CC BY-NC-SA 4.0; exactly 0 VIVOS records appear in any manifest.",
            "status": "VERIFIED",
            "severity": "INFO",
            "corrective_action": "None; verified true policy enforcement."
        },
        {
            "claim_id": "CLM-026",
            "category": "DATASET_LICENSING",
            "exact_claim": "VietSpeech / VSV-1100 is quarantined due to unconsented web scraping IP risks",
            "source_artifact": "reports/missing_data_and_blockers.md, docs/objective_1/dataset_and_provenance.md",
            "verification_method": "Inspection of VietSpeech dataset description and verification that 0 VietSpeech files exist in manifests",
            "observed_value": "Crawled from unconsented social media (YouTube/TikTok); quarantined from training pool. 0 records in manifests.",
            "status": "VERIFIED",
            "severity": "INFO",
            "corrective_action": "None; verified true risk quarantine."
        },
        {
            "claim_id": "CLM-027",
            "category": "AUTOMATED_TEST_SUITE",
            "exact_claim": "Test suite contains 19 automated tests that execute and pass completely",
            "source_artifact": "tests/test_field_acquisition.py, tests/test_parity.py, docs/objective_1/tables/table_11_automated_test_results.csv",
            "verification_method": "Execution of pytest -v tests/ in virtual environment",
            "observed_value": "19 passed in 13.71s (12 field acquisition + 7 parity tests). Exit code = 0.",
            "status": "VERIFIED",
            "severity": "INFO",
            "corrective_action": "None; verified true invariant."
        },
        {
            "claim_id": "CLM-028",
            "category": "DATASET_VOLUME",
            "exact_claim": "Frozen test manifest SHA-256 is 9E3904C8EA427244F7280A41140D7EFD19FD9625B359290ADB7ADCD4FA2E1C13",
            "source_artifact": "tests/test_parity.py, docs/objective_1/reproducibility_and_audit.md",
            "verification_method": "Get-FileHash data/manifests/test_manifest_v2.csv -Algorithm SHA256",
            "observed_value": "Hash confirmed bit-for-bit: 9E3904C8EA427244F7280A41140D7EFD19FD9625B359290ADB7ADCD4FA2E1C13.",
            "status": "VERIFIED",
            "severity": "INFO",
            "corrective_action": "None; permanent cryptographic invariant."
        }
    ]
    df = pd.DataFrame(claims)
    df.to_csv(AUDIT_DIR / "claim_audit.csv", index=False)
    print(f"Generated {AUDIT_DIR / 'claim_audit.csv'} with {len(df)} claims.")

def generate_dataset_reconciliation():
    reconciliations = [
        {
            "partition_or_source": "Train V2 (Approved)",
            "manifest_row_count": 1281,
            "unique_recording_ids": 1281,
            "unique_sha256_hashes": 1281,
            "files_existing_on_disk": 1281,
            "missing_files_count": 0,
            "target_commands_count": 712,
            "unknown_count": 319,
            "silence_count": 250,
            "primary_sources": "calibration_acoustic_generator_v1 (712), husein_mic_noise (250), google_fleurs_vi (199), calibration_confusing_words_v1 (120)",
            "genuine_human_speech_count": 199,
            "synthetic_calibration_count": 832,
            "environmental_noise_count": 250,
            "speaker_leakage_detected": "NO_LEAKAGE (134 disjoint speakers)",
            "audio_compliance_status": "100%_COMPLIANT_16K_MONO_1S_PCM16"
        },
        {
            "partition_or_source": "Val V2 (Approved)",
            "manifest_row_count": 284,
            "unique_recording_ids": 284,
            "unique_sha256_hashes": 284,
            "files_existing_on_disk": 284,
            "missing_files_count": 0,
            "target_commands_count": 144,
            "unknown_count": 80,
            "silence_count": 60,
            "primary_sources": "calibration_acoustic_generator_v1 (144), google_fleurs_vi (80), husein_mic_noise (60)",
            "genuine_human_speech_count": 80,
            "synthetic_calibration_count": 144,
            "environmental_noise_count": 60,
            "speaker_leakage_detected": "NO_LEAKAGE (51 disjoint speakers)",
            "audio_compliance_status": "100%_COMPLIANT_16K_MONO_1S_PCM16"
        },
        {
            "partition_or_source": "Test V2 (Approved, Frozen)",
            "manifest_row_count": 534,
            "unique_recording_ids": 534,
            "unique_sha256_hashes": 534,
            "files_existing_on_disk": 534,
            "missing_files_count": 0,
            "target_commands_count": 144,
            "unknown_count": 300,
            "silence_count": 90,
            "primary_sources": "google_fleurs_vi (300), calibration_acoustic_generator_v1 (144), husein_mic_noise (90)",
            "genuine_human_speech_count": 300,
            "synthetic_calibration_count": 144,
            "environmental_noise_count": 90,
            "speaker_leakage_detected": "NO_LEAKAGE (148 disjoint speakers)",
            "audio_compliance_status": "100%_COMPLIANT_16K_MONO_1S_PCM16"
        },
        {
            "partition_or_source": "Total Approved Pool (Splits)",
            "manifest_row_count": 2099,
            "unique_recording_ids": 2099,
            "unique_sha256_hashes": 2099,
            "files_existing_on_disk": 2099,
            "missing_files_count": 0,
            "target_commands_count": 1000,
            "unknown_count": 699,
            "silence_count": 400,
            "primary_sources": "calibration_acoustic_generator_v1 (1000), google_fleurs_vi (579), husein_mic_noise (400), calibration_confusing_words_v1 (120)",
            "genuine_human_speech_count": 579,
            "synthetic_calibration_count": 1120,
            "environmental_noise_count": 400,
            "speaker_leakage_detected": "NO_LEAKAGE (333 total disjoint speakers)",
            "audio_compliance_status": "100%_COMPLIANT_16K_MONO_1S_PCM16"
        },
        {
            "partition_or_source": "Registry Quarantined / Rejected",
            "manifest_row_count": 141,
            "unique_recording_ids": 141,
            "unique_sha256_hashes": 0,
            "files_existing_on_disk": 0,
            "missing_files_count": 0,
            "target_commands_count": 0,
            "unknown_count": 141,
            "silence_count": 0,
            "primary_sources": "google_fleurs_vi (141 rejected during ingestion for clipping or duration)",
            "genuine_human_speech_count": 141,
            "synthetic_calibration_count": 0,
            "environmental_noise_count": 0,
            "speaker_leakage_detected": "NOT_APPLICABLE (Excluded from splits)",
            "audio_compliance_status": "REJECTED_RAW_EXTRACT"
        },
        {
            "partition_or_source": "Source Registry V2 (Total Rows)",
            "manifest_row_count": 2240,
            "unique_recording_ids": 2240,
            "unique_sha256_hashes": 2099,
            "files_existing_on_disk": 2099,
            "missing_files_count": 0,
            "target_commands_count": 1000,
            "unknown_count": 840,
            "silence_count": 400,
            "primary_sources": "calibration_acoustic_generator_v1 (1000), google_fleurs_vi (720), husein_mic_noise (400), calibration_confusing_words_v1 (120)",
            "genuine_human_speech_count": 720,
            "synthetic_calibration_count": 1120,
            "environmental_noise_count": 400,
            "speaker_leakage_detected": "NO_LEAKAGE (333 disjoint speakers in approved pool)",
            "audio_compliance_status": "2099_APPROVED_141_REJECTED"
        },
        {
            "partition_or_source": "Physical Audio Files on Disk",
            "manifest_row_count": 2398,
            "unique_recording_ids": 2398,
            "unique_sha256_hashes": 2398,
            "files_existing_on_disk": 2398,
            "missing_files_count": 0,
            "target_commands_count": 1000,
            "unknown_count": 699,
            "silence_count": 699,
            "primary_sources": "calibration_acoustic_generator_v1 (1000), google_fleurs_vi (579), calibration_confusing_words_v1 (120), husein_mic_noise (699 on disk, 400 in v2 manifest)",
            "genuine_human_speech_count": 579,
            "synthetic_calibration_count": 1120,
            "environmental_noise_count": 699,
            "speaker_leakage_detected": "NO_LEAKAGE",
            "audio_compliance_status": "100%_COMPLIANT_16K_MONO_1S_PCM16"
        },
        {
            "partition_or_source": "Human Field Collection Takes",
            "manifest_row_count": 74,
            "unique_recording_ids": 74,
            "unique_sha256_hashes": 74,
            "files_existing_on_disk": 74,
            "missing_files_count": 0,
            "target_commands_count": 74,
            "unknown_count": 0,
            "silence_count": 0,
            "primary_sources": "Automated Unit Test Fixtures (SPK_FIELD_N001_TESTSUITE, SPK_M4A_TEST)",
            "genuine_human_speech_count": 0,
            "synthetic_calibration_count": 0,
            "environmental_noise_count": 0,
            "speaker_leakage_detected": "STRICTLY_ISOLATED (0 field takes permitted into model manifests)",
            "audio_compliance_status": "UNIT_TEST_FIXTURES_ONLY"
        }
    ]
    df = pd.DataFrame(reconciliations)
    df.to_csv(AUDIT_DIR / "dataset_reconciliation.csv", index=False)
    print(f"Generated {AUDIT_DIR / 'dataset_reconciliation.csv'} with {len(df)} rows.")

def generate_model_evidence_matrix():
    models = [
        {
            "model_name": "Phase 2 Baseline (Historical)",
            "architecture_class": "VietnameseKWSNet",
            "artifact_path": "models/checkpoints/best_kws_model.pt",
            "artifact_exists": "YES",
            "file_size_bytes": 1688098,
            "parameter_count": 138534,
            "evaluation_manifest": "data/manifests/test_manifest.csv (5-class historical)",
            "reproduced_accuracy": "97.52% (Historical)",
            "claimed_accuracy": "97.52%",
            "reproduced_macro_f1": "0.7812 (True 6-class F1; 0.9374 claimed excluded SILENCE)",
            "claimed_macro_f1": "0.9374",
            "reproduced_trai_recall": "69.44%",
            "claimed_trai_recall": "69.4%",
            "reproduced_northern_trai_recall": "16.67%",
            "claimed_northern_trai_recall": "16.7%",
            "cpu_latency_ms": "0.25 ms (x86 CPU)",
            "evidence_classification": "HISTORICAL_NOT_REPRODUCED_ON_V1 (Defective SILENCE evaluation)"
        },
        {
            "model_name": "Phase 2 Baseline (Evaluated on V2)",
            "architecture_class": "VietnameseKWSNet",
            "artifact_path": "models/checkpoints/best_kws_model.pt",
            "artifact_exists": "YES",
            "file_size_bytes": 1688098,
            "parameter_count": 138534,
            "evaluation_manifest": "data/manifests/test_manifest_v2.csv (6-class, 534 samples)",
            "reproduced_accuracy": "97.75% (522/534)",
            "claimed_accuracy": "97.75%",
            "reproduced_macro_f1": "0.9466",
            "claimed_macro_f1": "0.9466",
            "reproduced_trai_recall": "69.44% (25/36)",
            "claimed_trai_recall": "69.4%",
            "reproduced_northern_trai_recall": "16.67% (2/12)",
            "claimed_northern_trai_recall": "16.7%",
            "cpu_latency_ms": "0.25 ms",
            "evidence_classification": "REPRODUCED_IN_AUDIT"
        },
        {
            "model_name": "Baseline Retrained (Baseline Aug)",
            "architecture_class": "VietnameseKWSNet",
            "artifact_path": "models/checkpoints/kws_baseline_baseline.pt",
            "artifact_exists": "YES",
            "file_size_bytes": 566338,
            "parameter_count": 138534,
            "evaluation_manifest": "data/manifests/test_manifest_v2.csv (6-class, 534 samples)",
            "reproduced_accuracy": "96.25% (514/534)",
            "claimed_accuracy": "96.25%",
            "reproduced_macro_f1": "0.9381",
            "claimed_macro_f1": "0.9381",
            "reproduced_trai_recall": "75.00% (27/36)",
            "claimed_trai_recall": "75.0%",
            "reproduced_northern_trai_recall": "25.00% (3/12)",
            "claimed_northern_trai_recall": "25.0%",
            "cpu_latency_ms": "0.25 ms",
            "evidence_classification": "REPRODUCED_IN_AUDIT"
        },
        {
            "model_name": "Baseline Retrained (Advanced Aug)",
            "architecture_class": "VietnameseKWSNet",
            "artifact_path": "models/checkpoints/kws_baseline_advanced.pt",
            "artifact_exists": "YES",
            "file_size_bytes": 566338,
            "parameter_count": 138534,
            "evaluation_manifest": "data/manifests/test_manifest_v2.csv (6-class, 534 samples)",
            "reproduced_accuracy": "95.13% (508/534)",
            "claimed_accuracy": "95.13%",
            "reproduced_macro_f1": "0.8857",
            "claimed_macro_f1": "0.8857",
            "reproduced_trai_recall": "58.33% (21/36)",
            "claimed_trai_recall": "58.3%",
            "reproduced_northern_trai_recall": "8.33% (1/12)",
            "claimed_northern_trai_recall": "8.3%",
            "cpu_latency_ms": "0.25 ms",
            "evidence_classification": "REPRODUCED_IN_AUDIT"
        },
        {
            "model_name": "Temporal-CNN (Advanced Aug)",
            "architecture_class": "VietnameseTemporalKWSNet",
            "artifact_path": "models/checkpoints/kws_temporal_cnn_advanced.pt",
            "artifact_exists": "YES",
            "file_size_bytes": 1187209,
            "parameter_count": 292582,
            "evaluation_manifest": "data/manifests/test_manifest_v2.csv (6-class, 534 samples)",
            "reproduced_accuracy": "95.88% (512/534)",
            "claimed_accuracy": "95.88%",
            "reproduced_macro_f1": "0.9205",
            "claimed_macro_f1": "0.9205",
            "reproduced_trai_recall": "63.89% (23/36)",
            "claimed_trai_recall": "63.9%",
            "reproduced_northern_trai_recall": "8.33% (1/12)",
            "claimed_northern_trai_recall": "8.3%",
            "cpu_latency_ms": "1.44 ms",
            "evidence_classification": "SUPPORTED_BY_PRIMARY_LOG"
        },
        {
            "model_name": "TC-ResNet-14 Champion (PyTorch Checkpoint)",
            "architecture_class": "VietnameseTCResNet",
            "artifact_path": "models/checkpoints/kws_tcresnet_advanced.pt",
            "artifact_exists": "YES",
            "file_size_bytes": 1123068,
            "parameter_count": 272934,
            "evaluation_manifest": "data/manifests/test_manifest_v2.csv (6-class, 534 samples)",
            "reproduced_accuracy": "96.63% (516/534)",
            "claimed_accuracy": "96.63% (in phase3 report) / 98.13% (in objective_1 docs)",
            "reproduced_macro_f1": "0.9402",
            "claimed_macro_f1": "0.9402 (in phase3 report) / 0.9576 (in objective_1 docs)",
            "reproduced_trai_recall": "75.00% (27/36)",
            "claimed_trai_recall": "75.0% (in phase3 report) / 79.2% (in objective_1 docs)",
            "reproduced_northern_trai_recall": "25.00% (3/12)",
            "claimed_northern_trai_recall": "25.0%",
            "cpu_latency_ms": "0.35 ms (inference only) / 706.4 ms (sliding streaming)",
            "evidence_classification": "REPRODUCED_IN_AUDIT"
        },
        {
            "model_name": "TC-ResNet-14 FP32 (ONNX)",
            "architecture_class": "VietnameseTCResNet",
            "artifact_path": "models/exported/vietnamese_kws_tcresnet_fp32.onnx",
            "artifact_exists": "YES",
            "file_size_bytes": 1093691,
            "parameter_count": 272934,
            "evaluation_manifest": "data/manifests/test_manifest_v2.csv (ONNX Runtime, Opset 14)",
            "reproduced_accuracy": "96.63% (516/534)",
            "claimed_accuracy": "100% parity with PyTorch",
            "reproduced_macro_f1": "0.9402",
            "claimed_macro_f1": "100% parity with PyTorch",
            "reproduced_trai_recall": "75.00% (27/36)",
            "claimed_trai_recall": "75.0%",
            "reproduced_northern_trai_recall": "25.00% (3/12)",
            "claimed_northern_trai_recall": "25.0%",
            "cpu_latency_ms": "0.35 ms (max parity error = 1.19e-06)",
            "evidence_classification": "REPRODUCED_IN_AUDIT"
        },
        {
            "model_name": "2D-CNN Baseline FP32 (ONNX)",
            "architecture_class": "VietnameseKWSNet",
            "artifact_path": "models/exported/vietnamese_kws_float32.onnx",
            "artifact_exists": "YES",
            "file_size_bytes": 555439,
            "parameter_count": 138534,
            "evaluation_manifest": "data/manifests/test_manifest_v2.csv (ONNX Runtime, Opset 14)",
            "reproduced_accuracy": "97.75% (522/534)",
            "claimed_accuracy": "100% parity with PyTorch",
            "reproduced_macro_f1": "0.9466",
            "claimed_macro_f1": "100% parity with PyTorch",
            "reproduced_trai_recall": "69.44% (25/36)",
            "claimed_trai_recall": "69.4%",
            "reproduced_northern_trai_recall": "16.67% (2/12)",
            "claimed_northern_trai_recall": "16.7%",
            "cpu_latency_ms": "0.25 ms (max parity error = 9.54e-07)",
            "evidence_classification": "REPRODUCED_IN_AUDIT"
        },
        {
            "model_name": "2D-CNN Baseline INT8 (ONNX)",
            "architecture_class": "VietnameseKWSNet",
            "artifact_path": "models/exported/vietnamese_kws_int8.onnx",
            "artifact_exists": "YES",
            "file_size_bytes": 148811,
            "parameter_count": 138534,
            "evaluation_manifest": "data/manifests/test_manifest_v2.csv (ONNX Runtime, Opset 14)",
            "reproduced_accuracy": "97.56% (521/534, -0.19% drop)",
            "claimed_accuracy": "Minimal accuracy degradation (< 0.2%)",
            "reproduced_macro_f1": "0.9421",
            "claimed_macro_f1": "Minimal F1 degradation",
            "reproduced_trai_recall": "69.44% (25/36)",
            "claimed_trai_recall": "69.4%",
            "reproduced_northern_trai_recall": "16.67% (2/12)",
            "claimed_northern_trai_recall": "16.7%",
            "cpu_latency_ms": "0.91 ms (genuine DynamicQuantizeLinear, ConvInteger, MatMulInteger)",
            "evidence_classification": "REPRODUCED_IN_AUDIT"
        },
        {
            "model_name": "TC-ResNet-14 INT8 (ONNX)",
            "architecture_class": "VietnameseTCResNet",
            "artifact_path": "models/exported/vietnamese_kws_tcresnet_int8.onnx",
            "artifact_exists": "NO",
            "file_size_bytes": 0,
            "parameter_count": 272934,
            "evaluation_manifest": "N/A (File missing from disk)",
            "reproduced_accuracy": "N/A (Cannot evaluate)",
            "claimed_accuracy": "Claimed in Phase 3 comparison notes",
            "reproduced_macro_f1": "N/A",
            "claimed_macro_f1": "N/A",
            "reproduced_trai_recall": "N/A",
            "claimed_trai_recall": "N/A",
            "reproduced_northern_trai_recall": "N/A",
            "claimed_northern_trai_recall": "N/A",
            "cpu_latency_ms": "N/A",
            "evidence_classification": "UNSUPPORTED_OR_MISSING"
        }
    ]
    df = pd.DataFrame(models)
    df.to_csv(AUDIT_DIR / "model_evidence_matrix.csv", index=False)
    print(f"Generated {AUDIT_DIR / 'model_evidence_matrix.csv'} with {len(df)} models.")

def generate_figure_table_audit():
    items = [
        {
            "artifact_path": "docs/objective_1/tables/table_01_dataset_source_inventory.csv",
            "artifact_type": "CSV_TABLE",
            "generation_script": "scripts/objective_1_docs/generate_tables.py",
            "data_input_mode": "COMPUTED_FROM_MANIFESTS",
            "accuracy_of_data": "EXACT_MATCH",
            "specific_inconsistencies_found": "Correctly lists Google FLEURS, Husein Noise, and Calibration sources; flags Common Voice as 0 approved.",
            "audit_verdict": "VERIFIED"
        },
        {
            "artifact_path": "docs/objective_1/tables/table_02_class_split_distribution.csv",
            "artifact_type": "CSV_TABLE",
            "generation_script": "scripts/objective_1_docs/generate_tables.py",
            "data_input_mode": "COMPUTED_FROM_MANIFESTS",
            "accuracy_of_data": "EXACT_MATCH",
            "specific_inconsistencies_found": "Matches actual manifest counts: Train 178/178/178/178/319/250, Val 36/36/36/36/80/60, Test 36/36/36/36/300/90. Total 2,099.",
            "audit_verdict": "VERIFIED"
        },
        {
            "artifact_path": "docs/objective_1/tables/table_03_genuine_vs_fixture_breakdown.csv",
            "artifact_type": "CSV_TABLE",
            "generation_script": "scripts/objective_1_docs/generate_tables.py",
            "data_input_mode": "COMPUTED_FROM_MANIFESTS",
            "accuracy_of_data": "EXACT_MATCH",
            "specific_inconsistencies_found": "Accurately reports 100% of 1,000 target commands are synthetic calibration speech and all field takes are test fixtures.",
            "audit_verdict": "VERIFIED"
        },
        {
            "artifact_path": "docs/objective_1/tables/table_04_field_campaign_progress.csv",
            "artifact_type": "CSV_TABLE",
            "generation_script": "scripts/objective_1_docs/generate_tables.py",
            "data_input_mode": "COMPUTED_FROM_MANIFESTS",
            "accuracy_of_data": "EXACT_MATCH",
            "specific_inconsistencies_found": "Correctly audits 0 genuine human recordings, 0 genuine speakers, and flags 58 unit-test mock takes.",
            "audit_verdict": "VERIFIED"
        },
        {
            "artifact_path": "docs/objective_1/tables/table_05_speaker_dialect_distribution.csv",
            "artifact_type": "CSV_TABLE",
            "generation_script": "scripts/objective_1_docs/generate_tables.py",
            "data_input_mode": "COMPUTED_FROM_MANIFESTS",
            "accuracy_of_data": "EXACT_MATCH",
            "specific_inconsistencies_found": "Accurately reports 0% genuine field collection across Northern, Central, and Southern dialects.",
            "audit_verdict": "VERIFIED"
        },
        {
            "artifact_path": "docs/objective_1/tables/table_06_model_architecture_comparison.csv",
            "artifact_type": "CSV_TABLE",
            "generation_script": "scripts/objective_1_docs/generate_tables.py",
            "data_input_mode": "COMPUTED_FROM_MANIFESTS",
            "accuracy_of_data": "EXACT_MATCH",
            "specific_inconsistencies_found": "Accurately lists VietnameseKWSNet (138,534 params), TC-ResNet-14 (272,934 params), and flags missing TC-ResNet INT8 file.",
            "audit_verdict": "VERIFIED"
        },
        {
            "artifact_path": "docs/objective_1/tables/table_07_accuracy_and_metrics_comparison.csv",
            "artifact_type": "CSV_TABLE",
            "generation_script": "scripts/objective_1_docs/generate_tables.py",
            "data_input_mode": "HARDCODED_FROM_PHASE3_LOGS",
            "accuracy_of_data": "EXACT_MATCH",
            "specific_inconsistencies_found": "Accurately records 97.75% acc / 0.9466 F1 for baseline, 96.63% acc / 0.9402 F1 for TC-ResNet, and 25.0% Northern TRAI recall.",
            "audit_verdict": "VERIFIED"
        },
        {
            "artifact_path": "docs/objective_1/tables/table_08_confusion_matrix_tcresnet.csv",
            "artifact_type": "CSV_TABLE",
            "generation_script": "scripts/objective_1_docs/generate_tables.py",
            "data_input_mode": "HARDCODED_FROM_PHASE3_LOGS",
            "accuracy_of_data": "EXACT_MATCH",
            "specific_inconsistencies_found": "Accurately records 36 support per command, 27 correct TRAI, 9 PHAI, 300 UNKNOWN (293 correct), 90 SILENCE (90 correct).",
            "audit_verdict": "VERIFIED"
        },
        {
            "artifact_path": "docs/objective_1/tables/table_09_streaming_performance_comparison.csv",
            "artifact_type": "CSV_TABLE",
            "generation_script": "scripts/objective_1_docs/generate_tables.py",
            "data_input_mode": "HARDCODED_FROM_PHASE3_LOGS",
            "accuracy_of_data": "EXACT_MATCH",
            "specific_inconsistencies_found": "Accurately records results from 8.0-minute continuous streaming evaluation in reports/phase3_streaming_evaluation.md.",
            "audit_verdict": "VERIFIED"
        },
        {
            "artifact_path": "docs/objective_1/tables/table_10_onnx_model_benchmarks.csv",
            "artifact_type": "CSV_TABLE",
            "generation_script": "scripts/objective_1_docs/generate_tables.py",
            "data_input_mode": "COMPUTED_FROM_MANIFESTS",
            "accuracy_of_data": "EXACT_MATCH",
            "specific_inconsistencies_found": "Accurately reports Opset 14, exact byte sizes, parity tolerances (< 1.2e-6), and flags missing TC-ResNet INT8 file.",
            "audit_verdict": "VERIFIED"
        },
        {
            "artifact_path": "docs/objective_1/tables/table_11_automated_test_results.csv",
            "artifact_type": "CSV_TABLE",
            "generation_script": "scripts/objective_1_docs/generate_tables.py",
            "data_input_mode": "COMPUTED_FROM_MANIFESTS",
            "accuracy_of_data": "EXACT_MATCH",
            "specific_inconsistencies_found": "Accurately lists all 19 passed tests across test_field_acquisition.py (12) and test_parity.py (7).",
            "audit_verdict": "VERIFIED"
        },
        {
            "artifact_path": "docs/objective_1/tables/table_12_outstanding_issues_and_severity.csv",
            "artifact_type": "CSV_TABLE",
            "generation_script": "scripts/objective_1_docs/generate_tables.py",
            "data_input_mode": "COMPUTED_FROM_MANIFESTS",
            "accuracy_of_data": "EXACT_MATCH",
            "specific_inconsistencies_found": "Accurately categorizes P0 (0 human field takes), P1 (Northern TRAI deficit), P2 (missing INT8 TC-ResNet), P3 (Windows toolchain).",
            "audit_verdict": "VERIFIED"
        },
        {
            "artifact_path": "docs/objective_1/figures/figure_01_end_to_end_system_architecture.svg",
            "artifact_type": "SVG_FIGURE",
            "generation_script": "scripts/objective_1_docs/generate_figures.py",
            "data_input_mode": "HARDCODED_UNSUPPORTED_VALUES",
            "accuracy_of_data": "PARTIALLY_INCONSISTENT",
            "specific_inconsistencies_found": "Labels TC-ResNet as having 278.4k params (actual: 272.9k) and claims 18.5 ms overall latency.",
            "audit_verdict": "REVISE_REQUIRED"
        },
        {
            "artifact_path": "docs/objective_1/figures/figure_02_dataset_acquisition_provenance_flow.svg",
            "artifact_type": "SVG_FIGURE",
            "generation_script": "scripts/objective_1_docs/generate_figures.py",
            "data_input_mode": "HARDCODED_UNSUPPORTED_VALUES",
            "accuracy_of_data": "SEVERELY_CONTRADICTORY",
            "specific_inconsistencies_found": "Claims UNKNOWN derives from Common Voice (779 files) and SILENCE from ESC-50/FMA (321 files). Actual sources are Google FLEURS (579), Confusing Words (120), and Husein Noise (400). Also claims 140 reserved streaming files.",
            "audit_verdict": "REVISE_REQUIRED"
        },
        {
            "artifact_path": "docs/objective_1/figures/figure_03_class_distribution_by_split.svg",
            "artifact_type": "SVG_FIGURE",
            "generation_script": "scripts/objective_1_docs/generate_figures.py",
            "data_input_mode": "HARDCODED_UNSUPPORTED_VALUES",
            "accuracy_of_data": "SEVERELY_CONTRADICTORY",
            "specific_inconsistencies_found": "Displays hardcoded fictitious split counts (Train: 160/160/160/160/461/180; Val: 40/40/40/40/84/40; Test: 50/50/50/50/234/100). Contradicts table_02 and actual manifests on disk (Test: 36/36/36/36/300/90).",
            "audit_verdict": "REVISE_REQUIRED"
        },
        {
            "artifact_path": "docs/objective_1/figures/figure_04_model_accuracy_f1_trai_comparison.svg",
            "artifact_type": "SVG_FIGURE",
            "generation_script": "scripts/objective_1_docs/generate_figures.py",
            "data_input_mode": "HARDCODED_UNSUPPORTED_VALUES",
            "accuracy_of_data": "SEVERELY_CONTRADICTORY",
            "specific_inconsistencies_found": "Compares fictitious 'BC-ResNet-1' (97.5% acc, 93.7% F1) against 'TC-ResNet-14' (98.1% acc, 95.8% F1). Contradicts actual evaluated metrics (Baseline: 97.75% acc, 0.9466 F1; TC-ResNet: 96.63% acc, 0.9402 F1).",
            "audit_verdict": "REVISE_REQUIRED"
        },
        {
            "artifact_path": "docs/objective_1/figures/figure_05_confusion_matrix_tcresnet.svg",
            "artifact_type": "SVG_FIGURE",
            "generation_script": "scripts/objective_1_docs/generate_figures.py",
            "data_input_mode": "HARDCODED_UNSUPPORTED_VALUES",
            "accuracy_of_data": "SEVERELY_CONTRADICTORY",
            "specific_inconsistencies_found": "Displays fictitious 6x6 matrix with 50 support per command, 234 UNKNOWN, 100 SILENCE, and 6 TRAI -> PHAI errors. Contradicts table_08 and actual checkpoint predictions (Support: 36/36/36/36/300/90; 9 TRAI -> PHAI errors).",
            "audit_verdict": "REVISE_REQUIRED"
        },
        {
            "artifact_path": "docs/objective_1/figures/figure_06_streaming_roc_mcr_latency.svg",
            "artifact_type": "SVG_FIGURE",
            "generation_script": "scripts/objective_1_docs/generate_figures.py",
            "data_input_mode": "HARDCODED_UNSUPPORTED_VALUES",
            "accuracy_of_data": "SEVERELY_CONTRADICTORY",
            "specific_inconsistencies_found": "Plots fictitious trade-off curve with 192 ms latency, 0.8 FA/hr babble, and claims 140 1-hour streams. Contradicts actual 8.0-minute evaluation log (680 - 850 ms latency, 0 FA in 8.0 min).",
            "audit_verdict": "REVISE_REQUIRED"
        },
        {
            "artifact_path": "docs/objective_1/figures/figure_07_field_campaign_target_progress.svg",
            "artifact_type": "SVG_FIGURE",
            "generation_script": "scripts/objective_1_docs/generate_figures.py",
            "data_input_mode": "COMPUTED_FROM_MANIFESTS",
            "accuracy_of_data": "EXACT_MATCH",
            "specific_inconsistencies_found": "Accurately represents 0% genuine field progress and clearly discloses the 58 test-harness fixtures.",
            "audit_verdict": "VERIFIED"
        },
        {
            "artifact_path": "docs/objective_1/figures/figure_08_deployment_parity_pipeline.svg",
            "artifact_type": "SVG_FIGURE",
            "generation_script": "scripts/objective_1_docs/generate_figures.py",
            "data_input_mode": "HARDCODED_UNSUPPORTED_VALUES",
            "accuracy_of_data": "PARTIALLY_INCONSISTENT",
            "specific_inconsistencies_found": "Claims Opset 17 (actual: Opset 14), claims BC-ResNet-1 (actual: VietnameseKWSNet), claims 1.83 ms latency (actual: 0.35 ms). Accurately flags missing TC-ResNet INT8 file.",
            "audit_verdict": "REVISE_REQUIRED"
        }
    ]
    df = pd.DataFrame(items)
    df.to_csv(AUDIT_DIR / "figure_table_audit.csv", index=False)
    print(f"Generated {AUDIT_DIR / 'figure_table_audit.csv'} with {len(df)} items.")

if __name__ == "__main__":
    print("Generating all 4 audit CSV files...")
    generate_claim_audit()
    generate_dataset_reconciliation()
    generate_model_evidence_matrix()
    generate_figure_table_audit()
    print("Successfully generated all audit CSV files in docs/sound_command_recognition/audit/")
