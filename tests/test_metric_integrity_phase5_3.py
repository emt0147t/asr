"""
Phase 5.3 Unit Tests: Final Metric Semantics & Latency Association Integrity
Repository: D:\trainidentificationaudio

This test suite verifies:
1. Standard Levenshtein sequence WER vs Temporal Event matching discrepancy handling.
2. Overlapping timing windows and repeated commands (LEN -> LEN, gap=0ms).
3. Wrong-label detection classification (Substitution: 1 FP and 1 FN in event matching).
4. Empty ground truth (N_gt = 0 -> Recall, Precision, F1, Sequence WER, Exact Match = None / N/A).
5. Multiple detections competing for one ground-truth command (strict 1-to-1 bijection).
6. Latency association using strictly the documented one-to-one matching rule.

Governance Labels:
- EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED
- NOT_VALIDATED_ON_NATURAL_SPEECH
- NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION
- NOT_VALIDATED_PENDING_DATA
"""

import sys
import pytest
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4" / "demo"
SCRIPTS_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4" / "scripts"

for p in [str(PROJECT_ROOT), str(DEMO_DIR), str(SCRIPTS_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from audit_phase5_3_final_metrics import (
    compute_standard_sequence_wer, compute_temporal_event_metrics
)


class TestPhase53MetricSemantics:
    """Rigorous unit tests for Phase 5.3 metric semantics and latency association."""

    def test_levenshtein_wer_vs_temporal_event_matching_discrepancy(self):
        """
        Verify that Sequence Levenshtein WER and Temporal Event Matching can legitimately differ
        when acoustic lag shifts an event past its ground-truth tolerance window into an adjacent window.
        """
        # Ground truth: 3 commands
        gt = [
            {"command": "TRAI", "start_sec": 0.50, "end_sec": 0.80},
            {"command": "LEN", "start_sec": 0.85, "end_sec": 1.15},    # window: [0.70, 1.70]
            {"command": "XUONG", "start_sec": 1.40, "end_sec": 1.70}  # window: [1.25, 2.25]
        ]
        # Prediction: TRAI at 0.82 (matches GT 0), LEN delayed to 1.85 (past GT 1 window, inside GT 2 window!)
        pred = [
            {"command": "TRAI", "timestamp_sec": 0.82, "infer_latency_ms": 2.0},
            {"command": "LEN", "timestamp_sec": 1.85, "infer_latency_ms": 2.0}
        ]

        gt_cmds = [g["command"] for g in gt]
        det_cmds = [p["command"] for p in pred]

        # 1. Path A: Standard Sequence Levenshtein (ignoring timestamps)
        # Ref: ['TRAI', 'LEN', 'XUONG'], Hyp: ['TRAI', 'LEN']
        # Matches: TRAI->TRAI, LEN->LEN, XUONG deleted
        seq_res = compute_standard_sequence_wer(gt_cmds, det_cmds)
        assert seq_res["seq_s"] == 0
        assert seq_res["seq_d"] == 1
        assert seq_res["seq_ins"] == 0
        assert seq_res["seq_wer_pct"] == 33.3

        # 2. Path B: Temporal Event Matching (enforcing [-0.15s, +0.55s] windows)
        # Det 0 ('TRAI' at 0.82): Matches GT 0 ('TRAI') -> TP=1
        # Det 1 ('LEN' at 1.85): Distance to GT 1 (1.15) is +0.70s (> 0.55s) -> CANNOT match GT 1!
        # Distance to GT 2 (1.70) is +0.15s (within window) -> Matches GT 2 with wrong label ('LEN' != 'XUONG') -> SUB=1!
        # GT 1 is unreached -> DEL=1!
        ev_res = compute_temporal_event_metrics(pred, gt)
        assert ev_res["event_tp"] == 1
        assert ev_res["event_sub"] == 1
        assert ev_res["event_del"] == 1
        assert ev_res["event_ins"] == 0
        assert ev_res["event_fp"] == 1  # FP = INS + SUB = 0 + 1 = 1
        assert ev_res["event_fn"] == 2  # FN = DEL + SUB = 1 + 1 = 2

        # Verify discrepancy is preserved and understood
        assert seq_res["seq_s"] != ev_res["event_sub"]
        assert seq_res["seq_wer_pct"] != round((ev_res["event_sub"] + ev_res["event_del"] + ev_res["event_ins"]) / len(gt) * 100.0, 1)

    def test_overlapping_timing_windows_and_repeated_commands(self):
        """Repeated identical commands with 0ms gap (LEN -> LEN -> PHAI)."""
        gt = [
            {"command": "LEN", "start_sec": 0.50, "end_sec": 0.85},
            {"command": "LEN", "start_sec": 0.85, "end_sec": 1.20},
            {"command": "PHAI", "start_sec": 1.20, "end_sec": 1.55}
        ]
        pred = [
            {"command": "LEN", "timestamp_sec": 0.89, "infer_latency_ms": 2.2},
            {"command": "LEN", "timestamp_sec": 1.24, "infer_latency_ms": 2.2},
            {"command": "PHAI", "timestamp_sec": 1.59, "infer_latency_ms": 2.2}
        ]

        seq_res = compute_standard_sequence_wer([g["command"] for g in gt], [p["command"] for p in pred])
        assert seq_res["seq_wer_pct"] == 0.0
        assert seq_res["exact_sequence_match"] is True

        ev_res = compute_temporal_event_metrics(pred, gt)
        assert ev_res["event_tp"] == 3
        assert ev_res["event_fp"] == 0
        assert ev_res["event_fn"] == 0
        assert ev_res["event_precision_pct"] == 100.0
        assert ev_res["event_recall_pct"] == 100.0
        assert ev_res["event_f1_pct"] == 100.0

    def test_wrong_label_detection_classification(self):
        """A wrong-label prediction inside a GT window is a Substitution, counting as 1 FP and 1 FN."""
        gt = [
            {"command": "LEN", "start_sec": 0.50, "end_sec": 0.85}
        ]
        pred = [
            {"command": "XUONG", "timestamp_sec": 0.88, "infer_latency_ms": 2.2}
        ]

        # Sequence WER: 1 substitution
        seq_res = compute_standard_sequence_wer(["LEN"], ["XUONG"])
        assert seq_res["seq_s"] == 1
        assert seq_res["seq_d"] == 0
        assert seq_res["seq_ins"] == 0
        assert seq_res["seq_wer_pct"] == 100.0

        # Temporal Event: TP=0, SUB=1, FP=1, FN=1, Precision=0.0%, Recall=0.0%
        ev_res = compute_temporal_event_metrics(pred, gt)
        assert ev_res["event_tp"] == 0
        assert ev_res["event_sub"] == 1
        assert ev_res["event_del"] == 0
        assert ev_res["event_ins"] == 0
        assert ev_res["event_fp"] == 1  # FP = INS + SUB = 0 + 1 = 1
        assert ev_res["event_fn"] == 1  # FN = DEL + SUB = 0 + 1 = 1
        assert ev_res["event_precision_pct"] == 0.0
        assert ev_res["event_recall_pct"] == 0.0
        assert ev_res["event_f1_pct"] == 0.0

    def test_zero_ground_truth_policy(self):
        """When N_gt = 0 (e.g. closed game gate), metrics must report None (N/A) without division by zero."""
        # Case A: Pure idle silence, no detections
        seq_res_empty = compute_standard_sequence_wer([], [])
        assert seq_res_empty["n_gt"] == 0
        assert seq_res_empty["seq_wer_pct"] is None
        assert seq_res_empty["exact_sequence_match"] is None

        ev_res_empty = compute_temporal_event_metrics([], [])
        assert ev_res_empty["n_gt"] == 0
        assert ev_res_empty["event_precision_pct"] is None
        assert ev_res_empty["event_recall_pct"] is None
        assert ev_res_empty["event_f1_pct"] is None
        assert ev_res_empty["event_fp"] == 0

        # Case B: Spurious false activation during closed gate
        pred_spurious = [{"command": "LEN", "timestamp_sec": 1.0, "infer_latency_ms": 2.0}]
        seq_res_spurious = compute_standard_sequence_wer([], ["LEN"])
        assert seq_res_spurious["n_gt"] == 0
        assert seq_res_spurious["seq_ins"] == 1
        assert seq_res_spurious["seq_wer_pct"] is None

        ev_res_spurious = compute_temporal_event_metrics(pred_spurious, [])
        assert ev_res_spurious["n_gt"] == 0
        assert ev_res_spurious["event_ins"] == 1
        assert ev_res_spurious["event_fp"] == 1
        assert ev_res_spurious["event_precision_pct"] is None
        assert ev_res_spurious["event_recall_pct"] is None

    def test_multiple_detections_competing_for_one_ground_truth(self):
        """Multiple detections competing for one GT command: strict 1-to-1 injection."""
        gt = [
            {"command": "LEN", "start_sec": 0.50, "end_sec": 0.85}
        ]
        # Three detections inside the single GT window
        pred = [
            {"command": "LEN", "timestamp_sec": 0.88, "infer_latency_ms": 2.0},
            {"command": "LEN", "timestamp_sec": 0.92, "infer_latency_ms": 2.0},
            {"command": "LEN", "timestamp_sec": 0.98, "infer_latency_ms": 2.0}
        ]

        ev_res = compute_temporal_event_metrics(pred, gt)
        # Exactly ONE detection matches GT as True Positive
        # The remaining TWO detections must be counted as Insertions
        assert ev_res["event_tp"] == 1
        assert ev_res["event_ins"] == 2
        assert ev_res["event_fp"] == 2
        assert ev_res["event_del"] == 0
        assert ev_res["event_sub"] == 0
        assert len(ev_res["matches"]) == 1

        seq_res = compute_standard_sequence_wer(["LEN"], ["LEN", "LEN", "LEN"])
        assert seq_res["seq_ins"] == 2
        assert seq_res["seq_wer_pct"] == 200.0

    def test_latency_association_strictly_via_matched_pairs(self):
        """Verify that latency is extracted strictly from matched pairs (j, i) in M, never independent nearest neighbors."""
        gt = [
            {"command": "LEN", "start_sec": 0.50, "end_sec": 0.85},
            {"command": "PHAI", "start_sec": 1.00, "end_sec": 1.35}
        ]
        pred = [
            {"command": "LEN", "timestamp_sec": 0.89, "infer_latency_ms": 2.5},     # matched to GT 0
            {"command": "XUONG", "timestamp_sec": 3.00, "infer_latency_ms": 2.0},   # unmatched insertion!
            {"command": "PHAI", "timestamp_sec": 1.40, "infer_latency_ms": 3.0}     # matched to GT 1
        ]

        ev_res = compute_temporal_event_metrics(pred, gt)
        # Even though there are 3 detections, only 2 matched pairs exist
        assert len(ev_res["matches"]) == 2
        assert len(ev_res["lag_latencies"]) == 2
        assert len(ev_res["e2e_latencies"]) == 2

        # The unmatched insertion at 3.00s MUST NOT appear in latency arrays
        lag_times = ev_res["lag_latencies"]
        e2e_times = ev_res["e2e_latencies"]

        assert round(lag_times[0], 1) == 40.0
        assert round(e2e_times[0], 1) == 42.5
        assert round(lag_times[1], 1) == 50.0
        assert round(e2e_times[1], 1) == 53.0

        for lag, e2e in zip(lag_times, e2e_times):
            assert e2e > lag
