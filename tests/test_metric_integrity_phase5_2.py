"""
Phase 5.2 Unit Tests: Event Matching & Latency Consistency
Repository: D:\trainidentificationaudio

This test suite verifies:
1. Repeated identical commands where multiple tolerance windows overlap.
2. Two predictions falling inside the same ground-truth window.
3. One prediction compatible with two adjacent ground-truth windows.
4. One missing command plus an extra prediction.
5. A wrong-label prediction in an overlapping timing window.
6. Empty predictions and empty ground truth edge cases.
7. Strict one-to-one bijection and temporal monotonicity in matching.
8. Per-trial latency calculation consistency (t_lag < t_e2e, true per-trial distribution).

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

from audit_phase5_2_matching_latency import match_events_monotonic_one_to_one


class TestPhase52EventMatchingAndLatency:
    """Rigorous unit tests for Phase 5.2 event matching and latency consistency."""

    def test_repeated_identical_overlapping_tolerance_windows(self):
        """Repeated identical commands (LEN -> LEN, gap=0ms) where tolerance windows overlap."""
        gt = [
            {"command": "LEN", "start_sec": 0.50, "end_sec": 0.85, "center_sec": 0.675},
            {"command": "LEN", "start_sec": 0.85, "end_sec": 1.20, "center_sec": 1.025}
        ]
        # Two predictions emitted at valley points
        pred = [
            {"command": "LEN", "timestamp_sec": 0.88, "infer_latency_ms": 2.2},
            {"command": "LEN", "timestamp_sec": 1.23, "infer_latency_ms": 2.2}
        ]

        res = match_events_monotonic_one_to_one(pred, gt)
        # Expected: Strict one-to-one monotonic pairing
        # Det 0 -> GT 0, Det 1 -> GT 1
        assert res["tp"] == 2
        assert res["fp"] == 0
        assert res["fn"] == 0
        assert res["sub"] == 0
        assert res["ins"] == 0
        assert res["del"] == 0

        wer = (res["sub"] + res["del"] + res["ins"]) / len(gt)
        assert wer == 0.0
        assert [p["command"] for p in pred] == [g["command"] for g in gt]

    def test_two_predictions_falling_inside_same_window(self):
        """Two predictions falling inside the same ground-truth window (over-segmentation / bounce)."""
        gt = [
            {"command": "LEN", "start_sec": 0.50, "end_sec": 0.85, "center_sec": 0.675}
        ]
        # Double trigger: one at 0.88, another at 0.95
        pred = [
            {"command": "LEN", "timestamp_sec": 0.88, "infer_latency_ms": 2.2},
            {"command": "LEN", "timestamp_sec": 0.95, "infer_latency_ms": 2.2}
        ]

        res = match_events_monotonic_one_to_one(pred, gt)
        # 1-to-1 matching constraint: GT can match at most ONE prediction!
        # First matches GT (TP=1), second has no unmatched GT -> Insertion (I=1, FP=1)
        assert res["tp"] == 1
        assert res["ins"] == 1
        assert res["fp"] == 1
        assert res["sub"] == 0
        assert res["del"] == 0
        assert res["fn"] == 0

        wer = (res["sub"] + res["del"] + res["ins"]) / len(gt)
        assert wer == 1.0  # (0 + 0 + 1) / 1 = 100%
        assert len(res["matches"]) == 1

    def test_one_prediction_compatible_with_two_adjacent_windows(self):
        """One prediction compatible with two adjacent windows (LEN -> PHAI, only PHAI detected)."""
        gt = [
            {"command": "LEN", "start_sec": 0.50, "end_sec": 0.85, "center_sec": 0.675},
            {"command": "PHAI", "start_sec": 0.85, "end_sec": 1.20, "center_sec": 1.025}
        ]
        # Only one detection: PHAI at 1.05 (falls in GT 0 window [0.70, 1.40] AND GT 1 window [1.05, 1.75])
        pred = [
            {"command": "PHAI", "timestamp_sec": 1.05, "infer_latency_ms": 2.2}
        ]

        res = match_events_monotonic_one_to_one(pred, gt)
        # Optimal matching must pair PHAI with GT 1 (PHAI) because label matches (TP=1),
        # instead of pairing with GT 0 (LEN) which would create a spurious substitution!
        assert res["tp"] == 1
        assert res["del"] == 1  # GT 0 deleted
        assert res["fn"] == 1
        assert res["sub"] == 0
        assert res["ins"] == 0
        assert res["fp"] == 0

        wer = (res["sub"] + res["del"] + res["ins"]) / len(gt)
        assert wer == 0.5  # (0 + 1 + 0) / 2 = 50%
        assert res["matches"][0][1] == 1  # Matched GT index 1 (PHAI)

    def test_one_missing_command_plus_extra_prediction(self):
        """One missing command plus an extra prediction (LEN -> PHAI, detected LEN at 0.88, XUONG at 3.00)."""
        gt = [
            {"command": "LEN", "start_sec": 0.50, "end_sec": 0.85, "center_sec": 0.675},
            {"command": "PHAI", "start_sec": 0.90, "end_sec": 1.25, "center_sec": 1.075}
        ]
        pred = [
            {"command": "LEN", "timestamp_sec": 0.88, "infer_latency_ms": 2.2},
            {"command": "XUONG", "timestamp_sec": 3.00, "infer_latency_ms": 2.2}
        ]

        res = match_events_monotonic_one_to_one(pred, gt)
        assert res["tp"] == 1   # LEN matches GT 0
        assert res["del"] == 1  # PHAI is missed (Deletion)
        assert res["ins"] == 1  # XUONG at 3.00 is an Insertion
        assert res["sub"] == 0
        assert res["fp"] == 1   # FP = INS + SUB = 1 + 0 = 1
        assert res["fn"] == 1   # FN = DEL + SUB = 1 + 0 = 1

        wer = (res["sub"] + res["del"] + res["ins"]) / len(gt)
        assert wer == 1.0  # (0 + 1 + 1) / 2 = 100%

    def test_wrong_label_prediction_in_overlapping_timing_window(self):
        """A wrong-label prediction in an overlapping timing window."""
        gt = [
            {"command": "LEN", "start_sec": 0.50, "end_sec": 0.85, "center_sec": 0.675},
            {"command": "PHAI", "start_sec": 0.90, "end_sec": 1.25, "center_sec": 1.075}
        ]
        # Only one prediction: XUONG at 1.15
        pred = [
            {"command": "XUONG", "timestamp_sec": 1.15, "infer_latency_ms": 2.2}
        ]

        res = match_events_monotonic_one_to_one(pred, gt)
        # XUONG matches GT 1 (closer, dist = -0.10) as Substitution
        # GT 0 is Deleted
        assert res["tp"] == 0
        assert res["sub"] == 1
        assert res["del"] == 1
        assert res["ins"] == 0
        assert res["fp"] == 1  # FP = INS + SUB = 0 + 1 = 1
        assert res["fn"] == 2  # FN = DEL + SUB = 1 + 1 = 2

        wer = (res["sub"] + res["del"] + res["ins"]) / len(gt)
        assert wer == 1.0  # (1 + 1 + 0) / 2 = 100%

    def test_empty_predictions_and_empty_ground_truth(self):
        """Edge cases for empty inputs."""
        # 1. Both empty
        res_both_empty = match_events_monotonic_one_to_one([], [])
        assert res_both_empty["tp"] == 0
        assert res_both_empty["fp"] == 0
        assert res_both_empty["fn"] == 0
        assert res_both_empty["sub"] == 0
        assert res_both_empty["del"] == 0
        assert res_both_empty["ins"] == 0

        # 2. Pred empty, GT present
        gt = [{"command": "LEN", "start_sec": 0.5, "end_sec": 0.85}]
        res_pred_empty = match_events_monotonic_one_to_one([], gt)
        assert res_pred_empty["del"] == 1
        assert res_pred_empty["fn"] == 1
        assert res_pred_empty["tp"] == 0

        # 3. GT empty, Pred present (Negative control activation)
        pred = [{"command": "LEN", "timestamp_sec": 1.0, "infer_latency_ms": 2.0}]
        res_gt_empty = match_events_monotonic_one_to_one(pred, [])
        assert res_gt_empty["ins"] == 1
        assert res_gt_empty["fp"] == 1
        assert res_gt_empty["tp"] == 0

    def test_monotonic_matching_strict_bijection(self):
        """Verify strict one-to-one bijection and strict temporal monotonicity."""
        gt = [
            {"command": "LEN", "start_sec": 0.5, "end_sec": 0.8},
            {"command": "XUONG", "start_sec": 1.0, "end_sec": 1.3},
            {"command": "TRAI", "start_sec": 1.5, "end_sec": 1.8},
            {"command": "PHAI", "start_sec": 2.0, "end_sec": 2.3}
        ]
        pred = [
            {"command": "LEN", "timestamp_sec": 0.85, "infer_latency_ms": 2.0},
            {"command": "XUONG", "timestamp_sec": 1.35, "infer_latency_ms": 2.0},
            {"command": "TRAI", "timestamp_sec": 1.85, "infer_latency_ms": 2.0},
            {"command": "PHAI", "timestamp_sec": 2.35, "infer_latency_ms": 2.0}
        ]

        res = match_events_monotonic_one_to_one(pred, gt)
        matches = res["matches"]
        assert len(matches) == 4

        det_indices = [m[0] for m in matches]
        gt_indices = [m[1] for m in matches]

        # Check uniqueness (one-to-one bijection)
        assert len(set(det_indices)) == len(det_indices)
        assert len(set(gt_indices)) == len(gt_indices)

        # Check strict monotonicity
        assert all(det_indices[i] < det_indices[i + 1] for i in range(len(det_indices) - 1))
        assert all(gt_indices[i] < gt_indices[i + 1] for i in range(len(gt_indices) - 1))

    def test_per_trial_latency_distribution_consistency(self):
        """Verify per-trial latency calculation: t_lag = t_det - t_offset, t_e2e = t_lag + t_infer."""
        gt = [
            {"command": "LEN", "start_sec": 0.50, "end_sec": 0.85},
            {"command": "PHAI", "start_sec": 1.00, "end_sec": 1.35}
        ]
        pred = [
            {"command": "LEN", "timestamp_sec": 0.89, "infer_latency_ms": 2.5},   # lag = 40 ms, e2e = 42.5 ms
            {"command": "PHAI", "timestamp_sec": 1.40, "infer_latency_ms": 3.0}   # lag = 50 ms, e2e = 53.0 ms
        ]

        res = match_events_monotonic_one_to_one(pred, gt)
        lag_times = res["lag_latencies"]
        e2e_times = res["e2e_latencies"]

        assert len(lag_times) == 2
        assert len(e2e_times) == 2

        # In every single trial, t_e2e MUST be strictly greater than t_lag when infer > 0
        for lag, e2e in zip(lag_times, e2e_times):
            assert e2e > lag

        assert round(lag_times[0], 1) == 40.0
        assert round(e2e_times[0], 1) == 42.5
        assert round(lag_times[1], 1) == 50.0
        assert round(e2e_times[1], 1) == 53.0

        # Percentile computed from the distribution
        p50_lag = float(np.percentile(lag_times, 50))
        p50_e2e = float(np.percentile(e2e_times, 50))
        assert p50_lag < p50_e2e
