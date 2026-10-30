"""
Phase 5.1 Synthetic Metric Unit Tests: Metric Integrity & Sequence Alignment Verification
Repository: D:\trainidentificationaudio

This test suite verifies:
1. Exact match (0 sequence errors, WER = 0%)
2. Single deletion (D=1, WER = 50%)
3. Single insertion (I=1, WER = 50%)
4. Single substitution (S=1, WER = 50%, FP=1, FN=1)
5. Multiple errors (S, D, I combination)
6. Edge cases: empty predictions, empty ground truth, duplicate detections in tolerance window, out-of-order detections
7. Mathematical identity: WER == 100% - Recall iff I == 0
8. Time-window event matching vs Levenshtein sequence alignment comparison

Governance Labels:
- EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED
- NOT_VALIDATED_ON_NATURAL_SPEECH
- NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION
- NOT_VALIDATED_PENDING_DATA
"""

import sys
import pytest
from pathlib import Path
from typing import List, Dict, Any, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4" / "demo"
SCRIPTS_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4" / "scripts"

for p in [str(PROJECT_ROOT), str(DEMO_DIR), str(SCRIPTS_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from benchmark_phase5_audit import match_events_rigorous
from audit_phase5_1_metrics import levenshtein_align


class TestSyntheticMetricIntegrity:
    """Rigorous unit tests for KWS metric calculation integrity."""

    def test_exact_match_zero_errors(self):
        """GT ['LEN', 'PHAI'], Pred ['LEN', 'PHAI'] -> 0 errors, WER = 0%, Exact Match = True."""
        gt = [
            {"command": "LEN", "start_sec": 0.5, "end_sec": 0.85, "center_sec": 0.675},
            {"command": "PHAI", "start_sec": 0.9, "end_sec": 1.25, "center_sec": 1.075}
        ]
        pred = [
            {"command": "LEN", "timestamp_sec": 0.88, "infer_latency_ms": 2.0},
            {"command": "PHAI", "timestamp_sec": 1.28, "infer_latency_ms": 2.0}
        ]

        gt_cmds = [g["command"] for g in gt]
        pred_cmds = [p["command"] for p in pred]

        # 1. Levenshtein Sequence Alignment
        s_lev, d_lev, i_lev = levenshtein_align(gt_cmds, pred_cmds)
        assert s_lev == 0
        assert d_lev == 0
        assert i_lev == 0
        wer_lev = (s_lev + d_lev + i_lev) / len(gt)
        assert wer_lev == 0.0

        # 2. Time-Window Event Matching
        res = match_events_rigorous(pred, gt)
        assert res["tp"] == 2
        assert res["fp"] == 0
        assert res["fn"] == 0
        assert res["sub"] == 0
        assert res["del"] == 0
        assert res["ins"] == 0

        recall = res["tp"] / len(gt)
        precision = res["tp"] / (res["tp"] + res["fp"])
        wer_event = (res["sub"] + res["del"] + res["ins"]) / len(gt)

        assert recall == 1.0
        assert precision == 1.0
        assert wer_event == 0.0
        assert (pred_cmds == gt_cmds) is True

    def test_single_deletion(self):
        """GT ['LEN', 'PHAI'], Pred ['LEN'] -> D = 1, WER = 50%, Exact Match = False."""
        gt = [
            {"command": "LEN", "start_sec": 0.5, "end_sec": 0.85, "center_sec": 0.675},
            {"command": "PHAI", "start_sec": 0.9, "end_sec": 1.25, "center_sec": 1.075}
        ]
        pred = [
            {"command": "LEN", "timestamp_sec": 0.88, "infer_latency_ms": 2.0}
        ]

        gt_cmds = [g["command"] for g in gt]
        pred_cmds = [p["command"] for p in pred]

        # 1. Levenshtein Sequence Alignment
        s_lev, d_lev, i_lev = levenshtein_align(gt_cmds, pred_cmds)
        assert s_lev == 0
        assert d_lev == 1
        assert i_lev == 0
        wer_lev = (s_lev + d_lev + i_lev) / len(gt)
        assert wer_lev == 0.5

        # 2. Time-Window Event Matching
        res = match_events_rigorous(pred, gt)
        assert res["tp"] == 1
        assert res["fp"] == 0
        assert res["fn"] == 1  # FN = DEL + SUB = 1 + 0 = 1
        assert res["sub"] == 0
        assert res["del"] == 1
        assert res["ins"] == 0

        recall = res["tp"] / len(gt)
        precision = res["tp"] / (res["tp"] + res["fp"])
        wer_event = (res["sub"] + res["del"] + res["ins"]) / len(gt)

        assert recall == 0.5
        assert precision == 1.0
        assert wer_event == 0.5
        assert (pred_cmds == gt_cmds) is False

    def test_single_insertion(self):
        """GT ['LEN', 'PHAI'], Pred ['LEN', 'XUONG', 'PHAI'] -> I = 1, WER = 50%, Exact Match = False."""
        gt = [
            {"command": "LEN", "start_sec": 0.5, "end_sec": 0.85, "center_sec": 0.675},
            {"command": "PHAI", "start_sec": 2.0, "end_sec": 2.35, "center_sec": 2.175}
        ]
        # XUONG is inserted at t=1.40 (outside tolerance window [-0.15, +0.55] of both words)
        pred = [
            {"command": "LEN", "timestamp_sec": 0.88, "infer_latency_ms": 2.0},
            {"command": "XUONG", "timestamp_sec": 1.40, "infer_latency_ms": 2.0},
            {"command": "PHAI", "timestamp_sec": 2.38, "infer_latency_ms": 2.0}
        ]

        gt_cmds = [g["command"] for g in gt]
        pred_cmds = [p["command"] for p in pred]

        # 1. Levenshtein Sequence Alignment
        s_lev, d_lev, i_lev = levenshtein_align(gt_cmds, pred_cmds)
        assert s_lev == 0
        assert d_lev == 0
        assert i_lev == 1
        wer_lev = (s_lev + d_lev + i_lev) / len(gt)
        assert wer_lev == 0.5

        # 2. Time-Window Event Matching
        res = match_events_rigorous(pred, gt)
        assert res["tp"] == 2
        assert res["fp"] == 1  # FP = INS + SUB = 1 + 0 = 1
        assert res["fn"] == 0
        assert res["sub"] == 0
        assert res["del"] == 0
        assert res["ins"] == 1

        recall = res["tp"] / len(gt)
        precision = res["tp"] / (res["tp"] + res["fp"])
        wer_event = (res["sub"] + res["del"] + res["ins"]) / len(gt)

        assert recall == 1.0
        assert precision == 2 / 3
        assert wer_event == 0.5
        # Notice: When I > 0, WER (50%) != 100% - Recall (0%)!
        assert wer_event != (1.0 - recall)

    def test_single_substitution(self):
        """GT ['LEN', 'PHAI'], Pred ['LEN', 'XUONG'] -> S = 1, WER = 50%, Exact Match = False."""
        gt = [
            {"command": "LEN", "start_sec": 0.5, "end_sec": 0.85, "center_sec": 0.675},
            {"command": "PHAI", "start_sec": 0.9, "end_sec": 1.25, "center_sec": 1.075}
        ]
        # XUONG temporally matches PHAI (dist = 1.28 - 1.25 = +0.03s), but has wrong class
        pred = [
            {"command": "LEN", "timestamp_sec": 0.88, "infer_latency_ms": 2.0},
            {"command": "XUONG", "timestamp_sec": 1.28, "infer_latency_ms": 2.0}
        ]

        gt_cmds = [g["command"] for g in gt]
        pred_cmds = [p["command"] for p in pred]

        # 1. Levenshtein Sequence Alignment
        s_lev, d_lev, i_lev = levenshtein_align(gt_cmds, pred_cmds)
        assert s_lev == 1
        assert d_lev == 0
        assert i_lev == 0
        wer_lev = (s_lev + d_lev + i_lev) / len(gt)
        assert wer_lev == 0.5

        # 2. Time-Window Event Matching
        res = match_events_rigorous(pred, gt)
        assert res["tp"] == 1
        # IMPORTANT: A substitution counts as 1 FP AND 1 FN in event matching!
        assert res["sub"] == 1
        assert res["del"] == 0
        assert res["ins"] == 0
        assert res["fp"] == 1  # FP = INS + SUB = 0 + 1 = 1
        assert res["fn"] == 1  # FN = DEL + SUB = 0 + 1 = 1

        recall = res["tp"] / len(gt)
        precision = res["tp"] / (res["tp"] + res["fp"])
        wer_event = (res["sub"] + res["del"] + res["ins"]) / len(gt)

        assert recall == 0.5
        assert precision == 0.5
        assert wer_event == 0.5
        # Since I == 0, WER == 100% - Recall!
        assert wer_event == (1.0 - recall)

    def test_multiple_errors_compound(self):
        """GT ['LEN', 'PHAI', 'TRAI'], Pred ['LEN', 'XUONG', 'PHAI', 'PHAI'] -> multiple errors."""
        ref = ["LEN", "PHAI", "TRAI"]
        hyp = ["LEN", "XUONG", "PHAI", "PHAI"]

        s, d, ins = levenshtein_align(ref, hyp)
        # Optimal edit path:
        # LEN -> LEN (match)
        # insert XUONG (ins + 1)
        # PHAI -> PHAI (match)
        # TRAI -> PHAI (sub + 1)
        # Cost = 2 (1 INS, 1 SUB, 0 DEL)
        assert s == 1
        assert d == 0
        assert ins == 1
        wer = (s + d + ins) / len(ref)
        assert round(wer, 4) == round(2 / 3, 4)

    def test_edge_case_empty_predictions(self):
        """Empty prediction list: all GT commands are deletions."""
        gt = [
            {"command": "LEN", "start_sec": 0.5, "end_sec": 0.85, "center_sec": 0.675},
            {"command": "PHAI", "start_sec": 0.9, "end_sec": 1.25, "center_sec": 1.075}
        ]
        pred = []

        res = match_events_rigorous(pred, gt)
        assert res["tp"] == 0
        assert res["fp"] == 0
        assert res["fn"] == 2
        assert res["del"] == 2
        assert res["sub"] == 0
        assert res["ins"] == 0

        s_lev, d_lev, i_lev = levenshtein_align([g["command"] for g in gt], [])
        assert d_lev == 2
        assert s_lev == 0
        assert i_lev == 0

    def test_edge_case_empty_ground_truth(self):
        """Empty ground truth (e.g. pure silence / negative test): all detections are insertions."""
        gt = []
        pred = [
            {"command": "LEN", "timestamp_sec": 1.0, "infer_latency_ms": 2.0},
            {"command": "PHAI", "timestamp_sec": 2.5, "infer_latency_ms": 2.0}
        ]

        res = match_events_rigorous(pred, gt)
        assert res["tp"] == 0
        assert res["fp"] == 2
        assert res["fn"] == 0
        assert res["ins"] == 2
        assert res["sub"] == 0
        assert res["del"] == 0

        s_lev, d_lev, i_lev = levenshtein_align([], ["LEN", "PHAI"])
        assert i_lev == 2
        assert s_lev == 0
        assert d_lev == 0

    def test_duplicate_detections_in_same_window(self):
        """Two detections for a single ground truth word: first matches (TP), second is insertion (FP)."""
        gt = [
            {"command": "LEN", "start_sec": 0.5, "end_sec": 0.85, "center_sec": 0.675}
        ]
        pred = [
            {"command": "LEN", "timestamp_sec": 0.88, "infer_latency_ms": 2.0},  # matches GT
            {"command": "LEN", "timestamp_sec": 0.95, "infer_latency_ms": 2.0}   # duplicate -> insertion!
        ]

        res = match_events_rigorous(pred, gt)
        assert res["tp"] == 1
        assert res["ins"] == 1
        assert res["fp"] == 1
        assert res["del"] == 0
        assert res["fn"] == 0

    def test_out_of_order_detections_temporal_resolution(self):
        """Temporal matching matches based on timestamp regardless of arrival order in list."""
        gt = [
            {"command": "LEN", "start_sec": 0.5, "end_sec": 0.85, "center_sec": 0.675},
            {"command": "PHAI", "start_sec": 2.0, "end_sec": 2.35, "center_sec": 2.175}
        ]
        # Arrives inverted: PHAI first, then LEN
        pred = [
            {"command": "PHAI", "timestamp_sec": 2.38, "infer_latency_ms": 2.0},
            {"command": "LEN", "timestamp_sec": 0.88, "infer_latency_ms": 2.0}
        ]

        res = match_events_rigorous(pred, gt)
        assert res["tp"] == 2
        assert res["fp"] == 0
        assert res["fn"] == 0
        assert res["ins"] == 0
        assert res["del"] == 0

    def test_mathematical_identity_wer_equals_complement_of_recall(self):
        """
        Proof test:
        N_gt = TP + S + D.
        Recall = TP / N_gt.
        WER = (S + D + I) / N_gt = (N_gt - TP + I) / N_gt = 1 - Recall + (I / N_gt).
        Therefore: WER == 100% - Recall IFF I == 0.
        """
        for tp in range(0, 11):
            for s in range(0, 5):
                for d in range(0, 5):
                    n_gt = tp + s + d
                    if n_gt == 0:
                        continue
                    # Case 1: I == 0
                    i = 0
                    recall = tp / n_gt
                    wer = (s + d + i) / n_gt
                    assert abs(wer - (1.0 - recall)) < 1e-9

                    # Case 2: I > 0
                    for i_nonzero in [1, 2, 3]:
                        wer_with_i = (s + d + i_nonzero) / n_gt
                        assert wer_with_i > (1.0 - recall)
                        assert abs(wer_with_i - ((1.0 - recall) + i_nonzero / n_gt)) < 1e-9
