"""
Event Matching & Annotation Agreement Engine for Continuous Vietnamese KWS.
Complies with docs/dataset_protocol.md and reports/phase9a3_annotation_agreement_protocol.md:
  - Temporal boundary matching: IoU >= 0.50 and center distance <= 100 ms.
  - Greedy bipartite matching with deterministic tie-breaking.
  - Globally optimal Maximum-Cardinality Maximum-Weight bipartite matching via scipy.optimize.linear_sum_assignment.
  - Layer 1 segmentation metrics: Precision, Recall, F1, Onset MAE, Offset MAE.
  - Layer 2 semantic agreement metrics: Contingency table, Observed Agreement (Po),
    Expected Agreement (Pe), Cohen's Kappa, Positive Specific Agreement (Ppos), Negative Specific Agreement (Pneg).
"""

import math
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    linear_sum_assignment = None


@dataclass
class AnnotationEvent:
    event_id: str
    start_time: float
    end_time: float
    label: str = "CONVERSATIONAL"
    intent: str = "CONVERSATIONAL"
    addressee: str = "HUMAN"
    target_lexeme: str = "NONE"
    phonetic_relation: str = "NONE"
    speaker_role: str = "PRIMARY_ENROLLED"


def compute_iou(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    """Computes temporal Intersection-over-Union between two 1D intervals."""
    if start_a >= end_a or start_b >= end_b or start_a < 0.0 or start_b < 0.0:
        return 0.0
    intersection = max(0.0, min(end_a, end_b) - max(start_a, start_b))
    union = max(end_a, end_b) - min(start_a, start_b)
    if union <= 0.0:
        return 0.0
    return float(intersection / union)


def compute_center_distance(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    """Computes absolute difference in midpoints (in seconds)."""
    mid_a = (start_a + end_a) / 2.0
    mid_b = (start_b + end_b) / 2.0
    return float(abs(mid_a - mid_b))


def is_valid_candidate(
    start_a: float,
    end_a: float,
    start_b: float,
    end_b: float,
    iou_threshold: float = 0.50,
    center_dist_threshold_sec: float = 0.100
) -> Tuple[bool, float, float]:
    """
    Checks if two intervals satisfy Phase 9A.3 alignment criteria:
      1. IoU >= iou_threshold (default 0.50)
      2. |mid_a - mid_b| <= center_dist_threshold_sec (default 0.100s / 100ms)
    Returns (is_valid, iou, center_dist).
    """
    iou = compute_iou(start_a, end_a, start_b, end_b)
    cdist = compute_center_distance(start_a, end_a, start_b, end_b)
    is_valid = (iou >= iou_threshold) and (cdist <= center_dist_threshold_sec)
    return is_valid, iou, cdist


def validate_annotation_events(events: List[AnnotationEvent], list_name: str = "events") -> None:
    """
    Strictly validates annotation events under Phase 9A.6.3 contract:
      1. Every item must be an AnnotationEvent instance.
      2. Event IDs must be non-empty strings.
      3. All event IDs within each input list must be strictly unique.
      4. Timestamps (start_time, end_time) must be real numeric values (not bool, None, str).
      5. Neither start_time nor end_time may be NaN.
      6. Neither start_time nor end_time may be Infinite (+inf or -inf).
      7. Timestamps must be non-negative: start_time >= 0.0 and end_time >= 0.0.
      8. Interval duration must be strictly positive: end_time > start_time (rejects zero-duration and reversed).
    Raises TypeError on non-numeric or malformed types; raises ValueError on numeric/logical violations.
    """
    seen_ids = set()
    for idx, e in enumerate(events):
        if not isinstance(e, AnnotationEvent):
            raise TypeError(f"Element at index {idx} in {list_name} is not an AnnotationEvent: {type(e)}")

        # Event ID validation
        if not isinstance(e.event_id, str) or not e.event_id.strip():
            raise ValueError(f"Event at index {idx} in {list_name} has invalid or empty event_id: '{e.event_id}'")
        if e.event_id in seen_ids:
            raise ValueError(
                f"Duplicate event_id '{e.event_id}' detected in {list_name} at index {idx}. "
                "Event IDs must be strictly unique within each input list to guarantee deterministic matching."
            )
        seen_ids.add(e.event_id)

        # Type validation for timestamps (bool is subclass of int in Python, check explicitly)
        if isinstance(e.start_time, bool) or not isinstance(e.start_time, (int, float)):
            raise TypeError(f"Event '{e.event_id}' in {list_name} has non-numeric start_time: {type(e.start_time).__name__} ({e.start_time})")
        if isinstance(e.end_time, bool) or not isinstance(e.end_time, (int, float)):
            raise TypeError(f"Event '{e.event_id}' in {list_name} has non-numeric end_time: {type(e.end_time).__name__} ({e.end_time})")

        # NaN validation
        if math.isnan(e.start_time) or math.isnan(e.end_time):
            raise ValueError(f"Event '{e.event_id}' in {list_name} contains NaN timestamp: start={e.start_time}, end={e.end_time}")

        # Infinity validation
        if math.isinf(e.start_time) or math.isinf(e.end_time):
            raise ValueError(f"Event '{e.event_id}' in {list_name} contains infinite timestamp: start={e.start_time}, end={e.end_time}")

        # Negative timestamp validation
        if e.start_time < 0.0 or e.end_time < 0.0:
            raise ValueError(f"Event '{e.event_id}' in {list_name} has negative timestamp: start={e.start_time}, end={e.end_time}")

        # Zero-duration validation
        if e.start_time == e.end_time:
            raise ValueError(f"Event '{e.event_id}' in {list_name} has zero duration: start={e.start_time}, end={e.end_time}")

        # Reversed interval validation
        if e.end_time < e.start_time:
            raise ValueError(f"Event '{e.event_id}' in {list_name} has reversed interval (end < start): start={e.start_time}, end={e.end_time}")


def match_events_greedy(
    events_a: List[AnnotationEvent],
    events_b: List[AnnotationEvent],
    iou_threshold: float = 0.50,
    center_dist_threshold_sec: float = 0.100
) -> List[Dict[str, Any]]:
    """
    Greedy Bipartite Alignment:
      1. Validates event ID uniqueness in both input lists.
      2. Collects all candidate pairs satisfying IoU >= threshold and cdist <= threshold.
      3. Sorts descending by IoU, tie-breaking deterministically by ascending cdist, event_a.event_id, event_b.event_id.
      4. Greedily pairs highest IoU, removing matched indices.
    """
    validate_annotation_events(events_a, "events_a")
    validate_annotation_events(events_b, "events_b")

    candidates = []
    for i, ea in enumerate(events_a):
        for j, eb in enumerate(events_b):
            valid, iou, cdist = is_valid_candidate(
                ea.start_time, ea.end_time, eb.start_time, eb.end_time,
                iou_threshold=iou_threshold,
                center_dist_threshold_sec=center_dist_threshold_sec
            )
            if valid:
                candidates.append((iou, cdist, i, j, ea, eb))

    # Deterministic sort: -iou (highest first), cdist (lowest first), ea.event_id, eb.event_id
    candidates.sort(key=lambda x: (-x[0], x[1], str(x[4].event_id), str(x[5].event_id)))

    matched_a = set()
    matched_b = set()
    matches = []

    for iou, cdist, i, j, ea, eb in candidates:
        if i not in matched_a and j not in matched_b:
            matched_a.add(i)
            matched_b.add(j)
            matches.append({
                "index_a": i,
                "index_b": j,
                "event_a": ea,
                "event_b": eb,
                "iou": iou,
                "center_distance": cdist,
                "onset_diff": abs(ea.start_time - eb.start_time),
                "offset_diff": abs(ea.end_time - eb.end_time),
            })

    # Sort final matches deterministically by stable event identity
    matches.sort(key=lambda m: (m["event_a"].start_time, m["event_a"].end_time, str(m["event_a"].event_id)))
    return matches


def match_events_optimal(
    events_a: List[AnnotationEvent],
    events_b: List[AnnotationEvent],
    iou_threshold: float = 0.50,
    center_dist_threshold_sec: float = 0.100
) -> List[Dict[str, Any]]:
    """
    Maximum-Cardinality Maximum-Weight Bipartite Matching:
      Formulated as Hungarian assignment via scipy.optimize.linear_sum_assignment.
      Assigns priority to cardinality by adding large weight K to all valid candidate edges:
        Benefit(i, j) = K + IoU(i, j) + tie_break
      where K = max(1000.0, float(max(n_a, n_b) * 10.0)).
      Cost matrix: C(i, j) = max_benefit - Benefit(i, j).
      Guarantees lexicographic optimization:
        1. Maximize number of valid matches |M|.
        2. Maximize sum of IoU weights.
        3. Minimize center distance.
        4. Deterministic stable-ID tie-breaking invariant under input permutation.
    """
    validate_annotation_events(events_a, "events_a")
    validate_annotation_events(events_b, "events_b")

    if not _HAS_SCIPY:
        raise ImportError(
            "scipy>=1.10.0 is required for optimal Hungarian bipartite event matching. "
            "Please ensure scipy is installed according to requirements.txt."
        )

    n_a = len(events_a)
    n_b = len(events_b)
    if n_a == 0 or n_b == 0:
        return []

    # Canonical sorting by (start_time, end_time, event_id).
    # Since event IDs are strictly validated to be unique within each list,
    # this sort key is strictly unique across all events, making the canonical
    # ordering 100% permutation-invariant without any reliance on original input index.
    indexed_a = sorted(
        list(enumerate(events_a)),
        key=lambda x: (x[1].start_time, x[1].end_time, str(x[1].event_id))
    )
    indexed_b = sorted(
        list(enumerate(events_b)),
        key=lambda x: (x[1].start_time, x[1].end_time, str(x[1].event_id))
    )

    K = max(1000.0, float(max(n_a, n_b) * 10.0))
    benefit_matrix = np.zeros((n_a, n_b), dtype=np.float64)
    info_matrix: Dict[Tuple[int, int], Tuple[float, float]] = {}

    for r, (orig_i, ea) in enumerate(indexed_a):
        for c, (orig_j, eb) in enumerate(indexed_b):
            valid, iou, cdist = is_valid_candidate(
                ea.start_time, ea.end_time, eb.start_time, eb.end_time,
                iou_threshold=iou_threshold,
                center_dist_threshold_sec=center_dist_threshold_sec
            )
            if valid:
                # Strict lexicographic formulation:
                # 1. Primary: Cardinality bonus K (strictly dominates any m vs m-1 configuration).
                # 2. Secondary: IoU weight in [0.50, 1.00].
                # 3. Tertiary: Center proximity in [0.0, 1.0] scaled by 1e-4.
                # 4. Quaternary: Canonical index tie-break in [0.0, 1.0) scaled by -1e-7 for deterministic resolution.
                cdist_norm = min(1.0, cdist / max(1e-6, center_dist_threshold_sec))
                center_proximity = 1.0 - cdist_norm
                idx_norm = float(r * n_b + c) / float(max(1, n_a * n_b))
                tie_break = (1e-4 * center_proximity) - (1e-7 * idx_norm)
                benefit_matrix[r, c] = K + iou + tie_break
                info_matrix[(r, c)] = (iou, cdist)

    # If no valid candidate edges exist, return empty match list immediately
    if not info_matrix:
        return []

    # Cost matrix for minimization
    max_b = np.max(benefit_matrix)
    cost_matrix = max_b - benefit_matrix

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    for r, c in zip(row_ind, col_ind):
        if benefit_matrix[r, c] >= K:  # Valid matched edge
            iou, cdist = info_matrix[(r, c)]
            orig_i, ea = indexed_a[r]
            orig_j, eb = indexed_b[c]
            matches.append({
                "index_a": int(orig_i),
                "index_b": int(orig_j),
                "event_a": ea,
                "event_b": eb,
                "iou": iou,
                "center_distance": cdist,
                "onset_diff": abs(ea.start_time - eb.start_time),
                "offset_diff": abs(ea.end_time - eb.end_time),
            })

    # Sort final matches deterministically by stable event identity
    matches.sort(key=lambda m: (m["event_a"].start_time, m["event_a"].end_time, str(m["event_a"].event_id)))
    return matches


def compute_layer1_segmentation_metrics(
    matches: List[Dict[str, Any]],
    n_annotator_a: int,
    n_annotator_b: int
) -> Dict[str, float]:
    """Computes Layer 1 segmentation precision, recall, F1, and mean boundary deviations."""
    n_matched = len(matches)
    precision = float(n_matched / n_annotator_a) if n_annotator_a > 0 else 1.0
    recall = float(n_matched / n_annotator_b) if n_annotator_b > 0 else 1.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0.0 else 0.0

    if n_matched > 0:
        onset_mae = float(np.mean([m["onset_diff"] for m in matches]))
        offset_mae = float(np.mean([m["offset_diff"] for m in matches]))
        mean_iou = float(np.mean([m["iou"] for m in matches]))
    else:
        onset_mae = 0.0
        offset_mae = 0.0
        mean_iou = 0.0

    return {
        "n_matched": n_matched,
        "n_annotator_a": n_annotator_a,
        "n_annotator_b": n_annotator_b,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "onset_mae_sec": onset_mae,
        "offset_mae_sec": offset_mae,
        "mean_iou": mean_iou,
    }


def compute_layer2_semantic_agreement(
    matches: List[Dict[str, Any]],
    is_command_fn=None
) -> Dict[str, Any]:
    """
    Computes Layer 2 2x2 contingency matrix and agreement statistics across matched pairs.
    Default command predicate: intent == 'INTENDED_COMMAND_TO_SYSTEM' and addressee == 'KWS_DEVICE'.
    """
    if is_command_fn is None:
        def is_command_fn(ev: AnnotationEvent) -> bool:
            return (
                ev.intent == "INTENDED_COMMAND_TO_SYSTEM"
                and ev.addressee == "KWS_DEVICE"
            )

    A = 0  # Concordant positive
    B = 0  # A pos, B neg
    C = 0  # A neg, B pos
    D = 0  # Concordant neg

    for m in matches:
        ea: AnnotationEvent = m["event_a"]
        eb: AnnotationEvent = m["event_b"]
        a_is_cmd = is_command_fn(ea)
        b_is_cmd = is_command_fn(eb)

        if a_is_cmd and b_is_cmd:
            A += 1
        elif a_is_cmd and not b_is_cmd:
            B += 1
        elif not a_is_cmd and b_is_cmd:
            C += 1
        else:
            D += 1

    N = A + B + C + D
    if N == 0:
        return {
            "A": 0, "B": 0, "C": 0, "D": 0, "N": 0,
            "Po": 1.0, "Pe": 1.0, "kappa": 1.0,
            "Ppos": 1.0, "Pneg": 1.0
        }

    Po = (A + D) / N
    Pe = ((A + B) / N) * ((A + C) / N) + ((C + D) / N) * ((B + D) / N)

    if (1.0 - Pe) > 1e-9:
        kappa = (Po - Pe) / (1.0 - Pe)
    else:
        kappa = 1.0 if Po == 1.0 else 0.0

    Ppos = (2 * A) / (2 * A + B + C) if (2 * A + B + C) > 0 else 1.0
    Pneg = (2 * D) / (2 * D + B + C) if (2 * D + B + C) > 0 else 1.0

    return {
        "A": A,
        "B": B,
        "C": C,
        "D": D,
        "N": N,
        "Po": float(Po),
        "Pe": float(Pe),
        "kappa": float(kappa),
        "Ppos": float(Ppos),
        "Pneg": float(Pneg),
    }
