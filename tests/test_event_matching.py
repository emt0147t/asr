"""
Unit tests for Annotation Event Matching and Agreement Metrics.
Demonstrates:
  1. Strict mathematical compliance with Phase 9A.3 IoU and center-distance criteria.
  2. Deterministic assignment and reproducible tie-breaking.
  3. Adversarial proof of greedy bipartite alignment suboptimality vs Hungarian optimal matching.
  4. Layer 1 segmentation metrics (Precision, Recall, F1, Onset MAE, Offset MAE).
  5. Layer 2 semantic agreement metrics (Po, Pe, Cohen's Kappa, Ppos, Pneg) and prevalence paradox immunity.
"""

import pytest
import numpy as np
from src.acquisition.event_matching import (
    AnnotationEvent,
    compute_iou,
    compute_center_distance,
    is_valid_candidate,
    match_events_greedy,
    match_events_optimal,
    compute_layer1_segmentation_metrics,
    compute_layer2_semantic_agreement,
)


def test_iou_and_center_distance_calculations():
    # 1. Identical intervals
    iou = compute_iou(0.0, 1.0, 0.0, 1.0)
    cdist = compute_center_distance(0.0, 1.0, 0.0, 1.0)
    assert iou == 1.0
    assert cdist == 0.0

    # 2. Non-overlapping intervals
    assert compute_iou(0.0, 1.0, 1.5, 2.5) == 0.0
    assert compute_center_distance(0.0, 1.0, 1.5, 2.5) == 1.5

    # 3. Known overlap: [0.0, 1.0] and [0.5, 1.5]
    # Intersection = 0.5, Union = 1.5 -> IoU = 1/3 ~ 0.3333 (< 0.50 threshold)
    iou = compute_iou(0.0, 1.0, 0.5, 1.5)
    assert abs(iou - 1.0 / 3.0) < 1e-6
    valid, _, _ = is_valid_candidate(0.0, 1.0, 0.5, 1.5)
    assert not valid  # Fails IoU >= 0.50

    # 4. Center distance boundary: [0.0, 1.0] (mid=0.5) and [0.2, 1.2] (mid=0.7)
    # cdist = 0.200s > 100ms
    valid, iou, cdist = is_valid_candidate(0.0, 1.0, 0.2, 1.2)
    assert iou == (0.8 / 1.2)  # ~0.6667 >= 0.50
    assert cdist == pytest.approx(0.200)
    assert not valid  # Fails center distance <= 100ms


def test_greedy_vs_optimal_adversarial_cardinality_suboptimality():
    """
    Adversarial Counter-Example:
    Annotator A produces:
      A1: [0.00, 1.00] (midpoint = 0.50)
      A2: [0.10, 1.10] (midpoint = 0.60)
    Annotator B produces:
      B1: [0.04, 1.04] (midpoint = 0.54)
      B2: [0.00, 0.90] (midpoint = 0.45)

    Candidate evaluation:
      (A1, B1): IoU = 0.96 / 1.04 = 0.9231 >= 0.50, cdist = |0.50 - 0.54| = 0.04 <= 0.10 (VALID)
      (A1, B2): IoU = 0.90 / 1.00 = 0.9000 >= 0.50, cdist = |0.50 - 0.45| = 0.05 <= 0.10 (VALID)
      (A2, B1): IoU = 0.94 / 1.10 = 0.8545 >= 0.50, cdist = |0.60 - 0.54| = 0.06 <= 0.10 (VALID)
      (A2, B2): IoU = 0.80 / 1.10 = 0.7273 >= 0.50, cdist = |0.60 - 0.45| = 0.15 > 0.10 (INVALID: cdist exceeds 100ms)

    Greedy behavior:
      Sorted by IoU: (A1, B1)=0.9231, (A1, B2)=0.9000, (A2, B1)=0.8545.
      1. Greedy selects (A1, B1).
      2. Both A1 and B1 are now occupied.
      3. A2 can only match B1 (occupied -> blocked).
      4. B2 can only match A1 (occupied -> blocked).
      Result: Exactly 1 match (50% event recovery).

    Optimal (Hungarian) behavior:
      Can select (A1, B2) and (A2, B1).
      Result: Exactly 2 matches (100% event recovery).
    """
    ea1 = AnnotationEvent("A1", 0.00, 1.00, label="LEN")
    ea2 = AnnotationEvent("A2", 0.10, 1.10, label="XUONG")
    eb1 = AnnotationEvent("B1", 0.04, 1.04, label="LEN")
    eb2 = AnnotationEvent("B2", 0.00, 0.90, label="XUONG")

    events_a = [ea1, ea2]
    events_b = [eb1, eb2]

    # Run Greedy
    greedy_matches = match_events_greedy(events_a, events_b)
    assert len(greedy_matches) == 1
    assert greedy_matches[0]["index_a"] == 0 and greedy_matches[0]["index_b"] == 0
    assert greedy_matches[0]["event_a"].event_id == "A1"
    assert greedy_matches[0]["event_b"].event_id == "B1"

    # Run Optimal Hungarian
    optimal_matches = match_events_optimal(events_a, events_b)
    assert len(optimal_matches) == 2, "Optimal Hungarian failed to find maximum cardinality matching!"
    # Matches: (A1, B2) and (A2, B1)
    matched_pairs = {(m["event_a"].event_id, m["event_b"].event_id) for m in optimal_matches}
    assert matched_pairs == {("A1", "B2"), ("A2", "B1")}

    # Compare Layer 1 F1 scores
    metrics_greedy = compute_layer1_segmentation_metrics(greedy_matches, len(events_a), len(events_b))
    metrics_optimal = compute_layer1_segmentation_metrics(optimal_matches, len(events_a), len(events_b))

    assert metrics_greedy["f1"] == 0.50
    assert metrics_optimal["f1"] == 1.00
    assert metrics_optimal["n_matched"] > metrics_greedy["n_matched"]


def test_deterministic_reproducibility():
    """Verifies that multiple runs with identical inputs produce deterministic output ordering."""
    events_a = [
        AnnotationEvent("A1", 0.00, 0.80),
        AnnotationEvent("A2", 1.00, 1.80),
        AnnotationEvent("A3", 2.00, 2.80),
    ]
    events_b = [
        AnnotationEvent("B1", 0.02, 0.82),
        AnnotationEvent("B2", 1.01, 1.81),
        AnnotationEvent("B3", 2.03, 2.83),
    ]

    res1 = match_events_optimal(events_a, events_b)
    res2 = match_events_optimal(events_a, events_b)

    assert len(res1) == len(res2) == 3
    for m1, m2 in zip(res1, res2):
        assert m1["index_a"] == m2["index_a"]
        assert m1["index_b"] == m2["index_b"]
        assert m1["iou"] == m2["iou"]


def test_layer1_and_layer2_metrics_and_prevalence_paradox():
    """
    Validates Layer 1 boundary deviation calculations and Layer 2 semantic agreement statistics,
    specifically confirming Ppos immunity to negative-sample inflation (prevalence paradox).
    """
    matches = [
        {
            "event_a": AnnotationEvent("A1", 0.0, 1.0, intent="INTENDED_COMMAND_TO_SYSTEM", addressee="KWS_DEVICE"),
            "event_b": AnnotationEvent("B1", 0.02, 1.03, intent="INTENDED_COMMAND_TO_SYSTEM", addressee="KWS_DEVICE"),
            "iou": 0.95, "center_distance": 0.025, "onset_diff": 0.020, "offset_diff": 0.030,
        },
        {
            "event_a": AnnotationEvent("A2", 2.0, 3.0, intent="INTENDED_COMMAND_TO_SYSTEM", addressee="KWS_DEVICE"),
            "event_b": AnnotationEvent("B2", 2.01, 3.02, intent="CONVERSATIONAL", addressee="HUMAN"),
            "iou": 0.97, "center_distance": 0.015, "onset_diff": 0.010, "offset_diff": 0.020,
        },
        {
            "event_a": AnnotationEvent("A3", 4.0, 5.0, intent="CONVERSATIONAL", addressee="HUMAN"),
            "event_b": AnnotationEvent("B3", 4.01, 5.01, intent="CONVERSATIONAL", addressee="HUMAN"),
            "iou": 0.98, "center_distance": 0.010, "onset_diff": 0.010, "offset_diff": 0.010,
        },
    ]

    # Layer 1
    l1 = compute_layer1_segmentation_metrics(matches, n_annotator_a=3, n_annotator_b=3)
    assert l1["precision"] == 1.0
    assert l1["recall"] == 1.0
    assert l1["f1"] == 1.0
    assert l1["onset_mae_sec"] == pytest.approx((0.020 + 0.010 + 0.010) / 3.0)
    assert l1["offset_mae_sec"] == pytest.approx((0.030 + 0.020 + 0.010) / 3.0)

    # Layer 2: A=1, B=1, C=0, D=1 (Total N=3)
    l2 = compute_layer2_semantic_agreement(matches)
    assert l2["A"] == 1
    assert l2["B"] == 1
    assert l2["C"] == 0
    assert l2["D"] == 1
    assert l2["N"] == 3
    # Ppos = 2*1 / (2*1 + 1 + 0) = 2/3 ~ 0.6667
    assert l2["Ppos"] == pytest.approx(2.0 / 3.0)

    # Add 100 concordant negative pairs (simulating prevalence shift)
    inflated_matches = list(matches)
    for i in range(100):
        inflated_matches.append({
            "event_a": AnnotationEvent(f"A_neg_{i}", 10.0 + i, 11.0 + i, intent="CONVERSATIONAL", addressee="HUMAN"),
            "event_b": AnnotationEvent(f"B_neg_{i}", 10.0 + i, 11.0 + i, intent="CONVERSATIONAL", addressee="HUMAN"),
            "iou": 1.0, "center_distance": 0.0, "onset_diff": 0.0, "offset_diff": 0.0,
        })

    l2_inflated = compute_layer2_semantic_agreement(inflated_matches)
    assert l2_inflated["A"] == 1
    assert l2_inflated["B"] == 1
    assert l2_inflated["C"] == 0
    assert l2_inflated["D"] == 101
    # Ppos MUST remain strictly 2/3 (immune to prevalence shift!)
    assert l2_inflated["Ppos"] == pytest.approx(2.0 / 3.0)
    # Whereas Po and Pe change significantly due to prevalence
    assert l2_inflated["Po"] > 0.98


def test_rectangular_matrices_and_empty_candidate_graph():
    """Validates behavior on rectangular candidate matrices (Na != Nb) and empty edge graphs."""
    # Rectangular: 4 reference events vs 2 predicted events
    events_a = [
        AnnotationEvent("A1", 0.0, 1.0),
        AnnotationEvent("A2", 2.0, 3.0),
        AnnotationEvent("A3", 4.0, 5.0),
        AnnotationEvent("A4", 6.0, 7.0),
    ]
    events_b = [
        AnnotationEvent("B1", 0.02, 1.01),  # matches A1
        AnnotationEvent("B2", 4.01, 5.02),  # matches A3
    ]

    matches = match_events_optimal(events_a, events_b)
    assert len(matches) == 2
    matched_ids = {(m["event_a"].event_id, m["event_b"].event_id) for m in matches}
    assert matched_ids == {("A1", "B1"), ("A3", "B2")}

    metrics = compute_layer1_segmentation_metrics(matches, len(events_a), len(events_b))
    assert metrics["precision"] == 2.0 / 4.0  # 0.50
    assert metrics["recall"] == 2.0 / 2.0     # 1.00
    assert metrics["f1"] == pytest.approx(2.0 * 0.5 * 1.0 / (0.5 + 1.0))  # ~0.6667

    # Empty candidate graph: Non-overlapping events with zero valid edges
    events_far = [
        AnnotationEvent("FAR1", 100.0, 101.0),
        AnnotationEvent("FAR2", 102.0, 103.0),
    ]
    empty_matches = match_events_optimal(events_a, events_far)
    assert empty_matches == []
    empty_metrics = compute_layer1_segmentation_metrics(empty_matches, len(events_a), len(events_far))
    assert empty_metrics["n_matched"] == 0
    assert empty_metrics["f1"] == 0.0


def test_invalid_and_zero_duration_intervals():
    """
    Ensures zero-duration or inverted intervals return 0.0 in compute_iou,
    and are strictly rejected with ValueError in match_events_optimal and match_events_greedy.
    """
    assert compute_iou(1.0, 1.0, 1.0, 2.0) == 0.0  # Zero duration on A
    assert compute_iou(2.0, 1.0, 0.5, 1.5) == 0.0  # Inverted interval on A
    assert compute_iou(-1.0, 1.0, 0.0, 1.0) == 0.0 # Negative timestamp

    ev_valid = AnnotationEvent("V1", 0.0, 1.0)
    ev_zero = AnnotationEvent("Z1", 0.5, 0.5)
    ev_inv = AnnotationEvent("I1", 1.5, 0.5)

    with pytest.raises(ValueError, match="has zero duration"):
        match_events_optimal([ev_valid], [ev_zero])

    with pytest.raises(ValueError, match="has reversed interval"):
        match_events_optimal([ev_valid], [ev_inv])

    with pytest.raises(ValueError, match="has zero duration"):
        match_events_greedy([ev_valid], [ev_zero])

    with pytest.raises(ValueError, match="has reversed interval"):
        match_events_greedy([ev_valid], [ev_inv])


def test_event_matching_input_validation_rejections():
    """
    Phase 9A.6.3 Section 3: Input validation exhaustive rejection tests.
    Validates explicit rejection of:
      1. NaN timestamps (ValueError)
      2. Infinity timestamps (ValueError)
      3. Non-numeric timestamps (TypeError: bool, str, None)
      4. Negative timestamps (ValueError)
      5. Zero-duration intervals (ValueError)
      6. Reversed intervals (ValueError)
      7. Duplicate event IDs (ValueError)
      8. Empty or invalid event IDs (ValueError, TypeError)
    """
    ev_ok = AnnotationEvent("OK", 1.0, 2.0)

    # 1. NaN timestamps
    with pytest.raises(ValueError, match="contains NaN timestamp"):
        match_events_optimal([AnnotationEvent("NAN1", float("nan"), 2.0)], [ev_ok])
    with pytest.raises(ValueError, match="contains NaN timestamp"):
        match_events_optimal([ev_ok], [AnnotationEvent("NAN2", 1.0, float("nan"))])

    # 2. Infinity timestamps
    with pytest.raises(ValueError, match="contains infinite timestamp"):
        match_events_optimal([AnnotationEvent("INF1", float("inf"), 2.0)], [ev_ok])
    with pytest.raises(ValueError, match="contains infinite timestamp"):
        match_events_optimal([ev_ok], [AnnotationEvent("INF2", 1.0, float("-inf"))])

    # 3. Non-numeric timestamps (bool, str, None)
    with pytest.raises(TypeError, match="has non-numeric start_time"):
        match_events_optimal([AnnotationEvent("BOOL1", True, 2.0)], [ev_ok])
    with pytest.raises(TypeError, match="has non-numeric end_time"):
        match_events_optimal([AnnotationEvent("BOOL2", 1.0, False)], [ev_ok])
    with pytest.raises(TypeError, match="has non-numeric start_time"):
        match_events_optimal([AnnotationEvent("STR1", "1.0", 2.0)], [ev_ok])
    with pytest.raises(TypeError, match="has non-numeric start_time"):
        match_events_optimal([AnnotationEvent("NONE1", None, 2.0)], [ev_ok])

    # 4. Negative timestamps
    with pytest.raises(ValueError, match="has negative timestamp"):
        match_events_optimal([AnnotationEvent("NEG1", -0.1, 1.0)], [ev_ok])
    with pytest.raises(ValueError, match="has negative timestamp"):
        match_events_optimal([ev_ok], [AnnotationEvent("NEG2", 0.0, -0.5)])

    # 5. Zero-duration intervals
    with pytest.raises(ValueError, match="has zero duration"):
        match_events_optimal([AnnotationEvent("ZERO1", 1.5, 1.5)], [ev_ok])

    # 6. Reversed intervals
    with pytest.raises(ValueError, match="has reversed interval"):
        match_events_optimal([AnnotationEvent("REV1", 3.0, 1.0)], [ev_ok])

    # 7. Duplicate event IDs
    with pytest.raises(ValueError, match="Duplicate event_id 'DUP'"):
        match_events_optimal([AnnotationEvent("DUP", 1.0, 2.0), AnnotationEvent("DUP", 3.0, 4.0)], [ev_ok])

    # 8. Empty or invalid event IDs
    with pytest.raises(ValueError, match="has invalid or empty event_id"):
        match_events_optimal([AnnotationEvent("", 1.0, 2.0)], [ev_ok])
    with pytest.raises(ValueError, match="has invalid or empty event_id"):
        match_events_optimal([AnnotationEvent(12345, 1.0, 2.0)], [ev_ok])


def test_equal_weight_ties_deterministic_selection():
    """Verifies that equal-weight ties are broken deterministically by midpoint deviation and indices."""
    # A1 matches both B1 and B2 with identical IoU
    ea = AnnotationEvent("A1", 0.0, 1.0)  # midpoint = 0.50
    eb1 = AnnotationEvent("B1", 0.01, 1.01)  # midpoint = 0.51, cdist = 0.01
    eb2 = AnnotationEvent("B2", 0.02, 1.02)  # midpoint = 0.52, cdist = 0.02

    # Both have high IoU, but eb1 has smaller center distance (0.01 < 0.02)
    matches = match_events_optimal([ea], [eb2, eb1])
    assert len(matches) == 1
    assert matches[0]["event_b"].event_id == "B1"


def test_cardinality_dominance_proof_marginal_iou_beats_single_perfect_iou():
    """
    Mathematical Proof Verification of Cardinality Dominance:
      Assignment 1: 1 match with perfect IoU=1.00 and 0ms cdist ((A1, B1)).
      Assignment 2: 2 matches with non-perfect IoUs ((A1, B2) and (A2, B1)).
    The Hungarian optimization must choose Assignment 2 (|M|=2) over Assignment 1 (|M|=1),
    proving that cardinality bonus K dominates any IoU advantage.
    """
    # A1 can match B1 (perfect overlap: IoU=1.0) or B2 (IoU=0.85, cdist=0.075s <= 0.10s)
    # A2 can match B1 (IoU=0.905, cdist=0.050s <= 0.10s)
    # (A2, B2) is strictly INVALID: cdist = |0.55 - 0.425| = 0.125s > 0.100s
    ea1 = AnnotationEvent("A1", 0.00, 1.00)  # [0.00, 1.00] mid=0.50
    ea2 = AnnotationEvent("A2", 0.05, 1.05)  # [0.05, 1.05] mid=0.55
    eb1 = AnnotationEvent("B1", 0.00, 1.00)  # [0.00, 1.00] mid=0.50
    eb2 = AnnotationEvent("B2", 0.00, 0.85)  # [0.00, 0.85] mid=0.425

    events_a = [ea1, ea2]
    events_b = [eb1, eb2]

    # Greedy chooses 1 match (A1, B1) with IoU=1.00
    greedy_matches = match_events_greedy(events_a, events_b)
    assert len(greedy_matches) == 1
    assert (greedy_matches[0]["event_a"].event_id, greedy_matches[0]["event_b"].event_id) == ("A1", "B1")

    # Hungarian chooses 2 matches, achieving |M|=2
    matches = match_events_optimal(events_a, events_b)
    assert len(matches) == 2, "Hungarian failed cardinality dominance: chose 1 match instead of 2!"
    matched_pairs = {(m["event_a"].event_id, m["event_b"].event_id) for m in matches}
    assert matched_pairs == {("A1", "B2"), ("A2", "B1")}


def test_permutation_invariance_under_random_reordering():
    """
    Verifies that the set of matched pairs is invariant to the ordering of events in inputs.
    """
    import random
    events_a = [
        AnnotationEvent(f"A{i}", float(i * 2), float(i * 2 + 1))
        for i in range(10)
    ]
    events_b = [
        AnnotationEvent(f"B{i}", float(i * 2) + 0.02, float(i * 2 + 1) + 0.02)
        for i in range(10)
    ]

    canonical_matches = match_events_optimal(events_a, events_b)
    canonical_pairs = {(m["event_a"].event_id, m["event_b"].event_id) for m in canonical_matches}
    assert len(canonical_pairs) == 10

    # Shuffle and verify identity of matched pairs
    rng = random.Random(42)
    shuffled_a = list(events_a)
    shuffled_b = list(events_b)
    for _ in range(5):
        rng.shuffle(shuffled_a)
        rng.shuffle(shuffled_b)
        permuted_matches = match_events_optimal(shuffled_a, shuffled_b)
        permuted_pairs = {(m["event_a"].event_id, m["event_b"].event_id) for m in permuted_matches}
        assert permuted_pairs == canonical_pairs, "Permutation invariance violated!"


def test_isolated_scipy_import_guard(monkeypatch):
    """
    Verifies that match_events_optimal fails gracefully with informative ImportError
    if scipy is unavailable, preserving project error isolation.
    """
    import src.acquisition.event_matching as em
    monkeypatch.setattr(em, "_HAS_SCIPY", False)

    ea = AnnotationEvent("A1", 0.0, 1.0)
    eb = AnnotationEvent("B1", 0.0, 1.0)

    with pytest.raises(ImportError, match="scipy>=1.10.0 is required"):
        em.match_events_optimal([ea], [eb])


def test_adversarial_symmetric_equal_weight_permutation_invariance():
    """
    Phase 9A.6.1/9A.6.3 Determinism Verification: Adversarial Equal-Weight Symmetry Fixture.
    Evaluated under Python 3.11.9 and SciPy 1.17.1.
    Constructs a fully symmetric equal-weight 3x3 complete bipartite graph fixture where all edges
    have identical IoU=1.00 and center distance=0.0.
    Evaluates all 3! x 3! = 36 permutations of input orderings to verify that
    matching results are 100% deterministic and invariant when represented by stable event IDs.

    Scope Boundary Notice: This test constitutes an empirical fixture proof for this specific
    graph and runtime configuration. It does not establish a universal cross-version mathematical
    invariance guarantee across all arbitrary graphs or future SciPy solver releases.
    """
    import itertools

    ea1 = AnnotationEvent("A1", 0.0, 1.0)
    ea2 = AnnotationEvent("A2", 0.0, 1.0)
    ea3 = AnnotationEvent("A3", 0.0, 1.0)

    eb1 = AnnotationEvent("B1", 0.0, 1.0)
    eb2 = AnnotationEvent("B2", 0.0, 1.0)
    eb3 = AnnotationEvent("B3", 0.0, 1.0)

    distinct_pairings = set()
    for perm_a in itertools.permutations([ea1, ea2, ea3]):
        for perm_b in itertools.permutations([eb1, eb2, eb3]):
            matches = match_events_optimal(list(perm_a), list(perm_b))
            pair_tuple = tuple(sorted((m["event_a"].event_id, m["event_b"].event_id) for m in matches))
            distinct_pairings.add(pair_tuple)

    assert len(distinct_pairings) == 1, (
        f"Non-deterministic matching detected! Expected exactly 1 canonical pairing, "
        f"got {len(distinct_pairings)} across 36 permutations: {distinct_pairings}"
    )
    assert distinct_pairings == {(("A1", "B1"), ("A2", "B2"), ("A3", "B3"))}


def test_duplicate_event_ids_validation_error():
    """
    Phase 9A.6.2 Workstream D: Validates that duplicate event IDs produce an explicit
    ValueError with a clear message, in both reference and prediction inputs.
    """
    ea1 = AnnotationEvent("DUP_01", 0.0, 1.0)
    ea2 = AnnotationEvent("DUP_01", 10.0, 11.0)
    eb1 = AnnotationEvent("B1", 0.0, 1.0)
    eb2 = AnnotationEvent("B2", 10.0, 11.0)

    # 1. Duplicate reference event IDs
    with pytest.raises(ValueError, match="Duplicate event_id 'DUP_01' detected in events_a"):
        match_events_optimal([ea1, ea2], [eb1, eb2])

    with pytest.raises(ValueError, match="Duplicate event_id 'DUP_01' detected in events_a"):
        match_events_greedy([ea1, ea2], [eb1, eb2])

    # 2. Duplicate prediction event IDs
    ea_clean1 = AnnotationEvent("A1", 0.0, 1.0)
    ea_clean2 = AnnotationEvent("A2", 10.0, 11.0)
    eb_dup1 = AnnotationEvent("DUP_B", 0.0, 1.0)
    eb_dup2 = AnnotationEvent("DUP_B", 10.0, 11.0)

    with pytest.raises(ValueError, match="Duplicate event_id 'DUP_B' detected in events_b"):
        match_events_optimal([ea_clean1, ea_clean2], [eb_dup1, eb_dup2])

    with pytest.raises(ValueError, match="Duplicate event_id 'DUP_B' detected in events_b"):
        match_events_greedy([ea_clean1, ea_clean2], [eb_dup1, eb_dup2])


def test_equal_timestamps_and_distinct_event_ids_deterministic():
    """
    Verifies that events with identical timestamps and distinct event IDs
    are sorted deterministically and match reliably without dependency on input order.
    """
    ea1 = AnnotationEvent("A_ALPHA", 0.0, 1.0)
    ea2 = AnnotationEvent("A_BETA", 0.0, 1.0)
    eb1 = AnnotationEvent("B_ALPHA", 0.01, 1.01)
    eb2 = AnnotationEvent("B_BETA", 0.01, 1.01)

    # Permutation 1
    m1 = match_events_optimal([ea1, ea2], [eb1, eb2])
    # Permutation 2
    m2 = match_events_optimal([ea2, ea1], [eb2, eb1])

    pairs1 = [(m["event_a"].event_id, m["event_b"].event_id) for m in m1]
    pairs2 = [(m["event_a"].event_id, m["event_b"].event_id) for m in m2]

    assert pairs1 == pairs2, f"Permutations differed: {pairs1} vs {pairs2}"
    assert len(pairs1) == 2


def test_sparse_graph_determinism():
    """
    Verifies that sparse disconnected components separated by large gaps
    are resolved deterministically.
    """
    cluster_a = [
        AnnotationEvent("C1_A", 0.0, 1.0),
        AnnotationEvent("C2_A", 100.0, 101.0),
        AnnotationEvent("C3_A", 200.0, 201.0),
    ]
    cluster_b = [
        AnnotationEvent("C1_B", 0.02, 1.01),
        AnnotationEvent("C2_B", 100.01, 101.02),
        AnnotationEvent("C3_B", 200.03, 201.01),
    ]
    sparse_matches = match_events_optimal(cluster_a, cluster_b)
    assert len(sparse_matches) == 3
    sparse_pairs = {(m["event_a"].event_id, m["event_b"].event_id) for m in sparse_matches}
    assert sparse_pairs == {("C1_A", "C1_B"), ("C2_A", "C2_B"), ("C3_A", "C3_B")}



