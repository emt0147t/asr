"""
Development Tuning Script for VAD Candidates on vad_dev_corpus (Phase 15).
Systematically tests parameter grids for Candidate B and Candidate C
strictly on DEV corpus to find the optimal pre-registered configuration.
"""

import sys
import io
import json
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "phase15_vad"))

from vad_segmenters import DualThresholdAdaptiveVAD, SpeechBandRatioVAD
from eval_stage_a_segmentation import evaluate_segmenter_on_corpus

manifest_dev = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "manifests" / "vad_dev_corpus_manifest.csv"

def run_tuning():
    print("=" * 80)
    print("  DEVELOPMENT PARAMETER TUNING ON vad_dev_corpus")
    print("=" * 80)

    # Grid for Candidate B (Dual-Threshold Adaptive)
    b_grid = [
        {"hangover_clean_ms": 100, "hangover_noisy_ms": 160, "onset_snr_offset_db": 10.0, "offset_snr_offset_db": 6.0},
        {"hangover_clean_ms": 80,  "hangover_noisy_ms": 140, "onset_snr_offset_db": 10.0, "offset_snr_offset_db": 6.0},
        {"hangover_clean_ms": 70,  "hangover_noisy_ms": 120, "onset_snr_offset_db": 10.0, "offset_snr_offset_db": 7.0},
        {"hangover_clean_ms": 60,  "hangover_noisy_ms": 100, "onset_snr_offset_db": 10.0, "offset_snr_offset_db": 8.0},
        {"hangover_clean_ms": 70,  "hangover_noisy_ms": 120, "onset_snr_offset_db": 8.0,  "offset_snr_offset_db": 6.0},
    ]

    print("\n--- Testing Candidate B Grid ---")
    b_results = []
    for p in b_grid:
        vad = DualThresholdAdaptiveVAD(**p)
        res = evaluate_segmenter_on_corpus(vad, manifest_dev)
        b_results.append((p, res))
        print(f"B cfg: {p} -> F1={res['f1_score']:.4f}, Merges={res['total_merges']}, Cuts={res['total_premature_cuts']}, Matched={res['total_matched']}/216")

    # Grid for Candidate C (Speech-Band Ratio)
    c_grid = [
        {"hangover_ms": 140, "min_speech_band_ratio": 0.32, "energy_threshold_db": -46.0},
        {"hangover_ms": 100, "min_speech_band_ratio": 0.30, "energy_threshold_db": -46.0},
        {"hangover_ms": 80,  "min_speech_band_ratio": 0.30, "energy_threshold_db": -48.0},
        {"hangover_ms": 80,  "min_speech_band_ratio": 0.28, "energy_threshold_db": -48.0},
        {"hangover_ms": 70,  "min_speech_band_ratio": 0.28, "energy_threshold_db": -48.0},
    ]

    print("\n--- Testing Candidate C Grid ---")
    c_results = []
    for p in c_grid:
        vad = SpeechBandRatioVAD(**p)
        res = evaluate_segmenter_on_corpus(vad, manifest_dev)
        c_results.append((p, res))
        print(f"C cfg: {p} -> F1={res['f1_score']:.4f}, Merges={res['total_merges']}, Cuts={res['total_premature_cuts']}, Matched={res['total_matched']}/216")

if __name__ == "__main__":
    run_tuning()
