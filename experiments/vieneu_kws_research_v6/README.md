# Vietnamese Keyword Spotting Research Campaign — v6 (Phases 60–64)

**Workspace:** `D:\trainidentificationaudio`  
**Experiment Directory:** `experiments/vieneu_kws_research_v6/`  
**Status Banner:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

## Overview

This experiment directory documents the complete artifact lineage, datasets, models, validation calibrations, locked test benchmarks, and dual operational decisions for the Vietnamese Keyword Spotting (`TRAI`, `PHAI`, `LEN`, `XUONG`, `REJECT`) Research Campaign v6.

## Structure
- `data/`: Dataset v6 manifests (`manifest.csv`, `splits/`), raw 48kHz and 16kHz normalized audio, and `benchmark_metadata.json`.
- `models/`: Architecture definitions, PyTorch training scripts (`train_ablation_campaign.py`), and model checkpoints (`checkpoints/`).
- `calibration/`: Validation-only threshold searches and locked parameter configurations (`selected_configs.json`).
- `predictions/`: Serialized test summaries (`locked_test_summary.json`), per-clip predictions (`per_clip_predictions.csv`), and reconciliation audits (`phase60_reproduced_metrics.json`).
- `reports/`: Complete suite of peer-reviewed markdown reports spanning Phases 60 to 64.

## Dual Operational Verdicts
- **Mode A (Bounded-Utterance):** `FEASIBLE_BOUNDED_CANDIDATE_ESTABLISHED` / `NO_FEASIBLE_VALIDATION_OPERATING_POINT` (per-seed criteria).
- **Mode B (Continuous Open-Mic):** `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION` (strictly preserved across all phases).
