# Vietnamese Keyword Spotting Research Campaign — v7 (Phases 65–69)

**Workspace:** `D:\trainidentificationaudio`  
**Experiment Directory:** `experiments/vieneu_kws_research_v7/`  
**Status Banner:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

## Overview

This experiment directory documents the complete artifact lineage, failure diagnosis, controlled model training, validation calibrations, locked test benchmarks, and dual operational decisions for the Vietnamese Keyword Spotting (`TRAI`, `PHAI`, `LEN`, `XUONG`, `REJECT`) Research Campaign v7.

## Structure
- `data/`: Diagnostic datasets, error taxonomies (`error_taxonomy.csv`).
- `models/`: Architecture definitions, PyTorch training scripts (`train_ablation.py`), and model checkpoints (`checkpoints/`).
- `calibration/`: Validation-only threshold searches and locked parameter configurations (`selected_configs.json`).
- `predictions/`: Serialized test summaries (`locked_test_summary.json`), per-clip predictions (`per_clip_predictions.csv`), and reproduced metrics (`reproduced_metrics.json`).
- `reports/`: Complete suite of peer-reviewed markdown reports spanning Phases 65 to 69.

## Dual Operational Verdicts
- **Mode A (Bounded-Utterance):** `NO_FEASIBLE_VALIDATION_OPERATING_POINT`.
- **Mode B (Continuous Open-Mic):** `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`.
