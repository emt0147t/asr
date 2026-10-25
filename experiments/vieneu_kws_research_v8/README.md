# Vietnamese Keyword Spotting Research Campaign — v8 (Phases 70–74)

**Workspace:** `D:\trainidentificationaudio`  
**Experiment Directory:** `experiments/vieneu_kws_research_v8/`  
**Status Banner:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

## Overview

This experiment directory documents the complete artifact lineage, Candidate B metric audit, validation Pareto study, controlled hard-negative weight ablation, robust calibration, fresh locked test benchmark, and dual operational decisions for the Vietnamese Keyword Spotting (`TRAI`, `PHAI`, `LEN`, `XUONG`, `REJECT`) Research Campaign v8.

## Directory Structure
- `data/`: Fresh benchmark manifests (`manifest.csv`, `splits/`), fresh synthesized audio clips (`audio/`).
- `models/`: Training scripts (`train_ablation_v8.py`), checkpoints (`checkpoints/`), and training summaries (`training_summary.json`).
- `calibration/`: Validation Pareto frontiers (`threshold_frontier.json`) and frozen operating parameters (`selected_configs.json`).
- `predictions/`: Locked test evaluation summaries (`locked_test_summary.json`), per-clip predictions (`per_clip_predictions.csv`), and validation operating points (`validation_operating_points.csv`).
- `reports/`: Complete suite of peer-reviewed markdown reports spanning Phases 70 to 74.

## Dual Operational Verdicts
- **Mode A (Bounded-Utterance):** `FEASIBLE_BOUNDED_CANDIDATE_ESTABLISHED`.
- **Mode B (Continuous Open-Mic):** `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`.
