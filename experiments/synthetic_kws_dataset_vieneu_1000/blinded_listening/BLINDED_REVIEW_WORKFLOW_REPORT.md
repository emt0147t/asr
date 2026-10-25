# Vietnamese Synthetic KWS — Blinded Listening Review Workflow Report

**Package Directory:** `experiments/synthetic_kws_dataset_vieneu_1000/blinded_listening/`  
**Evaluation Scope:** 80 Stratified Clips (`blind_eval_01.wav` .. `blind_eval_80.wav`)  
**Phase ID:** Phase 25 — Complete the 80-Clip Blinded Listening Workflow  
**Date:** October 17, 2026  
**Human Verification Status:** **`PENDING_HUMAN_LISTENER_VERIFICATION`**  
**Official Project Governance Status:** **`NOT_VALIDATED_ON_NATURAL_SPEECH`**  

---

## 1. Executive Summary & Objective

Phase 25 establishes a local, zero-dependency, web-based evaluation interface and analysis pipeline for human perceptual quality assessment of the 80 stratified synthetic Vietnamese keyword clips.

This workflow guarantees:
1. **Convenient Local Review:** A single-page local web application allowing seamless audio playback, keyboard shortcuts, quick clip jumping, and responsive rating collection.
2. **Strict Blind Isolation:** Complete isolation of the answer key. The review interface, server process, HTTP endpoints, and client DOM contain **zero references or leaks** of target command labels, preset voice names, original sample IDs, split partitions, or RNG seeds.
3. **Robust Incremental Persistence:** Reviews are saved to disk on every submission (`blinded_listening_responses.csv`). Closing the browser or restarting the local server resumes exactly where the reviewer left off.
4. **Anti-Fabrication Guarantee:** Zero human ratings are simulated or inferred. Unreviewed clips remain strictly blank and unanswered. The analysis tool will not compute accuracy or perceptual scores until real ratings are entered.

---

## 2. Launch Instructions & Server Specifications

### 2.1 Starting the Local Review Server
The server is implemented in pure Python using standard library modules (`http.server`, `urllib.parse`, `csv`, `json`). It requires no third-party web frameworks.

To start the server:
```powershell
.\.venv\Scripts\python.exe experiments/synthetic_kws_dataset_vieneu_1000/blinded_listening/run_blinded_review.py
```

### 2.2 Server Options & Network Binding
| Argument | Default | Description |
| :--- | :---: | :--- |
| `--host` | `127.0.0.1` | Network interface to bind. Strictly defaults to loopback to prevent external network exposure. |
| `--port` | `8765` | TCP port for the local web server. |
| `--no-browser` | `False` | Disables automatic browser launch (useful for headless or manual browser navigation). |

Once started, open a web browser to:
**`http://127.0.0.1:8765/`**

---

## 3. Evaluation Schema & Input Field Definitions

For each of the 80 anonymized clips, the interface collects six structured evaluation fields:

| Field | Input Control | Options / Range | Operational Definition |
| :--- | :---: | :--- | :--- |
| **1. Recognized Command** | Radio Buttons | `TRAI`, `PHAI`, `LEN`, `XUONG`, `OTHER`, `UNCERTAIN` | **Required.** The Vietnamese command word perceived by the listener without prior bias. |
| **2. Perceived Tone** | Radio Buttons | `SAC`, `HOI`, `NGANG`, `OTHER`, `UNCERTAIN` | **Required.** The auditory lexical tone trajectory: `SAC` (high-rising), `HOI` (dipping/falling), `NGANG` (mid-level flat). |
| **3. Intelligibility** | 1–5 Scale | `1` (Unintelligible)<br>`2` (Ambiguous / Difficult)<br>`3` (Recognizable with effort)<br>`4` (Clear speech)<br>`5` (Crystal clear) | **Required.** Phonetic clarity and ease of understanding. Target threshold: $\ge 4.0$. |
| **4. Naturalness** | 1–5 Scale | `1` (Harsh robotic / buzzer)<br>`2` (Unnatural artifacts)<br>`3` (Synthetic but acceptable)<br>`4` (Natural speech)<br>`5` (Completely human-like) | **Required.** Perceptual human likeness and acoustic smoothness. Target threshold: $\ge 3.5$. |
| **5. Acoustic Defects** | Multi-Checkboxes | `Clipping`, `Distortion`, `Truncation`, `Buzzer`, `Clean` | Specific defect flags. Selecting any defect automatically unchecks `Clean`; selecting `Clean` unchecks all defects. |
| **6. Evaluator Notes** | Free Textarea | Optional string | Qualitative observations on regional accent, glottalization, or subtle acoustic anomalies. |

---

## 4. Architectural Privacy & Answer Key Isolation

To maintain scientific integrity during blinded human evaluation:
- **Key Quarantine:** The true labels and synthesis parameters are stored strictly in:
  - [`blinded_listening/blinded_listening_key.json`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_dataset_vieneu_1000/blinded_listening/blinded_listening_key.json)
  - [`blinded_listening/blinded_listening_key.csv`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_dataset_vieneu_1000/blinded_listening/blinded_listening_key.csv)
- **Zero Server Key Reference:** The review server [`run_blinded_review.py`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_dataset_vieneu_1000/blinded_listening/run_blinded_review.py) does not import, read, or parse the answer key.
- **Payload Minimization:** The `/api/manifest` endpoint returns only:
  ```json
  [
    {"blind_clip_id": "BLIND_EVAL_01", "audio_url": "/audio/blind_eval_01.wav"},
    {"blind_clip_id": "BLIND_EVAL_02", "audio_url": "/audio/blind_eval_02.wav"}
  ]
  ```
  Neither the web UI nor the client DOM can extract or deduce the true command label.

---

## 5. Incremental Persistence & Response Accounting

- **Incremental Auto-Save:** Every time the evaluator clicks **Save & Next** or navigates clips, a `POST /api/save_response` request writes the record to [`blinded_listening_responses.csv`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_dataset_vieneu_1000/blinded_listening/blinded_listening_responses.csv) via atomic file renaming.
- **State Recovery:** When the evaluator closes and reopens the interface, the client automatically queries `/api/responses` and highlights reviewed clips in green on the 80-clip quick jump grid, automatically loading the first unreviewed clip.
- **Distinction of Unreviewed Clips:** An unreviewed clip has an empty `perceived_command` field. The analysis tool will not treat missing entries as errors or defaults; they are counted strictly as `missing` reviews.

---

## 6. Automated Validation Test Results

Automated regression and workflow tests were executed in `tests/test_blinded_listening_workflow.py`:
```
tests/test_blinded_listening_workflow.py::test_all_80_clips_unique_and_complete PASSED
tests/test_blinded_listening_workflow.py::test_all_audio_files_exist_and_playable PASSED
tests/test_blinded_listening_workflow.py::test_strict_metadata_isolation_and_privacy PASSED
tests/test_blinded_listening_workflow.py::test_empty_review_sheet_reports_pending_without_fabrication PASSED
tests/test_blinded_listening_workflow.py::test_simulated_isolated_review_persistence_and_reloading PASSED
============================== 5 passed in 0.59s ==============================
```
Combined regression suite across all components: **74 passed in 7.20s**.

---

## 7. File Locations & Output Paths

| Workflow Artifact | Absolute File Path | Description |
| :--- | :--- | :--- |
| **Review Web Server** | [`blinded_listening/run_blinded_review.py`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_dataset_vieneu_1000/blinded_listening/run_blinded_review.py) | Standalone local HTTP server with interactive UI |
| **Analysis Pipeline** | [`blinded_listening/analyze_blinded_review.py`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_dataset_vieneu_1000/blinded_listening/analyze_blinded_review.py) | Analysis CLI comparing responses against key |
| **Active Response File** | [`blinded_listening/blinded_listening_responses.csv`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_dataset_vieneu_1000/blinded_listening/blinded_listening_responses.csv) | Destination where human submissions are persisted |
| **Master Blank Review Sheet** | [`blinded_listening/blinded_listening_review_sheet.csv`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_dataset_vieneu_1000/blinded_listening/blinded_listening_review_sheet.csv) | Schema definition and fallback review sheet |
| **Original Sheet Backup** | [`blinded_listening/blinded_listening_review_sheet_original.csv`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_dataset_vieneu_1000/blinded_listening/blinded_listening_review_sheet_original.csv) | Untouched original backup |
| **Quarantined Answer Key** | [`blinded_listening/blinded_listening_key.csv`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_dataset_vieneu_1000/blinded_listening/blinded_listening_key.csv)<br>[`blinded_listening/blinded_listening_key.json`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_dataset_vieneu_1000/blinded_listening/blinded_listening_key.json) | Ground truth answers (quarantined) |
| **Audio Corpus (80 WAVs)** | [`blinded_listening/audio/`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_dataset_vieneu_1000/blinded_listening/audio/) | 80 randomized, anonymized 16 kHz 1.0s WAVs |
| **Evaluation Report** | [`blinded_listening/HUMAN_LISTENING_EVALUATION_REPORT.md`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_dataset_vieneu_1000/blinded_listening/HUMAN_LISTENING_EVALUATION_REPORT.md) | Audit report generated by analysis tool |
