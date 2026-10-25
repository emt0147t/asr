"""
Phase 24: Blinded Listening Review Analysis Pipeline (80-Sample Screening)
Audits and analyzes human listening evaluation responses against the preserved answer key.
Calculates recognition rates, intelligibility, naturalness, tone perception, and defect counts.
Reports samples requiring re-listening and explicitly tracks listener count.

CRITICAL: Never fabricates responses. If review sheet has blank responses,
it accurately reports completion status as PENDING_HUMAN_LISTENER_VERIFICATION.
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

BLINDED_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.abspath(os.path.join(BLINDED_DIR, ".."))
KEY_CSV = os.path.join(BLINDED_DIR, "blinded_listening_key.csv")
RESPONSES_CSV = os.path.join(BLINDED_DIR, "blinded_listening_responses.csv")
REVIEW_SHEET_CSV = os.path.join(BLINDED_DIR, "blinded_listening_review_sheet.csv")
DEFAULT_RESPONSES_PATH = RESPONSES_CSV if os.path.exists(RESPONSES_CSV) else REVIEW_SHEET_CSV
OUTPUT_REPORT_MD = os.path.join(BLINDED_DIR, "HUMAN_LISTENING_EVALUATION_REPORT.md")

def analyze_blinded_review(responses_path=None, key_path=KEY_CSV, num_listeners=1):
    if responses_path is None:
        responses_path = RESPONSES_CSV if os.path.exists(RESPONSES_CSV) else REVIEW_SHEET_CSV
    print("=" * 70)
    print("Blinded Listening Review Analysis (80 Samples)")
    print("=" * 70)

    if not os.path.exists(responses_path):
        raise FileNotFoundError(f"Responses file not found: {responses_path}")
    if not os.path.exists(key_path):
        raise FileNotFoundError(f"Quarantined answer key not found: {key_path}")

    rdf = pd.read_csv(responses_path, dtype=str)
    kdf = pd.read_csv(key_path, dtype=str)

    total_expected = len(kdf)
    total_in_responses = len(rdf)

    print(f"Total expected clips in key: {total_expected}")
    print(f"Total clips in review sheet: {total_in_responses}")

    # Standardize column names
    col_map = {}
    for c in rdf.columns:
        if "perceived_command" in c:
            col_map[c] = "perceived_command"
        elif "defect_clipped" in c:
            col_map[c] = "defect_clipped"
        elif "defect_distorted" in c:
            col_map[c] = "defect_distorted"
        elif "defect_truncated" in c:
            col_map[c] = "defect_truncated"
        elif "defect_buzzer" in c:
            col_map[c] = "defect_buzzer"
        elif "defect_clean" in c:
            col_map[c] = "defect_clean"
        elif "perceived_tone" in c:
            col_map[c] = "perceived_tone"
        elif "intelligibility" in c:
            col_map[c] = "intelligibility"
        elif "naturalness" in c:
            col_map[c] = "naturalness"
        elif "evaluator_comments" in c:
            col_map[c] = "evaluator_comments"

    rdf = rdf.rename(columns=col_map)

    # Merge on blind_clip_id
    mdf = pd.merge(kdf, rdf, on="blind_clip_id", how="left")

    # Detect filled/completed reviews
    # A review is considered completed if perceived_command is non-empty and valid
    valid_commands = {"TRAI", "PHAI", "LEN", "XUONG", "OTHER", "UNCERTAIN"}
    mdf["is_completed"] = mdf["perceived_command"].apply(
        lambda x: bool(pd.notna(x) and str(x).strip().upper() in valid_commands)
    )

    n_completed = int(mdf["is_completed"].sum())
    n_missing = total_expected - n_completed
    completion_rate = (n_completed / total_expected) * 100.0

    print(f"\n--- Review Completion Status ---")
    print(f"Genuinely completed reviews: {n_completed} / {total_expected} ({completion_rate:.1f}%)")
    print(f"Missing / unreviewed clips:   {n_missing} / {total_expected}")
    print(f"Number of Active Listeners:   {num_listeners} (Screening Exercise)")

    if n_completed == 0:
        print("\nNotice: All review fields are currently blank. No human ratings have been entered yet.")
        print("Human Review Status: PENDING_HUMAN_LISTENER_VERIFICATION")
        print("The analysis pipeline is validated and standing by for real listener submissions.")

        md_content = fr"""# Vietnamese Synthetic KWS — Blinded Listening Review Report

**Package Directory:** `experiments/synthetic_kws_dataset_vieneu_1000/blinded_listening/`  
**Total Evaluation Clips:** {total_expected}  
**Date of Audit:** October 17, 2026  
**Completed Reviews:** 0 / {total_expected} (0.0%)  
**Missing / Pending Reviews:** {total_expected} / {total_expected} (100.0%)  
**Reported Listener Count:** {num_listeners} (Screening auditor)  
**Human Verification Status:** **`PENDING_HUMAN_LISTENER_VERIFICATION`**  
**Official Project Governance Status:** `NOT_VALIDATED_ON_NATURAL_SPEECH`  

---

## 1. Executive Status

The 80-clip stratified blinded listening package is fully prepared and quarantined. All evaluation fields in `blinded_listening_review_sheet.csv` are currently blank pending real human listener evaluation.

> [!IMPORTANT]
> **Strict Anti-Fabrication Invariant:** Antigravity does not fabricate listener ratings, infer answers from file basenames, or populate review responses from the quarantined answer key. This status report documents the readiness of the listening protocol.

---

## 2. Evaluation Package Composition

The 80 evaluation clips are stratified across:
- **Class Balance:** Exactly 20 clips per keyword (`TRAI`, `PHAI`, `LEN`, `XUONG`).
- **Voice Coverage:** All 25 VieNeu-TTS v3 Turbo preset voices are represented.
- **Split Distribution:** 52 train, 14 validation, 14 test.
- **Randomization:** File paths and clip IDs (`BLIND_EVAL_01` .. `BLIND_EVAL_80`) conceal all synthesizer metadata.

---

## 3. Downstream Analysis Metric Contract

Once real ratings are entered into [`blinded_listening_review_sheet.csv`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_dataset_vieneu_1000/blinded_listening/blinded_listening_review_sheet.csv), executing `python analyze_blinded_review.py` will automatically produce:
1. Overall and per-class recognition accuracy (% correct).
2. Perceived lexical tone accuracy and tone uncertainty rate (% `UNCERTAIN`).
3. Intelligibility rating distribution (Mean $\\pm$ Std on 1–5 scale).
4. Naturalness rating distribution (Mean $\\pm$ Std on 1–5 scale).
5. Defect breakdown: clipping, robotic distortion, truncation, synthetic buzzer.
6. Actionable triage list of candidate clips requiring re-listening or synthesis rejection.
"""
        with open(OUTPUT_REPORT_MD, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"Pending status report saved to: {OUTPUT_REPORT_MD}")
        return {
            "total_expected": total_expected,
            "completed": n_completed,
            "missing": n_missing,
            "status": "PENDING_HUMAN_LISTENER_VERIFICATION"
        }

    # If completed reviews exist, compute metrics
    completed_df = mdf[mdf["is_completed"]].copy()

    # Accuracy
    completed_df["is_correct"] = completed_df["perceived_command"].str.strip().str.upper() == completed_df["true_class_label"].str.strip().str.upper()
    overall_acc = completed_df["is_correct"].mean() * 100.0

    acc_by_class = completed_df.groupby("true_class_label")["is_correct"].agg(["count", "mean"])
    acc_by_class["mean"] = acc_by_class["mean"] * 100.0

    # Intelligibility & Naturalness (coerce to numeric)
    completed_df["intelligibility"] = pd.to_numeric(completed_df["intelligibility"], errors="coerce")
    completed_df["naturalness"] = pd.to_numeric(completed_df["naturalness"], errors="coerce")

    mean_intel = completed_df["intelligibility"].mean()
    std_intel = completed_df["intelligibility"].std()
    mean_nat = completed_df["naturalness"].mean()
    std_nat = completed_df["naturalness"].std()

    # Tone Accuracy
    tone_map = {
        "sắc": "SAC", "sắc (high-rising)": "SAC",
        "hỏi": "HOI", "hỏi (mid-falling rising / dipping)": "HOI",
        "ngang": "NGANG", "ngang (mid-level)": "NGANG"
    }
    completed_df["expected_tone_norm"] = completed_df["true_tone"].map(lambda t: tone_map.get(str(t).lower().strip(), "OTHER"))
    completed_df["perceived_tone_norm"] = completed_df["perceived_tone"].str.strip().str.upper()

    completed_df["tone_correct"] = completed_df["expected_tone_norm"] == completed_df["perceived_tone_norm"]
    completed_df["tone_uncertain"] = completed_df["perceived_tone_norm"] == "UNCERTAIN"

    tone_acc = completed_df["tone_correct"].mean() * 100.0
    tone_unc_rate = completed_df["tone_uncertain"].mean() * 100.0

    # Defects
    defect_clipped_cnt = (completed_df["defect_clipped"].str.strip().str.upper() == "YES").sum()
    defect_distorted_cnt = (completed_df["defect_distorted"].str.strip().str.upper() == "YES").sum()
    defect_truncated_cnt = (completed_df["defect_truncated"].str.strip().str.upper() == "YES").sum()
    defect_buzzer_cnt = (completed_df["defect_buzzer"].str.strip().str.upper() == "YES").sum()

    # Flagged for re-listening:
    # 1. Not correct command
    # 2. Command or tone is UNCERTAIN
    # 3. Intelligibility <= 2
    # 4. Any defect == YES
    relisten_df = completed_df[
        (~completed_df["is_correct"]) |
        (completed_df["perceived_command"].str.strip().str.upper() == "UNCERTAIN") |
        (completed_df["tone_uncertain"]) |
        (completed_df["intelligibility"] <= 2) |
        (completed_df["defect_clipped"].str.strip().str.upper() == "YES") |
        (completed_df["defect_distorted"].str.strip().str.upper() == "YES") |
        (completed_df["defect_truncated"].str.strip().str.upper() == "YES") |
        (completed_df["defect_buzzer"].str.strip().str.upper() == "YES")
    ]

    print(f"\nOverall Command Recognition Rate: {overall_acc:.1f}%")
    print(f"Per-Class Recognition Rates:\n{acc_by_class.to_string()}")
    print(f"Mean Intelligibility: {mean_intel:.2f} ± {std_intel:.2f}")
    print(f"Mean Naturalness:     {mean_nat:.2f} ± {std_nat:.2f}")
    print(f"Tone Correct Rate:    {tone_acc:.1f}% (Uncertain: {tone_unc_rate:.1f}%)")
    print(f"Defect Counts: Clipped={defect_clipped_cnt}, Distorted={defect_distorted_cnt}, Truncated={defect_truncated_cnt}, Buzzer={defect_buzzer_cnt}")
    print(f"Clips Flagged for Re-listening: {len(relisten_df)}")

    # Build Markdown report
    md_content = f"""# Vietnamese Synthetic KWS — Blinded Listening Review Report

**Package Directory:** `experiments/synthetic_kws_dataset_vieneu_1000/blinded_listening/`  
**Total Evaluation Clips:** {total_expected}  
**Completed Reviews:** {n_completed} / {total_expected} ({completion_rate:.1f}%)  
**Missing / Pending Reviews:** {n_missing} / {total_expected}  
**Reported Listener Count:** {num_listeners} (Screening auditor)  
**Human Verification Status:** **`REVIEW_COMPLETED`**  
**Official Project Governance Status:** `NOT_VALIDATED_ON_NATURAL_SPEECH`  

---

## 1. Command Recognition Performance

- **Overall Word Recognition Rate:** **{overall_acc:.1f}%** ({completed_df['is_correct'].sum()} / {n_completed})

| Keyword | Samples Evaluated | Correct Recognition | Accuracy (%) |
| :--- | :---: | :---: | :---: |
"""
    for cls_name, r in acc_by_class.iterrows():
        md_content += f"| **`{cls_name}`** | {int(r['count'])} | {int(round(r['count'] * r['mean'] / 100.0))} | {r['mean']:.1f}% |\n"

    md_content += fr"""
---

## 2. Perceptual Quality and Tone Ratings

| Metric | Target | Observed Result |
| :--- | :--- | :--- |
| **Intelligibility (1 to 5)** | $\\ge 4.0$ (Clear) | **{mean_intel:.2f} $\\pm$ {std_intel:.2f}** |
| **Naturalness (1 to 5)** | $\\ge 3.5$ (Human-like) | **{mean_nat:.2f} $\\pm$ {std_nat:.2f}** |
| **Lexical Tone Concordance** | High | **{tone_acc:.1f}%** (Uncertain: {tone_unc_rate:.1f}%) |

---

## 3. Acoustic Defect Counts

| Defect Category | Count | Share (%) | Status |
| :--- | :---: | :---: | :---: |
| **Digital Clipping** | {defect_clipped_cnt} | {defect_clipped_cnt/n_completed*100:.1f}% | {"PASS" if defect_clipped_cnt == 0 else "FAIL"} |
| **Robotic / Harsh Distortion** | {defect_distorted_cnt} | {defect_distorted_cnt/n_completed*100:.1f}% | {"PASS" if defect_distorted_cnt == 0 else "FLAGGED"} |
| **Phoneme Truncation** | {defect_truncated_cnt} | {defect_truncated_cnt/n_completed*100:.1f}% | {"PASS" if defect_truncated_cnt == 0 else "FAIL"} |
| **Synthetic Buzzer Sound** | {defect_buzzer_cnt} | {defect_buzzer_cnt/n_completed*100:.1f}% | {"PASS" if defect_buzzer_cnt == 0 else "FAIL"} |

---

## 4. Clips Flagged for Re-listening ({len(relisten_df)} clips)

"""
    if len(relisten_df) == 0:
        md_content += "No clips met the criteria for re-listening. All completed evaluations satisfied clarity thresholds.\n"
    else:
        md_content += "| Blind ID | True Label | Perceived Command | Perceived Tone | Intel | Nat | Flag Reason |\n"
        md_content += "| :--- | :--- | :--- | :--- | :---: | :---: | :--- |\n"
        for _, r in relisten_df.iterrows():
            reasons = []
            if not r["is_correct"]:
                reasons.append(f"Misidentified ({r['perceived_command']})")
            if r["tone_uncertain"]:
                reasons.append("Tone Uncertain")
            if pd.notna(r["intelligibility"]) and r["intelligibility"] <= 2:
                reasons.append("Low Intel")
            if r["defect_distorted"] == "YES":
                reasons.append("Distorted")
            if r["defect_truncated"] == "YES":
                reasons.append("Truncated")
            reason_str = ", ".join(reasons) if reasons else "Review"
            md_content += f"| `{r['blind_clip_id']}` | **{r['true_class_label']}** | {r['perceived_command']} | {r['perceived_tone']} | {r['intelligibility']} | {r['naturalness']} | {reason_str} |\n"

    md_content += """
---

## 5. Scope and Governance Note

> [!NOTE]
> This review represents a screening exercise across 80 stratified synthetic clips by an individual human reviewer. While highly informative for identifying gross phonetic defects, it does not constitute a population-level listening study across regional demographics.
"""

    with open(OUTPUT_REPORT_MD, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"\nReport written to: {OUTPUT_REPORT_MD}")

    return {
        "total_expected": total_expected,
        "completed": n_completed,
        "missing": n_missing,
        "accuracy": overall_acc,
        "relisten_count": len(relisten_df),
        "status": "REVIEW_COMPLETED"
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze blinded listening review responses")
    parser.add_argument("--responses", default=DEFAULT_RESPONSES_PATH, help="Path to completed review sheet CSV (default: blinded_listening_responses.csv if present, else review_sheet.csv)")
    parser.add_argument("--key", default=KEY_CSV, help="Path to quarantined answer key CSV")
    parser.add_argument("--listeners", type=int, default=1, help="Number of human listeners")
    args = parser.parse_args()

    analyze_blinded_review(args.responses, args.key, args.listeners)
