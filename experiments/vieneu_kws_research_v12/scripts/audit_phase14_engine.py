"""
Audit Phase 14 Inspection Engine.
Verifies OOV split lineage, call paths of smoke test, exact streaming sequence logs,
and reproduces metrics directly from raw data.
"""

import sys
import io
import json
import hashlib
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
V12_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12"
MANIFEST_ALL = V12_DIR / "manifests" / "dataset_v12_manifest.csv"
TRAIN_MANIFEST = V12_DIR / "manifests" / "train_manifest.csv"
VAL_MANIFEST = V12_DIR / "manifests" / "val_manifest.csv"
TEST_MANIFEST = V12_DIR / "manifests" / "test_manifest.csv"
STREAMING_MANIFEST = V12_DIR / "manifests" / "streaming_eval_corpus_manifest.csv"
V11_EVAL_JSON = V12_DIR / "reports" / "eval_v11_baseline.json"
E1_EVAL_JSON = V12_DIR / "reports" / "eval_e1_results.json"

df_all = pd.read_csv(MANIFEST_ALL)
df_train = pd.read_csv(TRAIN_MANIFEST)
df_val = pd.read_csv(VAL_MANIFEST)
df_test = pd.read_csv(TEST_MANIFEST)
df_stream = pd.read_csv(STREAMING_MANIFEST)

with open(V11_EVAL_JSON, encoding="utf-8") as f:
    v11_data = json.load(f)
with open(E1_EVAL_JSON, encoding="utf-8") as f:
    e1_data = json.load(f)

print("=" * 80)
print("AUDIT A: OOV SPLIT LINEAGE & SAMPLE VERIFICATION")
print("=" * 80)

oov_all = df_all[df_all["reject_subgroup"] == "oov_word"].copy()
print(f"Total OOV clips: {len(oov_all)}")
for sp in ["train", "val", "test"]:
    sub = oov_all[oov_all["split"] == sp]
    print(f"  {sp}: {len(sub)} clips, {sub['oov_lexeme'].nunique()} unique lexemes")

focus_words = ["ghế", "cửa", "mở", "bàn", "nhà"]
for w in focus_words:
    sub = oov_all[oov_all["oov_lexeme"].str.lower() == w]
    print(f"\nLexeme: '{w}' (Total rows: {len(sub)})")
    for sp in ["train", "val", "test"]:
        s_sp = sub[sub["split"] == sp]
        if len(s_sp) > 0:
            voices = list(s_sp["voice_or_speaker_id"].unique())
            utts = s_sp["source_utterance_id"].unique()
            paths = s_sp["audio_path"].tolist()
            exist_count = sum(1 for p in paths if (PROJECT_ROOT / p).exists())
            print(f"  Split: {sp} | Count: {len(s_sp)} | Exists on disk: {exist_count}/{len(s_sp)}")
            print(f"    Voices ({len(voices)}): {voices}")
            print(f"    Source Utterance IDs ({len(utts)}): {list(utts)}")
            print(f"    Sample audio paths: {paths[:2]}")

print("\n--- Disjointness Verification across splits ---")
train_voices = set(df_train["voice_or_speaker_id"].dropna().unique())
val_voices = set(df_val["voice_or_speaker_id"].dropna().unique())
test_voices = set(df_test["voice_or_speaker_id"].dropna().unique())

print(f"Train voices: {len(train_voices)} | Val voices: {len(val_voices)} | Test voices: {len(test_voices)}")
print(f"Voice overlap Train-Val:  {train_voices.intersection(val_voices)}")
print(f"Voice overlap Train-Test: {train_voices.intersection(test_voices)}")
print(f"Voice overlap Val-Test:   {val_voices.intersection(test_voices)}")

train_oov_lex = set(df_train[df_train["reject_subgroup"] == "oov_word"]["oov_lexeme"].dropna().str.lower())
val_oov_lex = set(df_val[df_val["reject_subgroup"] == "oov_word"]["oov_lexeme"].dropna().str.lower())
test_oov_lex = set(df_test[df_test["reject_subgroup"] == "oov_word"]["oov_lexeme"].dropna().str.lower())

print(f"Train OOV lexemes: {len(train_oov_lex)}")
print(f"Val OOV lexemes:   {len(val_oov_lex)}")
print(f"Test OOV lexemes:  {len(test_oov_lex)}")
print(f"OOV Lexeme Overlap Train-Test: {train_oov_lex.intersection(test_oov_lex)}")
print(f"OOV Lexeme Overlap Val-Test:   {val_oov_lex.intersection(test_oov_lex)}")
print(f"OOV Lexeme Overlap Train-Val:  {train_oov_lex.intersection(val_oov_lex)}")

train_utts = set(df_train["source_utterance_id"].dropna().unique())
val_utts = set(df_val["source_utterance_id"].dropna().unique())
test_utts = set(df_test["source_utterance_id"].dropna().unique())
print(f"Utterance Overlap Train-Val:  {train_utts.intersection(val_utts)}")
print(f"Utterance Overlap Train-Test: {train_utts.intersection(test_utts)}")
print(f"Utterance Overlap Val-Test:   {val_utts.intersection(test_utts)}")

print("\n" + "=" * 80)
print("AUDIT C: STREAMING ERROR DECOMPOSITION & DEEP TRACE")
print("=" * 80)

v11_str = v11_data["streaming_evaluation"]
e1_str = e1_data["streaming_evaluation"]

v11_seqs = pd.DataFrame(v11_str["sequences"])
e1_seqs = pd.DataFrame(e1_str["sequences"])

print(f"v11 Exact Sequence Accuracy: {v11_str['exact_sequence_accuracy']*100:.2f}% ({v11_seqs['is_order_exact'].sum()}/40)")
print(f"E1  Exact Sequence Accuracy: {e1_str['exact_sequence_accuracy']*100:.2f}% ({e1_seqs['is_order_exact'].sum()}/40)")
print(f"v11 Matched: {v11_str['total_matched_events']}, Misses: {v11_str['total_missed_events']}, Dups: {v11_str['total_duplicate_events']}, False Ins: {v11_str['total_false_insertions']}")
print(f"E1  Matched: {e1_str['total_matched_events']}, Misses: {e1_str['total_missed_events']}, Dups: {e1_str['total_duplicate_events']}, False Ins: {e1_str['total_false_insertions']}")

# Investigate why 60 misses and 36 false insertions occurred in both
print("\nComparing v11 and E1 detected events sequence by sequence:")
identical_detections = 0
different_detections = 0
for idx in range(len(v11_seqs)):
    r11 = v11_seqs.iloc[idx]
    re1 = e1_seqs.iloc[idx]
    d11 = r11["detected"]
    de1 = re1["detected"]
    gt = r11["ground_truth"]
    if d11 == de1:
        identical_detections += 1
    else:
        different_detections += 1
        print(f"Sequence {r11['sequence_id']} ({r11['sequence_type']}, {r11['condition']}):")
        print(f"  GT:       {gt}")
        print(f"  v11 Det:  {d11} (Exact: {r11['is_order_exact']})")
        print(f"  E1  Det:  {de1} (Exact: {re1['is_order_exact']})")

print(f"\nTotal sequences with identical detections between v11 and E1: {identical_detections}/40")
print(f"Total sequences with different detections between v11 and E1: {different_detections}/40")

# Detailed metrics per model
def compute_detailed_streaming_metrics(df_sub, name):
    tot_seq = len(df_sub)
    exact_seq = int(df_sub["is_order_exact"].sum())
    tot_tgt = int(df_sub["target_count"].sum())
    tot_matched = int(df_sub["matched_count"].sum())
    tot_miss = int(df_sub["missed_count"].sum())
    tot_dup = int(df_sub["duplicate_count"].sum())
    tot_f_ins = int(df_sub["false_insertion_count"].sum())
    
    return {
        "model": name,
        "total_sequences": tot_seq,
        "exact_sequences": exact_seq,
        "exact_seq_rate": round(exact_seq / tot_seq, 4),
        "total_target_events": tot_tgt,
        "matched_target_events": tot_matched,
        "target_recall": round(tot_matched / tot_tgt, 4),
        "missed_events": tot_miss,
        "miss_rate": round(tot_miss / tot_tgt, 4),
        "duplicate_events": tot_dup,
        "false_insertions": tot_f_ins,
    }

print("\nDetailed Summary Metrics:")
print(compute_detailed_streaming_metrics(v11_seqs, "v11_frozen"))
print(compute_detailed_streaming_metrics(e1_seqs, "e1_controlled"))
