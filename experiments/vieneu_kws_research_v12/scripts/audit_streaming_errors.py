import json
import sys
import io
from pathlib import Path
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
V12_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12"

with open(V12_DIR / "reports" / "eval_v11_baseline.json", encoding="utf-8") as f:
    v11_str = json.load(f)["streaming_evaluation"]
with open(V12_DIR / "reports" / "eval_e1_results.json", encoding="utf-8") as f:
    e1_str = json.load(f)["streaming_evaluation"]

v11_seqs = pd.DataFrame(v11_str["sequences"])
e1_seqs = pd.DataFrame(e1_str["sequences"])

print("=" * 80)
print("  STREAMING EVALUATION OVERALL ERROR DECOMPOSITION (40 SEQUENCES)")
print("=" * 80)
print(f"Total Sequences:                       40")
print(f"Complete Correct Sequences in Order:   v11 = {v11_seqs['is_order_exact'].sum()}/40 ({v11_seqs['is_order_exact'].mean()*100:.1f}%) | E1 = {e1_seqs['is_order_exact'].sum()}/40 ({e1_seqs['is_order_exact'].mean()*100:.1f}%)")
print(f"Total Target Events:                   124")
print(f"Matched Target Events:                 v11 = {v11_str['total_matched_events']} ({v11_str['target_event_recall']*100:.1f}%) | E1 = {e1_str['total_matched_events']} ({e1_str['target_event_recall']*100:.1f}%)")
print(f"Missed Target Events:                  v11 = {v11_str['total_missed_events']} | E1 = {e1_str['total_missed_events']}")
print(f"Duplicate Events:                      v11 = {v11_str['total_duplicate_events']} | E1 = {e1_str['total_duplicate_events']}")
print(f"False Command Insertions (from OOV):   v11 = {v11_str['total_false_insertions']} | E1 = {e1_str['total_false_insertions']}")

print("\n" + "=" * 80)
print("  BREAKDOWN BY SEQUENCE PATTERN")
print("=" * 80)
print(f"{'Sequence Type':<25} | {'Count':<5} | {'v11 Correct':<12} | {'E1 Correct':<12} | {'v11 Misses':<10} | {'E1 Misses':<10} | {'v11 F.Ins':<10} | {'E1 F.Ins':<10}")
print("-" * 80)
for stype in sorted(e1_seqs["sequence_type"].unique()):
    s11 = v11_seqs[v11_seqs["sequence_type"] == stype]
    se1 = e1_seqs[e1_seqs["sequence_type"] == stype]
    c11 = int(s11["is_order_exact"].sum())
    ce1 = int(se1["is_order_exact"].sum())
    m11 = int(s11["missed_count"].sum())
    me1 = int(se1["missed_count"].sum())
    fi11 = int(s11["false_insertion_count"].sum())
    fie1 = int(se1["false_insertion_count"].sum())
    print(f"{stype:<25} | {len(s11):<5} | {c11:>2}/{len(s11)} ({c11/len(s11)*100:>4.1f}%) | {ce1:>2}/{len(se1)} ({ce1/len(se1)*100:>4.1f}%) | {m11:<10} | {me1:<10} | {fi11:<10} | {fie1:<10}")

print("\n" + "=" * 80)
print("  BREAKDOWN BY ACOUSTIC CONDITION (SNR)")
print("=" * 80)
print(f"{'Condition':<15} | {'Count':<5} | {'v11 Correct':<12} | {'E1 Correct':<12} | {'v11 Misses':<10} | {'E1 Misses':<10} | {'v11 F.Ins':<10} | {'E1 F.Ins':<10}")
print("-" * 80)
for cond in ["clean", "snr15", "snr10", "snr05"]:
    s11 = v11_seqs[v11_seqs["condition"] == cond]
    se1 = e1_seqs[e1_seqs["condition"] == cond]
    c11 = int(s11["is_order_exact"].sum())
    ce1 = int(se1["is_order_exact"].sum())
    m11 = int(s11["missed_count"].sum())
    me1 = int(se1["missed_count"].sum())
    fi11 = int(s11["false_insertion_count"].sum())
    fie1 = int(se1["false_insertion_count"].sum())
    print(f"{cond:<15} | {len(s11):<5} | {c11:>2}/{len(s11)} ({c11/len(s11)*100:>4.1f}%) | {ce1:>2}/{len(se1)} ({ce1/len(se1)*100:>4.1f}%) | {m11:<10} | {me1:<10} | {fi11:<10} | {fie1:<10}")

print("\n" + "=" * 80)
print("  DETAILED ERROR SAMPLES IN E1 (Where E1 Failed)")
print("=" * 80)
for idx, r in e1_seqs[~e1_seqs["is_order_exact"]].iterrows():
    print(f"[{r['sequence_id']}] Type: {r['sequence_type']} | Cond: {r['condition']}")
    print(f"  GT:       {r['ground_truth']}")
    print(f"  Detected: {r['detected']}")
    print(f"  Misses: {r['missed_count']}, Dups: {r['duplicate_count']}, False Ins: {r['false_insertion_count']}")
