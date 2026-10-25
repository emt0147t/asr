import json
import sys
import io
from pathlib import Path
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
V12_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12"

with open(V12_DIR / "reports" / "eval_v11_baseline.json", encoding="utf-8") as f:
    v11 = json.load(f)
with open(V12_DIR / "reports" / "eval_e1_results.json", encoding="utf-8") as f:
    e1 = json.load(f)

v11_off = v11["offline_evaluation"]
e1_off = e1["offline_evaluation"]
v11_str = v11["streaming_evaluation"]
e1_str = e1["streaming_evaluation"]

print("================================================================================")
print("  OFFLINE LOCKED TEST SET COMPARISON (1,280 samples @ theta = 0.20)")
print("================================================================================")
print(f"{'Metric':<35} | {'v11 Frozen Baseline':<20} | {'E1 Controlled':<20} | {'Delta':<10}")
print("-" * 92)
print(f"{'Overall Accuracy':<35} | {v11_off['overall_accuracy']*100:>18.2f}% | {e1_off['overall_accuracy']*100:>18.2f}% | {((e1_off['overall_accuracy']-v11_off['overall_accuracy'])*100):>+9.2f}%")
print(f"{'Macro F1 (all 5 classes)':<35} | {v11_off['macro_f1']:>19.4f}  | {e1_off['macro_f1']:>19.4f}  | {(e1_off['macro_f1']-v11_off['macro_f1']):>+9.4f}")
print(f"{'Commands Macro F1':<35} | {v11_off['command_macro_f1']:>19.4f}  | {e1_off['command_macro_f1']:>19.4f}  | {(e1_off['command_macro_f1']-v11_off['command_macro_f1']):>+9.4f}")
print(f"{'Clean Command Recall':<35} | {v11_off['clean_command_recall']*100:>18.2f}% | {e1_off['clean_command_recall']*100:>18.2f}% | {((e1_off['clean_command_recall']-v11_off['clean_command_recall'])*100):>+9.2f}%")
print(f"{'Noisy Command Recall':<35} | {v11_off['noisy_command_recall']*100:>18.2f}% | {e1_off['noisy_command_recall']*100:>18.2f}% | {((e1_off['noisy_command_recall']-v11_off['noisy_command_recall'])*100):>+9.2f}%")

print("\nPer-Class F1 Scores:")
for c in ["TRAI", "PHAI", "LEN", "XUONG", "REJECT"]:
    f1_11 = v11_off["per_class"][c]["f1_score"]
    f1_e1 = e1_off["per_class"][c]["f1_score"]
    print(f"  {c:<8}: v11 = {f1_11:.4f} -> E1 = {f1_e1:.4f} (Delta: {f1_e1 - f1_11:+.4f})")

print("\nNegative Subgroup Rejection Rates (Higher is better):")
for sg, label in [
    ("conversation", "FLEURS Conversational"),
    ("environmental_noise", "Environmental Noise"),
    ("silence_non_speech", "Silence / Non-speech"),
    ("oov_word", "OOV Monosyllables Total"),
    ("oov_ghe_isolated", "Unseen 'ghế' (Isolated)"),
    ("oov_other_unseen", "Unseen Other Monosyllables")
]:
    r11 = v11_off["reject_breakdown"][sg]["rejection_rate"] * 100
    re1 = e1_off["reject_breakdown"][sg]["rejection_rate"] * 100
    fa11 = v11_off["reject_breakdown"][sg].get("false_acceptance_count", v11_off["reject_breakdown"][sg].get("false_accepts", 0))
    fae1 = e1_off["reject_breakdown"][sg].get("false_acceptance_count", e1_off["reject_breakdown"][sg].get("false_accepts", 0))
    total_sg = v11_off["reject_breakdown"][sg]["count"]
    print(f"  {label:<30}: v11 = {r11:>6.2f}% (FA: {fa11:>2}/{total_sg}) -> E1 = {re1:>6.2f}% (FA: {fae1:>2}/{total_sg}) | Delta: {re1 - r11:>+6.2f}%")

print("\nTarget Recall by SNR Tier:")
for snr in ["20dB", "15dB", "10dB", "5dB", "0dB"]:
    r11 = v11_off["snr_breakdown"][snr]["recall"] * 100
    re1 = e1_off["snr_breakdown"][snr]["recall"] * 100
    print(f"  {snr:<8}: v11 = {r11:>6.2f}% -> E1 = {re1:>6.2f}% | Delta: {re1 - r11:>+6.2f}%")

print("\n" + "=" * 80)
print("  STREAMING EVALUATION COMPARISON (40 sequences)")
print("=" * 80)
print(f"{'Exact Sequence Accuracy':<35} | {v11_str['exact_sequence_accuracy']*100:>18.2f}% | {e1_str['exact_sequence_accuracy']*100:>18.2f}% | {((e1_str['exact_sequence_accuracy']-v11_str['exact_sequence_accuracy'])*100):>+9.2f}%")
print(f"{'Target Event Recall':<35} | {v11_str['target_event_recall']*100:>18.2f}% | {e1_str['target_event_recall']*100:>18.2f}% | {((e1_str['target_event_recall']-v11_str['target_event_recall'])*100):>+9.2f}%")
print(f"{'Target Event Precision':<35} | {v11_str['target_event_precision']*100:>18.2f}% | {e1_str['target_event_precision']*100:>18.2f}% | {((e1_str['target_event_precision']-v11_str['target_event_precision'])*100):>+9.2f}%")
print(f"{'Duplicate Events':<35} | {v11_str['total_duplicate_events']:>19d}  | {e1_str['total_duplicate_events']:>19d}  | {e1_str['total_duplicate_events']-v11_str['total_duplicate_events']:>+9d}")
print(f"{'False Insertions from OOV':<35} | {v11_str['total_false_insertions']:>19d}  | {e1_str['total_false_insertions']:>19d}  | {e1_str['total_false_insertions']-v11_str['total_false_insertions']:>+9d}")

print("\nInference Latency (x86 CPU):")
print(f"  v11 P95: {v11_off['latency']['p95_ms']:.2f} ms")
print(f"  E1  P95: {e1_off['latency']['p95_ms']:.2f} ms")
