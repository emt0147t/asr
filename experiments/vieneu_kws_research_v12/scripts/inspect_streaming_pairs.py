import json
import sys
import io
from pathlib import Path
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

V12_DIR = Path('experiments/vieneu_kws_research_v12')
v11 = json.load(open(V12_DIR / 'reports/eval_v11_baseline.json', encoding='utf-8'))['streaming_evaluation']
e1 = json.load(open(V12_DIR / 'reports/eval_e1_results.json', encoding='utf-8'))['streaming_evaluation']

v11_seqs = v11['sequences']
e1_seqs = e1['sequences']

diff_matched = 0
diff_missed = 0
diff_false_ins = 0

print("=== SEQUENCE-LEVEL EVENT DIFFERENCES BETWEEN v11 AND E1 ===")
for i, (s11, se1) in enumerate(zip(v11_seqs, e1_seqs)):
    m11, me1 = s11['matched_count'], se1['matched_count']
    ms11, mse1 = s11['missed_count'], se1['missed_count']
    fi11, fie1 = s11['false_insertion_count'], se1['false_insertion_count']
    d11, de1 = s11['duplicate_count'], se1['duplicate_count']
    
    if (m11 != me1) or (ms11 != mse1) or (fi11 != fie1) or (d11 != de1):
        print(f"\nSeq {s11['sequence_id']} ({s11['sequence_type']}, {s11['condition']}):")
        print(f"  GT: {s11['ground_truth']}")
        print(f"  v11 det: {s11['detected']} -> match={m11}, miss={ms11}, fi={fi11}, dup={d11}")
        print(f"  E1  det: {se1['detected']} -> match={me1}, miss={mse1}, fi={fie1}, dup={de1}")
        diff_matched += abs(m11 - me1)
        diff_missed += abs(ms11 - mse1)
        diff_false_ins += abs(fi11 - fie1)

print(f"\nNet Differences across sequences:")
print(f"  Total target events across 40 sequences: 124")
print(f"  v11 matched: {sum(s['matched_count'] for s in v11_seqs)} / 124")
print(f"  E1  matched: {sum(s['matched_count'] for s in e1_seqs)} / 124")
print(f"  v11 missed:  {sum(s['missed_count'] for s in v11_seqs)} / 124")
print(f"  E1  missed:  {sum(s['missed_count'] for s in e1_seqs)} / 124")
print(f"  v11 false insertions: {sum(s['false_insertion_count'] for s in v11_seqs)}")
print(f"  E1  false insertions: {sum(s['false_insertion_count'] for s in e1_seqs)}")
print(f"  v11 duplicates:       {sum(s['duplicate_count'] for s in v11_seqs)}")
print(f"  E1  duplicates:       {sum(s['duplicate_count'] for s in e1_seqs)}")
