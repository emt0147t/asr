import json
import io
import sys
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
V12_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12"

sys.path.insert(0, str(V12_DIR / "scripts"))
from eval_kws import KWSEvaluator, preprocess_offline_audio

ev_v11 = KWSEvaluator(PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4" / "bundle" / "checkpoint" / "variant_v11_hard_negative_weighted_seed42.pt")
ev_e1 = KWSEvaluator(V12_DIR / "checkpoints" / "exp_v12_e1_baseline_scale_seed42.pt")

df_manifest = pd.read_csv(V12_DIR / "manifests" / "dataset_v12_manifest.csv")
oov_df = df_manifest[df_manifest["reject_subgroup"] == "oov_word"].copy()

print(f"Total OOV clips in Dataset v12: {len(oov_df)}")
print("Distribution by split:")
for split, count in oov_df["split"].value_counts().items():
    print(f"  {split}: {count} clips ({oov_df[oov_df['split']==split]['oov_lexeme'].nunique()} unique lexemes)")

# Focus on specific lexemes: ghế, cửa, mở, bàn, nhà and others
focus_lexemes = ["ghế", "cửa", "mở", "bàn", "nhà"]

print("\n" + "=" * 90)
print(f"{'Lexeme':<10} | {'Split':<8} | {'Total':<6} | {'v11 REJ':<8} | {'v11 FA':<8} | {'v11 Predictions':<18} | {'E1 REJ':<8} | {'E1 FA':<8} | {'E1 Predictions':<18}")
print("-" * 90)

for lexeme in focus_lexemes:
    sub = oov_df[oov_df["oov_lexeme"].str.lower() == lexeme]
    splits = sub["split"].unique()
    for s in splits:
        s_sub = sub[sub["split"] == s]
        v11_preds = []
        e1_preds = []
        for idx, row in s_sub.iterrows():
            audio = preprocess_offline_audio(PROJECT_ROOT / row["audio_path"])
            p11, _, _, _ = ev_v11.predict_clip(audio)
            pe1, _, _, _ = ev_e1.predict_clip(audio)
            v11_preds.append(p11)
            e1_preds.append(pe1)
        
        v11_rej = sum(1 for p in v11_preds if p == "REJECT")
        v11_fa = len(v11_preds) - v11_rej
        e1_rej = sum(1 for p in e1_preds if p == "REJECT")
        e1_fa = len(e1_preds) - e1_rej
        v11_dist = dict(Counter(v11_preds))
        e1_dist = dict(Counter(e1_preds))
        print(f"{lexeme:<10} | {s:<8} | {len(s_sub):<6} | {v11_rej:<8} | {v11_fa:<8} | {str(v11_dist):<18} | {e1_rej:<8} | {e1_fa:<8} | {str(e1_dist):<18}")

# Also check historical staging "cửa" files
cua_staging = list((V12_DIR / "audio" / "staging_oov").glob("*c*a*.wav"))
print(f"\nHistorical Staging 'cửa' clips ({len(cua_staging)} clips):")
v11_c_preds = []
e1_c_preds = []
for f in cua_staging:
    audio = preprocess_offline_audio(f)
    p11, c11, _, _ = ev_v11.predict_clip(audio)
    pe1, ce1, _, _ = ev_e1.predict_clip(audio)
    v11_c_preds.append((p11, c11))
    e1_c_preds.append((pe1, ce1))
    print(f"  {f.name}: v11 -> {p11} ({c11:.3f}) | E1 -> {pe1} ({ce1:.3f})")

# Test set OOV breakdown (all 31 test lexemes)
print("\n" + "=" * 90)
print("  TEST SET OOV LEXEMES (256 clips across 31 unseen words)")
print("=" * 90)
test_oov = oov_df[oov_df["split"] == "test"]
test_lexemes = sorted(test_oov["oov_lexeme"].unique())

print(f"{'Lexeme':<12} | {'Count':<6} | {'v11 REJ':<8} | {'v11 FA':<8} | {'v11 FAs':<20} | {'E1 REJ':<8} | {'E1 FA':<8} | {'E1 FAs':<20}")
print("-" * 90)

e1_test_fa_total = 0
v11_test_fa_total = 0

for lex in test_lexemes:
    sub = test_oov[test_oov["oov_lexeme"] == lex]
    v11_p = []
    e1_p = []
    for idx, row in sub.iterrows():
        audio = preprocess_offline_audio(PROJECT_ROOT / row["audio_path"])
        p11, _, _, _ = ev_v11.predict_clip(audio)
        pe1, _, _, _ = ev_e1.predict_clip(audio)
        v11_p.append(p11)
        e1_p.append(pe1)
    
    v11_rej = sum(1 for p in v11_p if p == "REJECT")
    v11_fa = len(v11_p) - v11_rej
    e1_rej = sum(1 for p in e1_p if p == "REJECT")
    e1_fa = len(e1_p) - e1_rej
    
    v11_fa_cmds = [p for p in v11_p if p != "REJECT"]
    e1_fa_cmds = [p for p in e1_p if p != "REJECT"]

    v11_test_fa_total += v11_fa
    e1_test_fa_total += e1_fa

    print(f"{lex:<12} | {len(sub):<6} | {v11_rej:<8} | {v11_fa:<8} | {str(dict(Counter(v11_fa_cmds))):<20} | {e1_rej:<8} | {e1_fa:<8} | {str(dict(Counter(e1_fa_cmds))):<20}")

print("-" * 90)
print(f"Total Test OOV FA: v11 = {v11_test_fa_total} / {len(test_oov)} ({v11_test_fa_total/len(test_oov)*100:.1f}%) | E1 = {e1_test_fa_total} / {len(test_oov)} ({e1_test_fa_total/len(test_oov)*100:.1f}%)")
