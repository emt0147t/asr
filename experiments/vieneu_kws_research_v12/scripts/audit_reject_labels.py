"""
Audit FLEURS conversational REJECT clips for potential target command contamination.
Checks whether full source transcripts contain 'trái', 'phải', 'lên', 'xuống'.
"""

import sys
import io
import re
from pathlib import Path
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
FLEURS_DIR = PROJECT_ROOT / "data" / "raw" / "fleurs_vi"
MANIFEST_PATH = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "manifests" / "dataset_v12_manifest.csv"

cols = ['id', 'wav_name', 'raw_text', 'norm_text', 'tokens', 'num_samples', 'gender']
df_dev = pd.read_csv(FLEURS_DIR / 'dev.tsv', sep='\t', header=None, names=cols)
df_test = pd.read_csv(FLEURS_DIR / 'test.tsv', sep='\t', header=None, names=cols)
df_fleurs = pd.concat([df_dev, df_test], ignore_index=True)
df_fleurs['stem'] = df_fleurs['wav_name'].apply(lambda x: Path(x).stem)

v12_df = pd.read_csv(MANIFEST_PATH)
conv_df = v12_df[v12_df['reject_subgroup'] == 'conversation'].copy()
conv_merged = conv_df.merge(df_fleurs, left_on='source_utterance_id', right_on='stem', how='inner')

target_patterns = {
    'TRAI': re.compile(r'\b(trái|trai)\b', re.IGNORECASE),
    'PHAI': re.compile(r'\b(phải|phai)\b', re.IGNORECASE),
    'LEN': re.compile(r'\b(lên|len)\b', re.IGNORECASE),
    'XUONG': re.compile(r'\b(xuống|xuong)\b', re.IGNORECASE),
}

hits = []
for idx, r in conv_merged.iterrows():
    text_to_check = str(r['norm_text']) + ' ' + str(r['raw_text'])
    detected = []
    for cmd, pattern in target_patterns.items():
        if pattern.search(text_to_check):
            detected.append(cmd)
    if detected:
        hits.append({
            'sample_id': r['sample_id'],
            'split': r['split'],
            'source_utterance_id': r['source_utterance_id'],
            'detected_commands': detected,
            'raw_text': r['raw_text']
        })

print(f"Total conversation clips audited: {len(conv_merged)}")
print(f"Clips whose source transcript contains any target word: {len(hits)} / {len(conv_merged)}")

by_cmd = {}
for h in hits:
    for cmd in h['detected_commands']:
        by_cmd[cmd] = by_cmd.get(cmd, 0) + 1
print(f"By command breakdown in transcripts: {by_cmd}")

by_split = {}
for h in hits:
    by_split[h['split']] = by_split.get(h['split'], 0) + 1
print(f"By split breakdown: {by_split}")

print("\nSample occurrences (first 10):")
for h in hits[:10]:
    print(f"  [{h['split']}] {h['sample_id']} (src: {h['source_utterance_id']}): {h['detected_commands']} -> \"{h['raw_text']}\"")
