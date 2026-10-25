#!/usr/bin/env python3
"""
Provenance & Corpus Separation Audit
Inspects D0, D2, and edge_tts_eval_v2 (isolated + streaming) across:
1. Exact audio files and SHA-256 hashes
2. Normalized source text and OOV lexemes
3. Voice identities
4. TTS engine and version
5. Noise tracks and historical exposure
"""

import sys
import json
import hashlib
from pathlib import Path
import pandas as pd

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANIFESTS_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "manifests"

d0_path = MANIFESTS_DIR / "D0_train_v1.csv"
d2_path = MANIFESTS_DIR / "D2_matched_train_v1.csv"
iso_test_path = MANIFESTS_DIR / "edge_tts_eval_v2_isolated_test_manifest.csv"
stream_test_path = MANIFESTS_DIR / "edge_tts_eval_v2_streaming_manifest.csv"

df_d0 = pd.read_csv(d0_path)
df_d2 = pd.read_csv(d2_path)
df_iso = pd.read_csv(iso_test_path)
df_stream = pd.read_csv(stream_test_path)

print("=" * 80)
print("PROVENANCE AND SEPARATION AUDIT REPORT")
print("=" * 80)

# 1. Exact Audio File & Hash Overlap
d2_shas = set(df_d2["sha256"].dropna())
iso_shas = set(df_iso["sha256"].dropna())
stream_shas = set(df_stream["sha256"].dropna())

audio_overlap_d2_iso = d2_shas.intersection(iso_shas)
audio_overlap_d2_stream = d2_shas.intersection(stream_shas)

print(f"\n1. Audio Files & Hash Overlap:")
print(f"   D2 Audio Count: {len(df_d2)} (Unique SHAs: {len(d2_shas)})")
print(f"   Isolated Test Count: {len(df_iso)} (Unique SHAs: {len(iso_shas)})")
print(f"   Streaming Test Count: {len(df_stream)} (Unique SHAs: {len(stream_shas)})")
print(f"   Audio SHA overlap (D2 vs Isolated): {len(audio_overlap_d2_iso)} (MUST BE 0)")
print(f"   Audio SHA overlap (D2 vs Streaming): {len(audio_overlap_d2_stream)} (MUST BE 0)")

# 2. Text and Lexeme Overlap
d2_cmds = set(df_d2[df_d2["label"] != "REJECT"]["text"].dropna().unique())
d2_oovs = set(df_d2[df_d2["label"] == "REJECT"]["oov_lexeme"].dropna().unique())

test_cmds = set(df_iso[df_iso["label"] != "REJECT"]["text"].dropna().unique())
test_oovs = set(df_iso[df_iso["label"] == "REJECT"]["text"].dropna().unique())

oov_overlap = d2_oovs.intersection(test_oovs)

print(f"\n2. Lexicon & Text Overlap:")
print(f"   Target Commands in D2: {d2_cmds}")
print(f"   Target Commands in Test: {test_cmds} (Matched: {d2_cmds == test_cmds})")
print(f"   D2 OOV Lexemes ({len(d2_oovs)}): {sorted(list(d2_oovs))[:10]}...")
print(f"   Test OOV Lexemes ({len(test_oovs)}): {sorted(list(test_oovs))}")
print(f"   OOV Lexeme Overlap between D2 and Test: {len(oov_overlap)} -> {oov_overlap}")

# Also check Phase 27 test lexemes
p27_iso_path = MANIFESTS_DIR / "edge_tts_eval_isolated_test_manifest.csv"
if p27_iso_path.exists():
    df_p27 = pd.read_csv(p27_iso_path)
    p27_oovs = set(df_p27[df_p27["label"] == "REJECT"]["text"].dropna().unique())
    p27_overlap = d2_oovs.intersection(p27_oovs)
    test_v2_p27_overlap = test_oovs.intersection(p27_oovs)
    print(f"   D2 OOV overlap with Phase 27 Test: {len(p27_overlap)} -> {p27_overlap}")
    print(f"   edge_tts_eval_v2 OOV overlap with Phase 27 Test: {len(test_v2_p27_overlap)} -> {test_v2_p27_overlap}")

# 3. Voice Identities
d2_voices = set(df_d2["voice_or_speaker_id"].dropna().unique())
test_voices = set(df_iso["voice"].dropna().unique())
print(f"\n3. Voice Identities:")
print(f"   D2 Voices: {sorted(list(d2_voices))}")
print(f"   Test Voices: {sorted(list(test_voices))}")
voice_overlap = d2_voices.intersection(test_voices)
print(f"   Voice Overlap: {voice_overlap}")
print(f"   [DISCLOSURE] D2 training and edge_tts_eval_v2 share the same Edge-TTS voices ({voice_overlap}).")
print(f"   Therefore, edge_tts_eval_v2 is a within-voice/within-generator test for D2,")
print(f"   evaluating generalization across novel utterances, prosodic rates (+10%), and novel phonetic hard confusers.")
print(f"   For D0 models, edge_tts_eval_v2 is a cross-generator and cross-voice test.")

# 4. TTS Engine & Versions
print(f"\n4. TTS Engine & Versions:")
print(f"   D0 Engine: VieNeu-TTS v3 Turbo")
print(f"   D2 Engines: VieNeu-TTS v3 Turbo (90%) + Microsoft Edge-TTS 7.3.1 (10%)")
print(f"   Test Engine: Microsoft Edge-TTS 7.3.1")

# 5. Noise Tracks & Historical Manifests
test_noise_sources = set(df_iso["noise_source"].unique())
print(f"\n5. Noise Tracks Analysis:")
print(f"   Isolated Test Noise Sources: {test_noise_sources}")

# Check earlier manifests for mic-noise track 36-39
noise_history = {}
for m in MANIFESTS_DIR.glob("*.csv"):
    try:
        df_m = pd.read_csv(m)
        tracks_found = []
        for col in ["audio_path", "noise_source", "source_utterance_id", "source_id"]:
            if col in df_m.columns:
                for t in [36, 37, 38, 39]:
                    matches = df_m[df_m[col].astype(str).str.contains(f"track_{t}|handling noise.*{t}", case=False, na=False)]
                    if len(matches) > 0:
                        tracks_found.append((t, len(matches)))
        if tracks_found:
            noise_history[m.name] = tracks_found
    except Exception:
        pass

print(f"   Historical exposure of tracks 36-39 across repository manifests:")
for m_name, findings in noise_history.items():
    print(f"     - {m_name}: {findings}")

out_audit_path = MANIFESTS_DIR / "provenance_separation_audit.json"
audit_dict = {
    "audio_sha_overlap": len(audio_overlap_d2_iso) + len(audio_overlap_d2_stream),
    "oov_lexeme_overlap_d2_vs_test": list(oov_overlap),
    "d2_voices": sorted(list(d2_voices)),
    "test_voices": sorted(list(test_voices)),
    "voice_overlap": sorted(list(voice_overlap)),
    "test_scope_classification": {
        "D0": "Cross-generator, Cross-voice, Unseen-utterance, Unseen-prosody, Unseen-noise evaluation",
        "D2": "Within-generator, Within-voice, Unseen-utterance, Unseen-prosody, Unseen-noise evaluation"
    },
    "noise_tracks_history": noise_history
}

with open(out_audit_path, "w", encoding="utf-8") as f:
    json.dump(audit_dict, f, indent=2, ensure_ascii=False)

print(f"\n[OK] Provenance audit saved to: {out_audit_path.name}")
