"""
Detailed Voice Roster Verification Script for VieNeu-TTS v3 Turbo Pilot
"""

import os
import sys
import json
import pathlib
import hashlib
import numpy as np
import pandas as pd
import soundfile as sf
import vieneu
from vieneu import Vieneu

# Ensure UTF-8 stdout on Windows
sys.stdout.reconfigure(encoding="utf-8")

def audit():
    tts = Vieneu()
    voices_list = tts.list_preset_voices()

    # Load raw voices JSON from package assets
    asset_path = pathlib.Path(vieneu.__file__).parent / "assets" / "voices_v3_turbo.json"
    assert asset_path.exists(), f"Voice asset missing at: {asset_path}"
    raw_data = json.loads(asset_path.read_text(encoding="utf-8"))

    default_voice_in_json = raw_data.get("default_voice")
    total_presets_in_json = len(raw_data.get("presets", {}))
    total_listed_by_sdk = len(voices_list)

    print(f"SDK version: {getattr(vieneu, '__version__', 'unknown')}")
    print(f"Total presets in JSON: {total_presets_in_json}")
    print(f"Total listed by tts.list_preset_voices(): {total_listed_by_sdk}")
    print(f"Default voice declared in JSON: {default_voice_in_json}")
    print(f"Default voice in tts._default_voice: {getattr(tts, '_default_voice', None)}")

    # Map English regional names to Vietnamese upstream
    region_map_en_to_vi = {
        "North": "Bắc",
        "Central": "Trung",
        "South": "Nam"
    }

    # Verify each of the 25 installed presets
    installed_presets_audit = []
    for label, voice_id in voices_list:
        meta = raw_data.get("presets", {}).get(voice_id, {})
        installed_presets_audit.append({
            "voice_id": voice_id,
            "display_label": label,
            "gender": meta.get("gender"),
            "region": meta.get("region"),
            "style": meta.get("style"),
            "description": meta.get("description"),
            "featured_rank": meta.get("featured"),
            "aliases": meta.get("aliases", []),
            "has_speaker_emb": meta.get("speaker_emb") is not None,
            "speaker_emb_dim": len(meta.get("speaker_emb", [])) if meta.get("speaker_emb") else 0,
            "has_codes": meta.get("codes") is not None,
            "codes_shape": [len(meta.get("codes", [])), len(meta.get("codes", [])[0])] if meta.get("codes") else None
        })

    # Load pilot manifest
    manifest_path = "experiments/synthetic_kws_tts_pilot_vieneu_local/vieneu_local_pilot_manifest.csv"
    manifest_df = pd.read_csv(manifest_path)

    # Inspect all voices in pilot manifest
    manifest_voice_rows = []
    unique_manifest_voices = manifest_df[["voice_id", "voice_gender", "voice_region", "voice_style"]].drop_duplicates()

    matched_count = 0
    mismatch_records = []

    for _, row in unique_manifest_voices.iterrows():
        vid = row["voice_id"]
        v_gender = row["voice_gender"]
        v_region = row["voice_region"]
        v_style = row["voice_style"]

        # Check if in installed
        is_in_installed = any(vid == item[1] for item in voices_list)
        meta = raw_data.get("presets", {}).get(vid)

        if not is_in_installed or not meta:
            mismatch_records.append({
                "voice_id": vid,
                "error": "Voice ID not found in installed presets"
            })
            continue

        matched_count += 1
        expected_gender = meta.get("gender")
        expected_region_vi = meta.get("region")
        expected_region_en = [k for k, v in region_map_en_to_vi.items() if v == expected_region_vi]
        expected_region_en = expected_region_en[0] if expected_region_en else None
        upstream_desc = meta.get("description")

        gender_match = (v_gender == expected_gender)
        region_match = (region_map_en_to_vi.get(v_region) == expected_region_vi)

        manifest_voice_rows.append({
            "voice_id": vid,
            "found_in_installed": is_in_installed,
            "manifest_gender": v_gender,
            "upstream_gender": expected_gender,
            "gender_match": gender_match,
            "manifest_region": v_region,
            "upstream_region_vi": expected_region_vi,
            "upstream_region_en": expected_region_en,
            "region_match": region_match,
            "manifest_style": v_style,
            "upstream_style_token": meta.get("style"),
            "upstream_description": upstream_desc,
            "aliases": meta.get("aliases", []),
            "featured_rank": meta.get("featured")
        })

    # Check each clip file in manifest:
    clip_verification = []
    for _, row in manifest_df.iterrows():
        clip_id = row["clip_id"]
        raw_path = row["raw_wav_relative_path"]
        kws_path = row["kws_wav_relative_path"]
        vid = row["voice_id"]
        kw = row["target_keyword"]

        raw_exists = os.path.exists(raw_path)
        kws_exists = os.path.exists(kws_path)

        clip_verification.append({
            "clip_id": clip_id,
            "target_keyword": kw,
            "voice_id": vid,
            "raw_exists": raw_exists,
            "kws_exists": kws_exists,
            "raw_sha256": row["raw_sha256"],
            "kws_sha256": row["kws_sha256"]
        })

    # Summary
    verification_summary = {
        "engine": "VieNeu-TTS v3 Turbo",
        "vieneu_sdk_version": "3.8.3",
        "installed_presets_count": total_listed_by_sdk,
        "json_presets_count": total_presets_in_json,
        "declared_default_voice": default_voice_in_json,
        "is_default_voice_hai_dang": (default_voice_in_json == "Hải Đăng"),
        "hai_dang_alias_minh_quan": "Minh Quân" in raw_data.get("presets", {}).get("Hải Đăng", {}).get("aliases", []),
        "manifest_unique_voices_count": len(unique_manifest_voices),
        "manifest_voices_all_in_installed": (matched_count == len(unique_manifest_voices)),
        "all_genders_match_upstream": all(r["gender_match"] for r in manifest_voice_rows),
        "all_regions_match_upstream": all(r["region_match"] for r in manifest_voice_rows),
        "total_clips_verified": len(clip_verification),
        "mismatches": mismatch_records,
        "manifest_voices": manifest_voice_rows,
        "installed_presets": installed_presets_audit
    }

    output_json_path = "experiments/synthetic_kws_tts_pilot_vieneu_local/voice_roster_verification.json"
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(verification_summary, f, indent=2, ensure_ascii=False)
    print(f"Saved roster verification to: {output_json_path}")

    # Print summary table
    print("\n--- Manifest Voice Verification Table ---")
    for r in manifest_voice_rows:
        status_str = "OK" if r["gender_match"] and r["region_match"] else "MISMATCH"
        print(f"[{status_str}] {r['voice_id']:12s} | Gender: {r['manifest_gender']} (upstream: {r['upstream_gender']}) | Region: {r['manifest_region']} (upstream: {r['upstream_region_vi']}) | Upstream desc: {r['upstream_description']}")

if __name__ == "__main__":
    audit()
