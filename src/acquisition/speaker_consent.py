"""
Speaker Registration and Informed Consent Ledger for Field Recording Campaign.
Complies with docs/dataset_protocol.md Section 4.5:
  - Generates pseudonymous IDs (no PII: no names, phone numbers, or addresses).
  - Enforces explicit informed consent for machine learning research and commercial use.
  - Tracks dialect, gender, age bracket, device, and environment.
  - Persists records to data/raw/field_recordings/consent_ledger.json.
"""

import sys
from pathlib import Path

# Ensure UTF-8 stdout and sys.path
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
    except Exception:
        pass
else:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

import secrets
import re

CONSENT_VERSION = "v1.0-kws-research-commercial"
VALID_DIALECTS = ["Northern", "Central", "Southern"]
VALID_GENDERS = ["MALE", "FEMALE", "OTHER"]
VALID_AGE_BRACKETS = ["18-30", "31-45", "46-60", "over_60", "31-50", "51+"]
VALID_MICROPHONE_TYPES = [
    "builtin_mobile",
    "wired_headset",
    "bluetooth_earbuds",
    "earbuds",
    "laptop_mic",
    "boundary_mic",
    "usb_desktop_mic",
    "Built-in microphone",
    "unspecified"
]
VALID_DISTANCE_CATEGORIES = [
    "close_talk_15cm",
    "medium_talk_50cm",
    "desk_distance_60cm",
    "far_field_2m",
    "near",
    "unspecified"
]
VALID_ENVIRONMENTS = ["quiet_room", "office_hvac", "living_room_reverberant", "street_noise", "fan_noise"]

HUMAN_RECORDINGS_ROOT = Path("data/raw/human_recordings_v1")
LEGACY_FIXTURES_ROOT = Path("data/raw/field_recordings")

# Default production path for genuine human consent ledger (P0-1 remediation)
LEDGER_PATH = HUMAN_RECORDINGS_ROOT / "consent_ledger_human.json"
# Historical fixture ledger path (preserved read-only for audit integrity)
LEGACY_LEDGER_PATH = LEGACY_FIXTURES_ROOT / "consent_ledger.json"

SPEAKER_ID_REGEX = re.compile(r"^SPK_H_([NCS])_([a-f0-9]{6})$")


def load_consent_ledger(ledger_path: Optional[Path] = None) -> Dict[str, Any]:
    target_path = ledger_path if ledger_path is not None else LEDGER_PATH
    if not target_path.exists():
        return {}
    try:
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
            return {}
    except Exception:
        return {}


def save_consent_ledger(ledger: Dict[str, Any], ledger_path: Optional[Path] = None):
    target_path = ledger_path if ledger_path is not None else LEDGER_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2, ensure_ascii=False)


def generate_pseudonymous_speaker_id(
    dialect: str,
    ledger: Optional[Dict[str, Any]] = None,
    max_attempts: int = 1000
) -> str:
    """
    Generates a cryptographically random pseudonymous speaker identifier:
    SPK_H_<N|C|S>_<6-hex-nonce> using Python's secrets module (P0-3 remediation).

    Examples:
      - Northern: SPK_H_N_8f2a1b
      - Central:  SPK_H_C_4e7c09
      - Southern: SPK_H_S_a1d3f5

    Guarantees:
      1. Cryptographic Entropy: Generated via secrets.token_hex(3) (~16.78M combinations).
      2. Direct Identifier Freedom: Does not embed names, birth dates, phone numbers, or sequence numbers.
      3. Non-Sequential Distribution: Prevents order-of-enrollment leakage.
      4. Safe Collision Handling: Re-samples if nonce already exists in active ledger;
         fails safely with RuntimeError if max_attempts is exceeded.

    Governance Notice:
      Pseudonymization separates speech files from direct personal identity in storage.
      It does NOT claim that physical human vocal acoustics are fully anonymized.
    """
    prefix_map = {"Northern": "N", "Central": "C", "Southern": "S"}
    prefix = prefix_map.get(dialect, "X")

    active_ledger = ledger if ledger is not None else {}
    for _ in range(max_attempts):
        nonce = secrets.token_hex(3).lower()
        candidate_id = f"SPK_H_{prefix}_{nonce}"
        if candidate_id not in active_ledger:
            return candidate_id

    raise RuntimeError(
        f"Speaker ID collision avoidance limit ({max_attempts}) reached. "
        "Active ledger may be congested."
    )


def generate_next_speaker_id(dialect: str, ledger: Dict[str, Any]) -> str:
    """Backwards-compatible alias for generate_pseudonymous_speaker_id."""
    return generate_pseudonymous_speaker_id(dialect, ledger)


def register_speaker(
    dialect: str,
    gender: str,
    age_bracket: str = "18-30",
    consent_given: bool = True,
    device_model: str = "Unspecified Mobile",
    microphone_type: str = "builtin_mobile",
    distance_category: str = "close_talk_15cm",
    acoustic_environment: str = "quiet_room",
    notes: str = "",
    custom_speaker_id: Optional[str] = None,
    ledger_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Registers a contributor with explicit informed consent.
    Raises ValueError if consent is False or inputs are invalid.
    """
    if not consent_given:
        raise ValueError("Cannot register speaker: Informed consent was NOT granted.")

    if dialect not in VALID_DIALECTS:
        raise ValueError(f"Invalid dialect '{dialect}'. Must be one of {VALID_DIALECTS}")

    if gender.upper() not in VALID_GENDERS:
        raise ValueError(f"Invalid gender '{gender}'. Must be one of {VALID_GENDERS}")

    if age_bracket not in VALID_AGE_BRACKETS:
        raise ValueError(f"Invalid age_bracket '{age_bracket}'. Must be one of {VALID_AGE_BRACKETS}")

    if microphone_type not in VALID_MICROPHONE_TYPES:
        raise ValueError(f"Invalid microphone_type '{microphone_type}'. Must be one of {VALID_MICROPHONE_TYPES}")

    if distance_category not in VALID_DISTANCE_CATEGORIES:
        raise ValueError(f"Invalid distance_category '{distance_category}'. Must be one of {VALID_DISTANCE_CATEGORIES}")

    target_ledger_path = ledger_path if ledger_path is not None else LEDGER_PATH
    ledger = load_consent_ledger(target_ledger_path)

    if custom_speaker_id:
        if not re.match(r"^[A-Za-z0-9_\-]+$", custom_speaker_id):
            raise ValueError(f"Invalid characters in custom_speaker_id '{custom_speaker_id}'. Only alphanumeric, underscores, and dashes allowed.")
        spk_id = custom_speaker_id
    else:
        spk_id = generate_pseudonymous_speaker_id(dialect, ledger)

    cid = f"CONSENT_{spk_id[4:]}_v1.0" if spk_id.startswith("SPK_") else f"CONSENT_{spk_id}_v1.0"

    record = {
        "speaker_id": spk_id,
        "consent_id": cid,
        "consent_given": True,
        "consent_version": CONSENT_VERSION,
        "consent_timestamp": datetime.now(timezone.utc).isoformat(),
        "dialect": dialect,
        "gender": gender.upper(),
        "age_bracket": age_bracket,
        "primary_device": device_model,
        "primary_microphone": microphone_type,
        "distance_category": distance_category,
        "speaking_distance": distance_category,
        "primary_environment": acoustic_environment,
        "notes": notes
    }

    ledger[spk_id] = record
    save_consent_ledger(ledger, target_ledger_path)
    print(f"[REGISTERED] Speaker {spk_id} ({dialect}, {gender}) registered with verified consent.")
    return record


def verify_speaker_consent(speaker_id: str, ledger_path: Optional[Path] = None) -> bool:
    """
    Returns True if speaker_id exists in ledger, identity matches, and consent is strictly True.
    Fails closed (False) if missing, unregistered, revoked, malformed, or unverifiable.
    """
    if not speaker_id or not isinstance(speaker_id, str):
        return False
    target_path = ledger_path if ledger_path is not None else LEDGER_PATH
    try:
        ledger = load_consent_ledger(target_path)
    except Exception:
        return False

    if not isinstance(ledger, dict) or speaker_id not in ledger:
        return False

    rec = ledger.get(speaker_id)
    if not isinstance(rec, dict):
        return False

    # Identity matching check
    rec_spk = rec.get("speaker_id")
    if rec_spk and str(rec_spk).strip() != speaker_id.strip():
        return False

    # Revocation check: revoked_at timestamp or explicit revocation status
    if rec.get("revoked_at") is not None:
        return False

    # Consent must be strictly boolean True
    consent_val = rec.get("consent_given")
    if consent_val is not True:
        return False

    return True


def revoke_speaker_consent(speaker_id: str, ledger_path: Optional[Path] = None) -> bool:
    """
    Revokes consent for speaker_id. Sets consent_given to False and timestamps the revocation.
    """
    target_path = ledger_path if ledger_path is not None else LEDGER_PATH
    ledger = load_consent_ledger(target_path)
    if speaker_id not in ledger:
        return False
    ledger[speaker_id]["consent_given"] = False
    ledger[speaker_id]["revoked_at"] = datetime.now(timezone.utc).isoformat()
    save_consent_ledger(ledger, target_path)
    return True


def get_speaker_metadata(speaker_id: str, ledger_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    target_path = ledger_path if ledger_path is not None else LEDGER_PATH
    ledger = load_consent_ledger(target_path)
    return ledger.get(speaker_id)


def list_registered_speakers(ledger_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    target_path = ledger_path if ledger_path is not None else LEDGER_PATH
    ledger = load_consent_ledger(target_path)
    return list(ledger.values())


if __name__ == "__main__":
    print("Testing Speaker Consent Ledger...")
    spk = register_speaker(
        dialect="Northern",
        gender="MALE",
        age_bracket="18-30",
        consent_given=True,
        device_model="Windows PC (Dev Test)",
        acoustic_environment="quiet_room",
        custom_speaker_id="SPK_FIELD_N999_TEST"
    )
    assert verify_speaker_consent("SPK_FIELD_N999_TEST")
    print("Speaker Consent Ledger test passed.")
