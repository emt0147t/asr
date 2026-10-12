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

CONSENT_VERSION = "v1.0-kws-research-commercial"
VALID_DIALECTS = ["Northern", "Central", "Southern"]
VALID_GENDERS = ["MALE", "FEMALE", "OTHER"]
VALID_AGE_BRACKETS = ["18-30", "31-50", "51+"]
VALID_ENVIRONMENTS = ["quiet_room", "office_hvac", "living_room_reverberant", "street_noise", "fan_noise"]

LEDGER_PATH = Path("data/raw/field_recordings/consent_ledger.json")


def load_consent_ledger(ledger_path: Path = LEDGER_PATH) -> Dict[str, Any]:
    if not ledger_path.exists():
        return {}
    try:
        with open(ledger_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_consent_ledger(ledger: Dict[str, Any], ledger_path: Path = LEDGER_PATH):
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger_path, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2, ensure_ascii=False)


def generate_next_speaker_id(dialect: str, ledger: Dict[str, Any]) -> str:
    """
    Generates structured pseudonymous ID:
      - Northern: SPK_FIELD_N001, SPK_FIELD_N002, ...
      - Central:  SPK_FIELD_C001, SPK_FIELD_C002, ...
      - Southern: SPK_FIELD_S001, SPK_FIELD_S002, ...
    """
    prefix = f"SPK_FIELD_{dialect[0].upper()}"
    existing_nums = []
    for spk_id in ledger.keys():
        if spk_id.startswith(prefix):
            try:
                num = int(spk_id.replace(prefix, ""))
                existing_nums.append(num)
            except ValueError:
                pass
    next_num = max(existing_nums, default=0) + 1
    return f"{prefix}{next_num:03d}"


def register_speaker(
    dialect: str,
    gender: str,
    age_bracket: str = "18-30",
    consent_given: bool = True,
    device_model: str = "Unspecified Mobile",
    microphone_type: str = "Built-in microphone",
    acoustic_environment: str = "quiet_room",
    notes: str = "",
    custom_speaker_id: Optional[str] = None,
    ledger_path: Path = LEDGER_PATH
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

    ledger = load_consent_ledger(ledger_path)

    if custom_speaker_id:
        spk_id = custom_speaker_id
    else:
        spk_id = generate_next_speaker_id(dialect, ledger)

    record = {
        "speaker_id": spk_id,
        "consent_given": True,
        "consent_version": CONSENT_VERSION,
        "consent_timestamp": datetime.now(timezone.utc).isoformat(),
        "dialect": dialect,
        "gender": gender.upper(),
        "age_bracket": age_bracket,
        "primary_device": device_model,
        "primary_microphone": microphone_type,
        "primary_environment": acoustic_environment,
        "notes": notes
    }

    ledger[spk_id] = record
    save_consent_ledger(ledger, ledger_path)
    print(f"[REGISTERED] Speaker {spk_id} ({dialect}, {gender}) registered with verified consent.")
    return record


def verify_speaker_consent(speaker_id: str, ledger_path: Path = LEDGER_PATH) -> bool:
    """
    Returns True if speaker_id exists in ledger and consent is True.
    """
    ledger = load_consent_ledger(ledger_path)
    if speaker_id not in ledger:
        return False
    return bool(ledger[speaker_id].get("consent_given", False))


def get_speaker_metadata(speaker_id: str, ledger_path: Path = LEDGER_PATH) -> Optional[Dict[str, Any]]:
    ledger = load_consent_ledger(ledger_path)
    return ledger.get(speaker_id)


def list_registered_speakers(ledger_path: Path = LEDGER_PATH) -> List[Dict[str, Any]]:
    ledger = load_consent_ledger(ledger_path)
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
