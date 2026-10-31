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
import os
import time
import socket
import threading
import contextlib
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

# Canonical Phase 9A.3 128-bit opaque pseudonymous speaker identifier regex
CANONICAL_SPEAKER_ID_REGEX = re.compile(r"^SPK_[a-f0-9]{32}$")
# Historical Phase 9A P0-3 remediation identifier regex (preserved for backward compatibility)
LEGACY_SPEAKER_ID_REGEX = re.compile(r"^SPK_H_([NCS])_([a-f0-9]{6})$")
# Master validation regex accepting canonical 128-bit opaque IDs and legacy P0-3 IDs
SPEAKER_ID_REGEX = re.compile(r"^(?:SPK_[a-f0-9]{32}|SPK_H_([NCS])_([a-f0-9]{6}))$")


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


_THREAD_LOCAL = threading.local()


def is_process_alive(pid: int) -> bool:
    """Checks whether a process with the given PID is currently running (diagnostic utility)."""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        import ctypes.wintypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        SYNCHRONIZE = 0x00100000
        h_process = ctypes.windll.kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE,
            False,
            pid
        )
        if not h_process:
            error = ctypes.windll.kernel32.GetLastError()
            ERROR_INVALID_PARAMETER = 87
            if error == ERROR_INVALID_PARAMETER:
                return False
            # Access denied indicates system/protected process, which is alive
            return True
        try:
            exit_code = ctypes.wintypes.DWORD()
            if ctypes.windll.kernel32.GetExitCodeProcess(h_process, ctypes.byref(exit_code)):
                STILL_ACTIVE = 259
                return exit_code.value == STILL_ACTIVE
            return True
        finally:
            ctypes.windll.kernel32.CloseHandle(h_process)
    else:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True


def _acquire_kernel_lock(fd: int) -> bool:
    """
    Attempts to acquire an exclusive, non-blocking kernel-level advisory lock on fd.
    On Windows: uses msvcrt.locking(fd, msvcrt.LK_NBLCK, 1).
    On POSIX: uses fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB).
    Returns True if acquired, False if contested.
    """
    if os.name == "nt":
        import msvcrt
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return True
        except (BlockingIOError, PermissionError):
            return False
        except OSError as e:
            if getattr(e, "winerror", None) in (33, 36) or e.errno in (13, 36):
                return False
            raise
    else:
        import fcntl
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except (BlockingIOError, PermissionError):
            return False
        except OSError as e:
            import errno
            if e.errno in (errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK):
                return False
            raise


def _release_kernel_lock(fd: int) -> None:
    """
    Releases the kernel-level advisory lock on fd.
    On Windows: uses msvcrt.locking(fd, msvcrt.LK_UNLCK, 1).
    On POSIX: uses fcntl.flock(fd, fcntl.LOCK_UN).
    """
    if os.name == "nt":
        import msvcrt
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
    else:
        import fcntl
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass


@contextlib.contextmanager
def ledger_lock(
    ledger_path: Optional[Path] = None,
    timeout: float = 5.0,
    poll_interval: float = 0.02,
    stale_timeout: float = 60.0
):
    """
    Acquires an exclusive OS-level advisory file lock on a stable lock file for atomic
    read-modify-write ledger transactions. Enforces the single-writer invariant across
    multi-threaded and multi-process callers.

    Architecture & Invariants (Phase 9A.6.2):
      1. Stable Lock File: Uses a persistent .lock file that is NEVER deleted or unlinked
         to acquire or release ownership, completely eliminating TOCTOU deletion races.
      2. Kernel-Level Ownership: Uses OS kernel locking (msvcrt.locking on Windows,
         fcntl.flock on POSIX). Lock ownership is strictly governed by the OS kernel.
      3. Automatic Crash Recovery: When a process holding the lock exits or is terminated
         (e.g., killed via SIGKILL / TerminateProcess), the OS kernel closes open file
         descriptors and automatically releases the lock immediately without manual intervention.
      4. Single Local Machine Scope: This implementation guarantees mutual exclusion on
         local filesystems. Network-mounted filesystems (NFS, SMB/CIFS) are not supported.
      5. Re-entrant (RLock) Semantics: Supports nested calls within the same thread.
      6. Diagnostic Metadata: Lock holder writes PID, host, and timestamp to the lock file
         for observability, but metadata does NOT authority lock ownership.
      7. Bounded Timeout: Raises TimeoutError if lock cannot be acquired within timeout.
    """
    target_path = ledger_path if ledger_path is not None else LEDGER_PATH
    canonical_lock_path = str(target_path.resolve().with_suffix(".lock"))
    lock_path = Path(canonical_lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    # Check thread-local re-entrancy
    if not hasattr(_THREAD_LOCAL, "active_locks"):
        _THREAD_LOCAL.active_locks = {}

    if canonical_lock_path in _THREAD_LOCAL.active_locks:
        _THREAD_LOCAL.active_locks[canonical_lock_path] += 1
        try:
            yield
        finally:
            _THREAD_LOCAL.active_locks[canonical_lock_path] -= 1
            if _THREAD_LOCAL.active_locks[canonical_lock_path] <= 0:
                del _THREAD_LOCAL.active_locks[canonical_lock_path]
        return

    start_time = time.monotonic()
    fd = None
    current_host = socket.gethostname()
    current_pid = os.getpid()

    # Open the stable lock file (creating if it does not exist)
    # Using O_RDWR | O_CREAT opens with shared access so other processes can also open and compete for lock
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o666)

    try:
        acquired = False
        while True:
            if _acquire_kernel_lock(fd):
                acquired = True
                break

            if time.monotonic() - start_time > timeout:
                diag_info = ""
                try:
                    with open(lock_path, "r", encoding="utf-8", errors="ignore") as f_diag:
                        content = f_diag.read().strip()
                        if content:
                            diag_info = f" [Current Holder: {content}]"
                except Exception:
                    pass
                raise TimeoutError(
                    f"Consent ledger lock contention timeout ({timeout}s) on {lock_path}."
                    f"{diag_info} Exclusive kernel lock held by another process."
                )
            time.sleep(poll_interval)

        # Write diagnostic payload for observability (never the authority for ownership)
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            lock_payload = (
                f"pid={current_pid};"
                f"host={current_host};"
                f"acquired={time.time()};"
                f"iso={datetime.now(timezone.utc).isoformat()}\n"
            ).encode("utf-8")
            os.write(fd, lock_payload)
            os.ftruncate(fd, len(lock_payload))
        except Exception:
            pass

        _THREAD_LOCAL.active_locks[canonical_lock_path] = 1
        try:
            yield
        finally:
            _THREAD_LOCAL.active_locks.pop(canonical_lock_path, None)
            # Release kernel lock
            _release_kernel_lock(fd)
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except Exception:
                pass


def save_consent_ledger(ledger: Dict[str, Any], ledger_path: Optional[Path] = None):
    target_path = ledger_path if ledger_path is not None else LEDGER_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(f".tmp_{secrets.token_hex(4)}")
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(ledger, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, target_path)
    except Exception:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass
        raise


def generate_canonical_speaker_id(
    ledger: Optional[Dict[str, Any]] = None,
    max_attempts: int = 1000
) -> str:
    """
    Generates a 128-bit cryptographically secure, fully opaque pseudonymous speaker identifier:
    SPK_<32-hex-characters> using Python's secrets module (Phase 9A.3 Canonical Contract).

    Guarantees:
      1. Zero PII/Demographic Leakage: Does NOT encode dialect, region, date, name, or contact info.
         Dialect is strictly maintained in separately controlled metadata.
      2. Cryptographic Entropy: 128 bits (secrets.token_hex(16)), yielding ~3.4e38 combinations.
         Collision probability across 1,000,000 participants is < 1e-18.
      3. Ledger Uniqueness: Re-samples if candidate exists in active ledger.
      4. Safe Failure: Raises RuntimeError if max_attempts is exceeded.
    """
    active_ledger = ledger if ledger is not None else {}
    for _ in range(max_attempts):
        nonce = secrets.token_hex(16).lower()
        candidate_id = f"SPK_{nonce}"
        if candidate_id not in active_ledger:
            return candidate_id

    raise RuntimeError(
        f"Speaker ID collision avoidance limit ({max_attempts}) reached. "
        "Active ledger may be congested."
    )


def generate_pseudonymous_speaker_id(
    dialect: Optional[str] = None,
    ledger: Optional[Dict[str, Any]] = None,
    max_attempts: int = 1000,
    canonical_opaque: bool = False
) -> str:
    """
    Generates a pseudonymous speaker identifier.
    If canonical_opaque=True or dialect is None, generates canonical 128-bit opaque ID (SPK_<32-hex>).
    Otherwise, preserves historical P0-3 dialect-prefixed ID (SPK_H_<N|C|S>_<6-hex>) for backward compatibility.
    """
    if canonical_opaque or dialect is None:
        return generate_canonical_speaker_id(ledger=ledger, max_attempts=max_attempts)

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
    use_canonical_id: bool = True,
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

    with ledger_lock(target_ledger_path):
        ledger = load_consent_ledger(target_ledger_path)

        if custom_speaker_id:
            if not re.match(r"^[A-Za-z0-9_\-]+$", custom_speaker_id):
                raise ValueError(f"Invalid characters in custom_speaker_id '{custom_speaker_id}'. Only alphanumeric, underscores, and dashes allowed.")
            if custom_speaker_id in ledger:
                raise ValueError(f"Cannot register speaker: ID '{custom_speaker_id}' already exists in consent ledger. Overwrite prevented.")
            spk_id = custom_speaker_id
        elif use_canonical_id:
            spk_id = generate_canonical_speaker_id(ledger)
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
    if rec.get("revoked_at") is not None or rec.get("revocation_status") in ("QUARANTINE_PENDING", "QUARANTINE_COMPLETED", "REVOKED"):
        return False

    # Consent must be strictly boolean True
    consent_val = rec.get("consent_given")
    if consent_val is not True:
        return False

    return True


def revoke_speaker_consent(speaker_id: str, ledger_path: Optional[Path] = None) -> bool:
    """
    Revokes consent for speaker_id. Sets consent_given to False, timestamps the revocation,
    and sets revocation_status to QUARANTINE_COMPLETED.
    Acquires ledger_lock to guarantee multi-threaded and multi-process concurrency safety.
    """
    target_path = ledger_path if ledger_path is not None else LEDGER_PATH
    with ledger_lock(target_path):
        ledger = load_consent_ledger(target_path)
        if speaker_id not in ledger:
            return False
        ledger[speaker_id]["consent_given"] = False
        ledger[speaker_id]["revoked_at"] = datetime.now(timezone.utc).isoformat()
        ledger[speaker_id]["revocation_status"] = "QUARANTINE_COMPLETED"
        save_consent_ledger(ledger, target_path)
        return True


def update_speaker_consent(
    speaker_id: str,
    updates: Dict[str, Any],
    ledger_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Atomically updates specified metadata fields for an existing speaker in the consent ledger.
    Guarantees single-writer consistency via ledger_lock.
    """
    target_path = ledger_path if ledger_path is not None else LEDGER_PATH
    with ledger_lock(target_path):
        ledger = load_consent_ledger(target_path)
        if speaker_id not in ledger:
            raise KeyError(f"Speaker ID '{speaker_id}' not found in ledger at {target_path}.")
        for k, v in updates.items():
            if k == "speaker_id":
                continue  # Immutable primary key
            ledger[speaker_id][k] = v
        ledger[speaker_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
        save_consent_ledger(ledger, target_path)
        return ledger[speaker_id]


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
