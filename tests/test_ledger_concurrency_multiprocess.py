"""
Unit and Multiprocess Integration Tests for OS-Level Advisory Ledger Locking.
Complies with Phase 9A.6.2 Workstream B:
  - Multi-process concurrency (registration, custom ID race, metadata update, revocation).
  - Process crash recovery (SIGKILL / TerminateProcess release, uncorrupted ledger).
  - Nested same-thread re-entrant transactions.
  - Bounded contention timeout without lock theft.
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
import pytest

from src.acquisition.speaker_consent import (
    ledger_lock,
    register_speaker,
    update_speaker_consent,
    revoke_speaker_consent,
    load_consent_ledger,
    save_consent_ledger,
    _acquire_kernel_lock,
    _release_kernel_lock,
    CANONICAL_SPEAKER_ID_REGEX,
)


def test_multiprocess_concurrent_registration_no_lost_records(tmp_path):
    """
    Verifies that multiple independent OS processes can register distinct speakers
    concurrently without losing records or corrupting JSON serialization.
    """
    ledger_path = tmp_path / "concurrent_reg_ledger.json"
    with open(ledger_path, "w", encoding="utf-8") as f:
        json.dump({}, f)

    worker_code = """
import sys, time
from pathlib import Path
from src.acquisition.speaker_consent import register_speaker

ledger_path = Path(sys.argv[1])
worker_id = sys.argv[2]
count = int(sys.argv[3])

for i in range(count):
    register_speaker(
        dialect="Northern" if i % 2 == 0 else "Southern",
        gender="FEMALE",
        consent_given=True,
        notes=f"worker_{worker_id}_rec_{i}",
        ledger_path=ledger_path
    )
    time.sleep(0.01)
"""
    num_workers = 3
    recs_per_worker = 5
    total_expected = num_workers * recs_per_worker

    procs = [
        subprocess.Popen(
            [sys.executable, "-c", worker_code, str(ledger_path), str(w), str(recs_per_worker)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        for w in range(num_workers)
    ]

    for p in procs:
        stdout, stderr = p.communicate(timeout=30)
        assert p.returncode == 0, f"Worker failed: {stderr}"

    # Verify ledger integrity
    with open(ledger_path, "r", encoding="utf-8") as f:
        ledger = json.load(f)

    assert len(ledger) == total_expected, f"Expected {total_expected} records, got {len(ledger)}"
    for spk_id, rec in ledger.items():
        assert CANONICAL_SPEAKER_ID_REGEX.match(spk_id) is not None
        assert rec["consent_given"] is True


def test_multiprocess_concurrent_custom_id_collision_prevention(tmp_path):
    """
    Verifies that two independent processes racing to register the exact same custom ID
    are serialized: exactly one succeeds and one is cleanly rejected with ValueError.
    """
    ledger_path = tmp_path / "collision_ledger.json"
    with open(ledger_path, "w", encoding="utf-8") as f:
        json.dump({}, f)

    custom_id = "SPK_RACE_ID_UNIQUE_001"

    worker_code = """
import sys
from pathlib import Path
from src.acquisition.speaker_consent import register_speaker

ledger_path = Path(sys.argv[1])
custom_id = sys.argv[2]

try:
    register_speaker(
        dialect="Central",
        gender="MALE",
        consent_given=True,
        custom_speaker_id=custom_id,
        ledger_path=ledger_path
    )
    sys.exit(0)  # SUCCESS
except ValueError as e:
    if "already exists" in str(e):
        sys.exit(42)  # COLLISION_REJECTED
    sys.exit(1)
"""
    p1 = subprocess.Popen([sys.executable, "-c", worker_code, str(ledger_path), custom_id])
    p2 = subprocess.Popen([sys.executable, "-c", worker_code, str(ledger_path), custom_id])

    ret1 = p1.wait(timeout=15)
    ret2 = p2.wait(timeout=15)

    outcomes = {ret1, ret2}
    assert outcomes == {0, 42}, f"Unexpected outcomes: ret1={ret1}, ret2={ret2}"

    with open(ledger_path, "r", encoding="utf-8") as f:
        ledger = json.load(f)
    assert len(ledger) == 1
    assert custom_id in ledger


def test_multiprocess_concurrent_registration_and_consent_update(tmp_path):
    """
    Verifies concurrent execution between speaker registration and metadata update.
    """
    ledger_path = tmp_path / "update_race_ledger.json"
    initial_rec = register_speaker(
        dialect="Northern",
        gender="MALE",
        consent_given=True,
        ledger_path=ledger_path,
        notes="initial"
    )
    spk_id = initial_rec["speaker_id"]

    code_updater = """
import sys, time
from pathlib import Path
from src.acquisition.speaker_consent import update_speaker_consent

ledger_path = Path(sys.argv[1])
spk_id = sys.argv[2]
update_speaker_consent(spk_id, {"notes": "updated_by_proc", "primary_device": "iPhone14"}, ledger_path=ledger_path)
"""
    code_registrar = """
import sys, time
from pathlib import Path
from src.acquisition.speaker_consent import register_speaker

ledger_path = Path(sys.argv[1])
register_speaker(dialect="Southern", gender="FEMALE", consent_given=True, notes="new_reg", ledger_path=ledger_path)
"""
    p_update = subprocess.Popen([sys.executable, "-c", code_updater, str(ledger_path), spk_id])
    p_reg = subprocess.Popen([sys.executable, "-c", code_registrar, str(ledger_path)])

    assert p_update.wait(timeout=15) == 0
    assert p_reg.wait(timeout=15) == 0

    with open(ledger_path, "r", encoding="utf-8") as f:
        ledger = json.load(f)

    assert len(ledger) == 2
    assert ledger[spk_id]["notes"] == "updated_by_proc"
    assert ledger[spk_id]["primary_device"] == "iPhone14"


def test_multiprocess_concurrent_registration_and_revocation(tmp_path):
    """
    Verifies concurrent execution between consent revocation and new speaker registration.
    """
    ledger_path = tmp_path / "revocation_race_ledger.json"
    spk_to_revoke = register_speaker(
        dialect="Central",
        gender="FEMALE",
        consent_given=True,
        ledger_path=ledger_path
    )["speaker_id"]

    code_revoker = """
import sys
from pathlib import Path
from src.acquisition.speaker_consent import revoke_speaker_consent

ledger_path = Path(sys.argv[1])
spk_id = sys.argv[2]
revoke_speaker_consent(spk_id, ledger_path=ledger_path)
"""
    code_registrar = """
import sys
from pathlib import Path
from src.acquisition.speaker_consent import register_speaker

ledger_path = Path(sys.argv[1])
register_speaker(dialect="Northern", gender="MALE", consent_given=True, notes="concurrent_reg", ledger_path=ledger_path)
"""
    p_rev = subprocess.Popen([sys.executable, "-c", code_revoker, str(ledger_path), spk_to_revoke])
    p_reg = subprocess.Popen([sys.executable, "-c", code_registrar, str(ledger_path)])

    assert p_rev.wait(timeout=15) == 0
    assert p_reg.wait(timeout=15) == 0

    with open(ledger_path, "r", encoding="utf-8") as f:
        ledger = json.load(f)

    assert len(ledger) == 2
    assert ledger[spk_to_revoke]["consent_given"] is False
    assert "revoked_at" in ledger[spk_to_revoke]


def test_multiprocess_crash_recovery_killed_lock_holder(tmp_path):
    """
    Verifies that if a process holding the OS kernel lock is killed abruptly (SIGKILL/TerminateProcess),
    the OS automatically frees the lock, and a subsequent process acquires it immediately
    without corrupting the ledger or requiring file deletion.
    """
    ledger_path = tmp_path / "crash_test_ledger.json"
    with open(ledger_path, "w", encoding="utf-8") as f:
        json.dump({"SEED": {"speaker_id": "SEED", "consent_given": True}}, f)

    # Spawn process that acquires lock and sleeps
    code_holder = """
import sys, time
from pathlib import Path
from src.acquisition.speaker_consent import ledger_lock

ledger_path = Path(sys.argv[1])
with ledger_lock(ledger_path, timeout=5.0):
    sys.stdout.write("LOCKED\\n")
    sys.stdout.flush()
    time.sleep(30)
"""
    p_holder = subprocess.Popen(
        [sys.executable, "-c", code_holder, str(ledger_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    line = p_holder.stdout.readline()
    assert "LOCKED" in line, f"Process failed to lock: {line}"

    # Kill the holder process abruptly while it holds the lock
    p_holder.kill()
    p_holder.wait()

    # Subsequent process registers new speaker; must succeed immediately without error
    new_rec = register_speaker(
        dialect="Southern",
        gender="MALE",
        consent_given=True,
        ledger_path=ledger_path,
        notes="post_crash_registration"
    )
    assert CANONICAL_SPEAKER_ID_REGEX.match(new_rec["speaker_id"]) is not None

    with open(ledger_path, "r", encoding="utf-8") as f:
        ledger = json.load(f)
    assert len(ledger) == 2
    assert "SEED" in ledger
    assert new_rec["speaker_id"] in ledger


def test_nested_lock_same_thread_reentrant(tmp_path):
    """
    Verifies that calling ledger_lock in a nested manner on the same thread
    acquires re-entrantly without self-deadlock, and only unbinds upon outer exit.
    """
    ledger_path = tmp_path / "nested_test_ledger.json"
    lock_path = ledger_path.with_suffix(".lock")

    with ledger_lock(ledger_path, timeout=0.5):
        with ledger_lock(ledger_path, timeout=0.5):
            with ledger_lock(ledger_path, timeout=0.5):
                rec = register_speaker(
                    dialect="Northern",
                    gender="FEMALE",
                    consent_given=True,
                    ledger_path=ledger_path,
                    notes="nested_3_levels"
                )
                assert CANONICAL_SPEAKER_ID_REGEX.match(rec["speaker_id"]) is not None

    assert lock_path.exists(), "Stable lock file must persist on disk"

    # Verify subsequent acquisition succeeds
    with ledger_lock(ledger_path, timeout=0.5):
        ledger = load_consent_ledger(ledger_path)
        assert len(ledger) == 1


def test_nested_lock_exception_cleanup(tmp_path):
    """
    Verifies that an exception inside a nested transaction unwinds cleanly
    and resets the thread-local depth so subsequent locks work normally.
    """
    ledger_path = tmp_path / "exception_test_ledger.json"

    with pytest.raises(ZeroDivisionError):
        with ledger_lock(ledger_path, timeout=0.5):
            with ledger_lock(ledger_path, timeout=0.5):
                _ = 1 / 0

    # Ensure thread-local depth was decremented and file can be locked again
    with ledger_lock(ledger_path, timeout=0.5):
        register_speaker(
            dialect="Central",
            gender="MALE",
            consent_given=True,
            ledger_path=ledger_path
        )


def test_timeout_contention_no_forced_theft(tmp_path):
    """
    Verifies that a live process holding the lock causes a bounded TimeoutError
    for contending callers without forcibly stealing ownership.
    """
    ledger_path = tmp_path / "timeout_test_ledger.json"

    code_holder = """
import sys, time
from pathlib import Path
from src.acquisition.speaker_consent import ledger_lock

ledger_path = Path(sys.argv[1])
with ledger_lock(ledger_path, timeout=5.0):
    sys.stdout.write("ACQUIRED\\n")
    sys.stdout.flush()
    time.sleep(1.0)
"""
    p_holder = subprocess.Popen(
        [sys.executable, "-c", code_holder, str(ledger_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    line = p_holder.stdout.readline()
    assert "ACQUIRED" in line

    # Contender with small timeout should time out
    start = time.time()
    with pytest.raises(TimeoutError, match="lock contention timeout"):
        with ledger_lock(ledger_path, timeout=0.2, poll_interval=0.02):
            pass
    elapsed = time.time() - start
    assert 0.15 <= elapsed <= 0.60, f"Timeout was not bounded: {elapsed:.2f}s"

    # Wait for holder to finish naturally
    p_holder.wait(timeout=5)
    assert p_holder.returncode == 0


def test_withdrawal_crash_injection_and_idempotent_recovery(tmp_path):
    """
    Phase 9A.6.3 Section 2: Withdrawal Crash Consistency and Idempotent Recovery.
    Verifies that:
      1. Consent revocation journal sets QUARANTINE_PENDING and consent_given=False.
      2. If process crashes at boundary before filesystem mutation, verify_speaker_consent
         immediately fails closed (returns False), preventing pipeline ingestion.
      3. Disk files remain untouched in original locations during crash state.
      4. recover_pending_revocations detects QUARANTINE_PENDING and completes the workflow
         (quarantines files, updates manifest, sets QUARANTINE_COMPLETED).
      5. Subsequent recovery run is completely idempotent (0 pending).
    """
    import pandas as pd
    from src.acquisition.speaker_consent import verify_speaker_consent
    from src.acquisition.revoke_speaker_consent import (
        execute_consent_revocation,
        recover_pending_revocations
    )

    sandbox_ledger = tmp_path / "consent_ledger.json"
    sandbox_manifest = tmp_path / "human_manifest.csv"
    sandbox_raw = tmp_path / "audio_raw"
    sandbox_std = tmp_path / "audio_16k_standardized"
    sandbox_quarantine = tmp_path / "quarantine"

    # 1. Register speaker
    spk_record = register_speaker(
        dialect="Northern",
        gender="FEMALE",
        consent_given=True,
        ledger_path=sandbox_ledger
    )
    spk_id = spk_record["speaker_id"]
    assert verify_speaker_consent(spk_id, ledger_path=sandbox_ledger) is True

    # 2. Create mock raw and standardized audio files and manifest
    spk_raw_dir = sandbox_raw / spk_id
    spk_raw_dir.mkdir(parents=True, exist_ok=True)
    raw_file = spk_raw_dir / "take_001.wav"
    raw_file.write_bytes(b"RIFF" + b"\x00" * 40)

    sandbox_std.mkdir(parents=True, exist_ok=True)
    std_file = sandbox_std / f"{spk_id}_take_001.wav"
    std_file.write_bytes(b"RIFF" + b"\x00" * 40)

    df_manifest = pd.DataFrame([{
        "recording_id": "REC_TEST_CRASH_001",
        "speaker_id": spk_id,
        "raw_filepath": str(raw_file),
        "processed_path": str(std_file),
        "status": "APPROVED",
        "exclusion_reason": ""
    }])
    sandbox_manifest.parent.mkdir(parents=True, exist_ok=True)
    df_manifest.to_csv(sandbox_manifest, index=False, encoding="utf-8")

    # 3. Inject crash at boundary between Phase 1 (ledger journal) and Phase 2 (filesystem)
    with pytest.raises(RuntimeError, match="CRASH_INJECTED_AT_BOUNDARY"):
        execute_consent_revocation(
            speaker_id=spk_id,
            reason="participant_withdrawal_with_crash",
            confirm=True,
            dry_run=False,
            manifest_path=sandbox_manifest,
            ledger_path=sandbox_ledger,
            quarantine_dir=sandbox_quarantine,
            raw_base_dir=sandbox_raw,
            std_base_dir=sandbox_std,
            _inject_crash_before_filesystem=True
        )

    # 4. Verify crash consistency:
    # Ledger must record QUARANTINE_PENDING and consent_given=False
    ledger_crashed = load_consent_ledger(sandbox_ledger)
    assert ledger_crashed[spk_id]["consent_given"] is False
    assert ledger_crashed[spk_id]["revocation_status"] == "QUARANTINE_PENDING"

    # Pipeline validation must fail closed immediately (cannot ingest unconsented audio)
    assert verify_speaker_consent(spk_id, ledger_path=sandbox_ledger) is False

    # Disk files must still be in source locations (not lost or half-quarantined)
    assert raw_file.exists()
    assert std_file.exists()

    # 5. Execute recovery mechanism
    rec_result = recover_pending_revocations(
        manifest_path=sandbox_manifest,
        ledger_path=sandbox_ledger,
        quarantine_dir=sandbox_quarantine,
        raw_base_dir=sandbox_raw,
        std_base_dir=sandbox_std
    )
    assert rec_result["status"] == "RECOVERY_COMPLETE"
    assert rec_result["recovered_speakers_count"] == 1
    assert spk_id in rec_result["recovered_speakers"]

    # 6. Verify state after recovery
    # Audio files relocated to quarantine
    assert not raw_file.exists()
    assert not std_file.exists()
    quar_spk_dir = sandbox_quarantine / "withdrawn_speakers" / spk_id
    assert quar_spk_dir.exists()
    assert len(list(quar_spk_dir.rglob("*.wav"))) == 2

    # Manifest rows marked REVOKED_WITHDRAWN
    df_man_after = pd.read_csv(sandbox_manifest)
    assert df_man_after.iloc[0]["status"] == "REVOKED_WITHDRAWN"
    assert "consent_revoked" in df_man_after.iloc[0]["exclusion_reason"]

    # Ledger marked QUARANTINE_COMPLETED
    ledger_recovered = load_consent_ledger(sandbox_ledger)
    assert ledger_recovered[spk_id]["consent_given"] is False
    assert ledger_recovered[spk_id]["revocation_status"] == "QUARANTINE_COMPLETED"
    assert "revoked_at" in ledger_recovered[spk_id]

    # Speaker consent remains False
    assert verify_speaker_consent(spk_id, ledger_path=sandbox_ledger) is False

    # 7. Idempotent second recovery: 0 pending
    rec_second = recover_pending_revocations(
        manifest_path=sandbox_manifest,
        ledger_path=sandbox_ledger,
        quarantine_dir=sandbox_quarantine,
        raw_base_dir=sandbox_raw,
        std_base_dir=sandbox_std
    )
    assert rec_second["status"] == "RECOVERY_COMPLETE"
    assert rec_second["recovered_speakers_count"] == 0
    assert rec_second["recovered_speakers"] == []


def test_withdrawal_content_integrity_same_size_different_hash(tmp_path):
    """
    Phase 9A.6.3-FINAL Acceptance Criteria:
    Verifies that file identity is NOT determined by file size alone.
    If a file exists in the destination quarantine folder with the SAME size but DIFFERENT SHA-256 hash,
    the source file must NOT be silently unlinked/deleted. Instead, it must be safely quarantined
    with a distinct collision-avoidance nonce so both artifacts are preserved.
    """
    import pandas as pd
    from src.acquisition.speaker_consent import register_speaker
    from src.acquisition.revoke_speaker_consent import execute_consent_revocation
    from src.acquisition.ingest_field_recordings import compute_sha256

    sandbox_ledger = tmp_path / "consent_ledger.json"
    sandbox_manifest = tmp_path / "human_manifest.csv"
    sandbox_raw = tmp_path / "audio_raw"
    sandbox_std = tmp_path / "audio_16k_standardized"
    sandbox_quarantine = tmp_path / "quarantine"

    spk_record = register_speaker(dialect="Central", gender="MALE", consent_given=True, ledger_path=sandbox_ledger)
    spk_id = spk_record["speaker_id"]

    # Source file: exactly 1024 bytes of character 'A'
    spk_raw_dir = sandbox_raw / spk_id
    spk_raw_dir.mkdir(parents=True, exist_ok=True)
    raw_file = spk_raw_dir / "collision_take.wav"
    raw_content = b"A" * 1024
    raw_file.write_bytes(raw_content)

    # Pre-existing quarantine destination file: exactly 1024 bytes of character 'B' (same size, different sha256!)
    dest_quar_dir = sandbox_quarantine / "withdrawn_speakers" / spk_id / "audio_raw"
    dest_quar_dir.mkdir(parents=True, exist_ok=True)
    existing_dest = dest_quar_dir / "collision_take.wav"
    dest_content = b"B" * 1024
    existing_dest.write_bytes(dest_content)

    assert raw_file.stat().st_size == existing_dest.stat().st_size == 1024
    assert compute_sha256(raw_file) != compute_sha256(existing_dest)

    df_manifest = pd.DataFrame([{
        "recording_id": "REC_COLLISION_SAME_SIZE",
        "speaker_id": spk_id,
        "raw_filepath": str(raw_file),
        "processed_path": "",
        "status": "APPROVED",
        "exclusion_reason": ""
    }])
    df_manifest.to_csv(sandbox_manifest, index=False, encoding="utf-8")

    # Execute revocation
    res = execute_consent_revocation(
        speaker_id=spk_id,
        confirm=True,
        manifest_path=sandbox_manifest,
        ledger_path=sandbox_ledger,
        quarantine_dir=sandbox_quarantine,
        raw_base_dir=sandbox_raw,
        std_base_dir=sandbox_std
    )
    assert res["status"] == "REVOKED_CONFIRMED"

    # Both files must be preserved! Original pre-existing dest file must NOT be overwritten!
    assert existing_dest.exists()
    assert existing_dest.read_bytes() == dest_content

    # Source file must be moved to a nonced file in the destination dir
    quar_files = list(dest_quar_dir.glob("collision_take*.wav"))
    assert len(quar_files) == 2, f"Expected 2 preserved files, found {quar_files}"
    nonced_files = [f for f in quar_files if f != existing_dest]
    assert len(nonced_files) == 1
    assert nonced_files[0].read_bytes() == raw_content


def test_withdrawal_crash_recovery_all_five_boundaries(tmp_path):
    """
    Phase 9A.6.3-FINAL Section 2: Exhaustive failure-boundary test matrix.
    Tests recovery at all 5 specified boundaries:
      Boundary 1: After QUARANTINE_PENDING journal persistence.
      Boundary 2: After the first file moves in a multi-file withdrawal.
      Boundary 3: After file copy but before source deletion.
      Boundary 4: After all file moves but before QUARANTINE_COMPLETED persistence.
      Boundary 5: During repeated recovery invocation (idempotency).
    """
    import pandas as pd
    from src.acquisition.speaker_consent import register_speaker, verify_speaker_consent
    from src.acquisition.revoke_speaker_consent import execute_consent_revocation, recover_pending_revocations

    # --- Boundary 1: Crash after QUARANTINE_PENDING journal ---
    sb1 = tmp_path / "b1"
    spk1 = register_speaker(dialect="Northern", gender="FEMALE", consent_given=True, ledger_path=sb1 / "ledger.json")["speaker_id"]
    f1 = sb1 / "raw" / spk1 / "f1.wav"
    f1.parent.mkdir(parents=True, exist_ok=True)
    f1.write_bytes(b"RIFF" + b"1" * 100)
    man1 = pd.DataFrame([{"recording_id": "R1", "speaker_id": spk1, "raw_filepath": str(f1), "processed_path": "", "status": "APPROVED", "exclusion_reason": ""}])
    man1.to_csv(sb1 / "man.csv", index=False)

    with pytest.raises(RuntimeError, match="CRASH_INJECTED_AT_BOUNDARY"):
        execute_consent_revocation(spk1, confirm=True, manifest_path=sb1 / "man.csv", ledger_path=sb1 / "ledger.json", quarantine_dir=sb1 / "quar", raw_base_dir=sb1 / "raw", std_base_dir=sb1 / "std", _crash_boundary="after_journal_pending")
    assert verify_speaker_consent(spk1, ledger_path=sb1 / "ledger.json") is False
    assert f1.exists()

    rec1 = recover_pending_revocations(manifest_path=sb1 / "man.csv", ledger_path=sb1 / "ledger.json", quarantine_dir=sb1 / "quar", raw_base_dir=sb1 / "raw", std_base_dir=sb1 / "std")
    assert rec1["recovered_speakers_count"] == 1
    assert not f1.exists()
    assert (sb1 / "quar" / "withdrawn_speakers" / spk1 / "audio_raw" / "f1.wav").exists()

    # --- Boundary 2: Crash after first file move in multi-file withdrawal ---
    sb2 = tmp_path / "b2"
    spk2 = register_speaker(dialect="Southern", gender="MALE", consent_given=True, ledger_path=sb2 / "ledger.json")["speaker_id"]
    f2_a = sb2 / "raw" / spk2 / "f2_a.wav"
    f2_b = sb2 / "raw" / spk2 / "f2_b.wav"
    f2_a.parent.mkdir(parents=True, exist_ok=True)
    f2_a.write_bytes(b"RIFF" + b"2A" * 50)
    f2_b.write_bytes(b"RIFF" + b"2B" * 50)
    man2 = pd.DataFrame([
        {"recording_id": "R2A", "speaker_id": spk2, "raw_filepath": str(f2_a), "processed_path": "", "status": "APPROVED", "exclusion_reason": ""},
        {"recording_id": "R2B", "speaker_id": spk2, "raw_filepath": str(f2_b), "processed_path": "", "status": "APPROVED", "exclusion_reason": ""}
    ])
    man2.to_csv(sb2 / "man.csv", index=False)

    with pytest.raises(RuntimeError, match="CRASH_INJECTED_AT_BOUNDARY"):
        execute_consent_revocation(spk2, confirm=True, manifest_path=sb2 / "man.csv", ledger_path=sb2 / "ledger.json", quarantine_dir=sb2 / "quar", raw_base_dir=sb2 / "raw", std_base_dir=sb2 / "std", _crash_boundary="after_first_file_move")
    assert verify_speaker_consent(spk2, ledger_path=sb2 / "ledger.json") is False
    # Exactly one file moved, one still in source
    assert (not f2_a.exists()) or (not f2_b.exists())
    assert f2_a.exists() or f2_b.exists()

    rec2 = recover_pending_revocations(manifest_path=sb2 / "man.csv", ledger_path=sb2 / "ledger.json", quarantine_dir=sb2 / "quar", raw_base_dir=sb2 / "raw", std_base_dir=sb2 / "std")
    assert rec2["recovered_speakers_count"] == 1
    assert not f2_a.exists()
    assert not f2_b.exists()
    assert len(list((sb2 / "quar" / "withdrawn_speakers" / spk2 / "audio_raw").glob("*.wav"))) == 2

    # --- Boundary 3: Crash after file copy but before source deletion ---
    sb3 = tmp_path / "b3"
    spk3 = register_speaker(dialect="Central", gender="FEMALE", consent_given=True, ledger_path=sb3 / "ledger.json")["speaker_id"]
    f3 = sb3 / "raw" / spk3 / "f3.wav"
    f3.parent.mkdir(parents=True, exist_ok=True)
    f3.write_bytes(b"RIFF" + b"3" * 100)
    man3 = pd.DataFrame([{"recording_id": "R3", "speaker_id": spk3, "raw_filepath": str(f3), "processed_path": "", "status": "APPROVED", "exclusion_reason": ""}])
    man3.to_csv(sb3 / "man.csv", index=False)

    with pytest.raises(RuntimeError, match="CRASH_INJECTED_AT_BOUNDARY"):
        execute_consent_revocation(spk3, confirm=True, manifest_path=sb3 / "man.csv", ledger_path=sb3 / "ledger.json", quarantine_dir=sb3 / "quar", raw_base_dir=sb3 / "raw", std_base_dir=sb3 / "std", _crash_boundary="after_copy_before_source_delete")
    assert verify_speaker_consent(spk3, ledger_path=sb3 / "ledger.json") is False
    # Both source and destination exist with identical content
    dest3 = sb3 / "quar" / "withdrawn_speakers" / spk3 / "audio_raw" / "f3.wav"
    assert f3.exists()
    assert dest3.exists()

    rec3 = recover_pending_revocations(manifest_path=sb3 / "man.csv", ledger_path=sb3 / "ledger.json", quarantine_dir=sb3 / "quar", raw_base_dir=sb3 / "raw", std_base_dir=sb3 / "std")
    assert rec3["recovered_speakers_count"] == 1
    # Residual source unlinked, destination preserved without duplicate nonce file
    assert not f3.exists()
    assert dest3.exists()
    assert len(list((sb3 / "quar" / "withdrawn_speakers" / spk3 / "audio_raw").glob("*.wav"))) == 1

    # --- Boundary 4: Crash after all file moves but before QUARANTINE_COMPLETED ---
    sb4 = tmp_path / "b4"
    spk4 = register_speaker(dialect="Northern", gender="MALE", consent_given=True, ledger_path=sb4 / "ledger.json")["speaker_id"]
    f4 = sb4 / "raw" / spk4 / "f4.wav"
    f4.parent.mkdir(parents=True, exist_ok=True)
    f4.write_bytes(b"RIFF" + b"4" * 100)
    man4 = pd.DataFrame([{"recording_id": "R4", "speaker_id": spk4, "raw_filepath": str(f4), "processed_path": "", "status": "APPROVED", "exclusion_reason": ""}])
    man4.to_csv(sb4 / "man.csv", index=False)

    with pytest.raises(RuntimeError, match="CRASH_INJECTED_AT_BOUNDARY"):
        execute_consent_revocation(spk4, confirm=True, manifest_path=sb4 / "man.csv", ledger_path=sb4 / "ledger.json", quarantine_dir=sb4 / "quar", raw_base_dir=sb4 / "raw", std_base_dir=sb4 / "std", _crash_boundary="after_files_before_completed")
    assert verify_speaker_consent(spk4, ledger_path=sb4 / "ledger.json") is False
    # File already moved, but ledger still QUARANTINE_PENDING
    assert not f4.exists()
    assert (sb4 / "quar" / "withdrawn_speakers" / spk4 / "audio_raw" / "f4.wav").exists()

    rec4 = recover_pending_revocations(manifest_path=sb4 / "man.csv", ledger_path=sb4 / "ledger.json", quarantine_dir=sb4 / "quar", raw_base_dir=sb4 / "raw", std_base_dir=sb4 / "std")
    assert rec4["recovered_speakers_count"] == 1
    # Finalized ledger
    assert not verify_speaker_consent(spk4, ledger_path=sb4 / "ledger.json")

    # --- Boundary 5: Repeated recovery invocation (idempotency) ---
    rec5 = recover_pending_revocations(manifest_path=sb4 / "man.csv", ledger_path=sb4 / "ledger.json", quarantine_dir=sb4 / "quar", raw_base_dir=sb4 / "raw", std_base_dir=sb4 / "std")
    assert rec5["status"] == "RECOVERY_COMPLETE"
    assert rec5["recovered_speakers_count"] == 0
    assert rec5["recovered_speakers"] == []
