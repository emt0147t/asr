"""
Interactive Human Field Recording Workflow for Vietnamese KWS.
Provides a simple recording interface with live microphone capture, playback, and re-recording.
Built using Windows MCI waveaudio (ctypes.windll.winmm) with zero external C dependencies.
Supports mock/dry-run mode for automated verification without hardware microphone.
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

import os
import time
import ctypes
import argparse
import tempfile
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import soundfile as sf

from src.acquisition.speaker_consent import (
    register_speaker, verify_speaker_consent, get_speaker_metadata,
    list_registered_speakers, VALID_DIALECTS, VALID_MICROPHONE_TYPES,
    VALID_DISTANCE_CATEGORIES, LEDGER_PATH
)
from src.acquisition.ingest_field_recordings import ingest_recording, VALID_COMMANDS

# Windows Multimedia API
winmm = ctypes.windll.winmm if os.name == 'nt' else None


class AudioRecorder:
    """
    Lightweight audio recorder and player using Windows MCI (winmm.dll).
    Records at 44.1 kHz 16-bit stereo (universally supported across Windows sound drivers),
    which is subsequently downmixed and resampled to 16 kHz mono during ingestion.
    """
    def __init__(self, sample_rate: int = 44100, channels: int = 2, bits_per_sample: int = 16):
        self.sample_rate = sample_rate
        self.channels = channels
        self.bits_per_sample = bits_per_sample
        self.alias = "field_recorder_alias"

    def _mci_cmd(self, cmd: str) -> Tuple[int, str]:
        buf = ctypes.create_string_buffer(256)
        err = winmm.mciSendStringA(cmd.encode('ascii'), buf, 256, 0)
        err_msg = ""
        if err != 0:
            err_buf = ctypes.create_string_buffer(256)
            winmm.mciGetErrorStringA(err, err_buf, 256)
            err_msg = err_buf.value.decode('latin1', errors='ignore')
        return err, err_msg

    def record_to_file(self, output_wav: Path, duration_sec: float = 1.6) -> bool:
        if not winmm:
            raise RuntimeError("AudioRecorder requires Windows OS (winmm.dll).")

        wav_path_str = str(output_wav.resolve())
        
        # Stop and close any previous instance
        self._mci_cmd(f"close {self.alias}")

        # Open waveaudio device
        err, msg = self._mci_cmd(f"open new type waveaudio alias {self.alias}")
        if err != 0:
            print(f"[ERROR] Failed to open waveaudio device: code {err} ({msg})")
            return False

        try:
            # Configure 44.1 kHz 16-bit stereo for universal Windows hardware support
            self._mci_cmd(f"set {self.alias} time format ms")
            self._mci_cmd(f"set {self.alias} channels {self.channels}")
            self._mci_cmd(f"set {self.alias} samplespersec {self.sample_rate}")
            self._mci_cmd(f"set {self.alias} bitspersample {self.bits_per_sample}")

            # Start recording
            err, msg = self._mci_cmd(f"record {self.alias}")
            if err != 0:
                print(f"[ERROR] MCI Record failed: code {err} ({msg})")
                return False

            time.sleep(duration_sec)

            # Stop and save
            self._mci_cmd(f"stop {self.alias}")
            err, msg = self._mci_cmd(f'save {self.alias} "{wav_path_str}"')
            if err != 0:
                print(f"[ERROR] MCI Save failed: code {err} ({msg})")
                return False
        finally:
            self._mci_cmd(f"close {self.alias}")

        return output_wav.exists() and output_wav.stat().st_size > 44

    def play_file(self, wav_path: Path):
        if not winmm:
            print("[PLAYBACK] Playback unavailable on non-Windows OS.")
            return
        # SND_FILENAME = 0x00020000, SND_SYNC = 0x0000
        SND_FILENAME = 0x00020000
        winmm.PlaySoundA(str(wav_path.resolve()).encode('ascii'), 0, SND_FILENAME)


def generate_mock_human_audio(command_word: str, duration_sec: float = 0.8) -> np.ndarray:
    """
    Generates a brief acoustic test signal for automated CLI verification in headless environments.
    """
    sr = 16000
    n = int(sr * duration_sec)
    t = np.linspace(0, duration_sec, n, endpoint=False)
    # Fundamental voice tone around 150 Hz with harmonics
    voice = 0.4 * np.sin(2 * np.pi * 150 * t) + 0.2 * np.sin(2 * np.pi * 300 * t)
    # Burst / envelope
    env = np.hanning(n)
    # Unique jitter so multiple mock takes have distinct SHA-256 hashes
    jitter = np.random.randn(n) * 0.005
    audio = (voice * env) + jitter
    return audio.astype(np.float32)


def run_recording_session(
    speaker_id: str,
    commands: List[str] = ["TRAI", "PHAI", "LEN", "XUONG"],
    takes_per_command: int = 3,
    session_id: str = "SES_001",
    device_model: str = "Realtek Laptop Mic",
    environment: str = "quiet_room",
    speaking_style: str = "normal",
    mock_mode: bool = False,
    microphone_type: Optional[str] = None,
    distance_category: Optional[str] = None,
    strict_metadata: bool = False,
    manifest_path: Optional[Path] = None,
    quarantine_path: Optional[Path] = None,
    raw_base_dir: Optional[Path] = None,
    std_base_dir: Optional[Path] = None,
    ledger_path: Optional[Path] = None,
) -> bool:
    print("=" * 65)
    print("VIETNAMESE KEYWORD SPOTTING — FIELD RECORDING CAMPAIGN CLI")
    print("=" * 65)

    # 1. Verify consent (GATE-08: Pre-recording mandatory invariant)
    if not verify_speaker_consent(speaker_id, ledger_path=ledger_path):
        print(f"\n[ERROR] GATE-08 ENFORCEMENT: Speaker '{speaker_id}' not found, consent not granted, revoked, or unverifiable.")
        print("Microphone recording is BLOCKED. Please register speaker first using --register.")
        return False

    spk_meta = get_speaker_metadata(speaker_id, ledger_path=ledger_path) or {}
    resolved_mic = microphone_type or spk_meta.get("primary_microphone")
    resolved_dist = distance_category or spk_meta.get("distance_category") or spk_meta.get("speaking_distance")
    effective_strict = True if not mock_mode else strict_metadata

    print(f"Speaker ID:   {speaker_id}")
    print(f"Dialect:      {spk_meta.get('dialect')}")
    print(f"Gender:       {spk_meta.get('gender')}")
    print(f"Mic Type:     {resolved_mic}")
    print(f"Distance:     {resolved_dist}")
    print(f"Device:       {device_model}")
    print(f"Environment:  {environment}")
    print(f"Session ID:   {session_id}")
    print(f"Mock Mode:    {mock_mode}")
    print(f"Strict Mode:  {effective_strict}")
    print("=" * 65)

    recorder = AudioRecorder(sample_rate=44100, channels=2)
    temp_dir = Path(tempfile.gettempdir()) / "kws_field_cli_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    total_ingested = 0

    try:
        for cmd in commands:
            cmd_upper = cmd.upper()
            if cmd_upper not in VALID_COMMANDS:
                continue
            viet_word = VALID_COMMANDS[cmd_upper]["word"]

            print(f"\n>>> TARGET COMMAND: [{cmd_upper}] ('{viet_word}')")
            if cmd_upper == "TRAI":
                print("    Hanoi/Northern: Affricate burst /tɕ/ + tone sắc (rising).")
            elif cmd_upper == "PHAI":
                print("    Pronunciation: Fricative /f/ + tone hỏi (dipping-rising).")
            elif cmd_upper == "LEN":
                print("    Pronunciation: Lateral /l/ + tone ngang (level).")
            elif cmd_upper == "XUONG":
                print("    Pronunciation: Sibilant /s/ + tone sắc (rising).")

            take_idx = 1
            while take_idx <= takes_per_command:
                print(f"\n--- [{cmd_upper}] Take {take_idx}/{takes_per_command} ---")
                temp_wav = temp_dir / f"temp_{cmd_upper}_{take_idx}.wav"

                if mock_mode:
                    print(f"[MOCK RECORDING] Capturing simulated utterance for '{viet_word}'...")
                    mock_audio = generate_mock_human_audio(viet_word)
                    sf.write(temp_wav, mock_audio, 16000, subtype="PCM_16")
                    action = "A"
                else:
                    input("Press [ENTER] and speak clearly after prompt...")
                    print("[RECORDING 1.6s...] Nói: " + viet_word.upper())
                    success = recorder.record_to_file(temp_wav, duration_sec=1.6)
                    if not success:
                        print("[WARNING] Hardware recording failed. Retrying take...")
                        continue
                    print("[RECORDING COMPLETE]")

                    # Review actions
                    while True:
                        choice = input("Action: [A]ccept & Ingest | [P]layback | [R]e-record | [S]kip: ").strip().upper()
                        if choice == "P":
                            print("Playing back recording...")
                            recorder.play_file(temp_wav)
                        elif choice in ["A", "R", "S"]:
                            action = choice
                            break
                        else:
                            print("Invalid choice. Enter A, P, R, or S.")

                if action == "S":
                    print(f"Skipped take {take_idx}.")
                    take_idx += 1
                    continue
                elif action == "R":
                    print("Discarding and re-recording...")
                    continue
                elif action == "A":
                    res = ingest_recording(
                        input_audio_path=temp_wav,
                        speaker_id=speaker_id,
                        command_label=cmd_upper,
                        session_id=session_id,
                        device_model=device_model,
                        environment=environment,
                        speaking_style=speaking_style,
                        take_index=take_idx,
                        provenance="human_field_recording" if not mock_mode else "test_harness_mock",
                        microphone_type=resolved_mic,
                        distance_category=resolved_dist,
                        strict_metadata=effective_strict,
                        manifest_path=manifest_path,
                        quarantine_path=quarantine_path,
                        raw_base_dir=raw_base_dir,
                        std_base_dir=std_base_dir,
                        ledger_path=ledger_path,
                    )
                    if res["status"] == "APPROVED":
                        total_ingested += 1
                        take_idx += 1
                    else:
                        print(f"[QUALITY REJECTION] {res['reason']}. Please re-record take.")
                        if mock_mode:
                            take_idx += 1
                        continue
    except KeyboardInterrupt:
        print("\n\n[PAUSED] Recording session interrupted by user. Progress saved.")
    finally:
        for f in temp_dir.glob("temp_*.wav"):
            try:
                f.unlink(missing_ok=True)
            except Exception:
                pass

    print(f"\n[SESSION COMPLETE] Successfully ingested {total_ingested} approved field recordings.")
    return True


def interactive_registration():
    print("\n--- NEW SPEAKER REGISTRATION ---")
    print("1. Northern (Giọng Bắc - Hanoi/Northern provinces)")
    print("2. Central  (Giọng Trung - Hue/Da Nang/Nghe An)")
    print("3. Southern (Giọng Nam - Ho Chi Minh City/Mekong Delta)")
    d_choice = input("Select Dialect (1/2/3): ").strip()
    dialect_map = {"1": "Northern", "2": "Central", "3": "Southern"}
    dialect = dialect_map.get(d_choice, "Northern")

    g_choice = input("Gender (M=Male / F=Female / O=Other) [M]: ").strip().upper() or "M"
    gender_map = {"M": "MALE", "F": "FEMALE", "O": "OTHER"}
    gender = gender_map.get(g_choice, "MALE")

    age = input("Age bracket (18-30 / 31-45 / 46-60 / over_60) [18-30]: ").strip() or "18-30"
    device = input("Device model [iPhone 13 / Laptop Mic]: ").strip() or "Realtek Laptop Mic"
    env = input("Environment [quiet_room / office_hvac / street_noise]: ").strip() or "quiet_room"

    print("\nMicrophone Types:")
    print("1. builtin_mobile    (Built-in Smartphone Mic)")
    print("2. wired_headset     (Wired 3.5mm/Type-C Headset)")
    print("3. bluetooth_earbuds  (Wireless Bluetooth Earbuds)")
    print("4. laptop_mic        (Internal Laptop Microphone)")
    print("5. usb_desktop_mic   (USB Condenser / Desktop Mic)")
    mic_choice = input("Select Microphone (1-5) [1]: ").strip() or "1"
    mic_map = {
        "1": "builtin_mobile",
        "2": "wired_headset",
        "3": "bluetooth_earbuds",
        "4": "laptop_mic",
        "5": "usb_desktop_mic"
    }
    mic_type = mic_map.get(mic_choice, "builtin_mobile")

    print("\nDistance Categories:")
    print("1. close_talk_15cm    (Handheld / Near-field ~15cm)")
    print("2. medium_talk_50cm   (Desk / Mid-field ~50cm)")
    print("3. far_field_2m       (Room / Far-field ~2m)")
    dist_choice = input("Select Distance (1-3) [1]: ").strip() or "1"
    dist_map = {
        "1": "close_talk_15cm",
        "2": "medium_talk_50cm",
        "3": "far_field_2m"
    }
    dist_cat = dist_map.get(dist_choice, "close_talk_15cm")

    print("\n[INFORMED CONSENT NOTICE]")
    print("I consent to the use of my voice recordings for Vietnamese machine learning research")
    print("and offline keyword spotting system evaluation without personal identifying information.")
    consent_input = input("Grant consent? (YES/no): ").strip().upper()
    if consent_input not in ["Y", "YES"]:
        print("[CANCELLED] Consent was not given. Registration aborted.")
        return None

    record = register_speaker(
        dialect=dialect,
        gender=gender,
        age_bracket=age,
        consent_given=True,
        device_model=device,
        microphone_type=mic_type,
        distance_category=dist_cat,
        acoustic_environment=env
    )
    return record["speaker_id"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vietnamese KWS Field Recording CLI")
    parser.add_argument("--speaker-id", "--speaker_id", dest="speaker_id", type=str, help="Existing Speaker ID")
    parser.add_argument("--register", action="store_true", help="Register a new speaker interactively")
    parser.add_argument("--dialect", type=str, choices=VALID_DIALECTS, help="Dialect if registering via CLI")
    parser.add_argument("--gender", type=str, choices=["MALE", "FEMALE", "OTHER"], help="Gender if registering via CLI")
    parser.add_argument("--age", type=str, choices=["18-30", "31-45", "46-60", "over_60", "unspecified"], help="Age bracket if registering via CLI")
    parser.add_argument("--mic", "--microphone_type", dest="microphone_type", type=str, choices=VALID_MICROPHONE_TYPES, help="Microphone type")
    parser.add_argument("--distance", "--distance_category", dest="distance_category", type=str, choices=VALID_DISTANCE_CATEGORIES, help="Distance category")
    parser.add_argument("--sub-category", "--sub_category", dest="sub_category", type=str, help="Sub-category (e.g. conversational_speech, background_room_tone)")
    parser.add_argument("--takes", type=int, default=3, help="Number of takes per command")
    parser.add_argument("--session", type=str, default="SES_001", help="Session identifier")
    parser.add_argument("--device", type=str, default="Default Mic", help="Device model")
    parser.add_argument("--env", type=str, default="quiet_room", help="Acoustic environment")
    parser.add_argument("--commands", nargs="+", default=["TRAI", "PHAI", "LEN", "XUONG"], help="Commands to record")
    parser.add_argument("--mock", action="store_true", help="Run in mock/dry-run mode for automated verification")
    parser.add_argument("--ingest-file", "--ingest_file", dest="ingest_file", type=str, help="Directly ingest an external audio file (.wav, .m4a, .aac, .flac)")
    parser.add_argument("--command", type=str, choices=list(VALID_COMMANDS.keys()), help="Target command label for --ingest-file")
    parser.add_argument("--take", type=int, default=1, help="Take index for --ingest-file")
    parser.add_argument("--strict", action="store_true", help="Enforce strict metadata validation for genuine human recordings")
    args = parser.parse_args()

    spk_id = args.speaker_id

    # Handle single file ingest mode
    if args.ingest_file:
        if not spk_id:
            raise ValueError("--speaker-id is required when using --ingest-file.")
        if not args.command:
            raise ValueError("--command is required when using --ingest-file (e.g. --command TRAI).")
        file_path = Path(args.ingest_file)
        if not file_path.exists():
            raise FileNotFoundError(f"Specified ingest file does not exist: {file_path}")
        print(f"Ingesting file '{file_path.name}' for speaker {spk_id} as [{args.command}]...")
        effective_strict = True if not args.mock else args.strict
        prov = "human_field_recording" if not args.mock else "test_harness_mock"
        result = ingest_recording(
            input_audio_path=file_path,
            speaker_id=spk_id,
            command_label=args.command,
            session_id=args.session,
            device_model=args.device,
            environment=args.env,
            take_index=args.take,
            sub_category=args.sub_category,
            microphone_type=args.microphone_type,
            distance_category=args.distance_category,
            age_bracket=args.age,
            strict_metadata=effective_strict,
            provenance=prov
        )
        print("Ingestion result:", result["status"])
        sys.exit(0 if result["status"] == "APPROVED" else 1)

    if args.register:
        if args.dialect and args.gender:
            rec = register_speaker(
                dialect=args.dialect,
                gender=args.gender,
                age_bracket=args.age or "18-30",
                consent_given=True,
                device_model=args.device,
                microphone_type=args.microphone_type or "builtin_mobile",
                distance_category=args.distance_category or "close_talk_15cm",
                acoustic_environment=args.env
            )
            spk_id = rec["speaker_id"]
        else:
            spk_id = interactive_registration()

    if not spk_id:
        existing = list_registered_speakers()
        if existing:
            print("Existing registered speakers:")
            for s in existing:
                print(f"  - {s['speaker_id']}: {s['dialect']}, {s['gender']}, Device: {s['primary_device']}")
            spk_id = input("Enter speaker ID to start session: ").strip()
        else:
            print("No speakers registered yet. Please run with --register.")
            sys.exit(0)

    if spk_id:
        success = run_recording_session(
            speaker_id=spk_id,
            commands=args.commands,
            takes_per_command=args.takes,
            session_id=args.session,
            device_model=args.device,
            environment=args.env,
            mock_mode=args.mock,
            microphone_type=args.microphone_type,
            distance_category=args.distance_category,
            strict_metadata=args.strict
        )
        if not success:
            sys.exit(1)
