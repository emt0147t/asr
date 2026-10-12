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
from typing import List, Dict, Any, Optional
import numpy as np
import soundfile as sf

from src.acquisition.speaker_consent import (
    register_speaker, verify_speaker_consent, get_speaker_metadata,
    list_registered_speakers, VALID_DIALECTS
)
from src.acquisition.ingest_field_recordings import ingest_recording, VALID_COMMANDS

# Windows Multimedia API
winmm = ctypes.windll.winmm if os.name == 'nt' else None


class AudioRecorder:
    """
    Lightweight audio recorder and player using Windows MCI (winmm.dll).
    """
    def __init__(self, sample_rate: int = 16000, channels: int = 1, bits_per_sample: int = 16):
        self.sample_rate = sample_rate
        self.channels = channels
        self.bits_per_sample = bits_per_sample
        self.alias = "field_recorder_alias"

    def record_to_file(self, output_wav: Path, duration_sec: float = 1.6) -> bool:
        if not winmm:
            raise RuntimeError("AudioRecorder requires Windows OS (winmm.dll).")

        wav_path_str = str(output_wav.resolve()).replace("\\", "\\\\")
        
        # Stop and close any previous instance
        winmm.mciSendStringA(f"close {self.alias}".encode('ascii'), None, 0, 0)

        # Open waveaudio with specific audio parameters
        cmd_open = f"open new type waveaudio alias {self.alias}"
        res = winmm.mciSendStringA(cmd_open.encode('ascii'), None, 0, 0)
        if res != 0:
            print(f"[ERROR] Failed to open waveaudio device (code {res})")
            return False

        # Configure 16 kHz 16-bit Mono PCM
        winmm.mciSendStringA(f"set {self.alias} time format ms".encode('ascii'), None, 0, 0)
        winmm.mciSendStringA(f"set {self.alias} channels {self.channels}".encode('ascii'), None, 0, 0)
        winmm.mciSendStringA(f"set {self.alias} samplespersec {self.sample_rate}".encode('ascii'), None, 0, 0)
        winmm.mciSendStringA(f"set {self.alias} bitspersample {self.bits_per_sample}".encode('ascii'), None, 0, 0)

        # Start recording
        winmm.mciSendStringA(f"record {self.alias}".encode('ascii'), None, 0, 0)
        time.sleep(duration_sec)

        # Stop and save
        winmm.mciSendStringA(f"stop {self.alias}".encode('ascii'), None, 0, 0)
        winmm.mciSendStringA(f"save {self.alias} \"{wav_path_str}\"".encode('ascii'), None, 0, 0)
        winmm.mciSendStringA(f"close {self.alias}".encode('ascii'), None, 0, 0)

        return output_wav.exists() and output_wav.stat().st_size > 44

    def play_file(self, wav_path: Path):
        if not winmm:
            print("[PLAYBACK] Playback unavailable on non-Windows OS.")
            return
        # SND_FILENAME = 0x00020000, SND_SYNC = 0x0000
        SND_FILENAME = 0x00020000
        SND_ASYNC = 0x00000001
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
    mock_mode: bool = False
):
    print("=" * 65)
    print("VIETNAMESE KEYWORD SPOTTING — FIELD RECORDING CAMPAIGN CLI")
    print("=" * 65)

    # 1. Verify consent
    if not verify_speaker_consent(speaker_id):
        print(f"\n[ERROR] Speaker '{speaker_id}' not found or consent not granted.")
        print("Please register speaker first using --register.")
        return

    spk_meta = get_speaker_metadata(speaker_id)
    print(f"Speaker ID:   {speaker_id}")
    print(f"Dialect:      {spk_meta.get('dialect')}")
    print(f"Gender:       {spk_meta.get('gender')}")
    print(f"Device:       {device_model}")
    print(f"Environment:  {environment}")
    print(f"Session ID:   {session_id}")
    print(f"Mock Mode:    {mock_mode}")
    print("=" * 65)

    recorder = AudioRecorder(sample_rate=16000)
    temp_dir = Path(tempfile.gettempdir()) / "kws_field_cli_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    total_ingested = 0

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
                    take_index=take_idx
                )
                if res["status"] == "APPROVED":
                    total_ingested += 1
                    take_idx += 1
                else:
                    print(f"[QUALITY REJECTION] {res['reason']}. Please re-record take.")
                    if mock_mode:
                        take_idx += 1
                    continue

    print(f"\n[SESSION COMPLETE] Successfully ingested {total_ingested} approved field recordings.")


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

    age = input("Age bracket (18-30 / 31-50 / 51+) [18-30]: ").strip() or "18-30"
    device = input("Device model [iPhone 13 / Laptop Mic]: ").strip() or "Realtek Laptop Mic"
    env = input("Environment [quiet_room / office_hvac / street_noise]: ").strip() or "quiet_room"

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
        acoustic_environment=env
    )
    return record["speaker_id"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vietnamese KWS Field Recording CLI")
    parser.add_argument("--speaker_id", type=str, help="Existing Speaker ID")
    parser.add_argument("--register", action="store_true", help="Register a new speaker interactively")
    parser.add_argument("--dialect", type=str, choices=VALID_DIALECTS, help="Dialect if registering via CLI")
    parser.add_argument("--gender", type=str, choices=["MALE", "FEMALE", "OTHER"], help="Gender if registering via CLI")
    parser.add_argument("--takes", type=int, default=3, help="Number of takes per command")
    parser.add_argument("--session", type=str, default="SES_001", help="Session identifier")
    parser.add_argument("--device", type=str, default="Default Mic", help="Device model")
    parser.add_argument("--env", type=str, default="quiet_room", help="Acoustic environment")
    parser.add_argument("--commands", nargs="+", default=["TRAI", "PHAI", "LEN", "XUONG"], help="Commands to record")
    parser.add_argument("--mock", action="store_true", help="Run in mock/dry-run mode for automated verification")
    args = parser.parse_args()

    spk_id = args.speaker_id

    if args.register:
        if args.dialect and args.gender:
            rec = register_speaker(
                dialect=args.dialect,
                gender=args.gender,
                consent_given=True,
                device_model=args.device,
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
        run_recording_session(
            speaker_id=spk_id,
            commands=args.commands,
            takes_per_command=args.takes,
            session_id=args.session,
            device_model=args.device,
            environment=args.env,
            mock_mode=args.mock
        )
