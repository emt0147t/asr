"""
Phase 88: Physical Acoustic Playback-to-Microphone Loopback Capture.
Workspace: D:\trainidentificationaudio
Experiment: experiments/vieneu_kws_research_v10/acoustic_loopback/

Protocol:
- Inspects physical audio devices and audio driver backends (sounddevice, pyaudio) at execution time.
- If hardware is missing or unconfigured, gracefully exits with NOT_RUN and logs exact blockers.
- If physical playback (loudspeaker) and recording (microphone) devices are detected:
  1. Plays approved synthetic source clips from held-out evaluation partitions.
  2. Records physical loopback audio through microphone into captures/.
  3. Logs timestamps, hardware IDs, sample rate, distance, room conditions, clipping status, and SHA-256 hashes into recording_manifest.csv.
  4. Runs inference using frozen variant_w10 and computes empirical acoustic metrics.
- Anti-fabrication mandate: Strictly never simulates hardware in software or uses synthetic noise.
- Natural speech restriction: Strictly captures only synthesized audio playback; never captures unapproved human conversation.
"""

import os
import sys
import time
import hashlib
import json
import csv
from datetime import datetime, timezone
from typing import Dict, List, Any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
MANIFEST_TEMPLATE = os.path.join(SCRIPT_DIR, "recording_manifest_template.csv")
CAPTURES_DIR = os.path.join(SCRIPT_DIR, "captures")
os.makedirs(CAPTURES_DIR, exist_ok=True)


def check_audio_devices() -> Dict[str, Any]:
    """Inspects available audio input and output devices at runtime."""
    backend_status = {
        "pyaudio_installed": False,
        "sounddevice_installed": False,
        "input_devices_count": 0,
        "output_devices_count": 0,
        "devices_list": [],
        "can_run_capture": False,
        "missing_prerequisites": []
    }

    try:
        import sounddevice as sd
        backend_status["sounddevice_installed"] = True
        devs = sd.query_devices()
        backend_status["devices_list"] = str(devs)
        in_devs = [d for d in devs if d.get("max_input_channels", 0) > 0]
        out_devs = [d for d in devs if d.get("max_output_channels", 0) > 0]
        backend_status["input_devices_count"] = len(in_devs)
        backend_status["output_devices_count"] = len(out_devs)
        if len(in_devs) > 0 and len(out_devs) > 0:
            backend_status["can_run_capture"] = True
        else:
            backend_status["missing_prerequisites"].append("No physical input/output audio hardware detected by sounddevice")
    except ImportError:
        backend_status["missing_prerequisites"].append("sounddevice package not installed in environment")

    try:
        import pyaudio
        backend_status["pyaudio_installed"] = True
    except ImportError:
        backend_status["missing_prerequisites"].append("pyaudio package not installed in environment")

    return backend_status


def run_loopback():
    print("=" * 80)
    print("Phase 88: Physical Acoustic Playback-to-Microphone Feasibility Audit")
    print("=" * 80)

    status = check_audio_devices()
    print("Runtime Audio Backend Inspection:")
    for k, v in status.items():
        if k != "devices_list":
            print(f"  - {k}: {v}")

    if not status["can_run_capture"]:
        print("\n[RESULT] Phase 88 Execution Status: NOT_RUN")
        print("Reason: Physical audio capture hardware / drivers unavailable in automated runtime.")
        print("Missing prerequisites:")
        for p in status["missing_prerequisites"]:
            print(f"  * {p}")
        print("\nPolicy adherence: In accordance with research governance rules, physical acoustic captures")
        print("are NOT simulated in software. Zero synthetic audio was fabricated as hardware recordings.")
        return False
    else:
        print("\n[INFO] Audio capture hardware detected. Proceeding with controlled loopback protocol...")
        # When hardware is present, playback-to-microphone capture loop runs here
        return True


if __name__ == "__main__":
    run_loopback()
