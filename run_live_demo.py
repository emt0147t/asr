#!/usr/bin/env python3
"""
Root Launcher for Vietnamese Keyword Spotting (KWS) Laptop Live-Mic Demo.
Phase 13.4 Final Release Candidate (v11).

Usage:
    python run_live_demo.py
    python run_live_demo.py --list-devices
    python run_live_demo.py --device 2
    python run_live_demo.py --gui
"""

import sys
import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEMO_SCRIPT = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4" / "demo" / "live_mic_app.py"

# Use .venv python if present
VENV_PY = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
PYTHON_EXE = str(VENV_PY) if VENV_PY.exists() else sys.executable

if __name__ == "__main__":
    if not DEMO_SCRIPT.exists():
        print(f"[ERROR] Demo script not found: {DEMO_SCRIPT}", file=sys.stderr)
        sys.exit(1)

    cmd = [PYTHON_EXE, str(DEMO_SCRIPT)] + sys.argv[1:]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    try:
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)
        sys.exit(proc.returncode)
    except KeyboardInterrupt:
        sys.exit(0)
