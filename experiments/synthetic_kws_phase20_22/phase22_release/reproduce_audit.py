"""
Reproducible Entry Point 1: Full Audit and Verification
Workspace: D:\\trainidentificationaudio

Executes complete dataset integrity audit, 3,000-sample bit-exact regeneration,
multi-seed model metric recomputations, diagnostic probes, and finite-sample uncertainty calculations.
"""

import sys
from pathlib import Path

# Insert repository root
sys.path.insert(0, str(Path(".").resolve()))
from experiments.synthetic_kws_phase20_22.phase20_audit.phase20_independent_audit import run_phase20_audit

if __name__ == "__main__":
    print("Running reproducible audit entry point...")
    results = run_phase20_audit()
    print("Audit run complete. Gate Verdict:", results.get("gate_verdict"))
