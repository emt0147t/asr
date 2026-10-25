#!/usr/bin/env python3
"""
Command-Line WAV Inference Tool for Vietnamese Bounded-Utterance KWS (Phase 91).
Frozen Candidate: variant_w10 (CorrectedVietnameseKWSNet)
Workspace: D:\trainidentificationaudio
Experiment: experiments/vieneu_kws_research_v10/deployment/

Accepts a user-supplied WAV file and returns:
- Predicted command (TRAI, PHAI, LEN, XUONG, or REJECT)
- Rejection status (ACCEPTED or REJECTED)
- Confidence and class probabilities
- Preprocessing and inference latencies
- Checkpoint hash verification

Exit Codes:
  0: Success
  1: General execution error
  2: Audio file not found
  3: Empty audio file (0 bytes or 0 frames)
  4: Corrupted audio file / unreadable header
  5: Invalid sample rate (strict mode)
  6: Invalid channel count (strict mode)
  7: Invalid duration (strict mode)
  8: Checkpoint not found
"""

import os
import sys
import time
import json
import argparse
import hashlib
from pathlib import Path

# Ensure workspace root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F

from experiments.vieneu_kws_research_v9.artifacts.inference import (
    CorrectedVietnameseKWSNet,
    LogMelFrontend,
    CLASSES
)
from experiments.vieneu_kws_research_v10.deployment.audio_validator import (
    AudioValidator,
    AudioValidationError,
    AudioFileNotFoundError,
    EmptyAudioError,
    CorruptedAudioError,
    InvalidSampleRateError,
    InvalidChannelCountError,
    InvalidDurationError,
    CheckpointNotFoundError
)

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "experiments"
    / "vieneu_kws_research_v9"
    / "artifacts"
    / "checkpoints"
    / "variant_w10_seed42.pt"
)

DEFAULT_THRESHOLD = 0.40

DISCLAIMERS = {
    "human_listener_verification": "PENDING_HUMAN_LISTENER_VERIFICATION",
    "natural_speech_validation": "NOT_VALIDATED_ON_NATURAL_SPEECH",
    "continuous_open_mic_verdict": "NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION"
}


def compute_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class PredictorEngine:
    def __init__(
        self,
        checkpoint_path: Path = DEFAULT_CHECKPOINT,
        onnx_path: Path | None = None,
        theta_cmd: float = DEFAULT_THRESHOLD,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.theta_cmd = theta_cmd
        self.frontend = LogMelFrontend().to(self.device)
        self.frontend.eval()

        self.onnx_mode = onnx_path is not None
        self.ort_session = None

        if self.onnx_mode:
            if not onnx_path.exists():
                raise CheckpointNotFoundError(f"ONNX model file not found: {onnx_path}")
            import onnxruntime as ort
            sess_opts = ort.SessionOptions()
            sess_opts.intra_op_num_threads = 1
            sess_opts.inter_op_num_threads = 1
            self.ort_session = ort.InferenceSession(str(onnx_path), sess_opts, providers=["CPUExecutionProvider"])
            self.model_path = onnx_path
            self.model_sha256 = compute_file_sha256(onnx_path)
            self.backend = "onnxruntime"
        else:
            if not checkpoint_path.exists():
                raise CheckpointNotFoundError(f"PyTorch checkpoint not found: {checkpoint_path}")
            self.model = CorrectedVietnameseKWSNet(num_classes=5).to(self.device)
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.model_path = checkpoint_path
            self.model_sha256 = compute_file_sha256(checkpoint_path)
            self.backend = f"pytorch_{device}"

    @torch.no_grad()
    def predict_wav(self, wav_path: Path | str, strict: bool = False) -> dict:
        t0 = time.perf_counter()

        # Step 1: Audio Loading & Preprocessing
        wav_tensor, orig_sr, meta = AudioValidator.load_and_validate(wav_path, strict=strict)
        wav_tensor = wav_tensor.to(self.device)

        # Step 2: Mel-Spectrogram extraction
        feats = self.frontend(wav_tensor)  # [1, 1, 40, 101]
        t1 = time.perf_counter()
        prep_ms = (t1 - t0) * 1000.0

        # Step 3: Model Inference
        t_inf_0 = time.perf_counter()
        if self.onnx_mode:
            input_name = self.ort_session.get_inputs()[0].name
            feats_np = feats.cpu().numpy()
            ort_outs = self.ort_session.run(None, {input_name: feats_np})
            logits_np = ort_outs[0]
            logits = torch.from_numpy(logits_np)
        else:
            logits = self.model(feats)

        probs = F.softmax(logits, dim=-1)
        t_inf_1 = time.perf_counter()
        inf_ms = (t_inf_1 - t_inf_0) * 1000.0
        total_ms = prep_ms + inf_ms

        cmd_probs = probs[0, :4]
        max_cmd_prob, cmd_idx = torch.max(cmd_probs, dim=-1)
        max_prob_val = float(max_cmd_prob.item())

        if max_prob_val >= self.theta_cmd:
            pred_class = CLASSES[cmd_idx.item()]
            rejection_status = "ACCEPTED"
            confidence = max_prob_val
        else:
            pred_class = "REJECT"
            rejection_status = "REJECTED"
            confidence = float(probs[0, 4].item())

        prob_dict = {CLASSES[i]: float(probs[0, i].item()) for i in range(5)}

        return {
            "status": "SUCCESS",
            "wav_path": str(wav_path),
            "audio_metadata": meta,
            "predicted_class": pred_class,
            "rejection_status": rejection_status,
            "confidence": round(confidence, 4),
            "command_threshold": self.theta_cmd,
            "probabilities": {k: round(v, 6) for k, v in prob_dict.items()},
            "backend": self.backend,
            "model_path": str(self.model_path),
            "model_sha256": self.model_sha256,
            "latency": {
                "preprocessing_ms": round(prep_ms, 2),
                "inference_ms": round(inf_ms, 2),
                "total_ms": round(total_ms, 2)
            },
            "governance": DISCLAIMERS
        }


def format_text_output(res: dict) -> str:
    m = res["audio_metadata"]
    lat = res["latency"]
    p = res["probabilities"]
    lines = [
        "=" * 80,
        "VIETNAMESE KEYWORD SPOTTING - OFFLINE WAV INFERENCE (PHASE 91)",
        "=" * 80,
        f"File:            {res['wav_path']}",
        f"Audio Info:      {m['original_sample_rate']} Hz, {m['original_channels']} ch, {m['original_frames']} frames ({m['original_duration_seconds']:.3f}s)",
        f"Validation:      {'STRICT (exact 16kHz mono 1.0s)' if m['strict_validation'] else 'PERMISSIVE (auto-resample & pad/trunc)'}",
        f"Backend:         {res['backend']}",
        f"Model Checkpoint: {Path(res['model_path']).name} (SHA256: {res['model_sha256'][:16]}...)",
        "-" * 80,
        f"PREDICTION:      {res['predicted_class']}",
        f"STATUS:          {res['rejection_status']} (threshold = {res['command_threshold']:.2f})",
        f"CONFIDENCE:      {res['confidence'] * 100:.2f}%",
        "-" * 80,
        "Class Probabilities:",
        f"  - TRAI:        {p['TRAI']:.4f}",
        f"  - PHAI:        {p['PHAI']:.4f}",
        f"  - LEN:         {p['LEN']:.4f}",
        f"  - XUONG:       {p['XUONG']:.4f}",
        f"  - REJECT:      {p['REJECT']:.4f}",
        "-" * 80,
        "Latency:",
        f"  - Preprocessing: {lat['preprocessing_ms']:.2f} ms",
        f"  - Inference:     {lat['inference_ms']:.2f} ms",
        f"  - Total:         {lat['total_ms']:.2f} ms",
        "-" * 80,
        "Governance Invariants:",
        f"  [!] Human Listening:       {DISCLAIMERS['human_listener_verification']}",
        f"  [!] Natural Speech:        {DISCLAIMERS['natural_speech_validation']}",
        f"  [!] Continuous Open-Mic:   {DISCLAIMERS['continuous_open_mic_verdict']}",
        "=" * 80
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Predict Vietnamese KWS keyword from a 16kHz WAV file (variant_w10)."
    )
    parser.add_argument("wav_path", type=str, help="Path to input WAV file.")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT),
                        help="Path to PyTorch .pt model checkpoint.")
    parser.add_argument("--onnx-model", type=str, default=None,
                        help="Path to ONNX model artifact (enables ONNX Runtime inference).")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="Command gating threshold theta_cmd (default 0.40).")
    parser.add_argument("--strict", action="store_true",
                        help="Enforce strict validation (16,000 Hz, mono, exactly 16,000 samples).")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Execution device (default: cpu).")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON.")

    args = parser.parse_args()

    # Audio file existence check before engine init
    wav_file = Path(args.wav_path)

    try:
        engine = PredictorEngine(
            checkpoint_path=Path(args.checkpoint),
            onnx_path=Path(args.onnx_model) if args.onnx_model else None,
            theta_cmd=args.threshold,
            device=args.device
        )
        res = engine.predict_wav(wav_file, strict=args.strict)

        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print(format_text_output(res))
        sys.exit(0)

    except AudioFileNotFoundError as e:
        err_dict = {"status": "ERROR", "error_type": "AudioFileNotFoundError", "error_message": str(e), "exit_code": 2}
        if args.json:
            print(json.dumps(err_dict, indent=2))
        else:
            print(f"Error [AudioFileNotFoundError]: {e}", file=sys.stderr)
        sys.exit(2)

    except EmptyAudioError as e:
        err_dict = {"status": "ERROR", "error_type": "EmptyAudioError", "error_message": str(e), "exit_code": 3}
        if args.json:
            print(json.dumps(err_dict, indent=2))
        else:
            print(f"Error [EmptyAudioError]: {e}", file=sys.stderr)
        sys.exit(3)

    except CorruptedAudioError as e:
        err_dict = {"status": "ERROR", "error_type": "CorruptedAudioError", "error_message": str(e), "exit_code": 4}
        if args.json:
            print(json.dumps(err_dict, indent=2))
        else:
            print(f"Error [CorruptedAudioError]: {e}", file=sys.stderr)
        sys.exit(4)

    except InvalidSampleRateError as e:
        err_dict = {"status": "ERROR", "error_type": "InvalidSampleRateError", "error_message": str(e), "exit_code": 5}
        if args.json:
            print(json.dumps(err_dict, indent=2))
        else:
            print(f"Error [InvalidSampleRateError]: {e}", file=sys.stderr)
        sys.exit(5)

    except InvalidChannelCountError as e:
        err_dict = {"status": "ERROR", "error_type": "InvalidChannelCountError", "error_message": str(e), "exit_code": 6}
        if args.json:
            print(json.dumps(err_dict, indent=2))
        else:
            print(f"Error [InvalidChannelCountError]: {e}", file=sys.stderr)
        sys.exit(6)

    except InvalidDurationError as e:
        err_dict = {"status": "ERROR", "error_type": "InvalidDurationError", "error_message": str(e), "exit_code": 7}
        if args.json:
            print(json.dumps(err_dict, indent=2))
        else:
            print(f"Error [InvalidDurationError]: {e}", file=sys.stderr)
        sys.exit(7)

    except CheckpointNotFoundError as e:
        err_dict = {"status": "ERROR", "error_type": "CheckpointNotFoundError", "error_message": str(e), "exit_code": 8}
        if args.json:
            print(json.dumps(err_dict, indent=2))
        else:
            print(f"Error [CheckpointNotFoundError]: {e}", file=sys.stderr)
        sys.exit(8)

    except Exception as e:
        err_dict = {"status": "ERROR", "error_type": type(e).__name__, "error_message": str(e), "exit_code": 1}
        if args.json:
            print(json.dumps(err_dict, indent=2))
        else:
            print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
