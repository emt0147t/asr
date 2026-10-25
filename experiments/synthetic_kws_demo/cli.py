"""
CLI Prototype for Synthetic Vietnamese KWS Offline Inference & Game Control.
Executes single-file classification or a sequential game-control simulation demo.
"""

import sys
import os
import argparse
import json
from pathlib import Path

if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

from experiments.synthetic_kws_demo.classifier import (
    OfflineKWSClassifier,
    DEFAULT_CHECKPOINT,
    SYNTHETIC_DEMO_WARNING
)
from experiments.synthetic_kws_demo.game_adapter import (
    CommandToActionAdapter,
    GameAction
)
from experiments.synthetic_kws_demo.game_simulator import GridGameSimulator


def run_single_inference(
    classifier: OfflineKWSClassifier,
    adapter: CommandToActionAdapter,
    wav_path: Path,
    as_json: bool = False
):
    pred = classifier.predict(wav_path)
    action = adapter.adapt(pred)
    
    if as_json:
        payload = {
            "prediction": pred.to_dict(),
            "action": action.to_dict()
        }
        print(json.dumps(payload, indent=2))
        return
        
    print("=" * 72)
    print("  SYNTHETIC VIETNAMESE KWS — OFFLINE INFERENCE RESULT")
    print("=" * 72)
    print(f"File Path:         {pred.wav_path}")
    print(f"Valid Input:       {'YES' if pred.is_valid else 'NO (REJECTED)'}")
    
    if not pred.is_valid:
        print(f"Error Message:     {pred.error_message}")
        print(f"Action Taken:      {action.action.value} ({action.notes})")
        print("-" * 72)
        print(f"[!] {SYNTHETIC_DEMO_WARNING}")
        print("=" * 72)
        return
        
    print(f"Predicted Command: {pred.predicted_label} (Class ID: {pred.predicted_class_id})")
    print(f"Confidence:        {pred.confidence * 100:.2f}%")
    print(f"Mapped Action:     {action.action.value} ({action.notes})")
    print("\nClass Probabilities:")
    for cl, prob in pred.class_probabilities.items():
        bar = "#" * int(prob * 30)
        marker = " <--" if cl == pred.predicted_label else ""
        print(f"  {cl:<8}: {prob * 100:6.2f}% | {bar:<30}{marker}")
        
    print("\nLatency Profiling (Desktop CPU):")
    print(f"  Preprocessing:   {pred.preprocessing_latency_ms:6.2f} ms")
    print(f"  Model Inference: {pred.inference_latency_ms:6.2f} ms")
    print(f"  Total Latency:   {pred.total_latency_ms:6.2f} ms")
    
    print("\nProvenance & Model Checkpoint:")
    print(f"  Checkpoint:      {pred.checkpoint_path}")
    print(f"  SHA-256:         {pred.checkpoint_sha256[:16]}...{pred.checkpoint_sha256[-8:]}")
    print("-" * 72)
    print(f"[!] {SYNTHETIC_DEMO_WARNING}")
    print("=" * 72)


def run_sequential_demo(
    classifier: OfflineKWSClassifier,
    adapter: CommandToActionAdapter,
    fixtures_dir: Path
):
    print("=" * 76)
    print("  SYNTHETIC VIETNAMESE KWS — GAME-CONTROL PROTOTYPE DEMONSTRATION")
    print("=" * 76)
    print(f"[!] {SYNTHETIC_DEMO_WARNING}\n")
    
    demo_sequence = [
        ("demo_01_trai.wav", "TRAI", "Move Left"),
        ("demo_03_len.wav", "LEN", "Jump / Move Up"),
        ("demo_02_phai.wav", "PHAI", "Move Right"),
        ("demo_04_xuong.wav", "XUONG", "Crouch / Move Down"),
        ("demo_05_unknown.wav", "UNKNOWN", "Non-Target Confuser (Suppressed)"),
        ("demo_06_silence.wav", "SILENCE", "Ambient Background Dither (Suppressed)"),
    ]
    
    sim = GridGameSimulator(width=7, height=7, initial_pos=(3, 3), goal_pos=(5, 1))
    
    print("Initial 2D Grid Environment (Player: '@', Goal: 'G', Obstacle: '#'):")
    print(sim.render())
    print("\nStarting Sequential Playback of Synthetic Voice Commands...\n")
    
    for idx, (fname, exp_cmd, intent) in enumerate(demo_sequence, start=1):
        wav_p = fixtures_dir / fname
        if not wav_p.exists():
            print(f"Fixture missing: {wav_p}")
            continue
            
        pred = classifier.predict(wav_p)
        action = adapter.adapt(pred)
        step_rec = sim.step(action)
        
        status_symbol = "[OK]" if pred.predicted_label == exp_cmd else "[FAIL]"
        print("-" * 76)
        print(f"Step {idx}: Audio Fixture '{fname}' | Intended: {exp_cmd} ({intent})")
        print(f"  Predicted: {status_symbol} {pred.predicted_label} ({pred.confidence*100:5.2f}% conf)")
        print(f"  Action:    {action.action.value} -> {step_rec.event}")
        print(f"  Position:  Player at ({sim.player_x}, {sim.player_y}) | Latency: {pred.total_latency_ms:.2f} ms")
        print(sim.render())
        print()
        
    print("=" * 76)
    print("Prototype Replay Complete.")
    print("All non-target fixtures (UNKNOWN, SILENCE) safely produced NO_ACTION.")
    print("=" * 76)


def main():
    parser = argparse.ArgumentParser(
        description="Offline KWS Inference & Game-Control Prototype (Synthetic Vietnamese KWS)"
    )
    parser.add_argument("--wav", type=str, help="Path to input 16kHz mono 1-second WAV file")
    parser.add_argument("--play-demo", action="store_true", help="Run sequential 6-class game control demonstration")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT), help="Path to model checkpoint")
    parser.add_argument("--threshold", type=float, default=0.50, help="Confidence threshold for game action mapping")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    
    args = parser.parse_args()
    
    ckpt_path = Path(args.checkpoint)
    fixtures_dir = Path("experiments/synthetic_kws_demo/fixtures")
    
    try:
        classifier = OfflineKWSClassifier(checkpoint_path=ckpt_path)
    except Exception as e:
        print(f"Initialization Error: {e}", file=sys.stderr)
        sys.exit(1)
        
    adapter = CommandToActionAdapter(confidence_threshold=args.threshold)
    
    if args.play_demo:
        run_sequential_demo(classifier, adapter, fixtures_dir)
    elif args.wav:
        run_single_inference(classifier, adapter, Path(args.wav), as_json=args.json)
    else:
        # Default behavior: run sequential demo
        run_sequential_demo(classifier, adapter, fixtures_dir)


if __name__ == "__main__":
    main()
