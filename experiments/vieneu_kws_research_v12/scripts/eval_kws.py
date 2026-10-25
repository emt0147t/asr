"""
Unified Evaluation Harness for Vietnamese Keyword Spotting Models.
Evaluates both Offline Locked Test Set (1,280 clips) and Streaming Multi-Command Corpus (40 sequences).
Compatible with frozen v11 baseline and newly trained E1 models.

Target: experiments/vieneu_kws_research_v12/scripts/eval_kws.py
"""

import sys
import io
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CLASSES = ["TRAI", "PHAI", "LEN", "XUONG", "REJECT"]
TARGET_CLASSES = ["TRAI", "PHAI", "LEN", "XUONG"]
DEFAULT_THRESHOLD = 0.20


# -----------------------------------------------------------------------------
# ARCHITECTURE & FRONTEND
# -----------------------------------------------------------------------------
class LogMelFrontend(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=512, win_length=400, hop_length=160, n_mels=40, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.mel = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm="slaney",
            n_mels=n_mels,
            mel_scale="slaney",
            f_min=20.0,
            f_max=8000.0,
        )

    def forward(self, x):
        mel_spec = self.mel(x)
        log_mel = torch.log(mel_spec + self.eps)
        if log_mel.dim() == 3:
            log_mel = log_mel.unsqueeze(1)
        return log_mel


class CorrectedVietnameseKWSNet(nn.Module):
    def __init__(self, num_classes=5, dropout_p=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # (40, 101) -> (20, 50)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # (20, 50) -> (10, 25)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # (10, 25) -> (5, 12)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))      # (5, 12) -> (1, 1)

        self.dropout = nn.Dropout(p=dropout_p)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)

        out = self.pool1(F.relu(self.bn1(self.conv1(x))))
        out = self.pool2(F.relu(self.bn2(self.conv2(out))))
        out = self.pool3(F.relu(self.bn3(self.conv3(out))))
        out = self.global_pool(F.relu(self.bn4(self.conv4(out))))

        out = torch.flatten(out, 1)
        out = self.dropout(F.relu(self.fc1(out)))
        logits = self.fc2(out)
        return logits


def preprocess_offline_audio(wav_path: Path, target_samples: int = 16000) -> np.ndarray:
    audio, sr = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)

    if len(audio) < target_samples:
        pad_right = target_samples - len(audio)
        audio = np.pad(audio, (0, pad_right), mode="constant", constant_values=0.0)
    elif len(audio) > target_samples:
        audio = audio[:target_samples]

    max_val = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.0
    if max_val > 1.0:
        audio = audio / max_val

    return audio.astype(np.float32)


# -----------------------------------------------------------------------------
# EVALUATION ENGINE
# -----------------------------------------------------------------------------
class KWSEvaluator:
    def __init__(self, checkpoint_path: Path, theta_cmd: float = DEFAULT_THRESHOLD, device: str = "cpu"):
        self.device = torch.device(device)
        self.theta_cmd = theta_cmd
        self.checkpoint_path = checkpoint_path

        self.frontend = LogMelFrontend().to(self.device)
        self.frontend.eval()

        self.model = CorrectedVietnameseKWSNet(num_classes=5).to(self.device)
        ckpt_data = torch.load(str(checkpoint_path), map_location=self.device)
        state_dict = ckpt_data["model_state_dict"] if isinstance(ckpt_data, dict) and "model_state_dict" in ckpt_data else ckpt_data
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict_clip(self, audio_1s: np.ndarray) -> Tuple[str, float, np.ndarray, float]:
        t0 = time.perf_counter()
        inp = torch.from_numpy(audio_1s).unsqueeze(0).to(self.device)
        with torch.no_grad():
            spec = self.frontend(inp)
            logits = self.model(spec)
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        dt_ms = (time.perf_counter() - t0) * 1000.0

        top_idx = int(np.argmax(probs))
        top_prob = float(probs[top_idx])

        if top_idx < 4 and top_prob >= self.theta_cmd:
            pred_class = TARGET_CLASSES[top_idx]
            conf = top_prob
        else:
            pred_class = "REJECT"
            conf = top_prob if top_idx == 4 else float(np.max(probs[:4]))

        return pred_class, conf, probs, dt_ms

    def evaluate_offline_manifest(self, manifest_csv: Path) -> Dict[str, Any]:
        df = pd.read_csv(manifest_csv)
        print(f"\n[Offline Evaluation] Evaluating {len(df)} clips from {manifest_csv.name}...")

        latencies = []
        records = []
        confusion_mat = np.zeros((5, 5), dtype=int)
        class_to_idx = {c: i for i, c in enumerate(CLASSES)}

        for idx, row in df.iterrows():
            wav_path = PROJECT_ROOT / row["audio_path"]
            audio = preprocess_offline_audio(wav_path)

            pred_class, conf, probs, dt_ms = self.predict_clip(audio)
            latencies.append(dt_ms)

            true_label = row["label"]
            ti = class_to_idx[true_label]
            pi = class_to_idx[pred_class]
            confusion_mat[ti, pi] += 1

            records.append({
                "sample_id": row["sample_id"],
                "true_label": true_label,
                "pred_class": pred_class,
                "confidence": round(conf, 4),
                "is_correct": bool(true_label == pred_class),
                "reject_subgroup": row.get("reject_subgroup", ""),
                "oov_lexeme": row.get("oov_lexeme", ""),
                "source_type": row.get("source_type", ""),
                "target_snr_db": row.get("target_snr_db", np.nan),
                "latency_ms": round(dt_ms, 2),
            })

        res_df = pd.DataFrame(records)

        # Per-class metrics
        per_class = {}
        for i, c in enumerate(CLASSES):
            tp = confusion_mat[i, i]
            fp = int(confusion_mat[:, i].sum() - tp)
            fn = int(confusion_mat[i, :].sum() - tp)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            per_class[c] = {
                "support": int(confusion_mat[i, :].sum()),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
            }

        macro_f1 = float(np.mean([per_class[c]["f1_score"] for c in CLASSES]))
        cmd_f1 = float(np.mean([per_class[c]["f1_score"] for c in TARGET_CLASSES]))
        overall_acc = float((confusion_mat.diagonal().sum()) / len(df))

        # Target command recall: clean vs noisy
        cmd_df = res_df[res_df["true_label"].isin(TARGET_CLASSES)]
        clean_cmds = cmd_df[cmd_df["source_type"].str.contains("clean", case=False, na=False)]
        noisy_cmds = cmd_df[cmd_df["source_type"].str.contains("noise_augmented", case=False, na=False)]

        clean_recall = float((clean_cmds["is_correct"]).mean()) if len(clean_cmds) > 0 else 0.0
        noisy_recall = float((noisy_cmds["is_correct"]).mean()) if len(noisy_cmds) > 0 else 0.0

        # SNR breakdown for target commands
        snr_breakdown = {}
        for snr_val in [20.0, 15.0, 10.0, 5.0, 0.0]:
            snr_sub = cmd_df[cmd_df["target_snr_db"] == snr_val]
            if len(snr_sub) > 0:
                acc_snr = float((snr_sub["is_correct"]).mean())
                snr_breakdown[f"{int(snr_val)}dB"] = {
                    "count": len(snr_sub),
                    "recall": round(acc_snr, 4),
                }

        # Negative breakdown (REJECT subgroups)
        rej_df = res_df[res_df["true_label"] == "REJECT"]
        rej_breakdown = {}
        for sg in ["oov_word", "conversation", "environmental_noise", "silence_non_speech"]:
            sub = rej_df[rej_df["reject_subgroup"] == sg]
            if len(sub) > 0:
                rej_rate = float((sub["pred_class"] == "REJECT").mean())
                fa_count = int((sub["pred_class"] != "REJECT").sum())
                rej_breakdown[sg] = {
                    "count": len(sub),
                    "rejection_rate": round(rej_rate, 4),
                    "false_acceptance_count": fa_count,
                    "false_acceptance_rate": round(1.0 - rej_rate, 4),
                }

        # OOV breakdown: known vs unseen (specifically 'ghe' and unseen test lexemes)
        oov_sub = rej_df[rej_df["reject_subgroup"] == "oov_word"]
        ghe_sub = oov_sub[oov_sub["oov_lexeme"].str.lower() == "ghế"]
        ghe_rej_rate = float((ghe_sub["pred_class"] == "REJECT").mean()) if len(ghe_sub) > 0 else 0.0
        other_unseen = oov_sub[oov_sub["oov_lexeme"].str.lower() != "ghế"]
        other_rej_rate = float((other_unseen["pred_class"] == "REJECT").mean()) if len(other_unseen) > 0 else 0.0

        rej_breakdown["oov_ghe_isolated"] = {
            "count": len(ghe_sub),
            "rejection_rate": round(ghe_rej_rate, 4),
            "false_accepts": int((ghe_sub["pred_class"] != "REJECT").sum()),
        }
        rej_breakdown["oov_other_unseen"] = {
            "count": len(other_unseen),
            "rejection_rate": round(other_rej_rate, 4),
            "false_accepts": int((other_unseen["pred_class"] != "REJECT").sum()),
        }

        # Latency statistics
        latencies_arr = np.array(latencies)
        latency_stats = {
            "mean_ms": round(float(np.mean(latencies_arr)), 2),
            "p50_ms": round(float(np.percentile(latencies_arr, 50)), 2),
            "p90_ms": round(float(np.percentile(latencies_arr, 90)), 2),
            "p95_ms": round(float(np.percentile(latencies_arr, 95)), 2),
        }

        print(f"  Overall Accuracy: {overall_acc * 100:.2f}%")
        print(f"  Macro F1:        {macro_f1:.4f} (Commands F1: {cmd_f1:.4f})")
        print(f"  Clean Recall:    {clean_recall * 100:.2f}% | Noisy Recall: {noisy_recall * 100:.2f}%")
        print(f"  Unseen 'ghế' Rejection Rate: {ghe_rej_rate * 100:.2f}% (FA: {rej_breakdown['oov_ghe_isolated']['false_accepts']}/{len(ghe_sub)})")
        print(f"  FLEURS Conv Rejection Rate:  {rej_breakdown['conversation']['rejection_rate'] * 100:.2f}% (FA: {rej_breakdown['conversation']['false_acceptance_count']}/{rej_breakdown['conversation']['count']})")
        print(f"  Latency P95:     {latency_stats['p95_ms']:.2f} ms")

        return {
            "manifest": str(manifest_csv),
            "total_samples": len(df),
            "overall_accuracy": round(overall_acc, 4),
            "macro_f1": round(macro_f1, 4),
            "command_macro_f1": round(cmd_f1, 4),
            "clean_command_recall": round(clean_recall, 4),
            "noisy_command_recall": round(noisy_recall, 4),
            "per_class": per_class,
            "confusion_matrix": confusion_mat.tolist(),
            "snr_breakdown": snr_breakdown,
            "reject_breakdown": rej_breakdown,
            "latency": latency_stats,
            "predictions": records,
        }

    def evaluate_streaming_corpus(self, streaming_manifest_csv: Path) -> Dict[str, Any]:
        df = pd.read_csv(streaming_manifest_csv)
        print(f"\n[Streaming Evaluation] Evaluating {len(df)} multi-command streams from {streaming_manifest_csv.name}...")

        total_target_events = 0
        total_detected_events = 0
        total_matched_events = 0
        total_missed_events = 0
        total_duplicate_events = 0
        total_false_insertions = 0
        order_errors = 0

        sequence_results = []

        for idx, row in df.iterrows():
            wav_path = PROJECT_ROOT / row["audio_path"]
            events_gt = json.loads(row["events_json"])
            target_gt = [e for e in events_gt if e.get("is_target", False)]
            total_target_events += len(target_gt)

            audio_stream, sr = sf.read(str(wav_path), dtype="float32")
            if sr != 16000:
                import soxr
                audio_stream = soxr.resample(audio_stream, sr, 16000)

            # Segmented detector emulation on stream (mimics live_mic_engine.py)
            detected_events = self._detect_stream_events(audio_stream)
            total_detected_events += len(detected_events)

            # Match detected events against ground truth target events
            matched_gt = set()
            seq_duplicates = 0
            seq_order_err = False
            seq_misses = 0
            seq_false_inserts = 0

            det_target_cmds = [d["command"] for d in detected_events]
            gt_target_cmds = [g["label"] for g in target_gt]

            # Greedy alignment with timestamp tolerance (0.55s)
            for d in detected_events:
                t_det = d["timestamp_sec"]
                cmd = d["command"]
                best_match = None
                best_dist = 999.0

                for gi, g in enumerate(target_gt):
                    if g["label"] == cmd:
                        dist = abs(t_det - ((g["start_sec"] + g["end_sec"]) / 2.0))
                        if dist < 0.65 and dist < best_dist:
                            best_dist = dist
                            best_match = gi

                if best_match is not None:
                    if best_match in matched_gt:
                        seq_duplicates += 1
                        total_duplicate_events += 1
                    else:
                        matched_gt.add(best_match)
                        total_matched_events += 1
                else:
                    seq_false_inserts += 1
                    total_false_insertions += 1

            seq_misses = len(target_gt) - len(matched_gt)
            total_missed_events += seq_misses

            # Check sequence order
            if det_target_cmds != gt_target_cmds:
                seq_order_err = True
                order_errors += 1

            sequence_results.append({
                "sequence_id": row["sequence_id"],
                "sequence_type": row["sequence_type"],
                "condition": row["condition"],
                "ground_truth": gt_target_cmds,
                "detected": det_target_cmds,
                "is_order_exact": bool(not seq_order_err),
                "target_count": len(target_gt),
                "matched_count": len(matched_gt),
                "missed_count": seq_misses,
                "duplicate_count": seq_duplicates,
                "false_insertion_count": seq_false_inserts,
            })

        seq_df = pd.DataFrame(sequence_results)
        exact_order_rate = float((seq_df["is_order_exact"]).mean())
        target_recall = float(total_matched_events / max(1, total_target_events))
        target_precision = float(total_matched_events / max(1, total_detected_events))

        # Breakdown by sequence type
        type_breakdown = {}
        for stype in df["sequence_type"].unique():
            sub = seq_df[seq_df["sequence_type"] == stype]
            type_breakdown[stype] = {
                "count": len(sub),
                "exact_sequence_accuracy": round(float((sub["is_order_exact"]).mean()), 4),
                "target_recall": round(float(sub["matched_count"].sum() / max(1, sub["target_count"].sum())), 4),
                "false_insertions": int(sub["false_insertion_count"].sum()),
            }

        # Breakdown by acoustic condition
        cond_breakdown = {}
        for cond in df["condition"].unique():
            sub = seq_df[seq_df["condition"] == cond]
            cond_breakdown[cond] = {
                "count": len(sub),
                "exact_sequence_accuracy": round(float((sub["is_order_exact"]).mean()), 4),
                "target_recall": round(float(sub["matched_count"].sum() / max(1, sub["target_count"].sum())), 4),
                "false_insertions": int(sub["false_insertion_count"].sum()),
            }

        print(f"  Exact Sequence Accuracy:    {exact_order_rate * 100:.2f}% ({int(seq_df['is_order_exact'].sum())}/{len(df)})")
        print(f"  Target Event Recall:        {target_recall * 100:.2f}% ({total_matched_events}/{total_target_events})")
        print(f"  Target Event Precision:     {target_precision * 100:.2f}% ({total_matched_events}/{total_detected_events})")
        print(f"  Missed Commands:            {total_missed_events}")
        print(f"  Duplicate Events:           {total_duplicate_events}")
        print(f"  False Insertions (from OOV): {total_false_insertions}")

        return {
            "total_sequences": len(df),
            "exact_sequence_accuracy": round(exact_order_rate, 4),
            "target_event_recall": round(target_recall, 4),
            "target_event_precision": round(target_precision, 4),
            "total_target_events": total_target_events,
            "total_detected_events": total_detected_events,
            "total_matched_events": total_matched_events,
            "total_missed_events": total_missed_events,
            "total_duplicate_events": total_duplicate_events,
            "total_false_insertions": total_false_insertions,
            "order_errors": order_errors,
            "by_sequence_type": type_breakdown,
            "by_condition": cond_breakdown,
            "sequences": sequence_results,
        }

    def _detect_stream_events(self, stream_audio: np.ndarray, chunk_ms: int = 20) -> List[Dict[str, Any]]:
        """
        Emulates live_mic_engine.py UtteranceSegmenter + Inferencer on offline audio.
        """
        chunk_len = int(16000 * (chunk_ms / 1000.0))  # 320 samples
        events = []

        preroll_len = int(16000 * 0.15)  # 150 ms preroll
        silence_thresh_len = int(16000 * 0.30)  # 300 ms silence
        energy_thresh_db = -48.0

        state = "WAITING"
        accumulated: List[np.ndarray] = []
        consec_silence = 0
        preroll_buf = np.zeros(preroll_len, dtype=np.float32)
        sample_cursor = 0

        for i in range(0, len(stream_audio), chunk_len):
            chunk = stream_audio[i : i + chunk_len]
            sample_cursor += len(chunk)
            rms = float(np.sqrt(np.mean(chunk ** 2))) + 1e-7
            db = 20.0 * np.log10(rms)

            is_active = (db >= energy_thresh_db)

            if state == "WAITING":
                # Maintain preroll
                n = len(chunk)
                if n >= preroll_len:
                    preroll_buf[:] = chunk[-preroll_len:]
                else:
                    preroll_buf[:-n] = preroll_buf[n:]
                    preroll_buf[-n:] = chunk

                if is_active:
                    state = "ACCUMULATING"
                    accumulated = [preroll_buf.copy(), chunk.copy()]
                    consec_silence = 0
            else:  # ACCUMULATING
                accumulated.append(chunk.copy())
                if not is_active:
                    consec_silence += len(chunk)
                else:
                    consec_silence = 0

                cur_samples = sum(len(c) for c in accumulated)
                if consec_silence >= silence_thresh_len or cur_samples >= 19200:  # 1.2s max
                    # Finalize segment
                    total_speech = np.concatenate(accumulated)
                    if len(total_speech) >= 1920:  # >= 120ms
                        # Format into 1.0s window right-padded with zeros
                        if len(total_speech) < 16000:
                            pad_right = 16000 - len(total_speech)
                            formatted_1s = np.pad(total_speech, (0, pad_right), mode="constant")
                        else:
                            formatted_1s = total_speech[:16000]

                        # Predict
                        pred_cmd, conf, _, _ = self.predict_clip(formatted_1s)
                        if pred_cmd in TARGET_CLASSES:
                            event_time = (sample_cursor - consec_silence) / 16000.0
                            events.append({
                                "command": pred_cmd,
                                "timestamp_sec": round(event_time, 3),
                                "confidence": round(conf, 4),
                            })

                    state = "WAITING"
                    accumulated = []
                    consec_silence = 0

        return events


def main():
    parser = argparse.ArgumentParser(description="Vietnamese KWS Model Evaluation Harness")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt")
    parser.add_argument("--test-manifest", type=str, default="experiments/vieneu_kws_research_v12/manifests/test_manifest.csv")
    parser.add_argument("--streaming-manifest", type=str, default="experiments/vieneu_kws_research_v12/manifests/streaming_eval_corpus_manifest.csv")
    parser.add_argument("--theta", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-json", type=str, required=True, help="Output JSON path for results")
    args = parser.parse_args()

    evaluator = KWSEvaluator(checkpoint_path=Path(args.checkpoint), theta_cmd=args.theta, device=args.device)

    # 1. Offline evaluation on locked test manifest
    offline_res = evaluator.evaluate_offline_manifest(Path(args.test_manifest))

    # 2. Streaming evaluation on multi-command corpus
    streaming_res = evaluator.evaluate_streaming_corpus(Path(args.streaming_manifest))

    full_results = {
        "checkpoint": str(args.checkpoint),
        "theta_cmd": args.theta,
        "device": args.device,
        "offline_evaluation": offline_res,
        "streaming_evaluation": streaming_res,
    }

    out_p = Path(args.output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)

    print(f"\nEvaluation completed. Results saved to: {out_p}")


if __name__ == "__main__":
    main()
