#!/usr/bin/env python3
"""
Phase 51: Locked Confirmatory Evaluation Harness (gTTS + ESC-50 Fold 2)
Repository: D:\\trainidentificationaudio
Target: experiments/vieneu_kws_research_v12/scripts/eval_phase51_confirmatory_test.py

Evaluates:
- Baseline v11 (default model)
- Baseline E1
- Phase 28 D2_H1 Candidate (Pre-registered Low-FA profile)
- Phase 26 Candidate D0_H1 seed44 (Pre-registered Balanced-Sensitivity profile)
- Phase 28 D0_H1 Candidate (Exploratory)

Under:
- Track A: Isolated 4-Command Classification (240 primary clips + 48 tone probes)
- Track B: Keyword Detection in Context / Continuous Audio (50 sequences, 230 events)

Strict Protocol Invariants:
- Fixed frontend: LogMelFrontend (40 mels, win=400, hop=160, n_fft=512)
- Fixed VAD: SpeechBandRatioVAD (300-3400 Hz, ratio >= 0.32, energy >= -46 dBFS)
- Fixed decision threshold: theta_cmd = 0.20
- Fixed capacity: 138,469 parameters
"""

import sys
import io
import time
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from scipy import stats

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANIFESTS_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "manifests"
REPORTS_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "reports"

sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "scripts"))
from train_phase28_factorial import CorrectedVietnameseKWSNet, LogMelFrontend, CLASSES, CLASS_TO_IDX, compute_sha256

ISOLATED_MANIFEST = MANIFESTS_DIR / "PHASE50_LOCKED_MANIFEST.csv"
TONE_PROBE_MANIFEST = MANIFESTS_DIR / "PHASE50_TONE_PROBE_MANIFEST.csv"
STREAMING_MANIFEST = MANIFESTS_DIR / "PHASE50_STREAMING_MANIFEST.csv"
EVENT_LEDGER_CSV = MANIFESTS_DIR / "PHASE50_EVENT_LEDGER.csv"
EVENT_LEDGER_JSON = MANIFESTS_DIR / "PHASE50_EVENT_LEDGER.json"

MODELS_CONFIG = {
    "Baseline_v11_seed42": {
        "path": PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4" / "bundle" / "checkpoint" / "variant_v11_hard_negative_weighted_seed42.pt",
        "description": "Canonical v11 default model (Phase 13.4)",
        "expected_hash": "3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2",
        "role": "planned_baseline"
    },
    "Baseline_E1_seed42": {
        "path": PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "checkpoints" / "exp_v12_e1_baseline_scale_seed42.pt",
        "description": "Phase 18 E1 baseline scale model",
        "expected_hash": "f3c0fe8f7ccebb8fe354cb50acb247e0674232044706db0187feb92b1706b878",
        "role": "planned_baseline"
    },
    "Phase28_D2_H1_Candidate": {
        "path": PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "candidates" / "phase28_d2_candidate" / "candidate_checkpoint.pt",
        "description": "Pre-registered Candidate: Low False-Alarm Profile",
        "expected_hash": "c6102b4876ddeaa453547c211c5cfc322511c957e18fd488a705a164b18601c8",
        "role": "preregistered_candidate_low_fa"
    },
    "Phase26_Candidate_D0_H1_seed44": {
        "path": PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "candidates" / "phase26_candidate_d0_h1_seed44" / "candidate_checkpoint.pt",
        "description": "Pre-registered Candidate: Balanced Sensitivity Profile",
        "expected_hash": "00b4fc71f863788648d5ea388a99c4cbd4441d9b9f63adc5e56154e329df63f3",
        "role": "preregistered_candidate_balanced"
    },
    "Phase28_D0_H1_Candidate": {
        "path": PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "candidates" / "phase28_candidate" / "candidate_checkpoint.pt",
        "description": "Phase 28 D0_H1 Candidate (seed42)",
        "expected_hash": "84a3457f105058bb7ca843b830013d00cb6d2e9ce737cb389267a84fcf51a6af",
        "role": "exploratory"
    }
}


class SpeechBandRatioVAD:
    def __init__(self, sample_rate=16000, speech_band_hz=(300.0, 3400.0), min_speech_band_ratio=0.32, energy_threshold_db=-46.0):
        self.sr = sample_rate
        self.f_min, self.f_max = speech_band_hz
        self.min_ratio = min_speech_band_ratio
        self.energy_thresh = 10.0 ** (energy_threshold_db / 10.0)

    def is_speech(self, frame: np.ndarray) -> bool:
        if len(frame) == 0:
            return False
        energy = np.mean(frame ** 2)
        if energy < self.energy_thresh:
            return False

        w = np.hanning(len(frame))
        fft_vals = np.abs(np.fft.rfft(frame * w)) ** 2
        freqs = np.fft.rfftfreq(len(frame), d=1.0 / self.sr)

        tot_p = np.sum(fft_vals)
        if tot_p <= 1e-12:
            return False

        band_mask = (freqs >= self.f_min) & (freqs <= self.f_max)
        band_p = np.sum(fft_vals[band_mask])
        ratio = band_p / tot_p
        return ratio >= self.min_ratio


def load_model(ckpt_path: Path, device: torch.device) -> CorrectedVietnameseKWSNet:
    model = CorrectedVietnameseKWSNet(num_classes=5, dropout_p=0.30).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def evaluate_isolated(model: torch.nn.Module, frontend: torch.nn.Module, df_iso: pd.DataFrame, device: torch.device) -> Dict[str, Any]:
    preds = []
    targets = []

    for idx, row in df_iso.iterrows():
        wav_path = PROJECT_ROOT / row["audio_path"]
        audio, sr = sf.read(str(wav_path), dtype="float32")
        if len(audio) < 16000:
            audio = np.pad(audio, (0, 16000 - len(audio)), mode="constant")
        elif len(audio) > 16000:
            audio = audio[:16000]

        wav_t = torch.from_numpy(audio).unsqueeze(0).to(device)
        with torch.no_grad():
            spec = frontend(wav_t)
            logits = model(spec)
            probs = F.softmax(logits, dim=-1)[0]
            target_probs = probs[:4]
            max_p, max_idx = torch.max(target_probs, dim=-1)
            pred = int(max_idx.item()) if max_p.item() >= 0.20 else 4

        preds.append(pred)
        targets.append(CLASS_TO_IDX[row["label"]])

    y_true = np.array(targets)
    y_pred = np.array(preds)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2, 3, 4], average=None, zero_division=0)

    macro_f1 = float(np.mean(f1[:4]))
    overall_macro_f1 = float(np.mean(f1))
    acc = float(np.mean(y_true == y_pred))

    oov_mask = (y_true == 4)
    oov_fa_count = int(np.sum(y_pred[oov_mask] != 4))
    oov_total = int(np.sum(oov_mask))
    oov_fa_rate = oov_fa_count / oov_total if oov_total > 0 else 0.0

    phonetic_mask = (df_iso["subgroup"] == "test_oov_phonetic").values
    phonetic_fa = int(np.sum(y_pred[phonetic_mask] != 4))
    phonetic_total = int(np.sum(phonetic_mask))

    general_mask = (df_iso["subgroup"] == "test_oov_general").values
    general_fa = int(np.sum(y_pred[general_mask] != 4))
    general_total = int(np.sum(general_mask))

    # SNR breakdown
    snr_breakdown = {}
    for snr in ["clean", "15.0", "10.0", "5.0"]:
        snr_m = (df_iso["target_snr_db"] == snr).values
        if np.sum(snr_m) > 0:
            yt_s = y_true[snr_m]
            yp_s = y_pred[snr_m]
            _, _, f1_s, _ = precision_recall_fscore_support(yt_s, yp_s, labels=[0, 1, 2, 3], average=None, zero_division=0)
            snr_breakdown[str(snr)] = {
                "macro_f1": float(np.mean(f1_s)),
                "accuracy": float(np.mean(yt_s == yp_s)),
                "oov_fa_rate": float(np.mean(yp_s[yt_s == 4] != 4)) if np.sum(yt_s == 4) > 0 else 0.0
            }

    return {
        "macro_f1": macro_f1,
        "overall_macro_f1": overall_macro_f1,
        "accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "per_class": {
            CLASSES[i]: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f1[i]),
                "support": int(s[i])
            }
            for i in range(5)
        },
        "oov_total": oov_total,
        "oov_fa_count": oov_fa_count,
        "oov_fa_rate": oov_fa_rate,
        "phonetic_oov_total": phonetic_total,
        "phonetic_oov_fa_count": phonetic_fa,
        "phonetic_oov_fa_rate": phonetic_fa / phonetic_total if phonetic_total > 0 else 0.0,
        "general_oov_total": general_total,
        "general_oov_fa_count": general_fa,
        "general_oov_fa_rate": general_fa / general_total if general_total > 0 else 0.0,
        "snr_breakdown": snr_breakdown,
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist()
    }


def evaluate_tone_probes(model: torch.nn.Module, frontend: torch.nn.Module, df_probe: pd.DataFrame, device: torch.device) -> Dict[str, Any]:
    preds = []
    for idx, row in df_probe.iterrows():
        wav_path = PROJECT_ROOT / row["audio_path"]
        audio, _ = sf.read(str(wav_path), dtype="float32")
        if len(audio) < 16000:
            audio = np.pad(audio, (0, 16000 - len(audio)), mode="constant")
        elif len(audio) > 16000:
            audio = audio[:16000]

        wav_t = torch.from_numpy(audio).unsqueeze(0).to(device)
        with torch.no_grad():
            spec = frontend(wav_t)
            logits = model(spec)
            probs = F.softmax(logits, dim=-1)[0]
            target_probs = probs[:4]
            max_p, max_idx = torch.max(target_probs, dim=-1)
            pred = int(max_idx.item()) if max_p.item() >= 0.20 else 4
        preds.append(pred)

    y_pred = np.array(preds)
    fa_count = int(np.sum(y_pred != 4))
    total = len(y_pred)
    fa_rate = fa_count / total if total > 0 else 0.0
    ucb95_clip = (1.0 - (1.0 - 0.95) ** (1.0 / total)) if fa_count == 0 else (fa_rate + 1.96 * np.sqrt(fa_rate * (1 - fa_rate) / total))
    ucb95_inst = (1.0 - (1.0 - 0.95) ** (1.0 / 12)) if fa_count == 0 else None

    return {
        "probe_total": total,
        "probe_fa_count": fa_count,
        "probe_fa_rate": fa_rate,
        "probe_ucb95_clip_far": float(ucb95_clip),
        "probe_ucb95_instance_far": float(ucb95_inst) if ucb95_inst else None
    }


def evaluate_streaming(model: torch.nn.Module, frontend: torch.nn.Module, df_stream: pd.DataFrame, events_dict: Dict[str, List], device: torch.device) -> Dict[str, Any]:
    vad = SpeechBandRatioVAD(speech_band_hz=(300.0, 3400.0), min_speech_band_ratio=0.32, energy_threshold_db=-46.0)

    exact_matches = 0
    total_seqs = len(df_stream)

    total_gt_cmds = 0
    total_detected_cmds = 0
    true_positive_cmds = 0

    substitutions = 0
    false_insertions = 0
    duplicates = 0
    premature_cuts = 0

    silence_seq_fa = 0
    total_silence_duration_s = 0.0

    oov_seq_fa = 0
    total_oov_duration_s = 0.0

    latencies = []
    step_samples = 1600
    win_samples = 16000

    for idx, row in df_stream.iterrows():
        seq_id = row["sequence_id"]
        seq_type = row["sequence_type"]
        wav_path = PROJECT_ROOT / row["audio_path"]
        audio, _ = sf.read(str(wav_path), dtype="float32")
        seq_len = len(audio)
        dur_s = seq_len / 16000.0

        gt_events = events_dict.get(seq_id, [])
        gt_cmds = [ev for ev in gt_events if ev["label"] != "REJECT"]

        if seq_type == "pure_silence_noise":
            total_silence_duration_s += dur_s
        elif seq_type == "pure_oov_conversation":
            total_oov_duration_s += dur_s

        detected_events = []
        last_trigger_pos = -999999

        for pos in range(0, seq_len - win_samples + 1, step_samples):
            if (pos - last_trigger_pos) < 8000:
                continue

            frame = audio[pos : pos + win_samples]
            recent_frame = frame[-3200:]
            if not vad.is_speech(recent_frame):
                continue

            wav_t = torch.from_numpy(frame).unsqueeze(0).to(device)
            with torch.no_grad():
                spec = frontend(wav_t)
                logits = model(spec)
                probs = F.softmax(logits, dim=-1)[0]
                target_probs = probs[:4]
                max_p, max_idx = torch.max(target_probs, dim=-1)

                if max_p.item() >= 0.20:
                    cmd_label = CLASSES[max_idx.item()]
                    trigger_time = (pos + win_samples) / 16000.0
                    detected_events.append({
                        "label": cmd_label,
                        "time": trigger_time,
                        "prob": float(max_p.item())
                    })
                    last_trigger_pos = pos

        detected_cmd_seq = [d["label"] for d in detected_events]
        gt_cmd_seq = [g["label"] for g in gt_cmds]

        is_exact = (detected_cmd_seq == gt_cmd_seq)
        if is_exact:
            exact_matches += 1

        if seq_type == "pure_silence_noise":
            silence_seq_fa += len(detected_events)
        elif seq_type == "pure_oov_conversation":
            oov_seq_fa += len(detected_events)

        total_gt_cmds += len(gt_cmds)
        total_detected_cmds += len(detected_events)

        matched_gt = set()
        matched_det = set()

        for d_i, det in enumerate(detected_events):
            best_match = None
            for g_i, gt in enumerate(gt_cmds):
                if (gt["start_time"] - 0.2) <= det["time"] <= (gt["end_time"] + 0.6):
                    best_match = (g_i, gt)
                    break

            if best_match is not None:
                g_i, gt = best_match
                if det["label"] == gt["label"]:
                    if g_i not in matched_gt:
                        matched_gt.add(g_i)
                        matched_det.add(d_i)
                        true_positive_cmds += 1
                        lat = max(0.0, det["time"] - gt["start_time"])
                        latencies.append(lat)
                        if lat < 0.15:
                            premature_cuts += 1
                    else:
                        duplicates += 1
                else:
                    substitutions += 1
            else:
                false_insertions += 1

    cmd_recall = true_positive_cmds / total_gt_cmds if total_gt_cmds > 0 else 0.0
    cmd_precision = true_positive_cmds / total_detected_cmds if total_detected_cmds > 0 else 0.0
    cmd_f1 = 2 * cmd_precision * cmd_recall / (cmd_precision + cmd_recall) if (cmd_precision + cmd_recall) > 0 else 0.0

    silence_min = total_silence_duration_s / 60.0
    silence_fa_per_min = silence_seq_fa / silence_min if silence_min > 0 else 0.0
    silence_ucb95 = (3.0 / silence_min) if (silence_seq_fa == 0 and silence_min > 0) else (silence_fa_per_min + 1.96 * np.sqrt(silence_fa_per_min / silence_min))

    oov_min = total_oov_duration_s / 60.0
    oov_fa_per_min = oov_seq_fa / oov_min if oov_min > 0 else 0.0
    oov_ucb95 = (3.0 / oov_min) if (oov_seq_fa == 0 and oov_min > 0) else (oov_fa_per_min + 1.96 * np.sqrt(oov_fa_per_min / oov_min))

    p50_lat = float(np.median(latencies)) * 1000.0 if len(latencies) > 0 else 0.0
    p95_lat = float(np.percentile(latencies, 95)) * 1000.0 if len(latencies) > 0 else 0.0

    return {
        "total_sequences": total_seqs,
        "exact_sequence_matches": exact_matches,
        "exact_sequence_accuracy": exact_matches / total_seqs if total_seqs > 0 else 0.0,
        "total_gt_commands": total_gt_cmds,
        "total_detected_commands": total_detected_cmds,
        "true_positive_commands": true_positive_cmds,
        "command_recall": cmd_recall,
        "command_precision": cmd_precision,
        "command_f1": cmd_f1,
        "error_taxonomy": {
            "misses": total_gt_cmds - true_positive_cmds,
            "substitutions": substitutions,
            "false_insertions": false_insertions,
            "duplicates": duplicates,
            "premature_cuts": premature_cuts
        },
        "silence_duration_min": round(silence_min, 3),
        "silence_false_alarms": silence_seq_fa,
        "silence_fa_per_min": round(silence_fa_per_min, 2),
        "silence_ucb95_fa_per_min": round(silence_ucb95, 2),
        "oov_duration_min": round(oov_min, 3),
        "oov_false_alarms": oov_seq_fa,
        "oov_fa_per_min": round(oov_fa_per_min, 2),
        "oov_ucb95_fa_per_min": round(oov_ucb95, 2),
        "p50_latency_ms": round(p50_lat, 1),
        "p95_latency_ms": round(p95_lat, 1)
    }


def main():
    print("=" * 80)
    print("PHASE 51: LOCKED CONFIRMATORY EVALUATION ON INDEPENDENT gTTS + ESC-50 FOLD 2")
    print("=" * 80)

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load manifests
    df_iso = pd.read_csv(ISOLATED_MANIFEST)
    df_probe = pd.read_csv(TONE_PROBE_MANIFEST)
    df_stream = pd.read_csv(STREAMING_MANIFEST)
    with open(EVENT_LEDGER_JSON, "r", encoding="utf-8") as f:
        events_dict = json.load(f)

    print(f"Loaded Track A Isolated Manifest: {len(df_iso)} clips (SHA-256: {compute_sha256(ISOLATED_MANIFEST)})")
    print(f"Loaded Track A Tone Probe Manifest: {len(df_probe)} clips (SHA-256: {compute_sha256(TONE_PROBE_MANIFEST)})")
    print(f"Loaded Track B Streaming Manifest: {len(df_stream)} sequences (SHA-256: {compute_sha256(STREAMING_MANIFEST)})")

    # Load models
    models = {}
    checkpoint_hashes = {}
    frontend = LogMelFrontend(sample_rate=16000, n_mels=40, win_length=400, hop_length=160, n_fft=512).to(device)
    frontend.eval()

    print("\n--- Verifying Frozen Checkpoints ---")
    for m_name, m_cfg in MODELS_CONFIG.items():
        actual_hash = compute_sha256(m_cfg["path"])
        checkpoint_hashes[m_name] = actual_hash
        assert actual_hash == m_cfg["expected_hash"], f"Hash mismatch for {m_name}"
        print(f"  [VERIFIED] {m_name:30s}: {actual_hash} ({m_cfg['role']})")
        models[m_name] = load_model(m_cfg["path"], device)

    # 1. Evaluate Track A: Isolated Performance
    print("\n--- 1. Evaluating Track A: Isolated Classification (240 Clips) ---")
    isolated_results = {}
    for m_name, model in models.items():
        res = evaluate_isolated(model, frontend, df_iso, device)
        isolated_results[m_name] = res
        print(f"  {m_name:30s}: 4-Cmd F1: {res['macro_f1']*100:.2f}% | 5-Class F1: {res['overall_macro_f1']*100:.2f}% | Ph-FAR: {res['phonetic_oov_fa_rate']*100:.2f}% | Gen-FAR: {res['general_oov_fa_rate']*100:.2f}%")

    # 2. Evaluate Track A: Tone Probes
    print("\n--- 2. Evaluating Track A: Tone Probes (48 Clips) ---")
    probe_results = {}
    for m_name, model in models.items():
        p_res = evaluate_tone_probes(model, frontend, df_probe, device)
        probe_results[m_name] = p_res
        print(f"  {m_name:30s}: Probe FAR: {p_res['probe_fa_rate']*100:.2f}% ({p_res['probe_fa_count']}/{p_res['probe_total']}) | UCB95 Clip: {p_res['probe_ucb95_clip_far']*100:.2f}%")

    # 3. Evaluate Track B: Streaming Evaluation
    print("\n--- 3. Evaluating Track B: Streaming Continuous Audio (50 Sequences) ---")
    streaming_results = {}
    for m_name, model in models.items():
        s_res = evaluate_streaming(model, frontend, df_stream, events_dict, device)
        streaming_results[m_name] = s_res
        print(f"  {m_name:30s}: Exact Seq: {s_res['exact_sequence_matches']}/50 ({s_res['exact_sequence_accuracy']*100:.1f}%) | Ev-F1: {s_res['command_f1']*100:.2f}% | Sil-FA/m: {s_res['silence_fa_per_min']} | OOV-FA/m: {s_res['oov_fa_per_min']}")

    # 4. Dependency-Aware Paired Clustered Comparisons on Track A
    print("\n--- 4. Computing Clustered Statistical Comparisons on Track A (N=60 Synthesis Instances) ---")
    df_iso_copy = df_iso.copy()
    clustered_accs = {}
    for m_name in MODELS_CONFIG.keys():
        y_true = np.array(isolated_results[m_name]["y_true"])
        y_pred = np.array(isolated_results[m_name]["y_pred"])
        df_iso_copy["correct"] = (y_true == y_pred).astype(float)
        clustered_accs[m_name] = df_iso_copy.groupby("synthesis_instance_id")["correct"].mean()

    df_clustered_trackA = pd.DataFrame(clustered_accs)

    paired_comparisons = {}
    pairs = [
        ("Phase28_D2_H1_Candidate", "Baseline_v11_seed42"),
        ("Phase28_D2_H1_Candidate", "Baseline_E1_seed42"),
        ("Phase28_D2_H1_Candidate", "Phase26_Candidate_D0_H1_seed44"),
        ("Phase26_Candidate_D0_H1_seed44", "Baseline_E1_seed42")
    ]

    for m1, m2 in pairs:
        diffs = df_clustered_trackA[m1] - df_clustered_trackA[m2]
        mean_diff = float(diffs.mean())
        se = float(diffs.std(ddof=1) / np.sqrt(len(diffs)))
        t_stat = mean_diff / se if se > 0 else 0.0
        p_val_t = float(stats.t.sf(abs(t_stat), df=len(diffs)-1) * 2)
        ci95 = [float(mean_diff - 2.001 * se), float(mean_diff + 2.001 * se)]

        paired_comparisons[f"{m1}_vs_{m2}"] = {
            "mean_delta_pp": mean_diff * 100.0,
            "ci95_pp": [ci95[0] * 100.0, ci95[1] * 100.0],
            "t_statistic": t_stat,
            "p_value_t": p_val_t,
            "n_clusters": len(diffs)
        }
        print(f"  {m1} vs {m2}: Mean Delta={mean_diff*100:+.2f} pp (95% CI: [{ci95[0]*100:+.2f}, {ci95[1]*100:+.2f}]), p={p_val_t:.6e}")

    # 5. Pre-Registered Engineering Criteria Validation
    print("\n--- 5. Pre-Registered Confirmatory Gate Audit ---")
    # Low-FA Candidate: D2_H1
    d2_sil_fa = streaming_results["Phase28_D2_H1_Candidate"]["silence_fa_per_min"]
    d2_oov_fa = streaming_results["Phase28_D2_H1_Candidate"]["oov_fa_per_min"]
    d2_iso_oov = isolated_results["Phase28_D2_H1_Candidate"]["oov_fa_rate"]
    d2_macro_f1 = isolated_results["Phase28_D2_H1_Candidate"]["macro_f1"]

    pass_d2_low_fa = (
        d2_sil_fa <= 1.0 and
        d2_oov_fa <= 6.0 and
        d2_iso_oov <= 0.20 and
        d2_macro_f1 >= 0.50
    )

    # Balanced Candidate: Phase 26
    p26_macro_f1 = isolated_results["Phase26_Candidate_D0_H1_seed44"]["macro_f1"]
    p26_event_f1 = streaming_results["Phase26_Candidate_D0_H1_seed44"]["command_f1"]
    p26_sil_fa = streaming_results["Phase26_Candidate_D0_H1_seed44"]["silence_fa_per_min"]
    p26_iso_oov = isolated_results["Phase26_Candidate_D0_H1_seed44"]["oov_fa_rate"]

    pass_p26_balanced = (
        p26_macro_f1 >= 0.60 and
        p26_event_f1 >= 0.55 and
        p26_sil_fa <= 3.0 and
        p26_iso_oov <= 0.25
    )

    print(f"Candidate Profile A (Low-FA Profile, D2_H1):")
    print(f"  Silence FA/min: {d2_sil_fa} <= 1.0 -> {d2_sil_fa <= 1.0}")
    print(f"  OOV FA/min:     {d2_oov_fa} <= 6.0 -> {d2_oov_fa <= 6.0}")
    print(f"  Isolated OOV:   {d2_iso_oov*100:.2f}% <= 20.0% -> {d2_iso_oov <= 0.20}")
    print(f"  Macro-F1:       {d2_macro_f1*100:.2f}% >= 50.0% -> {d2_macro_f1 >= 0.50}")
    print(f"  Low-FA Profile Pass: {pass_d2_low_fa}")

    print(f"\nCandidate Profile B (Balanced Sensitivity, Phase 26):")
    print(f"  Macro-F1:       {p26_macro_f1*100:.2f}% >= 60.0% -> {p26_macro_f1 >= 0.60}")
    print(f"  Streaming Ev-F1:{p26_event_f1*100:.2f}% >= 55.0% -> {p26_event_f1 >= 0.55}")
    print(f"  Silence FA/min: {p26_sil_fa} <= 3.0 -> {p26_sil_fa <= 3.0}")
    print(f"  Isolated OOV:   {p26_iso_oov*100:.2f}% <= 25.0% -> {p26_iso_oov <= 0.25}")
    print(f"  Balanced Profile Pass: {pass_p26_balanced}")

    # Remove raw array dumps before serializing JSON
    iso_clean = {}
    for m_name, res in isolated_results.items():
        res_copy = dict(res)
        res_copy.pop("y_true", None)
        res_copy.pop("y_pred", None)
        iso_clean[m_name] = res_copy

    out_master_dict = {
        "metadata": {
            "evaluation_phase": "Phase 51 (Locked Confirmatory Evaluation)",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "speech_generator": "Google Translate Vietnamese TTS (gTTS 2.5.4)",
            "noise_background": "Karol Piczak ESC-50 Fold 2 Indoor Tracks",
            "trackA_manifest_sha256": compute_sha256(ISOLATED_MANIFEST),
            "trackA_tone_probe_manifest_sha256": compute_sha256(TONE_PROBE_MANIFEST),
            "trackB_streaming_manifest_sha256": compute_sha256(STREAMING_MANIFEST),
            "trackB_event_ledger_sha256": compute_sha256(EVENT_LEDGER_CSV),
            "checkpoint_hashes": checkpoint_hashes,
            "governance_labels": [
                "EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED",
                "NOT_VALIDATED_ON_NATURAL_SPEECH",
                "NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION",
                "NOT_VALIDATED_PENDING_DATA"
            ]
        },
        "trackA_isolated_results": iso_clean,
        "trackA_tone_probe_results": probe_results,
        "trackB_streaming_results": streaming_results,
        "trackA_clustered_paired_comparisons": paired_comparisons,
        "candidate_criteria_audit": {
            "d2_h1_low_fa_profile": {
                "passed": pass_d2_low_fa,
                "silence_fa_per_min": d2_sil_fa,
                "oov_fa_per_min": d2_oov_fa,
                "isolated_oov_far": d2_iso_oov,
                "trackA_macro_f1": d2_macro_f1
            },
            "phase26_balanced_profile": {
                "passed": pass_p26_balanced,
                "trackA_macro_f1": p26_macro_f1,
                "streaming_event_f1": p26_event_f1,
                "silence_fa_per_min": p26_sil_fa,
                "isolated_oov_far": p26_iso_oov
            }
        }
    }

    out_json = REPORTS_DIR / "PHASE51_FINAL_EVALUATION_RESULTS.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out_master_dict, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Saved PHASE51_FINAL_EVALUATION_RESULTS.json (SHA-256: {compute_sha256(out_json)})")

    elapsed = time.time() - t0
    print(f"[DONE] Confirmatory evaluation completed in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
