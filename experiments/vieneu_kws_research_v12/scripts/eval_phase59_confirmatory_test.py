#!/usr/bin/env python3
"""
Phase 59: Locked Confirmatory Evaluation Runner (Piper VAIS-1000 + ESC-50 Fold 3)
Repository: D:\\trainidentificationaudio
Target: experiments/vieneu_kws_research_v12/scripts/eval_phase59_confirmatory_test.py

Evaluates the 5 pre-registered models:
1. Baseline_v11_seed42 (Legacy default)
2. Baseline_E1_seed42 (Frozen baseline)
3. Phase28_D2_H1_Candidate (Primary Low-FA Candidate)
4. Phase26_Candidate_D0_H1_seed44 (Primary Balanced Candidate)
5. Candidate_D3_C_seed42 (Registered Developmental Challenger)

Under locked theta_cmd = 0.20, fixed Log-Mel frontend, REJECT-inclusive 5-class argmax.
"""

import sys
import os
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from scipy import stats
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANIFESTS_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "manifests"
REPORTS_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "reports"

ISOLATED_MANIFEST = MANIFESTS_DIR / "PHASE58_CONFIRMATORY_MANIFEST.csv"
TONE_PROBE_MANIFEST = MANIFESTS_DIR / "PHASE58_TONE_PROBE_MANIFEST.csv"
STREAMING_MANIFEST = MANIFESTS_DIR / "PHASE58_STREAMING_MANIFEST.csv"
EVENT_LEDGER_CSV = MANIFESTS_DIR / "PHASE58_EVENT_LEDGER.csv"
EVENT_LEDGER_JSON = MANIFESTS_DIR / "PHASE58_EVENT_LEDGER.json"

CLASSES = ["TRAI", "PHAI", "LEN", "XUONG", "REJECT"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


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
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

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


class SpeechBandRatioVAD:
    def __init__(self, sample_rate: int = 16000, speech_band_hz: Tuple[float, float] = (300.0, 3400.0), min_speech_band_ratio: float = 0.32, energy_threshold_db: float = -46.0):
        self.sample_rate = sample_rate
        self.f_low, self.f_high = speech_band_hz
        self.min_ratio = min_speech_band_ratio
        self.energy_threshold = 10.0 ** (energy_threshold_db / 10.0)

    def is_speech(self, frame_samples: np.ndarray) -> bool:
        energy = np.mean(frame_samples ** 2)
        if energy < self.energy_threshold:
            return False
        spectrum = np.abs(np.fft.rfft(frame_samples * np.hanning(len(frame_samples)))) ** 2
        freqs = np.fft.rfftfreq(len(frame_samples), 1.0 / self.sample_rate)
        band_mask = (freqs >= self.f_low) & (freqs <= self.f_high)
        band_energy = np.sum(spectrum[band_mask])
        total_energy = np.sum(spectrum)
        if total_energy <= 1e-12:
            return False
        return (band_energy / total_energy) >= self.min_ratio


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_frozen_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    model = CorrectedVietnameseKWSNet(num_classes=5, dropout_p=0.0).to(device)
    raw = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        st = raw["model_state_dict"]
    elif isinstance(raw, dict) and "state_dict" in raw:
        st = raw["state_dict"]
    elif isinstance(raw, dict):
        st = raw
    else:
        st = raw.state_dict()
    clean_st = {k.replace("module.", ""): v for k, v in st.items()}
    model.load_state_dict(clean_st, strict=True)
    model.eval()
    return model


# -----------------------------------------------------------------------------
# EVALUATION FUNCTIONS
# -----------------------------------------------------------------------------
def evaluate_isolated_trackA(model: nn.Module, frontend: nn.Module, df_manifest: pd.DataFrame, device: torch.device) -> Dict[str, Any]:
    preds = []
    trues = []

    for idx, row in df_manifest.iterrows():
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
        trues.append(CLASS_TO_IDX[row["label"]])

    y_true = np.array(trues)
    y_pred = np.array(preds)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4]).tolist()
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2, 3, 4], average=None, zero_division=0)

    macro_f1 = float(np.mean(f1[:4]))
    overall_macro_f1 = float(np.mean(f1))
    acc = float(np.mean(y_true == y_pred))

    target_mask = (y_true != 4)
    target_correct = int(np.sum(y_true[target_mask] == y_pred[target_mask]))
    target_accuracy = float(target_correct / np.sum(target_mask))

    oov_mask = (y_true == 4)
    oov_total = int(np.sum(oov_mask))
    oov_fa_count = int(np.sum(y_pred[oov_mask] != 4))
    oov_fa_rate = float(oov_fa_count / oov_total) if oov_total > 0 else 0.0

    phon_mask = (df_manifest["subgroup"] == "test_oov_phonetic").to_numpy()
    phon_total = int(np.sum(phon_mask))
    phon_fa = int(np.sum(y_pred[phon_mask] != 4))

    gen_mask = (df_manifest["subgroup"] == "test_oov_general").to_numpy()
    gen_total = int(np.sum(gen_mask))
    gen_fa = int(np.sum(y_pred[gen_mask] != 4))

    return {
        "macro_f1": macro_f1,
        "overall_macro_f1": overall_macro_f1,
        "accuracy": acc,
        "target_total": int(np.sum(target_mask)),
        "target_correct": target_correct,
        "target_accuracy": target_accuracy,
        "confusion_matrix": cm,
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
        "phonetic_oov_total": phon_total,
        "phonetic_oov_fa_count": phon_fa,
        "phonetic_oov_fa_rate": float(phon_fa / phon_total) if phon_total > 0 else 0.0,
        "general_oov_total": gen_total,
        "general_oov_fa_count": gen_fa,
        "general_oov_fa_rate": float(gen_fa / gen_total) if gen_total > 0 else 0.0,
        "y_true": y_true,
        "y_pred": y_pred
    }


def evaluate_tone_probes(model: nn.Module, frontend: nn.Module, df_probe: pd.DataFrame, device: torch.device) -> Dict[str, Any]:
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
    ucb95_inst = (1.0 - (1.0 - 0.95) ** (1.0 / 15)) if fa_count == 0 else None

    return {
        "probe_total": total,
        "probe_fa_count": fa_count,
        "probe_fa_rate": fa_rate,
        "probe_ucb95_clip_far": float(ucb95_clip),
        "probe_ucb95_instance_far": float(ucb95_inst) if ucb95_inst else None
    }


def evaluate_streaming(model: nn.Module, frontend: nn.Module, df_stream: pd.DataFrame, events_dict: Dict[str, List], device: torch.device) -> Dict[str, Any]:
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
        for d_i, det in enumerate(detected_events):
            best_match = None
            for g_i, gt in enumerate(gt_cmds):
                if (gt["start_time"] - 0.2) <= det["time"] <= (gt["end_time"] + 0.6):
                    best_match = (g_i, gt)
                    break

            if best_match is not None:
                g_i, gt = best_match
                if g_i not in matched_gt:
                    matched_gt.add(g_i)
                    if det["label"] == gt["label"]:
                        true_positive_cmds += 1
                    else:
                        substitutions += 1
                else:
                    duplicates += 1
            else:
                false_insertions += 1

    cmd_recall = true_positive_cmds / total_gt_cmds if total_gt_cmds > 0 else 0.0
    cmd_precision = true_positive_cmds / total_detected_cmds if total_detected_cmds > 0 else 0.0
    cmd_f1 = (2 * cmd_precision * cmd_recall) / (cmd_precision + cmd_recall) if (cmd_precision + cmd_recall) > 0 else 0.0

    sil_min = total_silence_duration_s / 60.0
    sil_fa_per_min = silence_seq_fa / sil_min if sil_min > 0 else 0.0
    sil_ucb95 = (silence_seq_fa + 1.96 * np.sqrt(max(1, silence_seq_fa))) / sil_min if sil_min > 0 else 0.0

    oov_min = total_oov_duration_s / 60.0
    oov_fa_per_min = oov_seq_fa / oov_min if oov_min > 0 else 0.0
    oov_ucb95 = (oov_seq_fa + 1.96 * np.sqrt(max(1, oov_seq_fa))) / oov_min if oov_min > 0 else 0.0

    return {
        "total_sequences": total_seqs,
        "exact_sequence_matches": exact_matches,
        "exact_sequence_accuracy": exact_matches / total_seqs,
        "total_gt_commands": total_gt_cmds,
        "total_detected_commands": total_detected_cmds,
        "true_positive_commands": true_positive_cmds,
        "command_recall": cmd_recall,
        "command_precision": cmd_precision,
        "command_f1": cmd_f1,
        "error_taxonomy": {
            "misses": total_gt_cmds - true_positive_cmds - substitutions,
            "substitutions": substitutions,
            "false_insertions": false_insertions,
            "duplicates": duplicates
        },
        "silence_duration_min": round(sil_min, 3),
        "silence_false_alarms": silence_seq_fa,
        "silence_fa_per_min": round(sil_fa_per_min, 2),
        "silence_ucb95_fa_per_min": round(sil_ucb95, 2),
        "oov_duration_min": round(oov_min, 3),
        "oov_false_alarms": oov_seq_fa,
        "oov_fa_per_min": round(oov_fa_per_min, 2),
        "oov_ucb95_fa_per_min": round(oov_ucb95, 2)
    }


def main():
    print("=" * 80)
    print("PHASE 59: LOCKED CONFIRMATORY EVALUATION (PIPER VAIS-1000 + ESC-50 FOLD 3)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}")

    # Checkpoint configurations
    models_to_eval = {
        "Baseline_v11_seed42": {
            "path": PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4" / "bundle" / "checkpoint" / "variant_v11_hard_negative_weighted_seed42.pt",
            "sha256": "3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2"
        },
        "Baseline_E1_seed42": {
            "path": PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "checkpoints" / "exp_v12_e1_baseline_scale_seed42.pt",
            "sha256": "f3c0fe8f7ccebb8fe354cb50acb247e0674232044706db0187feb92b1706b878"
        },
        "Phase28_D2_H1_Candidate": {
            "path": PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "candidates" / "phase28_d2_candidate" / "candidate_checkpoint.pt",
            "sha256": "c6102b4876ddeaa453547c211c5cfc322511c957e18fd488a705a164b18601c8"
        },
        "Phase26_Candidate_D0_H1_seed44": {
            "path": PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "candidates" / "phase26_candidate_d0_h1_seed44" / "candidate_checkpoint.pt",
            "sha256": "00b4fc71f863788648d5ea388a99c4cbd4441d9b9f63adc5e56154e329df63f3"
        },
        "Candidate_D3_C_seed42": {
            "path": PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "training_phase54_d3" / "D3_C_seed42" / "best_checkpoint.pt",
            "sha256": "da5b414328c33644b07c0667c3f9926637baa5f1672119f86393182c53c38ba6"
        }
    }

    models = {}
    checkpoint_hashes = {}
    frontend = LogMelFrontend().to(device).eval()

    print("\nVerifying Checkpoint Hashes:")
    for m_name, spec in models_to_eval.items():
        p = spec["path"]
        assert p.exists(), f"Missing checkpoint: {p}"
        h = compute_sha256(p)
        assert h == spec["sha256"], f"Hash mismatch on {m_name}: {h} != {spec['sha256']}"
        models[m_name] = load_frozen_model(p, device)
        checkpoint_hashes[m_name] = h
        print(f"  [OK] {m_name:<30s}: {h[:16]}... verified.")

    # Load Phase 58 Manifests
    df_trackA = pd.read_csv(ISOLATED_MANIFEST)
    df_probe = pd.read_csv(TONE_PROBE_MANIFEST)
    df_stream = pd.read_csv(STREAMING_MANIFEST)

    with open(EVENT_LEDGER_JSON, "r", encoding="utf-8") as f:
        events_dict = json.load(f)

    print(f"\nLoaded Test Manifests:")
    print(f"  Track A Primary:     {len(df_trackA)} clips")
    print(f"  Track A Tone Probes: {len(df_probe)} clips")
    print(f"  Track B Streaming:   {len(df_stream)} sequences")

    # 1. Evaluate Track A Isolated
    print("\n" + "=" * 80)
    print("RUNNING TRACK A ISOLATED EVALUATION (420 CLIPS)")
    print("=" * 80)

    trackA_results = {}
    for m_name, model in models.items():
        res = evaluate_isolated_trackA(model, frontend, df_trackA, device)
        trackA_results[m_name] = res
        print(f"\nModel: {m_name}")
        print(f"  Overall Accuracy:      {res['accuracy']*100:.2f}% ({int(res['accuracy']*len(df_trackA))}/{len(df_trackA)})")
        print(f"  4-Command Macro-F1:    {res['macro_f1']*100:.2f}%")
        print(f"  Target-Only Accuracy:  {res['target_accuracy']*100:.2f}% ({res['target_correct']}/{res['target_total']})")
        print(f"  OOV False Alarm Rate:  {res['oov_fa_rate']*100:.2f}% ({res['oov_fa_count']}/{res['oov_total']})")
        print(f"  Phonetic OOV FAR:      {res['phonetic_oov_fa_rate']*100:.2f}% ({res['phonetic_oov_fa_count']}/{res['phonetic_oov_total']})")
        print(f"  General OOV FAR:       {res['general_oov_fa_rate']*100:.2f}% ({res['general_oov_fa_count']}/{res['general_oov_total']})")
        print(f"  Per-Class Recall:      TRAI: {res['per_class']['TRAI']['recall']*100:.1f}%, PHAI: {res['per_class']['PHAI']['recall']*100:.1f}%, LEN: {res['per_class']['LEN']['recall']*100:.1f}%, XUONG: {res['per_class']['XUONG']['recall']*100:.1f}%, REJECT: {res['per_class']['REJECT']['recall']*100:.1f}%")

    # 2. Evaluate Track A Tone Probes
    print("\n" + "=" * 80)
    print("RUNNING TRACK A TONE PROBE EVALUATION (60 CLIPS)")
    print("=" * 80)

    probe_results = {}
    for m_name, model in models.items():
        p_res = evaluate_tone_probes(model, frontend, df_probe, device)
        probe_results[m_name] = p_res
        print(f"Model: {m_name:<30s} | FA: {p_res['probe_fa_count']:2d}/{p_res['probe_total']} ({p_res['probe_fa_rate']*100:.2f}%) | UCB95 Clip: {p_res['probe_ucb95_clip_far']*100:.2f}% | UCB95 Inst: {p_res['probe_ucb95_instance_far']*100 if p_res['probe_ucb95_instance_far'] else 'N/A'}")

    # 3. Evaluate Track B Streaming Continuous Audio
    print("\n" + "=" * 80)
    print("RUNNING TRACK B STREAMING EVALUATION (50 SEQUENCES, 58 EVENTS)")
    print("=" * 80)

    streaming_results = {}
    for m_name, model in models.items():
        s_res = evaluate_streaming(model, frontend, df_stream, events_dict, device)
        streaming_results[m_name] = s_res
        print(f"\nModel: {m_name}")
        print(f"  Exact Sequence Matches:  {s_res['exact_sequence_matches']}/{s_res['total_sequences']} ({s_res['exact_sequence_accuracy']*100:.1f}%)")
        print(f"  Command Event F1:        {s_res['command_f1']*100:.2f}% (Recall: {s_res['command_recall']*100:.1f}%, Precision: {s_res['command_precision']*100:.1f}%)")
        print(f"  Silence FA/min:          {s_res['silence_fa_per_min']} FA/min (UCB95: {s_res['silence_ucb95_fa_per_min']})")
        print(f"  Conversational OOV FA/min:{s_res['oov_fa_per_min']} FA/min (UCB95: {s_res['oov_ucb95_fa_per_min']})")

    # 4. Dependency-Aware Paired Clustered Inference (105 Primary Clean Waveforms)
    print("\n" + "=" * 80)
    print("DEPENDENCY-AWARE CLUSTERED PAIRED INFERENCE (N_inst = 105)")
    print("=" * 80)

    inst_ids = sorted(list(df_trackA["synthesis_instance_id"].unique()))
    assert len(inst_ids) == 105, f"Expected 105 instance clusters, got {len(inst_ids)}"

    cluster_scores = {m: [] for m in models}
    for inst_id in inst_ids:
        c_mask = (df_trackA["synthesis_instance_id"] == inst_id).to_numpy()
        for m_name in models:
            y_t = trackA_results[m_name]["y_true"][c_mask]
            y_p = trackA_results[m_name]["y_pred"][c_mask]
            acc_i = float(np.mean(y_t == y_p))
            cluster_scores[m_name].append(acc_i)

    paired_comparisons = {}
    comp_pairs = [
        ("Candidate_D3_C_seed42", "Baseline_v11_seed42"),
        ("Candidate_D3_C_seed42", "Phase28_D2_H1_Candidate"),
        ("Candidate_D3_C_seed42", "Phase26_Candidate_D0_H1_seed44"),
        ("Phase28_D2_H1_Candidate", "Baseline_v11_seed42"),
        ("Phase26_Candidate_D0_H1_seed44", "Baseline_v11_seed42")
    ]

    for m_a, m_b in comp_pairs:
        diffs = np.array(cluster_scores[m_a]) - np.array(cluster_scores[m_b])
        mean_diff = float(np.mean(diffs) * 100)
        t_stat, p_val = stats.ttest_rel(cluster_scores[m_a], cluster_scores[m_b])
        ci95 = stats.t.interval(0.95, df=len(diffs)-1, loc=np.mean(diffs), scale=stats.sem(diffs))
        ci95_pp = [float(ci95[0] * 100), float(ci95[1] * 100)]

        paired_comparisons[f"{m_a}_vs_{m_b}"] = {
            "mean_delta_pp": mean_diff,
            "ci95_pp": ci95_pp,
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "n_clusters": len(diffs)
        }
        print(f"{m_a} vs {m_b}:")
        print(f"  Mean Delta: {mean_diff:+.2f} pp | 95% CI: [{ci95_pp[0]:+.2f}, {ci95_pp[1]:+.2f}] | t = {t_stat:.4f} | p = {p_val:.4e}")

    # 5. Pre-Registered Engineering Criteria Audit
    print("\n" + "=" * 80)
    print("AUDIT OF PRE-REGISTERED ENGINEERING CRITERIA")
    print("=" * 80)

    # Candidate 1: Low-FA Profile (D2_H1)
    d2_sil_fa = streaming_results["Phase28_D2_H1_Candidate"]["silence_fa_per_min"]
    d2_oov_fa = streaming_results["Phase28_D2_H1_Candidate"]["oov_fa_per_min"]
    d2_iso_oov = trackA_results["Phase28_D2_H1_Candidate"]["oov_fa_rate"]
    d2_macro_f1 = trackA_results["Phase28_D2_H1_Candidate"]["macro_f1"]
    pass_d2_low_fa = bool(d2_sil_fa <= 1.0 and d2_oov_fa <= 6.0 and d2_iso_oov <= 0.20 and d2_macro_f1 >= 0.50)

    print(f"\n1. Low-FA Profile (Phase 28 D2_H1):")
    print(f"  Silence FA/min:  {d2_sil_fa} <= 1.0 -> {d2_sil_fa <= 1.0}")
    print(f"  OOV FA/min:      {d2_oov_fa} <= 6.0 -> {d2_oov_fa <= 6.0}")
    print(f"  Isolated OOV:    {d2_iso_oov*100:.2f}% <= 20.0% -> {d2_iso_oov <= 0.20}")
    print(f"  Macro-F1:        {d2_macro_f1*100:.2f}% >= 50.0% -> {d2_macro_f1 >= 0.50}")
    print(f"  Verdict:         {'PASS' if pass_d2_low_fa else 'FAIL'}")

    # Candidate 2: Balanced Sensitivity Profile (Phase 26)
    p26_macro_f1 = trackA_results["Phase26_Candidate_D0_H1_seed44"]["macro_f1"]
    p26_event_f1 = streaming_results["Phase26_Candidate_D0_H1_seed44"]["command_f1"]
    p26_sil_fa = streaming_results["Phase26_Candidate_D0_H1_seed44"]["silence_fa_per_min"]
    p26_iso_oov = trackA_results["Phase26_Candidate_D0_H1_seed44"]["oov_fa_rate"]
    pass_p26_balanced = bool(p26_macro_f1 >= 0.60 and p26_event_f1 >= 0.50 and p26_sil_fa <= 3.0 and p26_iso_oov <= 0.25)

    print(f"\n2. Balanced Sensitivity Profile (Phase 26 seed44):")
    print(f"  Macro-F1:        {p26_macro_f1*100:.2f}% >= 60.0% -> {p26_macro_f1 >= 0.60}")
    print(f"  Streaming Ev-F1: {p26_event_f1*100:.2f}% >= 50.0% -> {p26_event_f1 >= 0.50}")
    print(f"  Silence FA/min:  {p26_sil_fa} <= 3.0 -> {p26_sil_fa <= 3.0}")
    print(f"  Isolated OOV:    {p26_iso_oov*100:.2f}% <= 25.0% -> {p26_iso_oov <= 0.25}")
    print(f"  Verdict:         {'PASS' if pass_p26_balanced else 'FAIL'}")

    # Candidate 3: Developmental Challenger Profile (D3_C)
    d3_macro_f1 = trackA_results["Candidate_D3_C_seed42"]["macro_f1"]
    d3_min_rec = min(trackA_results["Candidate_D3_C_seed42"]["per_class"][c]["recall"] for c in CLASSES[:4])
    d3_event_f1 = streaming_results["Candidate_D3_C_seed42"]["command_f1"]
    d3_sil_fa = streaming_results["Candidate_D3_C_seed42"]["silence_fa_per_min"]
    d3_iso_oov = trackA_results["Candidate_D3_C_seed42"]["oov_fa_rate"]
    pass_d3_challenger = bool(d3_macro_f1 >= 0.60 and d3_min_rec >= 0.40 and d3_event_f1 >= 0.50 and d3_sil_fa <= 2.0 and d3_iso_oov <= 0.25)

    print(f"\n3. Developmental Challenger Profile (D3_C seed42):")
    print(f"  Macro-F1:        {d3_macro_f1*100:.2f}% >= 60.0% -> {d3_macro_f1 >= 0.60}")
    print(f"  Min Cmd Recall:  {d3_min_rec*100:.2f}% >= 40.0% -> {d3_min_rec >= 0.40}")
    print(f"  Streaming Ev-F1: {d3_event_f1*100:.2f}% >= 50.0% -> {d3_event_f1 >= 0.50}")
    print(f"  Silence FA/min:  {d3_sil_fa} <= 2.0 -> {d3_sil_fa <= 2.0}")
    print(f"  Isolated OOV:    {d3_iso_oov*100:.2f}% <= 25.0% -> {d3_iso_oov <= 0.25}")
    print(f"  Verdict:         {'PASS' if pass_d3_challenger else 'FAIL'}")

    # Compile CSV table
    csv_rows = []
    for m_name in models:
        t_res = trackA_results[m_name]
        p_res = probe_results[m_name]
        s_res = streaming_results[m_name]

        csv_rows.append({
            "model_name": m_name,
            "trackA_accuracy": round(t_res["accuracy"] * 100, 2),
            "trackA_macro_f1": round(t_res["macro_f1"] * 100, 2),
            "trackA_5class_f1": round(t_res["overall_macro_f1"] * 100, 2),
            "target_accuracy": round(t_res["target_accuracy"] * 100, 2),
            "trai_recall": round(t_res["per_class"]["TRAI"]["recall"] * 100, 2),
            "phai_recall": round(t_res["per_class"]["PHAI"]["recall"] * 100, 2),
            "len_recall": round(t_res["per_class"]["LEN"]["recall"] * 100, 2),
            "xuong_recall": round(t_res["per_class"]["XUONG"]["recall"] * 100, 2),
            "oov_far": round(t_res["oov_fa_rate"] * 100, 2),
            "phonetic_oov_far": round(t_res["phonetic_oov_fa_rate"] * 100, 2),
            "general_oov_far": round(t_res["general_oov_fa_rate"] * 100, 2),
            "tone_probe_far": round(p_res["probe_fa_rate"] * 100, 2),
            "stream_exact_match_pct": round(s_res["exact_sequence_accuracy"] * 100, 2),
            "stream_command_f1": round(s_res["command_f1"] * 100, 2),
            "silence_fa_per_min": s_res["silence_fa_per_min"],
            "oov_fa_per_min": s_res["oov_fa_per_min"]
        })

    df_csv = pd.DataFrame(csv_rows)
    metrics_csv_path = REPORTS_DIR / "PHASE59_CONFIRMATORY_METRICS.csv"
    df_csv.to_csv(metrics_csv_path, index=False)
    print(f"\nSaved Confirmatory Metrics CSV to {metrics_csv_path}")

    # Remove non-serializable raw numpy arrays from JSON
    clean_trackA = {}
    for m_name, res in trackA_results.items():
        res_c = dict(res)
        res_c.pop("y_true", None)
        res_c.pop("y_pred", None)
        clean_trackA[m_name] = res_c

    master_results = {
        "metadata": {
            "phase": "Phase 59 (Locked Confirmatory Evaluation)",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "speech_generator": "Piper TTS (vi_VN-vais1000-medium.onnx, IEEE DataPort VAIS-1000)",
            "noise_background": "Karol Piczak ESC-50 Fold 3 Indoor Tracks",
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
        "trackA_isolated_results": clean_trackA,
        "trackA_tone_probe_results": probe_results,
        "trackB_streaming_results": streaming_results,
        "paired_clustered_comparisons": paired_comparisons,
        "candidate_criteria_audit": {
            "d2_h1_low_fa_profile": {"passed": pass_d2_low_fa, "silence_fa": d2_sil_fa, "oov_fa": d2_oov_fa, "iso_oov": d2_iso_oov, "macro_f1": d2_macro_f1},
            "p26_balanced_profile": {"passed": pass_p26_balanced, "macro_f1": p26_macro_f1, "event_f1": p26_event_f1, "silence_fa": p26_sil_fa, "iso_oov": p26_iso_oov},
            "d3_challenger_profile": {"passed": pass_d3_challenger, "macro_f1": d3_macro_f1, "min_rec": d3_min_rec, "event_f1": d3_event_f1, "silence_fa": d3_sil_fa, "iso_oov": d3_iso_oov}
        }
    }

    out_json_path = REPORTS_DIR / "PHASE59_FINAL_EVALUATION_RESULTS.json"
    out_json_path.write_text(json.dumps(master_results, indent=2), encoding="utf-8")
    print(f"Saved Master Evaluation JSON to {out_json_path}")
    print("\nPhase 59 Locked Confirmatory Evaluation completed successfully!")


if __name__ == "__main__":
    main()
