"""
Phase 45 Independent Audit Script.
Workspace: D:\trainidentificationaudio
Target: Audit v2 experiment outputs, checkpoints, predictions, data, and CTC implementation.
"""

import os
import sys
import json
import math
import hashlib
import numpy as np
import pandas as pd
import soundfile as sf
import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, BASE_DIR)

from src.models.kws_cnn import VietnameseKWSNet, get_model_param_count
from experiments.vieneu_kws_cascade_v1.train_verifier import TemporalKWSVerifier
from experiments.vieneu_kws_research_v2.models.ctc_kws.ctc_model import (
    VietnameseCharTokenizer,
    VietnameseCTCKWSNet,
    get_ctc_param_count
)
from src.preprocessing.feature_extractor import LogMelFeatureExtractor
from experiments.vieneu_kws_v2.evaluate import match_events_and_evaluate


def audit_architectures_and_checkpoints():
    print("--- 1. Auditing Architectures and Checkpoints ---")
    results = {}
    
    # 1. Stage 1 (VietnameseKWSNet)
    m1 = VietnameseKWSNet(num_classes=6)
    p1 = get_model_param_count(m1)
    results["stage1_param_count"] = p1
    
    # Checkpoint v2 Stage 1
    s1_ckpt_path = os.path.join(BASE_DIR, "experiments/vieneu_kws_research_v2/models/hard_negative_cascade/checkpoints/stage1_retrained_seed42.pt")
    s1_ckpt = torch.load(s1_ckpt_path, map_location="cpu", weights_only=False)
    m1.load_state_dict(s1_ckpt["model_state"])
    results["stage1_checkpoint_keys"] = list(s1_ckpt.keys())
    results["stage1_val_f1"] = s1_ckpt["val_macro_f1"]
    
    # 2. Stage 2 (TemporalKWSVerifier)
    m2 = TemporalKWSVerifier(num_classes=5)
    p2 = get_model_param_count(m2)
    results["stage2_param_count"] = p2
    results["stage2_has_tcn"] = hasattr(m2, "tcn")
    results["stage2_has_gru"] = any("gru" in name.lower() or "rnn" in name.lower() for name, _ in m2.named_modules())
    
    s2_ckpt_path = os.path.join(BASE_DIR, "experiments/vieneu_kws_research_v2/models/hard_negative_cascade/checkpoints/stage2_retrained_seed42.pt")
    s2_ckpt = torch.load(s2_ckpt_path, map_location="cpu", weights_only=False)
    m2.load_state_dict(s2_ckpt["model_state"])
    results["stage2_checkpoint_keys"] = list(s2_ckpt.keys())
    results["stage2_val_f1"] = s2_ckpt["val_macro_f1"]
    
    # 3. Candidate B CTC
    tok = VietnameseCharTokenizer()
    m_ctc = VietnameseCTCKWSNet(vocab_size=tok.vocab_size)
    p_ctc = get_ctc_param_count(m_ctc)
    results["ctc_param_count"] = p_ctc
    results["ctc_vocab_size"] = tok.vocab_size
    
    ctc_ckpt_path = os.path.join(BASE_DIR, "experiments/vieneu_kws_research_v2/models/ctc_kws/checkpoints/best_ctc_seed42.pt")
    ctc_ckpt = torch.load(ctc_ckpt_path, map_location="cpu", weights_only=False)
    m_ctc.load_state_dict(ctc_ckpt["model_state"])
    results["ctc_checkpoint_keys"] = list(ctc_ckpt.keys())
    results["ctc_val_keyword_acc"] = ctc_ckpt["val_keyword_accuracy"]
    
    return results


def audit_preprocessing_and_input_shapes():
    print("--- 2. Auditing Preprocessing & Shapes ---")
    results = {}
    extractor = LogMelFeatureExtractor(sample_rate=16000, n_mels=40, win_length=400, hop_length=160)
    
    # 1.0s window = 16000 samples
    wav_1s = torch.randn(2, 16000)
    feat_1s = extractor(wav_1s)
    results["feat_1s_shape"] = list(feat_1s.shape)  # Expected: [2, 40, 101]
    
    # Stage 1 forward
    m1 = VietnameseKWSNet(num_classes=6)
    out1 = m1(feat_1s)
    results["stage1_out_shape"] = list(out1.shape)  # Expected: [2, 6]
    
    # CTC forward
    tok = VietnameseCharTokenizer()
    m_ctc = VietnameseCTCKWSNet(vocab_size=tok.vocab_size)
    out_ctc = m_ctc(feat_1s)
    results["ctc_out_shape"] = list(out_ctc.shape)  # Expected: [101, 2, 91]
    
    # 1.2s window = 19200 samples
    wav_1_2s = torch.randn(2, 19200)
    feat_1_2s = extractor(wav_1_2s)
    results["feat_1_2s_shape"] = list(feat_1_2s.shape)  # Expected: [2, 40, 121]
    
    m2 = TemporalKWSVerifier(num_classes=5)
    out2 = m2(feat_1_2s)
    results["stage2_out_shape"] = list(out2.shape)  # Expected: [2, 5]
    
    return results


def audit_metric_recomputation():
    print("--- 3 & 4 & 5. Auditing Metric Recomputation and FA/hr ---")
    results = {}
    
    # Read saved predictions
    preds_csv = os.path.join(BASE_DIR, "experiments/vieneu_kws_research_v2/predictions/per_event_predictions.csv")
    preds_df = pd.read_csv(preds_csv)
    
    # Read stream durations and ground truths
    streams_dir = os.path.join(BASE_DIR, "experiments/vieneu_kws_research_v2/data/continuous_streams")
    
    stream_files = {
        "val_commands": ("val_stream_continuous_commands.wav", "val_stream_continuous_commands_ground_truth.json"),
        "val_nontarget": ("val_stream_continuous_nontarget.wav", "val_stream_continuous_nontarget_ground_truth.json"),
        "test_commands": ("test_stream_continuous_commands.wav", "test_stream_continuous_commands_ground_truth.json"),
        "test_nontarget": ("test_stream_continuous_nontarget.wav", "test_stream_continuous_nontarget_ground_truth.json"),
    }
    
    stream_info = {}
    for k, (wav_f, gt_f) in stream_files.items():
        w_path = os.path.join(streams_dir, wav_f)
        info = sf.info(w_path)
        with open(os.path.join(streams_dir, gt_f), "r", encoding="utf-8") as f:
            gt_data = json.load(f)
        stream_info[k] = {
            "duration_sec": info.duration,
            "duration_hours": info.duration / 3600.0,
            "gt_event_count": len(gt_data),
            "gt_df": pd.DataFrame(gt_data)
        }
    
    results["stream_info"] = {
        k: {"dur_sec": v["duration_sec"], "dur_hours": v["duration_hours"], "num_events": v["gt_event_count"]}
        for k, v in stream_info.items()
    }
    
    # Check saved test summary vs recomputed
    test_summary_p = os.path.join(BASE_DIR, "experiments/vieneu_kws_research_v2/predictions/locked_test_summary.json")
    with open(test_summary_p, "r", encoding="utf-8") as f:
        saved_test_summary = json.load(f)
        
    candidates = ["1_historical_cascade", "2_hard_negative_stage1", "3_hard_negative_cascade", "4_ctc_kws"]
    seeds = [42, 43, 44]
    
    recomputed = {}
    for cand in candidates:
        recomputed[cand] = {}
        for s in seeds:
            # Command stream
            cmd_preds = preds_df[(preds_df["candidate"] == cand) & (preds_df["seed"] == s) & (preds_df["stream"] == "test_commands")]
            cmd_triggers = cmd_preds[["trigger_time_sec", "pred_label", "confidence"]].to_dict(orient="records")
            cmd_eval = match_events_and_evaluate(cmd_triggers, stream_info["test_commands"]["gt_df"], stream_info["test_commands"]["duration_sec"])
            
            # Non-target stream
            nt_preds = preds_df[(preds_df["candidate"] == cand) & (preds_df["seed"] == s) & (preds_df["stream"] == "test_nontarget")]
            nt_triggers = nt_preds[["trigger_time_sec", "pred_label", "confidence"]].to_dict(orient="records")
            nt_eval = match_events_and_evaluate(nt_triggers, stream_info["test_nontarget"]["gt_df"], stream_info["test_nontarget"]["duration_sec"])
            
            saved_cmd = saved_test_summary[cand][f"seed{s}"]["command_stream"]
            saved_nt = saved_test_summary[cand][f"seed{s}"]["nontarget_stream"]
            
            # Check equality
            assert cmd_eval["recall"] == saved_cmd["recall"], f"Recall mismatch for {cand} seed {s}"
            assert cmd_eval["true_positives"] == saved_cmd["true_positives"]
            assert nt_eval["false_positives"] == saved_nt["false_positives"]
            assert abs(nt_eval["fa_per_hour"] - saved_nt["fa_per_hour"]) < 1e-4
            
            recomputed[cand][f"seed{s}"] = {
                "cmd_recall": cmd_eval["recall"],
                "nt_fp": nt_eval["false_positives"],
                "nt_fa_per_hr": nt_eval["fa_per_hour"],
                "nt_duration_hours": nt_eval["stream_duration_hours"]
            }
            
    results["recomputed_verified"] = True
    results["recomputed_metrics"] = recomputed
    return results


def audit_data_manifests_and_leakage():
    print("--- 6 & 7. Auditing Data Manifests and Leakage ---")
    data_dir = os.path.join(BASE_DIR, "experiments/vieneu_kws_research_v2/data")
    train_df = pd.read_csv(os.path.join(data_dir, "splits/train_manifest.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "splits/val_manifest.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "splits/test_manifest.csv"))
    text_inv_df = pd.read_csv(os.path.join(data_dir, "text_inventory.csv"))
    
    # 1. Voice disjointness
    train_voices = set(train_df["voice_id"].unique())
    val_voices = set(val_df["voice_id"].unique())
    test_voices = set(test_df["voice_id"].unique())
    
    voice_leak_tv = train_voices.intersection(val_voices)
    voice_leak_tt = train_voices.intersection(test_voices)
    voice_leak_vt = val_voices.intersection(test_voices)
    
    # 2. Non-target text disjointness
    train_nt_texts = set(train_df[train_df["class_label"] == "UNKNOWN"]["text"].unique())
    val_nt_texts = set(val_df[val_df["class_label"] == "UNKNOWN"]["text"].unique())
    test_nt_texts = set(test_df[test_df["class_label"] == "UNKNOWN"]["text"].unique())
    
    text_leak_tv = train_nt_texts.intersection(val_nt_texts)
    text_leak_tt = train_nt_texts.intersection(test_nt_texts)
    text_leak_vt = val_nt_texts.intersection(test_nt_texts)
    
    # 3. Audio file path overlap
    train_files = set(train_df["kws_wav_path"].unique())
    val_files = set(val_df["kws_wav_path"].unique())
    test_files = set(test_df["kws_wav_path"].unique())
    
    file_leak_tv = train_files.intersection(val_files)
    file_leak_tt = train_files.intersection(test_files)
    file_leak_vt = val_files.intersection(test_files)
    
    return {
        "num_train_voices": len(train_voices),
        "num_val_voices": len(val_voices),
        "num_test_voices": len(test_voices),
        "voice_leakage": {"train_val": list(voice_leak_tv), "train_test": list(voice_leak_tt), "val_test": list(voice_leak_vt)},
        "num_train_nt_texts": len(train_nt_texts),
        "num_val_nt_texts": len(val_nt_texts),
        "num_test_nt_texts": len(test_nt_texts),
        "text_leakage": {"train_val": list(text_leak_tv), "train_test": list(text_leak_tt), "val_test": list(text_leak_vt)},
        "file_path_leakage": {"train_val": len(file_leak_tv), "train_test": len(file_leak_tt), "val_test": len(file_leak_vt)}
    }


def audit_ctc_implementation():
    print("--- 10. Auditing CTC Implementation ---")
    tok = VietnameseCharTokenizer()
    m_ctc = VietnameseCTCKWSNet(vocab_size=tok.vocab_size)
    
    # Test valid target lengths and sequence lengths
    # Input has 101 frames -> CTC sequence length T = 101
    sample_text = "xuống"
    target_ids = tok.encode(sample_text)
    tgt_len = len(target_ids)
    
    # Verify input length > target length
    assert 101 > tgt_len, f"CTC sequence length 101 must be greater than target length {tgt_len}"
    
    # Forward pass
    dummy_wav = torch.randn(1, 40, 101)
    log_probs = m_ctc(dummy_wav)  # (T=101, B=1, Vocab=91)
    
    loss_fn = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    loss = loss_fn(
        log_probs,
        torch.tensor([target_ids], dtype=torch.long),
        torch.tensor([101], dtype=torch.long),
        torch.tensor([tgt_len], dtype=torch.long)
    )
    
    # Check repeated characters decoding in CTC
    # In CTC, if a character is repeated in sequence, e.g. "l - - - ê - - n", greedy decode collapses blanks and repeats
    sample_seq = [0, tok.char_to_id['l'], 0, tok.char_to_id['l'], 0]
    decoded = tok.decode(sample_seq)
    
    return {
        "ctc_loss_finite": bool(torch.isfinite(loss).item()),
        "ctc_loss_val": float(loss.item()),
        "token_count": tok.vocab_size,
        "blank_idx": 0,
        "sample_repeated_char_decode": decoded,
        "sample_text_decoded": tok.decode(tok.encode(sample_text))
    }


def main():
    report = {}
    report["architectures"] = audit_architectures_and_checkpoints()
    report["preprocessing"] = audit_preprocessing_and_input_shapes()
    report["metrics"] = audit_metric_recomputation()
    report["data_leakage"] = audit_data_manifests_and_leakage()
    report["ctc_audit"] = audit_ctc_implementation()
    
    out_p = os.path.join(os.path.dirname(__file__), "audit_findings.json")
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nAudit complete. Summary saved to {out_p}")


if __name__ == "__main__":
    main()
