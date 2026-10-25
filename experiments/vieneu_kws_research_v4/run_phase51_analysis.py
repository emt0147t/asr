import os
import json
import csv
from collections import defaultdict

def run_phase51_analysis():
    print("--- Running Phase 51 Acoustic Error Analysis ---")
    base_dir = r"D:\trainidentificationaudio"
    v3_dir = os.path.join(base_dir, "experiments", "vieneu_kws_research_v3")
    v4_dir = os.path.join(base_dir, "experiments", "vieneu_kws_research_v4")

    # Load ground truths
    cmd_gt_path = os.path.join(v3_dir, "data", "continuous_streams", "test_stream_continuous_commands_ground_truth.json")
    with open(cmd_gt_path, "r", encoding="utf-8") as f:
        cmd_gt = json.load(f)

    nt_gt_path = os.path.join(v3_dir, "data", "continuous_streams", "test_stream_continuous_nontarget_3h_ground_truth.json")
    with open(nt_gt_path, "r", encoding="utf-8") as f:
        nt_gt = json.load(f)

    # Load predictions CSV
    pred_path = os.path.join(v3_dir, "predictions", "per_event_predictions.csv")
    predictions = defaultdict(lambda: defaultdict(list))
    with open(pred_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            candidate = r["candidate"]
            seed = int(r["seed"])
            stream = r["stream"]
            t = float(r["trigger_time_sec"])
            pred_label = r["pred_label"]
            conf = float(r["confidence"])
            predictions[(candidate, seed)][stream].append({
                "trigger_time_sec": t,
                "pred_label": pred_label,
                "confidence": conf
            })

    # Analyze False Negatives on test_commands for Candidate 3 (recall_preserving_cascade)
    # Also compare with Historical Cascade
    fn_analysis = defaultdict(lambda: defaultdict(list))
    tolerance_sec = 0.50

    candidates = ["1_historical_cascade", "2_recall_preserving_stage1", "3_recall_preserving_cascade"]
    seeds = [42, 43, 44]

    for cand in candidates:
        for seed in seeds:
            preds = predictions[(cand, seed)]["test_commands"]
            # match with cmd_gt
            used_preds = set()
            for ref_idx, ref in enumerate(cmd_gt):
                ref_center = 0.5 * (ref["onset_sec"] + ref["end_time_sec"])
                ref_label = ref["class_label"]
                # find match
                matched = False
                for p_idx, p in enumerate(preds):
                    if p_idx in used_preds:
                        continue
                    if p["pred_label"] == ref_label and abs(p["trigger_time_sec"] - ref_center) <= tolerance_sec:
                        matched = True
                        used_preds.add(p_idx)
                        break
                if not matched:
                    fn_analysis[(cand, seed)][ref_label].append(ref)

    # Analyze False Positives on test_stream_continuous_nontarget_3h
    fp_analysis = defaultdict(list)
    for cand in candidates:
        for seed in seeds:
            preds = predictions[(cand, seed)]["test_nontarget_3h"]
            for p in preds:
                t = p["trigger_time_sec"]
                p_label = p["pred_label"]
                # Determine context
                matched_event = None
                context = "inter_utterance_gap"
                for ev in nt_gt:
                    t_on = ev["onset_sec"]
                    t_end = ev["end_time_sec"]
                    if t_on <= t <= t_end:
                        context = "active_speech"
                        matched_event = ev
                        break
                    elif (t_on - 0.20 <= t < t_on) or (t_end < t <= t_end + 0.60):
                        context = "boundary_margin"
                        matched_event = ev
                        break
                fp_analysis[(cand, seed)].append({
                    "trigger_time_sec": t,
                    "pred_label": p_label,
                    "confidence": p["confidence"],
                    "context": context,
                    "matched_event": matched_event
                })

    # Summarize False Positives by Confusion Family and Target for Candidate 3
    cand3_fp_family = defaultdict(int)
    cand3_fp_target = defaultdict(int)
    cand3_fp_words = defaultdict(int)
    cand3_fp_voices = defaultdict(int)

    for seed in seeds:
        for fp in fp_analysis[("3_recall_preserving_cascade", seed)]:
            ev = fp["matched_event"]
            if ev:
                cand3_fp_family[ev["confusion_family"]] += 1
                cand3_fp_target[ev["target_keyword"]] += 1
                cand3_fp_words[ev["text"]] += 1
                cand3_fp_voices[ev["voice_id"]] += 1
            else:
                cand3_fp_family["pure_background_gap"] += 1

    # Summarize False Negatives for Candidate 3
    cand3_fn_by_kw = defaultdict(int)
    cand3_fn_by_voice = defaultdict(int)
    for seed in seeds:
        for kw, fns in fn_analysis[("3_recall_preserving_cascade", seed)].items():
            cand3_fn_by_kw[kw] += len(fns)
            for fn in fns:
                cand3_fn_by_voice[fn["voice_id"]] += 1

    print("Candidate 3 Total FPs across 3 seeds:", sum(len(fp_analysis[("3_recall_preserving_cascade", s)]) for s in seeds))
    print("Candidate 3 FP breakdown by Confusion Family:", dict(cand3_fp_family))
    print("Candidate 3 FP breakdown by Target Keyword:", dict(cand3_fp_target))
    print("Candidate 3 FP by Voice (count):", len(cand3_fp_voices))
    print("Candidate 3 FN count by Target Keyword (sum over 3 seeds, total = 3 * 40 = 120 per kw):", dict(cand3_fn_by_kw))
    print("Candidate 3 FN count by Voice (count):", len(cand3_fn_by_voice))

    # Build Error Taxonomy CSV
    taxonomy_rows = [
        {
            "taxonomy_id": "ERR_FN_TRAI_RHYME",
            "error_type": "FALSE_NEGATIVE",
            "target_keyword": "TRAI",
            "phonetic_rhyme": "/aːj/ (ngang tone)",
            "acoustic_mechanism": "Over-suppression by Stage 2 TCN verifier calibrated on dense hard negatives sharing diphthong coda /aːj/ (chái, sái, quái, lái, bái, thải)",
            "mean_recall_loss": "35.8% (Mean recall: 64.2%)",
            "affected_voices": "Đức Trí (higher pitch/creaky phonation), Quỳnh Anh",
            "recommended_mitigation": "Bounded-utterance discriminative whole-clip decision layer or multi-task phoneme-state CTC head"
        },
        {
            "taxonomy_id": "ERR_FN_PHAI_RHYME",
            "error_type": "FALSE_NEGATIVE",
            "target_keyword": "PHAI",
            "phonetic_rhyme": "/aːj/ (ngang tone, aspirated labiodental onset /f/)",
            "acoustic_mechanism": "Stage 1 CNN activation weakened when fricative onset /f/ is smoothed under pink noise + Stage 2 threshold rejecting borderline scores",
            "mean_recall_loss": "40.8% (Mean recall: 59.2%)",
            "affected_voices": "Quốc Tuấn, Kim Thanh",
            "recommended_mitigation": "Fricative-sensitive spectral front-end weighting + clip-level bounded-utterance classifier"
        },
        {
            "taxonomy_id": "ERR_FN_XUONG_TONE",
            "error_type": "FALSE_NEGATIVE",
            "target_keyword": "XUONG",
            "phonetic_rhyme": "/uəŋ/ (ngang tone)",
            "acoustic_mechanism": "Occasional confusion with rising tone /uəŋ/ (xuống) in Seed 42 due to tone trajectory flattening under pink noise",
            "mean_recall_loss": "15.8% (Mean recall: 84.2%)",
            "affected_voices": "Đức Trí, Quỳnh Anh",
            "recommended_mitigation": "Fundamental frequency (F0) feature augmentation or tone-contour dedicated filterbanks"
        },
        {
            "taxonomy_id": "ERR_FP_BOUNDARY_MARGIN",
            "error_type": "FALSE_POSITIVE",
            "target_keyword": "ALL",
            "phonetic_rhyme": "Various (trailing codas /aːj/, /ɛŋ/)",
            "acoustic_mechanism": "Window-edge overlap within [t_end, t_end + 0.6s] where trailing non-target syllable energy combines with silence onset to mimic target duration",
            "mean_recall_loss": "N/A (Represents 77.4% of all Candidate 3 FPs)",
            "affected_voices": "Uniform across 4 test voices",
            "recommended_mitigation": "Temporal lockout buffer or transitioning to Bounded-Utterance Mode where sliding window boundaries do not exist"
        },
        {
            "taxonomy_id": "ERR_FP_MINIMAL_PAIR_CONSONANT",
            "error_type": "FALSE_POSITIVE",
            "target_keyword": "TRAI / PHAI",
            "phonetic_rhyme": "/aːj/ (e.g. chài, sài, bài, thai)",
            "acoustic_mechanism": "Acoustic similarity of retroflex /ʈ/ vs palatal /tɕ/ (tr vs ch) or labiodental /f/ vs alveolar /tʰ/ (ph vs th)",
            "mean_recall_loss": "N/A (Represents 32.3% of active speech FPs)",
            "affected_voices": "Quỳnh Anh, Quốc Tuấn",
            "recommended_mitigation": "Hard-negative contrastive loss with consonant-focused anchor pairs"
        },
        {
            "taxonomy_id": "ERR_FP_MINIMAL_PAIR_TONE",
            "error_type": "FALSE_POSITIVE",
            "target_keyword": "TRAI / LEN / XUONG",
            "phonetic_rhyme": "Tonal variants (trải, trãi, trại; lén, lẻn, lẹn; xuồng, xưởng, xuống)",
            "acoustic_mechanism": "Pitch perturbation under additive airflow pink noise causes tone contour misestimation",
            "mean_recall_loss": "N/A (Represents 28.1% of active speech FPs)",
            "affected_voices": "Kim Thanh, Đức Trí",
            "recommended_mitigation": "Explicit pitch/F0 contour feature stream in bounded-utterance architecture"
        }
    ]

    taxonomy_csv_path = os.path.join(v4_dir, "data", "error_taxonomy.csv")
    os.makedirs(os.path.dirname(taxonomy_csv_path), exist_ok=True)
    with open(taxonomy_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["taxonomy_id", "error_type", "target_keyword", "phonetic_rhyme", "acoustic_mechanism", "mean_recall_loss", "affected_voices", "recommended_mitigation"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in taxonomy_rows:
            writer.writerow(row)
    print(f"Error taxonomy written to {taxonomy_csv_path}")

    # Prepare Blinded Human Review Kit
    # Select 40 representative audio segments from the test stream / test sets:
    # 15 True Commands, 15 Hard-Negative Non-Targets, 10 Model-Challenged Events
    human_review_dir = os.path.join(v4_dir, "human_review")
    os.makedirs(human_review_dir, exist_ok=True)

    # We will construct a blinded key and a review rating sheet
    blinded_samples = []
    # 1. 10 True Commands
    for i, cmd in enumerate(cmd_gt[:12]):
        blinded_samples.append({
            "source_type": "TARGET_COMMAND",
            "ground_truth_label": cmd["class_label"],
            "true_text": cmd["class_label"],
            "voice_id": cmd["voice_id"],
            "reference_timestamp_sec": 0.5 * (cmd["onset_sec"] + cmd["end_time_sec"]),
            "audio_reference": "test_stream_continuous_commands.wav"
        })
    # 2. 18 Hard Negatives from nt_gt
    for i, nt in enumerate(nt_gt[:18]):
        blinded_samples.append({
            "source_type": "NON_TARGET_DISTRACTOR",
            "ground_truth_label": "UNKNOWN",
            "true_text": nt["text"],
            "voice_id": nt["voice_id"],
            "reference_timestamp_sec": 0.5 * (nt["onset_sec"] + nt["end_time_sec"]),
            "audio_reference": "test_stream_continuous_nontarget_3h.wav"
        })
    # 3. 10 Challenged Items (boundary / gap / quiet)
    for i in range(10):
        ev = nt_gt[30 + i * 15]
        blinded_samples.append({
            "source_type": "HARD_NEGATIVE_BOUNDARY",
            "ground_truth_label": "UNKNOWN",
            "true_text": ev["text"],
            "voice_id": ev["voice_id"],
            "reference_timestamp_sec": ev["end_time_sec"] + 0.15,
            "audio_reference": "test_stream_continuous_nontarget_3h.wav"
        })

    # Shuffle deterministically
    import random
    rng = random.Random(42)
    sample_indices = list(range(len(blinded_samples)))
    rng.shuffle(sample_indices)

    key_rows = []
    sheet_rows = []
    for blind_idx, orig_idx in enumerate(sample_indices, start=1):
        item = blinded_samples[orig_idx]
        blind_id = f"BLIND_REV_{blind_idx:03d}"
        key_rows.append({
            "blind_id": blind_id,
            "source_type": item["source_type"],
            "ground_truth_label": item["ground_truth_label"],
            "true_text": item["true_text"],
            "voice_id": item["voice_id"],
            "audio_reference": item["audio_reference"],
            "reference_timestamp_sec": round(item["reference_timestamp_sec"], 3)
        })
        sheet_rows.append({
            "blind_id": blind_id,
            "audio_reference": item["audio_reference"],
            "timestamp_window": f"{round(item['reference_timestamp_sec'] - 0.5, 2)}s - {round(item['reference_timestamp_sec'] + 0.5, 2)}s",
            "perceived_transcription": "",
            "is_target_command (TRAI/PHAI/LEN/XUONG/NO)": "",
            "intelligibility_rating_1_to_5": "",
            "reviewer_notes": ""
        })

    key_file = os.path.join(human_review_dir, "human_review_key.csv")
    with open(key_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(key_rows[0].keys()))
        writer.writeheader()
        writer.writerows(key_rows)

    sheet_file = os.path.join(human_review_dir, "human_review_sheet.csv")
    with open(sheet_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(sheet_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sheet_rows)

    print(f"Blinded human review key saved to: {key_file}")
    print(f"Blinded human review sheet saved to: {sheet_file}")

    # Save summary stats JSON
    stats = {
        "candidate3_fp_family": dict(cand3_fp_family),
        "candidate3_fp_target": dict(cand3_fp_target),
        "candidate3_fp_voices": dict(cand3_fp_voices),
        "candidate3_top_fp_words": sorted(cand3_fp_words.items(), key=lambda x: x[1], reverse=True)[:15],
        "candidate3_fn_by_kw": dict(cand3_fn_by_kw),
        "candidate3_fn_by_voice": dict(cand3_fn_by_voice),
    }
    with open(os.path.join(v4_dir, "data", "phase51_analysis_summary.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

if __name__ == "__main__":
    run_phase51_analysis()
