import os

DOCS_DIR = r"d:\trainidentificationaudio\reports\cnn_comprehensive"

replacements = [
    # 01_PROJECT_SCOPE_AND_REQUIREMENTS.md
    ("tests/test_kws_streaming.py", "src/models/evaluate_streaming.py"),
    
    # 06_ERROR_ANALYSIS_AND_RESEARCH_LESSONS.md
    ("experiments/vieneu_kws_research_v12/bundle/src/infer.py", "experiments/vieneu_kws_research_v12/bundle/infer.py"),
    
    # 08_HUMAN_LISTENING_AND_VALIDATION.md
    ("data/raw/field_recordings/phase90_listening/", "experiments/vieneu_kws_research_v10/human_review/"),
    ("experiments/vieneu_kws_research_v9/human_review/reviewer_01_responses.csv", "experiments/vieneu_kws_research_v10/human_review/responses/reviewer_01_responses.csv"),
    ("experiments/vieneu_kws_research_v9/human_review/validated_responses.csv", "experiments/vieneu_kws_research_v10/human_review/phase90_validated_responses.csv"),
    
    # 10_CHECKPOINT_LINEAGE_AND_RELEASE_ARTIFACTS.md
    ("bundle/src/infer.py", "bundle/infer.py"),
    ("bundle/config/preprocessing_contract.json", "bundle/preprocessing_contract.json"),
    
    # 14_EVIDENCE_GAPS_AND_DISCREPANCIES.md
    ("reports/phase92_master_summary.md", "experiments/vieneu_kws_research_v10/reports/PHASE92_CROSS_PLATFORM_PARITY.md"),
    
    # E_SOURCE_REFERENCE_INDEX.md
    ("experiments/vieneu_kws_research_v92/reports/PHASE92_CROSS_PLATFORM_PARITY.md", "experiments/vieneu_kws_research_v10/reports/PHASE92_CROSS_PLATFORM_PARITY.md"),
    ("experiments/vieneu_kws_research_v91/reports/PHASE91_EDGE_DEPLOYMENT.md", "experiments/vieneu_kws_research_v10/reports/PHASE91_CNN_DEPLOYMENT.md"),
    ("experiments/vieneu_kws_research_v90/reports/PHASE90_HUMAN_LISTENING_STUDY.md", "experiments/vieneu_kws_research_v10/reports/PHASE90_HUMAN_LISTENING_VALIDATION.md"),
    ("experiments/vieneu_kws_research_v13_4/bundle/src/infer.py", "experiments/vieneu_kws_research_v13_4/bundle/infer.py"),
    ("experiments/vieneu_kws_research_v13_4/bundle/src/model.py", "experiments/vieneu_kws_research_v10/deployment/export_model.py"),
    ("experiments/vieneu_kws_research_v13_4/bundle/src/frontend.py", "experiments/vieneu_kws_research_v10/parity/portable_frontend.py"),
    ("experiments/vieneu_kws_research_v11/src/train_v11.py", "experiments/vieneu_kws_research_v11/train_v11.py"),
    ("experiments/vieneu_kws_research_v13_4/bundle/config/preprocessing_contract.json", "experiments/vieneu_kws_research_v13_4/bundle/preprocessing_contract.json"),
]

for root, dirs, files in os.walk(DOCS_DIR):
    for f in files:
        if f.endswith(".md"):
            filepath = os.path.join(root, f)
            with open(filepath, "r", encoding="utf-8") as file:
                content = file.read()
            modified = False
            for old_s, new_s in replacements:
                if old_s in content:
                    content = content.replace(old_s, new_s)
                    modified = True
            if modified:
                with open(filepath, "w", encoding="utf-8") as file:
                    file.write(content)
                print(f"[UPDATED LINKS] {f}")

print("[DONE] Link fix script executed.")
