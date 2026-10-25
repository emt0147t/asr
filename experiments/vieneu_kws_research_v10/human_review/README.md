# Blinded Human Listening Evaluation Package & Local App

**Workspace:** `D:\trainidentificationaudio`  
**Package Directory:** `experiments/vieneu_kws_research_v10/human_review/`  
**Review Corpus:** 120 randomized clips (16 kHz WAV, anonymized)  
**Status Banner:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Directory Structure
```
human_review/
├── README.md                      # Package overview & operating instructions
├── reviewer_instructions.md       # Comprehensive rubrics for human raters
├── response_template.csv          # Blank CSV template (120 rows)
├── review_manifest.csv            # Internal answer key (RESTRICTED: DO NOT SERVE TO RATERS)
├── review_app.py                  # Standalone local HTTP web application (127.0.0.1)
├── audio/                         # 120 anonymized audio files (review_clip_001.wav - 120.wav)
└── responses/                     # Saved reviewer submissions (CSV format)
```

---

## 2. Running the Local Review Application
To launch the review server locally:
```powershell
python experimentsieneu_kws_research_v10\human_revieweview_app.py --port 8080
```
Then navigate to:
```
http://127.0.0.1:8080/
```

### Security & Blinding Guarantees:
- **Localhost only:** Bound strictly to `127.0.0.1`.
- **Zero answer key leakage:** The web application reads only `audio/` and writes to `responses/`. It has zero routes exposing `review_manifest.csv`.
- **Zero external network requests:** Self-contained vanilla HTML/CSS/JS without external CDN calls.
- **Autosave & Resume:** Progress is saved automatically to `responses/reviewer_<id>_responses.csv`.
