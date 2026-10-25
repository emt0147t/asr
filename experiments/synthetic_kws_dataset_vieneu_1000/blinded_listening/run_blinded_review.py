"""
Local Blinded Listening Review Web Application for Vietnamese Synthetic KWS (80 Clips)
Provides a clean, local web UI for human evaluators to listen to 80 anonymized clips
and submit perceptual ratings without exposing target labels, voice presets, or the answer key.

Usage:
  python run_blinded_review.py                # Starts local server on http://127.0.0.1:8765
  python run_blinded_review.py --port 8080    # Custom port
  python run_blinded_review.py --no-browser   # Do not auto-open browser
"""

import os
import sys
import csv
import json
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler
import webbrowser
import argparse

sys.stdout.reconfigure(encoding="utf-8")

BLINDED_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BLINDED_DIR, "audio")
REVIEW_SHEET_CSV = os.path.join(BLINDED_DIR, "blinded_listening_review_sheet.csv")
RESPONSES_CSV = os.path.join(BLINDED_DIR, "blinded_listening_responses.csv")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vietnamese KWS — Blinded Listening Evaluation (80 Clips)</title>
<style>
  :root {
    --primary: #2563eb;
    --primary-hover: #1d4ed8;
    --bg: #f8fafc;
    --card: #ffffff;
    --text: #0f172a;
    --text-muted: #64748b;
    --border: #e2e8f0;
    --success: #16a34a;
    --success-bg: #dcfce7;
    --warning: #f59e0b;
    --danger: #dc2626;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    padding: 24px;
    display: flex;
    justify-content: center;
    line-height: 1.5;
  }
  .container { max-width: 820px; width: 100%; }
  .header {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  }
  .title { font-size: 20px; font-weight: 700; color: var(--text); margin-bottom: 4px; }
  .subtitle { font-size: 13px; color: var(--text-muted); margin-bottom: 16px; }
  .progress-wrap { margin-top: 12px; }
  .progress-header {
    display: flex; justify-content: space-between; font-size: 13px;
    font-weight: 600; color: var(--text-muted); margin-bottom: 6px;
  }
  .progress-bar-bg {
    background: var(--border); height: 8px; border-radius: 4px; overflow: hidden;
  }
  .progress-bar-fill {
    background: var(--success); height: 100%; width: 0%; transition: width 0.3s;
  }
  .grid-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 20px;
  }
  .grid-title {
    font-size: 13px; font-weight: 600; color: var(--text-muted); margin-bottom: 10px;
    display: flex; justify-content: space-between;
  }
  .clip-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(36px, 1fr));
    gap: 6px;
    max-height: 120px;
    overflow-y: auto;
    padding: 4px;
  }
  .grid-item {
    font-size: 11px;
    font-weight: 600;
    text-align: center;
    padding: 6px 0;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: #fff;
    cursor: pointer;
    transition: all 0.15s;
    user-select: none;
  }
  .grid-item:hover { border-color: var(--primary); background: #f0f7ff; }
  .grid-item.active { border-color: var(--primary); background: #bfdbfe; color: #1e3a8a; }
  .grid-item.completed { background: var(--success-bg); border-color: #86efac; color: #166534; }
  .grid-item.completed.active { border: 2px solid var(--primary); }

  .player-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  }
  .clip-header {
    display: flex; justify-content: space-between; align-items: center;
    border-bottom: 1px solid var(--border); padding-bottom: 14px; margin-bottom: 18px;
  }
  .clip-id { font-size: 18px; font-weight: 700; color: var(--primary); }
  .audio-section {
    background: #f1f5f9; border-radius: 10px; padding: 18px;
    display: flex; flex-direction: column; align-items: center; gap: 12px; margin-bottom: 22px;
  }
  audio { width: 100%; max-width: 500px; }
  .shortcut-tip { font-size: 11px; color: var(--text-muted); }

  .form-section { margin-bottom: 22px; }
  .section-label {
    font-size: 14px; font-weight: 600; color: #334155; margin-bottom: 10px;
    display: flex; align-items: center; gap: 6px;
  }
  .required-badge {
    color: var(--danger); font-size: 12px;
  }
  .options-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 8px;
  }
  .btn-choice {
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 12px;
    background: #fff;
    cursor: pointer;
    font-size: 13px;
    font-weight: 500;
    text-align: center;
    transition: all 0.15s;
    user-select: none;
  }
  .btn-choice:hover { border-color: var(--primary); background: #f8fafc; }
  .btn-choice.selected {
    border-color: var(--primary); background: #eff6ff; color: #1d4ed8; font-weight: 600;
  }

  .scale-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 8px;
  }
  .scale-item {
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 6px;
    background: #fff;
    cursor: pointer;
    text-align: center;
    transition: all 0.15s;
    user-select: none;
  }
  .scale-item:hover { border-color: var(--primary); background: #f8fafc; }
  .scale-item.selected {
    border-color: var(--primary); background: #eff6ff; color: #1d4ed8; font-weight: 600;
  }
  .scale-num { font-size: 16px; font-weight: 700; margin-bottom: 2px; }
  .scale-desc { font-size: 10px; color: var(--text-muted); line-height: 1.2; }

  .defects-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 8px;
  }
  .defect-toggle {
    display: flex; align-items: center; gap: 8px;
    border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px;
    cursor: pointer; font-size: 13px; background: #fff; user-select: none;
  }
  .defect-toggle:hover { background: #f8fafc; }
  .defect-toggle input { cursor: pointer; }

  .comments-input {
    width: 100%; border: 1px solid var(--border); border-radius: 8px;
    padding: 10px 12px; font-size: 13px; font-family: inherit; resize: vertical; min-height: 50px;
  }
  .comments-input:focus { outline: none; border-color: var(--primary); }

  .nav-bar {
    display: flex; justify-content: space-between; align-items: center; margin-top: 24px;
  }
  .btn-nav {
    border: none; border-radius: 8px; padding: 10px 20px;
    font-size: 14px; font-weight: 600; cursor: pointer; transition: all 0.15s;
  }
  .btn-prev { background: #e2e8f0; color: #475569; }
  .btn-prev:hover { background: #cbd5e1; }
  .btn-next { background: var(--primary); color: #fff; }
  .btn-next:hover { background: var(--primary-hover); }
  .save-status { font-size: 12px; color: var(--success); font-weight: 500; }

  .finish-panel {
    background: var(--card); border: 1px solid var(--border); border-radius: 12px;
    padding: 36px 24px; text-align: center; display: none;
  }
  .finish-title { font-size: 22px; font-weight: 700; color: var(--success); margin-bottom: 12px; }
  .finish-desc { font-size: 14px; color: var(--text-muted); margin-bottom: 24px; }
</style>
</head>
<body>
<div class="container">
  <!-- Header -->
  <div class="header">
    <div class="title">Vietnamese Synthetic KWS — Blinded Listening Review</div>
    <div class="subtitle">Phase 25 Local Evaluation Screening Interface (80 Clips) • Strictly Anonymized</div>
    <div class="progress-wrap">
      <div class="progress-header">
        <span id="progress-text">Reviewed: 0 / 80 (0%)</span>
        <span id="remaining-text">80 remaining</span>
      </div>
      <div class="progress-bar-bg">
        <div class="progress-bar-fill" id="progress-fill"></div>
      </div>
    </div>
  </div>

  <!-- Quick Grid -->
  <div class="grid-card">
    <div class="grid-title">
      <span>Clip Quick Jump Navigator</span>
      <span style="font-weight: normal; font-size: 11px;">Green = Reviewed • Blue = Current</span>
    </div>
    <div class="clip-grid" id="clip-grid"></div>
  </div>

  <!-- Main Player Card -->
  <div class="player-card" id="review-panel">
    <div class="clip-header">
      <span class="clip-id" id="display-clip-id">BLIND_EVAL_01</span>
      <span style="font-size: 12px; color: var(--text-muted);" id="display-clip-index">Clip 1 of 80</span>
    </div>

    <!-- Audio Player -->
    <div class="audio-section">
      <audio id="audio-player" controls preload="auto"></audio>
      <div class="shortcut-tip">Tip: Press [Space] to Play/Pause • [R] to Replay from start</div>
    </div>

    <!-- Section 1: Perceived Command -->
    <div class="form-section">
      <div class="section-label">
        1. Recognized Vietnamese Command Word
        <span class="required-badge">* Required</span>
      </div>
      <div class="options-grid" id="command-options">
        <div class="btn-choice" data-val="TRAI">Trái (TRAI)</div>
        <div class="btn-choice" data-val="PHAI">Phải (PHAI)</div>
        <div class="btn-choice" data-val="LEN">Lên (LEN)</div>
        <div class="btn-choice" data-val="XUONG">Xuống (XUONG)</div>
        <div class="btn-choice" data-val="OTHER">Other Word</div>
        <div class="btn-choice" data-val="UNCERTAIN">Uncertain / Unclear</div>
      </div>
    </div>

    <!-- Section 2: Perceived Tone -->
    <div class="form-section">
      <div class="section-label">
        2. Perceived Lexical Tone (Auditory Contour)
        <span class="required-badge">* Required</span>
      </div>
      <div class="options-grid" id="tone-options">
        <div class="btn-choice" data-val="SAC">Sắc (Rising)</div>
        <div class="btn-choice" data-val="HOI">Hỏi (Dipping/Falling)</div>
        <div class="btn-choice" data-val="NGANG">Ngang (Mid-level flat)</div>
        <div class="btn-choice" data-val="OTHER">Other Tone</div>
        <div class="btn-choice" data-val="UNCERTAIN">Uncertain</div>
      </div>
    </div>

    <!-- Section 3: Intelligibility (1 to 5) -->
    <div class="form-section">
      <div class="section-label">
        3. Intelligibility Rating (1 = Inaudible, 5 = Crystal Clear)
        <span class="required-badge">* Required</span>
      </div>
      <div class="scale-grid" id="intel-scale">
        <div class="scale-item" data-val="1"><div class="scale-num">1</div><div class="scale-desc">Unintelligible</div></div>
        <div class="scale-item" data-val="2"><div class="scale-num">2</div><div class="scale-desc">Ambiguous / Hard</div></div>
        <div class="scale-item" data-val="3"><div class="scale-num">3</div><div class="scale-desc">Recognizable with effort</div></div>
        <div class="scale-item" data-val="4"><div class="scale-num">4</div><div class="scale-desc">Clear speech</div></div>
        <div class="scale-item" data-val="5"><div class="scale-num">5</div><div class="scale-desc">Crystal clear</div></div>
      </div>
    </div>

    <!-- Section 4: Naturalness (1 to 5) -->
    <div class="form-section">
      <div class="section-label">
        4. Naturalness Rating (1 = Harsh Robotic, 5 = Indistinguishable from Human)
        <span class="required-badge">* Required</span>
      </div>
      <div class="scale-grid" id="natural-scale">
        <div class="scale-item" data-val="1"><div class="scale-num">1</div><div class="scale-desc">Harsh robotic / buzzer</div></div>
        <div class="scale-item" data-val="2"><div class="scale-num">2</div><div class="scale-desc">Unnatural artifacts</div></div>
        <div class="scale-item" data-val="3"><div class="scale-num">3</div><div class="scale-desc">Synthetic but acceptable</div></div>
        <div class="scale-item" data-val="4"><div class="scale-num">4</div><div class="scale-desc">Natural speech</div></div>
        <div class="scale-item" data-val="5"><div class="scale-num">5</div><div class="scale-desc">Completely human-like</div></div>
      </div>
    </div>

    <!-- Section 5: Acoustic Defects -->
    <div class="form-section">
      <div class="section-label">5. Observed Acoustic Defects</div>
      <div class="defects-grid">
        <label class="defect-toggle">
          <input type="checkbox" id="defect-clipped">
          <span>Digital Clipping / Crackle</span>
        </label>
        <label class="defect-toggle">
          <input type="checkbox" id="defect-distorted">
          <span>Robotic Distortion</span>
        </label>
        <label class="defect-toggle">
          <input type="checkbox" id="defect-truncated">
          <span>Phoneme Truncation</span>
        </label>
        <label class="defect-toggle">
          <input type="checkbox" id="defect-buzzer">
          <span>Buzzer / Non-speech tone</span>
        </label>
        <label class="defect-toggle">
          <input type="checkbox" id="defect-clean">
          <span>Clean / Defect-free</span>
        </label>
      </div>
    </div>

    <!-- Section 6: Notes -->
    <div class="form-section">
      <div class="section-label">6. Evaluator Notes (Optional)</div>
      <textarea class="comments-input" id="comments" placeholder="Notes on accent, tone subtleties, glottal constriction, or acoustic anomalies..."></textarea>
    </div>

    <!-- Nav Bar -->
    <div class="nav-bar">
      <button class="btn-nav btn-prev" id="btn-prev">← Previous</button>
      <span class="save-status" id="save-status">Saved incrementally to disk</span>
      <button class="btn-nav btn-next" id="btn-next">Save & Next →</button>
    </div>
  </div>

  <!-- Finish Panel -->
  <div class="finish-panel" id="finish-panel">
    <div class="finish-title">Evaluation Complete!</div>
    <div class="finish-desc">
      All 80 clips have been reviewed and saved incrementally to disk.<br>
      Output file: <code>blinded_listening_responses.csv</code>
    </div>
    <div style="display: flex; justify-content: center; gap: 12px;">
      <button class="btn-nav btn-prev" id="btn-review-again">Review Clips Again</button>
      <button class="btn-nav btn-next" id="btn-download-csv">Download Responses CSV</button>
    </div>
  </div>
</div>

<script>
let manifest = [];
let responses = {};
let currentIndex = 0;

const player = document.getElementById('audio-player');

async function init() {
  try {
    const mRes = await fetch('/api/manifest');
    manifest = await mRes.json();

    const rRes = await fetch('/api/responses');
    responses = await rRes.json();

    buildGrid();
    renderClip(findFirstUnreviewedIndex());
  } catch (err) {
    console.error('Initialization error:', err);
  }
}

function findFirstUnreviewedIndex() {
  for (let i = 0; i < manifest.length; i++) {
    const id = manifest[i].blind_clip_id;
    if (!responses[id] || !responses[id].perceived_command) {
      return i;
    }
  }
  return 0;
}

function buildGrid() {
  const grid = document.getElementById('clip-grid');
  grid.innerHTML = '';
  manifest.forEach((item, idx) => {
    const div = document.createElement('div');
    div.className = 'grid-item';
    div.innerText = (idx + 1);
    div.title = item.blind_clip_id;
    div.id = 'grid-' + idx;
    div.addEventListener('click', () => {
      saveCurrentState();
      renderClip(idx);
    });
    grid.appendChild(div);
  });
  updateGridStatus();
}

function updateGridStatus() {
  let completedCount = 0;
  manifest.forEach((item, idx) => {
    const el = document.getElementById('grid-' + idx);
    if (!el) return;
    const r = responses[item.blind_clip_id];
    el.classList.remove('active', 'completed');
    if (r && r.perceived_command) {
      el.classList.add('completed');
      completedCount++;
    }
    if (idx === currentIndex) {
      el.classList.add('active');
    }
  });

  const total = manifest.length;
  const pct = total > 0 ? Math.round((completedCount / total) * 100) : 0;
  document.getElementById('progress-text').innerText = `Reviewed: ${completedCount} / ${total} (${pct}%)`;
  document.getElementById('remaining-text').innerText = `${total - completedCount} remaining`;
  document.getElementById('progress-fill').style.width = pct + '%';
}

function renderClip(index) {
  currentIndex = index;
  const item = manifest[currentIndex];
  document.getElementById('display-clip-id').innerText = item.blind_clip_id;
  document.getElementById('display-clip-index').innerText = `Clip ${currentIndex + 1} of ${manifest.length}`;

  player.src = item.audio_url;
  player.load();

  // Reset or load response
  const r = responses[item.blind_clip_id] || {};
  setSelectedChoice('command-options', r.perceived_command || null);
  setSelectedChoice('tone-options', r.perceived_tone || null);
  setSelectedChoice('intel-scale', r.intelligibility ? String(r.intelligibility) : null);
  setSelectedChoice('natural-scale', r.naturalness ? String(r.naturalness) : null);

  document.getElementById('defect-clipped').checked = (r.defect_clipped === 'YES');
  document.getElementById('defect-distorted').checked = (r.defect_distorted === 'YES');
  document.getElementById('defect-truncated').checked = (r.defect_truncated === 'YES');
  document.getElementById('defect-buzzer').checked = (r.defect_buzzer === 'YES');
  document.getElementById('defect-clean').checked = (r.defect_clean === 'YES');

  document.getElementById('comments').value = r.comments || '';

  document.getElementById('btn-prev').disabled = (currentIndex === 0);
  document.getElementById('btn-next').innerText = (currentIndex === manifest.length - 1) ? 'Finish Evaluation' : 'Save & Next →';

  updateGridStatus();
}

function setSelectedChoice(containerId, value) {
  const container = document.getElementById(containerId);
  const items = container.querySelectorAll('.btn-choice, .scale-item');
  items.forEach(el => {
    if (el.getAttribute('data-val') === value) {
      el.classList.add('selected');
    } else {
      el.classList.remove('selected');
    }
  });
}

function getSelectedChoice(containerId) {
  const container = document.getElementById(containerId);
  const selected = container.querySelector('.selected');
  return selected ? selected.getAttribute('data-val') : null;
}

// Choice button clicks
['command-options', 'tone-options', 'intel-scale', 'natural-scale'].forEach(cid => {
  const container = document.getElementById(cid);
  container.addEventListener('click', (e) => {
    const target = e.target.closest('.btn-choice, .scale-item');
    if (!target) return;
    const val = target.getAttribute('data-val');
    setSelectedChoice(cid, val);
  });
});

// Defect toggles
const defectBoxes = ['defect-clipped', 'defect-distorted', 'defect-truncated', 'defect-buzzer'];
defectBoxes.forEach(id => {
  document.getElementById(id).addEventListener('change', () => {
    if (document.getElementById(id).checked) {
      document.getElementById('defect-clean').checked = false;
    }
  });
});
document.getElementById('defect-clean').addEventListener('change', () => {
  if (document.getElementById('defect-clean').checked) {
    defectBoxes.forEach(id => { document.getElementById(id).checked = false; });
  }
});

function saveCurrentState() {
  const item = manifest[currentIndex];
  if (!item) return;

  const cmd = getSelectedChoice('command-options');
  const tone = getSelectedChoice('tone-options');
  const intel = getSelectedChoice('intel-scale');
  const natural = getSelectedChoice('natural-scale');

  const cleanBox = document.getElementById('defect-clean');
  const anyDefect = defectBoxes.some(id => document.getElementById(id).checked);

  // If user made choices, record review
  if (cmd || tone || intel || natural) {
    responses[item.blind_clip_id] = {
      blind_clip_id: item.blind_clip_id,
      audio_file_path: item.audio_file_path,
      perceived_command: cmd || '',
      defect_clipped: document.getElementById('defect-clipped').checked ? 'YES' : 'NO',
      defect_distorted: document.getElementById('defect-distorted').checked ? 'YES' : 'NO',
      defect_truncated: document.getElementById('defect-truncated').checked ? 'YES' : 'NO',
      defect_buzzer: document.getElementById('defect-buzzer').checked ? 'YES' : 'NO',
      defect_clean: (cleanBox.checked || !anyDefect) ? 'YES' : 'NO',
      perceived_tone: tone || '',
      intelligibility: intel ? parseInt(intel, 10) : '',
      naturalness: natural ? parseInt(natural, 10) : '',
      comments: document.getElementById('comments').value.trim()
    };
  }
}

async function persistCurrentToServer() {
  saveCurrentState();
  const item = manifest[currentIndex];
  const r = responses[item.blind_clip_id];
  if (!r) return;

  try {
    const res = await fetch('/api/save_response', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(r)
    });
    if (res.ok) {
      const statusEl = document.getElementById('save-status');
      statusEl.innerText = `Saved ${item.blind_clip_id} to disk`;
      setTimeout(() => { statusEl.innerText = 'Saved incrementally to disk'; }, 2000);
      updateGridStatus();
    }
  } catch (err) {
    console.error('Save error:', err);
  }
}

// Keyboard shortcuts
window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') return;
  if (e.code === 'Space') {
    e.preventDefault();
    if (player.paused) player.play(); else player.pause();
  } else if (e.key === 'r' || e.key === 'R') {
    e.preventDefault();
    player.currentTime = 0;
    player.play();
  }
});

document.getElementById('btn-prev').addEventListener('click', async () => {
  await persistCurrentToServer();
  if (currentIndex > 0) renderClip(currentIndex - 1);
});

document.getElementById('btn-next').addEventListener('click', async () => {
  const cmd = getSelectedChoice('command-options');
  if (!cmd) {
    alert('Please select a perceived command word before proceeding.');
    return;
  }
  const tone = getSelectedChoice('tone-options');
  if (!tone) {
    alert('Please select a perceived lexical tone before proceeding.');
    return;
  }
  const intel = getSelectedChoice('intel-scale');
  if (!intel) {
    alert('Please select an intelligibility score (1-5).');
    return;
  }
  const natural = getSelectedChoice('natural-scale');
  if (!natural) {
    alert('Please select a naturalness score (1-5).');
    return;
  }

  await persistCurrentToServer();

  if (currentIndex < manifest.length - 1) {
    renderClip(currentIndex + 1);
  } else {
    document.getElementById('review-panel').style.display = 'none';
    document.getElementById('finish-panel').style.display = 'block';
  }
});

document.getElementById('btn-review-again').addEventListener('click', () => {
  document.getElementById('finish-panel').style.display = 'none';
  document.getElementById('review-panel').style.display = 'block';
  renderClip(0);
});

document.getElementById('btn-download-csv').addEventListener('click', () => {
  window.location.href = '/api/export_csv';
});

init();
</script>
</body>
</html>
"""

class BlindedReviewHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        # Silence benign GET 200 logs to keep server output clean
        pass

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        path = url.path

        if path in ["/", "/index.html"]:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
            return

        elif path == "/api/manifest":
            # Returns strictly blinded metadata: blind_clip_id and audio_url
            items = []
            if os.path.exists(REVIEW_SHEET_CSV):
                with open(REVIEW_SHEET_CSV, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        bid = row.get("blind_clip_id")
                        if bid:
                            items.append({
                                "blind_clip_id": bid,
                                "audio_file_path": row.get("audio_file_path", f"experiments/synthetic_kws_dataset_vieneu_1000/blinded_listening/audio/{bid.lower()}.wav"),
                                "audio_url": f"/audio/{bid.lower()}.wav"
                            })
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(items, ensure_ascii=False).encode("utf-8"))
            return

        elif path == "/api/responses":
            responses_dict = {}
            target_csv = RESPONSES_CSV if os.path.exists(RESPONSES_CSV) else REVIEW_SHEET_CSV
            if os.path.exists(target_csv):
                with open(target_csv, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        bid = row.get("blind_clip_id")
                        cmd = None
                        for k, v in row.items():
                            if "perceived_command" in k and v and v.strip():
                                cmd = v.strip().upper()
                                break
                        if bid and cmd:
                            responses_dict[bid] = {
                                "blind_clip_id": bid,
                                "audio_file_path": row.get("audio_file_path", ""),
                                "perceived_command": cmd,
                                "defect_clipped": row.get("defect_clipped (YES/NO)", "NO"),
                                "defect_distorted": row.get("defect_distorted (YES/NO)", "NO"),
                                "defect_truncated": row.get("defect_truncated (YES/NO)", "NO"),
                                "defect_buzzer": row.get("defect_buzzer (YES/NO)", "NO"),
                                "defect_clean": row.get("defect_clean (YES/NO)", "YES"),
                                "perceived_tone": row.get("perceived_tone (SAC/HOI/NGANG/OTHER/UNCERTAIN)", ""),
                                "intelligibility": row.get("intelligibility (1 to 5, 5=crystal clear)", ""),
                                "naturalness": row.get("naturalness (1 to 5, 5=indistinguishable from human)", ""),
                                "comments": row.get("evaluator_comments", "")
                            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(responses_dict, ensure_ascii=False).encode("utf-8"))
            return

        elif path == "/api/export_csv":
            target_csv = RESPONSES_CSV if os.path.exists(RESPONSES_CSV) else REVIEW_SHEET_CSV
            if os.path.exists(target_csv):
                with open(target_csv, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Disposition", 'attachment; filename="blinded_listening_responses.csv"')
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_response(404)
                self.end_headers()
            return

        elif path.startswith("/audio/"):
            filename = os.path.basename(path)
            audio_path = os.path.join(AUDIO_DIR, filename)
            if os.path.exists(audio_path) and filename.endswith(".wav"):
                with open(audio_path, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_response(404)
                self.end_headers()
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        url = urllib.parse.urlparse(self.path)
        path = url.path

        if path == "/api/save_response":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            try:
                record = json.loads(body)
                bid = record.get("blind_clip_id")
                if not bid:
                    self.send_response(400)
                    self.end_headers()
                    return

                # Read existing responses or review sheet
                rows_map = {}
                headers = [
                    "blind_clip_id",
                    "audio_file_path",
                    "perceived_command (TRAI/PHAI/LEN/XUONG/OTHER/UNCERTAIN)",
                    "defect_clipped (YES/NO)",
                    "defect_distorted (YES/NO)",
                    "defect_truncated (YES/NO)",
                    "defect_buzzer (YES/NO)",
                    "defect_clean (YES/NO)",
                    "perceived_tone (SAC/HOI/NGANG/OTHER/UNCERTAIN)",
                    "intelligibility (1 to 5, 5=crystal clear)",
                    "naturalness (1 to 5, 5=indistinguishable from human)",
                    "evaluator_comments"
                ]

                # Initialize rows from REVIEW_SHEET_CSV to preserve order 1..80
                if os.path.exists(REVIEW_SHEET_CSV):
                    with open(REVIEW_SHEET_CSV, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for r in reader:
                            rows_map[r["blind_clip_id"]] = r

                # Overlay any prior responses from RESPONSES_CSV
                if os.path.exists(RESPONSES_CSV):
                    with open(RESPONSES_CSV, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for r in reader:
                            rows_map[r["blind_clip_id"]] = r

                # Update with current record
                updated_row = rows_map.get(bid, {})
                updated_row["blind_clip_id"] = bid
                updated_row["audio_file_path"] = record.get("audio_file_path", updated_row.get("audio_file_path", f"experiments/synthetic_kws_dataset_vieneu_1000/blinded_listening/audio/{bid.lower()}.wav"))
                updated_row["perceived_command (TRAI/PHAI/LEN/XUONG/OTHER/UNCERTAIN)"] = record.get("perceived_command", "")
                updated_row["defect_clipped (YES/NO)"] = record.get("defect_clipped", "NO")
                updated_row["defect_distorted (YES/NO)"] = record.get("defect_distorted", "NO")
                updated_row["defect_truncated (YES/NO)"] = record.get("defect_truncated", "NO")
                updated_row["defect_buzzer (YES/NO)"] = record.get("defect_buzzer", "NO")
                updated_row["defect_clean (YES/NO)"] = record.get("defect_clean", "YES")
                updated_row["perceived_tone (SAC/HOI/NGANG/OTHER/UNCERTAIN)"] = record.get("perceived_tone", "")
                updated_row["intelligibility (1 to 5, 5=crystal clear)"] = record.get("intelligibility", "")
                updated_row["naturalness (1 to 5, 5=indistinguishable from human)"] = record.get("naturalness", "")
                updated_row["evaluator_comments"] = record.get("comments", "")
                rows_map[bid] = updated_row

                # Atomically write to RESPONSES_CSV
                tmp_file = RESPONSES_CSV + ".tmp"
                with open(tmp_file, "w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writeheader()
                    for k in sorted(rows_map.keys(), key=lambda x: int(x.split("_")[-1])):
                        writer.writerow(rows_map[k])

                if os.path.exists(RESPONSES_CSV):
                    os.remove(RESPONSES_CSV)
                os.rename(tmp_file, RESPONSES_CSV)

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status":"saved"}')
                return
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(str(e).encode("utf-8"))
                return

        self.send_response(404)
        self.end_headers()

def run_server(host=DEFAULT_HOST, port=DEFAULT_PORT, auto_open=True):
    server_address = (host, port)
    httpd = HTTPServer(server_address, BlindedReviewHandler)
    url = f"http://{host}:{port}/"
    print("=" * 70)
    print(f"Blinded Listening Review Server running at: {url}")
    print("Press Ctrl+C to stop the server.")
    print("=" * 70)

    if auto_open:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        httpd.server_close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run blinded listening review server")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port (default: 8765)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")
    args = parser.parse_args()

    run_server(args.host, args.port, not args.no_browser)
