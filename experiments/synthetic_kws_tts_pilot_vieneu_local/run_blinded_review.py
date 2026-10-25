"""
Local Blinded Listening Review Web Application and CLI for Vietnamese Synthetic KWS
Allows human listeners to evaluate 24 anonymized KWS clips without exposing
the underlying synthesizer (VieNeu-TTS, Edge-TTS, V5.2) or intended keywords.

Usage:
  python run_blinded_review.py         # Launches browser interface on http://127.0.0.1:8765
  python run_blinded_review.py --cli   # Runs terminal-based interactive review
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

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(APP_DIR, "blinded_listening_manifest.csv")
RESPONSES_CSV_PATH = os.path.join(APP_DIR, "blinded_listening_responses.csv")
PORT = 8765

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vietnamese KWS — Blinded Listening Review</title>
<style>
  :root {
    --primary: #1976d2;
    --primary-dark: #115293;
    --bg: #f8fafc;
    --card: #ffffff;
    --text: #1e293b;
    --text-muted: #64748b;
    --border: #e2e8f0;
    --success: #16a34a;
  }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    margin: 0;
    padding: 24px;
    display: flex;
    justify-content: center;
  }
  .container {
    max-width: 760px;
    width: 100%;
  }
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 28px;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    margin-bottom: 20px;
  }
  .header {
    margin-bottom: 24px;
  }
  h1 {
    font-size: 22px;
    margin: 0 0 6px 0;
    color: #0f172a;
  }
  .subtext {
    font-size: 14px;
    color: var(--text-muted);
    line-height: 1.5;
  }
  .progress-wrap {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 20px;
  }
  .progress-bar-bg {
    flex: 1;
    height: 8px;
    background: #e2e8f0;
    border-radius: 4px;
    overflow: hidden;
  }
  .progress-bar-fill {
    height: 100%;
    background: var(--primary);
    width: 0%;
    transition: width 0.3s;
  }
  .clip-badge {
    display: inline-block;
    background: #e0f2fe;
    color: #0369a1;
    font-weight: 600;
    font-size: 13px;
    padding: 4px 10px;
    border-radius: 6px;
    margin-bottom: 16px;
  }
  .player-box {
    background: #f1f5f9;
    border-radius: 8px;
    padding: 16px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
    margin-bottom: 24px;
  }
  audio {
    width: 100%;
    max-width: 500px;
  }
  .form-section {
    margin-bottom: 22px;
  }
  .form-section-title {
    font-size: 15px;
    font-weight: 600;
    margin-bottom: 10px;
    color: #334155;
  }
  .options-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 10px;
  }
  .btn-option {
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 14px;
    background: #fff;
    cursor: pointer;
    font-size: 14px;
    font-weight: 500;
    text-align: center;
    transition: all 0.15s;
    user-select: none;
  }
  .btn-option:hover {
    border-color: var(--primary);
    background: #f0f7ff;
  }
  .btn-option.selected {
    border-color: var(--primary);
    background: #e0f2fe;
    color: #0369a1;
    font-weight: 600;
  }
  .checkbox-label {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 10px;
    background: #fff;
    border: 1px solid var(--border);
    border-radius: 6px;
    cursor: pointer;
    font-size: 13px;
  }
  .checkbox-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 8px;
  }
  .scale-grid {
    display: flex;
    gap: 10px;
  }
  .scale-item {
    flex: 1;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 6px;
    text-align: center;
    cursor: pointer;
    font-size: 13px;
    background: #fff;
    transition: all 0.15s;
  }
  .scale-item:hover {
    background: #f8fafc;
    border-color: var(--primary);
  }
  .scale-item.selected {
    background: #e0f2fe;
    border-color: var(--primary);
    color: #0369a1;
    font-weight: 600;
  }
  .scale-score {
    font-size: 16px;
    font-weight: 700;
    display: block;
    margin-bottom: 2px;
  }
  textarea {
    width: 100%;
    box-sizing: border-box;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px;
    font-family: inherit;
    font-size: 13px;
    resize: vertical;
  }
  .actions {
    display: flex;
    justify-content: space-between;
    margin-top: 28px;
    padding-top: 16px;
    border-top: 1px solid var(--border);
  }
  .btn {
    padding: 10px 20px;
    border-radius: 8px;
    border: none;
    cursor: pointer;
    font-weight: 600;
    font-size: 14px;
    transition: all 0.15s;
  }
  .btn-primary {
    background: var(--primary);
    color: #fff;
  }
  .btn-primary:hover {
    background: var(--primary-dark);
  }
  .btn-secondary {
    background: #f1f5f9;
    color: #475569;
  }
  .btn-secondary:hover {
    background: #e2e8f0;
  }
  .status-msg {
    margin-top: 12px;
    font-size: 13px;
    text-align: center;
    color: var(--success);
    font-weight: 600;
  }
  .finish-box {
    text-align: center;
    padding: 30px 20px;
  }
</style>
</head>
<body>
<div class="container">
  <div class="card">
    <div class="header">
      <h1>Vietnamese KWS — Blinded Listening Evaluation</h1>
      <div class="subtext">
        Anonymized perceptual evaluation of Vietnamese synthetic command audio.<br>
        Source models and target words are completely blinded. Please listen to each audio clip and provide your honest impression.
      </div>
    </div>

    <div class="progress-wrap">
      <span id="progress-text" style="font-size: 13px; font-weight: 600; color: #475569;">Clip 1 of 24</span>
      <div class="progress-bar-bg">
        <div id="progress-fill" class="progress-bar-fill"></div>
      </div>
    </div>

    <div id="review-panel">
      <div class="clip-badge" id="clip-badge">BLIND_EVAL_01</div>

      <div class="player-box">
        <audio id="audio-player" controls preload="auto"></audio>
        <div style="font-size: 12px; color: var(--text-muted);">Click Play or press Spacebar. Adjust volume as needed.</div>
      </div>

      <!-- 1. Perceived Command -->
      <div class="form-section">
        <div class="form-section-title">1. Perceived Vietnamese Command (Bạn nghe thấy từ gì?) *</div>
        <div class="options-grid" id="command-options">
          <div class="btn-option" data-val="TRAI">TRÁI</div>
          <div class="btn-option" data-val="PHAI">PHẢI</div>
          <div class="btn-option" data-val="LEN">LÊN</div>
          <div class="btn-option" data-val="XUONG">XUỐNG</div>
          <div class="btn-option" data-val="OTHER">OTHER (Từ khác)</div>
          <div class="btn-option" data-val="UNCERTAIN">UNCERTAIN (Không rõ)</div>
        </div>
      </div>

      <!-- 2. Acoustic Defects & Artifacts -->
      <div class="form-section">
        <div class="form-section-title">2. Acoustic Quality & Artifacts (Khuyết tật âm thanh nếu có)</div>
        <div class="checkbox-grid">
          <label class="checkbox-label"><input type="checkbox" id="defect-clipped"> Bị rè / vỡ tiếng (Clipped)</label>
          <label class="checkbox-label"><input type="checkbox" id="defect-distorted"> Bị méo âm / nghẹt mũi (Distorted)</label>
          <label class="checkbox-label"><input type="checkbox" id="defect-truncated"> Bị ngắt cụt đầu/đuôi (Truncated)</label>
          <label class="checkbox-label"><input type="checkbox" id="defect-buzzer"> Giọng robot / tiếng vo ve giả tạo (Buzzer)</label>
          <label class="checkbox-label"><input type="checkbox" id="defect-clean"> Âm thanh trong rõ, không khuyết tật (Clean)</label>
        </div>
      </div>

      <!-- 3. Tone Correctness -->
      <div class="form-section">
        <div class="form-section-title">3. Perceived Tone (Thanh điệu nghe thấy)</div>
        <div class="options-grid" id="tone-options">
          <div class="btn-option" data-val="SAC">Sắc (high-rising)</div>
          <div class="btn-option" data-val="HOI">Hỏi (dipping / falling-rising)</div>
          <div class="btn-option" data-val="NGANG">Ngang (mid-level)</div>
          <div class="btn-option" data-val="OTHER_TONE">Thanh khác / Sai thanh</div>
          <div class="btn-option" data-val="UNSURE_TONE">Không chắc</div>
        </div>
      </div>

      <!-- 4. Intelligibility Score (1-5) -->
      <div class="form-section">
        <div class="form-section-title">4. Intelligibility (Độ dễ nghe / nhận diện được từ) *</div>
        <div class="scale-grid" id="intel-scale">
          <div class="scale-item" data-val="1"><span class="scale-score">1</span>Không nhận ra từ</div>
          <div class="scale-item" data-val="2"><span class="scale-score">2</span>Rất khó nghe</div>
          <div class="scale-item" data-val="3"><span class="scale-score">3</span>Nghe được cần chú ý</div>
          <div class="scale-item" data-val="4"><span class="scale-score">4</span>Rõ ràng, dễ nghe</div>
          <div class="scale-item" data-val="5"><span class="scale-score">5</span>Cực kỳ rõ nét</div>
        </div>
      </div>

      <!-- 5. Naturalness Score (1-5) -->
      <div class="form-section">
        <div class="form-section-title">5. Naturalness (Độ tự nhiên của giọng nói) *</div>
        <div class="scale-grid" id="natural-scale">
          <div class="scale-item" data-val="1"><span class="scale-score">1</span>Hoàn toàn máy móc / buzz</div>
          <div class="scale-item" data-val="2"><span class="scale-score">2</span>Giọng robot rõ rệt</div>
          <div class="scale-item" data-val="3"><span class="scale-score">3</span>Giọng máy tạm chấp nhận</div>
          <div class="scale-item" data-val="4"><span class="scale-score">4</span>Tự nhiên, gần người thật</div>
          <div class="scale-item" data-val="5"><span class="scale-score">5</span>Như người thật nói</div>
        </div>
      </div>

      <!-- 6. Comments -->
      <div class="form-section">
        <div class="form-section-title">6. Comments / Ghi chú bổ sung (tùy chọn)</div>
        <textarea id="comments" rows="2" placeholder="Ví dụ: Giọng miền Nam nhẹ; phụ âm đầu bật hơi tốt; bị lẫn tiếng ồn nền..."></textarea>
      </div>

      <div class="actions">
        <button class="btn btn-secondary" id="btn-prev" disabled>← Previous Clip</button>
        <button class="btn btn-secondary" id="btn-save-draft">Save Progress</button>
        <button class="btn btn-primary" id="btn-next">Next Clip →</button>
      </div>
      <div id="status-msg" class="status-msg"></div>
    </div>

    <div id="finish-panel" class="finish-box" style="display: none;">
      <h2 style="color: var(--success); margin-bottom: 12px;">Evaluation Complete! 🎉</h2>
      <p style="color: var(--text-muted); font-size: 15px; margin-bottom: 24px;">
        You have successfully reviewed all 24 anonymized audio clips.<br>
        Your responses have been saved to <code>blinded_listening_responses.csv</code>.
      </p>
      <div style="display: flex; gap: 12px; justify-content: center;">
        <button class="btn btn-primary" id="btn-download-csv">Download Responses CSV</button>
        <button class="btn btn-secondary" id="btn-review-again">Review Again</button>
      </div>
    </div>

  </div>
</div>

<script>
let manifest = [];
let currentIndex = 0;
let responses = {};

// Load manifest from server
async function init() {
  const res = await fetch('/api/manifest');
  manifest = await res.json();
  
  // Try loading saved responses if any
  try {
    const savedRes = await fetch('/api/load_responses');
    if (savedRes.ok) {
      const savedData = await savedRes.json();
      if (savedData && Object.keys(savedData).length > 0) {
        responses = savedData;
      }
    }
  } catch (e) {}

  renderClip(0);
}

function renderClip(index) {
  if (index < 0 || index >= manifest.length) return;
  currentIndex = index;

  const item = manifest[index];
  document.getElementById('clip-badge').innerText = item.blind_clip_id + ' (' + (index + 1) + '/' + manifest.length + ')';
  document.getElementById('progress-text').innerText = 'Clip ' + (index + 1) + ' of ' + manifest.length;
  document.getElementById('progress-fill').style.width = ((index + 1) / manifest.length * 100) + '%';

  const player = document.getElementById('audio-player');
  player.src = '/audio/' + encodeURIComponent(item.audio_file_path);

  // Clear selections
  document.querySelectorAll('#command-options .btn-option').forEach(el => el.classList.remove('selected'));
  document.querySelectorAll('#tone-options .btn-option').forEach(el => el.classList.remove('selected'));
  document.querySelectorAll('#intel-scale .scale-item').forEach(el => el.classList.remove('selected'));
  document.querySelectorAll('#natural-scale .scale-item').forEach(el => el.classList.remove('selected'));
  document.getElementById('defect-clipped').checked = false;
  document.getElementById('defect-distorted').checked = false;
  document.getElementById('defect-truncated').checked = false;
  document.getElementById('defect-buzzer').checked = false;
  document.getElementById('defect-clean').checked = false;
  document.getElementById('comments').value = '';

  // Restore saved response if existing
  const r = responses[item.blind_clip_id];
  if (r) {
    if (r.perceived_command) {
      const el = document.querySelector('#command-options .btn-option[data-val="' + r.perceived_command + '"]');
      if (el) el.classList.add('selected');
    }
    if (r.perceived_tone) {
      const el = document.querySelector('#tone-options .btn-option[data-val="' + r.perceived_tone + '"]');
      if (el) el.classList.add('selected');
    }
    if (r.intelligibility) {
      const el = document.querySelector('#intel-scale .scale-item[data-val="' + r.intelligibility + '"]');
      if (el) el.classList.add('selected');
    }
    if (r.naturalness) {
      const el = document.querySelector('#natural-scale .scale-item[data-val="' + r.naturalness + '"]');
      if (el) el.classList.add('selected');
    }
    document.getElementById('defect-clipped').checked = !!r.defect_clipped;
    document.getElementById('defect-distorted').checked = !!r.defect_distorted;
    document.getElementById('defect-truncated').checked = !!r.defect_truncated;
    document.getElementById('defect-buzzer').checked = !!r.defect_buzzer;
    document.getElementById('defect-clean').checked = !!r.defect_clean;
    document.getElementById('comments').value = r.comments || '';
  }

  document.getElementById('btn-prev').disabled = (index === 0);
  document.getElementById('btn-next').innerText = (index === manifest.length - 1) ? 'Finish & Save ✓' : 'Next Clip →';
  document.getElementById('status-msg').innerText = '';
}

// Option button selection handling
function setupGroup(containerId, allowEmpty) {
  document.querySelectorAll('#' + containerId + ' > *').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('#' + containerId + ' > *').forEach(b => b.classList.remove('selected'));
      btn.classList.add('selected');
    });
  });
}
setupGroup('command-options');
setupGroup('tone-options');
setupGroup('intel-scale');
setupGroup('natural-scale');

// Clean checkbox exclusivity
document.getElementById('defect-clean').addEventListener('change', (e) => {
  if (e.target.checked) {
    document.getElementById('defect-clipped').checked = false;
    document.getElementById('defect-distorted').checked = false;
    document.getElementById('defect-truncated').checked = false;
    document.getElementById('defect-buzzer').checked = false;
  }
});
['defect-clipped', 'defect-distorted', 'defect-truncated', 'defect-buzzer'].forEach(id => {
  document.getElementById(id).addEventListener('change', () => {
    if (document.getElementById(id).checked) {
      document.getElementById('defect-clean').checked = false;
    }
  });
});

function saveCurrentState() {
  const item = manifest[currentIndex];
  const cmd = document.querySelector('#command-options .btn-option.selected')?.getAttribute('data-val') || '';
  const tone = document.querySelector('#tone-options .btn-option.selected')?.getAttribute('data-val') || '';
  const intel = document.querySelector('#intel-scale .scale-item.selected')?.getAttribute('data-val') || '';
  const natural = document.querySelector('#natural-scale .scale-item.selected')?.getAttribute('data-val') || '';

  responses[item.blind_clip_id] = {
    blind_clip_id: item.blind_clip_id,
    audio_file_path: item.audio_file_path,
    perceived_command: cmd,
    defect_clipped: document.getElementById('defect-clipped').checked,
    defect_distorted: document.getElementById('defect-distorted').checked,
    defect_truncated: document.getElementById('defect-truncated').checked,
    defect_buzzer: document.getElementById('defect-buzzer').checked,
    defect_clean: document.getElementById('defect-clean').checked,
    perceived_tone: tone,
    intelligibility: intel,
    naturalness: natural,
    comments: document.getElementById('comments').value.trim()
  };
}

async function persistToServer() {
  saveCurrentState();
  try {
    const res = await fetch('/api/save_responses', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(responses)
    });
    if (res.ok) {
      document.getElementById('status-msg').innerText = 'Progress saved successfully.';
      setTimeout(() => { document.getElementById('status-msg').innerText = ''; }, 2000);
    }
  } catch (e) {
    console.error('Error saving:', e);
  }
}

document.getElementById('btn-save-draft').addEventListener('click', persistToServer);

document.getElementById('btn-prev').addEventListener('click', () => {
  saveCurrentState();
  if (currentIndex > 0) renderClip(currentIndex - 1);
});

document.getElementById('btn-next').addEventListener('click', async () => {
  saveCurrentState();
  const current = responses[manifest[currentIndex].blind_clip_id];
  if (!current.perceived_command) {
    alert('Please select a perceived command before proceeding.');
    return;
  }
  if (!current.intelligibility) {
    alert('Please provide an intelligibility score (1-5).');
    return;
  }
  if (!current.naturalness) {
    alert('Please provide a naturalness score (1-5).');
    return;
  }

  await persistToServer();

  if (currentIndex < manifest.length - 1) {
    renderClip(currentIndex + 1);
  } else {
    // Show finish panel
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
    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        path = url.path

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
            return

        elif path == "/api/manifest":
            items = []
            if os.path.exists(MANIFEST_PATH):
                with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        items.append({
                            "blind_clip_id": row["blind_clip_id"],
                            "audio_file_path": row["audio_file_path"]
                        })
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(items, ensure_ascii=False).encode("utf-8"))
            return

        elif path == "/api/load_responses":
            responses = {}
            if os.path.exists(RESPONSES_CSV_PATH):
                with open(RESPONSES_CSV_PATH, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        cid = row.get("blind_clip_id")
                        if cid:
                            responses[cid] = row
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(responses, ensure_ascii=False).encode("utf-8"))
            return

        elif path == "/api/export_csv":
            if os.path.exists(RESPONSES_CSV_PATH):
                with open(RESPONSES_CSV_PATH, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Disposition", "attachment; filename=\"blinded_listening_responses.csv\"")
                self.end_headers()
                self.wfile.write(content)
                return
            else:
                self.send_response(404)
                self.end_headers()
                return

        elif path.startswith("/audio/"):
            rel_audio_path = urllib.parse.unquote(path[len("/audio/"):])
            full_audio_path = os.path.join(ROOT_DIR, rel_audio_path)
            if not os.path.exists(full_audio_path):
                # Try direct path
                full_audio_path = os.path.abspath(rel_audio_path)
            if os.path.exists(full_audio_path) and full_audio_path.endswith(".wav"):
                try:
                    with open(full_audio_path, "rb") as f:
                        audio_bytes = f.read()
                    self.send_response(200)
                    self.send_header("Content-Type", "audio/wav")
                    self.send_header("Content-Length", str(len(audio_bytes)))
                    self.send_header("Accept-Ranges", "bytes")
                    self.end_headers()
                    self.wfile.write(audio_bytes)
                    return
                except Exception as e:
                    self.send_response(500)
                    self.end_headers()
                    return
            else:
                self.send_response(404)
                self.end_headers()
                return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        url = urllib.parse.urlparse(self.path)
        if url.path == "/api/save_responses":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            data = json.loads(body)

            # Write to CSV
            fieldnames = [
                "blind_clip_id", "audio_file_path", "perceived_command",
                "defect_clipped", "defect_distorted", "defect_truncated",
                "defect_buzzer", "defect_clean", "perceived_tone",
                "intelligibility", "naturalness", "comments"
            ]
            with open(RESPONSES_CSV_PATH, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for cid in sorted(data.keys()):
                    writer.writerow(data[cid])

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "saved_count": len(data)}).encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()

def run_cli_mode():
    print("=" * 70)
    print("Vietnamese KWS Blinded Listening Evaluation — Terminal Review Mode")
    print("=" * 70)
    print("Instructions: For each clip, listen to the audio file and enter your response.")
    print("Commands: TRAI, PHAI, LEN, XUONG, OTHER, UNCERTAIN\n")

    if not os.path.exists(MANIFEST_PATH):
        print(f"Error: Manifest not found at {MANIFEST_PATH}")
        return

    items = []
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        items = list(reader)

    responses = []
    valid_cmds = ["TRAI", "PHAI", "LEN", "XUONG", "OTHER", "UNCERTAIN"]

    for i, item in enumerate(items, 1):
        cid = item["blind_clip_id"]
        rel_path = item["audio_file_path"]
        full_path = os.path.join(ROOT_DIR, rel_path)
        print(f"\n[{i}/{len(items)}] Evaluating {cid}")
        print(f"Audio file: {full_path}")
        print("Play audio with your local media player if terminal cannot play audio directly.")

        # Input perceived command
        cmd = ""
        while cmd not in valid_cmds:
            cmd = input(f"Perceived command ({'/'.join(valid_cmds)}): ").strip().upper()

        clipped = input("Sounds clipped? (y/n): ").strip().lower().startswith("y")
        distorted = input("Sounds distorted/unnatural? (y/n): ").strip().lower().startswith("y")
        truncated = input("Sounds truncated/cut off? (y/n): ").strip().lower().startswith("y")

        intel = ""
        while intel not in ["1", "2", "3", "4", "5"]:
            intel = input("Intelligibility (1-5, 5=crystal clear): ").strip()

        natural = ""
        while natural not in ["1", "2", "3", "4", "5"]:
            natural = input("Naturalness (1-5, 5=like human): ").strip()

        comments = input("Comments (optional): ").strip()

        responses.append({
            "blind_clip_id": cid,
            "audio_file_path": rel_path,
            "perceived_command": cmd,
            "defect_clipped": clipped,
            "defect_distorted": distorted,
            "defect_truncated": truncated,
            "defect_buzzer": False,
            "defect_clean": not (clipped or distorted or truncated),
            "perceived_tone": "",
            "intelligibility": intel,
            "naturalness": natural,
            "comments": comments
        })

    fieldnames = [
        "blind_clip_id", "audio_file_path", "perceived_command",
        "defect_clipped", "defect_distorted", "defect_truncated",
        "defect_buzzer", "defect_clean", "perceived_tone",
        "intelligibility", "naturalness", "comments"
    ]
    with open(RESPONSES_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in responses:
            writer.writerow(r)

    print(f"\nResponses saved to: {RESPONSES_CSV_PATH}")

def main():
    parser = argparse.ArgumentParser(description="Blinded Listening Review Tool")
    parser.add_argument("--cli", action="store_true", help="Run interactive terminal evaluation")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically launch web browser")
    args = parser.parse_args()

    if args.cli:
        run_cli_mode()
        return

    server = HTTPServer(("127.0.0.1", PORT), BlindedReviewHandler)
    url = f"http://127.0.0.1:{PORT}"
    print("=" * 70)
    print(f"Blinded Listening Review Server running at: {url}")
    print(f"Serving manifest: {MANIFEST_PATH}")
    print("Responses will be saved to: " + RESPONSES_CSV_PATH)
    print("Press Ctrl+C to stop the server.")
    print("=" * 70)

    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping review server.")
        server.server_close()

if __name__ == "__main__":
    main()
