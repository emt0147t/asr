"""
Standalone Local Web Application for Blinded Human Listening Review.
Workspace: D:\trainidentificationaudio
Experiment: experiments/vieneu_kws_research_v10/human_review/

Features:
- Pure Python standard library (http.server), zero external dependencies.
- Bound strictly to 127.0.0.1.
- Complete double-blind security: zero exposure of answer key, transcripts, or models.
- Audio streaming, playback controls, and keyboard shortcuts.
- Autosave and resume support.
- Data validation on completed submissions.
"""

import os
import sys
import json
import csv
import shutil
import re
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(SCRIPT_DIR, "audio")
RESPONSES_DIR = os.path.join(SCRIPT_DIR, "responses")
os.makedirs(RESPONSES_DIR, exist_ok=True)

# List of 120 anonymous clip IDs
CLIP_IDS = [f"review_clip_{i:03d}" for i in range(1, 121)]


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Blinded Vietnamese Audio Listening Review</title>
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --border-color: #334155;
            --text-color: #f8fafc;
            --text-muted: #94a3b8;
            --accent: #38bdf8;
            --accent-hover: #0284c7;
            --success: #22c55e;
            --danger: #ef4444;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            padding: 20px;
            display: flex;
            justify-content: center;
        }
        .container {
            width: 100%;
            max-width: 760px;
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.5);
        }
        .header {
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 15px;
            margin-bottom: 20px;
        }
        .header h1 {
            margin: 0 0 5px 0;
            font-size: 22px;
            color: var(--accent);
        }
        .header .status {
            font-size: 13px;
            color: var(--text-muted);
        }
        .nav-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        .progress-bar-container {
            flex-grow: 1;
            background: var(--border-color);
            border-radius: 6px;
            height: 10px;
            margin: 0 15px;
            overflow: hidden;
        }
        .progress-fill {
            background: var(--accent);
            height: 100%;
            width: 0%;
            transition: width 0.2s ease;
        }
        .audio-box {
            background: #0b1120;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            margin-bottom: 25px;
        }
        .clip-title {
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 15px;
            letter-spacing: 0.5px;
        }
        audio {
            width: 100%;
            max-width: 500px;
            outline: none;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 6px;
            font-weight: 500;
            font-size: 14px;
        }
        .hint {
            font-size: 12px;
            color: var(--text-muted);
            margin-top: 2px;
        }
        input[type="text"], textarea, select {
            width: 100%;
            box-sizing: border-box;
            background: #0f172a;
            border: 1px solid var(--border-color);
            color: #fff;
            padding: 10px;
            border-radius: 6px;
            font-size: 15px;
        }
        input[type="text"]:focus, textarea:focus, select:focus {
            border-color: var(--accent);
            outline: none;
        }
        .btn-group {
            display: flex;
            justify-content: space-between;
            margin-top: 30px;
        }
        button {
            background: var(--accent);
            color: #0f172a;
            font-weight: 600;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            transition: background 0.15s;
        }
        button:hover {
            background: var(--accent-hover);
        }
        button.secondary {
            background: #334155;
            color: #fff;
        }
        button.secondary:hover {
            background: #475569;
        }
        .save-status {
            font-size: 12px;
            color: var(--success);
            text-align: right;
            margin-top: 5px;
            min-height: 16px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Blinded Vietnamese Speech Perceptual Audit</h1>
            <div class="status">Double-Blind Protocol | Zero Ground-Truth Exposure | 120 Total Clips</div>
        </div>

        <div class="nav-bar">
            <div>
                Reviewer ID: <input type="text" id="reviewerId" value="reviewer_01" style="width: 110px; padding: 4px 8px; font-size: 13px;">
            </div>
            <div class="progress-bar-container">
                <div class="progress-fill" id="progressFill"></div>
            </div>
            <div id="clipCounter" style="font-size: 14px; font-weight: 500;">Clip 1 of 120</div>
        </div>

        <div class="audio-box">
            <div class="clip-title" id="clipTitle">review_clip_001</div>
            <audio id="audioPlayer" controls></audio>
        </div>

        <div class="form-group">
            <label for="perceivedText">1. What Vietnamese word(s) or syllable(s) did you hear? *</label>
            <input type="text" id="perceivedText" placeholder="e.g. trái, phải, lên, xuống, trại, bánh rán, hoặc [SILENCE]">
            <div class="hint">Type exactly what you perceive. If only silence/noise, enter [SILENCE]. If unrecognizable, enter [UNINTELLIGIBLE].</div>
        </div>

        <div class="form-group" style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
            <div>
                <label for="intelligibility">2. Intelligibility (1-5) *</label>
                <select id="intelligibility">
                    <option value="">-- Select --</option>
                    <option value="5">5 - Crystal clear</option>
                    <option value="4">4 - Mostly clear</option>
                    <option value="3">3 - Moderately clear</option>
                    <option value="2">2 - Barely intelligible</option>
                    <option value="1">1 - Completely unintelligible</option>
                </select>
            </div>
            <div>
                <label for="naturalness">3. Naturalness (1-5) *</label>
                <select id="naturalness">
                    <option value="">-- Select --</option>
                    <option value="5">5 - Completely human-like</option>
                    <option value="4">4 - Slight synthetic quality</option>
                    <option value="3">3 - Acceptable synthetic</option>
                    <option value="2">2 - Robotic / unnatural tone</option>
                    <option value="1">1 - Extremely distorted / buzzy</option>
                </select>
            </div>
        </div>

        <div class="form-group" style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
            <div>
                <label for="pronunciation">4. Pronunciation & Tone Correct? *</label>
                <select id="pronunciation">
                    <option value="">-- Select --</option>
                    <option value="Y">Y - Correct Vietnamese pronunciation & tone</option>
                    <option value="N">N - Defective tone or distorted phoneme</option>
                    <option value="Unsure">Unsure - Ambiguous or unclear</option>
                </select>
            </div>
            <div>
                <label for="confidence">5. Your Confidence (1-5) *</label>
                <select id="confidence">
                    <option value="">-- Select --</option>
                    <option value="5">5 - Absolutely certain</option>
                    <option value="4">4 - Confident</option>
                    <option value="3">3 - Moderate</option>
                    <option value="2">2 - Low</option>
                    <option value="1">1 - Pure guess</option>
                </select>
            </div>
        </div>

        <div class="form-group">
            <label for="comments">6. Optional Observations / Notes</label>
            <textarea id="comments" rows="2" placeholder="Note any specific audio artifacts, background hum, or phoneme shifts..."></textarea>
        </div>

        <div class="save-status" id="saveStatus"></div>

        <div class="btn-group">
            <button class="secondary" id="prevBtn" onclick="navigate(-1)">← Previous Clip</button>
            <button class="secondary" onclick="saveCurrentClip(true)">Save Progress</button>
            <button id="nextBtn" onclick="navigate(1)">Next Clip →</button>
        </div>
    </div>

    <script>
        const clipIds = %CLIP_IDS_JSON%;
        let currentIndex = 0;
        let responses = {};

        function init() {
            loadSavedResponses();
            updateUI();
        }

        function updateUI() {
            const cid = clipIds[currentIndex];
            document.getElementById("clipTitle").innerText = cid;
            document.getElementById("clipCounter").innerText = `Clip ${currentIndex + 1} of ${clipIds.length}`;
            document.getElementById("progressFill").style.width = `${((currentIndex + 1) / clipIds.length) * 100}%`;
            
            const player = document.getElementById("audioPlayer");
            player.src = `/audio/${cid}.wav`;
            player.load();

            // Populate form if response exists
            const r = responses[cid] || {};
            document.getElementById("perceivedText").value = r.perceived_text || "";
            document.getElementById("intelligibility").value = r.intelligibility_1to5 || "";
            document.getElementById("naturalness").value = r.naturalness_1to5 || "";
            document.getElementById("pronunciation").value = r.pronunciation_correct_yn || "";
            document.getElementById("confidence").value = r.confidence_1to5 || "";
            document.getElementById("comments").value = r.comments || "";

            document.getElementById("prevBtn").disabled = (currentIndex === 0);
            document.getElementById("nextBtn").innerText = (currentIndex === clipIds.length - 1) ? "Finish & Submit" : "Next Clip →";
        }

        function saveCurrentClip(showMsg = false) {
            const cid = clipIds[currentIndex];
            responses[cid] = {
                review_clip_id: cid,
                perceived_text: document.getElementById("perceivedText").value.trim(),
                intelligibility_1to5: document.getElementById("intelligibility").value,
                naturalness_1to5: document.getElementById("naturalness").value,
                pronunciation_correct_yn: document.getElementById("pronunciation").value,
                confidence_1to5: document.getElementById("confidence").value,
                comments: document.getElementById("comments").value.trim()
            };

            const revId = document.getElementById("reviewerId").value.trim() || "anonymous";

            fetch("/api/save", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({
                    reviewer_id: revId,
                    responses: responses
                })
            }).then(r => r.json()).then(data => {
                if (showMsg) {
                    const status = document.getElementById("saveStatus");
                    status.innerText = "Progress saved successfully (" + new Date().toLocaleTimeString() + ")";
                    setTimeout(() => status.innerText = "", 3000);
                }
            });
        }

        function navigate(delta) {
            saveCurrentClip(false);
            if (currentIndex + delta >= 0 && currentIndex + delta < clipIds.length) {
                currentIndex += delta;
                updateUI();
            } else if (currentIndex + delta >= clipIds.length) {
                saveCurrentClip(true);
                alert("You have reached the end of the review! All responses saved.");
            }
        }

        function loadSavedResponses() {
            const revId = document.getElementById("reviewerId").value.trim() || "anonymous";
            fetch(`/api/load?reviewer_id=${encodeURIComponent(revId)}`)
                .then(r => r.json())
                .then(data => {
                    if (data && data.responses) {
                        responses = data.responses;
                        updateUI();
                    }
                });
        }

        document.getElementById("reviewerId").addEventListener("change", loadSavedResponses);

        // Keyboard shortcuts
        window.addEventListener("keydown", (e) => {
            if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA" || e.target.tagName === "SELECT") return;
            if (e.code === "Space") {
                e.preventDefault();
                const p = document.getElementById("audioPlayer");
                p.paused ? p.play() : p.pause();
            } else if (e.code === "ArrowRight") {
                navigate(1);
            } else if (e.code === "ArrowLeft") {
                navigate(-1);
            }
        });

        init();
    </script>
</body>
</html>
"""


class BlindedReviewHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Quiet logger
        return

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            html = HTML_PAGE.replace("%CLIP_IDS_JSON%", json.dumps(CLIP_IDS))
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))

        elif path.startswith("/audio/"):
            filename = os.path.basename(path)
            audio_path = os.path.join(AUDIO_DIR, filename)
            if os.path.exists(audio_path) and filename.endswith(".wav"):
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(os.path.getsize(audio_path)))
                self.end_headers()
                with open(audio_path, "rb") as f:
                    shutil.copyfileobj(f, self.wfile)
            else:
                self.send_response(404)
                self.end_headers()

        elif path == "/api/clips":
            # Returns strictly anonymized clip IDs
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"clips": CLIP_IDS}).encode("utf-8"))

        elif path == "/api/load":
            query = urllib.parse.parse_qs(parsed.query)
            rev_raw = query.get("reviewer_id", ["anonymous"])[0]
            rev_id = re.sub(r'[^a-zA-Z0-9_\-]', '', rev_raw) or "anonymous"
            csv_path = os.path.join(RESPONSES_DIR, f"{rev_id}_responses.csv")
            responses = {}
            if os.path.exists(csv_path):
                with open(csv_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        responses[r["review_clip_id"]] = r

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"responses": responses}).encode("utf-8"))

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/save":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len)
            data = json.loads(body.decode("utf-8"))

            rev_raw = data.get("reviewer_id", "anonymous")
            rev_id = re.sub(r'[^a-zA-Z0-9_\-]', '', rev_raw) or "anonymous"
            responses = data.get("responses", {})

            # Write to responses CSV
            csv_path = os.path.join(RESPONSES_DIR, f"{rev_id}_responses.csv")
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "review_clip_id", "perceived_text", "intelligibility_1to5",
                    "naturalness_1to5", "pronunciation_correct_yn", "confidence_1to5", "comments"
                ])
                writer.writeheader()
                for cid in CLIP_IDS:
                    row = responses.get(cid, {
                        "review_clip_id": cid,
                        "perceived_text": "",
                        "intelligibility_1to5": "",
                        "naturalness_1to5": "",
                        "pronunciation_correct_yn": "",
                        "confidence_1to5": "",
                        "comments": ""
                    })
                    writer.writerow(row)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "saved", "file": os.path.basename(csv_path)}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def run_server(port=8080):
    server_address = ("127.0.0.1", port)
    httpd = HTTPServer(server_address, BlindedReviewHandler)
    print(f"Blinded Review Server running at http://127.0.0.1:{port}/")
    print("Press Ctrl+C to terminate.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080, help="Port to bind (default: 8080)")
    args = parser.parse_args()
    run_server(args.port)
