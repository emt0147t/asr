"""
Script: generate_figures.py
Generates 8 publication-grade vector SVG figures for Objective 1 documentation.
Strictly reflects verified empirical repository evidence, audited manifests, and evaluated checkpoints.
Zero external dependencies beyond standard library (pure Python XML/SVG generation).
"""

import os
from pathlib import Path

FIGURES_DIR = Path("docs/objective_1/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Common SVG styles and helpers
COMMON_STYLE = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    text { font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
    .code { font-family: 'JetBrains Mono', Consolas, Monaco, monospace; }
    .shadow { filter: drop-shadow(0px 4px 8px rgba(0, 0, 0, 0.06)); }
    .card-shadow { filter: drop-shadow(0px 2px 4px rgba(15, 23, 42, 0.08)); }
"""

def save_svg(filename: str, content: str):
    filepath = FIGURES_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content.strip())
    print(f"Generated: {filepath}")

# -----------------------------------------------------------------------------
# Figure 1: End-to-End System Architecture
# -----------------------------------------------------------------------------
def generate_figure_01():
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 680" width="1200" height="680">
  <defs>
    <style>{COMMON_STYLE}</style>
    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0F172A"/>
      <stop offset="100%" stop-color="#1E293B"/>
    </linearGradient>
    <linearGradient id="cardGrad1" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#1E293B"/>
      <stop offset="100%" stop-color="#334155"/>
    </linearGradient>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#38BDF8"/>
    </marker>
  </defs>

  <rect width="1200" height="680" fill="url(#bgGrad)" rx="16"/>

  <text x="50" y="55" font-size="24" font-weight="700" fill="#F8FAFC">Figure 1: End-to-End Real-Time Vietnamese KWS System Architecture</text>
  <text x="50" y="82" font-size="14" fill="#94A3B8">Streaming 16 kHz Audio Pipeline &bull; Temporal Convolutional Backbone (VietnameseTCResNet, 272.9k Params) &bull; State Machine Gating</text>

  <!-- Stage 1: Audio Input & Buffering -->
  <g transform="translate(50, 120)">
    <rect width="230" height="490" rx="12" fill="url(#cardGrad1)" stroke="#334155" stroke-width="1.5" class="card-shadow"/>
    <path d="M 0 12 Q 0 0 12 0 L 218 0 Q 230 0 230 12 L 230 42 L 0 42 Z" fill="#0284C7"/>
    <text x="115" y="26" font-size="14" font-weight="700" fill="#FFFFFF" text-anchor="middle">STAGE 1: AUDIO INPUT</text>
    
    <rect x="20" y="60" width="190" height="65" rx="8" fill="#0F172A" stroke="#0284C7" stroke-width="1"/>
    <text x="115" y="85" font-size="13" font-weight="600" fill="#E2E8F0" text-anchor="middle">Microphone Stream</text>
    <text x="115" y="105" font-size="11" fill="#38BDF8" text-anchor="middle" class="code">16 kHz &bull; Mono &bull; PCM16</text>

    <line x1="115" y1="125" x2="115" y2="150" stroke="#38BDF8" stroke-width="2" marker-end="url(#arrow)"/>
    <rect x="20" y="150" width="190" height="75" rx="8" fill="#0F172A" stroke="#334155" stroke-width="1"/>
    <text x="115" y="175" font-size="13" font-weight="600" fill="#E2E8F0" text-anchor="middle">Chunk Ingestion</text>
    <text x="115" y="195" font-size="11" fill="#94A3B8" text-anchor="middle">Step size: 100 ms (1,600 spl)</text>
    <text x="115" y="212" font-size="10" fill="#64748B" text-anchor="middle">10 inferences / sec rate</text>

    <line x1="115" y1="225" x2="115" y2="250" stroke="#38BDF8" stroke-width="2" marker-end="url(#arrow)"/>
    <rect x="20" y="250" width="190" height="95" rx="8" fill="#0F172A" stroke="#0284C7" stroke-width="1.5"/>
    <text x="115" y="275" font-size="13" font-weight="600" fill="#38BDF8" text-anchor="middle">Circular Ring Buffer</text>
    <text x="115" y="295" font-size="11" fill="#E2E8F0" text-anchor="middle" class="code">Window = 1.0 s (16,000 spl)</text>
    <text x="115" y="315" font-size="11" fill="#94A3B8" text-anchor="middle">Overlap: 90% (900 ms)</text>
    <text x="115" y="332" font-size="10" fill="#64748B" text-anchor="middle">FIFO sliding window</text>

    <line x1="115" y1="345" x2="115" y2="370" stroke="#38BDF8" stroke-width="2" marker-end="url(#arrow)"/>
    <rect x="20" y="370" width="190" height="85" rx="8" fill="#0F172A" stroke="#334155" stroke-width="1"/>
    <text x="115" y="395" font-size="13" font-weight="600" fill="#E2E8F0" text-anchor="middle">Signal Conditioning</text>
    <text x="115" y="415" font-size="11" fill="#94A3B8" text-anchor="middle">Pre-emphasis: &alpha; = 0.97</text>
    <text x="115" y="433" font-size="11" fill="#94A3B8" text-anchor="middle">Dynamic range normalization</text>
  </g>

  <line x1="280" y1="365" x2="310" y2="365" stroke="#38BDF8" stroke-width="3" marker-end="url(#arrow)"/>

  <!-- Stage 2: Feature Extraction -->
  <g transform="translate(315, 120)">
    <rect width="240" height="490" rx="12" fill="url(#cardGrad1)" stroke="#334155" stroke-width="1.5" class="card-shadow"/>
    <path d="M 0 12 Q 0 0 12 0 L 228 0 Q 240 0 240 12 L 240 42 L 0 42 Z" fill="#7C3AED"/>
    <text x="120" y="26" font-size="14" font-weight="700" fill="#FFFFFF" text-anchor="middle">STAGE 2: SPECTROGRAM</text>

    <rect x="20" y="60" width="200" height="75" rx="8" fill="#0F172A" stroke="#7C3AED" stroke-width="1"/>
    <text x="120" y="85" font-size="13" font-weight="600" fill="#E2E8F0" text-anchor="middle">STFT Framing</text>
    <text x="120" y="105" font-size="11" fill="#C4B5FD" text-anchor="middle">Win = 480 spl (30 ms)</text>
    <text x="120" y="122" font-size="11" fill="#94A3B8" text-anchor="middle">Hop = 160 spl (10 ms)</text>

    <line x1="120" y1="135" x2="120" y2="160" stroke="#A78BFA" stroke-width="2" marker-end="url(#arrow)"/>

    <rect x="20" y="160" width="200" height="85" rx="8" fill="#0F172A" stroke="#7C3AED" stroke-width="1"/>
    <text x="120" y="185" font-size="13" font-weight="600" fill="#E2E8F0" text-anchor="middle">Mel Filterbank</text>
    <text x="120" y="205" font-size="11" fill="#C4B5FD" text-anchor="middle">40 Triangular Filters</text>
    <text x="120" y="223" font-size="11" fill="#94A3B8" text-anchor="middle">Freq range: 20 Hz &ndash; 7,600 Hz</text>

    <line x1="120" y1="245" x2="120" y2="270" stroke="#A78BFA" stroke-width="2" marker-end="url(#arrow)"/>

    <rect x="20" y="270" width="200" height="85" rx="8" fill="#0F172A" stroke="#7C3AED" stroke-width="1"/>
    <text x="120" y="295" font-size="13" font-weight="600" fill="#E2E8F0" text-anchor="middle">Log &amp; Normalization</text>
    <text x="120" y="315" font-size="11" fill="#C4B5FD" text-anchor="middle">Log Compression: ln(S + 1e-6)</text>
    <text x="120" y="333" font-size="11" fill="#94A3B8" text-anchor="middle">Per-utterance CMVN</text>

    <line x1="120" y1="355" x2="120" y2="385" stroke="#A78BFA" stroke-width="2" marker-end="url(#arrow)"/>

    <rect x="20" y="385" width="200" height="70" rx="8" fill="#0F172A" stroke="#A78BFA" stroke-width="1.5"/>
    <text x="120" y="410" font-size="13" font-weight="600" fill="#A78BFA" text-anchor="middle">Input Feature Tensor</text>
    <text x="120" y="432" font-size="12" fill="#FFFFFF" text-anchor="middle" class="code">[1, 40, 101] (squeeze channel)</text>
  </g>

  <line x1="555" y1="365" x2="585" y2="365" stroke="#A78BFA" stroke-width="3" marker-end="url(#arrow)"/>

  <!-- Stage 3: VietnameseTCResNet Backbone -->
  <g transform="translate(590, 120)">
    <rect width="280" height="490" rx="12" fill="url(#cardGrad1)" stroke="#334155" stroke-width="1.5" class="card-shadow"/>
    <path d="M 0 12 Q 0 0 12 0 L 268 0 Q 280 0 280 12 L 280 42 L 0 42 Z" fill="#D97706"/>
    <text x="140" y="26" font-size="14" font-weight="700" fill="#FFFFFF" text-anchor="middle">STAGE 3: TC-ResNet BACKBONE</text>

    <rect x="20" y="55" width="240" height="50" rx="6" fill="#0F172A" stroke="#D97706" stroke-width="1"/>
    <text x="140" y="76" font-size="12" font-weight="600" fill="#FCD34D" text-anchor="middle">Conv1 (Initial Frequency Conv)</text>
    <text x="140" y="94" font-size="10" fill="#94A3B8" text-anchor="middle" class="code">Conv1D(c=24, k=3, s=1) &bull; BN &bull; ReLU</text>

    <g transform="translate(20, 115)">
      <rect width="240" height="185" rx="8" fill="#172554" stroke="#3B82F6" stroke-width="1.5"/>
      <text x="120" y="20" font-size="12" font-weight="700" fill="#60A5FA" text-anchor="middle">6x Temporal Residual Blocks</text>
      
      <rect x="10" y="32" width="220" height="35" rx="4" fill="#0F172A"/>
      <text x="120" y="54" font-size="11" fill="#93C5FD" text-anchor="middle">ResBlock 1 &amp; 2 (C_out = 36, Dil=1)</text>

      <rect x="10" y="73" width="220" height="35" rx="4" fill="#0F172A"/>
      <text x="120" y="95" font-size="11" fill="#93C5FD" text-anchor="middle">ResBlock 3 &amp; 4 (C_out = 48, Dil=2)</text>

      <rect x="10" y="114" width="220" height="35" rx="4" fill="#0F172A"/>
      <text x="120" y="136" font-size="11" fill="#93C5FD" text-anchor="middle">ResBlock 5 &amp; 6 (C_out = 64, Dil=4)</text>

      <text x="120" y="168" font-size="9" fill="#94A3B8" text-anchor="middle">1D Temporal Conv (k=9) + Skip Connection</text>
    </g>

    <rect x="20" y="310" width="240" height="50" rx="6" fill="#0F172A" stroke="#334155" stroke-width="1"/>
    <text x="140" y="331" font-size="12" font-weight="600" fill="#E2E8F0" text-anchor="middle">Mean + Max Pooling Layer</text>
    <text x="140" y="348" font-size="10" fill="#94A3B8" text-anchor="middle" class="code">Concat(MeanPool, MaxPool) &rarr; [1, 128]</text>

    <rect x="20" y="370" width="240" height="55" rx="6" fill="#0F172A" stroke="#D97706" stroke-width="1"/>
    <text x="140" y="391" font-size="12" font-weight="600" fill="#FCD34D" text-anchor="middle">Linear Classification Head</text>
    <text x="140" y="411" font-size="10" fill="#94A3B8" text-anchor="middle" class="code">Dropout(0.2) &rarr; Linear(128, 6) &rarr; Logits</text>

    <text x="140" y="445" font-size="11" font-weight="600" fill="#38BDF8" text-anchor="middle">272,934 Parameters &bull; 1,068 KiB FP32</text>
    <text x="140" y="462" font-size="10" fill="#94A3B8" text-anchor="middle">CPU Frame Latency: 0.35 ms</text>
  </g>

  <line x1="870" y1="365" x2="900" y2="365" stroke="#F59E0B" stroke-width="3" marker-end="url(#arrow)"/>

  <!-- Stage 4: Post-Processing & Streaming Gating -->
  <g transform="translate(905, 120)">
    <rect width="245" height="490" rx="12" fill="url(#cardGrad1)" stroke="#334155" stroke-width="1.5" class="card-shadow"/>
    <path d="M 0 12 Q 0 0 12 0 L 233 0 Q 245 0 245 12 L 245 42 L 0 42 Z" fill="#059669"/>
    <text x="122" y="26" font-size="14" font-weight="700" fill="#FFFFFF" text-anchor="middle">STAGE 4: STREAMING GATE</text>

    <rect x="20" y="60" width="205" height="75" rx="8" fill="#0F172A" stroke="#059669" stroke-width="1"/>
    <text x="122" y="85" font-size="13" font-weight="600" fill="#E2E8F0" text-anchor="middle">EMA Posterior Smoothing</text>
    <text x="122" y="105" font-size="11" fill="#34D399" text-anchor="middle" class="code">p_t = &alpha;&middot;p_raw + (1-&alpha;)&middot;p_t-1</text>
    <text x="122" y="122" font-size="10" fill="#94A3B8" text-anchor="middle">&alpha; = 0.60 (smoothing factor)</text>

    <line x1="122" y1="135" x2="122" y2="160" stroke="#34D399" stroke-width="2" marker-end="url(#arrow)"/>

    <rect x="20" y="160" width="205" height="85" rx="8" fill="#0F172A" stroke="#059669" stroke-width="1"/>
    <text x="122" y="185" font-size="13" font-weight="600" fill="#E2E8F0" text-anchor="middle">Dual-Threshold Gate</text>
    <text x="122" y="205" font-size="11" fill="#34D399" text-anchor="middle">Operating Thresh: &tau; &ge; 0.70</text>
    <text x="122" y="222" font-size="11" fill="#94A3B8" text-anchor="middle">Margin over UNKNOWN &ge; 0.15</text>
    <text x="122" y="237" font-size="10" fill="#64748B" text-anchor="middle">M=2 frame persistence</text>

    <line x1="122" y1="245" x2="122" y2="270" stroke="#34D399" stroke-width="2" marker-end="url(#arrow)"/>

    <rect x="20" y="270" width="205" height="75" rx="8" fill="#0F172A" stroke="#059669" stroke-width="1"/>
    <text x="122" y="295" font-size="13" font-weight="600" fill="#E2E8F0" text-anchor="middle">Refractory Lockout</text>
    <text x="122" y="315" font-size="11" fill="#34D399" text-anchor="middle">Cooldown Window: 600 ms</text>
    <text x="122" y="332" font-size="10" fill="#94A3B8" text-anchor="middle">Zero duplicate activations</text>

    <line x1="122" y1="345" x2="122" y2="375" stroke="#34D399" stroke-width="2" marker-end="url(#arrow)"/>

    <rect x="20" y="375" width="205" height="95" rx="8" fill="#064E3B" stroke="#10B981" stroke-width="1.5"/>
    <text x="122" y="400" font-size="12" font-weight="700" fill="#A7F3D0" text-anchor="middle">DISPATCHED ACTION</text>
    <text x="122" y="423" font-size="14" font-weight="800" fill="#FFFFFF" text-anchor="middle">TRAI | PHAI | LEN | XUONG</text>
    <text x="122" y="445" font-size="11" fill="#6EE7B7" text-anchor="middle">Trigger latency: ~706 ms</text>
    <text x="122" y="460" font-size="10" fill="#A7F3D0" text-anchor="middle">Single-frame inference: 0.35 ms</text>
  </g>

  <!-- Bottom Legend / Summary -->
  <rect x="50" y="625" width="1100" height="35" rx="6" fill="#1E293B" stroke="#334155"/>
  <text x="600" y="647" font-size="12" fill="#94A3B8" text-anchor="middle">Real-Time Invariant: Frame CPU Inference: 0.35 ms (at 100 ms hop) &bull; Continuous Detection Latency: ~706 ms &bull; Evaluated Stream Duration: 8.0 Minutes</text>
</svg>"""
    save_svg("figure_01_end_to_end_system_architecture.svg", svg)

# -----------------------------------------------------------------------------
# Figure 2: Dataset Acquisition & Provenance Flow
# -----------------------------------------------------------------------------
def generate_figure_02():
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 700" width="1200" height="700">
  <defs>
    <style>{COMMON_STYLE}</style>
    <linearGradient id="bgGrad2" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0F172A"/>
      <stop offset="100%" stop-color="#1E293B"/>
    </linearGradient>
    <marker id="arrowCyan" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#0EA5E9"/>
    </marker>
    <marker id="arrowRose" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#F43F5E"/>
    </marker>
  </defs>

  <rect width="1200" height="700" fill="url(#bgGrad2)" rx="16"/>

  <text x="50" y="50" font-size="24" font-weight="700" fill="#F8FAFC">Figure 2: Dataset Acquisition, Provenance Tracking &amp; Ingestion Lineage</text>
  <text x="50" y="75" font-size="14" fill="#94A3B8">Audited Data Sources: Synthetic Calibration Commands, Google FLEURS Negatives, Husein Noise &bull; 0 Genuine Human Takes</text>

  <!-- Left: Raw Audio Ingestion Tracks -->
  <g transform="translate(50, 105)">
    <rect width="320" height="250" rx="10" fill="#1E293B" stroke="#38BDF8" stroke-width="1.5"/>
    <path d="M 0 10 Q 0 0 10 0 L 310 0 Q 320 0 320 10 L 320 35 L 0 35 Z" fill="#0369A1" opacity="0.4"/>
    <text x="160" y="24" font-size="13" font-weight="700" fill="#38BDF8" text-anchor="middle">APPROVED PIPELINE AUDIO (2,099 WAVs)</text>
    
    <rect x="15" y="45" width="290" height="55" rx="6" fill="#0F172A" stroke="#0284C7"/>
    <text x="25" y="65" font-size="12" font-weight="600" fill="#E2E8F0">1. Parametric Synthetic Commands</text>
    <text x="25" y="82" font-size="11" fill="#38BDF8" class="code">TRAI, PHAI, LEN, XUONG (1,000 files)</text>
    <text x="25" y="94" font-size="10" fill="#94A3B8">Source: calibration_acoustic_generator_v1</text>

    <rect x="15" y="110" width="290" height="55" rx="6" fill="#0F172A" stroke="#334155"/>
    <text x="25" y="130" font-size="12" font-weight="600" fill="#E2E8F0">2. Speech Distractors (UNKNOWN)</text>
    <text x="25" y="147" font-size="11" fill="#94A3B8" class="code">Google FLEURS (579) + Confusing Words (120)</text>
    <text x="25" y="159" font-size="10" fill="#64748B">Total: 699 approved files &bull; License: CC-BY-4.0</text>

    <rect x="15" y="175" width="290" height="55" rx="6" fill="#0F172A" stroke="#334155"/>
    <text x="25" y="195" font-size="12" font-weight="600" fill="#E2E8F0">3. Environmental Noise (SILENCE)</text>
    <text x="25" y="212" font-size="11" fill="#94A3B8" class="code">husein_mic_noise (400 approved files)</text>
    <text x="25" y="224" font-size="10" fill="#64748B">License: CC-BY-4.0 &bull; Handling, rumble, quiet</text>
  </g>

  <!-- Track B: Human Field Recording Campaign Track -->
  <g transform="translate(50, 380)">
    <rect width="320" height="280" rx="10" fill="#1E293B" stroke="#F43F5E" stroke-width="1.5"/>
    <path d="M 0 10 Q 0 0 10 0 L 310 0 Q 320 0 320 10 L 320 35 L 0 35 Z" fill="#BE123C" opacity="0.3"/>
    <text x="160" y="24" font-size="13" font-weight="700" fill="#FB7185" text-anchor="middle">HUMAN FIELD CAMPAIGN (CURRENT STATE)</text>

    <rect x="15" y="45" width="290" height="60" rx="6" fill="#0F172A" stroke="#F43F5E"/>
    <text x="25" y="65" font-size="12" font-weight="600" fill="#FB7185">Consent Ledger Status</text>
    <text x="25" y="82" font-size="11" fill="#CBD5E1">Recorded: 2 entries in ledger</text>
    <text x="25" y="97" font-size="10" fill="#F43F5E" font-weight="600">&bull; ALL 2 ARE HARNESS FIXTURES (0 genuine)</text>

    <rect x="15" y="115" width="290" height="60" rx="6" fill="#0F172A" stroke="#F43F5E"/>
    <text x="25" y="135" font-size="12" font-weight="600" fill="#FB7185">Field Collection Manifest</text>
    <text x="25" y="152" font-size="11" fill="#CBD5E1">Manifest entries: 74 test fixture takes</text>
    <text x="25" y="167" font-size="10" fill="#F43F5E" font-weight="600">&bull; ALL 74 ARE HARNESS FIXTURES (0 genuine)</text>

    <rect x="15" y="185" width="290" height="80" rx="6" fill="#2A1215" stroke="#E11D48"/>
    <text x="25" y="205" font-size="12" font-weight="700" fill="#FDA4AF">Audited Human Speech: 0 Utterances</text>
    <text x="25" y="222" font-size="11" fill="#FECDD3">Target Quota: 1,200 genuine utterances</text>
    <text x="25" y="238" font-size="10" fill="#FDA4AF">Dialect Target: 50% North, 25% Central, 25% South</text>
    <text x="25" y="253" font-size="10" fill="#FECDD3">Completion: 0.0% &bull; CRITICAL P0 BLOCKER</text>
  </g>

  <!-- Middle: Ingestion Quality Gate -->
  <g transform="translate(420, 105)">
    <rect width="360" height="555" rx="10" fill="#1E293B" stroke="#475569" stroke-width="1.5"/>
    <path d="M 0 10 Q 0 0 10 0 L 350 0 Q 360 0 360 10 L 360 35 L 0 35 Z" fill="#334155" opacity="0.6"/>
    <text x="180" y="24" font-size="13" font-weight="700" fill="#F1F5F9" text-anchor="middle">INGESTION QUALITY GATE (AUDIT RULES)</text>

    <rect x="20" y="50" width="320" height="75" rx="6" fill="#0F172A" stroke="#64748B"/>
    <text x="35" y="72" font-size="12" font-weight="600" fill="#38BDF8">1. Acoustic Format Standardization</text>
    <text x="35" y="90" font-size="11" fill="#CBD5E1">&bull; Sample Rate: Exactly 16,000 Hz</text>
    <text x="35" y="105" font-size="11" fill="#CBD5E1">&bull; Channels: Mono &bull; Depth: 16-bit PCM</text>
    <text x="35" y="118" font-size="10" fill="#94A3B8">&bull; Length: Exactly 16,000 samples (1.000 s)</text>

    <rect x="20" y="135" width="320" height="115" rx="6" fill="#0F172A" stroke="#64748B"/>
    <text x="35" y="157" font-size="12" font-weight="600" fill="#FCD34D">2. Automated Quality Filtering</text>
    <text x="35" y="175" font-size="11" fill="#CBD5E1">&bull; Clipping Check: &ge; 3 consecutive samples at 0 dBFS</text>
    <text x="35" y="190" font-size="11" fill="#CBD5E1">&bull; Silence Floor: RMS &lt; -45.0 dBFS &rarr; Quarantined</text>
    <text x="35" y="205" font-size="11" fill="#CBD5E1">&bull; Duration Check: 0.25s &le; duration &le; 5.0s</text>
    <text x="35" y="220" font-size="11" fill="#CBD5E1">&bull; 141 FLEURS raw clips quarantined for low RMS</text>
    <text x="35" y="235" font-size="10" fill="#94A3B8">&bull; Format: WAV / M4A (M4A decoding verified in tests)</text>

    <rect x="20" y="260" width="320" height="95" rx="6" fill="#0F172A" stroke="#64748B"/>
    <text x="35" y="282" font-size="12" font-weight="600" fill="#34D399">3. Integrity &amp; Provenance Checks</text>
    <text x="35" y="300" font-size="11" fill="#CBD5E1">&bull; SHA-256 Collision Check: Prevents duplicates</text>
    <text x="35" y="315" font-size="11" fill="#CBD5E1">&bull; Consent Verification: Speaker ID must exist in ledger</text>
    <text x="35" y="330" font-size="11" fill="#CBD5E1">&bull; Path Traversal Protection: Sanitizes filenames</text>
    <text x="35" y="343" font-size="10" fill="#94A3B8">&bull; Metadata validation: dialect, device, environment</text>

    <rect x="20" y="365" width="320" height="90" rx="6" fill="#0F172A" stroke="#64748B"/>
    <text x="35" y="387" font-size="12" font-weight="600" fill="#C084FC">4. Leakage-Proof Splitting Protocol</text>
    <text x="35" y="405" font-size="11" fill="#CBD5E1">&bull; Speaker-Disjoint Partitioning (0 speaker leak)</text>
    <text x="35" y="420" font-size="11" fill="#CBD5E1">&bull; Train (134 spk), Val (51 spk), Test (148 spk)</text>
    <text x="35" y="435" font-size="11" fill="#CBD5E1">&bull; Frozen Test Manifest (SHA-256 locked)</text>
    <text x="35" y="448" font-size="10" fill="#94A3B8">&bull; Total Approved Pool = 2,099 Files</text>

    <rect x="20" y="465" width="150" height="75" rx="6" fill="#2A1215" stroke="#E11D48"/>
    <text x="95" y="488" font-size="11" font-weight="700" fill="#FDA4AF" text-anchor="middle">QUARANTINE</text>
    <text x="95" y="508" font-size="10" fill="#FECDD3" text-anchor="middle">141 raw FLEURS clips</text>
    <text x="95" y="525" font-size="9" fill="#94A3B8" text-anchor="middle">Excluded from splits</text>

    <rect x="190" y="465" width="150" height="75" rx="6" fill="#064E3B" stroke="#10B981"/>
    <text x="265" y="488" font-size="11" font-weight="700" fill="#A7F3D0" text-anchor="middle">APPROVED</text>
    <text x="265" y="508" font-size="10" fill="#6EE7B7" text-anchor="middle">2,099 Clean WAVs</text>
    <text x="265" y="525" font-size="9" fill="#94A3B8" text-anchor="middle">Standardized 16k PCM</text>
  </g>

  <line x1="370" y1="230" x2="420" y2="230" stroke="#0EA5E9" stroke-width="2" marker-end="url(#arrowCyan)"/>
  <line x1="370" y1="520" x2="420" y2="400" stroke="#F43F5E" stroke-width="2" stroke-dasharray="4,4" marker-end="url(#arrowRose)"/>

  <!-- Right: Generated Manifest Manifests -->
  <g transform="translate(830, 105)">
    <rect width="320" height="555" rx="10" fill="#1E293B" stroke="#38BDF8" stroke-width="1.5"/>
    <path d="M 0 10 Q 0 0 10 0 L 310 0 Q 320 0 320 10 L 320 35 L 0 35 Z" fill="#0284C7" opacity="0.3"/>
    <text x="160" y="24" font-size="13" font-weight="700" fill="#38BDF8" text-anchor="middle">ACTUAL MANIFEST ARTIFACTS</text>

    <rect x="15" y="50" width="290" height="85" rx="6" fill="#0F172A" stroke="#334155"/>
    <text x="30" y="72" font-size="12" font-weight="600" fill="#E2E8F0">source_registry_v2.csv</text>
    <text x="30" y="90" font-size="11" fill="#38BDF8">2,240 Total Registered Records</text>
    <text x="30" y="105" font-size="10" fill="#94A3B8">2,099 Approved &bull; 141 Quarantined / Rejected</text>
    <text x="30" y="118" font-size="9" fill="#64748B">1,000 Commands &bull; 699 Unknown &bull; 400 Silence</text>

    <rect x="15" y="145" width="290" height="75" rx="6" fill="#0F172A" stroke="#334155"/>
    <text x="30" y="167" font-size="12" font-weight="600" fill="#E2E8F0">train_manifest_v2.csv</text>
    <text x="30" y="185" font-size="11" fill="#38BDF8">1,281 Approved Train Utterances</text>
    <text x="30" y="200" font-size="10" fill="#94A3B8">TRAI/PHAI/LEN/XUONG: 178 each (712 total)</text>
    <text x="30" y="213" font-size="9" fill="#64748B">UNKNOWN: 319 &bull; SILENCE: 250</text>

    <rect x="15" y="230" width="290" height="75" rx="6" fill="#0F172A" stroke="#334155"/>
    <text x="30" y="252" font-size="12" font-weight="600" fill="#E2E8F0">val_manifest_v2.csv</text>
    <text x="30" y="270" font-size="11" fill="#38BDF8">284 Approved Val Utterances</text>
    <text x="30" y="285" font-size="10" fill="#94A3B8">TRAI/PHAI/LEN/XUONG: 36 each (144 total)</text>
    <text x="30" y="298" font-size="9" fill="#64748B">UNKNOWN: 80 &bull; SILENCE: 60</text>

    <rect x="15" y="315" width="290" height="110" rx="6" fill="#0F172A" stroke="#10B981" stroke-width="1.5"/>
    <text x="30" y="337" font-size="12" font-weight="700" fill="#34D399">test_manifest_v2.csv (FROZEN)</text>
    <text x="30" y="355" font-size="11" fill="#FFFFFF">534 Evaluated Utterances</text>
    <text x="30" y="370" font-size="10" fill="#94A3B8">TRAI/PHAI/LEN/XUONG: 36 each (144 total)</text>
    <text x="30" y="383" font-size="10" fill="#94A3B8">UNKNOWN: 300 &bull; SILENCE: 90</text>
    <text x="30" y="398" font-size="9" fill="#34D399" class="code">SHA256: 9E3904C8EA427244F7280A41...</text>
    <text x="30" y="413" font-size="9" fill="#6EE7B7">Cryptographically Locked Invariant</text>

    <rect x="15" y="435" width="290" height="105" rx="6" fill="#0B132B" stroke="#0284C7"/>
    <text x="30" y="457" font-size="11" font-weight="700" fill="#38BDF8">AUDITED INTEGRITY STATUS</text>
    <text x="30" y="475" font-size="10" fill="#E2E8F0">&bull; 0 missing audio files on disk (2,099 found)</text>
    <text x="30" y="490" font-size="10" fill="#E2E8F0">&bull; Exact label mapping: [0..5] alignment</text>
    <text x="30" y="505" font-size="10" fill="#E2E8F0">&bull; Disjoint speaker partitions confirmed (0 leak)</text>
    <text x="30" y="520" font-size="10" fill="#F43F5E">&bull; Core Blocker: 100% synthetic commands</text>
  </g>

  <line x1="780" y1="380" x2="830" y2="380" stroke="#0EA5E9" stroke-width="2" marker-end="url(#arrowCyan)"/>
</svg>"""
    save_svg("figure_02_dataset_acquisition_provenance_flow.svg", svg)

# -----------------------------------------------------------------------------
# Figure 3: Class Distribution by Split (Verified Manifest Data)
# -----------------------------------------------------------------------------
def generate_figure_03():
    # Classes: TRAI, PHAI, LEN, XUONG, UNKNOWN, SILENCE
    # Train: 178, 178, 178, 178, 319, 250 (Total 1,281)
    # Val:    36,  36,  36,  36,  80,  60 (Total 284)
    # Test:   36,  36,  36,  36, 300,  90 (Total 534)
    # Scale: Max value 350. Chart height 380 px. Scale = 380/350 = 1.085 px/unit
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 600" width="1100" height="600">
  <defs>
    <style>{COMMON_STYLE}</style>
    <linearGradient id="bgGrad3" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0F172A"/>
      <stop offset="100%" stop-color="#1E293B"/>
    </linearGradient>
  </defs>

  <rect width="1100" height="600" fill="url(#bgGrad3)" rx="16"/>

  <text x="50" y="45" font-size="22" font-weight="700" fill="#F8FAFC">Figure 3: Audited Dataset Class Distribution Partitioned by Split (V2)</text>
  <text x="50" y="70" font-size="13" fill="#94A3B8">Verified Manifest Allocations across Train (1,281), Validation (284), and Held-Out Test (534) Partitions</text>

  <!-- Legend -->
  <g transform="translate(750, 40)">
    <rect x="0" y="0" width="16" height="16" rx="3" fill="#3B82F6"/>
    <text x="24" y="13" font-size="12" fill="#E2E8F0">Train (1,281)</text>

    <rect x="120" y="0" width="16" height="16" rx="3" fill="#10B981"/>
    <text x="144" y="13" font-size="12" fill="#E2E8F0">Val (284)</text>

    <rect x="220" y="0" width="16" height="16" rx="3" fill="#F59E0B"/>
    <text x="244" y="13" font-size="12" fill="#E2E8F0">Test (534)</text>
  </g>

  <!-- Gridlines for max=350. Baseline Y=500. Scale=1.085 -->
  <!-- 300: Y = 500 - 325.5 = 174.5 -->
  <line x1="80" y1="174" x2="1040" y2="174" stroke="#334155" stroke-dasharray="4,4"/>
  <text x="65" y="179" font-size="11" fill="#64748B" text-anchor="end">300</text>
  <!-- 200: Y = 500 - 217.0 = 283.0 -->
  <line x1="80" y1="283" x2="1040" y2="283" stroke="#334155" stroke-dasharray="4,4"/>
  <text x="65" y="288" font-size="11" fill="#64748B" text-anchor="end">200</text>
  <!-- 100: Y = 500 - 108.5 = 391.5 -->
  <line x1="80" y1="391.5" x2="1040" y2="391.5" stroke="#334155" stroke-dasharray="4,4"/>
  <text x="65" y="396" font-size="11" fill="#64748B" text-anchor="end">100</text>
  <!-- 0 baseline -->
  <line x1="80" y1="500" x2="1040" y2="500" stroke="#64748B" stroke-width="1.5"/>
  <text x="65" y="504" font-size="11" fill="#64748B" text-anchor="end">0</text>

  <!-- TRAI: Train=178 (h=193.1, y=306.9), Val=36 (h=39.1, y=460.9), Test=36 (h=39.1, y=460.9) -->
  <g transform="translate(130, 0)">
    <rect x="0" y="306.9" width="24" height="193.1" rx="4" fill="#3B82F6"/>
    <text x="12" y="299" font-size="11" font-weight="600" fill="#93C5FD" text-anchor="middle">178</text>
    <rect x="28" y="460.9" width="24" height="39.1" rx="4" fill="#10B981"/>
    <text x="40" y="453" font-size="11" font-weight="600" fill="#6EE7B7" text-anchor="middle">36</text>
    <rect x="56" y="460.9" width="24" height="39.1" rx="4" fill="#F59E0B"/>
    <text x="68" y="453" font-size="11" font-weight="600" fill="#FCD34D" text-anchor="middle">36</text>
    <text x="40" y="525" font-size="13" font-weight="700" fill="#F1F5F9" text-anchor="middle">TRAI</text>
    <text x="40" y="542" font-size="10" fill="#94A3B8" text-anchor="middle">Total: 250</text>
  </g>

  <!-- PHAI: Train=178, Val=36, Test=36 -->
  <g transform="translate(280, 0)">
    <rect x="0" y="306.9" width="24" height="193.1" rx="4" fill="#3B82F6"/>
    <text x="12" y="299" font-size="11" font-weight="600" fill="#93C5FD" text-anchor="middle">178</text>
    <rect x="28" y="460.9" width="24" height="39.1" rx="4" fill="#10B981"/>
    <text x="40" y="453" font-size="11" font-weight="600" fill="#6EE7B7" text-anchor="middle">36</text>
    <rect x="56" y="460.9" width="24" height="39.1" rx="4" fill="#F59E0B"/>
    <text x="68" y="453" font-size="11" font-weight="600" fill="#FCD34D" text-anchor="middle">36</text>
    <text x="40" y="525" font-size="13" font-weight="700" fill="#F1F5F9" text-anchor="middle">PHAI</text>
    <text x="40" y="542" font-size="10" fill="#94A3B8" text-anchor="middle">Total: 250</text>
  </g>

  <!-- LEN: Train=178, Val=36, Test=36 -->
  <g transform="translate(430, 0)">
    <rect x="0" y="306.9" width="24" height="193.1" rx="4" fill="#3B82F6"/>
    <text x="12" y="299" font-size="11" font-weight="600" fill="#93C5FD" text-anchor="middle">178</text>
    <rect x="28" y="460.9" width="24" height="39.1" rx="4" fill="#10B981"/>
    <text x="40" y="453" font-size="11" font-weight="600" fill="#6EE7B7" text-anchor="middle">36</text>
    <rect x="56" y="460.9" width="24" height="39.1" rx="4" fill="#F59E0B"/>
    <text x="68" y="453" font-size="11" font-weight="600" fill="#FCD34D" text-anchor="middle">36</text>
    <text x="40" y="525" font-size="13" font-weight="700" fill="#F1F5F9" text-anchor="middle">LEN</text>
    <text x="40" y="542" font-size="10" fill="#94A3B8" text-anchor="middle">Total: 250</text>
  </g>

  <!-- XUONG: Train=178, Val=36, Test=36 -->
  <g transform="translate(580, 0)">
    <rect x="0" y="306.9" width="24" height="193.1" rx="4" fill="#3B82F6"/>
    <text x="12" y="299" font-size="11" font-weight="600" fill="#93C5FD" text-anchor="middle">178</text>
    <rect x="28" y="460.9" width="24" height="39.1" rx="4" fill="#10B981"/>
    <text x="40" y="453" font-size="11" font-weight="600" fill="#6EE7B7" text-anchor="middle">36</text>
    <rect x="56" y="460.9" width="24" height="39.1" rx="4" fill="#F59E0B"/>
    <text x="68" y="453" font-size="11" font-weight="600" fill="#FCD34D" text-anchor="middle">36</text>
    <text x="40" y="525" font-size="13" font-weight="700" fill="#F1F5F9" text-anchor="middle">XUONG</text>
    <text x="40" y="542" font-size="10" fill="#94A3B8" text-anchor="middle">Total: 250</text>
  </g>

  <!-- UNKNOWN: Train=319 (h=346.1, y=153.9), Val=80 (h=86.8, y=413.2), Test=300 (h=325.5, y=174.5) -->
  <g transform="translate(730, 0)">
    <rect x="0" y="153.9" width="24" height="346.1" rx="4" fill="#3B82F6"/>
    <text x="12" y="146" font-size="11" font-weight="600" fill="#93C5FD" text-anchor="middle">319</text>
    <rect x="28" y="413.2" width="24" height="86.8" rx="4" fill="#10B981"/>
    <text x="40" y="405" font-size="11" font-weight="600" fill="#6EE7B7" text-anchor="middle">80</text>
    <rect x="56" y="174.5" width="24" height="325.5" rx="4" fill="#F59E0B"/>
    <text x="68" y="167" font-size="11" font-weight="600" fill="#FCD34D" text-anchor="middle">300</text>
    <text x="40" y="525" font-size="13" font-weight="700" fill="#F1F5F9" text-anchor="middle">UNKNOWN</text>
    <text x="40" y="542" font-size="10" fill="#94A3B8" text-anchor="middle">Total: 699</text>
  </g>

  <!-- SILENCE: Train=250 (h=271.3, y=228.7), Val=60 (h=65.1, y=434.9), Test=90 (h=97.7, y=402.3) -->
  <g transform="translate(880, 0)">
    <rect x="0" y="228.7" width="24" height="271.3" rx="4" fill="#3B82F6"/>
    <text x="12" y="221" font-size="11" font-weight="600" fill="#93C5FD" text-anchor="middle">250</text>
    <rect x="28" y="434.9" width="24" height="65.1" rx="4" fill="#10B981"/>
    <text x="40" y="427" font-size="11" font-weight="600" fill="#6EE7B7" text-anchor="middle">60</text>
    <rect x="56" y="402.3" width="24" height="97.7" rx="4" fill="#F59E0B"/>
    <text x="68" y="395" font-size="11" font-weight="600" fill="#FCD34D" text-anchor="middle">90</text>
    <text x="40" y="525" font-size="13" font-weight="700" fill="#F1F5F9" text-anchor="middle">SILENCE</text>
    <text x="40" y="542" font-size="10" fill="#94A3B8" text-anchor="middle">Total: 400</text>
  </g>

  <text x="550" y="580" font-size="12" fill="#94A3B8" text-anchor="middle">Audited Total Pool: Exactly 2,099 Files (1,000 Commands &bull; 699 Unknown &bull; 400 Silence) across Train, Val, and Test Manifests</text>
</svg>"""
    save_svg("figure_03_class_distribution_by_split.svg", svg)

# -----------------------------------------------------------------------------
# Figure 4: Model Accuracy, F1, and TRAI Recall Comparison (Verified Metrics)
# -----------------------------------------------------------------------------
def generate_figure_04():
    # Comparison: Baseline VietnameseKWSNet (best_kws_model.pt) vs. TC-ResNet-14 (kws_tcresnet_advanced.pt)
    # Verified metrics on test_manifest_v2.csv (N=534):
    # 1. Overall Accuracy: 97.75% vs. 96.63%
    # 2. 6-Class Macro-F1: 0.9466 (94.66%) vs. 0.9402 (94.02%)
    # 3. TRAI Recall: 69.44% (25/36) vs. 75.00% (27/36)
    # 4. Northern TRAI Recall: 16.67% (2/12) vs. 25.00% (3/12)
    # 5. Model Parameters: 138,534 vs. 272,934
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1150 620" width="1150" height="620">
  <defs>
    <style>{COMMON_STYLE}</style>
    <linearGradient id="bgGrad4" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0F172A"/>
      <stop offset="100%" stop-color="#1E293B"/>
    </linearGradient>
  </defs>

  <rect width="1150" height="620" fill="url(#bgGrad4)" rx="16"/>

  <text x="50" y="45" font-size="22" font-weight="700" fill="#F8FAFC">Figure 4: Comparative Performance: Baseline 2D-CNN vs. TC-ResNet-14</text>
  <text x="50" y="70" font-size="13" fill="#94A3B8">Empirically Verified Checkpoint Metrics on Held-Out Test Set (test_manifest_v2.csv, N = 534)</text>

  <!-- Legend -->
  <g transform="translate(730, 40)">
    <rect x="0" y="0" width="18" height="18" rx="4" fill="#64748B"/>
    <text x="26" y="14" font-size="12" font-weight="600" fill="#E2E8F0">2D-CNN Baseline (138.5k params)</text>

    <rect x="230" y="0" width="18" height="18" rx="4" fill="#0EA5E9"/>
    <text x="256" y="14" font-size="12" font-weight="600" fill="#E2E8F0">TC-ResNet-14 (272.9k params)</text>
  </g>

  <!-- Y axis: 0% to 100%, height 380px (Y: 120 to 500), 3.8 px / % -->
  <line x1="80" y1="120" x2="1080" y2="120" stroke="#334155" stroke-dasharray="4,4"/>
  <text x="65" y="125" font-size="11" fill="#64748B" text-anchor="end">100%</text>

  <line x1="80" y1="196" x2="1080" y2="196" stroke="#334155" stroke-dasharray="4,4"/>
  <text x="65" y="201" font-size="11" fill="#64748B" text-anchor="end">80%</text>

  <line x1="80" y1="272" x2="1080" y2="272" stroke="#334155" stroke-dasharray="4,4"/>
  <text x="65" y="277" font-size="11" fill="#64748B" text-anchor="end">60%</text>

  <line x1="80" y1="348" x2="1080" y2="348" stroke="#334155" stroke-dasharray="4,4"/>
  <text x="65" y="353" font-size="11" fill="#64748B" text-anchor="end">40%</text>

  <line x1="80" y1="424" x2="1080" y2="424" stroke="#334155" stroke-dasharray="4,4"/>
  <text x="65" y="429" font-size="11" fill="#64748B" text-anchor="end">20%</text>

  <line x1="80" y1="500" x2="1080" y2="500" stroke="#64748B" stroke-width="1.5"/>
  <text x="65" y="504" font-size="11" fill="#64748B" text-anchor="end">0%</text>

  <!-- 1. Overall Accuracy: 97.75% vs 96.63% -->
  <g transform="translate(140, 0)">
    <rect x="0" y="128.5" width="42" height="371.5" rx="5" fill="#64748B"/>
    <text x="21" y="120" font-size="11" font-weight="600" fill="#CBD5E1" text-anchor="middle">97.8%</text>
    <rect x="52" y="132.8" width="42" height="367.2" rx="5" fill="#0EA5E9"/>
    <text x="73" y="125" font-size="11" font-weight="700" fill="#38BDF8" text-anchor="middle">96.6%</text>
    <text x="47" y="525" font-size="13" font-weight="700" fill="#F1F5F9" text-anchor="middle">Test Accuracy</text>
    <text x="47" y="542" font-size="10" fill="#94A3B8" text-anchor="middle">522 vs 516 / 534</text>
  </g>

  <!-- 2. Macro-F1: 94.66% vs 94.02% -->
  <g transform="translate(350, 0)">
    <rect x="0" y="140.3" width="42" height="359.7" rx="5" fill="#64748B"/>
    <text x="21" y="132" font-size="11" font-weight="600" fill="#CBD5E1" text-anchor="middle">94.7%</text>
    <rect x="52" y="142.7" width="42" height="357.3" rx="5" fill="#0EA5E9"/>
    <text x="73" y="135" font-size="11" font-weight="700" fill="#38BDF8" text-anchor="middle">94.0%</text>
    <text x="47" y="525" font-size="13" font-weight="700" fill="#F1F5F9" text-anchor="middle">6-Class Macro-F1</text>
    <text x="47" y="542" font-size="10" fill="#94A3B8" text-anchor="middle">Full 6 classes evaluated</text>
  </g>

  <!-- 3. Overall TRAI Recall: 69.44% vs 75.00% -->
  <g transform="translate(560, 0)">
    <rect x="0" y="236.1" width="42" height="263.9" rx="5" fill="#64748B"/>
    <text x="21" y="228" font-size="11" font-weight="600" fill="#CBD5E1" text-anchor="middle">69.4%</text>
    <rect x="52" y="215.0" width="42" height="285.0" rx="5" fill="#0EA5E9"/>
    <text x="73" y="207" font-size="11" font-weight="700" fill="#38BDF8" text-anchor="middle">75.0%</text>
    <text x="47" y="525" font-size="13" font-weight="700" fill="#F1F5F9" text-anchor="middle">TRAI Recall (Overall)</text>
    <text x="47" y="542" font-size="10" fill="#34D399" text-anchor="middle">+5.56% gain (27 vs 25)</text>
  </g>

  <!-- 4. Northern Dialect TRAI: 16.67% vs 25.00% -->
  <g transform="translate(770, 0)">
    <rect x="0" y="436.7" width="42" height="63.3" rx="5" fill="#EF4444"/>
    <text x="21" y="428" font-size="11" font-weight="600" fill="#FCA5A5" text-anchor="middle">16.7%</text>
    <rect x="52" y="405.0" width="42" height="95.0" rx="5" fill="#F59E0B"/>
    <text x="73" y="397" font-size="11" font-weight="700" fill="#FCD34D" text-anchor="middle">25.0%</text>
    <text x="47" y="525" font-size="13" font-weight="700" fill="#F87171" text-anchor="middle">Northern TRAI Recall</text>
    <text x="47" y="542" font-size="10" fill="#FCD34D" text-anchor="middle">3/12 vs 2/12 (severely depressed)</text>
  </g>

  <!-- 5. PHAI Recall: 100.0% vs 100.0% -->
  <g transform="translate(980, 0)">
    <rect x="0" y="120.0" width="42" height="380.0" rx="5" fill="#64748B"/>
    <text x="21" y="112" font-size="11" font-weight="600" fill="#CBD5E1" text-anchor="middle">100%</text>
    <rect x="52" y="120.0" width="42" height="380.0" rx="5" fill="#0EA5E9"/>
    <text x="73" y="112" font-size="11" font-weight="700" fill="#38BDF8" text-anchor="middle">100%</text>
    <text x="47" y="525" font-size="13" font-weight="700" fill="#F1F5F9" text-anchor="middle">PHAI Recall</text>
    <text x="47" y="542" font-size="10" fill="#34D399" text-anchor="middle">36/36 correct</text>
  </g>

  <rect x="100" y="570" width="950" height="30" rx="6" fill="#1E293B" stroke="#334155"/>
  <text x="575" y="590" font-size="11" fill="#94A3B8" text-anchor="middle">Acoustic Analysis: TC-ResNet improves TRAI temporal onset modeling, but Northern TRAI recall remains capped at 25.0% due to synthetic data limitations.</text>
</svg>"""
    save_svg("figure_04_model_accuracy_f1_trai_comparison.svg", svg)

# -----------------------------------------------------------------------------
# Figure 5: Confusion Matrix for TC-ResNet-14 (Verified Predictions)
# -----------------------------------------------------------------------------
def generate_figure_05():
    # Matrix data: 6x6 from kws_tcresnet_advanced.pt on test_manifest_v2.csv (N=534)
    # Rows: True Class, Cols: Predicted Class
    # TRAI (36):    [27,  9,  0,  0,   0,  0]
    # PHAI (36):    [ 0, 36,  0,  0,   0,  0]
    # LEN (36):     [ 0,  0, 36,  0,   0,  0]
    # XUONG (36):   [ 0,  0,  2, 34,   0,  0]
    # UNKNOWN (300):[ 0,  0,  0,  0, 293,  7]
    # SILENCE (90): [ 0,  0,  0,  0,   0, 90]
    matrix = [
        [27, 9, 0, 0, 0, 0],
        [0, 36, 0, 0, 0, 0],
        [0, 0, 36, 0, 0, 0],
        [0, 0, 2, 34, 0, 0],
        [0, 0, 0, 0, 293, 7],
        [0, 0, 0, 0, 0, 90],
    ]
    totals = [36, 36, 36, 36, 300, 90]

    cells_svg = []
    for r in range(6):
        for c in range(6):
            val = matrix[r][c]
            x = 320 + c * 85
            y = 140 + r * 65
            pct = val / totals[r]
            
            if r == c:
                if pct > 0.95:
                    bg = "#065F46"
                    border = "#10B981"
                    text_color = "#ECFDF5"
                elif pct >= 0.75:
                    bg = "#047857"
                    border = "#34D399"
                    text_color = "#FFFFFF"
                else:
                    bg = "#B45309"
                    border = "#F59E0B"
                    text_color = "#FEF3C7"
            else:
                if val == 0:
                    bg = "#0F172A"
                    border = "#1E293B"
                    text_color = "#475569"
                elif val <= 2:
                    bg = "#1E293B"
                    border = "#334155"
                    text_color = "#94A3B8"
                else:
                    bg = "#7F1D1D"
                    border = "#EF4444"
                    text_color = "#FEE2E2"

            cells_svg.append(f"""
    <g transform="translate({x}, {y})">
      <rect width="80" height="60" rx="6" fill="{bg}" stroke="{border}" stroke-width="1.5"/>
      <text x="40" y="32" font-size="16" font-weight="700" fill="{text_color}" text-anchor="middle">{val}</text>
      <text x="40" y="48" font-size="10" fill="{text_color}" opacity="0.8" text-anchor="middle">{pct*100:.1f}%</text>
    </g>""")

    cells_str = "\n".join(cells_svg)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 680" width="1000" height="680">
  <defs>
    <style>{COMMON_STYLE}</style>
    <linearGradient id="bgGrad5" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0F172A"/>
      <stop offset="100%" stop-color="#1E293B"/>
    </linearGradient>
  </defs>

  <rect width="1000" height="680" fill="url(#bgGrad5)" rx="16"/>

  <text x="50" y="45" font-size="22" font-weight="700" fill="#F8FAFC">Figure 5: TC-ResNet-14 Confusion Matrix on Frozen Test Set (N = 534)</text>
  <text x="50" y="70" font-size="13" fill="#94A3B8">Empirical Multi-Class Results on test_manifest_v2.csv &bull; 9 TRAI misclassified as PHAI (25.0% error rate)</text>

  <!-- Predicted Class Header -->
  <text x="575" y="105" font-size="14" font-weight="700" fill="#38BDF8" text-anchor="middle">PREDICTED CLASS</text>
  
  <g transform="translate(320, 115)">
    <text x="40" y="15" font-size="12" font-weight="600" fill="#CBD5E1" text-anchor="middle">TRAI</text>
    <text x="125" y="15" font-size="12" font-weight="600" fill="#CBD5E1" text-anchor="middle">PHAI</text>
    <text x="210" y="15" font-size="12" font-weight="600" fill="#CBD5E1" text-anchor="middle">LEN</text>
    <text x="295" y="15" font-size="12" font-weight="600" fill="#CBD5E1" text-anchor="middle">XUONG</text>
    <text x="380" y="15" font-size="12" font-weight="600" fill="#CBD5E1" text-anchor="middle">UNKNOWN</text>
    <text x="465" y="15" font-size="12" font-weight="600" fill="#CBD5E1" text-anchor="middle">SILENCE</text>
  </g>

  <!-- True Class Header -->
  <text x="130" y="335" font-size="14" font-weight="700" fill="#38BDF8" text-anchor="middle" transform="rotate(-90 130 335)">TRUE CLASS</text>

  <!-- Row Labels -->
  <g transform="translate(180, 140)">
    <text x="110" y="35" font-size="12" font-weight="600" fill="#CBD5E1" text-anchor="end">TRAI (N=36)</text>
    <text x="110" y="100" font-size="12" font-weight="600" fill="#CBD5E1" text-anchor="end">PHAI (N=36)</text>
    <text x="110" y="165" font-size="12" font-weight="600" fill="#CBD5E1" text-anchor="end">LEN (N=36)</text>
    <text x="110" y="230" font-size="12" font-weight="600" fill="#CBD5E1" text-anchor="end">XUONG (N=36)</text>
    <text x="110" y="295" font-size="12" font-weight="600" fill="#CBD5E1" text-anchor="end">UNKNOWN (N=300)</text>
    <text x="110" y="360" font-size="12" font-weight="600" fill="#CBD5E1" text-anchor="end">SILENCE (N=90)</text>
  </g>

  <!-- Grid Cells -->
  {cells_str}

  <!-- Callout box for TRAI -> PHAI error -->
  <g transform="translate(845, 140)">
    <rect width="130" height="120" rx="8" fill="#1E293B" stroke="#EF4444" stroke-width="1.5"/>
    <text x="65" y="25" font-size="11" font-weight="700" fill="#F87171" text-anchor="middle">CRITICAL ERROR</text>
    <text x="65" y="45" font-size="10" fill="#E2E8F0" text-anchor="middle">TRAI misclassified</text>
    <text x="65" y="60" font-size="10" fill="#E2E8F0" text-anchor="middle">as PHAI (9 times)</text>
    <text x="65" y="80" font-size="9" fill="#94A3B8" text-anchor="middle">Shared /aːj/ rhyme</text>
    <text x="65" y="95" font-size="9" fill="#94A3B8" text-anchor="middle">Weak onset burst</text>
    <text x="65" y="110" font-size="9" fill="#FCA5A5" text-anchor="middle">25.0% error rate</text>
  </g>

  <!-- Summary Statistics Footer -->
  <rect x="180" y="555" width="790" height="90" rx="8" fill="#1E293B" stroke="#334155"/>
  <text x="200" y="580" font-size="12" font-weight="700" fill="#38BDF8">Audited Metrics on 534 Verified Held-Out Test Samples:</text>
  <text x="200" y="600" font-size="11" fill="#CBD5E1">&bull; Overall Accuracy: 96.63% (516 / 534 correct) &bull; Macro-F1 (6 Classes): 0.9402</text>
  <text x="200" y="618" font-size="11" fill="#CBD5E1">&bull; Command-Only Recalls: TRAI = 75.0% &bull; PHAI = 100.0% &bull; LEN = 100.0% &bull; XUONG = 94.4%</text>
  <text x="200" y="635" font-size="11" fill="#94A3B8">&bull; Rejection Recalls: UNKNOWN = 97.67% (293/300) &bull; SILENCE = 100.0% (90/90)</text>
</svg>"""
    save_svg("figure_05_confusion_matrix_tcresnet.svg", svg)

# -----------------------------------------------------------------------------
# Figure 6: Streaming Detection Performance & Trade-offs (Verified 8-min Stream)
# -----------------------------------------------------------------------------
def generate_figure_06():
    # Real data from Table 09 / reports/phase3_streaming_evaluation.md
    # Stream duration: 8.0 minutes (0.133 hr); 144 target commands
    # tau=0.60: TP=118/144 (MCR=18.1%, Recall=81.9%), Latency=679.5 ms, 7.50 FA/hr
    # tau=0.70: TP=117/144 (MCR=18.8%, Recall=81.2%), Latency=706.4 ms, 0.00 FA/hr
    # tau=0.80: TP=109/144 (MCR=24.3%, Recall=75.7%), Latency=745.5 ms, 0.00 FA/hr
    # tau=0.85: TP=93/144  (MCR=35.4%, Recall=64.6%), Latency=786.6 ms, 0.00 FA/hr
    # tau=0.90: TP=66/144  (MCR=54.2%, Recall=45.8%), Latency=807.2 ms, 0.00 FA/hr
    # tau=0.95: TP=51/144  (MCR=64.6%, Recall=35.4%), Latency=849.7 ms, 0.00 FA/hr
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1150 620" width="1150" height="620">
  <defs>
    <style>{COMMON_STYLE}</style>
    <linearGradient id="bgGrad6" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0F172A"/>
      <stop offset="100%" stop-color="#1E293B"/>
    </linearGradient>
  </defs>

  <rect width="1150" height="620" fill="url(#bgGrad6)" rx="16"/>

  <text x="50" y="45" font-size="22" font-weight="700" fill="#F8FAFC">Figure 6: Continuous Streaming Detection Trade-off on 8.0-Minute Stream</text>
  <text x="50" y="70" font-size="13" fill="#94A3B8">Missed Command Rate (MCR) vs. Trigger Latency across Confidence Thresholds (&tau; &isin; [0.60, 0.95])</text>

  <!-- Plot Frame: (120, 110) to (670, 510). Width 550, Height 400 -->
  <!-- X: Missed Command Rate (%) from 10% to 70%. Width 500 px (X: 150 to 650). Scale = 500 / 60% = 8.33 px / % -->
  <!-- Y: Detection Latency (ms) from 650 ms to 900 ms. Height 330 px (Y: 150 to 480). Scale = 330 / 250 = 1.32 px / ms -->
  <g transform="translate(0, 0)">
    <rect x="120" y="110" width="550" height="400" rx="8" fill="#0B132B" stroke="#334155"/>

    <!-- Y Gridlines (Latency ms) -->
    <!-- 850 ms: y = 480 - (850-650)*1.32 = 480 - 264 = 216 -->
    <line x1="120" y1="216" x2="670" y2="216" stroke="#1E293B"/>
    <text x="110" y="220" font-size="11" fill="#64748B" text-anchor="end">850 ms</text>

    <!-- 800 ms: y = 480 - (800-650)*1.32 = 480 - 198 = 282 -->
    <line x1="120" y1="282" x2="670" y2="282" stroke="#1E293B"/>
    <text x="110" y="286" font-size="11" fill="#64748B" text-anchor="end">800 ms</text>

    <!-- 750 ms: y = 480 - (750-650)*1.32 = 480 - 132 = 348 -->
    <line x1="120" y1="348" x2="670" y2="348" stroke="#1E293B"/>
    <text x="110" y="352" font-size="11" fill="#64748B" text-anchor="end">750 ms</text>

    <!-- 700 ms: y = 480 - (700-650)*1.32 = 480 - 66 = 414 -->
    <line x1="120" y1="414" x2="670" y2="414" stroke="#1E293B"/>
    <text x="110" y="418" font-size="11" fill="#64748B" text-anchor="end">700 ms</text>

    <!-- 650 ms baseline: y = 480 -->
    <line x1="120" y1="480" x2="670" y2="480" stroke="#475569" stroke-width="1.5"/>
    <text x="110" y="484" font-size="11" fill="#64748B" text-anchor="end">650 ms</text>

    <!-- X Gridlines (MCR %) -->
    <!-- 20%: x = 150 + (20-10)*8.33 = 233.3 -->
    <line x1="233" y1="110" x2="233" y2="480" stroke="#1E293B" stroke-dasharray="3,3"/>
    <text x="233" y="500" font-size="11" fill="#64748B" text-anchor="middle">20%</text>

    <!-- 30%: x = 150 + (30-10)*8.33 = 316.6 -->
    <line x1="317" y1="110" x2="317" y2="480" stroke="#1E293B"/>
    <text x="317" y="500" font-size="11" fill="#64748B" text-anchor="middle">30%</text>

    <!-- 40%: x = 150 + (40-10)*8.33 = 400.0 -->
    <line x1="400" y1="110" x2="400" y2="480" stroke="#1E293B" stroke-dasharray="3,3"/>
    <text x="400" y="500" font-size="11" fill="#64748B" text-anchor="middle">40%</text>

    <!-- 50%: x = 150 + (50-10)*8.33 = 483.3 -->
    <line x1="483" y1="110" x2="483" y2="480" stroke="#1E293B"/>
    <text x="483" y="500" font-size="11" fill="#64748B" text-anchor="middle">50%</text>

    <!-- 60%: x = 150 + (60-10)*8.33 = 566.6 -->
    <line x1="567" y1="110" x2="567" y2="480" stroke="#1E293B" stroke-dasharray="3,3"/>
    <text x="567" y="500" font-size="11" fill="#64748B" text-anchor="middle">60%</text>

    <text x="395" y="525" font-size="12" font-weight="600" fill="#94A3B8" text-anchor="middle">Missed Command Rate (MCR %)</text>
    <text x="55" y="295" font-size="12" font-weight="600" fill="#94A3B8" text-anchor="middle" transform="rotate(-90 55 295)">Trigger Latency (ms)</text>

    <!-- Points:
         tau=0.60: MCR=18.1% (x=217), Lat=679.5 ms (y=441)
         tau=0.70: MCR=18.8% (x=223), Lat=706.4 ms (y=405)
         tau=0.80: MCR=24.3% (x=269), Lat=745.5 ms (y=354)
         tau=0.85: MCR=35.4% (x=362), Lat=786.6 ms (y=300)
         tau=0.90: MCR=54.2% (x=518), Lat=807.2 ms (y=272)
         tau=0.95: MCR=64.6% (x=605), Lat=849.7 ms (y=216)
    -->
    <path d="M 217 441 L 223 405 L 269 354 L 362 300 L 518 272 L 605 216" fill="none" stroke="#38BDF8" stroke-width="3"/>

    <!-- Point Markers -->
    <circle cx="217" cy="441" r="5" fill="#64748B"/>
    <text x="210" y="460" font-size="10" fill="#94A3B8">&tau;=0.60 (18.1%)</text>

    <!-- tau=0.70 (Recommended) -->
    <circle cx="223" cy="405" r="8" fill="#10B981" stroke="#FFFFFF" stroke-width="2"/>
    <g transform="translate(240, 385)">
      <rect width="180" height="50" rx="5" fill="#064E3B" stroke="#34D399" stroke-width="1.5"/>
      <text x="10" y="18" font-size="11" font-weight="700" fill="#A7F3D0">RECOMMENDED POINT</text>
      <text x="10" y="34" font-size="10" fill="#FFFFFF">&tau; = 0.70 &bull; 706 ms &bull; 18.8% MCR</text>
    </g>

    <circle cx="269" cy="354" r="5" fill="#F59E0B"/>
    <text x="285" y="358" font-size="10" fill="#FCD34D">&tau;=0.80 (24.3%)</text>

    <circle cx="362" cy="300" r="5" fill="#F59E0B"/>
    <text x="375" y="304" font-size="10" fill="#FCD34D">&tau;=0.85 (35.4%)</text>

    <circle cx="518" cy="272" r="5" fill="#EF4444"/>
    <text x="510" y="260" font-size="10" fill="#FCA5A5">&tau;=0.90 (54.2%)</text>

    <circle cx="605" cy="216" r="5" fill="#EF4444"/>
    <text x="605" y="205" font-size="10" fill="#FCA5A5" text-anchor="middle">&tau;=0.95 (64.6%)</text>
  </g>

  <!-- Right: Streaming Specifications & Audit Disclosure -->
  <g transform="translate(710, 110)">
    <rect width="390" height="400" rx="8" fill="#1E293B" stroke="#334155"/>
    <path d="M 0 8 Q 0 0 8 0 L 382 0 Q 390 0 390 8 L 390 35 L 0 35 Z" fill="#0284C7"/>
    <text x="195" y="23" font-size="13" font-weight="700" fill="#FFFFFF" text-anchor="middle">STREAMING AUDIT SPECIFICATIONS</text>

    <g transform="translate(20, 50)">
      <rect width="350" height="65" rx="6" fill="#0F172A" stroke="#334155"/>
      <text x="15" y="22" font-size="12" font-weight="600" fill="#38BDF8">Sliding Window Execution</text>
      <text x="15" y="39" font-size="11" fill="#E2E8F0">&bull; Window: 1.0 s &bull; Hop: 100 ms (10 Hz rate)</text>
      <text x="15" y="53" font-size="11" fill="#94A3B8">&bull; Single frame inference: 0.35 ms on CPU</text>
    </g>

    <g transform="translate(20, 125)">
      <rect width="350" height="65" rx="6" fill="#0F172A" stroke="#334155"/>
      <text x="15" y="22" font-size="12" font-weight="600" fill="#38BDF8">Temporal Gating &amp; Lockout</text>
      <text x="15" y="39" font-size="11" fill="#E2E8F0">&bull; Debounce lockout: 600 ms post-trigger</text>
      <text x="15" y="53" font-size="11" fill="#94A3B8">&bull; Confirmation: M = 2 consecutive frames</text>
    </g>

    <g transform="translate(20, 200)">
      <rect width="350" height="180" rx="6" fill="#0B132B" stroke="#F59E0B" stroke-width="1.5"/>
      <text x="15" y="22" font-size="12" font-weight="700" fill="#FCD34D">AUDITED DURATION &amp; POISSON BOUND</text>
      <text x="15" y="42" font-size="11" fill="#E2E8F0">&bull; Evaluated Stream: 8.0 Minutes (0.133 hr)</text>
      <text x="25" y="58" font-size="10" fill="#94A3B8">6.5 min FLEURS speech + 1.5 min noise</text>
      <text x="15" y="76" font-size="11" fill="#E2E8F0">&bull; Observed False Alarms: 0 FA in 8.0 min</text>
      <text x="15" y="94" font-size="11" fill="#CBD5E1">&bull; Poisson 95% Upper Bound: &le; 22.5 FA/hr</text>
      <text x="25" y="110" font-size="10" fill="#64748B">Rule of Three: 3 / 0.133 hr = 22.5 FA/hr</text>
      <text x="15" y="130" font-size="11" fill="#FDA4AF">&bull; Audit Finding: Claims of 140 1-hour streams</text>
      <text x="25" y="146" font-size="10" fill="#FDA4AF">were unverified extrapolations. True stream is 8 min.</text>
      <text x="15" y="166" font-size="10" fill="#A7F3D0">&bull; True Positives at &tau;=0.70: 117 / 144 commands</text>
    </g>
  </g>

  <text x="575" y="590" font-size="12" fill="#94A3B8" text-anchor="middle">Audited from evaluate_streaming.py and reports/phase3_streaming_evaluation.md &bull; Long-term FA rates require 10&ndash;50h audio</text>
</svg>"""
    save_svg("figure_06_streaming_roc_mcr_latency.svg", svg)

# -----------------------------------------------------------------------------
# Figure 7: Field Campaign Progress & Gap Analysis
# -----------------------------------------------------------------------------
def generate_figure_07():
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1150 640" width="1150" height="640">
  <defs>
    <style>{COMMON_STYLE}</style>
    <linearGradient id="bgGrad7" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0F172A"/>
      <stop offset="100%" stop-color="#1E293B"/>
    </linearGradient>
  </defs>

  <rect width="1150" height="640" fill="url(#bgGrad7)" rx="16"/>

  <text x="50" y="45" font-size="22" font-weight="700" fill="#F8FAFC">Figure 7: Human Field-Recording Campaign Audit &amp; Data Gap Analysis</text>
  <text x="50" y="70" font-size="13" fill="#94A3B8">Honest Verification: Planned Minimum Quotas vs. Audited Genuine Human Speech vs. Automated Test Fixtures</text>

  <g transform="translate(50, 110)">
    <rect width="550" height="490" rx="10" fill="#1E293B" stroke="#334155"/>
    <path d="M 0 10 Q 0 0 10 0 L 540 0 Q 550 0 550 10 L 550 40 L 0 40 Z" fill="#BE123C"/>
    <text x="275" y="26" font-size="14" font-weight="700" fill="#FFFFFF" text-anchor="middle">OVERALL CAMPAIGN COMPLETION STATUS</text>

    <g transform="translate(275, 140)">
      <circle cx="0" cy="0" r="75" fill="#0F172A" stroke="#334155" stroke-width="12"/>
      <circle cx="0" cy="0" r="75" fill="none" stroke="#F43F5E" stroke-width="12" stroke-dasharray="0 471"/>
      <text x="0" y="10" font-size="34" font-weight="800" fill="#FB7185" text-anchor="middle">0.0%</text>
      <text x="0" y="32" font-size="11" font-weight="600" fill="#94A3B8" text-anchor="middle">COMPLETE</text>
      <text x="0" y="105" font-size="14" font-weight="700" fill="#F43F5E" text-anchor="middle">0 / 1,200 Genuine Human Recordings</text>
    </g>

    <g transform="translate(30, 270)">
      <rect width="490" height="200" rx="8" fill="#0F172A" stroke="#334155"/>
      
      <text x="20" y="25" font-size="11" font-weight="700" fill="#94A3B8">METRIC</text>
      <text x="180" y="25" font-size="11" font-weight="700" fill="#94A3B8">TARGET GOAL</text>
      <text x="320" y="25" font-size="11" font-weight="700" fill="#94A3B8">GENUINE AUDITED</text>
      <text x="440" y="25" font-size="11" font-weight="700" fill="#94A3B8">GAP</text>
      <line x1="15" y1="35" x2="475" y2="35" stroke="#334155"/>

      <text x="20" y="58" font-size="12" font-weight="600" fill="#E2E8F0">TRAI Utterances</text>
      <text x="180" y="58" font-size="12" fill="#CBD5E1">300 samples</text>
      <text x="320" y="58" font-size="12" font-weight="700" fill="#EF4444">0 samples</text>
      <text x="440" y="58" font-size="12" font-weight="700" fill="#EF4444">-300</text>

      <text x="20" y="86" font-size="12" font-weight="600" fill="#E2E8F0">PHAI Utterances</text>
      <text x="180" y="86" font-size="12" fill="#CBD5E1">300 samples</text>
      <text x="320" y="86" font-size="12" font-weight="700" fill="#EF4444">0 samples</text>
      <text x="440" y="86" font-size="12" font-weight="700" fill="#EF4444">-300</text>

      <text x="20" y="114" font-size="12" font-weight="600" fill="#E2E8F0">LEN Utterances</text>
      <text x="180" y="114" font-size="12" fill="#CBD5E1">300 samples</text>
      <text x="320" y="114" font-size="12" font-weight="700" fill="#EF4444">0 samples</text>
      <text x="440" y="114" font-size="12" font-weight="700" fill="#EF4444">-300</text>

      <text x="20" y="142" font-size="12" font-weight="600" fill="#E2E8F0">XUONG Utterances</text>
      <text x="180" y="142" font-size="12" fill="#CBD5E1">300 samples</text>
      <text x="320" y="142" font-size="12" font-weight="700" fill="#EF4444">0 samples</text>
      <text x="440" y="142" font-size="12" font-weight="700" fill="#EF4444">-300</text>

      <text x="20" y="170" font-size="12" font-weight="600" fill="#E2E8F0">Distinct Human Speakers</text>
      <text x="180" y="170" font-size="12" fill="#CBD5E1">&ge; 20 speakers</text>
      <text x="320" y="170" font-size="12" font-weight="700" fill="#EF4444">0 speakers</text>
      <text x="440" y="170" font-size="12" font-weight="700" fill="#EF4444">-20</text>

      <text x="20" y="192" font-size="11" font-weight="600" fill="#E2E8F0">Verified Consent Forms</text>
      <text x="180" y="192" font-size="11" fill="#CBD5E1">&ge; 20 records</text>
      <text x="320" y="192" font-size="11" font-weight="700" fill="#EF4444">0 genuine</text>
      <text x="440" y="192" font-size="11" font-weight="700" fill="#EF4444">-20</text>
    </g>
  </g>

  <g transform="translate(630, 110)">
    <rect width="470" height="490" rx="10" fill="#1E293B" stroke="#334155"/>
    <path d="M 0 10 Q 0 0 10 0 L 460 0 Q 470 0 470 10 L 470 40 L 0 40 Z" fill="#334155"/>
    <text x="235" y="26" font-size="14" font-weight="700" fill="#FFFFFF" text-anchor="middle">DIALECT DEFICIT &amp; TEST FIXTURE AUDIT</text>

    <g transform="translate(25, 55)">
      <text x="0" y="18" font-size="13" font-weight="700" fill="#38BDF8">Dialect Target vs. Actual</text>
      
      <text x="0" y="45" font-size="12" fill="#CBD5E1">Northern (Target: 50% &bull; 600 clips)</text>
      <rect x="0" y="55" width="420" height="16" rx="4" fill="#0F172A"/>
      <text x="420" y="68" font-size="11" font-weight="700" fill="#EF4444" text-anchor="end">0.0% collected</text>

      <text x="0" y="95" font-size="12" fill="#CBD5E1">Central (Target: 25% &bull; 300 clips)</text>
      <rect x="0" y="105" width="420" height="16" rx="4" fill="#0F172A"/>
      <text x="420" y="118" font-size="11" font-weight="700" fill="#EF4444" text-anchor="end">0.0% collected</text>

      <text x="0" y="145" font-size="12" fill="#CBD5E1">Southern (Target: 25% &bull; 300 clips)</text>
      <rect x="0" y="155" width="420" height="16" rx="4" fill="#0F172A"/>
      <text x="420" y="168" font-size="11" font-weight="700" fill="#EF4444" text-anchor="end">0.0% collected</text>
    </g>

    <g transform="translate(25, 260)">
      <rect width="420" height="210" rx="8" fill="#0B132B" stroke="#0284C7" stroke-width="1.5"/>
      <text x="20" y="25" font-size="12" font-weight="700" fill="#38BDF8">EXPLICIT FIXTURE DISCLOSURE (NOT REAL DATA)</text>
      <text x="20" y="48" font-size="11" fill="#E2E8F0">Found in data/raw/field_recordings/:</text>
      <text x="20" y="68" font-size="11" fill="#CBD5E1">&bull; 74 Audio Takes (WAV and M4A)</text>
      <text x="20" y="86" font-size="10" fill="#94A3B8" class="code">&bull; SPK_FIELD_N001_TESTSUITE (unit test harness)</text>
      <text x="20" y="102" font-size="10" fill="#94A3B8" class="code">&bull; SPK_M4A_TEST (M4A decoding test fixture)</text>
      <text x="20" y="122" font-size="11" fill="#CBD5E1">&bull; Provenance Tag: test_harness / unit_test_mock</text>
      <text x="20" y="140" font-size="11" fill="#CBD5E1">&bull; Consent Ledger: 2 test fixture identities</text>
      
      <rect x="15" y="155" width="390" height="42" rx="6" fill="#1E293B" stroke="#F59E0B"/>
      <text x="25" y="172" font-size="10" font-weight="700" fill="#FCD34D">AUDIT INTEGRITY INVARIANT CONFIRMED:</text>
      <text x="25" y="188" font-size="10" fill="#E2E8F0">Test fixtures are strictly quarantined from training manifests.</text>
    </g>
  </g>
</svg>"""
    save_svg("figure_07_field_campaign_target_progress.svg", svg)

# -----------------------------------------------------------------------------
# Figure 8: Deployment & Parity Verification Pipeline
# -----------------------------------------------------------------------------
def generate_figure_08():
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1150 640" width="1150" height="640">
  <defs>
    <style>{COMMON_STYLE}</style>
    <linearGradient id="bgGrad8" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0F172A"/>
      <stop offset="100%" stop-color="#1E293B"/>
    </linearGradient>
    <marker id="arrowParity" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#38BDF8"/>
    </marker>
  </defs>

  <rect width="1150" height="640" fill="url(#bgGrad8)" rx="16"/>

  <text x="50" y="45" font-size="22" font-weight="700" fill="#F8FAFC">Figure 8: On-Device Deployment, Quantization &amp; Numerical Parity Flow</text>
  <text x="50" y="70" font-size="13" fill="#94A3B8">Verification Pipeline: PyTorch Golden Checkpoints &rarr; ONNX Opset 14 Export &rarr; INT8 Quantization Audit</text>

  <!-- Step 1: PyTorch Models -->
  <g transform="translate(50, 110)">
    <rect width="220" height="480" rx="10" fill="#1E293B" stroke="#334155" stroke-width="1.5"/>
    <path d="M 0 10 Q 0 0 10 0 L 210 0 Q 220 0 220 10 L 220 38 L 0 38 Z" fill="#EE4C2C"/>
    <text x="110" y="24" font-size="13" font-weight="700" fill="#FFFFFF" text-anchor="middle">1. PYTORCH CHECKPOINTS</text>

    <rect x="15" y="55" width="190" height="85" rx="6" fill="#0F172A" stroke="#334155"/>
    <text x="25" y="75" font-size="12" font-weight="600" fill="#F87171">Model Checkpoints</text>
    <text x="25" y="93" font-size="10" fill="#CBD5E1" class="code">kws_tcresnet_advanced.pt</text>
    <text x="25" y="108" font-size="10" fill="#CBD5E1" class="code">best_kws_model.pt</text>
    <text x="25" y="125" font-size="10" fill="#94A3B8">PyTorch 2.6.0+cu124</text>

    <rect x="15" y="155" width="190" height="110" rx="6" fill="#0F172A" stroke="#334155"/>
    <text x="25" y="175" font-size="12" font-weight="600" fill="#E2E8F0">Architecture Specs</text>
    <text x="25" y="195" font-size="11" fill="#94A3B8">&bull; TC-ResNet: 272,934 params</text>
    <text x="25" y="212" font-size="11" fill="#94A3B8">&bull; 2D-CNN: 138,534 params</text>
    <text x="25" y="229" font-size="11" fill="#94A3B8">&bull; Output: Logits [1, 6]</text>
    <text x="25" y="246" font-size="10" fill="#38BDF8">&bull; CPU Inference: 0.25 - 0.35 ms</text>

    <rect x="15" y="280" width="190" height="190" rx="6" fill="#0F172A" stroke="#334155"/>
    <text x="25" y="300" font-size="12" font-weight="600" fill="#E2E8F0">Verified Baselines</text>
    <text x="25" y="320" font-size="11" fill="#CBD5E1">&bull; TC-ResNet: 96.63% acc</text>
    <text x="25" y="338" font-size="11" fill="#CBD5E1">&bull; TC-ResNet Macro-F1: 0.9402</text>
    <text x="25" y="356" font-size="11" fill="#CBD5E1">&bull; Baseline: 97.75% acc</text>
    <text x="25" y="374" font-size="11" fill="#CBD5E1">&bull; Baseline Macro-F1: 0.9466</text>
    <text x="25" y="405" font-size="10" fill="#34D399">&bull; Golden Reference for</text>
    <text x="25" y="420" font-size="10" fill="#34D399">numerical parity check</text>
  </g>

  <line x1="270" y1="350" x2="310" y2="350" stroke="#38BDF8" stroke-width="2" marker-end="url(#arrowParity)"/>

  <!-- Step 2: ONNX FP32 Export -->
  <g transform="translate(315, 110)">
    <rect width="250" height="480" rx="10" fill="#1E293B" stroke="#334155" stroke-width="1.5"/>
    <path d="M 0 10 Q 0 0 10 0 L 240 0 Q 250 0 250 10 L 250 38 L 0 38 Z" fill="#0284C7"/>
    <text x="125" y="24" font-size="13" font-weight="700" fill="#FFFFFF" text-anchor="middle">2. ONNX FP32 EXPORT</text>

    <rect x="15" y="55" width="220" height="95" rx="6" fill="#0F172A" stroke="#0284C7"/>
    <text x="25" y="75" font-size="12" font-weight="600" fill="#38BDF8">Export Configuration</text>
    <text x="25" y="93" font-size="10" fill="#CBD5E1" class="code">torch.onnx.export()</text>
    <text x="25" y="108" font-size="10" fill="#94A3B8">Opset Version: 14</text>
    <text x="25" y="123" font-size="10" fill="#94A3B8">Dynamic Batch: True</text>
    <text x="25" y="138" font-size="10" fill="#94A3B8">Verified via onnx.checker</text>

    <rect x="15" y="165" width="220" height="95" rx="6" fill="#0F172A" stroke="#334155"/>
    <text x="25" y="185" font-size="12" font-weight="600" fill="#E2E8F0">TC-ResNet FP32 Model</text>
    <text x="25" y="203" font-size="10" fill="#38BDF8" class="code">tcresnet_fp32.onnx</text>
    <text x="25" y="220" font-size="11" fill="#CBD5E1">&bull; Size: 1,093.7 KB (1,068.1 KiB)</text>
    <text x="25" y="238" font-size="11" fill="#CBD5E1">&bull; Nodes: 29 (12 Conv, 1 Gemm)</text>
    <text x="25" y="253" font-size="10" fill="#34D399">&bull; Max Parity Error: 1.19e-06</text>

    <rect x="15" y="275" width="220" height="195" rx="6" fill="#0F172A" stroke="#334155"/>
    <text x="25" y="295" font-size="12" font-weight="600" fill="#E2E8F0">Baseline FP32 Model</text>
    <text x="25" y="313" font-size="10" fill="#94A3B8" class="code">vietnamese_kws_float32.onnx</text>
    <text x="25" y="330" font-size="11" fill="#CBD5E1">&bull; Size: 555.4 KB (542.4 KiB)</text>
    <text x="25" y="348" font-size="11" fill="#CBD5E1">&bull; Nodes: 16 (4 Conv, 2 Gemm)</text>
    <text x="25" y="366" font-size="11" fill="#CBD5E1">&bull; Max Parity Error: 9.54e-07</text>
    <text x="25" y="405" font-size="10" fill="#34D399">&bull; 100% Class Concordance</text>
    <text x="25" y="420" font-size="10" fill="#34D399">tested across 534 samples</text>
  </g>

  <line x1="565" y1="350" x2="605" y2="350" stroke="#38BDF8" stroke-width="2" marker-end="url(#arrowParity)"/>

  <!-- Step 3: Quantization Audit -->
  <g transform="translate(610, 110)">
    <rect width="250" height="480" rx="10" fill="#1E293B" stroke="#334155" stroke-width="1.5"/>
    <path d="M 0 10 Q 0 0 10 0 L 240 0 Q 250 0 250 10 L 250 38 L 0 38 Z" fill="#7C3AED"/>
    <text x="125" y="24" font-size="13" font-weight="700" fill="#FFFFFF" text-anchor="middle">3. QUANTIZATION AUDIT</text>

    <!-- Verified INT8 -->
    <rect x="15" y="55" width="220" height="150" rx="6" fill="#064E3B" stroke="#10B981"/>
    <text x="25" y="75" font-size="12" font-weight="700" fill="#A7F3D0">2D-CNN INT8 (VERIFIED)</text>
    <text x="25" y="93" font-size="10" fill="#FFFFFF" class="code">vietnamese_kws_int8.onnx</text>
    <text x="25" y="110" font-size="11" fill="#E2E8F0">&bull; Size: 148.8 KB (145.3 KiB)</text>
    <text x="25" y="127" font-size="11" fill="#E2E8F0">&bull; Compression: 3.73x reduction</text>
    <text x="25" y="143" font-size="10" fill="#A7F3D0">&bull; Verified Graph Operators:</text>
    <text x="35" y="158" font-size="9" fill="#CBD5E1" class="code">DynamicQuantizeLinear, ConvInt</text>
    <text x="25" y="177" font-size="10" fill="#6EE7B7">&bull; Accuracy: 97.56% (-0.19% drop)</text>
    <text x="25" y="193" font-size="10" fill="#6EE7B7">&bull; Opset 14 &bull; 12 UINT8 initializers</text>

    <!-- Missing INT8 (Audit finding) -->
    <rect x="15" y="220" width="220" height="150" rx="6" fill="#2A1215" stroke="#E11D48"/>
    <text x="25" y="240" font-size="12" font-weight="700" fill="#FDA4AF">TC-ResNet INT8 (MISSING!)</text>
    <text x="25" y="258" font-size="10" fill="#FECDD3" class="code">vietnamese_kws_tcresnet_int8.onnx</text>
    <text x="25" y="278" font-size="11" fill="#FECDD3">&bull; Cited in Phase 3 report notes</text>
    <text x="25" y="295" font-size="11" fill="#FDA4AF" font-weight="600">&bull; AUDIT FINDING: NOT ON DISK</text>
    <text x="25" y="313" font-size="10" fill="#CBD5E1">&bull; Target Size: ~280 KB</text>
    <text x="25" y="330" font-size="10" fill="#CBD5E1">&bull; Quantization script exists but</text>
    <text x="25" y="345" font-size="10" fill="#CBD5E1">artifact was not saved to disk</text>
    <text x="25" y="360" font-size="10" fill="#FECDD3">&bull; Severity: P2 Defect</text>

    <rect x="15" y="385" width="220" height="85" rx="6" fill="#0F172A" stroke="#334155"/>
    <text x="25" y="405" font-size="11" font-weight="600" fill="#E2E8F0">Quantization Method</text>
    <text x="25" y="422" font-size="10" fill="#94A3B8">&bull; ONNX Dynamic Quantization</text>
    <text x="25" y="437" font-size="10" fill="#94A3B8">&bull; Activation: uint8 dynamic</text>
    <text x="25" y="452" font-size="10" fill="#94A3B8">&bull; Weights: uint8 symmetric</text>
  </g>

  <line x1="860" y1="350" x2="900" y2="350" stroke="#38BDF8" stroke-width="2" marker-end="url(#arrowParity)"/>

  <!-- Step 4: Parity & Mobile Targets -->
  <g transform="translate(905, 110)">
    <rect width="200" height="480" rx="10" fill="#1E293B" stroke="#334155" stroke-width="1.5"/>
    <path d="M 0 10 Q 0 0 10 0 L 190 0 Q 200 0 200 10 L 200 38 L 0 38 Z" fill="#059669"/>
    <text x="100" y="24" font-size="13" font-weight="700" fill="#FFFFFF" text-anchor="middle">4. PARITY &amp; MOBILE</text>

    <rect x="15" y="55" width="170" height="110" rx="6" fill="#0F172A" stroke="#059669"/>
    <text x="25" y="75" font-size="11" font-weight="700" fill="#34D399">NUMERICAL PARITY</text>
    <text x="25" y="93" font-size="10" fill="#CBD5E1">&bull; TC-ResNet Error:</text>
    <text x="35" y="108" font-size="10" fill="#38BDF8" class="code">1.19e-06 (&lt; 1e-4)</text>
    <text x="25" y="125" font-size="10" fill="#CBD5E1">&bull; Class Concordance:</text>
    <text x="35" y="140" font-size="10" fill="#34D399" font-weight="600">100.0% (534/534)</text>
    <text x="25" y="155" font-size="9" fill="#94A3B8">0 prediction drift</text>

    <rect x="15" y="180" width="170" height="135" rx="6" fill="#0F172A" stroke="#334155"/>
    <text x="25" y="200" font-size="11" font-weight="700" fill="#38BDF8">CPU INFERENCE</text>
    <text x="25" y="220" font-size="10" fill="#CBD5E1">&bull; TC-ResNet: 0.35 ms</text>
    <text x="25" y="238" font-size="10" fill="#CBD5E1">&bull; Baseline FP32: 0.25 ms</text>
    <text x="25" y="256" font-size="10" fill="#CBD5E1">&bull; Baseline INT8: 0.91 ms</text>
    <text x="25" y="276" font-size="10" fill="#34D399">&bull; Streaming budget: 100 ms</text>
    <text x="25" y="294" font-size="9" fill="#94A3B8">&lt; 1% single CPU core</text>
    <text x="25" y="307" font-size="9" fill="#94A3B8">utilization</text>

    <rect x="15" y="330" width="170" height="140" rx="6" fill="#0F172A" stroke="#334155"/>
    <text x="25" y="350" font-size="11" font-weight="700" fill="#E2E8F0">MOBILE TARGETS</text>
    <text x="25" y="370" font-size="10" fill="#CBD5E1">&bull; Android:</text>
    <text x="35" y="385" font-size="10" fill="#94A3B8">ONNX Runtime Mobile</text>
    <text x="35" y="400" font-size="10" fill="#94A3B8">/ NNAPI Delegate</text>
    <text x="25" y="420" font-size="10" fill="#CBD5E1">&bull; iOS:</text>
    <text x="35" y="435" font-size="10" fill="#94A3B8">CoreML / ONNX Runtime</text>
    <text x="35" y="450" font-size="10" fill="#FDA4AF">(Blocked by Win11 host)</text>
  </g>
</svg>"""
    save_svg("figure_08_deployment_parity_pipeline.svg", svg)

if __name__ == "__main__":
    print("Generating all 8 verified vector SVG figures...")
    generate_figure_01()
    generate_figure_02()
    generate_figure_03()
    generate_figure_04()
    generate_figure_05()
    generate_figure_06()
    generate_figure_07()
    generate_figure_08()
    print("Successfully generated all 8 verified SVG figures in docs/objective_1/figures/")
