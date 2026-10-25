"""
Onset-Aware Synthetic Acoustic Generator for Vietnamese KWS V4.
Provides principled, phonetically motivated onset parameterization:
  1. Trái (/tɕa:j/ Northern, /ʈa:j/ Southern/Central):
     - Stop closure: 25ms of vocal suppression (silent gap)
     - Transient burst: 15ms high-frequency filtered noise (>3.5 kHz) with sharp attack (<3ms)
     - Followed by vowel formant resonance and rising tone sắc
  2. Phải (/fa:j/ across all dialects):
     - Continuous frication: 50ms diffuse bandpass noise (1.8 - 6.0 kHz) starting at t=0
     - Zero closure gap; lower acoustic intensity (smooth attack)
     - Followed by vowel formant resonance and dipping tone hỏi
  3. Lên (/le:n/), Xuống (/suəŋ/), UNKNOWN, and SILENCE retain identical parameters to V3 control.
"""

import sys
import os
import random
from pathlib import Path

if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd
import soundfile as sf
import scipy.signal as signal

from src.acquisition.collect_field_data import TARGET_COMMANDS
from src.acquisition.campaign_validator import compute_file_sha256

NON_TARGET_WORDS = [
    ("chào", "falling"),
    ("bạn", "heavy"),
    ("đi", "level"),
    ("xe", "level"),
    ("nhà", "falling"),
    ("số", "rising"),
    ("ngày", "falling"),
    ("học", "heavy")
]


def generate_vietnamese_onset_aware_utterance(
    word: str,
    sr: int = 16000,
    f0_base: float = 130.0,
    duration_sec: float = 0.5,
    dialect: str = "Northern"
) -> np.ndarray:
    """
    Parametric acoustic synthesis with class-specific onset modeling for TRAI vs. PHAI.
    """
    n_samples = int(sr * duration_sec)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)
    
    # 1. Pitch / F0 contour modeling Vietnamese tones
    if word in ["trái", "xuống"]:
        # Tone Sắc: starts mid-high, rises sharply
        f0 = f0_base * (1.0 + 0.35 * (t / duration_sec) ** 1.5)
    elif word == "phải":
        # Tone Hỏi: dips then rises
        f0 = f0_base * (1.0 - 0.20 * np.sin(np.pi * t / duration_sec) + 0.15 * (t / duration_sec) ** 2)
    elif word == "lên":
        # Tone Ngang: flat / level contour with slight micro-intonation
        f0 = f0_base * (1.0 + 0.03 * np.sin(2 * np.pi * 3 * t))
    else:
        # Default falling or level
        f0 = f0_base * (1.0 - 0.15 * (t / duration_sec))

    # Phase accumulation for variable F0
    phase = 2 * np.pi * np.cumsum(f0) / sr
    source = np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.25 * np.sin(3 * phase) + 0.12 * np.sin(4 * phase)

    # 2. Formant resonant filtering based on Vietnamese vowel phonology
    if word in ["trái", "phải"]:
        f1, f2, f3 = 800.0, 1350.0, 2600.0
        f1_track = f1 * (1.0 - 0.4 * (t / duration_sec))
        f2_track = f2 * (1.0 + 0.5 * (t / duration_sec))
        f3_track = np.full_like(t, f3)
    elif word == "lên":
        f1_track = np.linspace(500.0, 350.0, n_samples)
        f2_track = np.linspace(1900.0, 1600.0, n_samples)
        f3_track = np.full_like(t, 2700.0)
    elif word == "xuống":
        f1_track = np.linspace(400.0, 450.0, n_samples)
        f2_track = np.linspace(900.0, 1100.0, n_samples)
        f3_track = np.full_like(t, 2400.0)
    else:
        f1_track = np.linspace(600.0, 500.0, n_samples)
        f2_track = np.linspace(1500.0, 1400.0, n_samples)
        f3_track = np.full_like(t, 2500.0)

    # Filter vocal source via IIR resonators
    filtered = np.zeros_like(source)
    chunk_size = 320
    for i in range(0, n_samples, chunk_size):
        end = min(i + chunk_size, n_samples)
        mid = (i + end) // 2
        f1_c, f2_c, f3_c = f1_track[mid], f2_track[mid], f3_track[mid]
        
        b1, a1 = signal.iirpeak(f1_c, Q=5.0, fs=sr)
        b2, a2 = signal.iirpeak(f2_c, Q=7.0, fs=sr)
        b3, a3 = signal.iirpeak(f3_c, Q=9.0, fs=sr)
        
        chunk_src = source[i:end]
        chunk_out = signal.lfilter(b1, a1, chunk_src) * 1.0 + \
                    signal.lfilter(b2, a2, chunk_src) * 0.6 + \
                    signal.lfilter(b3, a3, chunk_src) * 0.3
        filtered[i:end] = chunk_out

    # 3. Class-specific onset acoustic modeling (TRAI vs. PHAI)
    output = np.copy(filtered)
    
    if word == "trái":
        # Voiceless stop/affricate (/tɕ/ Northern/Central, /ʈ/ Southern)
        # 1. Stop closure: 25ms of silence at onset
        closure_samples = int(0.025 * sr)
        output[:closure_samples] *= 0.05
        
        # 2. Transient burst: 15ms high-pass filtered noise (>3500 Hz)
        burst_samples = int(0.015 * sr)
        raw_noise = np.random.randn(burst_samples).astype(np.float32)
        b_hp, a_hp = signal.butter(4, 3500.0 / (sr / 2.0), btype='high')
        burst_filtered = signal.lfilter(b_hp, a_hp, raw_noise)
        # Sharp burst envelope: rises in 2ms, decays exponentially
        burst_env = np.exp(-np.linspace(0, 5, burst_samples))
        burst = burst_filtered * burst_env * 0.65
        
        burst_start = closure_samples
        burst_end = burst_start + burst_samples
        if burst_end <= n_samples:
            output[burst_start:burst_end] += burst
            
    elif word == "phải":
        # Voiceless labiodental fricative (/f/)
        # 1. Continuous diffuse frication: 50ms bandpass noise (1800 - 6000 Hz) starting at t=0
        fric_samples = int(0.050 * sr)
        raw_noise = np.random.randn(fric_samples).astype(np.float32)
        b_bp, a_bp = signal.butter(4, [1800.0 / (sr / 2.0), 6000.0 / (sr / 2.0)], btype='band')
        fric_filtered = signal.lfilter(b_bp, a_bp, raw_noise)
        # Smooth linear decay envelope without sharp transient
        fric_env = np.linspace(0.8, 0.1, fric_samples)
        frication = fric_filtered * fric_env * 0.35
        
        output[:fric_samples] += frication[:min(fric_samples, n_samples)]
        
    elif word == "xuống":
        # Sibilant fricative (/s/)
        fric_samples = int(0.060 * sr)
        raw_noise = np.random.randn(fric_samples).astype(np.float32)
        b_hp, a_hp = signal.butter(4, 4000.0 / (sr / 2.0), btype='high')
        fric_filtered = signal.lfilter(b_hp, a_hp, raw_noise) * 0.40
        output[:fric_samples] += fric_filtered[:min(fric_samples, n_samples)]

    # 4. Smooth amplitude envelope
    env = np.ones(n_samples)
    attack_len = int(0.06 * sr)
    decay_len = int(0.12 * sr)
    if word != "trái":
        env[:attack_len] = np.linspace(0.0, 1.0, attack_len)
    else:
        # Delayed attack for stop release
        env[:closure_samples] = 0.05
        post_closure = attack_len
        env[closure_samples:closure_samples + post_closure] = np.linspace(0.1, 1.0, post_closure)
    env[-decay_len:] = np.linspace(1.0, 0.0, decay_len)
    
    audio = output * env
    max_val = np.max(np.abs(audio)) + 1e-9
    audio = (audio / max_val) * 0.85
    return audio.astype(np.float32)


def generate_synthetic_silence_clip(sr: int = 16000, duration_sec: float = 1.0, mode: str = "dither") -> np.ndarray:
    n_samples = int(sr * duration_sec)
    if mode == "pure_silence":
        audio = np.zeros(n_samples, dtype=np.float32)
    elif mode == "dither":
        audio = np.random.randn(n_samples).astype(np.float32) * 1e-3
    elif mode == "room_tone":
        raw_noise = np.random.randn(n_samples).astype(np.float32)
        b, a = signal.butter(4, 1000.0 / (sr / 2.0), btype='low')
        filtered = signal.lfilter(b, a, raw_noise)
        rms_target = 10.0 ** (-50.0 / 20.0)
        current_rms = np.sqrt(np.mean(filtered ** 2)) + 1e-9
        audio = (filtered / current_rms) * rms_target
    else:
        audio = np.zeros(n_samples, dtype=np.float32)
    return np.clip(audio, -1.0, 1.0).astype(np.float32)
