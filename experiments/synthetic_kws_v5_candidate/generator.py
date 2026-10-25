"""
Synthetic KWS V5 Candidate Audio Generator.
Implements the Phase 12.2 Acoustic Modeling Specification:
  - Eliminates the V4 25ms stop-closure silence gap and static noise burst shortcuts.
  - Rosenberg polynomial glottal flow pulse with open quotient and spectral tilt.
  - Dynamic locus equation F2 transitions for initial consonant phonological contrast.
  - Variable stop closure duration with ambient leakage and glottal murmur (Gate 3).
  - Variable burst spectrum and continuous labiodental/sibilant frication.
  - Uniform temporal placement in 1.0s window (Gate 2).
  - 5 explicit ambient SILENCE regimes.
  - Algorithmic 20-word UNKNOWN vocabulary.
  - Pure deterministic per-sample seed generation.
"""

import os
import sys
import math
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

import numpy as np
import scipy.signal as signal
import soundfile as sf
import pandas as pd

from experiments.synthetic_kws_v5_candidate.config import (
    V5VoiceProfile,
    V5AcousticConfig,
    V5NoiseConfig,
    UNKNOWN_PILOT_VOCABULARY,
    TARGET_WORDS_SPEC,
)

SAMPLE_RATE = 16000
TOTAL_SAMPLES = 16000  # Exactly 1.0 second
CLASS_MAP = {
    "TRAI": 0,
    "PHAI": 1,
    "LEN": 2,
    "XUONG": 3,
    "UNKNOWN": 4,
    "SILENCE": 5
}


def sample_simulated_speaker(voice_id: str, rng: np.random.RandomState) -> V5VoiceProfile:
    """
    Deterministically samples vocal tract and glottal parameters for a synthetic speaker profile.
    """
    vtl_scale = float(rng.uniform(0.88, 1.12))
    f0_base = float(rng.uniform(110.0, 230.0))
    open_quotient = float(rng.uniform(0.45, 0.65))
    spectral_tilt_db = float(rng.uniform(-13.0, -7.0))
    jitter_pct = float(rng.uniform(0.4, 1.1))

    return V5VoiceProfile(
        voice_id=voice_id,
        vtl_scale=vtl_scale,
        f0_base=f0_base,
        open_quotient=open_quotient,
        spectral_tilt_db=spectral_tilt_db,
        jitter_pct=jitter_pct
    )


def generate_rosenberg_glottal_source(
    n_samples: int,
    f0_contour: np.ndarray,
    open_quotient: float,
    spectral_tilt_db: float,
    jitter_pct: float,
    rng: np.random.RandomState,
    sr: int = SAMPLE_RATE
) -> np.ndarray:
    """
    Generates a glottal source waveform using the Rosenberg polynomial flow pulse model
    with micro-intonation jitter and single-pole spectral tilt filtering.
    """
    # 1. Apply micro-jitter to F0
    jitter_noise = rng.uniform(-1.0, 1.0, n_samples)
    b_lp, a_lp = signal.butter(2, 35.0 / (sr / 2.0), btype='low')
    smooth_jitter = signal.lfilter(b_lp, a_lp, jitter_noise)
    jitter_factor = 1.0 + (jitter_pct * 0.01 * smooth_jitter)
    inst_f0 = np.clip(f0_contour * jitter_factor, 60.0, 500.0)

    # 2. Phase accumulation
    phase = 2.0 * np.pi * np.cumsum(inst_f0) / sr
    norm_phase = (phase / (2.0 * np.pi)) % 1.0

    # 3. Rosenberg polynomial glottal flow pulse
    # g(t) = 3*(t/Tp)^2 - 2*(t/Tp)^3 for t < Tp = Oq * T0, else 0
    oq = np.clip(open_quotient, 0.35, 0.75)
    s = np.clip(norm_phase / oq, 0.0, 1.0)
    source = np.where(norm_phase < oq, 3.0 * (s ** 2) - 2.0 * (s ** 3), 0.0)
    source = source - np.mean(source)

    # 4. Glottal spectral tilt: single-pole lowpass filter
    # Map spectral tilt (-14 to -6 dB/oct) to pole parameter alpha
    alpha = np.clip(0.40 + (-spectral_tilt_db - 6.0) * 0.045, 0.35, 0.80)
    b_tilt = [1.0 - alpha]
    a_tilt = [1.0, -alpha]
    filtered_source = signal.lfilter(b_tilt, a_tilt, source)

    norm = np.max(np.abs(filtered_source)) + 1e-9
    return (filtered_source / norm).astype(np.float32)


def apply_formant_resonators(
    source: np.ndarray,
    f1_track: np.ndarray,
    f2_track: np.ndarray,
    f3_track: np.ndarray,
    b1: float,
    b2: float,
    b3: float,
    sr: int = SAMPLE_RATE,
    chunk_size: int = 160
) -> np.ndarray:
    """
    Applies 3 parallel/cascade vocal tract formant resonators using time-varying
    IIR peak filters with continuous filter state retention across chunks.
    """
    n_samples = len(source)
    output = np.zeros(n_samples, dtype=np.float32)

    zi1 = np.zeros(2, dtype=np.float32)
    zi2 = np.zeros(2, dtype=np.float32)
    zi3 = np.zeros(2, dtype=np.float32)

    nyq = sr / 2.0

    for i in range(0, n_samples, chunk_size):
        end = min(i + chunk_size, n_samples)
        mid = (i + end) // 2

        f1_c = np.clip(f1_track[mid], 150.0, nyq - 200.0)
        f2_c = np.clip(f2_track[mid], 300.0, nyq - 200.0)
        f3_c = np.clip(f3_track[mid], 1200.0, nyq - 200.0)

        q1 = max(1.5, f1_c / max(b1, 40.0))
        q2 = max(2.0, f2_c / max(b2, 50.0))
        q3 = max(3.0, f3_c / max(b3, 60.0))

        b_iir1, a_iir1 = signal.iirpeak(f1_c, Q=q1, fs=sr)
        b_iir2, a_iir2 = signal.iirpeak(f2_c, Q=q2, fs=sr)
        b_iir3, a_iir3 = signal.iirpeak(f3_c, Q=q3, fs=sr)

        src_chunk = source[i:end].astype(np.float64)

        out1, zi1 = signal.lfilter(b_iir1, a_iir1, src_chunk, zi=zi1)
        out2, zi2 = signal.lfilter(b_iir2, a_iir2, src_chunk, zi=zi2)
        out3, zi3 = signal.lfilter(b_iir3, a_iir3, src_chunk, zi=zi3)

        chunk_out = 1.0 * out1 + 0.65 * out2 + 0.35 * out3
        output[i:end] = chunk_out.astype(np.float32)

    return output


def generate_v5_speech_clip(config: V5AcousticConfig, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Generates a single 1.0s (16000 samples) synthetic speech clip for the candidate dataset.
    Eliminates the V4 stop-closure shortcut and static templates.
    """
    rng = np.random.RandomState(config.seed)
    dur_sec = config.duration_sec
    n_speech = int(sr * dur_sec)
    t_speech = np.linspace(0, dur_sec, n_speech, endpoint=False)
    prof = config.voice_profile
    vtl = prof.vtl_scale
    f0_base = prof.f0_base

    # 1. Pitch / F0 contour by tone
    word = config.intended_text.strip().lower()
    
    # Identify tone category
    if word in ["trái", "xuống", "số", "núi", "mái", "cuống"]:
        # Sắc (rising)
        f0_track = f0_base * (1.0 + 0.34 * (t_speech / dur_sec) ** 1.35)
    elif word in ["phải", "biển"]:
        # Hỏi (dipping-rising)
        f0_track = f0_base * (1.0 - 0.22 * np.sin(np.pi * t_speech / dur_sec) + 0.16 * (t_speech / dur_sec) ** 2)
    elif word in ["lên", "đi", "xe", "hoa", "mây", "sông", "pha", "tai"]:
        # Ngang (level)
        f0_track = f0_base * (1.0 + 0.02 * np.sin(2.0 * np.pi * 3.0 * t_speech))
    elif word in ["chào", "nhà", "ngày", "rừng", "trời", "đèn"]:
        # Huyền (falling)
        f0_track = f0_base * (1.04 - 0.22 * (t_speech / dur_sec))
    elif word in ["bạn", "học"]:
        # Nặng (heavy / constricted falling)
        f0_track = f0_base * (1.0 - 0.30 * (t_speech / dur_sec) ** 0.8)
    else:
        f0_track = f0_base * (1.0 - 0.12 * (t_speech / dur_sec))

    # 2. Glottal source
    glottal = generate_rosenberg_glottal_source(
        n_samples=n_speech,
        f0_contour=f0_track,
        open_quotient=prof.open_quotient,
        spectral_tilt_db=prof.spectral_tilt_db,
        jitter_pct=prof.jitter_pct,
        rng=rng,
        sr=sr
    )

    # 3. Formant trajectories with VTL scaling and Locus Equation onset
    b1_val = float(rng.normal(90.0, 12.0))
    b2_val = float(rng.normal(120.0, 18.0))
    b3_val = float(rng.normal(160.0, 22.0))

    if word == "trái":
        # /tɕa:j/
        # Locus Equation: F2 onset locus is HIGH (1750-1950 Hz * vtl) for affricate /tɕ/
        f2_onset_locus = float(rng.uniform(1750.0, 1950.0)) * vtl
        f2_vowel = 1350.0 * vtl
        f2_offglide = 2050.0 * vtl

        transition_dur = float(rng.uniform(0.045, 0.075))
        n_trans = int(transition_dur * sr)
        
        f1_track = np.linspace(420.0 * vtl, 820.0 * vtl, n_speech)
        f2_track = np.zeros(n_speech)
        if n_trans < n_speech:
            f2_track[:n_trans] = np.linspace(f2_onset_locus, f2_vowel, n_trans)
            f2_track[n_trans:] = np.linspace(f2_vowel, f2_offglide, n_speech - n_trans)
        else:
            f2_track[:] = np.linspace(f2_onset_locus, f2_vowel, n_speech)
        f3_track = np.full(n_speech, 2650.0 * vtl)

    elif word == "phải":
        # /fa:j/
        # Locus Equation: F2 onset locus is LOW (1000-1200 Hz * vtl) for labiodental /f/
        f2_onset_locus = float(rng.uniform(1000.0, 1200.0)) * vtl
        f2_vowel = 1350.0 * vtl
        f2_offglide = 2050.0 * vtl

        transition_dur = float(rng.uniform(0.050, 0.080))
        n_trans = int(transition_dur * sr)

        f1_track = np.linspace(450.0 * vtl, 820.0 * vtl, n_speech)
        f2_track = np.zeros(n_speech)
        if n_trans < n_speech:
            f2_track[:n_trans] = np.linspace(f2_onset_locus, f2_vowel, n_trans)
            f2_track[n_trans:] = np.linspace(f2_vowel, f2_offglide, n_speech - n_trans)
        else:
            f2_track[:] = np.linspace(f2_onset_locus, f2_vowel, n_speech)
        f3_track = np.full(n_speech, 2650.0 * vtl)

    elif word == "lên":
        # /le:n/ - lateral approximant into front vowel /e/ and alveolar nasal /n/
        f1_track = np.linspace(460.0 * vtl, 370.0 * vtl, n_speech)
        f2_track = np.linspace(1750.0 * vtl, 1950.0 * vtl, n_speech)
        f3_track = np.full(n_speech, 2700.0 * vtl)

    elif word == "xuống":
        # /suəŋ/ - sibilant into back diphthong /uə/ and velar nasal /ŋ/
        f1_track = np.linspace(380.0 * vtl, 480.0 * vtl, n_speech)
        f2_track = np.linspace(850.0 * vtl, 1050.0 * vtl, n_speech)
        f3_track = np.full(n_speech, 2400.0 * vtl)

    else:
        # UNKNOWN syllables: dynamic formant mapping
        if any(v in word for v in ["a", "à", "á", "ạ", "ả"]):
            f1_base, f2_base = 800.0 * vtl, 1300.0 * vtl
        elif any(v in word for v in ["e", "è", "é", "ẹ", "ẻ", "ê", "ề", "ế"]):
            f1_base, f2_base = 450.0 * vtl, 1850.0 * vtl
        elif any(v in word for v in ["i", "ì", "í", "ị", "ỉ"]):
            f1_base, f2_base = 320.0 * vtl, 2200.0 * vtl
        elif any(v in word for v in ["o", "ò", "ó", "ọ", "ỏ", "ô", "ố", "ơ", "ớ"]):
            f1_base, f2_base = 520.0 * vtl, 980.0 * vtl
        elif any(v in word for v in ["u", "ù", "ú", "ụ", "ủ", "ư", "ứ"]):
            f1_base, f2_base = 380.0 * vtl, 880.0 * vtl
        else:
            f1_base, f2_base = 600.0 * vtl, 1400.0 * vtl

        f1_track = np.linspace(f1_base * 0.9, f1_base * 1.05, n_speech)
        f2_track = np.linspace(f2_base * 0.95, f2_base * 1.05, n_speech)
        f3_track = np.full(n_speech, 2550.0 * vtl)

    # 4. Formant resonant filtering
    vocal_resonated = apply_formant_resonators(
        glottal, f1_track, f2_track, f3_track, b1_val, b2_val, b3_val, sr=sr
    )

    # 5. Consonant onset synthesis & elimination of V4 shortcut
    output_speech = np.copy(vocal_resonated)

    if word == "trái":
        # Voiceless stop/affricate (/tɕ/)
        # Continuous closure variation: 16 to 32 ms
        closure_sec = float(rng.uniform(0.016, 0.032))
        closure_samples = int(closure_sec * sr)
        
        # Acoustic leakage during closure: glottal murmur + ambient room leakage
        # Attenuation factor ~ 0.08 to 0.14 (RMS ~ -20 dB to -17 dB below peak)
        # CRITICAL: NO ZERO CLAMP! Output amplitude in closure remains around 0.04-0.08
        leak_factor = float(rng.uniform(0.09, 0.15))
        output_speech[:closure_samples] *= leak_factor

        # Add low-frequency murmur leakage during closure
        t_close = np.linspace(0, closure_sec, closure_samples, endpoint=False)
        murmur = 0.04 * np.sin(2.0 * np.pi * f0_base * t_close)
        output_speech[:closure_samples] += murmur.astype(np.float32)

        # Release burst: 10 to 18 ms
        burst_sec = float(rng.uniform(0.010, 0.018))
        burst_samples = int(burst_sec * sr)
        burst_noise = rng.normal(0.0, 1.0, burst_samples).astype(np.float32)

        f_burst = float(rng.uniform(3400.0, 4400.0)) * vtl
        q_burst = float(rng.uniform(4.0, 7.0))
        b_bp, a_bp = signal.iirpeak(min(f_burst, (sr / 2.0) - 200.0), Q=q_burst, fs=sr)
        burst_filt = signal.lfilter(b_bp, a_bp, burst_noise)

        # Exponential decay envelope for burst
        burst_decay = np.exp(-np.linspace(0.0, 4.0, burst_samples))
        burst_amp = float(rng.uniform(0.35, 0.55))
        burst_signal = (burst_filt * burst_decay * burst_amp).astype(np.float32)

        burst_start = closure_samples
        burst_end = min(burst_start + burst_samples, n_speech)
        output_speech[burst_start:burst_end] += burst_signal[:burst_end - burst_start]

    elif word == "phải":
        # Continuous labiodental fricative (/f/)
        fric_sec = float(rng.uniform(0.045, 0.075))
        fric_samples = int(fric_sec * sr)
        fric_noise = rng.normal(0.0, 1.0, fric_samples).astype(np.float32)

        f_low = float(rng.uniform(1600.0, 2000.0))
        f_high = float(rng.uniform(5800.0, 6800.0))
        b_bp, a_bp = signal.butter(3, [f_low / (sr / 2.0), f_high / (sr / 2.0)], btype='band')
        fric_filt = signal.lfilter(b_bp, a_bp, fric_noise)

        # Smooth attack envelope (rise time 20-35ms, decay into vowel)
        rise_samples = int(fric_samples * 0.45)
        fric_env = np.ones(fric_samples, dtype=np.float32)
        fric_env[:rise_samples] = np.linspace(0.1, 1.0, rise_samples)
        fric_env[rise_samples:] = np.linspace(1.0, 0.2, fric_samples - rise_samples)
        fric_amp = float(rng.uniform(0.25, 0.40))
        fric_signal = (fric_filt * fric_env * fric_amp).astype(np.float32)

        fric_end = min(fric_samples, n_speech)
        output_speech[:fric_end] += fric_signal[:fric_end]

    elif word == "xuống":
        # Sibilant fricative (/s/)
        fric_sec = float(rng.uniform(0.050, 0.070))
        fric_samples = int(fric_sec * sr)
        fric_noise = rng.normal(0.0, 1.0, fric_samples).astype(np.float32)

        b_hp, a_hp = signal.butter(4, 3800.0 / (sr / 2.0), btype='high')
        fric_filt = signal.lfilter(b_hp, a_hp, fric_noise)

        fric_env = np.linspace(0.8, 0.2, fric_samples)
        fric_amp = float(rng.uniform(0.30, 0.45))
        fric_signal = (fric_filt * fric_env * fric_amp).astype(np.float32)

        fric_end = min(fric_samples, n_speech)
        output_speech[:fric_end] += fric_signal[:fric_end]

    elif word == "lên":
        # Voiced lateral attack: smooth rise, no sharp transient or closure
        pass

    else:
        # UNKNOWN onset synthesis by consonant class
        if word.startswith(("ch", "tr")):
            # Affricate
            close_s = int(0.020 * sr)
            output_speech[:close_s] *= 0.12
            burst_s = int(0.012 * sr)
            b_hp, a_hp = signal.butter(3, 3500.0 / (sr / 2.0), btype='high')
            b_noise = signal.lfilter(b_hp, a_hp, rng.normal(0.0, 1.0, burst_s)) * 0.35
            output_speech[close_s:min(close_s + burst_s, n_speech)] += b_noise[:min(burst_s, n_speech - close_s)]
        elif word.startswith(("x", "s", "ph", "h")):
            # Fricative
            fric_s = int(0.045 * sr)
            b_bp, a_bp = signal.butter(3, [2000.0 / (sr / 2.0), 6500.0 / (sr / 2.0)], btype='band')
            f_noise = signal.lfilter(b_bp, a_bp, rng.normal(0.0, 1.0, fric_s)) * 0.25
            output_speech[:min(fric_s, n_speech)] += f_noise[:min(fric_s, n_speech)]
        elif word.startswith(("b", "đ", "c")):
            # Stop
            close_s = int(0.018 * sr)
            output_speech[:close_s] *= 0.10
            burst_s = int(0.010 * sr)
            b_bp, a_bp = signal.butter(3, [1200.0 / (sr / 2.0), 3200.0 / (sr / 2.0)], btype='band')
            b_noise = signal.lfilter(b_bp, a_bp, rng.normal(0.0, 1.0, burst_s)) * 0.30
            output_speech[close_s:min(close_s + burst_s, n_speech)] += b_noise[:min(burst_s, n_speech - close_s)]

    # 6. Utterance amplitude envelope
    attack_samples = int(float(rng.uniform(0.025, 0.050)) * sr)
    decay_samples = int(float(rng.uniform(0.080, 0.140)) * sr)
    env = np.ones(n_speech, dtype=np.float32)
    
    if word not in ["trái"]:
        env[:attack_samples] = np.linspace(0.05, 1.0, attack_samples)
    else:
        # For trái: rise occurs after closure/burst
        post_close = closure_samples + burst_samples
        if post_close < n_speech:
            env[:closure_samples] = np.linspace(0.4, 0.6, closure_samples)
            env[closure_samples:post_close] = np.linspace(0.7, 1.0, post_close - closure_samples)
    env[-decay_samples:] = np.linspace(1.0, 0.02, decay_samples)

    active_speech = output_speech * env

    # Normalize active speech
    speech_peak = np.max(np.abs(active_speech)) + 1e-9
    active_speech = active_speech / speech_peak

    # 7. Placement in 1.0s (16000 samples) window
    clip = np.zeros(TOTAL_SAMPLES, dtype=np.float32)
    onset_idx = int(config.onset_offset_sec * sr)
    onset_idx = np.clip(onset_idx, 0, TOTAL_SAMPLES - n_speech)
    clip[onset_idx:onset_idx + n_speech] = active_speech

    # 8. Continuous background floor & SNR mixing
    # Synthesize background matching config.noise_type
    bg_noise = np.zeros(TOTAL_SAMPLES, dtype=np.float32)
    if config.noise_type == "room_tone":
        raw = rng.normal(0.0, 1.0, TOTAL_SAMPLES).astype(np.float32)
        b_bg, a_bg = signal.butter(3, float(rng.uniform(700.0, 1200.0)) / (sr / 2.0), btype='low')
        bg_noise = signal.lfilter(b_bg, a_bg, raw).astype(np.float32)
    elif config.noise_type == "dither":
        bg_noise = rng.normal(0.0, 1.0, TOTAL_SAMPLES).astype(np.float32)
    elif config.noise_type == "hum":
        t_clip = np.linspace(0, 1.0, TOTAL_SAMPLES, endpoint=False)
        f_hum = 50.0 if rng.rand() > 0.5 else 60.0
        bg_noise = (np.sin(2.0 * np.pi * f_hum * t_clip) +
                    0.4 * np.sin(2.0 * np.pi * 3.0 * f_hum * t_clip)).astype(np.float32)
    else:
        # Default low-level quiet floor
        bg_noise = rng.normal(0.0, 1.0, TOTAL_SAMPLES).astype(np.float32) * 1e-4

    # Scale background by SNR
    speech_rms = np.sqrt(np.mean(active_speech ** 2)) + 1e-9
    bg_rms = np.sqrt(np.mean(bg_noise ** 2)) + 1e-9
    snr_linear = 10.0 ** (config.snr_db / 20.0)
    target_bg_rms = speech_rms / snr_linear
    scaled_bg = bg_noise * (target_bg_rms / bg_rms)

    combined = clip + scaled_bg

    # 9. Normalization to target peak ceiling
    peak = np.max(np.abs(combined)) + 1e-9
    final_audio = (combined / peak) * config.peak_target
    return np.clip(final_audio, -1.0, 1.0).astype(np.float32)


def generate_v5_silence_clip(config: V5NoiseConfig, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Generates a 1.0s (16000 samples) non-speech / ambient SILENCE clip across 5 explicit regimes.
    """
    rng = np.random.RandomState(config.seed)
    n_samples = int(sr * config.duration_sec)
    regime = config.regime

    if regime == "exact_zeros":
        return np.zeros(n_samples, dtype=np.float32)

    elif regime == "low_level_dither":
        # Gaussian noise at -68 dBFS to -55 dBFS
        raw = rng.normal(0.0, 1.0, n_samples).astype(np.float32)
        rms_target = 10.0 ** (float(rng.uniform(-68.0, -55.0)) / 20.0)
        curr_rms = np.sqrt(np.mean(raw ** 2)) + 1e-9
        return (raw * (rms_target / curr_rms)).astype(np.float32)

    elif regime == "room_tone":
        # Butterworth lowpass filtered noise (-50 dBFS to -38 dBFS)
        raw = rng.normal(0.0, 1.0, n_samples).astype(np.float32)
        cutoff = float(rng.uniform(600.0, 1200.0))
        b_lp, a_lp = signal.butter(4, cutoff / (sr / 2.0), btype='low')
        filtered = signal.lfilter(b_lp, a_lp, raw).astype(np.float32)
        rms_target = 10.0 ** (float(rng.uniform(-50.0, -38.0)) / 20.0)
        curr_rms = np.sqrt(np.mean(filtered ** 2)) + 1e-9
        return (filtered * (rms_target / curr_rms)).astype(np.float32)

    elif regime == "synthetic_hum":
        # 50 Hz or 60 Hz fundamental + 3rd and 5th harmonics (-48 dBFS to -36 dBFS)
        t = np.linspace(0, config.duration_sec, n_samples, endpoint=False)
        f_hum = 50.0 if rng.rand() > 0.5 else 60.0
        hum = (np.sin(2.0 * np.pi * f_hum * t) +
               0.35 * np.sin(2.0 * np.pi * 3.0 * f_hum * t) +
               0.15 * np.sin(2.0 * np.pi * 5.0 * f_hum * t)).astype(np.float32)
        rms_target = 10.0 ** (float(rng.uniform(-48.0, -36.0)) / 20.0)
        curr_rms = np.sqrt(np.mean(hum ** 2)) + 1e-9
        return (hum * (rms_target / curr_rms)).astype(np.float32)

    elif regime == "non_speech_transient":
        # Quiet room tone floor (-55 dBFS) + 1 to 3 short impulse clicks (< 6 ms)
        raw = rng.normal(0.0, 1.0, n_samples).astype(np.float32)
        b_lp, a_lp = signal.butter(3, 800.0 / (sr / 2.0), btype='low')
        floor_tone = signal.lfilter(b_lp, a_lp, raw).astype(np.float32)
        rms_target = 10.0 ** (-55.0 / 20.0)
        curr_rms = np.sqrt(np.mean(floor_tone ** 2)) + 1e-9
        audio = floor_tone * (rms_target / curr_rms)

        num_clicks = int(rng.randint(1, 4))
        for _ in range(num_clicks):
            click_len = int(float(rng.uniform(0.002, 0.006)) * sr)
            pos = int(rng.randint(sr // 10, n_samples - click_len - sr // 10))
            click_raw = rng.normal(0.0, 1.0, click_len).astype(np.float32)
            b_click, a_click = signal.butter(2, [2000.0 / (sr / 2.0), 6000.0 / (sr / 2.0)], btype='band')
            click_filt = signal.lfilter(b_click, a_click, click_raw)
            click_env = np.exp(-np.linspace(0.0, 5.0, click_len))
            click_amp = float(rng.uniform(0.05, 0.14))
            audio[pos:pos + click_len] += (click_filt * click_env * click_amp).astype(np.float32)

        return np.clip(audio, -1.0, 1.0).astype(np.float32)

    else:
        return np.zeros(n_samples, dtype=np.float32)


def generate_v5_pilot_dataset(output_dir: Path, pilot_seed_base: int = 50000) -> pd.DataFrame:
    """
    Generates a deterministic 120-clip pilot dataset (20 clips per class x 6 classes)
    and saves WAV files and manifest.csv.
    """
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    master_rng = np.random.RandomState(pilot_seed_base)
    simulated_speakers = [
        sample_simulated_speaker(f"SPK_SYNTH_{i+1:03d}", master_rng)
        for i in range(20)
    ]

    records = []
    classes = ["TRAI", "PHAI", "LEN", "XUONG", "UNKNOWN", "SILENCE"]
    silence_regimes = [
        "exact_zeros", "exact_zeros", "exact_zeros", "exact_zeros",
        "low_level_dither", "low_level_dither", "low_level_dither", "low_level_dither",
        "room_tone", "room_tone", "room_tone", "room_tone",
        "synthetic_hum", "synthetic_hum", "synthetic_hum", "synthetic_hum",
        "non_speech_transient", "non_speech_transient", "non_speech_transient", "non_speech_transient"
    ]

    for class_label in classes:
        for idx in range(20):
            sample_seed = pilot_seed_base + (CLASS_MAP[class_label] * 100) + idx
            sample_rng = np.random.RandomState(sample_seed)
            recording_id = f"v5_pilot_{class_label.lower()}_{idx+1:03d}"
            wav_path = audio_dir / f"{recording_id}.wav"
            rel_path = f"experiments/synthetic_kws_v5_candidate/audio/{recording_id}.wav"

            if class_label == "SILENCE":
                regime = silence_regimes[idx]
                noise_cfg = V5NoiseConfig(
                    sample_id=recording_id,
                    regime=regime,
                    seed=sample_seed
                )
                audio = generate_v5_silence_clip(noise_cfg, sr=SAMPLE_RATE)

                # Save 16-bit PCM WAV
                sf.write(str(wav_path), audio, SAMPLE_RATE, subtype="PCM_16")
                with open(wav_path, "rb") as f:
                    file_sha = hashlib.sha256(f.read()).hexdigest().upper()

                records.append({
                    "recording_id": recording_id,
                    "split": "pilot",
                    "class_id": CLASS_MAP[class_label],
                    "class_label": class_label,
                    "transcript": f"__silence_{regime}__",
                    "generator_version": "v5.0-pilot",
                    "generator_recipe": "ambient_synthetic_v5",
                    "seed": sample_seed,
                    "simulated_speaker_id": "NONE",
                    "vtl_scale": 1.0,
                    "f0_base": 0.0,
                    "duration_sec": 1.0,
                    "onset_offset_sec": 0.0,
                    "snr_db": -99.0,
                    "noise_type": regime,
                    "sha256": file_sha,
                    "processed_path": rel_path,
                    "status": "VALID"
                })

            else:
                spk = simulated_speakers[idx % len(simulated_speakers)]
                if class_label in TARGET_WORDS_SPEC:
                    intended_word = TARGET_WORDS_SPEC[class_label]["word"]
                else:
                    intended_word = UNKNOWN_PILOT_VOCABULARY[idx][0]

                # Uniform onset distribution spanning 20ms to 350ms (Gate 2)
                onset_offset = float(sample_rng.uniform(0.025, 0.320))
                # Active utterance duration 300ms to 620ms
                duration_sec = float(sample_rng.uniform(0.32, 0.60))
                if onset_offset + duration_sec > 0.98:
                    duration_sec = 0.98 - onset_offset

                snr_db = float(sample_rng.uniform(20.0, 36.0))
                noise_type = sample_rng.choice(["room_tone", "dither", "hum"])

                acoustic_cfg = V5AcousticConfig(
                    sample_id=recording_id,
                    class_label=class_label,
                    intended_text=intended_word,
                    seed=sample_seed,
                    voice_profile=spk,
                    duration_sec=duration_sec,
                    onset_offset_sec=onset_offset,
                    snr_db=snr_db,
                    noise_type=noise_type,
                    peak_target=0.85
                )

                audio = generate_v5_speech_clip(acoustic_cfg, sr=SAMPLE_RATE)

                # Save 16-bit PCM WAV
                sf.write(str(wav_path), audio, SAMPLE_RATE, subtype="PCM_16")
                with open(wav_path, "rb") as f:
                    file_sha = hashlib.sha256(f.read()).hexdigest().upper()

                records.append({
                    "recording_id": recording_id,
                    "split": "pilot",
                    "class_id": CLASS_MAP[class_label],
                    "class_label": class_label,
                    "transcript": intended_word,
                    "generator_version": "v5.0-pilot",
                    "generator_recipe": "parametric_coarticulated_v5",
                    "seed": sample_seed,
                    "simulated_speaker_id": spk.voice_id,
                    "vtl_scale": round(spk.vtl_scale, 4),
                    "f0_base": round(spk.f0_base, 2),
                    "duration_sec": round(duration_sec, 4),
                    "onset_offset_sec": round(onset_offset, 4),
                    "snr_db": round(snr_db, 2),
                    "noise_type": noise_type,
                    "sha256": file_sha,
                    "processed_path": rel_path,
                    "status": "VALID"
                })

    df = pd.DataFrame(records)
    manifest_path = output_dir / "manifest.csv"
    df.to_csv(manifest_path, index=False)
    return df


def reproduce_pilot_clip(
    recording_id: str,
    pilot_seed_base: int = 50000
) -> Tuple[np.ndarray, str, Dict[str, Any]]:
    """
    Deterministically reproduces a single pilot clip by recording_id from the seed base.
    Returns (audio, sha256, metadata).
    """
    classes = ["TRAI", "PHAI", "LEN", "XUONG", "UNKNOWN", "SILENCE"]
    silence_regimes = [
        "exact_zeros", "exact_zeros", "exact_zeros", "exact_zeros",
        "low_level_dither", "low_level_dither", "low_level_dither", "low_level_dither",
        "room_tone", "room_tone", "room_tone", "room_tone",
        "synthetic_hum", "synthetic_hum", "synthetic_hum", "synthetic_hum",
        "non_speech_transient", "non_speech_transient", "non_speech_transient", "non_speech_transient"
    ]

    master_rng = np.random.RandomState(pilot_seed_base)
    simulated_speakers = [
        sample_simulated_speaker(f"SPK_SYNTH_{i+1:03d}", master_rng)
        for i in range(20)
    ]

    # Parse recording_id, e.g. "v5_pilot_trai_001"
    parts = recording_id.split("_")
    class_label = parts[2].upper()
    idx = int(parts[3]) - 1

    sample_seed = pilot_seed_base + (CLASS_MAP[class_label] * 100) + idx
    sample_rng = np.random.RandomState(sample_seed)

    if class_label == "SILENCE":
        regime = silence_regimes[idx]
        noise_cfg = V5NoiseConfig(
            sample_id=recording_id,
            regime=regime,
            seed=sample_seed
        )
        audio = generate_v5_silence_clip(noise_cfg, sr=SAMPLE_RATE)
        meta = {
            "recording_id": recording_id,
            "class_label": class_label,
            "regime": regime,
            "seed": sample_seed
        }
    else:
        spk = simulated_speakers[idx % len(simulated_speakers)]
        if class_label in TARGET_WORDS_SPEC:
            intended_word = TARGET_WORDS_SPEC[class_label]["word"]
        else:
            intended_word = UNKNOWN_PILOT_VOCABULARY[idx][0]

        onset_offset = float(sample_rng.uniform(0.025, 0.320))
        duration_sec = float(sample_rng.uniform(0.32, 0.60))
        if onset_offset + duration_sec > 0.98:
            duration_sec = 0.98 - onset_offset

        snr_db = float(sample_rng.uniform(20.0, 36.0))
        noise_type = sample_rng.choice(["room_tone", "dither", "hum"])

        acoustic_cfg = V5AcousticConfig(
            sample_id=recording_id,
            class_label=class_label,
            intended_text=intended_word,
            seed=sample_seed,
            voice_profile=spk,
            duration_sec=duration_sec,
            onset_offset_sec=onset_offset,
            snr_db=snr_db,
            noise_type=noise_type,
            peak_target=0.85
        )
        audio = generate_v5_speech_clip(acoustic_cfg, sr=SAMPLE_RATE)
        meta = {
            "recording_id": recording_id,
            "class_label": class_label,
            "intended_text": intended_word,
            "seed": sample_seed,
            "simulated_speaker_id": spk.voice_id
        }

    import io
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    sha256 = hashlib.sha256(buf.getvalue()).hexdigest().upper()
    return audio, sha256, meta


if __name__ == "__main__":
    out = Path("experiments/synthetic_kws_v5_candidate")
    df = generate_v5_pilot_dataset(out)
    print(f"Generated {len(df)} pilot clips at {out / 'manifest.csv'}")
