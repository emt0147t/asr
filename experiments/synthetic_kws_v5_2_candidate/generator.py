"""
Synthetic KWS V5.2 Candidate Generator
Isolated implementation remediating the remaining residual RMS variance:
- Peak headroom calibrated to 0.98 and speech RMS range [-17.0, -14.0] dBFS
- Eliminates premature soft-knee compression on unvoiced fricatives (PHAI)
- Maintains independent room noise floor and two-tier taxonomy
- 100% row-independent reproducibility from manifest row data
"""

import math
import hashlib
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
from scipy import signal
from typing import Dict, Any, Tuple, List

try:
    from experiments.synthetic_kws_v5_2_candidate.config import (
        V5_2VoiceProfile,
        V5_2AcousticConfig,
        V5_2NoiseConfig,
        UNKNOWN_PILOT_VOCABULARY_V5_2,
        TARGET_WORDS_SPEC_V5_2
    )
except ImportError:
    from config import (
        V5_2VoiceProfile,
        V5_2AcousticConfig,
        V5_2NoiseConfig,
        UNKNOWN_PILOT_VOCABULARY_V5_2,
        TARGET_WORDS_SPEC_V5_2
    )

SAMPLE_RATE = 16000
CLIP_DURATION_SEC = 1.0
TOTAL_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION_SEC)

SPEAKER_PROFILES_V5_2 = [
    V5_2VoiceProfile("SPK_V5_2_01_FEMALE_HIGH", 0.90, 220.0, 0.50, -10.0, 0.006),
    V5_2VoiceProfile("SPK_V5_2_02_FEMALE_MID", 0.94, 195.0, 0.55, -11.0, 0.007),
    V5_2VoiceProfile("SPK_V5_2_03_MALE_LOW", 1.10, 115.0, 0.62, -13.0, 0.008),
    V5_2VoiceProfile("SPK_V5_2_04_MALE_MID", 1.04, 135.0, 0.58, -12.0, 0.007),
    V5_2VoiceProfile("SPK_V5_2_05_CHILD_NEUTRAL", 0.86, 245.0, 0.48, -9.0, 0.005),
    V5_2VoiceProfile("SPK_V5_2_06_ADULT_DEEP", 1.14, 100.0, 0.65, -14.0, 0.009),
    V5_2VoiceProfile("SPK_V5_2_07_FEMALE_BREATHY", 0.92, 210.0, 0.68, -15.0, 0.010),
    V5_2VoiceProfile("SPK_V5_2_08_MALE_TIGHT", 1.02, 145.0, 0.45, -8.0, 0.005),
]

def sample_simulated_speaker_v5_2(voice_id: str, rng: np.random.RandomState) -> V5_2VoiceProfile:
    for spk in SPEAKER_PROFILES_V5_2:
        if spk.voice_id == voice_id:
            return V5_2VoiceProfile(
                voice_id=spk.voice_id,
                vtl_scale=float(spk.vtl_scale * rng.uniform(0.98, 1.02)),
                f0_base=float(spk.f0_base * rng.uniform(0.96, 1.04)),
                open_quotient=float(np.clip(spk.open_quotient * rng.uniform(0.95, 1.05), 0.35, 0.75)),
                spectral_tilt_db=float(spk.spectral_tilt_db + rng.uniform(-1.0, 1.0)),
                jitter_pct=float(np.clip(spk.jitter_pct * rng.uniform(0.90, 1.10), 0.002, 0.015))
            )
    return SPEAKER_PROFILES_V5_2[0]


def generate_rosenberg_glottal_source(
    n_samples: int,
    f0_track: np.ndarray,
    open_quotient: float,
    spectral_tilt_db: float,
    jitter_pct: float,
    rng: np.random.RandomState,
    sr: int = SAMPLE_RATE
) -> np.ndarray:
    source = np.zeros(n_samples, dtype=np.float32)
    phase = 0.0
    for i in range(n_samples):
        jitter_val = 1.0 + float(rng.uniform(-jitter_pct, jitter_pct))
        inst_f0 = f0_track[i] * jitter_val
        phase_inc = inst_f0 / float(sr)
        phase = (phase + phase_inc) % 1.0
        if phase < open_quotient:
            t_open = phase / open_quotient
            val = 0.5 * (1.0 - np.cos(np.pi * t_open))
        else:
            t_close = (phase - open_quotient) / (1.0 - open_quotient + 1e-9)
            val = np.cos(0.5 * np.pi * t_close)
        source[i] = float(val)
    
    source = source - np.mean(source)
    tilt_decay = float(10.0 ** (spectral_tilt_db / 40.0))
    b_tilt = [1.0 - tilt_decay]
    a_tilt = [1.0, -tilt_decay]
    filtered_source = signal.lfilter(b_tilt, a_tilt, source)
    norm_val = np.max(np.abs(filtered_source)) + 1e-9
    return (filtered_source / norm_val).astype(np.float32)


def apply_formant_resonators(
    glottal: np.ndarray,
    f1: np.ndarray,
    f2: np.ndarray,
    f3: np.ndarray,
    b1: float,
    b2: float,
    b3: float,
    sr: int = SAMPLE_RATE
) -> np.ndarray:
    n = len(glottal)
    out1 = np.zeros(n, dtype=np.float32)
    out2 = np.zeros(n, dtype=np.float32)
    out3 = np.zeros(n, dtype=np.float32)
    y1_1, y1_2 = 0.0, 0.0
    y2_1, y2_2 = 0.0, 0.0
    y3_1, y3_2 = 0.0, 0.0
    two_pi_dt = 2.0 * np.pi / float(sr)
    r1 = np.exp(-np.pi * b1 / float(sr))
    r2 = np.exp(-np.pi * b2 / float(sr))
    r3 = np.exp(-np.pi * b3 / float(sr))
    
    for i in range(n):
        theta1 = f1[i] * two_pi_dt
        a1_1 = 2.0 * r1 * np.cos(theta1)
        a1_2 = -r1 * r1
        g1 = 1.0 - r1
        y1_0 = g1 * glottal[i] + a1_1 * y1_1 + a1_2 * y1_2
        out1[i] = y1_0
        y1_2 = y1_1
        y1_1 = y1_0

        theta2 = f2[i] * two_pi_dt
        a2_1 = 2.0 * r2 * np.cos(theta2)
        a2_2 = -r2 * r2
        g2 = 1.0 - r2
        y2_0 = g2 * glottal[i] + a2_1 * y2_1 + a2_2 * y2_2
        out2[i] = y2_0
        y2_2 = y2_1
        y2_1 = y2_0

        theta3 = f3[i] * two_pi_dt
        a3_1 = 2.0 * r3 * np.cos(theta3)
        a3_2 = -r3 * r3
        g3 = 1.0 - r3
        y3_0 = g3 * glottal[i] + a3_1 * y3_1 + a3_2 * y3_2
        out3[i] = y3_0
        y3_2 = y3_1
        y3_1 = y3_0

    output = out1 * 0.55 + out2 * 0.35 + out3 * 0.15
    norm_factor = np.max(np.abs(output)) + 1e-9
    return (output / norm_factor).astype(np.float32)


def generate_v5_2_speech_clip(
    config: V5_2AcousticConfig,
    sr: int = SAMPLE_RATE
) -> Tuple[np.ndarray, float, float]:
    """
    Generates a 1.0s clip in V5.2 with calibrated speech RMS target and 0.98 peak headroom.
    """
    rng = np.random.RandomState(config.seed)
    dur_sec = config.duration_sec
    n_speech = int(sr * dur_sec)
    t_speech = np.linspace(0, dur_sec, n_speech, endpoint=False)
    prof = config.voice_profile
    vtl = prof.vtl_scale
    f0_base = prof.f0_base

    word = config.intended_text.strip().lower()

    # 1. Pitch contour
    if word == "trái":
        f0_track = f0_base * (1.0 + 0.35 * (t_speech / dur_sec) ** 1.2)
    elif word == "phải":
        mid = int(n_speech * 0.45)
        f0_track = np.zeros(n_speech, dtype=np.float32)
        f0_track[:mid] = f0_base * (1.0 - 0.20 * (t_speech[:mid] / (dur_sec * 0.45)))
        f0_track[mid:] = f0_base * (0.80 + 0.30 * ((t_speech[mid:] - dur_sec * 0.45) / (dur_sec * 0.55)))
    elif word == "lên":
        f0_track = np.full(n_speech, f0_base, dtype=np.float32)
    elif word == "xuống":
        f0_track = f0_base * (1.0 + 0.30 * (t_speech / dur_sec))
    else:
        f0_track = f0_base * (1.0 + 0.15 * np.sin(2.0 * np.pi * t_speech / dur_sec))

    # 2. Glottal source
    glottal = generate_rosenberg_glottal_source(
        n_speech, f0_track, prof.open_quotient, prof.spectral_tilt_db, prof.jitter_pct, rng, sr=sr
    )

    # 3. Formant trajectories
    b1_val, b2_val, b3_val = 80.0, 110.0, 150.0
    if word == "trái":
        f1_track = np.linspace(720.0 * vtl, 350.0 * vtl, n_speech)
        f2_track = np.linspace(1300.0 * vtl, 2150.0 * vtl, n_speech)
        f3_track = np.full(n_speech, 2700.0 * vtl)
    elif word == "phải":
        f1_track = np.linspace(780.0 * vtl, 360.0 * vtl, n_speech)
        f2_track = np.linspace(1350.0 * vtl, 2100.0 * vtl, n_speech)
        f3_track = np.full(n_speech, 2650.0 * vtl)
    elif word == "lên":
        f1_track = np.linspace(420.0 * vtl, 480.0 * vtl, n_speech)
        f2_track = np.linspace(1850.0 * vtl, 1750.0 * vtl, n_speech)
        f3_track = np.full(n_speech, 2550.0 * vtl)
    elif word == "xuống":
        f1_track = np.linspace(400.0 * vtl, 520.0 * vtl, n_speech)
        f2_track = np.linspace(850.0 * vtl, 950.0 * vtl, n_speech)
        f3_track = np.full(n_speech, 2450.0 * vtl)
    else:
        f1_base, f2_base = 600.0 * vtl, 1400.0 * vtl
        f1_track = np.linspace(f1_base * 0.9, f1_base * 1.05, n_speech)
        f2_track = np.linspace(f2_base * 0.95, f2_base * 1.05, n_speech)
        f3_track = np.full(n_speech, 2550.0 * vtl)

    vocal_resonated = apply_formant_resonators(
        glottal, f1_track, f2_track, f3_track, b1_val, b2_val, b3_val, sr=sr
    )

    # 4. Consonant onset synthesis
    output_speech = np.copy(vocal_resonated)
    if word == "trái":
        closure_sec = float(rng.uniform(0.016, 0.032))
        closure_samples = int(closure_sec * sr)
        leak_factor = float(rng.uniform(0.09, 0.15))
        output_speech[:closure_samples] *= leak_factor
        t_close = np.linspace(0, closure_sec, closure_samples, endpoint=False)
        output_speech[:closure_samples] += (0.04 * np.sin(2.0 * np.pi * f0_base * t_close)).astype(np.float32)

        burst_sec = float(rng.uniform(0.010, 0.018))
        burst_samples = int(burst_sec * sr)
        b_bp, a_bp = signal.iirpeak(min(float(rng.uniform(3400.0, 4400.0)) * vtl, (sr / 2.0) - 200.0), Q=5.0, fs=sr)
        burst_noise = signal.lfilter(b_bp, a_bp, rng.normal(0.0, 1.0, burst_samples).astype(np.float32))
        burst_signal = (burst_noise * np.exp(-np.linspace(0.0, 4.0, burst_samples)) * float(rng.uniform(0.35, 0.50))).astype(np.float32)
        burst_start = closure_samples
        burst_end = min(burst_start + burst_samples, n_speech)
        output_speech[burst_start:burst_end] += burst_signal[:burst_end - burst_start]

    elif word == "phải":
        fric_sec = float(rng.uniform(0.045, 0.075))
        fric_samples = int(fric_sec * sr)
        f_low = float(rng.uniform(1600.0, 2000.0))
        f_high = float(rng.uniform(5800.0, 6800.0))
        b_bp, a_bp = signal.butter(3, [f_low / (sr / 2.0), f_high / (sr / 2.0)], btype='band')
        fric_filt = signal.lfilter(b_bp, a_bp, rng.normal(0.0, 1.0, fric_samples).astype(np.float32))
        rise_samples = int(fric_samples * 0.45)
        fric_env = np.ones(fric_samples, dtype=np.float32)
        fric_env[:rise_samples] = np.linspace(0.1, 1.0, rise_samples)
        fric_env[rise_samples:] = np.linspace(1.0, 0.2, fric_samples - rise_samples)
        fric_signal = (fric_filt * fric_env * float(rng.uniform(0.25, 0.35))).astype(np.float32)
        fric_end = min(fric_samples, n_speech)
        output_speech[:fric_end] += fric_signal[:fric_end]

    elif word == "xuống":
        fric_sec = float(rng.uniform(0.050, 0.070))
        fric_samples = int(fric_sec * sr)
        b_hp, a_hp = signal.butter(4, 3800.0 / (sr / 2.0), btype='high')
        fric_filt = signal.lfilter(b_hp, a_hp, rng.normal(0.0, 1.0, fric_samples).astype(np.float32))
        fric_signal = (fric_filt * np.linspace(0.8, 0.2, fric_samples) * float(rng.uniform(0.28, 0.40))).astype(np.float32)
        fric_end = min(fric_samples, n_speech)
        output_speech[:fric_end] += fric_signal[:fric_end]

    elif word == "lên":
        pass
    else:
        if word.startswith(("ch", "tr")):
            close_s = int(0.020 * sr)
            output_speech[:close_s] *= 0.12
        elif word.startswith(("s", "x", "ph", "h")):
            fric_s = int(0.050 * sr)
            b_hp, a_hp = signal.butter(3, 2500.0 / (sr / 2.0), btype='high')
            f_noise = signal.lfilter(b_hp, a_hp, rng.normal(0.0, 1.0, fric_s)) * 0.22
            output_speech[:min(fric_s, n_speech)] += f_noise[:min(fric_s, n_speech)]

    # 5. Envelope
    attack_samples = int(float(rng.uniform(0.025, 0.050)) * sr)
    decay_samples = int(float(rng.uniform(0.080, 0.140)) * sr)
    env = np.ones(n_speech, dtype=np.float32)
    if word != "trái":
        env[:attack_samples] = np.linspace(0.05, 1.0, attack_samples)
    else:
        post_close = closure_samples + burst_samples
        if post_close < n_speech:
            env[:closure_samples] = np.linspace(0.4, 0.6, closure_samples)
            env[closure_samples:post_close] = np.linspace(0.7, 1.0, post_close - closure_samples)
    env[-decay_samples:] = np.linspace(1.0, 0.02, decay_samples)
    raw_speech = output_speech * env

    # 6. Standardized Speech Gain
    raw_speech_rms = float(np.sqrt(np.mean(raw_speech ** 2))) + 1e-9
    target_speech_linear = float(10.0 ** (config.target_speech_rms_dbfs / 20.0))
    gain_factor = float(target_speech_linear / raw_speech_rms)
    scaled_speech = raw_speech * gain_factor

    # Soft headroom saturation limiter with 0.98 threshold (avoids premature fricative compression)
    peak_val = np.max(np.abs(scaled_speech))
    if peak_val > 0.98:
        scaled_speech = np.tanh(scaled_speech / 0.98) * 0.98

    # Place in 1.0s window
    clip = np.zeros(TOTAL_SAMPLES, dtype=np.float32)
    onset_idx = int(config.onset_offset_sec * sr)
    onset_idx = np.clip(onset_idx, 0, TOTAL_SAMPLES - n_speech)
    clip[onset_idx:onset_idx + n_speech] = scaled_speech

    # 7. Independent Ambient Background Noise
    bg_rng = np.random.RandomState(config.noise_seed)
    bg_raw = np.zeros(TOTAL_SAMPLES, dtype=np.float32)
    if config.noise_type == "room_tone":
        raw_n = bg_rng.normal(0.0, 1.0, TOTAL_SAMPLES).astype(np.float32)
        b_bg, a_bg = signal.butter(3, float(bg_rng.uniform(700.0, 1200.0)) / (sr / 2.0), btype='low')
        bg_raw = signal.lfilter(b_bg, a_bg, raw_n).astype(np.float32)
    elif config.noise_type == "dither":
        bg_raw = bg_rng.normal(0.0, 1.0, TOTAL_SAMPLES).astype(np.float32)
    elif config.noise_type == "hum":
        t_clip = np.linspace(0, 1.0, TOTAL_SAMPLES, endpoint=False)
        f_hum = 50.0 if bg_rng.rand() > 0.5 else 60.0
        bg_raw = (np.sin(2.0 * np.pi * f_hum * t_clip) +
                  0.4 * np.sin(2.0 * np.pi * 3.0 * f_hum * t_clip)).astype(np.float32)
    else:
        bg_raw = bg_rng.normal(0.0, 1.0, TOTAL_SAMPLES).astype(np.float32)

    target_bg_linear = float(10.0 ** (config.background_rms_dbfs / 20.0))
    raw_bg_rms = float(np.sqrt(np.mean(bg_raw ** 2))) + 1e-9
    scaled_bg = bg_raw * (target_bg_linear / raw_bg_rms)

    mixed = clip + scaled_bg
    final_peak = np.max(np.abs(mixed))
    if final_peak > 0.999:
        mixed = np.tanh(mixed / 0.999) * 0.999

    speech_rms = float(np.sqrt(np.mean(scaled_speech ** 2)))
    bg_rms = float(np.sqrt(np.mean(scaled_bg ** 2)))
    effective_snr_db = float(20.0 * np.log10((speech_rms + 1e-9) / (bg_rms + 1e-9)))

    return mixed.astype(np.float32), gain_factor, effective_snr_db


def generate_v5_2_silence_clip(config: V5_2NoiseConfig, sr: int = SAMPLE_RATE) -> np.ndarray:
    rng = np.random.RandomState(config.seed)
    sub = config.acoustic_subclass
    if sub == "ambient_digital_zero":
        dither = rng.uniform(-1.0 / 32768.0, 1.0 / 32768.0, TOTAL_SAMPLES).astype(np.float32)
        return dither
    elif sub == "ambient_room_tone":
        n = rng.normal(0.0, 1.0, TOTAL_SAMPLES).astype(np.float32)
        b, a = signal.butter(3, float(rng.uniform(600.0, 1200.0)) / (sr / 2.0), btype='low')
        filtered = signal.lfilter(b, a, n).astype(np.float32)
        rms = float(np.sqrt(np.mean(filtered ** 2))) + 1e-9
        target = float(10.0 ** (float(rng.uniform(-45.0, -35.0)) / 20.0))
        return (filtered * (target / rms)).astype(np.float32)
    elif sub == "ambient_low_freq_hum":
        t = np.linspace(0, 1.0, TOTAL_SAMPLES, endpoint=False)
        f_hum = 50.0 if rng.rand() > 0.5 else 60.0
        wave_sig = np.sin(2.0 * np.pi * f_hum * t) + 0.3 * np.sin(2.0 * np.pi * 3.0 * f_hum * t)
        noise = rng.normal(0.0, 0.2, TOTAL_SAMPLES)
        sig = (wave_sig + noise).astype(np.float32)
        rms = float(np.sqrt(np.mean(sig ** 2))) + 1e-9
        target = float(10.0 ** (float(rng.uniform(-42.0, -32.0)) / 20.0))
        return (sig * (target / rms)).astype(np.float32)
    else:
        n = rng.normal(0.0, 1.0, TOTAL_SAMPLES).astype(np.float32)
        rms = float(np.sqrt(np.mean(n ** 2))) + 1e-9
        target = float(10.0 ** (float(rng.uniform(-40.0, -35.0)) / 20.0))
        return (n * (target / rms)).astype(np.float32)


def reproduce_from_manifest_row(row: Dict[str, Any], sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Reconstructs exact bit-exact audio clip from manifest row dictionary.
    """
    c_label = str(row['class_label'])
    subclass = str(row['acoustic_subclass'])
    seed = int(row['seed'])
    dur_sec = float(row['duration_sec'])
    onset_sec = float(row['onset_offset_sec'])
    noise_seed = int(row['noise_seed'])
    noise_type = str(row['noise_type'])
    target_speech_rms_dbfs = float(row['target_speech_rms_dbfs'])
    background_rms_dbfs = float(row['background_rms_dbfs'])

    if c_label == "SILENCE":
        cfg = V5_2NoiseConfig(
            sample_id=str(row['recording_id']),
            class_label=c_label,
            acoustic_subclass=subclass,
            seed=seed,
            duration_sec=dur_sec,
            sr=sr
        )
        return generate_v5_2_silence_clip(cfg, sr=sr)
    else:
        prof = V5_2VoiceProfile(
            voice_id=str(row['simulated_speaker_id']),
            vtl_scale=float(row['vtl_scale']),
            f0_base=float(row['f0_base']),
            open_quotient=float(row['open_quotient']),
            spectral_tilt_db=float(row['spectral_tilt_db']),
            jitter_pct=float(row['jitter_pct'])
        )
        cfg = V5_2AcousticConfig(
            sample_id=str(row['recording_id']),
            class_label=c_label,
            acoustic_subclass=subclass,
            intended_text=str(row['transcript']),
            seed=seed,
            voice_profile=prof,
            duration_sec=dur_sec,
            onset_offset_sec=onset_sec,
            target_speech_rms_dbfs=target_speech_rms_dbfs,
            noise_type=noise_type,
            noise_seed=noise_seed,
            background_rms_dbfs=background_rms_dbfs
        )
        audio, _, _ = generate_v5_2_speech_clip(cfg, sr=sr)
        return audio
