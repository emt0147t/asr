"""
Live Microphone Pipeline and Engine for Vietnamese Keyword Spotting (KWS).
Release Candidate v11 / Phase 13.4 Frozen Model.

Mandatory Governance:
- Canonical Label: EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED
- NOT_VALIDATED_ON_NATURAL_SPEECH
- NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION
- NOT_VALIDATED_PENDING_DATA

Modes Supported:
1. "segmented" (DEFAULT & RECOMMENDED):
   - Speech onset detection + pre-roll buffer (150ms).
   - Accumulates complete command utterance until trailing silence (250-350ms).
   - Formats finalized segment into exact 1.0s window (right-padded with zeros).
   - Emits exactly 1 event per utterance.
   - Instantly re-arms with zero global cooldown latch.
   - Enables reliable multi-command sequences: e.g. TRAI -> XUONG -> PHAI and repeated TRAI -> TRAI.
2. "streaming" (LEGACY DEMO):
   - 160 ms sliding window over 16,000-sample circular ring buffer.
   - Debounce cooldown of 0.8s.
"""

import sys
import time
import queue
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field

import numpy as np
import sounddevice as sd

# Resampling utility
try:
    import soxr
    HAS_SOXR = True
except ImportError:
    HAS_SOXR = False
    from scipy import signal

# Add bundle directory to sys.path to reuse authoritative inferencer
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
BUNDLE_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4" / "bundle"
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from infer import VietnameseKWSInferencer, CLASSES, TARGET_CLASSES, DEFAULT_THRESHOLD

# Import existing project VAD from ablation study
VAD_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_vad_ablation_v1"
if str(VAD_DIR) not in sys.path:
    sys.path.insert(0, str(VAD_DIR))

try:
    from vad_engine import SpeechBandFormantVAD, ShortTimeEnergyGate
    HAS_PROJECT_VAD = True
except ImportError:
    HAS_PROJECT_VAD = False

# Import Phase 15 SpeechBandRatioVAD
PHASE15_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "phase15_vad"
if str(PHASE15_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE15_DIR))

try:
    from vad_segmenters import SpeechBandRatioVAD
    HAS_SPEECHBAND_VAD = True
except ImportError:
    HAS_SPEECHBAND_VAD = False


@dataclass
class DetectionEvent:
    """Records an accepted keyword activation event."""
    timestamp: float
    time_str: str
    command: str
    confidence: float
    probabilities: Dict[str, float]
    latency_ms: float
    vad_active: bool
    db_level: float
    segment_duration_ms: float = 0.0
    profile_name: str = "legacy_v11"
    model_artifact_hash: str = ""
    vad_config_hash: str = ""
    vad_algorithm: str = "UtteranceSegmenter"
    segment_start_sec: float = 0.0
    segment_end_sec: float = 0.0


class AudioRingBuffer:
    """Thread-safe circular audio buffer maintaining exactly 16,000 samples (1.0 s at 16 kHz)."""
    def __init__(self, capacity: int = 16000):
        self.capacity = capacity
        self.buffer = np.zeros(capacity, dtype=np.float32)
        self.lock = threading.Lock()
        self.samples_appended = 0

    def append(self, samples: np.ndarray):
        """Appends new 16 kHz mono samples into the ring buffer."""
        n = len(samples)
        if n == 0:
            return
        with self.lock:
            if n >= self.capacity:
                self.buffer[:] = samples[-self.capacity:]
            else:
                self.buffer[:-n] = self.buffer[n:]
                self.buffer[-n:] = samples
            self.samples_appended += n

    def get_window(self) -> np.ndarray:
        """Returns a snapshot copy of the current 1.0s (16,000 samples) window."""
        with self.lock:
            return self.buffer.copy()

    def reset(self):
        with self.lock:
            self.buffer.fill(0.0)
            self.samples_appended = 0


class UtteranceSegmenter:
    """
    Speech Segmenter for Segmented Command Mode.
    Detects speech onset, preserves pre-roll audio, accumulates speech,
    and finalizes the segment after a configurable silence interval (250-350ms).
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        silence_interval_ms: int = 300,
        preroll_ms: int = 150,
        energy_threshold_db: float = -48.0,
        min_speech_duration_ms: int = 120,
        max_segment_duration_ms: int = 1200
    ):
        self.sample_rate = sample_rate
        self.silence_interval_ms = silence_interval_ms
        self.silence_threshold_samples = int((silence_interval_ms / 1000.0) * sample_rate)
        self.preroll_samples = int((preroll_ms / 1000.0) * sample_rate)
        self.energy_threshold_db = energy_threshold_db
        self.min_speech_samples = int((min_speech_duration_ms / 1000.0) * sample_rate)
        self.max_segment_samples = int((max_segment_duration_ms / 1000.0) * sample_rate)

        # Pre-roll FIFO buffer
        self.preroll_buf = np.zeros(self.preroll_samples, dtype=np.float32)

        # State tracking
        self.state = "WAITING"  # "WAITING" or "ACCUMULATING"
        self.accumulated_chunks: List[np.ndarray] = []
        self.consecutive_silence_samples = 0
        self.active_speech_samples = 0
        self.current_db = -120.0
        self.noise_floor_db = energy_threshold_db
        self.is_speech_active = False

        # VAD engine
        if HAS_PROJECT_VAD:
            self.vad = SpeechBandFormantVAD(
                sample_rate=sample_rate,
                speech_band_hz=(300.0, 3400.0),
                min_speech_band_ratio=0.30,
                min_energy_db=energy_threshold_db,
                hangover_frames=4
            )
        else:
            self.vad = None

    def reset(self):
        self.state = "WAITING"
        self.accumulated_chunks = []
        self.consecutive_silence_samples = 0
        self.active_speech_samples = 0
        self.preroll_buf.fill(0.0)
        self.is_speech_active = False

    def process_chunk(self, chunk: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
        """
        Processes an incoming 16 kHz audio chunk.
        Returns:
            (finalized_1s_audio, segment_duration_ms) if a valid utterance segment concluded,
            otherwise None.
        """
        n_samples = len(chunk)
        if n_samples == 0:
            return None

        # Compute chunk acoustic energy
        rms = float(np.sqrt(np.mean(chunk ** 2))) + 1e-7
        db = 20.0 * np.log10(rms)
        self.current_db = db

        # Dynamic noise floor adaptation and effective speech threshold
        effective_thresh = max(self.energy_threshold_db, min(self.noise_floor_db + 5.0, -32.0))
        is_speech_chunk = (db >= effective_thresh)

        if self.state == "WAITING" and not is_speech_chunk:
            self.noise_floor_db = 0.95 * self.noise_floor_db + 0.05 * db

        self.is_speech_active = is_speech_chunk

        finalized_segment = None

        if self.state == "WAITING":
            # Maintain circular pre-roll buffer
            if n_samples >= self.preroll_samples:
                self.preroll_buf[:] = chunk[-self.preroll_samples:]
            else:
                self.preroll_buf[:-n_samples] = self.preroll_buf[n_samples:]
                self.preroll_buf[-n_samples:] = chunk

            # Speech onset detection
            if is_speech_chunk:
                self.state = "ACCUMULATING"
                self.accumulated_chunks = [self.preroll_buf.copy(), chunk.copy()]
                self.consecutive_silence_samples = 0
                self.active_speech_samples = n_samples

        elif self.state == "ACCUMULATING":
            self.accumulated_chunks.append(chunk.copy())

            if is_speech_chunk:
                self.consecutive_silence_samples = 0
                self.active_speech_samples += n_samples
            else:
                self.consecutive_silence_samples += n_samples

            # Calculate total accumulated duration
            total_samples = sum(len(c) for c in self.accumulated_chunks)

            # Check if utterance ended due to trailing silence or max duration cap
            silence_reached = (self.consecutive_silence_samples >= self.silence_threshold_samples)
            max_reached = (total_samples >= self.max_segment_samples)

            if silence_reached or max_reached:
                full_segment = np.concatenate(self.accumulated_chunks)
                actual_duration_ms = (len(full_segment) / self.sample_rate) * 1000.0

                # Validate minimum speech duration (filters out key taps, clicks, mic bumps)
                if self.active_speech_samples >= self.min_speech_samples:
                    # Format to exact 1.0s window using authoritative padding contract
                    if len(full_segment) < 16000:
                        audio_1s = np.pad(full_segment, (0, 16000 - len(full_segment)), mode="constant", constant_values=0.0)
                    else:
                        audio_1s = full_segment[:16000]

                    finalized_segment = (audio_1s.astype(np.float32), actual_duration_ms)

                # Reset state machine for next command
                self.state = "WAITING"
                self.accumulated_chunks = []
                self.consecutive_silence_samples = 0
                self.active_speech_samples = 0
                self.preroll_buf.fill(0.0)

        return finalized_segment


class SpeechBandRatioSegmenterAdapter:
    """
    Adapter wrapping SpeechBandRatioVAD (Phase 15/16) to provide identical
    chunk processing interface to UtteranceSegmenter for Profile B.
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        speech_band_hz: Tuple[float, float] = (300.0, 3400.0),
        min_speech_band_ratio: float = 0.32,
        energy_threshold_db: float = -46.0,
        hangover_ms: int = 140,
        preroll_ms: int = 150,
        min_speech_duration_ms: int = 100,
        max_segment_duration_ms: int = 1200
    ):
        if not HAS_SPEECHBAND_VAD:
            raise ImportError("SpeechBandRatioVAD is not available. Ensure phase15_vad is in sys.path.")
        self.sample_rate = sample_rate
        self.vad = SpeechBandRatioVAD(
            sample_rate=sample_rate,
            speech_band_hz=speech_band_hz,
            min_speech_band_ratio=min_speech_band_ratio,
            energy_threshold_db=energy_threshold_db,
            hangover_ms=hangover_ms,
            preroll_ms=preroll_ms,
            min_speech_duration_ms=min_speech_duration_ms,
            max_segment_duration_ms=max_segment_duration_ms
        )
        self.state = "WAITING"
        self.is_speech_active = False
        self.total_samples_processed = 0
        self.last_segment_info: Optional[Dict[str, Any]] = None

    def reset(self):
        self.vad.reset()
        self.state = "WAITING"
        self.is_speech_active = False
        self.total_samples_processed = 0
        self.last_segment_info = None

    def process_chunk(self, chunk: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
        chunk_time_sec = self.total_samples_processed / float(self.sample_rate)
        self.total_samples_processed += len(chunk)
        res = self.vad.process_chunk(chunk, chunk_time_sec)
        self.state = self.vad.state
        self.is_speech_active = (self.vad.state == "ACCUMULATING")
        if res is not None:
            self.last_segment_info = res
            audio_1s = res["segment_audio_1s"]
            dur_ms = res["duration_sec"] * 1000.0
            return audio_1s.astype(np.float32), dur_ms
        return None


class AudioStreamManager:
    """
    Manages non-blocking audio capture from microphone.
    Handles device discovery, native sampling rate querying, and resampling to 16,000 Hz.
    """
    def __init__(
        self,
        device_id: Optional[int] = None,
        target_sr: int = 16000,
        chunk_ms: int = 30,
        on_samples_callback: Optional[callable] = None
    ):
        self.target_sr = target_sr
        self.chunk_ms = chunk_ms
        self.on_samples_callback = on_samples_callback
        self.device_id = device_id

        self.native_sr = target_sr
        self.resample_required = False
        self.stream: Optional[sd.InputStream] = None
        self.is_running = False
        self.last_error = ""
        self.current_rms_db = -120.0
        self.peak_amp = 0.0

        self.raw_queue: queue.Queue = queue.Queue(maxsize=100)
        self.worker_thread: Optional[threading.Thread] = None

        # Acoustic test injection buffer for live stream mixing
        self.injection_buffer = np.zeros(0, dtype=np.float32)
        self.injection_lock = threading.Lock()

        self._configure_device()

    @staticmethod
    def list_input_devices() -> List[Dict[str, Any]]:
        """Returns all audio input devices visible via PortAudio."""
        devices = []
        hostapis = sd.query_hostapis()
        for i, dev in enumerate(sd.query_devices()):
            if dev['max_input_channels'] > 0:
                api_name = hostapis[dev['hostapi']]['name'] if dev['hostapi'] < len(hostapis) else "Unknown"
                devices.append({
                    "id": i,
                    "name": dev['name'],
                    "hostapi": api_name,
                    "channels": dev['max_input_channels'],
                    "default_sr": int(dev['default_samplerate'])
                })
        return devices

    def _configure_device(self):
        """Probes device configuration and determines native sample rate."""
        default_in = sd.default.device[0] if sd.default.device[0] is not None else 0
        if self.device_id is None:
            self.device_id = default_in

        try:
            info = sd.query_devices(self.device_id)
            self.device_name = info['name']
            self.device_api = sd.query_hostapis(info['hostapi'])['name']
            device_native_sr = int(info['default_samplerate'])
        except Exception as e:
            self.last_error = f"Device ID {self.device_id} query failed: {e}. Falling back to default."
            self.device_id = default_in
            info = sd.query_devices(self.device_id)
            self.device_name = info['name']
            self.device_api = sd.query_hostapis(info['hostapi'])['name']
            device_native_sr = int(info['default_samplerate'])

        # Check if 16000 Hz is supported natively
        try:
            sd.check_input_settings(device=self.device_id, channels=1, samplerate=self.target_sr, dtype='float32')
            self.native_sr = self.target_sr
            self.resample_required = False
        except Exception:
            # Must capture at device native SR and resample
            self.native_sr = device_native_sr
            self.resample_required = True

    def _audio_callback(self, indata, frames, time_info, status):
        """PortAudio real-time callback; pushes raw chunk to queue."""
        if status:
            self.last_error = f"Audio overflow/underflow: {status}"
        if self.is_running:
            try:
                self.raw_queue.put_nowait(indata.copy())
            except queue.Full:
                self.last_error = "Audio queue full; dropped frame"

    def _processing_worker(self):
        """Worker thread converting chunks to 16 kHz mono float32."""
        while self.is_running:
            try:
                chunk = self.raw_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Downmix to mono if stereo/multichannel
            if chunk.ndim > 1:
                mono = np.mean(chunk, axis=-1).astype(np.float32)
            else:
                mono = chunk.flatten().astype(np.float32)

            # Compute live signal metrics
            peak = float(np.max(np.abs(mono))) if len(mono) > 0 else 0.0
            rms = float(np.sqrt(np.mean(mono ** 2))) + 1e-7
            db = 20.0 * np.log10(rms)
            self.peak_amp = peak
            self.current_rms_db = db

            # Resample to 16 kHz if needed
            if self.resample_required and self.native_sr != self.target_sr:
                if HAS_SOXR:
                    mono_16k = soxr.resample(mono, self.native_sr, self.target_sr).astype(np.float32)
                else:
                    num_target = int(round(len(mono) * float(self.target_sr) / self.native_sr))
                    mono_16k = signal.resample(mono, num_target).astype(np.float32)
            else:
                mono_16k = mono

            # Mix injected acoustic test samples into 16 kHz stream if present
            with self.injection_lock:
                if len(self.injection_buffer) > 0:
                    n_inj = min(len(mono_16k), len(self.injection_buffer))
                    mono_16k[:n_inj] += self.injection_buffer[:n_inj]
                    self.injection_buffer = self.injection_buffer[n_inj:]
                    # Recalculate metrics reflecting injected signal
                    peak = float(np.max(np.abs(mono_16k))) if len(mono_16k) > 0 else 0.0
                    rms = float(np.sqrt(np.mean(mono_16k ** 2))) + 1e-7
                    self.peak_amp = peak
                    self.current_rms_db = 20.0 * np.log10(rms)

            if self.on_samples_callback:
                self.on_samples_callback(mono_16k)

    def inject_samples(self, samples: np.ndarray):
        """Acoustically mixes audio samples into the live microphone stream chronologically."""
        with self.injection_lock:
            if len(self.injection_buffer) == 0:
                self.injection_buffer = samples.astype(np.float32).copy()
            else:
                self.injection_buffer = np.concatenate([self.injection_buffer, samples.astype(np.float32)])

    def is_injecting(self) -> bool:
        """Returns True if injected samples are still playing into the stream."""
        with self.injection_lock:
            return len(self.injection_buffer) > 0

    def clear_injection(self):
        """Clears any pending injected audio samples."""
        with self.injection_lock:
            self.injection_buffer = np.zeros(0, dtype=np.float32)

    def start(self):
        """Starts input audio capture stream and worker thread."""
        if self.is_running:
            return

        self._configure_device()
        self.is_running = True
        self.raw_queue = queue.Queue(maxsize=100)

        # Start worker thread
        self.worker_thread = threading.Thread(target=self._processing_worker, daemon=True)
        self.worker_thread.start()

        # Blocksize calculation
        block_frames = int(self.native_sr * (self.chunk_ms / 1000.0))

        self.stream = sd.InputStream(
            device=self.device_id,
            channels=1,
            samplerate=self.native_sr,
            blocksize=block_frames,
            dtype='float32',
            callback=self._audio_callback
        )
        self.stream.start()

    def stop(self):
        """Cleanly stops and releases microphone stream."""
        self.is_running = False
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                self.last_error = f"Stream stop error: {e}"
            self.stream = None
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
            self.worker_thread = None


class LiveKWSPipeline:
    """
    Main Live KWS Pipeline.
    Supports both:
    1. 'segmented' (Default): Utterance accumulation & segmentation with instant re-arm.
    2. 'streaming' (Legacy): 160 ms sliding window over 16,000-sample circular ring buffer.
    """
    def __init__(
        self,
        mode: str = "segmented",
        profile: str = "legacy_v11",
        backend: str = "onnx",
        model_path: Optional[Path] = None,
        vad_config_path: Optional[Path] = None,
        model_hash: str = "",
        vad_config_hash: str = "",
        theta_cmd: float = DEFAULT_THRESHOLD,
        device_id: Optional[int] = None,
        vad_mode: str = "formant",
        silence_interval_ms: int = 300,
        preroll_ms: int = 150,
        energy_threshold_db: float = -48.0,
        update_interval_ms: int = 160,
        consecutive_frames: int = 2,
        cooldown_sec: float = 0.80,
        event_callback: Optional[callable] = None
    ):
        self.mode = mode.lower()
        self.profile_name = profile.lower()
        self.backend = backend.lower()
        self.theta_cmd = theta_cmd
        self.vad_mode = vad_mode
        self.silence_interval_ms = silence_interval_ms
        self.preroll_ms = preroll_ms
        self.energy_threshold_db = energy_threshold_db
        self.update_interval_ms = update_interval_ms
        self.consecutive_frames = consecutive_frames
        self.cooldown_sec = cooldown_sec
        self.event_callback = event_callback
        self.model_hash = model_hash
        self.vad_config_hash = vad_config_hash

        # Initialize Ring Buffer (used for streaming mode and display)
        self.ring_buffer = AudioRingBuffer(capacity=16000)

        # Initialize Segmenter based on profile
        if self.profile_name == "experimental_e1_speechband":
            import json
            cfg_p = vad_config_path or (PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "configs" / "selected_phase15_vad_config.json")
            with open(cfg_p, "r", encoding="utf-8") as f:
                vad_meta = json.load(f)
            params = dict(vad_meta.get("parameters", {}))
            params.pop("algorithm", None)
            if "speech_band_hz" in params and isinstance(params["speech_band_hz"], list):
                params["speech_band_hz"] = tuple(params["speech_band_hz"])
            self.segmenter = SpeechBandRatioSegmenterAdapter(**params)
            self.vad_algorithm = "SpeechBandRatioVAD"
        else:
            self.segmenter = UtteranceSegmenter(
                sample_rate=16000,
                silence_interval_ms=silence_interval_ms,
                preroll_ms=preroll_ms,
                energy_threshold_db=energy_threshold_db
            )
            self.vad_algorithm = "UtteranceSegmenter"

        # Initialize Authoritative Inferencer
        self.inferencer = VietnameseKWSInferencer(
            backend=self.backend,
            model_path=model_path,
            theta_cmd=self.theta_cmd
        )

        # For experimental Profile B, bind the canonical Slaney frontend to eliminate
        # the filterbank divergence and guarantee bitwise alignment with PyTorch reference
        if self.profile_name == "experimental_e1_speechband":
            from experiments.vieneu_kws_research_v12.scripts.audit_frontend_parity_phase20 import (
                CanonicalLogMelFrontendNumPy
            )
            self.inferencer.frontend = CanonicalLogMelFrontendNumPy()

        # Streaming mode VAD
        if HAS_PROJECT_VAD and vad_mode == "formant":
            self.vad = SpeechBandFormantVAD(
                sample_rate=16000, speech_band_hz=(300.0, 3400.0),
                min_speech_band_ratio=0.32, min_energy_db=energy_threshold_db,
                hangover_frames=8
            )
        elif HAS_PROJECT_VAD and vad_mode == "energy":
            self.vad = ShortTimeEnergyGate(energy_threshold_db=energy_threshold_db, min_speech_frames=5)
        else:
            self.vad = None

        # Audio Stream Manager
        self.stream_mgr = AudioStreamManager(
            device_id=device_id,
            target_sr=16000,
            chunk_ms=30,
            on_samples_callback=self._on_new_audio_chunk
        )

        # State tracking for streaming mode
        self.consecutive_candidate = None
        self.consecutive_count = 0
        self.last_trigger_time = -999.0
        self.last_trigger_cmd = None
        self.is_armed = True
        self.is_running = False

        # Live telemetry state for UI
        self.last_prediction: Dict[str, Any] = {
            "predicted_class": "REJECT",
            "confidence": 0.0,
            "probabilities": {c: 0.0 for c in CLASSES},
            "latency_ms": 0.0,
            "vad_active": False,
            "is_speech": False
        }
        self.recent_events: List[DetectionEvent] = []
        self.inference_thread: Optional[threading.Thread] = None

    def _on_new_audio_chunk(self, samples_16k: np.ndarray):
        """Called by audio streamer when new 16 kHz mono samples arrive."""
        # Always maintain ring buffer for visual VU & display
        self.ring_buffer.append(samples_16k)

        # In segmented mode, process chunk through UtteranceSegmenter
        if self.mode == "segmented" and self.is_running:
            result = self.segmenter.process_chunk(samples_16k)
            if result is not None:
                audio_1s, seg_dur_ms = result
                self._evaluate_segmented_utterance(audio_1s, seg_dur_ms)

    def _evaluate_segmented_utterance(self, audio_1s: np.ndarray, seg_dur_ms: float):
        """Runs inference on a finalized speech segment and emits event if accepted."""
        current_time = time.time()
        pred_class, conf, probs, dt_ms = self._run_in_memory_inference(audio_1s)

        # Update telemetry
        self.last_prediction = {
            "predicted_class": pred_class,
            "confidence": conf,
            "probabilities": probs,
            "latency_ms": dt_ms,
            "vad_active": True,
            "is_speech": True
        }

        # Command emission: if accepted command class (conf >= 0.20)
        if pred_class in TARGET_CLASSES:
            seg_s = 0.0
            seg_e = 0.0
            if hasattr(self.segmenter, "last_segment_info") and self.segmenter.last_segment_info is not None:
                seg_s = float(self.segmenter.last_segment_info.get("start_sec", 0.0))
                seg_e = float(self.segmenter.last_segment_info.get("end_sec", 0.0))

            event = DetectionEvent(
                timestamp=current_time,
                time_str=time.strftime("%H:%M:%S", time.localtime(current_time)),
                command=pred_class,
                confidence=conf,
                probabilities=probs,
                latency_ms=dt_ms,
                vad_active=True,
                db_level=self.stream_mgr.current_rms_db,
                segment_duration_ms=round(seg_dur_ms, 1),
                profile_name=self.profile_name,
                model_artifact_hash=self.model_hash[:8] if self.model_hash else "",
                vad_config_hash=self.vad_config_hash[:8] if self.vad_config_hash else "",
                vad_algorithm=self.vad_algorithm,
                segment_start_sec=seg_s,
                segment_end_sec=seg_e
            )
            self.recent_events.append(event)
            if len(self.recent_events) > 15:
                self.recent_events.pop(0)

            self.last_trigger_time = current_time
            self.last_trigger_cmd = pred_class

            if self.event_callback:
                self.event_callback(event)

    def _run_in_memory_inference(self, audio_1s: np.ndarray) -> Tuple[str, float, Dict[str, float], float]:
        """
        Executes exact model inference on 16,000 float32 samples.
        Strictly adheres to frozen contract:
        - Clamping: only if max(|x|) > 1.0
        - Log-Mel: [1, 1, 40, 101]
        - Threshold: theta_cmd = 0.20
        """
        t0 = time.perf_counter()

        # Amplitude clamping check (zero floor amplification)
        max_val = float(np.max(np.abs(audio_1s))) if len(audio_1s) > 0 else 0.0
        if max_val > 1.0:
            audio_1s = audio_1s / max_val

        if self.backend == "pytorch":
            spec = self.inferencer.frontend.compute(audio_1s)
            x = self.inferencer.torch.from_numpy(spec).to(self.inferencer.device)
            with self.inferencer.torch.no_grad():
                logits = self.inferencer.model(x)
                probs = self.inferencer.F.softmax(logits, dim=-1).cpu().numpy()[0]
        else:  # onnx
            spec = self.inferencer.frontend.compute(audio_1s)
            ort_out = self.inferencer.session.run(None, {"log_mel": spec})
            probs = ort_out[1][0]

        dt_ms = (time.perf_counter() - t0) * 1000.0

        # Validate output shape and finiteness
        assert len(probs) == 5, f"Expected 5 output probabilities, got {len(probs)}"
        assert np.all(np.isfinite(probs)), "Probabilities contain NaN or Inf values"

        # Canonical 5-class decision rule (Chapter 07 Preprocessing & Inference Contract):
        # c* = argmax over ALL 5 classes (TRAI, PHAI, LEN, XUONG, REJECT)
        # Accepted only if c* is a command (0..3) AND P(c*) >= theta_cmd (0.20)
        top_idx = int(np.argmax(probs))
        top_prob = float(probs[top_idx])

        if top_idx < 4 and top_prob >= self.theta_cmd:
            predicted_class = TARGET_CLASSES[top_idx]
            confidence = top_prob
        else:
            predicted_class = "REJECT"
            confidence = top_prob if top_idx == 4 else float(np.max(probs[:4]))

        probs_dict = {CLASSES[i]: round(float(probs[i]), 4) for i in range(5)}
        return predicted_class, round(confidence, 4), probs_dict, round(dt_ms, 2)

    def _streaming_inference_loop(self):
        """Worker thread executing streaming sliding-window inference every 160 ms (Legacy mode)."""
        step_samples = int(16000 * (self.update_interval_ms / 1000.0))  # 2,560 samples
        last_processed_count = 0

        while self.is_running:
            now_samples = self.ring_buffer.samples_appended
            if now_samples - last_processed_count < step_samples:
                time.sleep(0.02)
                continue

            last_processed_count = now_samples
            current_time = time.time()
            audio_window = self.ring_buffer.get_window()

            # 1. Voice Activity Detection
            is_speech = True
            if self.vad is not None:
                vad_res = self.vad.process_window(audio_window)
                is_speech = bool(vad_res[0])

            # 2. Inference
            if is_speech or self.vad_mode == "none":
                pred_class, conf, probs, dt_ms = self._run_in_memory_inference(audio_window)
            else:
                pred_class = "REJECT"
                conf = 0.0
                probs = {"TRAI": 0.01, "PHAI": 0.01, "LEN": 0.01, "XUONG": 0.01, "REJECT": 0.96}
                dt_ms = 0.1

            self.last_prediction = {
                "predicted_class": pred_class,
                "confidence": conf,
                "probabilities": probs,
                "latency_ms": dt_ms,
                "vad_active": is_speech,
                "is_speech": is_speech
            }

            # 3. Debounce & Cooldown Logic for streaming mode
            max_cmd_p = max(probs["TRAI"], probs["PHAI"], probs["LEN"], probs["XUONG"])

            if not self.is_armed:
                if max_cmd_p < 0.15 or not is_speech:
                    if current_time - self.last_trigger_time >= (self.cooldown_sec * 0.75):
                        self.is_armed = True
                        self.consecutive_count = 0
                        self.consecutive_candidate = None

            if pred_class in TARGET_CLASSES:
                if pred_class == self.consecutive_candidate:
                    self.consecutive_count += 1
                else:
                    self.consecutive_candidate = pred_class
                    self.consecutive_count = 1

                time_since_last = current_time - self.last_trigger_time
                if (
                    self.consecutive_count >= self.consecutive_frames
                    and time_since_last >= self.cooldown_sec
                    and self.is_armed
                ):
                    event = DetectionEvent(
                        timestamp=current_time,
                        time_str=time.strftime("%H:%M:%S", time.localtime(current_time)),
                        command=pred_class,
                        confidence=conf,
                        probabilities=probs,
                        latency_ms=dt_ms,
                        vad_active=is_speech,
                        db_level=self.stream_mgr.current_rms_db
                    )
                    self.recent_events.append(event)
                    if len(self.recent_events) > 15:
                        self.recent_events.pop(0)

                    self.last_trigger_time = current_time
                    self.last_trigger_cmd = pred_class
                    self.is_armed = False
                    self.consecutive_count = 0

                    if self.event_callback:
                        self.event_callback(event)
            else:
                self.consecutive_candidate = None
                self.consecutive_count = 0

    def start(self):
        """Starts audio capture and processing."""
        if self.is_running:
            return
        self.is_running = True
        self.segmenter.reset()
        self.stream_mgr.start()

        if self.mode == "streaming":
            self.inference_thread = threading.Thread(target=self._streaming_inference_loop, daemon=True)
            self.inference_thread.start()

    def stop(self):
        """Stops inference and cleans up audio capture."""
        self.is_running = False
        self.stream_mgr.stop()
        if self.inference_thread and self.inference_thread.is_alive():
            self.inference_thread.join(timeout=1.0)
            self.inference_thread = None

    def inject_samples(self, samples: np.ndarray):
        """Acoustically mixes audio samples into the active live microphone stream."""
        self.stream_mgr.inject_samples(samples)

    def is_injecting(self) -> bool:
        """Returns True if injected audio is still playing into the stream."""
        return self.stream_mgr.is_injecting()

    def clear_injection(self):
        """Clears pending injected audio samples."""
        self.stream_mgr.clear_injection()

    def get_status_dict(self) -> Dict[str, Any]:
        """Provides a complete status dictionary for UI rendering."""
        # Determine current speech activity based on mode
        if self.mode == "segmented":
            vad_active = self.segmenter.is_speech_active
            segmenter_state = self.segmenter.state
        else:
            vad_active = self.last_prediction.get("vad_active", False)
            segmenter_state = "STREAMING"

        return {
            "profile": self.profile_name,
            "model_hash": self.model_hash[:8] if self.model_hash else "",
            "vad_config_hash": self.vad_config_hash[:8] if self.vad_config_hash else "",
            "vad_algorithm": self.vad_algorithm,
            "mode": self.mode,
            "device_id": self.stream_mgr.device_id,
            "device_name": self.stream_mgr.device_name,
            "device_api": self.stream_mgr.device_api,
            "native_sr": self.stream_mgr.native_sr,
            "resample_required": self.stream_mgr.resample_required,
            "backend": self.backend,
            "theta_cmd": self.theta_cmd,
            "vad_mode": self.vad_mode,
            "silence_interval_ms": self.silence_interval_ms,
            "current_rms_db": round(self.stream_mgr.current_rms_db, 1),
            "peak_amp": round(self.stream_mgr.peak_amp, 4),
            "vad_active": vad_active,
            "segmenter_state": segmenter_state,
            "last_prediction": self.last_prediction,
            "last_trigger_cmd": self.last_trigger_cmd,
            "last_trigger_time": self.last_trigger_time,
            "is_armed": self.is_armed,
            "last_error": self.stream_mgr.last_error,
            "recent_events": self.recent_events
        }
