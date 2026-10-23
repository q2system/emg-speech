"""Continuous streaming inference: filters -> event detection -> LSTM prediction."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from .constants import N_CHANNELS
from .dsp import (
    EventDetector,
    EventSpan,
    NoiseProfile,
    StreamingPipeline,
    normalize_event,
)
from .model import CheckpointInfo, load_checkpoint, predict_probabilities
from .protocol import Frame


class RecognizerState(str, Enum):
    CALIBRATING = "calibrating"
    LISTENING = "listening"
    ACTIVE = "active"


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    probabilities: NDArray[np.float32]
    accepted: bool
    event_ms: float
    latency_ms: float
    timestamp: float


class StreamingRecognizer:
    """Turns a continuous frame stream into words.

    The recogniser is *not* thread-safe by design; run it inside one worker
    thread.  It starts in ``CALIBRATING`` and learns the idle noise floor from
    the first ``detection.calibration_ms`` of data, then listens forever.
    """

    def __init__(
        self,
        checkpoint: CheckpointInfo,
        *,
        min_confidence: float = 0.5,
        device: str | torch.device = "cpu",
    ) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        self.checkpoint = checkpoint
        self.label_map = list(checkpoint.label_map)
        self.min_confidence = float(min_confidence)
        self._device = torch.device(device)
        self._model = checkpoint.model.to(self._device).eval()
        self._dsp = checkpoint.dsp_config
        self._detection = checkpoint.detection_config
        fs = self._dsp.sample_rate

        self._pre = max(0, self._detection.samples(self._detection.pre_margin_ms, fs))
        self._post = max(0, self._detection.samples(self._detection.post_margin_ms, fs))
        self._max_event = max(
            1, self._detection.samples(self._detection.max_event_ms, fs)
        )
        off_debounce = max(
            1, self._detection.samples(self._detection.off_debounce_ms, fs)
        )
        on_debounce = max(
            1, self._detection.samples(self._detection.on_debounce_ms, fs)
        )
        self._capacity = (
            self._pre + self._max_event + self._post + off_debounce + on_debounce + 16
        )

        self._calibration_target = max(
            1, self._detection.samples(self._detection.calibration_ms, fs)
        )
        self._pipeline = StreamingPipeline(self._dsp)
        self._buffer: deque[NDArray[np.float32]] = deque(maxlen=self._capacity)
        self._detector: EventDetector | None = None
        self._profile: NoiseProfile | None = None
        self._calibration: list[float] = []
        self._state = RecognizerState.CALIBRATING
        self._index = 0
        self._level = 0.0
        self._last_filtered: NDArray[np.float32] | None = None
        self._pending: EventSpan | None = None
        self._pending_until = -1
        self._warmup()

    # ------------------------------------------------------------------ state
    @property
    def state(self) -> RecognizerState:
        return self._state

    @property
    def level(self) -> float:
        return self._level

    @property
    def last_filtered(self) -> NDArray[np.float32] | None:
        """Most recent filtered 9-channel frame (``None`` before the first one)."""

        return self._last_filtered

    @property
    def on_threshold(self) -> float | None:
        return self._detector.on_threshold if self._detector is not None else None

    @property
    def noise_profile(self) -> NoiseProfile | None:
        return self._profile

    @property
    def calibration_progress(self) -> float:
        if self._state is not RecognizerState.CALIBRATING:
            return 1.0
        return min(1.0, len(self._calibration) / self._calibration_target)

    @property
    def frames_seen(self) -> int:
        return self._index

    def _warmup(self) -> None:
        """Pay the first-inference cost now instead of during a live gesture."""

        dummy = torch.zeros(1, self._dsp.seq_len, N_CHANNELS, device=self._device)
        with torch.inference_mode():
            self._model(dummy)

    def reset(self) -> None:
        """Restart calibration (called after a serial reconnect, for example)."""

        self._pipeline.reset()
        self._buffer.clear()
        self._detector = None
        self._profile = None
        self._calibration.clear()
        self._state = RecognizerState.CALIBRATING
        self._index = 0
        self._level = 0.0
        self._last_filtered = None
        self._pending = None
        self._pending_until = -1

    # ------------------------------------------------------------------- feed
    def push(self, frame: Frame) -> Prediction | None:
        filtered, envelope = self._pipeline.process_frame(frame)
        self._last_filtered = filtered
        self._buffer.append(filtered)
        self._index += 1
        self._level = envelope

        if self._state is RecognizerState.CALIBRATING:
            self._calibration.append(envelope)
            if len(self._calibration) >= self._calibration_target:
                self._start_listening()
            return None

        detector = self._detector
        assert detector is not None

        span = detector.step(envelope)
        if span is not None:
            self._pending = span
            self._pending_until = span.end + self._post - 1

        if self._pending is not None and self._index - 1 >= self._pending_until:
            prediction = self._finalize(self._pending)
            self._pending = None
            self._state = (
                RecognizerState.ACTIVE
                if detector.is_active
                else RecognizerState.LISTENING
            )
            return prediction

        self._state = (
            RecognizerState.ACTIVE
            if detector.is_active or self._pending is not None
            else RecognizerState.LISTENING
        )
        return None

    def flush(self) -> Prediction | None:
        """Finalise an in-flight event when the stream is stopping."""

        if self._state is RecognizerState.CALIBRATING:
            return None
        span = self._pending
        if span is None and self._detector is not None and self._detector.is_active:
            start = self._detector.active_start
            if start is not None:
                span = EventSpan(start, self._index, self._level, self._index - 1)
        self._pending = None
        if span is None:
            return None
        self._state = RecognizerState.LISTENING
        return self._finalize(span)

    def _start_listening(self) -> None:
        profile = NoiseProfile.from_values(
            self._calibration, min_std=self._detection.min_noise_std
        )
        self._profile = profile
        self._calibration.clear()
        self._detector = EventDetector(
            self._detection, self._dsp.sample_rate, start_index=self._index
        )
        self._detector.set_noise_profile(profile)
        self._state = RecognizerState.LISTENING

    def _finalize(self, span: EventSpan) -> Prediction | None:
        frames = self._crop(span)
        if frames is None:
            return None
        sample = normalize_event(frames, self._dsp)

        start = time.perf_counter()
        probabilities = predict_probabilities(
            self._model,
            sample,
            device=self._device,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0

        index = int(np.argmax(probabilities))
        confidence = float(probabilities[index])
        return Prediction(
            label=self.label_map[index] if index < len(self.label_map) else "?",
            confidence=confidence,
            probabilities=probabilities,
            accepted=confidence >= self.min_confidence,
            event_ms=span.duration_ms(self._dsp.sample_rate),
            latency_ms=latency_ms,
            timestamp=time.time(),
        )

    def _crop(self, span: EventSpan) -> NDArray[np.float32] | None:
        oldest = self._index - len(self._buffer)
        start = max(span.start - self._pre, oldest, 0)
        end = min(span.end + self._post, self._index)
        if end - start < 4:
            return None
        offset = start - oldest
        rows = list(self._buffer)[offset : offset + (end - start)]
        if not rows:
            return None
        return np.stack(rows).astype(np.float32, copy=False)


def build_recognizer(
    checkpoint_path: str | Path,
    *,
    min_confidence: float = 0.5,
    device: str | torch.device = "auto",
) -> StreamingRecognizer:
    if isinstance(device, str) and device == "auto":
        from .train import resolve_device

        device = resolve_device("auto")
    info = load_checkpoint(checkpoint_path, device=device)
    return StreamingRecognizer(info, min_confidence=min_confidence, device=device)
