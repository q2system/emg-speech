"""EMG/IMU signal processing: streaming filters, event detection, normalisation.

The same code path is used by the acquisition tool (offline pass over a take) and
by the live inference engine (sample-by-sample), which guarantees that training
and deployment see identically processed data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfilt

from .config import DetectionConfig, DspConfig
from .constants import N_CHANNELS
from .protocol import Frame


class DspError(RuntimeError):
    """Raised when a signal cannot be processed."""


class EventNotFound(DspError):
    """Raised when a take contains no muscle activation above threshold."""


@dataclass(frozen=True)
class NoiseProfile:
    """Robust estimate of the idle-envelope distribution."""

    mean: float
    std: float

    @classmethod
    def from_values(
        cls, values: NDArray[np.floating] | list[float], *, min_std: float = 1e-3
    ) -> "NoiseProfile":
        samples = np.asarray(values, dtype=np.float64).reshape(-1)
        samples = samples[np.isfinite(samples)]
        if samples.size == 0:
            return cls(0.0, min_std)
        median = float(np.median(samples))
        mad = 1.4826 * float(np.median(np.abs(samples - median)))
        plain_std = float(np.std(samples))
        # MAD is robust to the odd movement artefact; plain std is a safety net
        # for very short calibration windows.
        std = max(mad, 0.5 * plain_std, min_std)
        return cls(mean=median, std=std)

    def on_threshold(self, cfg: DetectionConfig) -> float:
        return self.mean + cfg.on_sigma * self.std

    def off_threshold(self, cfg: DetectionConfig) -> float:
        return self.mean + cfg.off_sigma * self.std


@dataclass(frozen=True)
class EventSpan:
    """Detected activity, in absolute sample indices, ``end`` exclusive."""

    start: int
    end: int
    peak: float
    peak_index: int

    @property
    def duration(self) -> int:
        return max(0, self.end - self.start)

    def duration_ms(self, sample_rate: float) -> float:
        return self.duration / sample_rate * 1000.0


class StreamingPipeline:
    """Causal filters that can be fed one frame at a time or as a whole take."""

    def __init__(self, cfg: DspConfig) -> None:
        self.cfg = cfg
        self._n = len(cfg.channels)
        self._emg_idx = np.asarray(cfg.emg_channels, dtype=int)
        imu = [i for i in range(self._n) if i not in set(cfg.emg_channels)]
        self._imu_idx = np.asarray(imu, dtype=int)

        self._emg_sos = butter(
            4, cfg.emg_band_hz, btype="bandpass", fs=cfg.sample_rate, output="sos"
        )
        self._env_sos = butter(
            2, cfg.envelope_hz, btype="lowpass", fs=cfg.sample_rate, output="sos"
        )
        self.reset()

        self._imu_sos: NDArray[np.float64] | None = None
        self._imu_zi: NDArray[np.float64] | None = None
        if cfg.imu_lowpass_hz > 0 and self._imu_idx.size:
            self._imu_sos = butter(
                2, cfg.imu_lowpass_hz, btype="lowpass", fs=cfg.sample_rate, output="sos"
            )
            self._imu_zi = np.zeros(
                (self._imu_sos.shape[0], self._imu_idx.size, 2), dtype=np.float64
            )

    def reset(self) -> None:
        self._emg_zi = np.zeros(
            (self._emg_sos.shape[0], self._emg_idx.size, 2), dtype=np.float64
        )
        self._env_zi = np.zeros(
            (self._env_sos.shape[0], self._emg_idx.size, 2), dtype=np.float64
        )
        if getattr(self, "_imu_sos", None) is not None:
            self._imu_zi = np.zeros(
                (self._imu_sos.shape[0], self._imu_idx.size, 2), dtype=np.float64
            )

    def process_frame(self, frame: Frame) -> tuple[NDArray[np.float32], float]:
        """Filter one frame; returns ``(filtered_frame, envelope_value)``."""

        x = np.asarray(frame, dtype=np.float64).reshape(-1)
        if x.size != self._n:
            raise DspError(f"expected {self._n} channels, got {x.size}")
        filtered = x.copy()

        emg, self._emg_zi = sosfilt(
            self._emg_sos,
            x[self._emg_idx][:, None],
            axis=-1,
            zi=self._emg_zi,
        )
        envelope, self._env_zi = sosfilt(
            self._env_sos, np.abs(emg), axis=-1, zi=self._env_zi
        )
        filtered[self._emg_idx] = emg[:, 0]

        if self._imu_sos is not None and self._imu_zi is not None:
            imu, self._imu_zi = sosfilt(
                self._imu_sos,
                x[self._imu_idx][:, None],
                axis=-1,
                zi=self._imu_zi,
            )
            filtered[self._imu_idx] = imu[:, 0]

        return filtered.astype(np.float32), float(np.max(envelope))

    def process_array(
        self, frames: NDArray[np.floating]
    ) -> tuple[NDArray[np.float32], NDArray[np.float64]]:
        """Vectorised version of :meth:`process_frame` for a contiguous take.

        Filter state is *not* reset; call :meth:`reset` first for an independent
        pass (this is what :func:`extract_sample` does).
        """

        data = np.asarray(frames, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] != self._n:
            raise DspError(f"expected shape (n, {self._n}), got {data.shape}")
        if data.shape[0] == 0:
            return np.empty((0, self._n), dtype=np.float32), np.empty(0)

        source = data.T  # (channels, time)
        emg, self._emg_zi = sosfilt(
            self._emg_sos,
            source[self._emg_idx, :],
            axis=-1,
            zi=self._emg_zi,
        )
        envelope, self._env_zi = sosfilt(
            self._env_sos, np.abs(emg), axis=-1, zi=self._env_zi
        )
        filtered = source.copy()
        filtered[self._emg_idx, :] = emg
        if self._imu_sos is not None and self._imu_zi is not None:
            imu, self._imu_zi = sosfilt(
                self._imu_sos,
                source[self._imu_idx, :],
                axis=-1,
                zi=self._imu_zi,
            )
            filtered[self._imu_idx, :] = imu
        return (
            np.ascontiguousarray(filtered.T, dtype=np.float32),
            np.max(envelope, axis=0),
        )


class EventDetector:
    """Two-threshold state machine with debounce, refractory and noise tracking.

    A single active envelope sample is not enough: the detector requires
    ``on_debounce_ms`` consecutive samples above ``on_threshold`` before an event
    starts and ``off_debounce_ms`` consecutive samples below ``off_threshold``
    before it ends.  That rejects spikes and keeps pauses inside a word.
    """

    def __init__(
        self, cfg: DetectionConfig, sample_rate: float, *, start_index: int = 0
    ) -> None:
        self.cfg = cfg
        self.sample_rate = float(sample_rate)
        self._index_offset = int(start_index)
        self._on_debounce = max(1, int(round(cfg.on_debounce_ms / 1000.0 * sample_rate)))
        self._off_debounce = max(
            1, int(round(cfg.off_debounce_ms / 1000.0 * sample_rate))
        )
        self._min_samples = max(1, int(round(cfg.min_event_ms / 1000.0 * sample_rate)))
        self._max_samples = max(
            self._min_samples, int(round(cfg.max_event_ms / 1000.0 * sample_rate))
        )
        self._refractory = int(round(cfg.refractory_ms / 1000.0 * sample_rate))
        self._alpha = (
            1.0 - float(np.exp(-1.0 / (cfg.adapt_tau_s * sample_rate)))
            if cfg.adapt_noise
            else 0.0
        )

        self._profile: NoiseProfile | None = None
        self._mean = 0.0
        self._std = cfg.min_noise_std
        self._index = self._index_offset
        self._active = False
        self._start = 0
        self._above = 0
        self._below = 0
        self._peak = 0.0
        self._peak_index = 0
        self._cooldown_until = 0
        self.last_rejection: str | None = None

    # ------------------------------------------------------------------ setup
    def set_noise_profile(self, profile: NoiseProfile) -> None:
        self._profile = profile
        self._mean = profile.mean
        self._std = max(profile.std, self.cfg.min_noise_std)
        self._reset_state(keep_profile=True)

    def reset(self, *, keep_profile: bool = True) -> None:
        self._reset_state(keep_profile=keep_profile)

    def _reset_state(self, *, keep_profile: bool) -> None:
        if keep_profile and self._profile is not None:
            self._mean = self._profile.mean
            self._std = max(self._profile.std, self.cfg.min_noise_std)
        self._index = self._index_offset
        self._active = False
        self._start = 0
        self._above = 0
        self._below = 0
        self._peak = 0.0
        self._peak_index = 0
        self._cooldown_until = 0
        self.last_rejection = None

    # -------------------------------------------------------------- properties
    @property
    def has_profile(self) -> bool:
        return self._profile is not None

    @property
    def on_threshold(self) -> float:
        return self._mean + self.cfg.on_sigma * self._std

    @property
    def off_threshold(self) -> float:
        return self._mean + self.cfg.off_sigma * self._std

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def active_start(self) -> int | None:
        return self._start if self._active else None

    @property
    def active_duration_samples(self) -> int:
        return self._index - self._start if self._active else 0

    # ------------------------------------------------------------------- feed
    def step(self, value: float) -> EventSpan | None:
        """Feed one envelope sample.  Returns a span when an event just ended."""

        if self._profile is None:
            raise DspError("EventDetector requires a noise profile before step()")

        sample = float(value)
        if not np.isfinite(sample):
            sample = 0.0
        index = self._index
        self._index += 1

        if self._alpha and not self._active and sample < self.on_threshold:
            self._adapt(sample)

        on_threshold = self.on_threshold
        off_threshold = self.off_threshold

        if self._active:
            if sample > self._peak:
                self._peak = sample
                self._peak_index = index

            if sample < off_threshold:
                self._below += 1
                if self._below >= self._off_debounce:
                    end = index - self._below + 1
                    span = EventSpan(self._start, end, self._peak, self._peak_index)
                    self._finish(index)
                    if span.duration < self._min_samples:
                        self.last_rejection = (
                            f"muscle burst too short "
                            f"({span.duration_ms(self.sample_rate):.0f} ms)"
                        )
                        return None
                    return span
            else:
                self._below = 0

            if index - self._start + 1 >= self._max_samples:
                span = EventSpan(
                    self._start,
                    self._start + self._max_samples,
                    self._peak,
                    self._peak_index,
                )
                self._finish(index)
                return span
            return None

        if index < self._cooldown_until:
            self._above = 0
            return None

        if sample >= on_threshold:
            self._above += 1
            if self._above >= self._on_debounce:
                self._active = True
                self._start = index - self._above + 1
                self._peak = sample
                self._peak_index = index
                self._below = 0
        else:
            self._above = 0
        return None

    def _finish(self, index: int) -> None:
        self._active = False
        self._above = 0
        self._below = 0
        self._cooldown_until = index + self._refractory

    def _adapt(self, sample: float) -> None:
        previous = self._mean
        self._mean = previous + self._alpha * (sample - previous)
        variance = (1.0 - self._alpha) * (self._std**2) + self._alpha * (
            sample - previous
        ) ** 2
        self._std = max(float(np.sqrt(max(variance, 0.0))), self.cfg.min_noise_std)

    def run(
        self, envelope: NDArray[np.floating] | list[float]
    ) -> list[EventSpan]:
        """Offline pass over an envelope array (resets timing state)."""

        self.reset(keep_profile=True)
        spans: list[EventSpan] = []
        for value in np.asarray(envelope, dtype=np.float64).reshape(-1):
            span = self.step(float(value))
            if span is not None:
                spans.append(span)
        return spans


# --------------------------------------------------------------------- helpers
def resample_to_length(
    data: NDArray[np.floating], length: int
) -> NDArray[np.float32]:
    """Linear time-resampling to exactly ``length`` rows (channels preserved)."""

    array = np.asarray(data, dtype=np.float64)
    if array.ndim != 2:
        raise DspError(f"expected a 2-D array, got shape {array.shape}")
    if length < 1:
        raise DspError("length must be >= 1")
    rows = array.shape[0]
    if rows == 0:
        raise DspError("cannot resample an empty event")
    if rows == length:
        return array.astype(np.float32)
    if rows == 1:
        return np.repeat(array, length, axis=0).astype(np.float32)

    source = np.arange(rows, dtype=np.float64)
    target = np.linspace(0.0, rows - 1.0, length)
    output = np.empty((length, array.shape[1]), dtype=np.float64)
    for channel in range(array.shape[1]):
        output[:, channel] = np.interp(target, source, array[:, channel])
    return output.astype(np.float32)


def normalize_event(
    event: NDArray[np.floating], cfg: DspConfig
) -> NDArray[np.float32]:
    """Time-normalise to ``cfg.seq_len`` and make EMG amplitude-invariant."""

    output = resample_to_length(event, cfg.seq_len)
    if cfg.emg_rms_normalize and output.size:
        emg = output[:, list(cfg.emg_channels)]
        rms = float(np.sqrt(np.mean(np.square(emg, dtype=np.float64))))
        if rms > cfg.rms_floor:
            output[:, list(cfg.emg_channels)] = (emg / rms).astype(np.float32)
    return np.ascontiguousarray(output, dtype=np.float32)


@dataclass(frozen=True)
class ExtractedSample:
    data: NDArray[np.float32]
    event_ms: float
    peak_envelope: float
    on_threshold: float
    off_threshold: float
    source_frames: int
    event_start: int
    event_end: int
    warnings: tuple[str, ...] = field(default_factory=tuple)
    noise_profile: "NoiseProfile | None" = None


def _take_calibration_frames(dsp: DspConfig, detection: DetectionConfig, n: int) -> int:
    preferred = max(
        int(round(detection.pre_roll_ms / 1000.0 * dsp.sample_rate)),
        int(round(0.25 * dsp.sample_rate)),
    )
    # Never spend more than 40 % of the take on threshold estimation.
    return max(1, min(n, preferred, max(1, int(0.4 * n))))


def extract_sample(
    take: NDArray[np.floating],
    dsp_cfg: DspConfig,
    detection_cfg: DetectionConfig,
    noise_profile: NoiseProfile | None = None,
) -> ExtractedSample:
    """Run the complete DSP chain on one button-press take.

    Parameters
    ----------
    take:
        Raw ``(n, 9)`` frames captured from ``press - pre_roll`` to
        ``release + tail``.
    noise_profile:
        Optional idle measurement.  When omitted, the leading pre-roll is used.

    Raises
    ------
    EventNotFound
        If no valid burst is found.  The UI should reject the take rather than
        store silence in the training set.
    """

    data = np.asarray(take, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] != N_CHANNELS:
        raise DspError(f"expected shape (n, {N_CHANNELS}), got {data.shape}")
    if data.shape[0] < 8:
        raise DspError("take is too short to contain a gesture")

    pipeline = StreamingPipeline(dsp_cfg)
    filtered, envelope = pipeline.process_array(data)
    if envelope.size == 0:
        raise DspError("take is empty")

    if noise_profile is None:
        calibration = _take_calibration_frames(dsp_cfg, detection_cfg, len(data))
        noise_profile = NoiseProfile.from_values(
            envelope[:calibration], min_std=detection_cfg.min_noise_std
        )

    detector = EventDetector(detection_cfg, dsp_cfg.sample_rate)
    detector.set_noise_profile(noise_profile)
    spans = detector.run(envelope)
    if not spans:
        reason = detector.last_rejection or "no activity above threshold"
        raise EventNotFound(
            f"No usable muscle activity detected ({reason}). "
            f"Peak envelope {float(np.max(envelope)):.4g} vs. threshold "
            f"{detector.on_threshold:.4g}. Try lowering 'on_sigma', hold the "
            "button slightly longer, or check the electrode contact."
        )

    warnings: list[str] = []
    best = max(spans, key=lambda span: (span.peak, span.duration))
    if len(spans) > 1:
        warnings.append(
            f"{len(spans)} bursts detected; kept the strongest "
            f"({best.duration_ms(dsp_cfg.sample_rate):.0f} ms)"
        )

    pre = int(round(detection_cfg.pre_margin_ms / 1000.0 * dsp_cfg.sample_rate))
    post = int(round(detection_cfg.post_margin_ms / 1000.0 * dsp_cfg.sample_rate))
    start = max(0, best.start - pre)
    end = min(len(filtered), best.end + post)
    if start != best.start - pre or end != best.end + post:
        warnings.append("event touches the edge of the captured take")
    if end - start < 4:
        raise DspError("detected event is too short after cropping")

    normalized = normalize_event(filtered[start:end], dsp_cfg)
    return ExtractedSample(
        data=normalized,
        event_ms=best.duration_ms(dsp_cfg.sample_rate),
        peak_envelope=float(best.peak),
        on_threshold=float(noise_profile.on_threshold(detection_cfg)),
        off_threshold=float(noise_profile.off_threshold(detection_cfg)),
        source_frames=len(data),
        event_start=start,
        event_end=end,
        warnings=tuple(warnings),
        noise_profile=noise_profile,
    )
