"""Typed, framework-agnostic configuration objects.

Everything in this module is plain Python: it must stay importable without Qt
or PyTorch so that the acquisition tool, the training CLI and the inference
engine share exactly the same signal-processing contract.
"""

from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, TypeVar

from .constants import (
    CHANNELS,
    DATASET_FORMAT_VERSION,
    DATASET_META_FILENAME,
    EMG_CHANNELS,
    N_CHANNELS,
)
from .jsonio import read_json, write_json

T = TypeVar("T")

#: Default ESP32 -> host UART speed.  Change it together with the firmware.
#: 500 frames/s of keyed ASCII (~110 bytes/frame) is ~550 kbit/s once the
#: UART start/stop bits are counted, so 921600 is the safe default.
DEFAULT_BAUDRATE: int = 921600
#: Sampling rate assumed when nothing else is available (Hz).
DEFAULT_SAMPLE_RATE: float = 500.0
#: Number of time steps every sample is resampled to.
DEFAULT_SEQ_LEN: int = 128

#: Conservative size of one keyed ASCII frame (90 characters + CR/LF).
FRAME_BYTES: int = 110
#: UART framing overhead: 1 start bit + 8 data bits + 1 stop bit.
UART_BITS_PER_BYTE: int = 10
_BAUD_STEPS: tuple[int, ...] = (115200, 230400, 460800, 921600, 1_500_000, 2_000_000)


def required_baudrate(sample_rate: float, *, frame_bytes: int = FRAME_BYTES) -> int:
    """Smallest standard baud rate that can carry ``sample_rate`` ASCII frames.

    The estimate is deliberately pessimistic (full 4-digit EMG values even when
    the firmware sends fewer) plus 25 % headroom for jitter and retries.
    """

    bits_per_second = sample_rate * frame_bytes * UART_BITS_PER_BYTE * 1.25
    for candidate in _BAUD_STEPS:
        if candidate >= bits_per_second:
            return candidate
    return _BAUD_STEPS[-1]


def baudrate_warning(sample_rate: float, baudrate: int) -> str | None:
    """Explain (or return ``None``) when the selected baud rate is too slow."""

    needed = required_baudrate(sample_rate)
    if baudrate >= needed:
        return None
    return (
        f"{baudrate} baud cannot carry {sample_rate:g} frames/s (~{FRAME_BYTES} "
        f"bytes each). Use {needed} baud or lower the sample rate, otherwise "
        "frames arrive late and events are truncated."
    )


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def _known_kwargs(cls: type[T], data: Mapping[str, Any]) -> dict[str, Any]:
    """Filter ``data`` down to fields that ``cls`` actually declares.

    This keeps old checkpoints / metadata files loadable after a config field is
    renamed or removed.
    """

    names = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    return {key: value for key, value in data.items() if key in names}


@dataclass(frozen=True)
class DspConfig:
    """Signal-processing contract shared by acquisition and inference.

    Pipeline (per frame):
        1. EMG: causal 4th-order Butterworth band-pass -> full-wave rectification
           -> causal 2nd-order low-pass envelope, aggregated with ``max`` across
           EMG channels.
        2. IMU: causal low-pass to suppress sensor noise above the gesture band.
        3. Every extracted event is resampled to exactly ``seq_len`` samples and
           the EMG channels are scaled to unit RMS (amplitude invariance).
    """

    sample_rate: float = DEFAULT_SAMPLE_RATE
    seq_len: int = DEFAULT_SEQ_LEN
    channels: tuple[str, ...] = CHANNELS
    emg_channels: tuple[int, ...] = EMG_CHANNELS
    emg_band_hz: tuple[float, float] = (20.0, 120.0)
    envelope_hz: float = 8.0
    imu_lowpass_hz: float = 20.0
    emg_rms_normalize: bool = True
    rms_floor: float = 1e-3

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.seq_len < 8:
            raise ValueError("seq_len must be >= 8")
        if len(self.channels) != N_CHANNELS:
            raise ValueError(f"expected {N_CHANNELS} channels, got {len(self.channels)}")
        if not self.emg_channels:
            raise ValueError("emg_channels must not be empty")
        bad = [i for i in self.emg_channels if not 0 <= i < len(self.channels)]
        if bad:
            raise ValueError(f"emg_channels out of range: {bad}")

        nyquist = self.sample_rate / 2.0
        low, high = self.emg_band_hz
        if not 0.0 < low < high < nyquist:
            raise ValueError(
                f"EMG band {self.emg_band_hz} Hz is invalid for a "
                f"{self.sample_rate:g} Hz stream (Nyquist={nyquist:g} Hz)"
            )
        if not 0.0 < self.envelope_hz < nyquist:
            raise ValueError("envelope_hz must be below Nyquist")
        if self.imu_lowpass_hz and not 0.0 < self.imu_lowpass_hz < nyquist:
            raise ValueError("imu_lowpass_hz must be below Nyquist")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DspConfig":
        payload = _known_kwargs(cls, data)
        for key in ("channels", "emg_channels", "emg_band_hz"):
            if key in payload and payload[key] is not None:
                payload[key] = tuple(payload[key])
        return cls(**payload)


@dataclass(frozen=True)
class DetectionConfig:
    """Hysteresis-based muscle-activity detector parameters.

    Thresholds are expressed in robust standard deviations of the *idle*
    envelope, so they adapt to the ADC gain of the hardware:

        on_threshold  = noise_mean + on_sigma  * noise_std
        off_threshold = noise_mean + off_sigma * noise_std
    """

    on_sigma: float = 4.0
    off_sigma: float = 2.0
    min_noise_std: float = 1e-3
    on_debounce_ms: float = 20.0
    off_debounce_ms: float = 60.0
    min_event_ms: float = 150.0
    max_event_ms: float = 1600.0
    pre_margin_ms: float = 40.0
    post_margin_ms: float = 100.0
    refractory_ms: float = 220.0
    calibration_ms: float = 2000.0
    pre_roll_ms: float = 350.0
    adapt_noise: bool = True
    adapt_tau_s: float = 4.0

    def __post_init__(self) -> None:
        if self.on_sigma <= self.off_sigma:
            raise ValueError("on_sigma must be greater than off_sigma")
        if self.off_sigma <= 0:
            raise ValueError("off_sigma must be positive")
        if self.max_event_ms <= self.min_event_ms:
            raise ValueError("max_event_ms must exceed min_event_ms")
        for name in (
            "on_debounce_ms",
            "off_debounce_ms",
            "min_event_ms",
            "max_event_ms",
            "pre_margin_ms",
            "post_margin_ms",
            "refractory_ms",
            "calibration_ms",
            "pre_roll_ms",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.adapt_tau_s <= 0:
            raise ValueError("adapt_tau_s must be positive")

    def samples(self, milliseconds: float, sample_rate: float) -> int:
        return int(round(milliseconds / 1000.0 * sample_rate))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DetectionConfig":
        return cls(**_known_kwargs(cls, data))


@dataclass
class DatasetMeta:
    """Side-car metadata stored next to the ``.npy`` samples."""

    sample_rate: float
    seq_len: int
    channels: tuple[str, ...] = CHANNELS
    dsp: dict[str, Any] = field(default_factory=dict)
    detection: dict[str, Any] = field(default_factory=dict)
    format_version: int = DATASET_FORMAT_VERSION
    created_at: str = field(default_factory=utc_now_iso)

    @classmethod
    def from_configs(
        cls,
        dsp: DspConfig,
        detection: DetectionConfig,
    ) -> "DatasetMeta":
        return cls(
            sample_rate=dsp.sample_rate,
            seq_len=dsp.seq_len,
            channels=dsp.channels,
            dsp=dsp.to_dict(),
            detection=detection.to_dict(),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["channels"] = list(self.channels)
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DatasetMeta":
        payload = _known_kwargs(cls, data)
        if "channels" in payload:
            payload["channels"] = tuple(payload["channels"])
        return cls(**payload)

    @property
    def dsp_config(self) -> DspConfig:
        if self.dsp:
            return DspConfig.from_dict(self.dsp)
        return DspConfig(sample_rate=self.sample_rate, seq_len=self.seq_len)

    @property
    def detection_config(self) -> DetectionConfig:
        return DetectionConfig.from_dict(self.detection) if self.detection else DetectionConfig()

    @classmethod
    def load(cls, dataset_root: str | Path) -> "DatasetMeta | None":
        path = Path(dataset_root) / DATASET_META_FILENAME
        if not path.exists():
            return None
        meta = cls.from_dict(read_json(path))
        if meta.format_version > DATASET_FORMAT_VERSION:
            raise ValueError(
                f"Dataset format v{meta.format_version} is newer than this "
                f"application (v{DATASET_FORMAT_VERSION}). Please update."
            )
        return meta

    def save(self, dataset_root: str | Path) -> Path:
        path = Path(dataset_root) / DATASET_META_FILENAME
        write_json(path, self.to_dict())
        return path


@dataclass(frozen=True)
class ModelConfig:
    """Architecture of the sequence classifier."""

    conv_channels: int = 64
    kernel_size: int = 5
    conv_dropout: float = 0.1
    rnn_type: str = "lstm"
    hidden_size: int = 96
    num_layers: int = 2
    bidirectional: bool = True
    dropout: float = 0.3
    attention_dim: int = 64

    def __post_init__(self) -> None:
        if self.conv_channels <= 0 or self.hidden_size <= 0:
            raise ValueError("conv_channels and hidden_size must be positive")
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve sequence length")
        if self.num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if self.rnn_type not in {"lstm", "gru"}:
            raise ValueError("rnn_type must be 'lstm' or 'gru'")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelConfig":
        return cls(**_known_kwargs(cls, data))


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 40
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    val_fraction: float = 0.2
    seed: int = 42
    patience: int = 8
    min_delta: float = 1e-4
    num_workers: int = 0
    device: str = "auto"
    class_weights: bool = True
    balanced_sampler: bool = False
    augment: bool = False
    grad_clip: float = 1.0

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if not 0.0 < self.val_fraction < 1.0:
            raise ValueError("val_fraction must be in (0, 1)")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrainConfig":
        return cls(**_known_kwargs(cls, data))
