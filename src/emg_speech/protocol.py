"""UART protocol parsing for the 9-channel ESP32 stream.

Incoming frame::

    emg1:512,emg2:498,emg3:505,gx:1.2,gy:-0.4,gz:0.1,ax:0.02,ay:-0.01,az:0.98\n

Parsing is deliberately defensive: firmware boot logs, truncated lines, partial
reads and electrical noise must never crash the acquisition thread.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .constants import CHANNELS, N_CHANNELS

Frame = NDArray[np.float32]
"""One validated sample, shape ``(9,)``, channel order given by :data:`CHANNELS`."""

_KEY_ALIASES: dict[str, int] = {
    # EMG
    "emg1": 0,
    "emg_1": 0,
    "emg2": 1,
    "emg_2": 1,
    "emg3": 2,
    "emg_3": 2,
    # Gyroscope
    "gx": 3,
    "gyro_x": 3,
    "gy": 4,
    "gyro_y": 4,
    "gz": 5,
    "gyro_z": 5,
    # Accelerometer
    "ax": 6,
    "accel_x": 6,
    "ay": 7,
    "accel_y": 7,
    "az": 8,
    "accel_z": 8,
}

_MAX_ABS_VALUE: float = 1e9


class FrameParseError(ValueError):
    """Raised when a UART line cannot be converted into a valid frame."""


@dataclass
class ParseStats:
    parsed: int = 0
    malformed: int = 0
    last_error: str | None = None


def _normalise_key(key: str) -> str:
    return key.strip().lower().replace("-", "_").replace(" ", "")


def parse_frame(line: str, *, n_channels: int = N_CHANNELS) -> Frame:
    """Parse one line into a float32 frame.

    Two layouts are accepted:

    * keyed (the firmware format): ``emg1:1,emg2:2,...``
    * positional: ``1,2,3,4,5,6,7,8,9``

    Raises
    ------
    FrameParseError
        For empty lines, missing channels, non-numeric values or non-finite
        numbers.  The caller decides whether to skip the frame.
    """

    text = line.strip()
    if not text:
        raise FrameParseError("empty line")

    tokens = [token.strip() for token in text.split(",") if token.strip()]
    if not tokens:
        raise FrameParseError("no values")

    values = np.full(n_channels, np.nan, dtype=np.float64)

    if any(":" in token for token in tokens):
        seen: set[int] = set()
        for token in tokens:
            if ":" not in token:
                raise FrameParseError(f"mixed keyed/positional token: '{token}'")
            key, raw_value = token.split(":", 1)
            index = _KEY_ALIASES.get(_normalise_key(key))
            if index is None:
                # Unknown extra fields are tolerated (forward compatibility).
                continue
            if index >= n_channels:
                raise FrameParseError(f"channel '{key}' exceeds n_channels={n_channels}")
            values[index] = _to_float(raw_value, key)
            seen.add(index)
        missing = [CHANNELS[i] for i in range(n_channels) if i not in seen]
        if missing:
            raise FrameParseError(f"missing channels: {', '.join(missing)}")
    else:
        if len(tokens) != n_channels:
            raise FrameParseError(
                f"expected {n_channels} positional values, got {len(tokens)}"
            )
        values[:] = [_to_float(token, f"column {i}") for i, token in enumerate(tokens)]

    if not np.all(np.isfinite(values)):
        raise FrameParseError("non-finite value")
    return values.astype(np.float32, copy=False)


def _to_float(raw: str, field: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise FrameParseError(f"invalid number for '{field}': '{raw}'") from exc
    if not np.isfinite(value) or abs(value) > _MAX_ABS_VALUE:
        raise FrameParseError(f"out-of-range value for '{field}': '{raw}'")
    return value


class FrameParser:
    """Stateful parser that counts good and malformed lines."""

    def __init__(self, *, n_channels: int = N_CHANNELS) -> None:
        self._n_channels = n_channels
        self.stats = ParseStats()

    def parse(self, line: str) -> Frame | None:
        try:
            frame = parse_frame(line, n_channels=self._n_channels)
        except FrameParseError as exc:
            self.stats.malformed += 1
            self.stats.last_error = str(exc)
            return None
        self.stats.parsed += 1
        return frame

    def reset(self) -> None:
        self.stats = ParseStats()


class LineFramer:
    """Splits a byte stream into lines, tolerating partial and oversized reads."""

    def __init__(self, max_line_bytes: int = 512) -> None:
        if max_line_bytes < 16:
            raise ValueError("max_line_bytes is too small")
        self.max_line_bytes = max_line_bytes
        self._buffer = bytearray()
        self.dropped = 0

    def feed(self, chunk: bytes) -> list[str]:
        if not chunk:
            return []
        self._buffer.extend(chunk)
        lines: list[str] = []
        while True:
            newline = self._buffer.find(b"\n")
            if newline < 0:
                break
            raw = bytes(self._buffer[:newline])
            del self._buffer[: newline + 1]
            if len(raw) > self.max_line_bytes:
                self.dropped += 1
                continue
            text = raw.decode("ascii", errors="replace").strip()
            if text:
                lines.append(text)
        # A partial line larger than the cap can only be garbage: resynchronise.
        if len(self._buffer) > self.max_line_bytes:
            self._buffer.clear()
            self.dropped += 1
        return lines

    def reset(self) -> None:
        self._buffer.clear()
        self.dropped = 0
