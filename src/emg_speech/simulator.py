"""Continuous virtual biosignal source for hardware-free serial testing.

The simulator produces *raw* 9-channel frames (not pre-normalised dataset
samples), so it exercises the complete acquisition chain:

    UART framing -> band-pass -> envelope -> event detection -> normalisation

Two modes are available:

``auto``
    Emit a gesture every ``interval_seconds`` using a randomly chosen word.
    A :class:`Cue` is recorded for every gesture so the CLI/user knows when to
    press the record button.

``manual``
    Stay idle until :meth:`SimulatedSource.trigger` is called (the CLI maps
    this to typing a word on stdin).  This is the most controllable way to test
    the acquisition workflow.

The class implements the :class:`emg_speech.sources.FrameSource` protocol, but
it has no serial dependency: ``simulate.py`` decides whether frames go to a
real COM port, a com0com pair or a POSIX PTY.
"""

from __future__ import annotations

import queue
import time
import zlib
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .constants import CHANNELS, N_CHANNELS
from .protocol import Frame
from .sources import SourceError
from .storage import sanitize_class_name

DEFAULT_WORDS: tuple[str, ...] = ("alpha", "beta", "gamma", "delta")


@dataclass(frozen=True)
class WordStyle:
    """Deterministic synthesis parameters for one word."""

    carrier_hz: float
    envelope_kind: int
    motion_hz: float
    motion_axis: int
    accel_hz: float
    motion_phase: float
    accel_phase: float


#: The demo vocabulary mirrors ``synthetic.py`` so demo models behave the same.
_DEFAULT_STYLES: dict[str, WordStyle] = {
    "alpha": WordStyle(40.0, 0, 1.2, 0, 2.0, 0.0, 0.0),
    "beta": WordStyle(65.0, 1, 1.5, 1, 2.2, 1.0, 0.0),
    "gamma": WordStyle(90.0, 2, 1.8, 2, 2.4, 2.0, 0.0),
    "delta": WordStyle(110.0, 3, 2.1, 0, 2.6, 3.0, 0.0),
}


def style_for_word(word: str) -> WordStyle:
    """Return stable, distinguishable synthesis parameters for any word.

    Known demo words use hand-tuned carriers; anything else is hashed into the
    35–104 Hz band so re-running the simulator with the same labels produces the
    same gestures.
    """

    known = _DEFAULT_STYLES.get(word)
    if known is not None:
        return known
    digest = zlib.crc32(word.encode("utf-8"))
    return WordStyle(
        carrier_hz=35.0 + float(digest % 70),
        envelope_kind=int((digest >> 8) % 4),
        motion_hz=1.0 + float((digest >> 16) % 15) / 5.0,
        motion_axis=int((digest >> 24) % 3),
        accel_hz=2.0 + float((digest >> 20) % 10) / 5.0,
        motion_phase=float((digest >> 4) % 100) / 25.0,
        accel_phase=float((digest >> 12) % 100) / 25.0,
    )


@dataclass(frozen=True)
class Cue:
    """A gesture starts at ``frame`` (absolute sample index)."""

    frame: int
    word: str

    def time_s(self, sample_rate: float) -> float:
        return self.frame / sample_rate


@dataclass
class SimulationStats:
    frames_produced: int = 0
    gestures: int = 0
    word_counts: dict[str, int] = field(default_factory=dict)


def _envelope(kind: int, t: NDArray[np.float64]) -> NDArray[np.float64]:
    """Envelope shapes copied from the synthetic dataset generator."""

    if kind == 0:
        return np.sin(np.pi * t) ** 1.4
    if kind == 1:
        return np.abs(np.sin(2.0 * np.pi * t)) ** 2.0
    if kind == 2:
        return np.clip(t**1.2, 0.0, 1.0) * np.clip(2.0 - 1.5 * t, 0.0, 1.0)
    return np.clip(1.6 * np.sin(np.pi * t), 0.0, 1.0)


class SimulatedSource:
    """Generates an endless, real-time-paced stream of raw 9-channel frames."""

    def __init__(
        self,
        words: Sequence[str] = DEFAULT_WORDS,
        *,
        sample_rate: float = 500.0,
        mode: str = "auto",
        seed: int = 7,
        interval_seconds: float = 3.0,
        interval_jitter: float = 0.25,
        start_idle_seconds: float = 2.0,
        idle_emg_std: float = 0.02,
        idle_imu_std: float = 0.003,
        amplitude: float = 1.1,
        max_chunk: int = 4096,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if mode not in {"auto", "manual"}:
            raise ValueError("mode must be 'auto' or 'manual'")
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        if interval_jitter < 0 or interval_jitter >= 1.0:
            raise ValueError("interval_jitter must be in [0, 1)")
        if start_idle_seconds < 0:
            raise ValueError("start_idle_seconds must be non-negative")
        if idle_emg_std <= 0 or idle_imu_std < 0:
            raise ValueError("noise levels must be non-negative")
        if amplitude <= 0:
            raise ValueError("amplitude must be positive")
        if max_chunk < 1:
            raise ValueError("max_chunk must be positive")

        cleaned = tuple(dict.fromkeys(sanitize_class_name(word) for word in words))
        if not cleaned:
            raise ValueError("at least one word is required")

        self._words = cleaned
        self._sample_rate = float(sample_rate)
        self._mode = mode
        self._interval = float(interval_seconds)
        self._jitter = float(interval_jitter)
        self._start_idle = float(start_idle_seconds)
        self._idle_emg_std = float(idle_emg_std)
        self._idle_imu_std = float(idle_imu_std)
        self._amplitude = float(amplitude)
        self._max_chunk = int(max_chunk)
        self._seed = int(seed)
        self._rng = np.random.default_rng(seed)

        self._segment: NDArray[np.float32] | None = None
        self._cursor = 0
        self._produced = 0
        self._delivered = 0
        self._next_gesture_at = 0
        self._t0: float | None = None
        self._triggers: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._cues: list[Cue] = []
        self._stats = SimulationStats()

    # ------------------------------------------------------------- properties
    @property
    def description(self) -> str:
        return f"simulate:{','.join(self._words)} ({self._mode})"

    @property
    def words(self) -> tuple[str, ...]:
        return self._words

    @property
    def sample_rate(self) -> float:
        return self._sample_rate

    @property
    def stats(self) -> SimulationStats:
        return self._stats

    # ------------------------------------------------------------------ source
    def open(self) -> None:
        self._rng = np.random.default_rng(self._seed)
        self._segment = None
        self._cursor = 0
        self._produced = 0
        self._delivered = 0
        self._cues.clear()
        self._stats = SimulationStats()
        self._t0 = time.monotonic()
        self._next_gesture_at = int(round(self._start_idle * self._sample_rate))

    def close(self) -> None:
        self._t0 = None

    def read_available(self) -> list[Frame]:
        """Return the frames that should have been produced by now."""

        if self._t0 is None:
            raise SourceError("SimulatedSource is not open")
        target = int((time.monotonic() - self._t0) * self._sample_rate)
        count = min(max(0, target - self._delivered), self._max_chunk)
        if count <= 0:
            return []
        frames = self.pull(count)
        self._delivered += len(frames)
        return [row.copy() for row in frames]

    def pull(self, count: int) -> NDArray[np.float32]:
        """Generate ``count`` frames immediately (unpaced, used by tests/CLI)."""

        if count <= 0:
            return np.empty((0, N_CHANNELS), dtype=np.float32)

        chunks: list[NDArray[np.float32]] = []
        already = 0
        while already < count:
            if self._segment is None or self._cursor >= len(self._segment):
                self._segment = self._next_segment(self._produced + already)
                self._cursor = 0
            take = min(count - already, len(self._segment) - self._cursor)
            chunks.append(self._segment[self._cursor : self._cursor + take])
            self._cursor += take
            already += take
        self._produced += already
        self._stats.frames_produced = self._produced
        return np.ascontiguousarray(np.concatenate(chunks), dtype=np.float32)

    # ----------------------------------------------------------------- control
    def trigger(self, word: str) -> str:
        """Queue one manual gesture; returns the sanitised word."""

        name = sanitize_class_name(word)
        if name not in self._words:
            self._words = (*self._words, name)
        self._triggers.put(name)
        return name

    def drain_cues(self) -> list[Cue]:
        cues = self._cues
        self._cues = []
        return cues

    # -------------------------------------------------------------- generation
    def _next_segment(self, start: int) -> NDArray[np.float32]:
        if self._mode == "manual":
            word = self._next_trigger()
            if word is not None:
                return self._gesture_segment(start, word)
            return self._idle_segment(start, max(1, int(round(0.05 * self._sample_rate))))

        if start >= self._next_gesture_at:
            word = self._words[int(self._rng.integers(0, len(self._words)))]
            segment = self._gesture_segment(start, word)
            gap = self._interval * self._sample_rate
            gap *= 1.0 + float(self._rng.uniform(-self._jitter, self._jitter))
            self._next_gesture_at = start + max(len(segment), max(1, int(round(gap))))
            return segment

        remaining = self._next_gesture_at - start
        chunk = max(1, int(round(0.05 * self._sample_rate)))
        return self._idle_segment(start, min(chunk, max(1, remaining)))

    def _next_trigger(self) -> str | None:
        try:
            return self._triggers.get_nowait()
        except queue.Empty:
            return None

    def _gesture_segment(self, start: int, word: str) -> NDArray[np.float32]:
        """Build a synthetic take: 0.5 s pre-roll, burst, 0.5 s tail.

        The pre-roll gives the acquisition UI something to calibrate on and the
        class-specific IMU motion spans the whole take, exactly like
        :func:`emg_speech.synthetic._make_take`.  The cue points at the first
        frame of the burst, not at the start of the pre-roll.
        """

        style = style_for_word(word)
        pre = int(round(0.5 * self._sample_rate))
        post = int(round(0.5 * self._sample_rate))
        duration = float(self._rng.uniform(0.22, 0.40))
        body = max(1, int(round(duration * self._sample_rate)))
        total = pre + body + post
        # Take-relative time keeps the IMU/EMG phase relationship identical to
        # the synthetic training takes, which is what the demo model expects.
        t = np.arange(total, dtype=np.float64) / self._sample_rate

        emg = self._rng.normal(0.0, self._idle_emg_std, size=(total, 3))
        body_t = np.arange(body, dtype=np.float64) / self._sample_rate
        t_norm = np.linspace(0.0, 1.0, body)
        env = _envelope(style.envelope_kind, t_norm)
        for channel in range(3):
            frequency = style.carrier_hz * (1.0 + 0.03 * channel)
            frequency += float(self._rng.uniform(-2.0, 2.0))
            phase = float(self._rng.uniform(0.0, 2.0 * np.pi))
            carrier = np.sin(2.0 * np.pi * frequency * body_t + phase)
            carrier += 0.25 * np.sin(
                2.0 * np.pi * frequency * 1.7 * body_t + phase / 2.0
            )
            emg[pre : pre + body, channel] += self._amplitude * env * carrier

        gyro, accel = self._base_imu(t)
        gyro[:, style.motion_axis] += 0.04 * np.sin(
            2.0 * np.pi * style.motion_hz * t + style.motion_phase
        )
        accel[:, (style.motion_axis + 1) % 3] += 0.02 * np.sin(
            2.0 * np.pi * style.accel_hz * t + style.accel_phase
        )

        self._stats.gestures += 1
        self._stats.word_counts[word] = self._stats.word_counts.get(word, 0) + 1
        self._cues.append(Cue(start + pre, word))
        return self._assemble(emg, gyro, accel)

    def _idle_segment(self, start: int, count: int) -> NDArray[np.float32]:
        t = (start + np.arange(count, dtype=np.float64)) / self._sample_rate
        emg = self._rng.normal(0.0, self._idle_emg_std, size=(count, 3))
        gyro, accel = self._base_imu(t)
        return self._assemble(emg, gyro, accel)

    def _base_imu(
        self, t: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        count = t.size
        gyro = self._rng.normal(0.0, self._idle_imu_std, size=(count, 3))
        accel = self._rng.normal(0.0, self._idle_imu_std, size=(count, 3))
        accel[:, 2] += 0.98
        return gyro, accel

    @staticmethod
    def _assemble(
        emg: NDArray[np.float64],
        gyro: NDArray[np.float64],
        accel: NDArray[np.float64],
    ) -> NDArray[np.float32]:
        return np.ascontiguousarray(
            np.concatenate([emg, gyro, accel], axis=1), dtype=np.float32
        )


def encode_line(frame: Frame) -> str:
    """Encode one frame in the firmware's keyed ASCII format (no newline)."""

    values = np.asarray(frame, dtype=np.float64).reshape(-1)
    if values.size != N_CHANNELS:
        raise ValueError(f"expected {N_CHANNELS} channels, got {values.size}")
    parts = [
        f"{name}:{values[index]:.2f}" if index < 3 else f"{name}:{values[index]:.3f}"
        for index, name in enumerate(CHANNELS)
    ]
    return ",".join(parts)
