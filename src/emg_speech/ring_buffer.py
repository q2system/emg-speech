"""Thread-safe rolling buffer of raw frames.

The serial thread only appends; the UI thread snapshots arbitrary historical
ranges (for example "everything from 350 ms before the button press").  Global,
monotonically increasing frame indices make that bookkeeping exact even when old
frames are evicted.
"""

from __future__ import annotations

import itertools
import threading
from collections import deque

import numpy as np
from numpy.typing import NDArray

from .constants import N_CHANNELS
from .protocol import Frame


class FrameRingBuffer:
    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._frames: deque[Frame] = deque(maxlen=capacity)
        self._total = 0  # number of frames ever appended
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def next_index(self) -> int:
        """Global index that the next appended frame will receive."""

        with self._lock:
            return self._total

    @property
    def oldest_index(self) -> int:
        with self._lock:
            return self._total - len(self._frames)

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)

    def append(self, frame: Frame) -> None:
        with self._lock:
            self._frames.append(np.asarray(frame, dtype=np.float32))
            self._total += 1

    def extend(self, frames: list[Frame]) -> None:
        # ``len`` rather than truthiness: callers may pass a 2-D ``ndarray``.
        if len(frames) == 0:
            return
        with self._lock:
            for frame in frames:
                self._frames.append(np.asarray(frame, dtype=np.float32))
            self._total += len(frames)

    def clear(self) -> None:
        with self._lock:
            self._frames.clear()
            self._total = 0

    def get_range(self, start: int, end: int) -> tuple[NDArray[np.float32], int]:
        """Return frames with global indices in ``[start, end)``.

        The range is clamped to what is still buffered.  Returns the array
        (``(n, 9)``, possibly empty) and the global index of its first row.
        """

        with self._lock:
            oldest = self._total - len(self._frames)
            lo = max(start, oldest)
            hi = min(end, self._total)
            if hi <= lo:
                return np.empty((0, N_CHANNELS), dtype=np.float32), lo
            offset = lo - oldest
            rows = list(itertools.islice(self._frames, offset, offset + (hi - lo)))
            return np.stack(rows).astype(np.float32, copy=False), lo

    def snapshot(self, max_frames: int | None = None) -> NDArray[np.float32]:
        with self._lock:
            frames = list(self._frames)
        if max_frames is not None:
            frames = frames[-max_frames:]
        if not frames:
            return np.empty((0, N_CHANNELS), dtype=np.float32)
        return np.stack(frames).astype(np.float32, copy=False)
