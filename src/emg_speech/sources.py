"""Frame sources: real UART and a deterministic replay source for development."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from .config import DEFAULT_BAUDRATE
from .constants import N_CHANNELS
from .protocol import Frame, FrameParser, LineFramer
from .storage import scan_dataset

try:  # pragma: no cover - exercised only with hardware
    import serial
    from serial.tools import list_ports
except ImportError:  # pragma: no cover
    serial = None  # type: ignore[assignment]
    list_ports = None  # type: ignore[assignment]


class SourceError(RuntimeError):
    """Raised when a frame source cannot be opened or read."""


@runtime_checkable
class FrameSource(Protocol):
    @property
    def description(self) -> str: ...

    def open(self) -> None: ...

    def read_available(self) -> list[Frame]: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class PortInfo:
    device: str
    description: str


def available_ports() -> list[PortInfo]:
    """Enumerate serial ports without importing Qt."""

    if list_ports is None:  # pragma: no cover
        return []
    return [
        PortInfo(port.device, port.description or port.device)
        for port in list_ports.comports()
    ]


class SerialSource:
    """Non-blocking UART reader for the ESP32 stream."""

    def __init__(
        self,
        port: str,
        baudrate: int = DEFAULT_BAUDRATE,
        *,
        max_line_bytes: int = 512,
        deassert_control_lines: bool = True,
        settle_seconds: float = 0.0,
    ) -> None:
        if serial is None:  # pragma: no cover
            raise SourceError("pyserial is not installed (pip install pyserial)")
        self._port = port
        self._baudrate = baudrate
        self._deassert = deassert_control_lines
        self._settle_seconds = max(0.0, settle_seconds)
        self._serial: "serial.Serial | None" = None
        self._framer = LineFramer(max_line_bytes=max_line_bytes)
        self._parser = FrameParser()

    @property
    def description(self) -> str:
        return f"{self._port} @ {self._baudrate} baud"

    @property
    def stats(self):
        return self._parser.stats

    def open(self) -> None:
        if self._serial is not None and self._serial.is_open:
            return
        try:
            self._serial = serial.Serial(
                port=self._port,
                baudrate=self._baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0,
                write_timeout=0,
                xonxoff=False,
                rtscts=False,
                dsrdtr=False,
            )
            if self._deassert:
                # Keep the ESP32 auto-reset circuit from rebooting the board when
                # the port is opened.  Some USB bridges do not support this.
                try:
                    self._serial.dtr = False
                    self._serial.rts = False
                except (OSError, ValueError):
                    pass
            self._serial.reset_input_buffer()
        except Exception as exc:  # serial.SerialException / OSError / PermissionError
            self._serial = None
            raise SourceError(f"Could not open {self.description}: {exc}") from exc
        self._framer.reset()
        self._parser.reset()
        if self._settle_seconds:
            time.sleep(self._settle_seconds)

    def read_available(self) -> list[Frame]:
        if self._serial is None or not self._serial.is_open:
            raise SourceError("Serial port is not open")
        try:
            waiting = self._serial.in_waiting
            if not waiting:
                return []
            chunk = self._serial.read(min(waiting, 65536))
        except Exception as exc:
            self.close()
            raise SourceError(f"Serial read failed ({self.description}): {exc}") from exc

        frames: list[Frame] = []
        for line in self._framer.feed(chunk):
            frame = self._parser.parse(line)
            if frame is not None:
                frames.append(frame)
        return frames

    def close(self) -> None:
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None

    def __enter__(self) -> "SerialSource":
        self.open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _encode_lines(lines: Sequence[str | bytes]) -> bytes:
    encoded: list[bytes] = []
    for line in lines:
        if isinstance(line, bytes):
            encoded.append(line)
        else:
            encoded.append(line.encode("ascii", errors="replace"))
    return b"\n".join(encoded) + b"\n"


class SerialWriter:
    """Writes ASCII frames to a real or virtual (com0com) serial port.

    ``write_lines`` is deliberately non-blocking in spirit: if the operating
    system cannot accept data (no reader attached, buffer full), the bytes stay
    in a bounded pending buffer and old data is dropped instead of replaying a
    stale backlog when the GUI finally connects.
    """

    MAX_PENDING_BYTES = 256 * 1024

    def __init__(
        self,
        port: str,
        baudrate: int = DEFAULT_BAUDRATE,
        *,
        write_timeout: float = 0.5,
    ) -> None:
        if serial is None:  # pragma: no cover
            raise SourceError("pyserial is not installed (pip install pyserial)")
        self._port = port
        self._baudrate = int(baudrate)
        self._write_timeout = float(write_timeout)
        self._serial: "serial.Serial | None" = None
        self._pending = bytearray()
        self.dropped_bytes = 0

    @property
    def description(self) -> str:
        return f"{self._port} @ {self._baudrate} baud (writer)"

    @property
    def pending_bytes(self) -> int:
        return len(self._pending)

    def open(self) -> None:
        try:
            self._serial = serial.Serial(
                port=self._port,
                baudrate=self._baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0,
                write_timeout=self._write_timeout,
                xonxoff=False,
                rtscts=False,
                dsrdtr=False,
            )
            self._serial.reset_output_buffer()
        except Exception as exc:
            self._serial = None
            raise SourceError(
                f"Could not open {self.description}: {exc}"
            ) from exc

    def write_lines(self, lines: Sequence[str | bytes]) -> int:
        if self._serial is None or not self._serial.is_open:
            raise SourceError("Serial port is not open")
        if lines:
            self._pending.extend(_encode_lines(lines))
        self._trim_pending()
        if not self._pending:
            return 0
        try:
            written = int(self._serial.write(bytes(self._pending)))
        except Exception as exc:
            if serial is not None and isinstance(exc, serial.SerialTimeoutException):
                # The reader is not draining: prefer fresh frames over a stale
                # backlog once the app reconnects.
                self.dropped_bytes += len(self._pending)
                self._pending.clear()
                return 0
            raise SourceError(f"Serial write failed ({self.description}): {exc}") from exc
        del self._pending[:written]
        return written

    def _trim_pending(self) -> None:
        overflow = len(self._pending) - self.MAX_PENDING_BYTES
        if overflow > 0:
            del self._pending[:overflow]
            self.dropped_bytes += overflow

    def close(self) -> None:
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None
        self._pending.clear()

    def __enter__(self) -> "SerialWriter":
        self.open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class PtyWriter:
    """Writes to a POSIX pseudo-terminal pair (Linux/macOS).

    ``slave_name`` is the path the applications should open, for example by
    typing it into the port box or passing ``--port /dev/pts/7``.  The slave is
    kept open so the master never sees EIO when the GUI disconnects.
    """

    MAX_PENDING_BYTES = 256 * 1024

    def __init__(self) -> None:
        if os.name == "nt":  # pragma: no cover - Windows has no PTYs
            raise SourceError(
                "PTYs are not available on Windows; install com0com and use "
                "--port COMx instead."
            )
        import pty
        import tty

        master_fd, slave_fd = pty.openpty()
        tty.setraw(slave_fd)
        self._master = master_fd
        self._slave = slave_fd
        self._slave_name = os.ttyname(slave_fd)
        os.set_blocking(self._master, False)
        self._pending = bytearray()
        self.dropped_bytes = 0

    @property
    def slave_name(self) -> str:
        return self._slave_name

    @property
    def description(self) -> str:
        return f"pty: {self._slave_name}"

    @property
    def pending_bytes(self) -> int:
        return len(self._pending)

    def open(self) -> None:
        """No-op: the pair is created in ``__init__``."""

    def write_lines(self, lines: Sequence[str | bytes]) -> int:
        if self._master is None:
            raise SourceError("PTY is not open")
        if lines:
            self._pending.extend(_encode_lines(lines))
        overflow = len(self._pending) - self.MAX_PENDING_BYTES
        if overflow > 0:
            del self._pending[:overflow]
            self.dropped_bytes += overflow
        if not self._pending:
            return 0
        try:
            written = os.write(self._master, bytes(self._pending))
        except (BlockingIOError, InterruptedError):
            # Live streams should resume with fresh frames, not replay seconds
            # of stale data accumulated while no reader was attached.
            self.dropped_bytes += len(self._pending)
            self._pending.clear()
            return 0
        except OSError:
            # No slave reader attached / PTY disappeared: drop the backlog.
            self.dropped_bytes += len(self._pending)
            self._pending.clear()
            return 0
        del self._pending[:written]
        return written

    def close(self) -> None:
        for fd in (getattr(self, "_master", None), getattr(self, "_slave", None)):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        self._master = None
        self._slave = None
        self._pending.clear()


class ReplaySource:
    """Replays dataset samples with realistic idle noise, paced in real time.

    Useful to develop and demo the inference UI without the wearable.
    Samples are concatenated with rest periods and looped.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        sample_rate: float = 500.0,
        loop: bool = True,
        idle_seconds: tuple[float, float] = (0.7, 1.2),
        start_idle_seconds: float = 2.5,
        idle_emg_std: float = 0.03,
        idle_imu_std: float = 0.002,
        seed: int = 7,
        max_samples: int = 200,
    ) -> None:
        self._dataset_root = Path(dataset_root)
        self._sample_rate = sample_rate
        self._loop = loop
        self._idle_seconds = idle_seconds
        self._start_idle_seconds = max(0.0, start_idle_seconds)
        self._idle_emg_std = idle_emg_std
        self._idle_imu_std = idle_imu_std
        self._rng = np.random.default_rng(seed)
        self._max_samples = max_samples
        self._timeline: np.ndarray | None = None
        self._cursor = 0
        self._t0: float | None = None

    @property
    def description(self) -> str:
        return f"replay: {self._dataset_root}"

    def open(self) -> None:
        if self._timeline is None:
            self._timeline = self._build_timeline()
        self._cursor = 0
        self._t0 = time.monotonic()

    def _build_timeline(self) -> np.ndarray:
        classes = scan_dataset(self._dataset_root)
        if not classes:
            raise SourceError(
                f"No .npy samples found in '{self._dataset_root}'. "
                "Record a dataset first."
            )

        items: list[tuple[str, Path]] = []
        for label, paths in classes.items():
            for path in paths:
                items.append((label, path))
        self._rng.shuffle(items)
        items = items[: self._max_samples]

        # The IMU is continuous in real life.  Concatenating samples with
        # zero-IMU idle would create large gyro/accel steps that the classifier
        # has never seen, so idle segments hold the last sample's IMU value.
        last_imu = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.98], dtype=np.float32)

        def idle_frames(count: int) -> np.ndarray:
            segment = self._rng.normal(
                0.0, self._idle_imu_std, size=(count, N_CHANNELS)
            ).astype(np.float32)
            segment[:, :3] = self._rng.normal(
                0.0, self._idle_emg_std, size=(count, 3)
            )
            segment[:, 3:] += last_imu
            return segment

        segments: list[np.ndarray] = []
        if self._start_idle_seconds > 0:
            start_n = max(1, int(round(self._start_idle_seconds * self._sample_rate)))
            segments.append(idle_frames(start_n))

        for _, path in items:
            sample = np.load(path, allow_pickle=False)
            if sample.ndim != 2 or sample.shape[1] != N_CHANNELS:
                continue
            sample = sample.astype(np.float32, copy=False)
            idle_s = float(self._rng.uniform(*self._idle_seconds))
            idle_n = max(1, int(round(idle_s * self._sample_rate)))
            segments.append(idle_frames(idle_n))
            segments.append(sample)
            last_imu = sample[-1, 3:].copy()

        tail_n = max(1, int(round(0.8 * self._sample_rate)))
        segments.append(idle_frames(tail_n))
        timeline = np.concatenate(segments, axis=0)
        return np.ascontiguousarray(timeline, dtype=np.float32)

    def read_available(self) -> list[Frame]:
        if self._timeline is None or self._t0 is None:
            raise SourceError("Replay source is not open")
        total = len(self._timeline)
        elapsed = time.monotonic() - self._t0
        target = int(elapsed * self._sample_rate)
        if self._cursor >= total:
            if not self._loop:
                return []
            self._t0 = time.monotonic()
            self._cursor = 0
            target = 0
        available = min(target - self._cursor, 4096)
        if available <= 0:
            return []
        end = min(self._cursor + available, total)
        frames = self._timeline[self._cursor : end]
        self._cursor = end
        return [frame.copy() for frame in frames]

    def close(self) -> None:
        self._timeline = None
        self._cursor = 0
        self._t0 = None


def create_source(
    port: str,
    *,
    baudrate: int = DEFAULT_BAUDRATE,
    replay_root: str | Path | None = None,
    sample_rate: float = 500.0,
) -> FrameSource:
    """Build a source from a CLI-style port specification.

    ``replay:<path>`` or an explicit ``replay_root`` selects the replay source.
    """

    if replay_root is not None:
        return ReplaySource(replay_root, sample_rate=sample_rate)
    if port.startswith("replay:"):
        return ReplaySource(port.removeprefix("replay:"), sample_rate=sample_rate)
    return SerialSource(port, baudrate)
