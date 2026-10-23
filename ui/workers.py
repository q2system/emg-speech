"""Qt threading layer.  The only place where Qt and the core engines meet.

* :class:`SerialReaderThread` owns the blocking UART read loop.
* :class:`InferenceThread` additionally owns the model + streaming recognizer.
* :class:`ExtractionTask` performs DSP + disk I/O for one take on a thread pool.

The GUI thread never touches a serial port, a filter state or a tensor.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from PyQt6.QtCore import QObject, QRunnable, QThread, pyqtSignal

from emg_speech.config import DatasetMeta, DetectionConfig, DspConfig
from emg_speech.dsp import EventNotFound, NoiseProfile, extract_sample
from emg_speech.inference import StreamingRecognizer
from emg_speech.model import load_checkpoint
from emg_speech.protocol import Frame
from emg_speech.sources import FrameSource, SourceError
from emg_speech.storage import save_sample
from emg_speech.train import resolve_device

SourceFactory = Callable[[], FrameSource]


class WorkerError(RuntimeError):
    """Fatal worker error that should stop the thread."""


@dataclass(frozen=True)
class Telemetry:
    level: float
    threshold: float | None
    state: str
    calibration: float
    frames: int
    # Batches accumulated since the previous telemetry tick. ``levels`` drives
    # the scrolling envelope; ``channels`` contains the filtered 9-channel rows
    # for the "all signals" view.
    levels: tuple[float, ...] = ()
    channels: tuple[tuple[float, ...], ...] = ()


class SerialReaderThread(QThread):
    """Reads frames on a background thread and delivers them in small batches."""

    frames_batch = pyqtSignal(list)
    opened = pyqtSignal(str)
    closed = pyqtSignal()
    error = pyqtSignal(str)
    reconnecting = pyqtSignal(int)
    stats = pyqtSignal(int, int)

    def __init__(
        self,
        source_factory: SourceFactory,
        *,
        batch_interval: float = 0.02,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._factory = source_factory
        self._batch_interval = batch_interval
        self._stop_event = threading.Event()

    # -------------------------------------------------------------- lifecycle
    def stop(self) -> None:
        """Ask the thread to stop.  Call ``wait()`` afterwards."""

        self._stop_event.set()

    def run(self) -> None:  # noqa: D401 (Qt entry point)
        try:
            self._prepare()
        except Exception as exc:
            self.error.emit(str(exc))
            self.closed.emit()
            return

        attempt = 0
        while not self._stop_event.is_set():
            source: FrameSource | None = None
            try:
                source = self._factory()
                source.open()
            except SourceError as exc:
                attempt += 1
                self.error.emit(str(exc))
                if self._stop_event.is_set():
                    break
                self.reconnecting.emit(attempt)
                if self._stop_event.wait(min(3.0, 0.4 * attempt)):
                    break
                continue
            except Exception as exc:  # defensive: never kill the app silently
                self.error.emit(f"Unexpected source error: {exc}")
                break

            attempt = 0
            self.opened.emit(source.description)
            try:
                self._read_loop(source)
            except SourceError as exc:
                if not self._stop_event.is_set():
                    self.error.emit(str(exc))
                    self.reconnecting.emit(1)
            except Exception as exc:
                if not self._stop_event.is_set():
                    self.error.emit(f"Reader crashed: {exc}")
            finally:
                source.close()
                self._emit_stats(source)

            if not self._stop_event.is_set():
                self._stop_event.wait(1.0)

        try:
            self._shutdown_pipeline()
        finally:
            self.closed.emit()

    # ------------------------------------------------------------- overridable
    def _prepare(self) -> None:
        """Load models etc. before the source is opened."""

    def _consume(self, frames: list[Frame]) -> None:
        self.frames_batch.emit(frames)

    def _shutdown_pipeline(self) -> None:
        """Flush pending work after the read loop ends."""

    # ----------------------------------------------------------------- internals
    def _read_loop(self, source: FrameSource) -> None:
        pending: list[Frame] = []
        last_batch = time.monotonic()
        last_stats = last_batch

        while not self._stop_event.is_set():
            frames = source.read_available()
            if frames:
                pending.extend(frames)
            now = time.monotonic()
            if pending and (
                now - last_batch >= self._batch_interval or len(pending) >= 256
            ):
                batch = pending
                pending = []
                last_batch = now
                self._consume(batch)
            if now - last_stats >= 0.5:
                last_stats = now
                self._emit_stats(source)
            if not frames:
                time.sleep(0.001)

        if pending:
            self._consume(pending)

    def _emit_stats(self, source: FrameSource) -> None:
        stats = getattr(source, "stats", None)
        parsed = int(getattr(stats, "parsed", 0))
        malformed = int(getattr(stats, "malformed", 0))
        self.stats.emit(parsed, malformed)


class InferenceThread(SerialReaderThread):
    """Serial reader + streaming recognition, entirely off the GUI thread."""

    model_ready = pyqtSignal(object)
    prediction = pyqtSignal(object)
    telemetry = pyqtSignal(object)

    def __init__(
        self,
        checkpoint_path: str | Path,
        source_factory: SourceFactory,
        *,
        min_confidence: float = 0.5,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(source_factory, batch_interval=0.02, parent=parent)
        self._checkpoint_path = Path(checkpoint_path)
        self._min_confidence = float(min_confidence)
        self._lock = threading.Lock()
        self._recalibrate = threading.Event()
        self._recognizer: StreamingRecognizer | None = None
        self._last_telemetry = 0.0
        self._pending_levels: list[float] = []
        self._pending_channels: list[tuple[float, ...]] = []

    def _prepare(self) -> None:
        device = resolve_device("auto")
        info = load_checkpoint(self._checkpoint_path, device=device)
        self._recognizer = StreamingRecognizer(
            info,
            min_confidence=self.min_confidence,
            device=device,
        )
        self.model_ready.emit(
            {
                "label_map": info.label_map,
                "metrics": info.metrics,
                "trained_at": info.trained_at,
                "model_config": info.model_config,
                "dsp_config": info.dsp_config,
                "n_parameters": info.model.n_parameters,
                "device": str(device),
                "path": str(info.path),
            }
        )

    @property
    def min_confidence(self) -> float:
        with self._lock:
            return self._min_confidence

    def set_min_confidence(self, value: float) -> None:
        with self._lock:
            self._min_confidence = float(min(max(0.0, value), 1.0))
        if self._recognizer is not None:
            self._recognizer.min_confidence = self._min_confidence

    def request_recalibration(self) -> None:
        self._recalibrate.set()

    def _consume(self, frames: list[Frame]) -> None:
        recognizer = self._recognizer
        if recognizer is None:
            return
        if self._recalibrate.is_set():
            self._recalibrate.clear()
            recognizer.reset()
            self._pending_levels.clear()
            self._pending_channels.clear()
            self.telemetry.emit(
                Telemetry(0.0, None, recognizer.state.value, 0.0, recognizer.frames_seen)
            )
            return
        recognizer.min_confidence = self.min_confidence
        for frame in frames:
            prediction = recognizer.push(frame)
            if prediction is not None:
                self.prediction.emit(prediction)
            self._pending_levels.append(recognizer.level)
            filtered = recognizer.last_filtered
            if filtered is not None:
                self._pending_channels.append(tuple(float(v) for v in filtered))
        now = time.monotonic()
        if now - self._last_telemetry >= 0.05:
            self._last_telemetry = now
            levels = tuple(self._pending_levels)
            channels = tuple(self._pending_channels)
            self._pending_levels.clear()
            self._pending_channels.clear()
            self.telemetry.emit(
                Telemetry(
                    level=recognizer.level,
                    threshold=recognizer.on_threshold,
                    state=recognizer.state.value,
                    calibration=recognizer.calibration_progress,
                    frames=recognizer.frames_seen,
                    levels=levels,
                    channels=channels,
                )
            )

    def _shutdown_pipeline(self) -> None:
        if self._recognizer is None:
            return
        prediction = self._recognizer.flush()
        if prediction is not None:
            self.prediction.emit(prediction)


class ExtractionSignals(QObject):
    completed = pyqtSignal(object, str, str)  # ExtractedSample, path, class name
    failed = pyqtSignal(str)


class ExtractionTask(QRunnable):
    """DSP + save for one button-press take, executed on the global pool."""

    def __init__(
        self,
        take: NDArray[np.floating],
        *,
        dataset_root: str | Path,
        class_name: str,
        meta: DatasetMeta,
        dsp_config: DspConfig,
        detection_config: DetectionConfig,
        noise_profile: NoiseProfile | None,
    ) -> None:
        super().__init__()
        self.signals = ExtractionSignals()
        self._take = take
        self._dataset_root = Path(dataset_root)
        self._class_name = class_name
        self._meta = meta
        self._dsp = dsp_config
        self._detection = detection_config
        self._noise_profile = noise_profile

    def run(self) -> None:
        try:
            extracted = extract_sample(
                self._take,
                self._dsp,
                self._detection,
                self._noise_profile,
            )
            path = save_sample(
                self._dataset_root,
                self._class_name,
                extracted.data,
                self._meta,
            )
        except EventNotFound as exc:
            self.signals.failed.emit(str(exc))
        except Exception as exc:  # keep the GUI alive on any failure
            self.signals.failed.emit(f"Could not process take: {exc}")
        else:
            self.signals.completed.emit(extracted, str(path), self._class_name)
