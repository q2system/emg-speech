"""Application 2: real-time inference window (presentation layer only)."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCloseEvent, QTextCursor
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from emg_speech.config import DEFAULT_BAUDRATE, baudrate_warning
from emg_speech.inference import Prediction
from emg_speech.model import CheckpointError, CheckpointSummary, read_checkpoint_summary
from emg_speech.sources import ReplaySource, SerialSource

from .theme import ACCENT, DANGER, INFO, MUTED, TEXT, WARNING
from .widgets import LevelMeter, LiveWaveform, PortSelector
from .workers import InferenceThread, Telemetry


class InferenceWindow(QMainWindow):
    def __init__(
        self,
        *,
        checkpoint_path: str | Path | None = None,
        replay_root: str | Path | None = None,
        initial_port: str | None = None,
        baudrate: int = DEFAULT_BAUDRATE,
        min_confidence: float = 0.5,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._checkpoint_path: Path | None = None
        self._summary: CheckpointSummary | None = None
        self._replay_root = Path(replay_root).expanduser().resolve() if replay_root else None
        self._initial_port = initial_port
        self._initial_baud = baudrate
        self._min_confidence = float(min(max(min_confidence, 0.0), 1.0))

        self._thread: InferenceThread | None = None
        self._streaming = False
        self._stopping = False
        self._last_error: str | None = None
        self._words: list[str] = []

        self._build_ui()
        self._wire_signals()
        self.resize(1280, 820)
        self.setWindowTitle("EMG Speech — Real-time inference")
        if checkpoint_path:
            self._load_checkpoint(Path(checkpoint_path))

    # ------------------------------------------------------------------- UI
    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 14, 16, 10)
        root.setSpacing(12)

        header = QHBoxLayout()
        titles = QVBoxLayout()
        titles.setSpacing(2)
        title = QLabel("EMG Speech")
        title.setObjectName("title")
        subtitle = QLabel(
            f"Replay mode · {self._replay_root}"
            if self._replay_root
            else "Continuous subvocal recognition over UART"
        )
        subtitle.setObjectName("muted")
        titles.addWidget(title)
        titles.addWidget(subtitle)
        header.addLayout(titles, 1)

        self._port_selector = PortSelector()
        self._port_selector.set_baudrate(self._initial_baud)
        header.addWidget(self._port_selector, 0)
        if self._replay_root is not None:
            self._port_selector.hide()
        elif self._initial_port:
            self._port_selector.select_port(self._initial_port)
        root.addLayout(header)

        body = QHBoxLayout()
        body.setSpacing(12)
        body.addWidget(self._build_model_card(), 0)
        body.addWidget(self._build_transcript_card(), 1)
        root.addLayout(body, 1)
        self.setCentralWidget(central)

        self._status_serial = QLabel("serial: offline")
        self._status_frames = QLabel("0 frames")
        self._status_latency = QLabel("inference: —")
        self.statusBar().addPermanentWidget(self._status_serial)
        self.statusBar().addPermanentWidget(self._status_frames)
        self.statusBar().addPermanentWidget(self._status_latency)

    def _build_model_card(self) -> QGroupBox:
        card = QGroupBox("Model & controls")
        card.setFixedWidth(350)
        layout = QVBoxLayout(card)
        layout.setSpacing(9)

        self._model_path = QLineEdit()
        self._model_path.setReadOnly(True)
        self._model_path.setPlaceholderText("No checkpoint selected")
        self._browse_button = QPushButton("Browse…")
        path_row = QHBoxLayout()
        path_row.addWidget(self._model_path, 1)
        path_row.addWidget(self._browse_button)
        layout.addLayout(path_row)

        self._classes_label = QLabel("Classes: —")
        self._classes_label.setWordWrap(True)
        layout.addWidget(self._classes_label)

        self._metrics_label = QLabel("Load a trained checkpoint to begin.")
        self._metrics_label.setWordWrap(True)
        self._metrics_label.setObjectName("muted")
        layout.addWidget(self._metrics_label)

        self._runtime_label = QLabel("")
        self._runtime_label.setWordWrap(True)
        self._runtime_label.setObjectName("muted")
        layout.addWidget(self._runtime_label)

        self._start_button = QPushButton("Start listening")
        self._start_button.setObjectName("primaryButton")
        self._start_button.setMinimumHeight(46)
        self._start_button.setEnabled(False)
        layout.addWidget(self._start_button)

        self._recalibrate_button = QPushButton("Recalibrate noise floor")
        self._recalibrate_button.setEnabled(False)
        layout.addWidget(self._recalibrate_button)

        confidence_box = QGroupBox("Acceptance threshold")
        confidence_layout = QVBoxLayout(confidence_box)
        row = QHBoxLayout()
        row.addWidget(QLabel("Minimum confidence"))
        row.addStretch(1)
        self._confidence_value = QLabel(f"{self._min_confidence:.2f}")
        self._confidence_value.setObjectName("muted")
        row.addWidget(self._confidence_value)
        confidence_layout.addLayout(row)
        self._confidence = QSlider(Qt.Orientation.Horizontal)
        self._confidence.setRange(0, 95)
        self._confidence.setValue(int(round(self._min_confidence * 100)))
        confidence_layout.addWidget(self._confidence)
        hint = QLabel("Predictions below this probability are shown but not typed.")
        hint.setWordWrap(True)
        hint.setObjectName("muted")
        confidence_layout.addWidget(hint)
        layout.addWidget(confidence_box)

        self._calibration = QProgressBar()
        self._calibration.setRange(0, 100)
        self._calibration.setValue(0)
        self._calibration.setTextVisible(True)
        self._calibration.setFormat("calibrating %p%")
        self._calibration.hide()
        layout.addWidget(self._calibration)

        self._level_meter = LevelMeter()
        layout.addWidget(self._level_meter)

        hint = QLabel(
            "Stay silent for the first two seconds after connecting so the detector "
            "can learn the noise floor."
        )
        hint.setWordWrap(True)
        hint.setObjectName("muted")
        layout.addWidget(hint)
        layout.addStretch(1)
        return card

    def _build_transcript_card(self) -> QGroupBox:
        card = QGroupBox("Transcript")
        layout = QVBoxLayout(card)
        layout.setSpacing(8)

        toolbar = QHBoxLayout()
        self._state_label = QLabel("Idle")
        self._state_label.setObjectName("badge")
        toolbar.addWidget(self._state_label)
        self._word_count = QLabel("0 words")
        self._word_count.setObjectName("muted")
        toolbar.addWidget(self._word_count)
        toolbar.addStretch(1)
        self._clear_button = QPushButton("Clear")
        self._copy_button = QPushButton("Copy")
        self._save_button = QPushButton("Save…")
        toolbar.addWidget(self._clear_button)
        toolbar.addWidget(self._copy_button)
        toolbar.addWidget(self._save_button)
        layout.addLayout(toolbar)

        self._transcript = QPlainTextEdit()
        self._transcript.setObjectName("transcript")
        self._transcript.setReadOnly(True)
        self._transcript.setPlaceholderText(
            "Recognised words will appear here as you speak…"
        )
        layout.addWidget(self._transcript, 1)

        self._last_label = QLabel("No prediction yet.")
        self._last_label.setWordWrap(True)
        self._last_label.setMinimumHeight(38)
        self._last_label.setStyleSheet(f"color: {MUTED};")
        layout.addWidget(self._last_label)

        wave_header = QHBoxLayout()
        wave_title = QLabel("Live signal")
        wave_title.setObjectName("muted")
        wave_header.addWidget(wave_title)
        wave_header.addStretch(1)
        self._show_all_signals = QCheckBox("All 9 signals")
        self._show_all_signals.setToolTip(
            "Unchecked: detection envelope (default). Checked: all nine "
            "filtered channels, z-scored per channel for readability. The "
            "envelope threshold remains visible in both views."
        )
        wave_header.addWidget(self._show_all_signals)
        layout.addLayout(wave_header)

        self._waveform = LiveWaveform(capacity=2500)
        self._waveform.setMinimumHeight(170)
        self._show_all_signals.toggled.connect(self._waveform.set_show_channels)
        layout.addWidget(self._waveform, 0)
        return card

    def _wire_signals(self) -> None:
        self._browse_button.clicked.connect(self._choose_checkpoint)
        self._start_button.clicked.connect(self._toggle_stream)
        self._recalibrate_button.clicked.connect(self._recalibrate)
        self._confidence.valueChanged.connect(self._confidence_changed)
        self._clear_button.clicked.connect(self._clear_transcript)
        self._copy_button.clicked.connect(self._copy_transcript)
        self._save_button.clicked.connect(self._save_transcript)
        self._port_selector.selection_changed.connect(lambda _: self._update_buttons())
        self._port_selector.connect_requested.connect(self._connect_from_selector)
        self._port_selector.disconnect_requested.connect(self._stop_from_selector)

    # ------------------------------------------------------------ checkpoint
    def _choose_checkpoint(self) -> None:
        default_dir = str(Path.cwd() / "artifacts")
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select model checkpoint",
            default_dir,
            "PyTorch checkpoint (*.pt *.pth);;All files (*)",
        )
        if path:
            self._load_checkpoint(Path(path))

    def _load_checkpoint(self, path: Path) -> None:
        try:
            summary = read_checkpoint_summary(path)
        except CheckpointError as exc:
            QMessageBox.critical(self, "Invalid checkpoint", str(exc))
            return
        self._checkpoint_path = path
        self._summary = summary
        self._model_path.setText(str(path))
        classes = ", ".join(summary.label_map)
        self._classes_label.setText(
            f"Classes ({len(summary.label_map)}): {classes}"
        )
        metrics = summary.metrics
        pieces: list[str] = []
        if "val_accuracy" in metrics:
            pieces.append(f"validation accuracy {float(metrics['val_accuracy']):.1%}")
        if "val_macro_f1" in metrics:
            pieces.append(f"macro-F1 {float(metrics['val_macro_f1']):.3f}")
        self._metrics_label.setText(" · ".join(pieces) if pieces else "No metrics recorded.")
        self._runtime_label.setText(
            f"{summary.n_parameters:,} parameters · {summary.dsp_config.seq_len} "
            f"steps @ {summary.dsp_config.sample_rate:g} Hz · trained {summary.trained_at[:19]}"
        )
        self.setWindowTitle(f"EMG Speech — {path.name}")
        # Keep a fixed six-second strip chart at the checkpoint's sample rate.
        self._waveform.set_capacity(int(summary.dsp_config.sample_rate * 6))
        self._update_buttons()

    # --------------------------------------------------------------- stream
    def _connect_from_selector(self, _port: str, _baudrate: int) -> None:
        self._start_stream()

    def _stop_from_selector(self) -> None:
        self._stop_stream()

    def _toggle_stream(self) -> None:
        if self._thread is not None:
            self._stop_stream()
        else:
            self._start_stream()

    def _start_stream(self) -> None:
        if self._thread is not None:
            return
        if self._summary is None or self._checkpoint_path is None:
            self._status_message("Select a checkpoint first.", WARNING)
            return
        if self._replay_root is None:
            port = self._port_selector.current_port()
            if not port:
                self._status_message("Select a serial port first.", WARNING)
                return
            baud = self._port_selector.baudrate()
            warning = baudrate_warning(self._summary.dsp_config.sample_rate, baud)
            if warning:
                self._status_message(warning, WARNING)
            factory = lambda: SerialSource(port, baud)  # noqa: E731
            self._port_selector.set_busy(True, "Connecting…")
        else:
            sample_rate = self._summary.dsp_config.sample_rate
            factory = lambda: ReplaySource(self._replay_root, sample_rate=sample_rate)  # noqa: E731

        self._stopping = False
        self._last_error = None
        thread = InferenceThread(
            self._checkpoint_path,
            factory,
            min_confidence=self._min_confidence,
            parent=self,
        )
        thread.model_ready.connect(self._on_model_ready)
        thread.prediction.connect(self._on_prediction)
        thread.telemetry.connect(self._on_telemetry)
        thread.opened.connect(self._on_opened)
        thread.closed.connect(self._on_closed)
        thread.error.connect(self._on_error)
        thread.reconnecting.connect(self._on_reconnecting)
        thread.stats.connect(self._on_stats)
        thread.finished.connect(self._on_thread_finished)
        self._thread = thread
        self._browse_button.setEnabled(False)
        self._recalibrate_button.setEnabled(False)
        self._status_serial.setText("serial: starting")
        self._state_label.setText("Loading model")
        self._set_state_color(INFO)
        thread.start()
        self._update_buttons()

    def _stop_stream(self) -> None:
        if self._thread is None:
            return
        self._stopping = True
        self._thread.stop()
        self._status_message("Stopping…")
        self._update_buttons()

    def _on_model_ready(self, info: dict) -> None:
        self._last_error = None
        self._runtime_label.setText(
            f"{info['n_parameters']:,} parameters · {info['device']} · "
            f"{len(info['label_map'])} classes"
        )
        self._state_label.setText("Ready")
        self._set_state_color(ACCENT)

    def _on_opened(self, description: str) -> None:
        self._streaming = True
        self._last_error = None
        self._calibration.show()
        self._calibration.setValue(0)
        self._recalibrate_button.setEnabled(True)
        if self._replay_root is None:
            self._port_selector.set_busy(False)
            self._port_selector.set_connected(True, description)
        self._status_serial.setText(f"serial: {description}")
        self._state_label.setText("Calibrating")
        self._set_state_color(WARNING)
        self._update_buttons()

    def _on_closed(self) -> None:
        """Idempotent link reset; also used when the thread ends unexpectedly."""

        self._streaming = False
        self._recalibrate_button.setEnabled(False)
        self._calibration.hide()
        if self._replay_root is None:
            self._port_selector.set_connected(False)
            self._port_selector.set_busy(False)
        self._status_serial.setText("serial: offline")
        if self._last_error:
            self._state_label.setText("Error")
            self._set_state_color(DANGER)
        else:
            self._state_label.setText("Idle")
            self._set_state_color(MUTED)
        self._update_buttons()

    def _on_thread_finished(self) -> None:
        thread = self._thread
        self._thread = None
        self._stopping = False
        self._streaming = False
        if thread is not None:
            thread.deleteLater()
        # ``closed`` usually arrives first, but never rely on that: a thread
        # that fails before opening the port must not leave a dead UI behind.
        self._on_closed()
        self._browse_button.setEnabled(True)
        self._update_buttons()

    def _on_error(self, message: str) -> None:
        self._last_error = message
        self._status_message(message, DANGER)
        self._state_label.setText("Error")
        self._set_state_color(DANGER)

    def _on_reconnecting(self, attempt: int) -> None:
        if self._replay_root is None:
            self._port_selector.set_busy(True, "Retrying…")
        self._state_label.setText("Reconnecting")
        self._set_state_color(WARNING)
        self._status_message(f"Serial link lost, attempt {attempt}…")

    def _on_stats(self, parsed: int, malformed: int) -> None:
        if malformed:
            self._status_frames.setToolTip(f"{malformed} malformed UART lines")

    # ------------------------------------------------------------- telemetry
    def _on_telemetry(self, telemetry: Telemetry) -> None:
        self._waveform.push_values(telemetry.levels or (telemetry.level,))
        if telemetry.channels:
            self._waveform.push_channels(telemetry.channels)
        self._waveform.set_threshold(telemetry.threshold)
        self._waveform.set_active(telemetry.state == "active")
        self._level_meter.set_level(
            telemetry.level,
            telemetry.threshold,
            active=telemetry.state == "active",
        )
        self._status_frames.setText(f"{telemetry.frames:,} frames")

        if telemetry.state == "calibrating":
            self._calibration.show()
            self._calibration.setValue(int(telemetry.calibration * 100))
            self._state_label.setText("Calibrating")
            self._set_state_color(WARNING)
        elif telemetry.state == "active":
            self._calibration.hide()
            self._state_label.setText("Speech detected")
            self._set_state_color(DANGER)
        else:
            self._calibration.hide()
            self._state_label.setText("Listening")
            self._set_state_color(ACCENT)

    def _on_prediction(self, prediction: Prediction) -> None:
        self._status_latency.setText(f"inference: {prediction.latency_ms:.1f} ms")
        if prediction.accepted:
            self._append_word(prediction.label)
            self._last_label.setStyleSheet(f"color: {TEXT};")
            self._last_label.setText(
                f"“{prediction.label}” · {prediction.confidence:.0%} confidence · "
                f"{prediction.event_ms:.0f} ms burst · "
                f"{prediction.latency_ms:.1f} ms model latency"
            )
        else:
            self._last_label.setStyleSheet(f"color: {WARNING};")
            self._last_label.setText(
                f"Ignored “{prediction.label}” — confidence "
                f"{prediction.confidence:.0%} is below the "
                f"{self._min_confidence:.0%} acceptance threshold."
            )

    # ------------------------------------------------------------ transcript
    def _append_word(self, word: str) -> None:
        cursor = self._transcript.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self._transcript.setTextCursor(cursor)
        separator = " " if self._transcript.toPlainText() else ""
        self._transcript.insertPlainText(f"{separator}{word}")
        self._transcript.ensureCursorVisible()
        self._words.append(word)
        self._word_count.setText(f"{len(self._words)} words")

    def _clear_transcript(self) -> None:
        self._transcript.clear()
        self._words.clear()
        self._word_count.setText("0 words")
        self._last_label.setText("Transcript cleared.")
        self._last_label.setStyleSheet(f"color: {MUTED};")

    def _copy_transcript(self) -> None:
        QApplication.clipboard().setText(self._transcript.toPlainText())
        self._status_message("Transcript copied to the clipboard.")

    def _save_transcript(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save transcript",
            str(Path.cwd() / "transcript.txt"),
            "Text files (*.txt);;All files (*)",
        )
        if not path:
            return
        try:
            Path(path).write_text(self._transcript.toPlainText() + "\n", encoding="utf-8")
        except OSError as exc:
            QMessageBox.warning(self, "Could not save transcript", str(exc))
            return
        self._status_message(f"Transcript saved to {path}")

    # ------------------------------------------------------------------ misc
    def _recalibrate(self) -> None:
        if self._thread is None:
            return
        self._thread.request_recalibration()
        self._waveform.clear()
        self._calibration.show()
        self._calibration.setValue(0)
        self._state_label.setText("Calibrating")
        self._set_state_color(WARNING)
        self._status_message("Stay silent: relearning the noise floor…")

    def _confidence_changed(self, value: int) -> None:
        self._min_confidence = value / 100.0
        self._confidence_value.setText(f"{self._min_confidence:.2f}")
        if self._thread is not None:
            self._thread.set_min_confidence(self._min_confidence)

    def _update_buttons(self) -> None:
        if self._thread is not None:
            if self._stopping:
                self._start_button.setText("Stopping…")
                self._start_button.setEnabled(False)
            else:
                self._start_button.setText("Stop listening")
                self._start_button.setEnabled(True)
            return
        self._start_button.setText("Start listening")
        can_start = self._summary is not None
        if self._replay_root is None:
            can_start = can_start and bool(self._port_selector.current_port())
        self._start_button.setEnabled(can_start)
        self._recalibrate_button.setEnabled(False)

    def _set_state_color(self, color: str) -> None:
        self._state_label.setStyleSheet(f"color: {color}; border-color: {color};")

    def _status_message(self, message: str, color: str = MUTED) -> None:
        self.statusBar().showMessage(message, 8000)
        self._status_serial.setStyleSheet(f"color: {color};")

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if self._thread is not None:
            self._thread.stop()
            if not self._thread.wait(4000):
                self._thread.terminate()
                self._thread.wait(1000)
            self._thread = None
        super().closeEvent(event)
