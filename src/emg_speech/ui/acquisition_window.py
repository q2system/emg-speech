"""Application 1: acquisition and annotation window (presentation layer only)."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

from PyQt6.QtCore import QSignalBlocker, Qt, QThreadPool, QTimer, QUrl
from PyQt6.QtGui import QCloseEvent, QDesktopServices, QKeyEvent
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from emg_speech.config import (
    DEFAULT_BAUDRATE,
    DETECTION_MAX_EVENT_MS_RANGE,
    DETECTION_MIN_EVENT_MS_RANGE,
    DETECTION_ON_SIGMA_RANGE,
    DETECTION_POST_MARGIN_MS_RANGE,
    MAX_SAMPLE_RATE,
    MAX_SEQ_LEN,
    MIN_SAMPLE_RATE,
    MIN_SEQ_LEN,
    DatasetMeta,
    DetectionConfig,
    DspConfig,
    ProjectConfig,
    baudrate_warning,
    ensure_project_config_matches,
)
from emg_speech.constants import CHANNELS
from emg_speech.dsp import NoiseProfile, StreamingPipeline
from emg_speech.ring_buffer import FrameRingBuffer
from emg_speech.sources import SerialSource
from emg_speech.storage import (
    ClassNameError,
    ClassNotEmptyError,
    DatasetError,
    ensure_class,
    ensure_dataset_root,
    list_classes,
    remove_class,
    remove_sample,
)

from .theme import ACCENT, DANGER, INFO, MUTED, WARNING
from .widgets import LevelMeter, LiveWaveform, PortSelector, StaticWaveform
from .workers import ExtractionTask, SerialReaderThread


class AcquisitionWindow(QMainWindow):
    MAX_TAKE_MS = 8000
    TAIL_MS = 250
    MIN_TAKE_MS = 200

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        initial_port: str | None = None,
        baudrate: int = DEFAULT_BAUDRATE,
        project_config: ProjectConfig | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._dataset_root = ensure_dataset_root(dataset_root)
        self._initial_port = initial_port
        self._initial_baud = baudrate
        self._project_config = project_config

        self._meta: DatasetMeta | None = None
        self._base_dsp = project_config.dsp if project_config else DspConfig()
        self._base_detection = (
            project_config.detection if project_config else DetectionConfig()
        )
        self._config_warning: str | None = None
        self._profile: NoiseProfile | None = None
        self._settings_locked = False
        self._loading = True

        self._thread: SerialReaderThread | None = None
        self._streaming = False
        self._stopping = False
        self._recording = False
        self._pending = False
        self._anchor = 0
        self._release_index = 0
        self._take_class = ""
        self._take_started = 0.0
        self._last_level = 0.0
        self._session_frames = 0
        self._last_saved_path: Path | None = None
        self._pending_task: ExtractionTask | None = None
        self._last_error: str | None = None

        self._buffer = FrameRingBuffer(6000)
        self._live_pipeline = StreamingPipeline(self._base_dsp)

        self._tail_timer = QTimer(self)
        self._tail_timer.setSingleShot(True)
        self._tail_timer.timeout.connect(self._submit_take)
        self._max_timer = QTimer(self)
        self._max_timer.setSingleShot(True)
        self._max_timer.timeout.connect(lambda: self._end_take(auto=True))
        self._ui_timer = QTimer(self)
        self._ui_timer.setInterval(33)
        self._ui_timer.timeout.connect(self._refresh_live_ui)

        self._build_ui()
        self._load_dataset_state()
        self._wire_signals()
        self._loading = False
        self._refresh_classes()
        self._reset_stream_state()
        self._ui_timer.start()
        self.resize(1340, 840)
        self.setWindowTitle(f"EMG Acquisition — {self._dataset_root}")

    # ------------------------------------------------------------------- UI
    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 14, 16, 10)
        root.setSpacing(12)

        header = QHBoxLayout()
        titles = QVBoxLayout()
        titles.setSpacing(2)
        title = QLabel("EMG Acquisition")
        title.setObjectName("title")
        subtitle = QLabel(str(self._dataset_root))
        subtitle.setObjectName("muted")
        titles.addWidget(title)
        titles.addWidget(subtitle)
        header.addLayout(titles, 1)
        self._port_selector = PortSelector()
        if self._initial_port:
            self._port_selector.select_port(self._initial_port)
        self._port_selector.set_baudrate(self._initial_baud)
        header.addWidget(self._port_selector, 0)
        root.addLayout(header)

        body = QHBoxLayout()
        body.setSpacing(12)
        body.addWidget(self._build_dataset_card(), 0)
        body.addWidget(self._build_live_card(), 1)
        body.addWidget(self._build_control_card(), 0)
        root.addLayout(body, 1)
        self.setCentralWidget(central)

        self._serial_status = QLabel("serial: offline")
        self._rate_status = QLabel("0 frames")
        self._error_status = QLabel("0 malformed lines")
        self.statusBar().addPermanentWidget(self._serial_status)
        self.statusBar().addPermanentWidget(self._rate_status)
        self.statusBar().addPermanentWidget(self._error_status)

    def _build_dataset_card(self) -> QGroupBox:
        card = QGroupBox("Dataset & classes")
        card.setFixedWidth(280)
        layout = QVBoxLayout(card)
        layout.setSpacing(8)

        self._class_list = QListWidget()
        self._class_list.setAlternatingRowColors(False)
        layout.addWidget(self._class_list, 1)

        add_row = QHBoxLayout()
        self._new_class = QLineEdit()
        self._new_class.setPlaceholderText("new word, e.g. hello")
        add_row.addWidget(self._new_class, 1)
        self._add_button = QPushButton("Add")
        self._add_button.setToolTip("Create a folder for a new class")
        add_row.addWidget(self._add_button)
        layout.addLayout(add_row)

        action_row = QHBoxLayout()
        self._delete_class_button = QPushButton("Delete empty")
        self._delete_class_button.setObjectName("dangerButton")
        self._open_folder_button = QPushButton("Open folder")
        action_row.addWidget(self._delete_class_button)
        action_row.addWidget(self._open_folder_button)
        layout.addLayout(action_row)

        self._class_total = QLabel("0 classes · 0 samples")
        self._class_total.setObjectName("muted")
        layout.addWidget(self._class_total)

        hint = QLabel(
            "Tip: keep 20–40 clean takes per word. Empty folders never reach training."
        )
        hint.setWordWrap(True)
        hint.setObjectName("muted")
        layout.addWidget(hint)
        return card

    def _build_live_card(self) -> QGroupBox:
        card = QGroupBox("Live signal")
        layout = QVBoxLayout(card)
        layout.setSpacing(8)

        self._waveform = LiveWaveform()
        layout.addWidget(self._waveform, 3)
        self._level_meter = LevelMeter()
        layout.addWidget(self._level_meter)

        state_row = QHBoxLayout()
        self._state_label = QLabel("Offline")
        self._state_label.setObjectName("badge")
        self._state_hint = QLabel("Connect the ESP32 to start.")
        self._state_hint.setObjectName("muted")
        state_row.addWidget(self._state_label)
        state_row.addWidget(self._state_hint, 1)
        layout.addLayout(state_row)

        preview_row = QHBoxLayout()
        preview_row.setSpacing(8)
        preview_title = QLabel("Last accepted take")
        preview_title.setObjectName("muted")
        preview_row.addWidget(preview_title)
        preview_row.addStretch(1)
        self._preview_emg_only = QCheckBox("EMG only")
        self._preview_emg_only.setToolTip(
            "Show only EMG1–EMG3 (default). Untick to see all 9 channels; "
            "the all-channel view z-scores each trace for readability."
        )
        self._preview_emg_only.setChecked(True)
        preview_row.addWidget(self._preview_emg_only)
        layout.addLayout(preview_row)
        self._preview = StaticWaveform()
        self._preview.set_emg_only(self._preview_emg_only.isChecked())
        self._preview_emg_only.toggled.connect(self._preview.set_emg_only)
        layout.addWidget(self._preview, 2)

        self._take_info = QLabel("No take processed yet.")
        self._take_info.setWordWrap(True)
        layout.addWidget(self._take_info)
        self._warning_info = QLabel("")
        self._warning_info.setWordWrap(True)
        self._warning_info.setStyleSheet(f"color: {WARNING};")
        self._warning_info.hide()
        layout.addWidget(self._warning_info)
        return card

    def _build_control_card(self) -> QGroupBox:
        card = QGroupBox("Capture")
        card.setFixedWidth(315)
        layout = QVBoxLayout(card)
        layout.setSpacing(10)

        settings = QGroupBox("Signal shape")
        form = QFormLayout(settings)
        self._fs = QDoubleSpinBox()
        self._fs.setRange(MIN_SAMPLE_RATE, MAX_SAMPLE_RATE)
        self._fs.setSingleStep(50.0)
        self._fs.setDecimals(0)
        self._fs.setSuffix(" Hz")
        self._fs.setValue(self._base_dsp.sample_rate)
        self._fs.setToolTip("Must match the firmware sampling/transmission rate")
        self._seq_len = QSpinBox()
        self._seq_len.setRange(MIN_SEQ_LEN, MAX_SEQ_LEN)
        self._seq_len.setSingleStep(16)
        self._seq_len.setValue(self._base_dsp.seq_len)
        self._seq_len.setToolTip("Every take is resampled to exactly this many time steps")
        form.addRow("Sample rate", self._fs)
        form.addRow("Sequence length", self._seq_len)
        layout.addWidget(settings)

        detection = QGroupBox("Event detection")
        det_form = QFormLayout(detection)
        self._on_sigma = QDoubleSpinBox()
        self._on_sigma.setRange(*DETECTION_ON_SIGMA_RANGE)
        self._on_sigma.setSingleStep(0.5)
        self._on_sigma.setValue(4.0)
        self._on_sigma.setToolTip(
            "Onset threshold = noise floor + N × noise standard deviation"
        )
        self._min_event = QDoubleSpinBox()
        self._min_event.setRange(*DETECTION_MIN_EVENT_MS_RANGE)
        self._min_event.setSingleStep(25.0)
        self._min_event.setSuffix(" ms")
        self._min_event.setValue(150.0)
        self._max_event = QDoubleSpinBox()
        self._max_event.setRange(*DETECTION_MAX_EVENT_MS_RANGE)
        self._max_event.setSingleStep(100.0)
        self._max_event.setSuffix(" ms")
        self._max_event.setValue(1600.0)
        self._post_margin = QDoubleSpinBox()
        self._post_margin.setRange(*DETECTION_POST_MARGIN_MS_RANGE)
        self._post_margin.setSingleStep(20.0)
        self._post_margin.setSuffix(" ms")
        self._post_margin.setValue(100.0)
        det_form.addRow("Onset σ", self._on_sigma)
        det_form.addRow("Min event", self._min_event)
        det_form.addRow("Max event", self._max_event)
        det_form.addRow("Post margin", self._post_margin)
        layout.addWidget(detection)

        self._record_button = QPushButton("Hold to record")
        self._record_button.setObjectName("recordButton")
        self._record_button.setMinimumHeight(96)
        self._record_button.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        layout.addWidget(self._record_button)

        hint = QLabel(
            "Hold the button (or Space) while you speak the word, then release. "
            "Only the detected muscle burst is kept, padded and resampled."
        )
        hint.setWordWrap(True)
        hint.setObjectName("muted")
        layout.addWidget(hint)

        self._delete_take_button = QPushButton("Delete last saved take")
        self._delete_take_button.setObjectName("dangerButton")
        self._delete_take_button.setEnabled(False)
        layout.addWidget(self._delete_take_button)
        layout.addStretch(1)

        for button in (
            self._add_button,
            self._delete_class_button,
            self._open_folder_button,
            self._record_button,
            self._delete_take_button,
        ):
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._class_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        return card

    def _wire_signals(self) -> None:
        self._add_button.clicked.connect(self._add_class)
        self._new_class.returnPressed.connect(self._add_class)
        self._delete_class_button.clicked.connect(self._delete_class)
        self._open_folder_button.clicked.connect(self._open_folder)
        self._class_list.currentItemChanged.connect(self._on_class_changed)
        self._delete_take_button.clicked.connect(self._delete_last_take)

        self._port_selector.connect_requested.connect(self._connect_serial)
        self._port_selector.disconnect_requested.connect(self._disconnect_serial)

        self._record_button.pressed.connect(self._begin_take)
        self._record_button.released.connect(self._end_take)

        self._fs.valueChanged.connect(self._on_shape_changed)
        self._seq_len.valueChanged.connect(self._on_shape_changed)
        self._on_sigma.valueChanged.connect(self._on_detection_changed)
        self._min_event.valueChanged.connect(self._on_detection_changed)
        self._max_event.valueChanged.connect(self._on_detection_changed)
        self._post_margin.valueChanged.connect(self._on_detection_changed)

    # -------------------------------------------------------- dataset state
    def _load_dataset_state(self) -> None:
        meta = DatasetMeta.load(self._dataset_root)
        classes = list_classes(self._dataset_root)
        has_samples = any(info.count for info in classes)

        if meta is not None:
            self._meta = meta
            self._base_dsp = meta.dsp_config
            self._base_detection = meta.detection_config

        if not has_samples and self._project_config is not None:
            # No takes exist yet, so the project config may define the new
            # dataset contract.  Persist it now so dataset.json is consistent
            # even before the first recording.
            self._base_dsp = self._project_config.dsp
            self._base_detection = self._project_config.detection
            self._meta = DatasetMeta.from_configs(
                self._base_dsp, self._base_detection
            )
            self._meta.save(self._dataset_root)
        elif has_samples and self._project_config is not None:
            # Once samples exist their metadata is authoritative.  Explain the
            # conflict rather than silently changing the signal shape.
            try:
                ensure_project_config_matches(
                    self._project_config,
                    sample_rate=self._base_dsp.sample_rate,
                    seq_len=self._base_dsp.seq_len,
                    artifact=f"dataset '{self._dataset_root}'",
                )
            except ValueError as exc:
                self._config_warning = str(exc)

        blockers = [
            QSignalBlocker(self._fs),
            QSignalBlocker(self._seq_len),
            QSignalBlocker(self._on_sigma),
            QSignalBlocker(self._min_event),
            QSignalBlocker(self._max_event),
            QSignalBlocker(self._post_margin),
        ]
        self._fs.setValue(self._base_dsp.sample_rate)
        self._seq_len.setValue(self._base_dsp.seq_len)
        self._on_sigma.setValue(self._base_detection.on_sigma)
        self._min_event.setValue(self._base_detection.min_event_ms)
        self._max_event.setValue(self._base_detection.max_event_ms)
        self._post_margin.setValue(self._base_detection.post_margin_ms)
        del blockers

        self._apply_settings_lock(has_samples)

    def _apply_settings_lock(self, has_samples: bool | None = None) -> None:
        """Lock sample rate / sequence length once data exists, unlock when empty."""

        if has_samples is None:
            has_samples = any(
                info.count for info in list_classes(self._dataset_root)
            )
        self._settings_locked = has_samples
        enabled = not has_samples and not self._recording
        self._fs.setEnabled(enabled)
        self._seq_len.setEnabled(enabled)
        self._fs.setToolTip(
            "Locked: the dataset already contains samples"
            if has_samples
            else "Must match the firmware sampling/transmission rate"
        )
        self._seq_len.setToolTip(
            "Locked: the dataset already contains samples"
            if has_samples
            else "Every take is resampled to exactly this many time steps"
        )

    def _current_dsp_config(self) -> DspConfig:
        return replace(
            self._base_dsp,
            sample_rate=float(self._fs.value()),
            seq_len=int(self._seq_len.value()),
        )

    def _current_detection_config(self) -> DetectionConfig:
        return replace(
            self._base_detection,
            on_sigma=float(self._on_sigma.value()),
            min_event_ms=float(self._min_event.value()),
            max_event_ms=float(self._max_event.value()),
            post_margin_ms=float(self._post_margin.value()),
        )

    def _on_shape_changed(self) -> None:
        if self._loading or self._settings_locked:
            return
        try:
            dsp = self._current_dsp_config()
            detection = self._current_detection_config()
        except ValueError as exc:
            self._set_state("Invalid settings", DANGER)
            self._state_hint.setText(str(exc))
            return
        self._base_dsp = dsp
        self._meta = DatasetMeta.from_configs(dsp, detection)
        self._meta.save(self._dataset_root)
        self._reset_stream_state()
        self._set_state("Ready", ACCENT)
        self._state_hint.setText(
            f"Dataset locked to {dsp.sample_rate:g} Hz · {dsp.seq_len} time steps"
        )

    def _on_detection_changed(self) -> None:
        if self._loading:
            return
        try:
            detection = self._current_detection_config()
        except ValueError as exc:
            self._set_state("Invalid detection settings", DANGER)
            self._state_hint.setText(str(exc))
            return
        self._base_detection = detection
        if not self._settings_locked:
            dsp = self._current_dsp_config()
            self._meta = DatasetMeta.from_configs(dsp, detection)
            self._meta.save(self._dataset_root)

    def _reset_stream_state(self) -> None:
        try:
            dsp = self._current_dsp_config()
        except ValueError:
            return
        frames = max(1000, int(dsp.sample_rate * 12))
        self._buffer = FrameRingBuffer(frames)
        self._live_pipeline = StreamingPipeline(dsp)
        self._session_frames = 0
        self._last_level = 0.0
        self._waveform.clear()
        self._preview.clear()
        if self._config_warning:
            self._warning_info.setText(self._config_warning)
            self._warning_info.show()
        else:
            self._warning_info.hide()
        self._set_state("Offline" if not self._streaming else "Ready", MUTED)

    # -------------------------------------------------------------- classes
    def _refresh_classes(self, select: str | None = None) -> None:
        current = select or self._selected_class()
        self._class_list.blockSignals(True)
        self._class_list.clear()
        classes = list_classes(self._dataset_root)
        for info in classes:
            item = QListWidgetItem(f"{info.name}  ·  {info.count}")
            item.setData(Qt.ItemDataRole.UserRole, info.name)
            self._class_list.addItem(item)
            if info.name == current:
                self._class_list.setCurrentItem(item)
        self._class_list.blockSignals(False)
        total = sum(info.count for info in classes)
        self._class_total.setText(f"{len(classes)} classes · {total} samples")
        if classes and self._class_list.currentRow() < 0:
            self._class_list.setCurrentRow(0)
        self._on_class_changed()

    def _selected_class(self) -> str | None:
        item = self._class_list.currentItem()
        if item is None:
            return None
        value = item.data(Qt.ItemDataRole.UserRole)
        return str(value) if value else None

    def _add_class(self) -> None:
        raw = self._new_class.text().strip()
        if not raw:
            return
        try:
            path = ensure_class(self._dataset_root, raw)
        except (ClassNameError, DatasetError) as exc:
            self._set_state("Invalid class name", DANGER)
            self._state_hint.setText(str(exc))
            return
        self._new_class.clear()
        self._refresh_classes(select=path.name)
        self._set_state("Class ready", ACCENT)
        self._state_hint.setText(f"Record takes for '{path.name}'.")
        self._update_record_enabled()

    def _delete_class(self) -> None:
        name = self._selected_class()
        if not name:
            return
        try:
            remove_class(self._dataset_root, name)
        except ClassNotEmptyError as exc:
            QMessageBox.warning(self, "Class still contains data", str(exc))
            return
        except DatasetError as exc:
            QMessageBox.warning(self, "Cannot delete class", str(exc))
            return
        self._refresh_classes()
        self._apply_settings_lock()
        self._set_state("Class deleted", MUTED)
        self._state_hint.setText(f"Removed empty folder '{name}'.")
        self._update_record_enabled()

    def _open_folder(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._dataset_root)))

    def _on_class_changed(self) -> None:
        self._update_record_enabled()

    # --------------------------------------------------------------- serial
    def _connect_serial(self, port: str, baudrate: int) -> None:
        if self._thread is not None:
            return
        try:
            sample_rate = self._current_dsp_config().sample_rate
        except ValueError:
            sample_rate = self._base_dsp.sample_rate
        warning = baudrate_warning(sample_rate, baudrate)
        if warning:
            self._warning_info.setText(warning)
            self._warning_info.show()
        self._stopping = False
        self._last_error = None
        self._port_selector.set_connected(True, f"{port} @ {baudrate} baud")
        self._port_selector.set_busy(True, "Connecting…")
        self._serial_status.setText(f"serial: connecting to {port}")
        thread = SerialReaderThread(
            lambda: SerialSource(port, baudrate), parent=self
        )
        thread.frames_batch.connect(self._on_frames)
        thread.opened.connect(self._on_serial_opened)
        thread.error.connect(self._on_serial_error)
        thread.reconnecting.connect(self._on_reconnecting)
        thread.closed.connect(self._on_serial_closed)
        thread.finished.connect(self._on_thread_finished)
        thread.stats.connect(self._on_serial_stats)
        self._thread = thread
        thread.start()

    def _disconnect_serial(self) -> None:
        if self._thread is None:
            return
        self._stopping = True
        self._port_selector.set_busy(True, "Disconnecting…")
        self._serial_status.setText("serial: closing")
        self._update_record_enabled()
        self._thread.stop()

    def _on_serial_opened(self, description: str) -> None:
        self._streaming = True
        self._stopping = False
        self._last_error = None
        self._port_selector.set_busy(False)
        self._port_selector.set_connected(True, description)
        self._buffer.clear()
        self._live_pipeline.reset()
        self._waveform.clear()
        self._session_frames = 0
        self._serial_status.setText(f"serial: {description}")
        self._set_state("Listening", ACCENT)
        self._state_hint.setText("Press and hold to capture one word.")
        self._update_record_enabled()

    def _on_serial_error(self, message: str) -> None:
        self._last_error = message
        self._serial_status.setText("serial: error")
        self._error_status.setText(message)
        self._port_selector.set_busy(False)
        self._set_state("Error", DANGER)
        self._state_hint.setText(message)
        self._update_record_enabled()

    def _on_reconnecting(self, attempt: int) -> None:
        self._streaming = False
        self._set_state("Reconnecting", WARNING)
        self._state_hint.setText(
            f"Serial link lost. Attempt {attempt}… reconnect the device or press Disconnect."
        )
        self._update_record_enabled()

    def _on_serial_closed(self) -> None:
        """Idempotent link reset; also used when the thread ends unexpectedly."""

        self._streaming = False
        self._recording = False
        self._pending = False
        self._max_timer.stop()
        self._tail_timer.stop()
        self._port_selector.set_busy(False)
        self._port_selector.set_connected(False)
        self._serial_status.setText("serial: offline")
        if self._last_error:
            self._set_state("Error", DANGER)
            self._state_hint.setText(self._last_error)
        else:
            self._set_state("Offline", MUTED)
            self._state_hint.setText("Connect the ESP32 to start.")
        self._update_record_enabled()

    def _on_thread_finished(self) -> None:
        thread = self._thread
        self._thread = None
        stopping = self._stopping
        self._stopping = False
        if thread is not None:
            thread.deleteLater()
        # ``closed`` may already have been delivered; make sure a thread that
        # dies before emitting it (or after the window started closing) can
        # never leave the controls stuck in "connected".
        if self._streaming or self._recording or self._pending or stopping:
            self._on_serial_closed()
        self._port_selector.set_busy(False)

    def _on_serial_stats(self, parsed: int, malformed: int) -> None:
        self._error_status.setText(f"{malformed} malformed lines")
        self._rate_status.setText(f"{self._session_frames} frames")

    def _on_frames(self, frames: list) -> None:
        if not frames:
            return
        self._buffer.extend(frames)
        self._session_frames += len(frames)
        for frame in frames:
            _, envelope = self._live_pipeline.process_frame(frame)
            self._last_level = envelope
            self._waveform.push_values((envelope,))

    # --------------------------------------------------------------- taking
    def _begin_take(self) -> None:
        if self._loading or not self._streaming:
            return
        if self._recording or self._pending:
            return
        class_name = self._selected_class()
        if not class_name:
            self._set_state("Select a class", WARNING)
            self._state_hint.setText("Add or select a word on the left before recording.")
            return
        try:
            self._current_detection_config()
        except ValueError as exc:
            self._set_state("Invalid detection settings", DANGER)
            self._state_hint.setText(str(exc))
            return

        self._take_class = class_name
        self._anchor = self._buffer.next_index
        self._recording = True
        self._take_started = time.monotonic()
        self._tail_timer.stop()
        self._max_timer.start(self.MAX_TAKE_MS)
        self._record_button.setText("Release to stop")
        self._set_recording_style(True)
        self._set_state("Recording", DANGER)
        self._state_hint.setText(
            f"Speak '{class_name}' now, then release. Capturing up to "
            f"{self.MAX_TAKE_MS / 1000:.0f} s."
        )
        self._update_record_enabled()
        self._set_controls_enabled(False)

    def _end_take(self, *, auto: bool = False) -> None:
        if not self._recording:
            return
        self._recording = False
        self._max_timer.stop()
        self._release_index = self._buffer.next_index
        self._pending = True
        self._record_button.setText("Processing…")
        self._set_recording_style(False)
        self._tail_timer.start(self.TAIL_MS)
        self._set_state("Capturing tail", WARNING)
        suffix = " (maximum length reached)" if auto else ""
        self._state_hint.setText(
            f"Waiting {self.TAIL_MS} ms for the gesture to settle{suffix}…"
        )
        self._update_record_enabled()
        self._set_controls_enabled(True)

    def _submit_take(self) -> None:
        if not self._pending:
            return
        self._pending = False
        try:
            dsp = self._current_dsp_config()
            detection = self._current_detection_config()
        except ValueError as exc:
            self._on_take_failed(str(exc))
            return

        pre_roll = int(round(detection.pre_roll_ms / 1000.0 * dsp.sample_rate))
        tail = int(round(self.TAIL_MS / 1000.0 * dsp.sample_rate))
        start = max(0, self._anchor - pre_roll)
        end = self._release_index + tail
        take, _ = self._buffer.get_range(start, end)
        minimum = int(self.MIN_TAKE_MS / 1000.0 * dsp.sample_rate)
        if len(take) < minimum:
            self._on_take_failed(
                "Take is too short. Hold the button a little longer while speaking."
            )
            return

        meta = self._meta or DatasetMeta.from_configs(dsp, detection)
        self._meta = meta
        task = ExtractionTask(
            take,
            dataset_root=self._dataset_root,
            class_name=self._take_class,
            meta=meta,
            dsp_config=dsp,
            detection_config=detection,
            # Calibrate on this take's own pre-roll: a profile measured minutes
            # ago is stale after the electrodes settle or the subject moves.
            noise_profile=None,
        )
        task.signals.completed.connect(self._on_take_completed)
        task.signals.failed.connect(self._on_take_failed)
        self._pending_task = task
        self._set_state("Processing", INFO)
        self._state_hint.setText("Detecting the muscle burst and normalising length…")
        QThreadPool.globalInstance().start(task)

    def _on_take_completed(self, extracted, path: str, class_name: str) -> None:
        self._pending_task = None
        self._profile = extracted.noise_profile or self._profile
        labels = [
            name.upper() if name.startswith("emg") else name.capitalize()
            for name in CHANNELS
        ]
        self._preview.set_series(
            [
                extracted.data[:, index]
                for index in range(extracted.data.shape[1])
            ],
            labels[: extracted.data.shape[1]],
        )
        self._preview.set_emg_only(self._preview_emg_only.isChecked())
        self._take_info.setText(
            f"'{class_name}': {extracted.event_ms:.0f} ms burst, "
            f"{extracted.data.shape[0]} steps, peak {extracted.peak_envelope:.3g}. "
            f"Saved as {Path(path).name}"
        )
        if extracted.warnings:
            self._warning_info.setText(" · ".join(extracted.warnings))
            self._warning_info.show()
        else:
            self._warning_info.hide()
        self._last_saved_path = Path(path)
        self._delete_take_button.setEnabled(True)
        self._settings_locked = True
        self._fs.setEnabled(False)
        self._seq_len.setEnabled(False)
        self._refresh_classes(select=class_name)
        self._record_button.setText("Hold to record")
        self._apply_settings_lock()
        self._set_state("Saved", ACCENT)
        count = next(
            (info.count for info in list_classes(self._dataset_root) if info.name == class_name),
            0,
        )
        self._state_hint.setText(f"'{class_name}' now has {count} take(s).")
        self._update_record_enabled()

    def _on_take_failed(self, message: str) -> None:
        self._pending_task = None
        self._record_button.setText("Hold to record")
        self._take_info.setText(message)
        self._warning_info.hide()
        self._set_state("Take rejected", WARNING)
        self._state_hint.setText("Nothing was saved. Adjust the detection settings if this repeats.")
        self._set_recording_style(False)
        self._update_record_enabled()

    def _delete_last_take(self) -> None:
        if self._last_saved_path is None:
            return
        answer = QMessageBox.question(
            self,
            "Delete last take",
            f"Delete '{self._last_saved_path.name}' permanently?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            remove_sample(self._last_saved_path)
        except OSError as exc:
            QMessageBox.warning(self, "Could not delete", str(exc))
            return
        self._last_saved_path = None
        self._delete_take_button.setEnabled(False)
        self._refresh_classes()
        self._apply_settings_lock()
        self._take_info.setText("Last take deleted.")
        self._set_state("Deleted", MUTED)

    # ----------------------------------------------------------------- misc
    def _update_record_enabled(self) -> None:
        enabled = (
            self._streaming
            and not self._stopping
            and not self._pending
            and self._selected_class() is not None
            and self._thread is not None
        )
        self._record_button.setEnabled(enabled)

    def _set_controls_enabled(self, enabled: bool) -> None:
        self._class_list.setEnabled(enabled)
        self._add_button.setEnabled(enabled)
        self._delete_class_button.setEnabled(enabled)
        self._fs.setEnabled(enabled and not self._settings_locked)
        self._seq_len.setEnabled(enabled and not self._settings_locked)

    def _set_recording_style(self, recording: bool) -> None:
        self._record_button.setProperty("recording", recording)
        self._record_button.style().unpolish(self._record_button)
        self._record_button.style().polish(self._record_button)

    def _set_state(self, text: str, color: str = MUTED) -> None:
        self._state_label.setText(text)
        self._state_label.setStyleSheet(f"color: {color}; border-color: {color};")

    def _refresh_live_ui(self) -> None:
        threshold = None
        if self._profile is not None:
            try:
                threshold = self._profile.on_threshold(self._current_detection_config())
            except ValueError:
                threshold = None
        self._waveform.set_threshold(threshold)
        self._waveform.set_active(self._recording)
        self._level_meter.set_level(
            self._last_level, threshold, active=self._recording
        )
        if self._recording:
            elapsed = (time.monotonic() - self._take_started) * 1000.0
            self._state_hint.setText(
                f"Recording… {elapsed:.0f} ms. Release to stop and analyse the burst."
            )

    def _space_allowed(self) -> bool:
        focus = QApplication.focusWidget()
        return not isinstance(
            focus,
            (QLineEdit, QPlainTextEdit, QAbstractSpinBox, QComboBox),
        )

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if (
            event.key() == Qt.Key.Key_Space
            and not event.isAutoRepeat()
            and self._space_allowed()
        ):
            self._begin_take()
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if (
            event.key() == Qt.Key.Key_Space
            and not event.isAutoRepeat()
            and self._space_allowed()
        ):
            self._end_take()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self._ui_timer.stop()
        self._tail_timer.stop()
        self._max_timer.stop()
        if self._thread is not None:
            self._thread.stop()
            if not self._thread.wait(4000):
                # A crashed/blocked driver read must not keep the app alive.
                self._thread.terminate()
                self._thread.wait(1000)
            self._thread = None
        # Let an in-flight take finish writing before the dataset folder (and
        # this window) goes away; its signals are auto-disconnected afterwards.
        QThreadPool.globalInstance().waitForDone(4000)
        super().closeEvent(event)
