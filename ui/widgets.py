"""Reusable presentation-only widgets.  No DSP, serial or model code here."""

from __future__ import annotations

from collections import deque
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QPolygonF
from PyQt6.QtCore import QPointF
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from emg_speech.config import DEFAULT_BAUDRATE
from emg_speech.constants import CHANNELS, N_CHANNELS
from emg_speech.sources import PortInfo, available_ports

from .theme import (
    ACCENT,
    CHANNEL_COLORS,
    DANGER,
    EMG_COLORS,
    GRID,
    MUTED,
    SURFACE_ALT,
    TEXT,
    WARNING,
)


class LiveWaveform(QWidget):
    """Scrolling strip chart for the envelope or all filtered channels.

    New samples enter at the right edge and old samples fall off the left, so
    the display keeps a fixed-size time window instead of compressing the whole
    conversation into the widget.  ``show_channels`` selects the 9-channel view;
    the detection threshold is drawn in both modes.
    """

    _LEGEND_ROWS = 2
    _LEGEND_COLUMNS = 5

    def __init__(self, parent: QWidget | None = None, *, capacity: int = 900) -> None:
        super().__init__(parent)
        self._capacity = max(2, int(capacity))
        self._values: deque[float] = deque(maxlen=self._capacity)
        self._channel_values: list[deque[float]] = [
            deque(maxlen=self._capacity) for _ in range(N_CHANNELS)
        ]
        self._threshold: float | None = None
        self._active = False
        self._show_channels = False
        self._dirty = False
        self.setMinimumHeight(130)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._refresh)
        self._timer.start()

    # ------------------------------------------------------------------ data
    def push_values(self, values: Iterable[float]) -> None:
        changed = False
        for value in values:
            self._values.append(float(value))
            changed = True
        if changed:
            self._dirty = True

    def push_channels(self, frames: Iterable[Sequence[float]]) -> None:
        changed = False
        for frame in frames:
            row = list(frame)
            if len(row) != len(self._channel_values):
                continue
            for target, value in zip(self._channel_values, row):
                target.append(float(value))
            changed = True
        if changed:
            self._dirty = True

    def set_capacity(self, capacity: int) -> None:
        capacity = max(2, int(capacity))
        if capacity == self._capacity:
            return
        self._capacity = capacity
        self._values = deque(self._values, maxlen=capacity)
        self._channel_values = [
            deque(values, maxlen=capacity) for values in self._channel_values
        ]
        self._dirty = True

    def set_show_channels(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled != self._show_channels:
            self._show_channels = enabled
            self._dirty = True

    def set_threshold(self, threshold: float | None) -> None:
        self._threshold = threshold
        self._dirty = True

    def set_active(self, active: bool) -> None:
        if active != self._active:
            self._active = active
            self._dirty = True

    def clear(self) -> None:
        self._values.clear()
        for values in self._channel_values:
            values.clear()
        self._dirty = True

    @property
    def show_channels(self) -> bool:
        return self._show_channels

    @property
    def envelope_count(self) -> int:
        return len(self._values)

    @property
    def channel_sample_count(self) -> int:
        return max((len(values) for values in self._channel_values), default=0)

    def _refresh(self) -> None:
        if self._dirty:
            self._dirty = False
            self.update()

    # --------------------------------------------------------------- painting
    def _time_points(
        self,
        values: NDArray[np.float64],
        plot,
        *,
        centre: bool,
        peak: float,
    ) -> list[QPointF]:
        """Map the newest sample to the right edge and older ones to the left."""

        count = len(values)
        if count == 0:
            return []
        # Roughly one point per two horizontal pixels is plenty for a live strip
        # chart.  Drawing thousands of points per trace (9 traces × 3000) made
        # the all-channel view visibly laggy.
        target = max(1, int(plot.width() * 0.5))
        stride = max(1, count // target)
        shown = values[::stride]
        positions = np.arange(0, count, stride, dtype=np.int64)
        if positions[-1] != count - 1:
            shown = np.append(shown, values[-1])
            positions = np.append(positions, count - 1)
        step = plot.width() / max(1, self._capacity - 1)
        xs = plot.right() - (count - 1 - positions) * step
        if centre:
            ys = (
                plot.top()
                + plot.height() * 0.5
                - shown / peak * plot.height() * 0.45
            )
        else:
            ys = plot.bottom() - shown / peak * plot.height() * 0.9
        return [QPointF(float(x), float(y)) for x, y in zip(xs, ys)]

    def _threshold_fraction(self) -> float | None:
        threshold = self._threshold
        if threshold is None or threshold <= 0 or not self._values:
            return None
        levels = np.fromiter(self._values, dtype=np.float64, count=len(self._values))
        scale = max(float(levels.max()), float(threshold) * 1.3, 1e-6)
        return max(0.0, min(1.0, float(threshold) / scale))

    def _paint_threshold(self, painter: QPainter, plot, *, label: bool) -> None:
        fraction = self._threshold_fraction()
        if fraction is None:
            return
        y = plot.bottom() - fraction * plot.height()
        threshold_pen = QPen(QColor(WARNING), 1, Qt.PenStyle.DashLine)
        threshold_pen.setCosmetic(True)
        painter.setPen(threshold_pen)
        painter.drawLine(plot.left(), int(y), plot.right(), int(y))
        if label:
            painter.setPen(QColor(WARNING))
            painter.drawText(
                plot.adjusted(8, 0, -8, -4),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom,
                f"threshold {self._threshold:.3g}",
            )

    def _paint_legend(self, painter: QPainter, area) -> None:
        labels = [
            name.upper() if name.startswith("emg") else name.capitalize()
            for name in CHANNELS
        ]
        cell_width = max(1, (area.width() - 16) // self._LEGEND_COLUMNS)
        legend_top = area.bottom() - self._LEGEND_ROWS * 13 + 1
        font = painter.font()
        font.setPointSize(7)
        painter.setFont(font)
        for index, label in enumerate(labels):
            row = index // self._LEGEND_COLUMNS
            column = index % self._LEGEND_COLUMNS
            x = area.left() + 8 + column * cell_width
            y = legend_top + row * 13 + 6
            color = CHANNEL_COLORS[index % len(CHANNEL_COLORS)]
            swatch_pen = QPen(QColor(color), 2)
            swatch_pen.setCosmetic(True)
            painter.setPen(swatch_pen)
            painter.drawLine(QPointF(x, y), QPointF(x + 10, y))
            painter.setPen(QColor(TEXT))
            painter.drawText(QPointF(x + 14, y + 3), label)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt API)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        area = self.rect().adjusted(6, 6, -6, -6)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(SURFACE_ALT))
        painter.drawRoundedRect(area, 8, 8)

        grid_pen = QPen(QColor(GRID), 1)
        painter.setPen(grid_pen)
        for fraction in (0.25, 0.5, 0.75):
            y = int(area.top() + area.height() * fraction)
            painter.drawLine(area.left(), y, area.right(), y)

        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        painter.setPen(QColor(MUTED))
        if self._show_channels:
            painter.drawText(
                area.adjusted(8, 4, -8, 0),
                Qt.AlignmentFlag.AlignLeft,
                "All 9 signals · z-scored",
            )
            threshold = self._threshold
            badge = (
                f"threshold {float(threshold):.3g}"
                if threshold is not None and threshold > 0
                else "learning threshold"
            )
            painter.drawText(
                area.adjusted(8, 4, -8, 0), Qt.AlignmentFlag.AlignRight, badge
            )
            legend_height = self._LEGEND_ROWS * 13 + 2
            plot = area.adjusted(8, 18, -8, -(legend_height + 4))
            if plot.width() < 10 or plot.height() < 10:
                return
            if self.channel_sample_count == 0:
                painter.drawText(
                    area, Qt.AlignmentFlag.AlignCenter, "waiting for serial data"
                )
                return

            series: list[NDArray[np.float64]] = []
            for channel in self._channel_values:
                values = np.fromiter(channel, dtype=np.float64, count=len(channel))
                if values.size < 2:
                    series.append(values)
                    continue
                spread = float(values.std())
                series.append(
                    (values - float(values.mean())) / spread
                    if spread > 1e-9
                    else np.zeros_like(values)
                )
            peak = max(
                (float(np.max(np.abs(values))) for values in series if values.size),
                default=1e-6,
            )
            peak = max(peak, 1e-6)
            for index, values in enumerate(series):
                if values.size < 2:
                    continue
                points = self._time_points(values, plot, centre=True, peak=peak)
                # Cosmetic pens are essential here: on HiDPI screens a
                # non-cosmetic 1-logical-pixel pen is rendered as a wide,
                # antialiased path and is dramatically slower for 9 traces.
                pen = QPen(QColor(CHANNEL_COLORS[index % len(CHANNEL_COLORS)]), 1.0)
                pen.setCosmetic(True)
                painter.setPen(pen)
                painter.drawPolyline(QPolygonF(points))
            self._paint_threshold(painter, plot, label=False)
            self._paint_legend(painter, area)
            return

        painter.drawText(
            area.adjusted(8, 4, -8, 0),
            Qt.AlignmentFlag.AlignLeft,
            "EMG envelope",
        )
        if not self._values:
            painter.drawText(area, Qt.AlignmentFlag.AlignCenter, "waiting for serial data")
            return

        plot = area.adjusted(8, 18, -8, -16)
        if plot.width() < 10 or plot.height() < 10:
            return
        values = np.fromiter(self._values, dtype=np.float64, count=len(self._values))
        max_value = float(np.max(values))
        threshold = self._threshold or 0.0
        scale = max(max_value, threshold * 1.3, 1e-6)
        points = self._time_points(values, plot, centre=False, peak=scale)
        color = QColor(DANGER if self._active else ACCENT)
        envelope_pen = QPen(color, 1.8)
        envelope_pen.setCosmetic(True)
        painter.setPen(envelope_pen)
        painter.drawPolyline(QPolygonF(points))

        self._paint_threshold(painter, plot, label=True)
        painter.setPen(QColor(MUTED))
        painter.drawText(
            plot.adjusted(8, 0, -8, -4),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom,
            f"peak {max_value:.3g}",
        )


class StaticWaveform(QWidget):
    """Fixed-length preview of an extracted, time-normalised take.

    The all-channel view shows every recorded trace.  Because the EMG channels
    are normalised to unit RMS while the IMU channels stay in physical units,
    each trace is standardised (z-score) purely for display.  The ``emg_only``
    mode shows the three-curve EMG preview on one shared scale (the acquisition
    window selects this mode by default).
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._series: list[NDArray[np.float32]] = []
        self._labels: list[str] = []
        self._emg_only = False
        self.setMinimumHeight(150)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setToolTip(
            "All-channel view: each trace is standardised (z-scored) so the "
            "low-amplitude IMU channels stay visible. Tick 'EMG only' to compare "
            "the three EMG channels on one shared scale."
        )

    def set_series(
        self,
        series: Sequence[NDArray[np.floating]],
        labels: Sequence[str] | None = None,
    ) -> None:
        self._series = [
            np.array(values, dtype=np.float32, copy=True) for values in series
        ]
        self._labels = list(labels) if labels else [f"ch{i}" for i in range(len(series))]
        self.update()

    def set_emg_only(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled != self._emg_only:
            self._emg_only = enabled
            self.update()

    @property
    def emg_only(self) -> bool:
        return self._emg_only

    @property
    def channel_count(self) -> int:
        return len(self._series)

    @property
    def visible_count(self) -> int:
        return min(3, len(self._series)) if self._emg_only else len(self._series)

    def clear(self) -> None:
        self._series = []
        self._labels = []
        self.update()

    @staticmethod
    def _standardise(values: NDArray[np.float32]) -> NDArray[np.float64]:
        data = np.asarray(values, dtype=np.float64)
        if data.size == 0:
            return data
        spread = float(data.std())
        if spread < 1e-9:
            return np.zeros_like(data)
        return (data - float(data.mean())) / spread

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        area = self.rect().adjusted(6, 6, -6, -6)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(SURFACE_ALT))
        painter.drawRoundedRect(area, 8, 8)

        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        painter.setPen(QColor(MUTED))
        painter.drawText(
            area.adjusted(8, 4, -8, 0),
            Qt.AlignmentFlag.AlignLeft,
            "normalised take preview",
        )
        if not self._series:
            painter.drawText(
                area, Qt.AlignmentFlag.AlignCenter, "press and hold the record button"
            )
            return

        if self._emg_only:
            indices = list(range(min(3, len(self._series))))
            display = [self._series[index] for index in indices]
            badge = f"{len(display)} EMG channels"
        else:
            indices = list(range(len(self._series)))
            display = [self._standardise(self._series[index]) for index in indices]
            badge = f"{len(display)} channels · z-scored"
        painter.drawText(
            area.adjusted(8, 4, -8, 0), Qt.AlignmentFlag.AlignRight, badge
        )

        legend_rows = 1 if len(display) <= 4 else 2
        legend_height = legend_rows * 13 + 2
        plot = area.adjusted(8, 18, -8, -(legend_height + 2))
        if plot.width() < 10 or plot.height() < 10:
            return

        peak = max(
            (float(np.max(np.abs(values))) for values in display if values.size),
            default=1e-6,
        )
        peak = max(peak, 1e-6)
        for position, values in enumerate(display):
            if values.size < 2:
                continue
            x = np.linspace(0.0, 1.0, len(values))
            points = [
                QPointF(
                    plot.left() + float(px) * plot.width(),
                    plot.top()
                    + plot.height() * 0.5
                    - float(value) / peak * plot.height() * 0.45,
                )
                for px, value in zip(x, values)
            ]
            channel = indices[position]
            color = (
                EMG_COLORS[channel % len(EMG_COLORS)]
                if self._emg_only
                else CHANNEL_COLORS[channel % len(CHANNEL_COLORS)]
            )
            painter.setPen(QPen(QColor(color), 1.5))
            painter.drawPolyline(QPolygonF(points))

        # Compact legend: one row for 1–4 traces, two rows for 5+ traces.
        columns = (len(display) + legend_rows - 1) // legend_rows
        cell_width = max(1, (area.width() - 16) // max(1, columns))
        legend_top = area.bottom() - legend_height + 3
        font.setPointSize(7)
        painter.setFont(font)
        for position, channel in enumerate(indices):
            row = position // columns
            column = position % columns
            x = area.left() + 8 + column * cell_width
            y = legend_top + row * 13 + 6
            color = (
                EMG_COLORS[channel % len(EMG_COLORS)]
                if self._emg_only
                else CHANNEL_COLORS[channel % len(CHANNEL_COLORS)]
            )
            painter.setPen(QPen(QColor(color), 2))
            painter.drawLine(QPointF(x, y), QPointF(x + 10, y))
            painter.setPen(QColor(TEXT))
            label = self._labels[channel] if channel < len(self._labels) else ""
            painter.drawText(QPointF(x + 14, y + 3), label)


class LevelMeter(QWidget):
    """Instantaneous envelope level with threshold marker."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._level = 0.0
        self._threshold: float | None = None
        self._active = False
        self.setFixedHeight(34)

    def set_level(
        self, level: float, threshold: float | None = None, *, active: bool = False
    ) -> None:
        self._level = max(0.0, float(level))
        self._threshold = threshold
        self._active = active
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        area = self.rect().adjusted(6, 7, -6, -7)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(SURFACE_ALT))
        painter.drawRoundedRect(area, 5, 5)

        scale = max(self._threshold * 4.0 if self._threshold else 1.0, self._level, 1e-6)
        fraction = min(1.0, self._level / scale)
        fill = area.adjusted(0, 0, -int(area.width() * (1.0 - fraction)), 0)
        if fraction > 0:
            painter.setBrush(QColor(DANGER if self._active else ACCENT))
            painter.drawRoundedRect(fill, 5, 5)

        if self._threshold:
            x = area.left() + int(area.width() * min(1.0, self._threshold / scale))
            painter.setPen(QPen(QColor(WARNING), 1))
            painter.drawLine(x, area.top() - 2, x, area.bottom() + 2)

        painter.setPen(QColor(TEXT if self._active else MUTED))
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        threshold_text = f"  ·  thr {self._threshold:.3g}" if self._threshold else ""
        painter.drawText(
            area.adjusted(8, 0, -8, 0),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            f"level {self._level:.3g}{threshold_text}",
        )


class PortSelector(QWidget):
    """Serial port + baud selection and connect toggle.

    The port box is editable so a port that is not enumerated (a Windows COM
    port, a com0com pair or a POSIX PTY) can be typed in directly.
    """

    connect_requested = pyqtSignal(str, int)
    disconnect_requested = pyqtSignal()
    selection_changed = pyqtSignal(str)

    _NO_PORT_TEXT = "No ports — type one (e.g. COM3)"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self._port = QComboBox()
        self._port.setEditable(True)
        self._port.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self._port.setMinimumWidth(250)
        self._port.setToolTip(
            "Serial port: pick a detected one or type it manually "
            "(e.g. COM3 on Windows, /dev/ttyUSB0 or /dev/pts/7 elsewhere)"
        )
        line_edit = self._port.lineEdit()
        if line_edit is not None:
            line_edit.setPlaceholderText("select or type a port (e.g. COM3)")
            line_edit.setClearButtonEnabled(True)
        self._baud = QComboBox()
        for value in (115200, 230400, 460800, 921600, 1500000, 2000000):
            self._baud.addItem(str(value), value)
        self._baud.setToolTip("Must match the firmware Serial.begin() speed")
        self.set_baudrate(DEFAULT_BAUDRATE)

        self._refresh = QPushButton("↻")
        self._refresh.setToolTip("Rescan serial ports")
        self._refresh.setFixedWidth(38)
        self._refresh.clicked.connect(self.refresh_ports)

        self._port.currentIndexChanged.connect(self._emit_selection)
        self._port.editTextChanged.connect(self._emit_selection)

        self._connect = QPushButton("Connect")
        self._connect.setObjectName("primaryButton")
        self._connect.clicked.connect(self._on_clicked)
        self._connected = False

        layout.addWidget(QLabel("Port"))
        layout.addWidget(self._port, 1)
        layout.addWidget(self._baud)
        layout.addWidget(self._refresh)
        layout.addWidget(self._connect)
        self.refresh_ports()

    def _emit_selection(self, *_args: object) -> None:
        self.selection_changed.emit(self.current_port())

    def refresh_ports(self) -> None:
        if self._connected:
            return
        selected = self.current_port()
        ports = available_ports()
        self.set_ports(ports, selected=selected)

    def set_ports(self, ports: Sequence[PortInfo], selected: str | None = None) -> None:
        keep = selected if selected is not None else self.current_port()
        self._port.blockSignals(True)
        try:
            self._port.clear()
            for info in ports:
                text = (
                    info.device
                    if not info.description
                    else f"{info.device} — {info.description}"
                )
                self._port.addItem(text, info.device)
            if not ports:
                self._port.addItem(self._NO_PORT_TEXT, "")
            if keep:
                index = self._port.findData(keep)
                if index < 0:
                    index = self._port.findText(keep)
                if index >= 0:
                    self._port.setCurrentIndex(index)
                else:
                    self._port.setEditText(keep)
            elif ports:
                self._port.setCurrentIndex(0)
        finally:
            self._port.blockSignals(False)

    def current_port(self) -> str:
        """Return the selected/typed port, or ``""`` for the placeholder."""

        text = self._port.currentText().strip().strip("\"'")
        if not text or text == self._NO_PORT_TEXT:
            return ""
        index = self._port.findText(text)
        if index >= 0:
            data = self._port.itemData(index)
            return str(data) if data else ""
        return text

    def set_port_text(self, port: str) -> None:
        """Select ``port`` if listed, otherwise put it in the edit box."""

        text = (port or "").strip()
        if not text:
            return
        index = self._port.findData(text)
        if index < 0:
            index = self._port.findText(text)
        if index >= 0:
            self._port.setCurrentIndex(index)
        else:
            self._port.setEditText(text)
        self.selection_changed.emit(self.current_port())

    def select_port(self, port: str) -> None:
        self.set_port_text(port)

    def baudrate(self) -> int:
        data = self._baud.currentData()
        return int(data) if data else DEFAULT_BAUDRATE

    def set_baudrate(self, baudrate: int) -> None:
        index = self._baud.findData(int(baudrate))
        if index >= 0:
            self._baud.setCurrentIndex(index)

    def set_connected(self, connected: bool, description: str = "") -> None:
        self._connected = connected
        self._connect.setText("Disconnect" if connected else "Connect")
        self._connect.setObjectName("dangerButton" if connected else "primaryButton")
        self._connect.style().unpolish(self._connect)
        self._connect.style().polish(self._connect)
        self._port.setEnabled(not connected)
        self._baud.setEnabled(not connected)
        self._refresh.setEnabled(not connected)
        if description:
            self._port.setToolTip(description)

    def set_busy(self, busy: bool, text: str = "Connecting…") -> None:
        self._connect.setEnabled(not busy)
        if busy:
            self._connect.setText(text)

    def _on_clicked(self) -> None:
        if self._connected:
            self.disconnect_requested.emit()
            return
        port = self.current_port()
        if not port:
            return
        self.connect_requested.emit(port, self.baudrate())
