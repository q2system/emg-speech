"""Dark theme for both desktop applications."""

from __future__ import annotations

from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication

BACKGROUND = "#0b0f14"
SURFACE = "#131a22"
SURFACE_ALT = "#18212b"
BORDER = "#263341"
TEXT = "#e6edf3"
MUTED = "#8b9bab"
ACCENT = "#3ddc97"
ACCENT_DIM = "#1f7a58"
WARNING = "#f2b84b"
DANGER = "#ff6b6b"
INFO = "#5cc8ff"
GRID = "#1f2a36"

EMG_COLORS = ("#5cc8ff", "#3ddc97", "#f2b84b")
#: One colour per channel in ``emg_speech.constants.CHANNELS`` order.
CHANNEL_COLORS = (
    *EMG_COLORS,           # EMG1–3
    "#b48cff",             # Gx
    "#ff8fab",             # Gy
    "#ffa94d",             # Gz
    "#7bdff2",             # Ax
    "#c3f584",             # Ay
    "#f7aef8",             # Az
)


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(BACKGROUND))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Base, QColor(SURFACE))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(SURFACE_ALT))
    palette.setColor(QPalette.ColorRole.Text, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Button, QColor(SURFACE_ALT))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(ACCENT_DIM))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(SURFACE))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(MUTED))
    app.setPalette(palette)
    app.setStyleSheet(STYLESHEET)


STYLESHEET = f"""
QWidget {{
    background: {BACKGROUND};
    color: {TEXT};
    font-size: 13px;
}}
QMainWindow, QDialog {{
    background: {BACKGROUND};
}}
QGroupBox, QFrame#card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 10px;
    margin-top: 10px;
    padding: 12px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 6px;
    color: {MUTED};
    font-weight: 600;
    letter-spacing: 0.4px;
    text-transform: uppercase;
}}
QFrame#card {{
    margin-top: 0px;
}}
QLabel#title {{
    font-size: 20px;
    font-weight: 700;
}}
QLabel#subtitle, QLabel#muted {{
    color: {MUTED};
}}
QLabel#badge {{
    background: {SURFACE_ALT};
    border: 1px solid {BORDER};
    border-radius: 9px;
    padding: 3px 9px;
    color: {MUTED};
}}
QLabel#bigNumber {{
    font-size: 24px;
    font-weight: 700;
    color: {ACCENT};
}}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QListWidget {{
    background: {SURFACE_ALT};
    border: 1px solid {BORDER};
    border-radius: 7px;
    padding: 6px 8px;
    selection-background-color: {ACCENT_DIM};
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border: 1px solid {ACCENT_DIM};
}}
QComboBox::drop-down {{
    border: none;
    width: 22px;
}}
QComboBox QAbstractItemView {{
    background: {SURFACE_ALT};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT_DIM};
    outline: none;
}}
QListWidget::item {{
    padding: 7px 8px;
    border-radius: 6px;
}}
QListWidget::item:selected {{
    background: {ACCENT_DIM};
    color: white;
}}
QPushButton {{
    background: {SURFACE_ALT};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 8px 14px;
    font-weight: 600;
}}
QPushButton:hover {{
    border-color: {ACCENT_DIM};
}}
QPushButton:pressed {{
    background: #101820;
}}
QPushButton:disabled {{
    color: #55636f;
    border-color: #1d2731;
}}
QPushButton#primaryButton {{
    background: {ACCENT_DIM};
    border: 1px solid {ACCENT};
    color: #eafff6;
}}
QPushButton#primaryButton:hover {{
    background: {ACCENT};
    color: #04110b;
}}
QPushButton#recordButton {{
    background: #1b2632;
    border: 2px solid {DANGER};
    color: {TEXT};
    font-size: 18px;
    font-weight: 700;
    padding: 20px 12px;
    border-radius: 12px;
}}
QPushButton#recordButton:hover {{
    background: #25313f;
}}
QPushButton#recordButton[recording="true"] {{
    background: {DANGER};
    color: #1a0505;
}}
QPushButton#recordButton:disabled {{
    border-color: #2a3642;
    color: #55636f;
}}
QPushButton#dangerButton {{
    border-color: #5c2b2b;
    color: {DANGER};
}}
QPlainTextEdit#transcript {{
    font-size: 20px;
    line-height: 160%;
    padding: 14px;
    background: #0e141b;
}}
QSlider::groove:horizontal {{
    height: 4px;
    background: {BORDER};
    border-radius: 2px;
}}
QSlider::sub-page:horizontal {{
    background: {ACCENT_DIM};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    width: 14px;
    margin: -5px 0;
    border-radius: 7px;
    background: {ACCENT};
}}
QStatusBar {{
    background: {SURFACE};
    border-top: 1px solid {BORDER};
    color: {MUTED};
}}
QStatusBar QLabel {{
    color: {MUTED};
    padding: 0 6px;
}}
QSplitter::handle {{
    background: {BACKGROUND};
}}
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: {BORDER};
    border-radius: 5px;
    min-height: 24px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QToolTip {{
    background: {SURFACE_ALT};
    color: {TEXT};
    border: 1px solid {BORDER};
    padding: 6px;
}}
"""
