"""Application bootstrap: one QApplication, themed, with no business logic."""

from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from emg_speech.config import DEFAULT_BAUDRATE

from .acquisition_window import AcquisitionWindow
from .inference_window import InferenceWindow
from .theme import apply_theme


def _application() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    app.setApplicationName("EMG Speech")
    app.setOrganizationName("EMG Speech")
    apply_theme(app)
    return app


def run_acquisition(
    dataset_root: str | Path,
    *,
    port: str | None = None,
    baudrate: int = DEFAULT_BAUDRATE,
) -> int:
    app = _application()
    window = AcquisitionWindow(dataset_root, initial_port=port, baudrate=baudrate)
    window.show()
    return int(app.exec())


def run_inference(
    *,
    checkpoint: str | Path | None = None,
    replay_root: str | Path | None = None,
    port: str | None = None,
    baudrate: int = DEFAULT_BAUDRATE,
    min_confidence: float = 0.5,
) -> int:
    app = _application()
    window = InferenceWindow(
        checkpoint_path=checkpoint,
        replay_root=replay_root,
        initial_port=port,
        baudrate=baudrate,
        min_confidence=min_confidence,
    )
    window.show()
    return int(app.exec())
