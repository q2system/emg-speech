"""Application bootstrap: one QApplication, themed, with no business logic."""

from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from emg_speech.config import (
    DEFAULT_BAUDRATE,
    ensure_project_config_matches,
    load_project_config,
)
from emg_speech.jsonio import JsonIOError

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
    config_path: str | Path | None = None,
) -> int:
    try:
        project_config = load_project_config(
            config_path, required=config_path is not None
        )
    except (ValueError, OSError, JsonIOError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    app = _application()
    window = AcquisitionWindow(
        dataset_root,
        initial_port=port,
        baudrate=baudrate,
        project_config=project_config,
    )
    window.show()
    return int(app.exec())


def run_inference(
    *,
    checkpoint: str | Path | None = None,
    replay_root: str | Path | None = None,
    port: str | None = None,
    baudrate: int = DEFAULT_BAUDRATE,
    min_confidence: float = 0.5,
    config_path: str | Path | None = None,
) -> int:
    try:
        project_config = load_project_config(
            config_path, required=config_path is not None
        )
        if project_config is not None and checkpoint is not None:
            try:
                from emg_speech.model import CheckpointError, read_checkpoint_summary

                summary = read_checkpoint_summary(checkpoint)
            except CheckpointError:
                # Let the window present the checkpoint-specific error.
                summary = None
            if summary is not None:
                ensure_project_config_matches(
                    project_config,
                    sample_rate=summary.dsp_config.sample_rate,
                    seq_len=summary.dsp_config.seq_len,
                    artifact=f"checkpoint '{Path(checkpoint)}'",
                )
    except (ValueError, OSError, JsonIOError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    app = _application()
    window = InferenceWindow(
        checkpoint_path=checkpoint,
        replay_root=replay_root,
        initial_port=port,
        baudrate=baudrate,
        min_confidence=min_confidence,
        project_config=project_config,
    )
    window.show()
    return int(app.exec())
