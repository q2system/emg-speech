from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "emg_speech"
UI = ROOT / "ui"


def test_core_never_imports_qt() -> None:
    offenders: list[str] = []
    for path in CORE.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "PyQt" in text or "PySide" in text:
            offenders.append(path.name)
    assert not offenders, f"Qt leaked into the core layer: {offenders}"


def test_dsp_has_no_silent_dependencies() -> None:
    """The real-time DSP must not depend on Qt or torch (keeps it testable/fast)."""

    for name in ("dsp.py", "protocol.py", "config.py", "storage.py", "ring_buffer.py"):
        text = (CORE / name).read_text(encoding="utf-8")
        assert "PyQt" not in text
        assert "torch" not in text


def test_ui_uses_the_core_engines() -> None:
    acquisition = (UI / "acquisition_window.py").read_text(encoding="utf-8")
    inference = (UI / "inference_window.py").read_text(encoding="utf-8")
    workers = (UI / "workers.py").read_text(encoding="utf-8")
    assert "from emg_speech" in acquisition
    assert "from emg_speech" in inference
    assert "InferenceThread" in inference
    assert "QThread" in workers
    # No model imports may be duplicated inside the presentation files.
    assert "import torch" not in acquisition
    assert "import torch" not in inference


def test_ui_smoke(demo_dataset: Path, qt_app) -> None:
    from PyQt6.QtWidgets import QApplication

    from ui.acquisition_window import AcquisitionWindow
    from ui.inference_window import InferenceWindow

    acquisition = AcquisitionWindow(demo_dataset)
    try:
        assert acquisition._class_list.count() == 4
        assert acquisition._record_button.text() == "Hold to record"
        assert not acquisition._record_button.isEnabled()  # no serial connection
        acquisition._on_serial_stats(120, 3)
        assert "3 malformed" in acquisition._error_status.text()
    finally:
        acquisition.close()

    inference = InferenceWindow()
    try:
        assert not inference._start_button.isEnabled()
        assert inference._transcript.isReadOnly()
    finally:
        inference.close()

    QApplication.processEvents()
