from __future__ import annotations

import os

import pytest

# Must be set before the first PyQt import.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from emg_speech.synthetic import generate_dataset  # noqa: E402


@pytest.fixture(scope="session")
def demo_dataset(tmp_path_factory) -> "object":
    path = tmp_path_factory.mktemp("emg_dataset")
    report = generate_dataset(path, per_class=8, seed=11)
    assert report.created == 32, "synthetic generator should produce a balanced dataset"
    return path


@pytest.fixture(scope="session")
def qt_app():
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication

    from ui.theme import apply_theme

    app = QApplication.instance() or QApplication([])
    apply_theme(app)
    return app
