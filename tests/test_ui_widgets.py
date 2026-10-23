from __future__ import annotations

import numpy as np


def test_static_waveform_shows_all_channels_or_emg_only(qt_app) -> None:
    from ui.widgets import StaticWaveform

    widget = StaticWaveform()
    rng = np.random.default_rng(3)
    series = [rng.normal(size=128).astype(np.float32) for _ in range(9)]
    labels = ["EMG1", "EMG2", "EMG3", "Gx", "Gy", "Gz", "Ax", "Ay", "Az"]
    widget.set_series(series, labels)

    assert widget.channel_count == 9
    assert widget.visible_count == 9
    assert not widget.emg_only

    widget.set_emg_only(True)
    assert widget.emg_only
    assert widget.visible_count == 3

    widget.set_emg_only(False)
    assert widget.visible_count == 9

    widget.clear()
    assert widget.channel_count == 0


def test_take_preview_defaults_to_emg_only(tmp_path, qt_app) -> None:
    from emg_speech.dsp import ExtractedSample
    from ui.acquisition_window import AcquisitionWindow

    window = AcquisitionWindow(tmp_path)
    try:
        data = np.zeros((128, 9), dtype=np.float32)
        for channel in range(9):
            data[:, channel] = np.linspace(-1.0, 1.0, 128)
        extracted = ExtractedSample(
            data=data,
            event_ms=300.0,
            peak_envelope=1.0,
            on_threshold=0.1,
            off_threshold=0.05,
            source_frames=500,
            event_start=10,
            event_end=200,
        )
        window._on_take_completed(extracted, str(tmp_path / "hello" / "take.npy"), "hello")

        assert window._preview.channel_count == 9
        assert window._preview_emg_only.isChecked()
        assert window._preview.visible_count == 3

        window._preview_emg_only.setChecked(False)
        assert window._preview.visible_count == 9
        window._preview_emg_only.setChecked(True)
        assert window._preview.visible_count == 3
    finally:
        window.close()


def test_live_waveform_scrolls_with_a_fixed_window(qt_app) -> None:
    from ui.widgets import LiveWaveform

    widget = LiveWaveform(capacity=5)
    widget.resize(420, 180)
    widget.push_values(range(10))
    assert widget.envelope_count == 5
    assert list(widget._values) == [5.0, 6.0, 7.0, 8.0, 9.0]

    rows = [[float(index)] * 9 for index in range(10)]
    widget.push_channels(rows)
    assert widget.channel_sample_count == 5
    assert list(widget._channel_values[0]) == [5.0, 6.0, 7.0, 8.0, 9.0]

    widget.set_threshold(3.0)
    widget.set_show_channels(False)
    widget.grab()
    widget.set_show_channels(True)
    assert widget.show_channels
    widget.grab()

    # A 3000-point trace must be decimated to roughly one point per pixel;
    # otherwise nine full-rate polylines make the live view lag badly.
    from PyQt6.QtCore import QRectF

    points = widget._time_points(
        np.linspace(0.0, 1.0, 3000),
        QRectF(0.0, 0.0, 400.0, 150.0),
        centre=True,
        peak=1.0,
    )
    assert len(points) < 500

    widget.set_capacity(3)
    assert widget.envelope_count == 3
    assert widget.channel_sample_count == 3
    widget.clear()
    assert widget.envelope_count == 0
    assert widget.channel_sample_count == 0


def test_inference_defaults_to_envelope_and_toggles_all_signals(qt_app) -> None:
    from ui.inference_window import InferenceWindow

    window = InferenceWindow()
    try:
        assert not window._show_all_signals.isChecked()
        assert not window._waveform.show_channels

        window._show_all_signals.setChecked(True)
        assert window._waveform.show_channels
        window._show_all_signals.setChecked(False)
        assert not window._waveform.show_channels
    finally:
        window.close()


def test_inference_connect_button_starts_and_stops(monkeypatch, qt_app) -> None:
    from ui.inference_window import InferenceWindow

    window = InferenceWindow()
    started: list[bool] = []
    stopped: list[bool] = []
    monkeypatch.setattr(window, "_start_stream", lambda: started.append(True))
    monkeypatch.setattr(window, "_stop_stream", lambda: stopped.append(True))
    window._port_selector.set_port_text("/dev/ttyFAKE")

    window._port_selector._connect.click()
    assert started == [True]

    window._port_selector.set_connected(True)
    window._port_selector._connect.click()
    assert stopped == [True]
    window.close()


def test_port_selector_accepts_manual_port_names(qt_app) -> None:
    from emg_speech.sources import PortInfo
    from ui.widgets import PortSelector

    selector = PortSelector()
    assert selector.baudrate() == 921600

    selector.set_port_text("COM2")
    assert selector.current_port() == "COM2"

    # A typed port survives a refresh that does not enumerate it.
    selector.set_ports([PortInfo("/dev/ttyUSB0", "USB Serial")], selected="COM2")
    assert selector.current_port() == "COM2"

    # Selecting an enumerated port still returns its device name, not the label.
    selector.set_ports([PortInfo("COM2", "USB Serial")], selected="COM2")
    assert selector.current_port() == "COM2"

    line_edit = selector._port.lineEdit()
    assert line_edit is not None
    line_edit.setText("COM9")
    assert selector.current_port() == "COM9"

    selector._port.setEditText("")
    assert selector.current_port() == ""


def test_port_selector_placeholder_is_not_a_port(qt_app) -> None:
    from ui.widgets import PortSelector

    selector = PortSelector()
    selector.set_ports([], selected="")
    assert selector.current_port() == ""

    # A manually typed port survives an enumeration that no longer sees it.
    selector.set_ports([], selected="COM7")
    assert selector.current_port() == "COM7"
