from __future__ import annotations

from pathlib import Path


def test_public_api_delegates_to_the_gui(monkeypatch, tmp_path: Path) -> None:
    import emg_speech.api as api
    import emg_speech.ui.app as gui

    calls: list[tuple[tuple, dict]] = []

    def fake_acquisition(*args, **kwargs):
        calls.append((args, kwargs))
        return 7

    def fake_inference(**kwargs):
        calls.append(((), kwargs))
        return 8

    monkeypatch.setattr(gui, "run_acquisition", fake_acquisition)
    monkeypatch.setattr(gui, "run_inference", fake_inference)

    assert api.record(tmp_path, port="COM9", baudrate=123) == 7
    assert calls[-1] == ((tmp_path,), {"port": "COM9", "baudrate": 123})

    assert api.recognize("model.pt", replay="dataset_demo") == 8
    assert calls[-1][1]["checkpoint"] == "model.pt"
    assert calls[-1][1]["replay_root"] == "dataset_demo"


def test_train_model_builds_a_simple_config(monkeypatch) -> None:
    import emg_speech.api as api
    import emg_speech.train as training

    calls: list[tuple[tuple, dict]] = []

    def fake_train(*args, **kwargs):
        calls.append((args, kwargs))
        return "report"

    monkeypatch.setattr(training, "train", fake_train)

    assert api.train_model("data", "model.pt", epochs=7, device="cpu") == "report"
    args, kwargs = calls[0]
    assert args == ("data", "model.pt")
    assert kwargs["train_config"].epochs == 7
    assert kwargs["train_config"].device == "cpu"


def test_simulate_api_forwards_options(monkeypatch) -> None:
    import emg_speech.api as api
    import emg_speech.apps.simulate as simulator

    calls: list[list[str]] = []

    def fake_simulator(options):
        calls.append(options)
        return 3

    monkeypatch.setattr(simulator, "main", fake_simulator)

    assert api.simulate("--pty", "--interval", "1.5") == 3
    assert calls == [["--pty", "--interval", "1.5"]]
