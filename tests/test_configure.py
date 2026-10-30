from __future__ import annotations

from emg_speech.apps.configure import main
from emg_speech.apps.simulate import main as simulate_main
from emg_speech.config import DspConfig, ProjectConfig


def test_configure_command_writes_project_file(tmp_path, capsys) -> None:
    path = tmp_path / "emg_config.json"

    assert main(
        [
            "--config",
            str(path),
            "--sample-rate",
            "1000",
            "--seq-len",
            "256",
            "--on-sigma",
            "4.5",
        ]
    ) == 0

    config = ProjectConfig.load(path)
    assert config.sample_rate == 1000.0
    assert config.seq_len == 256
    assert config.detection.on_sigma == 4.5
    assert "written" in capsys.readouterr().out


def test_configure_command_shows_without_writing(tmp_path, capsys) -> None:
    path = tmp_path / "emg_config.json"

    assert main(["--config", str(path)]) == 0

    output = capsys.readouterr().out
    assert "not created yet" in output
    assert "500 Hz" in output
    assert not path.exists()


def test_configure_command_rejects_invalid_values(tmp_path, capsys) -> None:
    path = tmp_path / "emg_config.json"

    assert main(["--config", str(path), "--sample-rate", "50"]) == 2

    assert "Invalid configuration" in capsys.readouterr().err
    assert not path.exists()


def test_configure_command_can_reset(tmp_path) -> None:
    path = tmp_path / "emg_config.json"
    assert main(["--config", str(path), "--sample-rate", "1000"]) == 0
    assert main(["--config", str(path), "--reset"]) == 0

    config = ProjectConfig.load(path)
    assert config.sample_rate == 500.0
    assert config.seq_len == 128


def test_simulator_accepts_order_and_repeat(tmp_path, capsys) -> None:
    path = tmp_path / "emg_config.json"
    ProjectConfig(dsp=DspConfig(sample_rate=500.0, seq_len=128)).save(path)

    assert (
        simulate_main(
            [
                "--config",
                str(path),
                "--dry-run",
                "--words",
                "alpha,beta",
                "--order",
                "sequence",
                "--repeat",
                "2",
                "--start-idle",
                "0",
                "--duration",
                "4",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "Word order: sequence · repeat: 2" in output
    assert "alpha" in output
    assert "beta" in output


def test_simulator_uses_config_sample_rate(tmp_path, capsys) -> None:
    path = tmp_path / "emg_config.json"
    ProjectConfig(dsp=DspConfig(sample_rate=1000.0, seq_len=256)).save(path)

    assert (
        simulate_main(
            [
                "--config",
                str(path),
                "--dry-run",
                "--mode",
                "manual",
                "--duration",
                "0.1",
                "--start-idle",
                "0",
            ]
        )
        == 0
    )
    assert "at 1000 Hz" in capsys.readouterr().out
