from __future__ import annotations

import json

import pytest

from emg_speech.config import (
    DEFAULT_BAUDRATE,
    DatasetMeta,
    DetectionConfig,
    DspConfig,
    ProjectConfig,
    baudrate_warning,
    ensure_project_config_matches,
    load_project_config,
    required_baudrate,
)


def test_default_baudrate_carries_the_default_stream() -> None:
    assert baudrate_warning(DspConfig().sample_rate, DEFAULT_BAUDRATE) is None


@pytest.mark.parametrize("baudrate", [115200, 230400, 460800])
def test_slow_baudrates_are_flagged(baudrate: int) -> None:
    warning = baudrate_warning(500.0, baudrate)
    assert warning is not None
    assert str(required_baudrate(500.0)) in warning


def test_required_baudrate_grows_with_sample_rate() -> None:
    rates = [250.0, 500.0, 1000.0, 2000.0]
    needed = [required_baudrate(rate) for rate in rates]
    assert needed == sorted(needed)
    assert needed[0] >= 115200


def test_baudrate_warning_is_none_when_fast_enough() -> None:
    assert baudrate_warning(250.0, 921600) is None


def test_meta_roundtrip_keeps_dsp_and_detection(tmp_path) -> None:
    dsp = DspConfig(sample_rate=1000.0, seq_len=64)
    detection = DetectionConfig(on_sigma=5.0)
    meta = DatasetMeta.from_configs(dsp, detection)
    path = meta.save(tmp_path)

    restored = DatasetMeta.load(tmp_path)
    assert path.exists()
    assert restored is not None
    assert restored.dsp_config.sample_rate == 1000.0
    assert restored.dsp_config.seq_len == 64
    assert restored.detection_config.on_sigma == pytest.approx(5.0)


def test_project_config_roundtrip(tmp_path) -> None:
    config = ProjectConfig(
        dsp=DspConfig(sample_rate=1000.0, seq_len=256),
        detection=DetectionConfig(on_sigma=5.0),
    )
    path = config.save(tmp_path / "emg_config.json")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["format_version"] == 1
    assert payload["dsp"]["sample_rate"] == 1000.0

    restored = ProjectConfig.load(path)
    assert restored.sample_rate == 1000.0
    assert restored.seq_len == 256
    assert restored.detection.on_sigma == pytest.approx(5.0)


def test_load_project_config_is_optional(tmp_path) -> None:
    assert load_project_config(tmp_path / "missing.json") is None
    with pytest.raises(FileNotFoundError):
        load_project_config(tmp_path / "missing.json", required=True)


def test_project_config_accepts_flat_sample_shape(tmp_path) -> None:
    path = tmp_path / "emg_config.json"
    path.write_text(
        json.dumps({"sample_rate": 750, "seq_len": 192}),
        encoding="utf-8",
    )

    config = ProjectConfig.load(path)
    assert config.sample_rate == 750
    assert config.seq_len == 192


@pytest.mark.parametrize(
    "dsp",
    [
        DspConfig(sample_rate=241.0, seq_len=128),
        DspConfig(sample_rate=500.0, seq_len=8),
    ],
)
def test_project_config_enforces_acquisition_ranges(dsp: DspConfig) -> None:
    with pytest.raises(ValueError):
        ProjectConfig(dsp=dsp)


def test_project_config_enforces_detection_ranges() -> None:
    with pytest.raises(ValueError):
        ProjectConfig(detection=DetectionConfig(on_sigma=20.0))


def test_ensure_project_config_matches_reports_conflict() -> None:
    config = ProjectConfig(dsp=DspConfig(sample_rate=1000.0, seq_len=256))

    ensure_project_config_matches(
        config,
        sample_rate=1000.0,
        seq_len=256,
        artifact="dataset",
    )

    with pytest.raises(ValueError, match="Project config uses"):
        ensure_project_config_matches(
            config,
            sample_rate=500.0,
            seq_len=128,
            artifact="dataset",
        )
