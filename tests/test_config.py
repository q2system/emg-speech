from __future__ import annotations

import pytest

from emg_speech.config import (
    DEFAULT_BAUDRATE,
    DatasetMeta,
    DetectionConfig,
    DspConfig,
    baudrate_warning,
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
