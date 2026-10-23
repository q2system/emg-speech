from __future__ import annotations

import numpy as np
import pytest

from emg_speech.config import DatasetMeta, DetectionConfig, DspConfig
from emg_speech.storage import (
    ClassNameError,
    ClassNotEmptyError,
    DatasetMismatchError,
    class_counts,
    ensure_class,
    list_classes,
    remove_class,
    remove_sample,
    sanitize_class_name,
    save_sample,
    scan_dataset,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Hello World", "hello_world"),
        ("  EMG/Word! ", "emg_word"),
        ("naïve_tone", "naïve_tone"),
        ("run--fast", "run--fast"),
    ],
)
def test_sanitize_class_name(raw: str, expected: str) -> None:
    assert sanitize_class_name(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "///", "_hidden", ".hidden", "con"])
def test_sanitize_rejects_unsafe_names(raw: str) -> None:
    with pytest.raises(ClassNameError):
        sanitize_class_name(raw)


def test_save_scan_and_remove_sample(tmp_path) -> None:
    meta = DatasetMeta.from_configs(DspConfig(sample_rate=500, seq_len=16), detection=DetectionConfig())
    sample = np.random.default_rng(0).normal(size=(16, 9)).astype(np.float32)
    path = save_sample(tmp_path, "Hello World", sample, meta)
    assert path.exists()
    assert path.suffix == ".npy"
    np.testing.assert_allclose(np.load(path), sample)

    classes = list_classes(tmp_path)
    assert len(classes) == 1
    assert classes[0].name == "hello_world"
    assert class_counts(tmp_path) == {"hello_world": 1}
    assert scan_dataset(tmp_path)["hello_world"] == [path]

    remove_sample(path)
    assert not path.exists()


def test_metadata_shape_mismatch_is_rejected(tmp_path) -> None:
    meta = DatasetMeta.from_configs(DspConfig(seq_len=16), detection=DetectionConfig())
    sample = np.zeros((16, 9), dtype=np.float32)
    save_sample(tmp_path, "word", sample, meta)
    with pytest.raises(DatasetMismatchError):
        save_sample(tmp_path, "word", np.zeros((32, 9), dtype=np.float32), meta)


def test_metadata_rate_mismatch_is_rejected(tmp_path) -> None:
    first = DatasetMeta.from_configs(DspConfig(sample_rate=500, seq_len=16), detection=DetectionConfig())
    second = DatasetMeta.from_configs(DspConfig(sample_rate=1000, seq_len=16), detection=DetectionConfig())
    save_sample(tmp_path, "word", np.zeros((16, 9), dtype=np.float32), first)
    with pytest.raises(DatasetMismatchError):
        save_sample(tmp_path, "word", np.zeros((16, 9), dtype=np.float32), second)


def test_invalid_array_shape_is_rejected(tmp_path) -> None:
    meta = DatasetMeta.from_configs(DspConfig(seq_len=16), detection=DetectionConfig())
    with pytest.raises(DatasetMismatchError):
        save_sample(tmp_path, "word", np.zeros((16, 3), dtype=np.float32), meta)
    with pytest.raises(DatasetMismatchError):
        save_sample(tmp_path, "word", np.full((16, 9), np.nan), meta)


def test_only_empty_classes_can_be_deleted(tmp_path) -> None:
    ensure_class(tmp_path, "word")
    remove_class(tmp_path, "word")

    meta = DatasetMeta.from_configs(DspConfig(seq_len=16), detection=DetectionConfig())
    save_sample(tmp_path, "word", np.zeros((16, 9), dtype=np.float32), meta)
    with pytest.raises(ClassNotEmptyError):
        remove_class(tmp_path, "word")


def test_ring_buffer_accepts_a_2d_array() -> None:
    import numpy as np

    from emg_speech.ring_buffer import FrameRingBuffer

    buffer = FrameRingBuffer(8)
    frames = np.zeros((4, 9), dtype=np.float32)
    buffer.extend(frames)  # type: ignore[arg-type]
    assert len(buffer) == 4
    assert buffer.next_index == 4
    data, start = buffer.get_range(0, 4)
    assert start == 0
    assert data.shape == (4, 9)
