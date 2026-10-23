from __future__ import annotations

import numpy as np
import pytest

from emg_speech.config import DetectionConfig, DspConfig
from emg_speech.dsp import (
    EventDetector,
    EventNotFound,
    NoiseProfile,
    StreamingPipeline,
    extract_sample,
    normalize_event,
    resample_to_length,
)
from emg_speech.synthetic import _make_take

FS = 500.0
DSP = DspConfig(sample_rate=FS, seq_len=128)
DETECTION = DetectionConfig()


def test_streaming_and_batch_filters_are_identical() -> None:
    rng = np.random.default_rng(3)
    take = _make_take(rng, 1, sample_rate=FS)
    streaming = StreamingPipeline(DSP)
    per_frame = np.stack([streaming.process_frame(frame)[0] for frame in take[:400]])

    batch = StreamingPipeline(DSP)
    filtered, envelope = batch.process_array(take[:400])
    np.testing.assert_allclose(per_frame, filtered, atol=1e-6)
    assert envelope.shape == (400,)
    assert np.all(np.isfinite(envelope))


def test_noise_profile_uses_robust_statistics() -> None:
    values = np.full(100, 0.01)
    values[::10] = 5.0  # artefacts must not dominate the estimate
    profile = NoiseProfile.from_values(values, min_std=1e-3)
    assert profile.mean == pytest.approx(0.01)
    assert profile.std >= 1e-3
    assert profile.on_threshold(DETECTION) > profile.off_threshold(DETECTION)


def test_detects_a_single_burst_and_respects_margins() -> None:
    rng = np.random.default_rng(12)
    take = _make_take(rng, 2, sample_rate=FS)
    pipeline = StreamingPipeline(DSP)
    _, envelope = pipeline.process_array(take)
    profile = NoiseProfile.from_values(envelope[:175], min_std=DETECTION.min_noise_std)
    detector = EventDetector(DETECTION, FS)
    detector.set_noise_profile(profile)
    spans = detector.run(envelope)
    assert len(spans) == 1
    assert 150.0 <= spans[0].duration_ms(FS) <= 600.0
    assert spans[0].peak > profile.on_threshold(DETECTION)


def test_short_blip_is_rejected() -> None:
    detector = EventDetector(DETECTION, FS)
    detector.set_noise_profile(NoiseProfile(0.0, 1e-3))
    envelope = np.zeros(1500)
    envelope[500:510] = 1.0  # 20 ms
    assert detector.run(envelope) == []
    assert detector.last_rejection is not None
    assert "too short" in detector.last_rejection


def test_long_event_is_capped_at_max_duration() -> None:
    detector = EventDetector(DETECTION, FS)
    detector.set_noise_profile(NoiseProfile(0.0, 1e-3))
    spans = detector.run(np.ones(2000))
    assert len(spans) >= 1
    assert spans[0].duration_ms(FS) == pytest.approx(DETECTION.max_event_ms, abs=5)


def test_detector_honours_start_index_offset() -> None:
    detector = EventDetector(DETECTION, FS, start_index=1000)
    detector.set_noise_profile(NoiseProfile(0.0, 1e-3))
    envelope = np.zeros(400)
    envelope[100:300] = 1.0  # 400 ms
    spans = detector.run(envelope)
    assert len(spans) == 1
    assert spans[0].start >= 1095
    assert spans[0].end <= 1305


def test_resample_preserves_endpoints_and_length() -> None:
    data = np.repeat(np.arange(10, dtype=np.float32)[:, None], 9, axis=1)
    output = resample_to_length(data, 32)
    assert output.shape == (32, 9)
    assert output[0, 0] == pytest.approx(0.0)
    assert output[-1, 0] == pytest.approx(9.0)
    assert np.all(np.diff(output[:, 0]) >= 0)


def test_extract_sample_normalises_shape_and_emg_rms() -> None:
    rng = np.random.default_rng(5)
    take = _make_take(rng, 1, sample_rate=FS)
    extracted = extract_sample(take, DSP, DETECTION)
    assert extracted.data.shape == (DSP.seq_len, 9)
    assert extracted.data.dtype == np.float32
    assert np.all(np.isfinite(extracted.data))
    assert 150.0 <= extracted.event_ms <= 600.0
    rms = float(np.sqrt(np.mean(extracted.data[:, :3] ** 2)))
    assert rms == pytest.approx(1.0, abs=1e-3)
    assert extracted.noise_profile is not None


def test_extract_sample_is_amplitude_invariant() -> None:
    rng = np.random.default_rng(21)
    take = _make_take(rng, 3, sample_rate=FS)
    louder = take.copy()
    louder[:, :3] *= 5.0
    quiet_result = extract_sample(take, DSP, DETECTION)
    loud_result = extract_sample(louder, DSP, DETECTION)
    np.testing.assert_allclose(
        quiet_result.data[:, :3], loud_result.data[:, :3], atol=1e-4
    )
    np.testing.assert_allclose(
        quiet_result.data[:, 3:], loud_result.data[:, 3:], atol=1e-4
    )


def test_silence_raises_event_not_found() -> None:
    rng = np.random.default_rng(8)
    take = rng.normal(0.0, 0.02, size=(700, 9)).astype(np.float32)
    with pytest.raises(EventNotFound):
        extract_sample(take, DSP, DETECTION)


def test_normalize_event_handles_single_row() -> None:
    sample = normalize_event(np.ones((1, 9), dtype=np.float32), DSP)
    assert sample.shape == (DSP.seq_len, 9)
    assert np.allclose(sample[:, :3], 1.0)


def test_invalid_band_configuration_is_rejected() -> None:
    with pytest.raises(ValueError):
        DspConfig(sample_rate=100.0)
    with pytest.raises(ValueError):
        DetectionConfig(on_sigma=2.0, off_sigma=2.0)
