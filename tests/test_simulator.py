from __future__ import annotations

import os
import time

import numpy as np
import pytest

from emg_speech.config import DetectionConfig, DspConfig
from emg_speech.dsp import extract_sample
from emg_speech.protocol import parse_frame
from emg_speech.simulator import SimulatedSource, encode_line, style_for_word


def test_encoded_frame_matches_the_firmware_protocol() -> None:
    frame = np.array(
        [512, 498, 505, 1.2, -0.4, 0.1, 0.02, -0.01, 0.98], dtype=np.float32
    )
    line = encode_line(frame)
    assert len(line) < 120  # keeps the 921600 baud budget honest
    parsed = parse_frame(line)
    assert parsed.shape == (9,)
    assert np.allclose(parsed, frame, atol=0.06)


def test_encoded_idle_noise_survives_quantisation() -> None:
    """Coarse formatting must not turn the idle noise floor into zeros.

    At 0.02 EMG standard deviation, one decimal place quantises most samples to
    exactly 0.0, which makes ``NoiseProfile`` collapse and the detector reject
    real bursts on the serial path.
    """

    source = SimulatedSource(
        sample_rate=500.0, mode="manual", start_idle_seconds=0.0, seed=4
    )
    source.open()
    idle = source.pull(400)
    decoded = np.stack([parse_frame(encode_line(frame)) for frame in idle])
    assert float(decoded[:, :3].std()) > 0.005
    assert np.count_nonzero(decoded[:, :3]) > 500


def test_word_styles_are_stable_and_bounded() -> None:
    assert style_for_word("alpha") == style_for_word("alpha")
    assert style_for_word("hello") == style_for_word("hello")
    assert style_for_word("hello") != style_for_word("world")
    style = style_for_word("an_unusual_word")
    assert 35.0 <= style.carrier_hz <= 104.0
    assert 0 <= style.envelope_kind < 4
    assert 0 <= style.motion_axis < 3


def test_auto_mode_emits_detectable_gestures() -> None:
    fs = 500.0
    source = SimulatedSource(
        ("alpha", "beta"),
        sample_rate=fs,
        mode="auto",
        interval_seconds=1.0,
        start_idle_seconds=0.5,
        seed=5,
    )
    source.open()
    frames = source.pull(int(4.5 * fs))
    cues = source.drain_cues()

    assert len(cues) >= 2
    assert {cue.word for cue in cues} <= {"alpha", "beta"}
    assert source.stats.gestures == len(cues)
    assert frames.shape == (int(4.5 * fs), 9)
    assert np.all(np.isfinite(frames))

    cue = cues[0]
    start = max(0, cue.frame - int(0.4 * fs))
    end = min(len(frames), cue.frame + int(1.0 * fs))
    extracted = extract_sample(
        frames[start:end], DspConfig(sample_rate=fs), DetectionConfig()
    )
    assert extracted.data.shape == (128, 9)
    assert extracted.event_ms >= DetectionConfig().min_event_ms


def test_manual_mode_waits_for_a_trigger() -> None:
    fs = 500.0
    source = SimulatedSource(
        ("hello",),
        sample_rate=fs,
        mode="manual",
        start_idle_seconds=0.0,
        seed=2,
    )
    source.open()
    idle = source.pull(int(1.0 * fs))
    assert not source.drain_cues()
    assert source.stats.gestures == 0

    assert source.trigger("Hello World") == "hello_world"
    frames = source.pull(int(1.5 * fs))
    cues = source.drain_cues()
    assert len(cues) == 1
    assert cues[0].word == "hello_world"
    assert source.stats.word_counts == {"hello_world": 1}

    combined = np.concatenate([idle, frames], axis=0)
    start = max(0, cues[0].frame - int(0.35 * fs))
    end = min(len(combined), cues[0].frame + int(1.0 * fs))
    extracted = extract_sample(
        combined[start:end], DspConfig(sample_rate=fs), DetectionConfig()
    )
    assert extracted.data.shape == (128, 9)


def test_manual_mode_keeps_a_continuous_imu() -> None:
    fs = 500.0
    source = SimulatedSource(
        sample_rate=fs, mode="manual", start_idle_seconds=0.0, seed=9
    )
    source.open()
    first = source.pull(250)
    source.trigger("alpha")
    second = source.pull(500)
    timeline = np.concatenate([first, second], axis=0)
    steps = np.abs(np.diff(timeline[:, 3:], axis=0))
    assert float(steps.max()) < 0.1, "IMU must not jump between segments"


@pytest.mark.skipif(os.name == "nt", reason="POSIX PTYs only")
def test_pty_serial_roundtrip() -> None:
    from emg_speech.sources import PtyWriter, SerialSource

    writer = PtyWriter()
    source = SerialSource(writer.slave_name, 115200, deassert_control_lines=False)
    source.open()
    try:
        simulator = SimulatedSource(
            sample_rate=500.0, mode="manual", start_idle_seconds=0.0, seed=3
        )
        simulator.open()
        frames = simulator.pull(60)
        writer.write_lines([encode_line(frame) for frame in frames])

        received = []
        deadline = time.monotonic() + 3.0
        while len(received) < len(frames) and time.monotonic() < deadline:
            received.extend(source.read_available())
            time.sleep(0.002)

        assert len(received) >= 40, "the serial source should receive the PTY stream"
        assert np.allclose(
            np.stack(received[:20]), frames[:20], atol=0.06
        )
    finally:
        source.close()
        writer.close()
