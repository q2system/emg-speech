from __future__ import annotations

import numpy as np
import pytest

from emg_speech.constants import N_CHANNELS
from emg_speech.protocol import (
    FrameParseError,
    FrameParser,
    LineFramer,
    parse_frame,
)

VALID = (
    "emg1:512,emg2:498,emg3:505,gx:1.2,gy:-0.4,gz:0.1,"
    "ax:0.02,ay:-0.01,az:0.98"
)


def test_parse_keyed_frame_in_channel_order() -> None:
    frame = parse_frame(VALID)
    assert frame.shape == (N_CHANNELS,)
    assert frame.dtype == np.float32
    np.testing.assert_allclose(
        frame, [512, 498, 505, 1.2, -0.4, 0.1, 0.02, -0.01, 0.98], rtol=1e-6
    )


def test_parse_is_case_and_space_insensitive() -> None:
    line = " EMG1:1, emg2:2,emg3:3,GX:4,GY:5,GZ:6,AX:7,AY:8,AZ:9 \r"
    frame = parse_frame(line)
    np.testing.assert_array_equal(frame, np.arange(1, 10, dtype=np.float32))


def test_parse_positional_fallback() -> None:
    frame = parse_frame("1,2,3,4,5,6,7,8,9")
    np.testing.assert_array_equal(frame, np.arange(1, 10, dtype=np.float32))


def test_parse_tolerates_unknown_extra_fields() -> None:
    frame = parse_frame(VALID + ",battery:87")
    assert frame[0] == pytest.approx(512.0)


def test_parse_rejects_missing_channel() -> None:
    with pytest.raises(FrameParseError, match="missing channels"):
        parse_frame("emg1:1,emg2:2,emg3:3,gx:4,gy:5,gz:6,ax:7,ay:8")


def test_parse_rejects_bad_number() -> None:
    with pytest.raises(FrameParseError, match="invalid number"):
        parse_frame(VALID.replace("gx:1.2", "gx:oops"))


def test_parse_rejects_non_finite() -> None:
    with pytest.raises(FrameParseError):
        parse_frame(VALID.replace("gx:1.2", "gx:nan"))
    with pytest.raises(FrameParseError):
        parse_frame(VALID.replace("gx:1.2", "gx:inf"))


def test_parse_rejects_too_few_positional_values() -> None:
    with pytest.raises(FrameParseError, match="expected 9"):
        parse_frame("1,2,3")


def test_parser_counts_malformed_lines() -> None:
    parser = FrameParser()
    assert parser.parse(VALID) is not None
    assert parser.parse("garbage") is None
    assert parser.stats.parsed == 1
    assert parser.stats.malformed == 1
    assert parser.stats.last_error


def test_line_framer_handles_partial_reads_and_crlf() -> None:
    framer = LineFramer()
    assert framer.feed(b"emg1:1,emg2:2") == []
    lines = framer.feed(
        b",emg3:3,gx:4,gy:5,gz:6,ax:7,ay:8,az:9\r\n"
        b"1,2,3,4,5,6,7,8,9\n"
    )
    assert len(lines) == 2
    assert "\r" not in lines[0]


def test_line_framer_drops_oversized_garbage_and_resynchronises() -> None:
    framer = LineFramer(max_line_bytes=32)
    framer.feed(b"x" * 40)
    lines = framer.feed(b"1,2,3,4,5,6,7,8,9\n")
    assert len(lines) == 1
    assert framer.dropped == 1
    assert parse_frame(lines[0]).shape == (N_CHANNELS,)
