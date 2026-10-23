"""Create or update the project-wide EMG configuration file.

The file is stored as ``emg_config.json`` in the current working directory
unless ``--config PATH`` is supplied.  The recording, simulation, training and
inference commands read it automatically, so the same signal shape is used
everywhere.

Examples::

    emg-config --sample-rate 1000 --seq-len 256
    emg-config --sample-rate 500 --seq-len 128 --on-sigma 4.5
    emg-config --config configs/lab.json --sample-rate 750
    emg-config                 # show the current values without changing them
    emg-config --reset         # write the built-in defaults
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

from emg_speech.config import (
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SEQ_LEN,
    MAX_SAMPLE_RATE,
    MAX_SEQ_LEN,
    MIN_SAMPLE_RATE,
    MIN_SEQ_LEN,
    PROJECT_CONFIG_FILENAME,
    ProjectConfig,
    load_project_config,
    resolve_project_config_path,
)
from emg_speech.jsonio import JsonIOError


def _print_config(project: ProjectConfig, path, *, written: bool) -> None:
    state = "written" if written else "existing" if path.exists() else "not created yet"
    print(f"Config file: {path} ({state})")
    print(f"Sample rate: {project.sample_rate:g} Hz")
    print(f"Sequence length: {project.seq_len}")
    detection = project.detection
    print(
        "Detection: "
        f"onset {detection.on_sigma:g}σ · "
        f"event {detection.min_event_ms:g}–{detection.max_event_ms:g} ms · "
        f"post margin {detection.post_margin_ms:g} ms"
    )
    if not written:
        print(
            "Change values with e.g. "
            f"`emg-config --sample-rate {DEFAULT_SAMPLE_RATE:g} "
            f"--seq-len {DEFAULT_SEQ_LEN}`."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create or update the project-wide EMG signal configuration"
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help=f"config file to read/write (default: ./{PROJECT_CONFIG_FILENAME})",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=None,
        metavar="HZ",
        help=f"frames per second ({MIN_SAMPLE_RATE:g}–{MAX_SAMPLE_RATE:g})",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        metavar="N",
        help=f"time steps per extracted sample ({MIN_SEQ_LEN}–{MAX_SEQ_LEN})",
    )
    parser.add_argument(
        "--on-sigma",
        type=float,
        default=None,
        help="muscle-onset threshold in noise standard deviations",
    )
    parser.add_argument(
        "--min-event-ms",
        type=float,
        default=None,
        help="minimum accepted burst length in milliseconds",
    )
    parser.add_argument(
        "--max-event-ms",
        type=float,
        default=None,
        help="maximum accepted burst length in milliseconds",
    )
    parser.add_argument(
        "--post-margin-ms",
        type=float,
        default=None,
        help="extra signal saved after the burst in milliseconds",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="start from the built-in defaults before applying any changes",
    )
    args = parser.parse_args(argv)

    target = resolve_project_config_path(args.config)
    try:
        project = load_project_config(target) or ProjectConfig()
    except (ValueError, OSError, JsonIOError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    changed = args.reset or any(
        value is not None
        for value in (
            args.sample_rate,
            args.seq_len,
            args.on_sigma,
            args.min_event_ms,
            args.max_event_ms,
            args.post_margin_ms,
        )
    )

    if args.reset:
        project = ProjectConfig()

    try:
        dsp = project.dsp
        if args.sample_rate is not None:
            dsp = replace(dsp, sample_rate=args.sample_rate)
        if args.seq_len is not None:
            dsp = replace(dsp, seq_len=args.seq_len)

        detection = project.detection
        if args.on_sigma is not None:
            detection = replace(detection, on_sigma=args.on_sigma)
        if args.min_event_ms is not None:
            detection = replace(detection, min_event_ms=args.min_event_ms)
        if args.max_event_ms is not None:
            detection = replace(detection, max_event_ms=args.max_event_ms)
        if args.post_margin_ms is not None:
            detection = replace(detection, post_margin_ms=args.post_margin_ms)

        project = ProjectConfig(dsp=dsp, detection=detection)
    except ValueError as exc:
        print(f"Invalid configuration: {exc}", file=sys.stderr)
        return 2

    if changed:
        try:
            target = project.save(target)
        except (OSError, JsonIOError) as exc:
            print(f"Could not write '{target}': {exc}", file=sys.stderr)
            return 2

    _print_config(project, target, written=changed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
