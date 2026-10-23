"""Stream a live simulated biosignal into a serial port.

This lets you test the *hardware* code path (UART framing, threading, DSP,
event detection, live inference) without an ESP32.

POSIX (Linux/macOS) — built-in virtual port, no extra software:

    python simulate.py --pty
    # prints e.g. "/dev/pts/5"; type that path into the app's port box

Windows — require a com0com pair (see ``--setup-windows``):

    python simulate.py --setup-windows
    python simulate.py --port COM5 --baud 921600
    python record.py --port COM6 --baud 921600

Manual gesture control (type a word and press Enter):

    python simulate.py --pty --mode manual --words hello,world

Replay a recorded dataset instead of the synthetic generator:

    python simulate.py --pty --replay dataset

Self-test without opening any port:

    python simulate.py --dry-run --duration 6
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

from emg_speech.config import (
    DEFAULT_BAUDRATE,
    DetectionConfig,
    DspConfig,
)
from emg_speech.dsp import EventNotFound, extract_sample
from emg_speech.simulator import DEFAULT_WORDS, SimulatedSource, encode_line
from emg_speech.sources import (
    FrameSource,
    PtyWriter,
    ReplaySource,
    SerialWriter,
    SourceError,
    available_ports,
)

WINDOWS_HELP = """\
Windows cannot create a virtual COM port from Python.  Install the free
com0com null-modem driver (signed builds are available), then in an elevated
Command Prompt:

    setupc.exe install PortName=COM5 PortName=COM6
    setupc.exe list

Start the simulator on one end and the GUI on the other:

    python simulate.py --port COM5 --baud 921600
    python record.py --port COM6 --baud 921600
    python recognize.py   --checkpoint artifacts/model.pt --port COM6

Both ends must be different ports of the pair: Windows does not allow two
processes to open the same COM port.  The 921600 baud setting only matters for
a real ESP32; a virtual pair transfers data as fast as the OS allows.
"""


def _print_ports() -> int:
    ports = available_ports()
    if not ports:
        print("No serial ports detected.")
        print("Tip: the port box is editable — type a name such as COM3 or /dev/pts/7.")
        return 0
    for info in ports:
        print(f"{info.device:<20} {info.description}")
    return 0


def _build_source(args: argparse.Namespace) -> FrameSource:
    if args.replay:
        return ReplaySource(
            Path(args.replay),
            sample_rate=args.rate,
            loop=True,
        )

    words = args.words
    if args.word:
        words = (args.word,)
    return SimulatedSource(
        words=words,
        sample_rate=args.rate,
        mode=args.mode,
        seed=args.seed,
        interval_seconds=args.interval,
        start_idle_seconds=args.start_idle,
        idle_emg_std=args.noise,
        amplitude=args.amplitude,
    )


def _stdin_loop(source: FrameSource, stop: threading.Event) -> None:
    if not isinstance(source, SimulatedSource):
        return
    while not stop.is_set():
        try:
            line = sys.stdin.readline()
        except (OSError, ValueError):
            return
        if not line:
            return
        word = line.strip()
        if not word:
            continue
        try:
            name = source.trigger(word)
        except ValueError as exc:
            print(f"[sim] invalid word: {exc}", file=sys.stderr, flush=True)
            continue
        print(f"[sim] queued '{name}'", flush=True)


def _pump(
    source: FrameSource,
    writer: SerialWriter | PtyWriter,
    *,
    rate: float,
    duration: float,
    stop: threading.Event,
) -> int:
    started = time.monotonic()
    sent = 0
    last_report = started
    while not stop.is_set():
        now = time.monotonic()
        if duration and now - started >= duration:
            break

        frames = source.read_available()
        if frames:
            writer.write_lines([encode_line(frame) for frame in frames])
            sent += len(frames)

        drain = getattr(source, "drain_cues", None)
        if drain is not None:
            for cue in drain():
                print(
                    f"[sim] {cue.time_s(rate):7.2f}s  gesture '{cue.word}'",
                    flush=True,
                )

        if now - last_report >= 5.0:
            last_report = now
            stats = getattr(source, "stats", None)
            gestures = stats.gestures if stats is not None else 0
            print(
                f"[sim] sent {sent:,} frames ({sent / rate:,.1f}s) · "
                f"{gestures} gestures · pending {writer.pending_bytes:,} B · "
                f"dropped {writer.dropped_bytes:,} B",
                flush=True,
            )
        time.sleep(0.002)

    return sent


def _dry_run(args: argparse.Namespace) -> int:
    if args.replay:
        print("--dry-run cannot be combined with --replay", file=sys.stderr)
        return 2
    source = _build_source(args)
    if not isinstance(source, SimulatedSource):
        return 2
    source.open()
    duration = args.duration if args.duration > 0 else 5.0
    count = max(1, int(round(duration * args.rate)))
    frames = source.pull(count)
    cues = source.drain_cues()

    print(f"Generated {len(frames):,} frames ({len(frames) / args.rate:.1f}s) "
          f"at {args.rate:g} Hz")
    print(f"Words: {', '.join(source.words)}")
    print(f"Gestures: {source.stats.gestures}")
    for cue in cues:
        print(f"  {cue.time_s(args.rate):7.2f}s  {cue.word}")
    print(f"Encoded frame ({len(encode_line(frames[0]))} bytes):")
    print("  " + encode_line(frames[0]))

    if cues:
        dsp = DspConfig(sample_rate=args.rate)
        detection = DetectionConfig()
        cue = cues[0]
        start = max(0, cue.frame - int(0.35 * args.rate))
        end = min(len(frames), cue.frame + int(1.0 * args.rate))
        try:
            extracted = extract_sample(frames[start:end], dsp, detection)
        except EventNotFound as exc:
            print(f"Detector self-test: FAILED ({exc})", file=sys.stderr)
            return 1
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Detector self-test: FAILED ({exc})", file=sys.stderr)
            return 1
        print(
            f"Detector self-test: OK — {extracted.event_ms:.0f} ms burst, "
            f"{extracted.data.shape[0]} time steps"
        )
    return 0


def _make_writer(args: argparse.Namespace) -> SerialWriter | PtyWriter:
    if args.pty:
        return PtyWriter()
    if args.port:
        return SerialWriter(args.port, args.baud)
    raise ValueError("specify --port or --pty")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stream a live simulated 9-channel signal into a serial port"
    )
    parser.add_argument("--port", help="serial port to write to, e.g. COM5 or /dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUDRATE, help="UART baud rate")
    parser.add_argument("--pty", action="store_true", help="create a virtual PTY pair (Linux/macOS)")
    parser.add_argument("--list-ports", action="store_true", help="list detected serial ports")
    parser.add_argument("--setup-windows", action="store_true", help="show com0com instructions")
    parser.add_argument("--replay", metavar="DATASET", default=None, help="loop a recorded dataset instead")
    parser.add_argument(
        "--words",
        default=",".join(DEFAULT_WORDS),
        help=f"comma-separated vocabulary (default: {','.join(DEFAULT_WORDS)})",
    )
    parser.add_argument("--word", default=None, help="single word shorthand")
    parser.add_argument("--mode", choices=("auto", "manual"), default="auto")
    parser.add_argument("--interval", type=float, default=3.0, help="seconds between gestures (auto)")
    parser.add_argument("--start-idle", type=float, default=2.0, help="idle seconds before the first gesture")
    parser.add_argument("--noise", type=float, default=0.02, help="idle EMG standard deviation")
    parser.add_argument("--amplitude", type=float, default=1.1, help="gesture EMG amplitude")
    parser.add_argument("--rate", type=float, default=500.0, help="frames per second")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--duration", type=float, default=0.0, help="stop after N seconds (0 = forever)")
    parser.add_argument("--dry-run", action="store_true", help="generate without opening a port")
    args = parser.parse_args(argv)

    if args.list_ports:
        return _print_ports()
    if args.setup_windows:
        print(WINDOWS_HELP)
        return 0
    if args.rate <= 0 or args.interval <= 0:
        print("--rate and --interval must be positive", file=sys.stderr)
        return 2

    args.words = tuple(word.strip() for word in args.words.split(",") if word.strip())
    if not args.words and not args.word:
        print("--words must contain at least one word", file=sys.stderr)
        return 2

    if args.dry_run:
        return _dry_run(args)

    if args.pty and sys.platform.startswith("win"):
        print(WINDOWS_HELP, file=sys.stderr)
        return 2

    try:
        source = _build_source(args)
        writer = _make_writer(args)
    except (ValueError, SourceError) as exc:
        print(f"[sim] {exc}", file=sys.stderr)
        return 2

    if isinstance(writer, PtyWriter):
        print(
            f"[sim] virtual serial port: {writer.slave_name}\n"
            f"[sim] connect the GUI or CLI to --port {writer.slave_name}",
            flush=True,
        )

    stop = threading.Event()
    try:
        writer.open()
        source.open()
    except SourceError as exc:
        print(f"[sim] {exc}", file=sys.stderr)
        writer.close()
        return 2

    if args.mode == "manual" and isinstance(source, SimulatedSource):
        print("[sim] manual mode: type a word and press Enter to play it.", flush=True)
        threading.Thread(
            target=_stdin_loop, args=(source, stop), daemon=True
        ).start()
    elif isinstance(source, ReplaySource):
        print(f"[sim] replaying {args.replay} (looping)", flush=True)

    print(
        f"[sim] streaming {args.rate:g} frames/s from {source.description} "
        f"to {writer.description} — Ctrl+C to stop",
        flush=True,
    )
    if args.pty:
        time.sleep(1.0)  # let the user read/copy the PTY path

    sent = 0
    try:
        sent = _pump(
            source,
            writer,
            rate=args.rate,
            duration=args.duration,
            stop=stop,
        )
    except KeyboardInterrupt:
        sent = 0
    except (SourceError, OSError) as exc:
        print(f"[sim] write error: {exc}", file=sys.stderr)
    finally:
        stop.set()
        source.close()
        pending = writer.pending_bytes
        dropped = writer.dropped_bytes
        writer.close()

    stats = getattr(source, "stats", None)
    gestures = stats.gestures if stats is not None else 0
    print()
    print(
        f"[sim] stopped after {sent:,} frames · {gestures} gestures · "
        f"{pending:,} B pending · {dropped:,} B dropped"
    )
    if stats is not None and stats.word_counts:
        breakdown = ", ".join(
            f"{word}: {count}" for word, count in sorted(stats.word_counts.items())
        )
        print(f"[sim] gestures by word: {breakdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
