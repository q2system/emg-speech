"""Launch the real-time inference tool.

    python recognize.py --checkpoint artifacts/model.pt --port /dev/ttyUSB0
    python recognize.py --checkpoint artifacts/model.pt --port COM6
    python recognize.py --checkpoint artifacts/model.pt --replay dataset_demo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from emg_speech.config import DEFAULT_BAUDRATE


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Real-time EMG speech recognition")
    parser.add_argument(
        "--checkpoint", default="artifacts/model.pt", help="trained .pt checkpoint"
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="project config (default: ./emg_config.json when present)",
    )
    parser.add_argument("--port", default=None, help="serial port (ignored with --replay)")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUDRATE, help="UART baud rate")
    parser.add_argument(
        "--replay",
        default=None,
        metavar="DATASET",
        help="replay a recorded dataset instead of hardware (demo/testing)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="minimum probability required to type a word (0-1)",
    )
    args = parser.parse_args(argv)

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        print(
            f"Checkpoint '{checkpoint}' not found. Train one first:\n"
            f"  python train.py --data dataset --out {checkpoint}",
            file=sys.stderr,
        )
        return 1
    if args.replay and not Path(args.replay).exists():
        print(f"Replay dataset '{args.replay}' not found.", file=sys.stderr)
        return 1

    try:
        from emg_speech.ui.app import run_inference
    except ImportError as exc:  # pragma: no cover - dependency hint
        print(f"Could not start the GUI: {exc}", file=sys.stderr)
        print("Install the desktop dependencies with: pip install -r requirements.txt", file=sys.stderr)
        return 1

    return run_inference(
        checkpoint=checkpoint,
        replay_root=args.replay,
        port=args.port,
        baudrate=args.baud,
        min_confidence=args.min_confidence,
        config_path=args.config,
    )


if __name__ == "__main__":
    raise SystemExit(main())
