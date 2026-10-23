"""Launch the data acquisition & annotation tool.

    python -m apps.acquire --data dataset --port /dev/ttyUSB0 --baud 921600
    python -m apps.acquire --data dataset --port COM6 --baud 921600
"""

from __future__ import annotations

import argparse
import sys

from emg_speech.config import DEFAULT_BAUDRATE


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EMG data acquisition & annotation tool")
    parser.add_argument("--data", default="dataset", help="dataset folder (created if missing)")
    parser.add_argument("--port", default=None, help="serial port, e.g. /dev/ttyUSB0 or COM5")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUDRATE, help="UART baud rate")
    args = parser.parse_args(argv)

    try:
        from ui.app import run_acquisition
    except ImportError as exc:  # pragma: no cover - dependency hint
        print(f"Could not start the GUI: {exc}", file=sys.stderr)
        print("Install the desktop dependencies with: pip install -r requirements.txt", file=sys.stderr)
        return 1
    return run_acquisition(args.data, port=args.port, baudrate=args.baud)


if __name__ == "__main__":
    raise SystemExit(main())
