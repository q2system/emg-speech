"""Record labelled EMG examples, one word per button hold.

Steps:
    1. Choose a dataset folder.
    2. Connect the ESP32 (or type the serial port into the window).
    3. Hold Record, say one word, then release.
    4. Repeat for every class, then train with train.py.

Run:
    python record.py
    python record.py --data dataset --port /dev/ttyUSB0 --baud 921600
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from emg_speech import record
from emg_speech.config import DEFAULT_BAUDRATE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record labelled EMG examples")
    parser.add_argument("--data", default="dataset", help="folder for the dataset")
    parser.add_argument("--port", default=None, help="serial port, e.g. /dev/ttyUSB0 or COM5")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUDRATE, help="UART baud rate")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = Path(args.data)

    print("[1/3] Dataset folder:", dataset.resolve())
    print("[2/3] Serial port:", args.port or "(choose it in the window)")
    print("      Baud rate:", args.baud)
    print("[3/3] Opening the acquisition window...")
    print("      Hold Record, say one word, then release. Repeat for each class.")

    return record(dataset, port=args.port, baudrate=args.baud)


if __name__ == "__main__":
    raise SystemExit(main())
