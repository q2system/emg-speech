"""Create a fake ESP32 serial stream for testing without hardware.

Steps:
    1. Start this script with --pty (Linux/macOS) or a COM-port pair (Windows).
    2. Read the serial port name printed by the simulator.
    3. Type that name into record.py or recognize.py.

Run:
    python simulate.py --pty
    python simulate.py --pty --mode manual --words hello,world
    python simulate.py --port COM5 --baud 921600
    python simulate.py --pty --config emg_config.json
"""

from __future__ import annotations

import sys

from _bootstrap import add_src_to_path

add_src_to_path()

from emg_speech import simulate


def main() -> int:
    if len(sys.argv) == 1:
        print(__doc__)
        print("Choose a mode: --pty (Linux/macOS) or --port COM5 (Windows).")
        return 0
    if sys.argv[1] in {"-h", "--help"}:
        return simulate("--help")

    print("[1/3] Starting the simulated EMG signal source.")
    print("[2/3] The simulator will print a serial port name.")
    print("[3/3] Use that name in record.py or recognize.py. Press Ctrl+C to stop.")
    return simulate(*sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
