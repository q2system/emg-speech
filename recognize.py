"""Recognise words from a live 9-channel EMG stream.

    1. Read the checkpoint (weights, labels and DSP settings).
    2. Connect to the ESP32, or replay a recorded dataset.
    3. Detect each muscle burst, classify it, and type the word.

Run:
    python recognize.py --checkpoint artifacts/model.pt --port /dev/ttyUSB0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from emg_speech import recognize
from emg_speech.config import DEFAULT_BAUDRATE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recognise words from EMG")
    parser.add_argument("--checkpoint", default="artifacts/model.pt", help="trained model")
    parser.add_argument("--port", default=None, help="serial port, e.g. /dev/ttyUSB0 or COM5")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUDRATE, help="UART baud rate")
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="project config (default: ./emg_config.json when present)",
    )
    parser.add_argument(
        "--replay",
        default=None,
        metavar="DATASET",
        help="replay a recorded dataset instead of using hardware",
    )
    parser.add_argument("--min-confidence", type=float, default=0.5, help="typing threshold (0-1)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = Path(args.checkpoint)

    print("[1/4] Reading checkpoint:", checkpoint.resolve())
    if not checkpoint.exists():
        print(f"Checkpoint '{checkpoint}' was not found.", file=sys.stderr)
        print("Train one first with: python train.py", file=sys.stderr)
        return 1

    from emg_speech.model import CheckpointError, read_checkpoint_summary

    try:
        summary = read_checkpoint_summary(checkpoint)
    except CheckpointError as exc:
        print(f"Could not read the checkpoint: {exc}", file=sys.stderr)
        return 1

    print("      Classes:", ", ".join(summary.label_map))
    print("      Sample rate:", summary.dsp_config.sample_rate, "Hz")
    print("      Sequence length:", summary.dsp_config.seq_len, "samples")

    if args.replay:
        replay = Path(args.replay)
        print("[2/4] Replay dataset:", replay.resolve())
        if not replay.exists():
            print(f"Replay dataset '{replay}' was not found.", file=sys.stderr)
            return 1
        input_description = f"replay {replay}"
    else:
        print("[2/4] Serial port:", args.port or "(choose it in the window)")
        input_description = f"serial port {args.port or '(choose in window)'}"

    print("[3/4] Input:", input_description)
    print("      Config:", args.config or "emg_config.json (if present)")
    print("[4/4] Opening the recognition window...")

    return recognize(
        checkpoint,
        port=args.port,
        replay=args.replay,
        baudrate=args.baud,
        min_confidence=args.min_confidence,
        config_path=args.config,
    )


if __name__ == "__main__":
    raise SystemExit(main())
