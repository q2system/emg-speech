"""Train a word classifier from the recorded examples.

Steps:
    1. Read dataset/ and check that the classes are usable.
    2. Split the examples into training and validation sets.
    3. Train the CNN + BiLSTM model.
    4. Save the best checkpoint to artifacts/model.pt.

Run:
    python train.py
    python train.py --data dataset --out artifacts/model.pt --epochs 40
    python train.py --config emg_config.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from emg_speech import train_model
from emg_speech.jsonio import JsonIOError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the EMG word classifier")
    parser.add_argument("--data", default="dataset", help="dataset folder from record.py")
    parser.add_argument("--out", default="artifacts/model.pt", help="output checkpoint")
    parser.add_argument("--epochs", type=int, default=40, help="maximum training epochs")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | xpu | mps")
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="project config (default: ./emg_config.json when present)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = Path(args.data)
    checkpoint = Path(args.out)

    print("[1/4] Dataset:", dataset.resolve())
    print("      Config:", args.config or "emg_config.json (if present)")
    if not dataset.exists():
        print(f"Dataset folder '{dataset}' does not exist.", file=sys.stderr)
        print("Record examples first with: python record.py", file=sys.stderr)
        return 1

    print("[2/4] Validation split: built automatically from the dataset.")
    print("[3/4] Training for at most", args.epochs, "epochs on", args.device)
    print("[4/4] Best checkpoint will be saved to:", checkpoint.resolve())

    from emg_speech.model import CheckpointError
    from emg_speech.train import TrainingError

    try:
        report = train_model(
            dataset,
            checkpoint,
            epochs=args.epochs,
            device=args.device,
            config_path=args.config,
        )
    except (TrainingError, CheckpointError, ValueError, OSError, JsonIOError) as exc:
        print(f"\nTraining failed: {exc}", file=sys.stderr)
        return 1

    print()
    print(report.summary())
    print()
    print("Checkpoint saved to:", checkpoint.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
