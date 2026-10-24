"""Command-line options for ``pipeline_walkthrough.py``."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the EMG pipeline from saved examples through training."
    )
    parser.add_argument(
        "--data",
        default="dataset",
        help="dataset folder from record.py (use dataset_demo for synthetic data)",
    )
    parser.add_argument("--epochs", type=int, default=4, help="training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="mini-batch size")
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="Adam learning rate"
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.25,
        help="fraction held out for validation",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--device",
        default="cpu",
        help="cpu | cuda | xpu | mps | auto (default: cpu for a classroom run)",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="save eight PNG figures here instead of opening windows",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="do not open plot windows (use with --save-dir)",
    )
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if not 0.0 < args.val_fraction < 1.0:
        parser.error("--val-fraction must be between 0 and 1")
    return args
