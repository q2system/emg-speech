"""Train the LSTM classifier from a recorded dataset.

    python train.py --data dataset --out artifacts/model.pt --epochs 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from emg_speech.config import (
    ModelConfig,
    TrainConfig,
    load_project_config,
)
from emg_speech.jsonio import JsonIOError
from emg_speech.model import CheckpointError
from emg_speech.train import EpochMetrics, TrainingError, train


def _print_confusion(report) -> None:
    labels = report.label_map
    if report.confusion.size == 0:
        return
    width = max(8, *(len(label) + 1 for label in labels))
    header = " " * (width + 2) + " ".join(f"{label[:width]:>{width}}" for label in labels)
    print("\nValidation confusion matrix (rows = true, columns = predicted)")
    print(header)
    for label, row in zip(labels, report.confusion):
        values = " ".join(f"{int(value):>{width}}" for value in row)
        print(f"{label[:width]:<{width}}  {values}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the EMG sequence classifier")
    parser.add_argument("--data", default="dataset", help="dataset folder from `python record.py`")
    parser.add_argument("--out", default="artifacts/model.pt", help="output checkpoint")
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="project config (default: ./emg_config.json when present)",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0, help="DataLoader worker processes")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | xpu | mps")
    parser.add_argument("--hidden", type=int, default=96, help="LSTM hidden size")
    parser.add_argument("--layers", type=int, default=2, help="number of recurrent layers")
    parser.add_argument("--conv-channels", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument(
        "--bidirectional",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use a bidirectional recurrent layer (default: yes)",
    )
    parser.add_argument(
        "--class-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="weight the loss by inverse class frequency (default: yes)",
    )
    parser.add_argument(
        "--balanced-sampler",
        action="store_true",
        help="resample batches so every class is equally likely",
    )
    parser.add_argument(
        "--augment", action="store_true", help="time-shift/gain/noise augmentation"
    )
    args = parser.parse_args(argv)

    try:
        project_config = load_project_config(
            args.config, required=args.config is not None
        )
    except (ValueError, OSError, JsonIOError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    train_config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        val_fraction=args.val_fraction,
        patience=args.patience,
        seed=args.seed,
        num_workers=args.workers,
        device=args.device,
        class_weights=args.class_weights,
        balanced_sampler=args.balanced_sampler,
        augment=args.augment,
    )
    model_config = ModelConfig(
        conv_channels=args.conv_channels,
        hidden_size=args.hidden,
        num_layers=args.layers,
        bidirectional=args.bidirectional,
        dropout=args.dropout,
    )

    def on_epoch(metrics: EpochMetrics) -> None:
        print(
            f"epoch {metrics.epoch:03d} | "
            f"train loss {metrics.train_loss:.4f} acc {metrics.train_accuracy:.1%} | "
            f"val loss {metrics.val_loss:.4f} acc {metrics.val_accuracy:.1%} "
            f"F1 {metrics.val_macro_f1:.4f} | "
            f"lr {metrics.learning_rate:.1e} | {metrics.duration_s:.1f}s",
            flush=True,
        )

    print(f"Dataset: {Path(args.data).resolve()}")
    print(
        f"Model: {args.layers}× {'Bi' if args.bidirectional else ''}"
        f"LSTM({args.hidden}) + conv{args.conv_channels}"
    )
    try:
        import torch

        threads: int | str = torch.get_num_threads()
    except ImportError:  # pragma: no cover
        threads = "?"
    print(f"Device: {args.device} · torch threads: {threads}")

    try:
        report = train(
            args.data,
            args.out,
            train_config=train_config,
            model_config=model_config,
            on_epoch=on_epoch,
            project_config=project_config,
        )
    except (TrainingError, ValueError, CheckpointError) as exc:
        print(f"\nTraining failed: {exc}", file=sys.stderr)
        return 1

    print()
    print(report.summary())
    print("\nClass distribution:")
    for label, count in sorted(report.class_counts.items()):
        print(f"  {label:<16} {count}")
    _print_confusion(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
