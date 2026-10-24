"""Small printing helpers used by the pipeline walkthrough."""

from __future__ import annotations

from emg_speech.train import ClassificationMetrics


def section(number: int, title: str) -> None:
    """Print one clear divider before a pipeline section."""

    print("\n" + "=" * 78)
    print(f"SECTION {number} - {title}")
    print("=" * 78)


def welcome() -> None:
    print("EMG pipeline walkthrough")
    print("Starting from the examples already saved by the acquisition app.")
    print("Each section prints what happens and shows one figure.")


def goodbye() -> None:
    print("\nWalkthrough complete.")
    print("The real train.py adds early stopping and writes a checkpoint.")


def print_metric_table(
    train_metrics: ClassificationMetrics,
    val_metrics: ClassificationMetrics,
) -> None:
    rows = (
        ("Accuracy", train_metrics.accuracy, val_metrics.accuracy),
        ("Error rate", train_metrics.error, val_metrics.error),
        ("Sensitivity", train_metrics.sensitivity, val_metrics.sensitivity),
        ("Specificity", train_metrics.specificity, val_metrics.specificity),
        ("Precision", train_metrics.precision, val_metrics.precision),
        ("F1 score", train_metrics.f1, val_metrics.f1),
    )
    print(f"\n{'Metric':<14}{'Training':>12}{'Validation':>13}")
    for name, train_value, val_value in rows:
        print(f"{name:<14}{train_value:>11.2%}{val_value:>13.2%}")
