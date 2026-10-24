"""Training pipeline: stratified split, class balancing, early stopping, metrics."""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .config import (
    ModelConfig,
    ProjectConfig,
    TrainConfig,
    ensure_project_config_matches,
)
from .constants import N_CHANNELS
from .dataset import (
    DatasetIndex,
    EMGDataset,
    NormStats,
    balanced_sampler,
    class_weights,
    stratified_split,
)
from .jsonio import write_json
from .model import EMGSequenceClassifier, save_checkpoint


class TrainingError(RuntimeError):
    """Raised when training cannot start or finish."""


@dataclass(frozen=True)
class ClassificationMetrics:
    """Accuracy and one-vs-rest metrics for a classification split."""

    accuracy: float
    error: float
    sensitivity: float
    specificity: float
    precision: float
    f1: float

    def to_dict(self, prefix: str) -> dict[str, float]:
        return {
            f"{prefix}_accuracy": self.accuracy,
            f"{prefix}_error": self.error,
            f"{prefix}_sensitivity": self.sensitivity,
            f"{prefix}_specificity": self.specificity,
            f"{prefix}_precision": self.precision,
            f"{prefix}_macro_f1": self.f1,
        }


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_metrics: ClassificationMetrics
    val_loss: float
    val_metrics: ClassificationMetrics
    learning_rate: float
    duration_s: float

    @property
    def train_accuracy(self) -> float:
        return self.train_metrics.accuracy

    @property
    def train_error(self) -> float:
        return self.train_metrics.error

    @property
    def train_sensitivity(self) -> float:
        return self.train_metrics.sensitivity

    @property
    def train_specificity(self) -> float:
        return self.train_metrics.specificity

    @property
    def train_precision(self) -> float:
        return self.train_metrics.precision

    @property
    def train_macro_f1(self) -> float:
        return self.train_metrics.f1

    @property
    def val_accuracy(self) -> float:
        return self.val_metrics.accuracy

    @property
    def val_error(self) -> float:
        return self.val_metrics.error

    @property
    def val_sensitivity(self) -> float:
        return self.val_metrics.sensitivity

    @property
    def val_specificity(self) -> float:
        return self.val_metrics.specificity

    @property
    def val_precision(self) -> float:
        return self.val_metrics.precision

    @property
    def val_macro_f1(self) -> float:
        return self.val_metrics.f1

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            **self.train_metrics.to_dict("train"),
            "val_loss": self.val_loss,
            **self.val_metrics.to_dict("val"),
            "learning_rate": self.learning_rate,
            "duration_s": self.duration_s,
        }


@dataclass
class TrainingReport:
    checkpoint: Path
    best_epoch: int
    best_val_loss: float
    train_metrics: ClassificationMetrics
    val_metrics: ClassificationMetrics
    history: list[EpochMetrics]
    label_map: list[str]
    class_counts: dict[str, int]
    warnings: list[str] = field(default_factory=list)
    confusion: NDArray[np.int64] = field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.int64)
    )

    @property
    def best_val_accuracy(self) -> float:
        """Backward-compatible alias for the final validation accuracy."""

        return self.val_metrics.accuracy

    @property
    def best_val_macro_f1(self) -> float:
        """Backward-compatible alias for the final validation macro-F1."""

        return self.val_metrics.f1

    def summary(self) -> str:
        train = self.train_metrics
        val = self.val_metrics
        lines = [
            f"Best epoch: {self.best_epoch}",
            f"Validation loss: {self.best_val_loss:.4f}",
            "",
            f"{'Metric':<14}{'Training':>12}{'Validation':>13}",
            f"{'Accuracy':<14}{train.accuracy:>11.2%}{val.accuracy:>13.2%}",
            f"{'Error rate':<14}{train.error:>11.2%}{val.error:>13.2%}",
            f"{'Sensitivity':<14}{train.sensitivity:>11.2%}{val.sensitivity:>13.2%}",
            f"{'Specificity':<14}{train.specificity:>11.2%}{val.specificity:>13.2%}",
            f"{'Precision':<14}{train.precision:>11.2%}{val.precision:>13.2%}",
            f"{'F1 score':<14}{train.f1:>11.2%}{val.f1:>13.2%}",
            "",
            "Sensitivity, specificity, precision and F1 are macro-averaged over classes.",
            f"Checkpoint: {self.checkpoint}",
        ]
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"  - {warning}" for warning in self.warnings)
        return "\n".join(lines)


def resolve_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def confusion_matrix(
    targets: NDArray[np.int64], predictions: NDArray[np.int64], n_classes: int
) -> NDArray[np.int64]:
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(matrix, (targets, predictions), 1)
    return matrix


def _metrics_from_confusion(
    matrix: NDArray[np.int64],
) -> ClassificationMetrics:
    total = int(matrix.sum())
    if total <= 0:
        raise TrainingError("Cannot compute metrics on an empty set")

    accuracy = float(np.trace(matrix) / total)
    sensitivities: list[float] = []
    specificities: list[float] = []
    precisions: list[float] = []
    f1_scores: list[float] = []

    for class_index in range(matrix.shape[0]):
        true_positive = int(matrix[class_index, class_index])
        false_negative = int(matrix[class_index, :].sum()) - true_positive
        false_positive = int(matrix[:, class_index].sum()) - true_positive
        true_negative = total - true_positive - false_negative - false_positive

        # A class with no true samples and no predictions contributes nothing;
        # skipping it keeps the macro average stable for unused labels.
        if true_positive + false_negative + false_positive == 0:
            continue

        sensitivity = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        specificity = (
            true_negative / (true_negative + false_positive)
            if true_negative + false_positive
            else 0.0
        )
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        f1 = (
            2 * precision * sensitivity / (precision + sensitivity)
            if precision + sensitivity
            else 0.0
        )
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        precisions.append(precision)
        f1_scores.append(f1)

    def _mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    return ClassificationMetrics(
        accuracy=accuracy,
        error=1.0 - accuracy,
        sensitivity=_mean(sensitivities),
        specificity=_mean(specificities),
        precision=_mean(precisions),
        f1=_mean(f1_scores),
    )


def classification_metrics(
    targets: NDArray[np.int64],
    predictions: NDArray[np.int64],
    n_classes: int,
) -> ClassificationMetrics:
    """Return accuracy plus macro-averaged one-vs-rest metrics."""

    return _metrics_from_confusion(
        confusion_matrix(targets, predictions, n_classes)
    )


def macro_f1(
    targets: NDArray[np.int64], predictions: NDArray[np.int64], n_classes: int
) -> float:
    return classification_metrics(targets, predictions, n_classes).f1


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    grad_clip: float,
    n_classes: int,
) -> tuple[float, ClassificationMetrics]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    seen = 0
    targets_all: list[NDArray[np.int64]] = []
    predictions_all: list[NDArray[np.int64]] = []
    with torch.set_grad_enabled(training):
        for features, targets in loader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(features)
            loss = criterion(logits, targets)
            if training:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            total_loss += float(loss.item()) * features.shape[0]
            predictions_all.append(
                logits.argmax(dim=1).detach().cpu().numpy().astype(np.int64)
            )
            targets_all.append(targets.detach().cpu().numpy().astype(np.int64))
            seen += int(features.shape[0])
    if seen == 0:
        raise TrainingError("DataLoader yielded no samples")
    targets_array = np.concatenate(targets_all)
    predictions_array = np.concatenate(predictions_all)
    return (
        total_loss / seen,
        classification_metrics(targets_array, predictions_array, n_classes),
    )


@torch.inference_mode()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    n_classes: int,
) -> tuple[float, ClassificationMetrics, NDArray[np.int64], NDArray[np.int64]]:
    model.eval()
    total_loss = 0.0
    seen = 0
    targets_all: list[NDArray[np.int64]] = []
    predictions_all: list[NDArray[np.int64]] = []
    for features, targets in loader:
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(features)
        total_loss += float(criterion(logits, targets).item()) * features.shape[0]
        seen += int(features.shape[0])
        predictions_all.append(logits.argmax(dim=1).cpu().numpy().astype(np.int64))
        targets_all.append(targets.cpu().numpy().astype(np.int64))
    if seen == 0:
        raise TrainingError("Evaluation loader yielded no samples")
    targets_array = np.concatenate(targets_all)
    predictions_array = np.concatenate(predictions_all)
    matrix = confusion_matrix(targets_array, predictions_array, n_classes)
    return (
        total_loss / seen,
        _metrics_from_confusion(matrix),
        matrix,
        predictions_array,
    )


def train(
    dataset_root: str | Path,
    checkpoint_path: str | Path,
    *,
    train_config: TrainConfig | None = None,
    model_config: ModelConfig | None = None,
    on_epoch: Callable[[EpochMetrics], None] | None = None,
    project_config: ProjectConfig | None = None,
) -> TrainingReport:
    """Train a sequence classifier and save the best checkpoint.

    The checkpoint embeds architecture, label map, DSP/detection configuration
    and normalisation buffers, so the inference app only needs this one file.
    """

    cfg = train_config or TrainConfig()
    mcfg = model_config or ModelConfig()
    _seed_everything(cfg.seed)

    index = DatasetIndex.build(dataset_root)
    if project_config is not None:
        ensure_project_config_matches(
            project_config,
            sample_rate=index.meta.sample_rate,
            seq_len=index.meta.seq_len,
            artifact=f"dataset '{index.root}'",
        )
    if index.n_classes < 2:
        raise TrainingError(
            "At least two classes are required. Add another word to the dataset."
        )
    train_indices, val_indices = stratified_split(
        index.labels, val_fraction=cfg.val_fraction, seed=cfg.seed
    )
    train_dataset = EMGDataset(
        index, train_indices, augment=cfg.augment, seed=cfg.seed
    )
    train_eval_dataset = EMGDataset(index, train_indices)
    val_dataset = EMGDataset(index, val_indices)

    norm_stats = NormStats.compute(train_dataset)
    model = EMGSequenceClassifier(N_CHANNELS, index.n_classes, mcfg)
    model.set_norm_stats(norm_stats.mean, norm_stats.std)

    device = resolve_device(cfg.device)
    model.to(device)

    sampler = balanced_sampler(train_dataset, seed=cfg.seed) if cfg.balanced_sampler else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=cfg.num_workers > 0,
        drop_last=False,
    )
    train_eval_loader = DataLoader(
        train_eval_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=cfg.num_workers > 0,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=cfg.num_workers > 0,
        drop_last=False,
    )

    weights = class_weights(train_dataset.labels, index.n_classes) if cfg.class_weights else None
    criterion = nn.CrossEntropyLoss(weight=weights.to(device) if weights is not None else None)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5
    )

    target = Path(checkpoint_path)
    history_path = target.with_name(f"{target.stem}_history.json")
    history: list[EpochMetrics] = []
    best_f1 = -1.0
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    epochs_without_improvement = 0
    summary_payload: dict[str, Any] = {
        "dataset": str(index.root),
        "label_map": index.label_map,
        "class_counts": index.counts(),
        "train_config": cfg.to_dict(),
        "model_config": mcfg.to_dict(),
        "dsp_config": index.meta.dsp,
        "detection_config": index.meta.detection,
        "warnings": index.warnings,
        "history": [],
        "best": {},
    }

    for epoch in range(1, cfg.epochs + 1):
        started = time.perf_counter()
        train_loss, train_metrics = _train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            cfg.grad_clip,
            index.n_classes,
        )
        val_loss, val_metrics, _, _ = _evaluate(
            model, val_loader, criterion, device, index.n_classes
        )
        scheduler.step(val_loss)
        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            train_metrics=train_metrics,
            val_loss=val_loss,
            val_metrics=val_metrics,
            learning_rate=float(optimizer.param_groups[0]["lr"]),
            duration_s=time.perf_counter() - started,
        )
        history.append(metrics)
        if on_epoch is not None:
            on_epoch(metrics)

        improved = val_metrics.f1 > best_f1 + cfg.min_delta or (
            abs(val_metrics.f1 - best_f1) <= cfg.min_delta
            and val_loss < best_loss - 1e-6
        )
        if improved:
            best_f1, best_loss, best_epoch = val_metrics.f1, val_loss, epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
            checkpoint_metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                **train_metrics.to_dict("train"),
                "val_loss": val_loss,
                **val_metrics.to_dict("val"),
            }
            save_checkpoint(
                target,
                model,
                label_map=index.label_map,
                dsp_config=index.meta.dsp_config,
                detection_config=index.meta.detection_config,
                metrics={
                    **checkpoint_metrics,
                    "n_train": len(train_dataset),
                    "n_val": len(val_dataset),
                },
                train_config=cfg,
                dataset_root=index.root,
            )
            summary_payload["best"] = checkpoint_metrics
        else:
            epochs_without_improvement += 1

        summary_payload["history"] = [entry.to_dict() for entry in history]
        write_json(history_path, summary_payload)

        if epochs_without_improvement >= cfg.patience:
            break

    if best_state is None:
        raise TrainingError("Training finished without producing a checkpoint")

    model.load_state_dict(best_state)
    _, final_train_metrics, _, _ = _evaluate(
        model, train_eval_loader, criterion, device, index.n_classes
    )
    _, final_val_metrics, final_confusion, _ = _evaluate(
        model, val_loader, criterion, device, index.n_classes
    )
    return TrainingReport(
        checkpoint=target,
        best_epoch=best_epoch,
        best_val_loss=best_loss,
        train_metrics=final_train_metrics,
        val_metrics=final_val_metrics,
        history=history,
        label_map=index.label_map,
        class_counts=index.counts(),
        warnings=index.warnings,
        confusion=final_confusion,
    )


def load_training_history(checkpoint_path: str | Path) -> Mapping[str, Any]:
    target = Path(checkpoint_path)
    history_path = target.with_name(f"{target.stem}_history.json")
    with history_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
