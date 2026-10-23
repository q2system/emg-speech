"""Training pipeline: stratified split, class balancing, early stopping, metrics."""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
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


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    val_macro_f1: float
    learning_rate: float
    duration_s: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingReport:
    checkpoint: Path
    best_epoch: int
    best_val_loss: float
    best_val_accuracy: float
    best_val_macro_f1: float
    history: list[EpochMetrics]
    label_map: list[str]
    class_counts: dict[str, int]
    warnings: list[str] = field(default_factory=list)
    confusion: NDArray[np.int64] = field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.int64)
    )

    def summary(self) -> str:
        lines = [
            f"Best epoch: {self.best_epoch}",
            f"Validation loss: {self.best_val_loss:.4f}",
            f"Validation accuracy: {self.best_val_accuracy:.2%}",
            f"Validation macro-F1: {self.best_val_macro_f1:.4f}",
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


def macro_f1(
    targets: NDArray[np.int64], predictions: NDArray[np.int64], n_classes: int
) -> float:
    scores: list[float] = []
    for class_index in range(n_classes):
        true_positive = int(np.sum((predictions == class_index) & (targets == class_index)))
        false_positive = int(np.sum((predictions == class_index) & (targets != class_index)))
        false_negative = int(np.sum((predictions != class_index) & (targets == class_index)))
        if true_positive + false_positive + false_negative == 0:
            continue
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        scores.append(
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return float(np.mean(scores)) if scores else 0.0


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    grad_clip: float,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    seen = 0
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
            correct += int((logits.argmax(dim=1) == targets).sum().item())
            seen += int(features.shape[0])
    if seen == 0:
        raise TrainingError("DataLoader yielded no samples")
    return total_loss / seen, correct / seen


@torch.inference_mode()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    n_classes: int,
) -> tuple[float, float, float, NDArray[np.int64], NDArray[np.int64]]:
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
        raise TrainingError("Validation loader yielded no samples")
    targets_array = np.concatenate(targets_all)
    predictions_array = np.concatenate(predictions_all)
    accuracy = float(np.mean(targets_array == predictions_array))
    return (
        total_loss / seen,
        accuracy,
        macro_f1(targets_array, predictions_array, n_classes),
        confusion_matrix(targets_array, predictions_array, n_classes),
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
        train_loss, train_accuracy = _train_epoch(
            model, train_loader, criterion, optimizer, device, cfg.grad_clip
        )
        val_loss, val_accuracy, val_f1, _, _ = _evaluate(
            model, val_loader, criterion, device, index.n_classes
        )
        scheduler.step(val_loss)
        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
            val_macro_f1=val_f1,
            learning_rate=float(optimizer.param_groups[0]["lr"]),
            duration_s=time.perf_counter() - started,
        )
        history.append(metrics)
        if on_epoch is not None:
            on_epoch(metrics)

        improved = val_f1 > best_f1 + cfg.min_delta or (
            abs(val_f1 - best_f1) <= cfg.min_delta and val_loss < best_loss - 1e-6
        )
        if improved:
            best_f1, best_loss, best_epoch = val_f1, val_loss, epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
            save_checkpoint(
                target,
                model,
                label_map=index.label_map,
                dsp_config=index.meta.dsp_config,
                detection_config=index.meta.detection_config,
                metrics={
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "val_macro_f1": val_f1,
                    "train_accuracy": train_accuracy,
                    "n_train": len(train_dataset),
                    "n_val": len(val_dataset),
                },
                train_config=cfg,
                dataset_root=index.root,
            )
            summary_payload["best"] = {
                "epoch": epoch,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_macro_f1": val_f1,
            }
        else:
            epochs_without_improvement += 1

        summary_payload["history"] = [entry.to_dict() for entry in history]
        write_json(history_path, summary_payload)

        if epochs_without_improvement >= cfg.patience:
            break

    if best_state is None:
        raise TrainingError("Training finished without producing a checkpoint")

    model.load_state_dict(best_state)
    _, final_accuracy, final_f1, final_confusion, _ = _evaluate(
        model, val_loader, criterion, device, index.n_classes
    )
    return TrainingReport(
        checkpoint=target,
        best_epoch=best_epoch,
        best_val_loss=best_loss,
        best_val_accuracy=final_accuracy,
        best_val_macro_f1=final_f1,
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
