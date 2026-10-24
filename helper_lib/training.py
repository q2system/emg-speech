"""Forward pass, loss, training loop, and evaluation for the walkthrough."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from emg_speech.train import (
    ClassificationMetrics,
    classification_metrics,
    confusion_matrix,
)

from .data import BatchData, ModelBundle, PreparedData
from .reporting import print_metric_table, section


@dataclass
class ForwardResult:
    logits: torch.Tensor
    probabilities: torch.Tensor
    label_map: list[str]
    true_label: str
    predicted_label: str


@dataclass
class LossResult:
    loss: float
    per_sample: np.ndarray
    targets: np.ndarray
    label_map: list[str]
    gradient_names: list[str]
    gradient_norms: list[float]


@dataclass
class TrainingResult:
    label_map: list[str]
    best_epoch: int
    best_f1: float
    train_loss: float
    val_loss: float
    train_metrics: ClassificationMetrics
    val_metrics: ClassificationMetrics
    val_targets: np.ndarray
    val_predictions: np.ndarray
    history: dict[str, list[float]]


@dataclass
class EvaluationResult:
    label_map: list[str]
    train_metrics: ClassificationMetrics
    val_metrics: ClassificationMetrics
    matrix: np.ndarray


def forward_pass(bundle: ModelBundle, batch: BatchData) -> ForwardResult:
    """Section 4: run one batch through an untrained network."""

    section(5, "THE MODEL: A FORWARD PASS")
    print("Now push one real batch through the network from Section 4.")
    features = batch.features.to(bundle.device)
    bundle.model.eval()
    with torch.no_grad():
        logits = bundle.model(features).cpu()
        probabilities = torch.softmax(logits, dim=1)

    true_label = batch.label_map[int(batch.targets[0])]
    predicted_label = batch.label_map[int(probabilities[0].argmax())]
    print(f"Input shape : {tuple(batch.features.shape)}")
    print(f"Output shape: {tuple(logits.shape)}")
    print("One row per example, one score per word.")
    print(
        f"Example 0: true='{true_label}', "
        f"untrained prediction='{predicted_label}'"
    )
    print("The weights are still random, so this prediction is not meaningful yet.")

    return ForwardResult(
        logits=logits,
        probabilities=probabilities,
        label_map=batch.label_map,
        true_label=true_label,
        predicted_label=predicted_label,
    )


def compute_loss_and_gradients(
    bundle: ModelBundle,
    batch: BatchData,
) -> LossResult:
    """Section 5: compute the loss and backpropagate gradients."""

    section(6, "LOSS AND BACKPROPAGATION")
    print("The loss measures how wrong the current scores are.")
    print("Backpropagation computes a gradient for every trainable weight.")
    print("This section does not update the weights yet; the training loop does.")

    bundle.model.train()
    inputs = batch.features.to(bundle.device)
    labels = batch.targets.to(bundle.device)
    bundle.optimizer.zero_grad(set_to_none=True)
    logits = bundle.model(inputs)
    loss = bundle.criterion(logits, labels)
    per_sample = F.cross_entropy(
        logits,
        labels,
        weight=(
            bundle.criterion.weight
            if isinstance(bundle.criterion, nn.CrossEntropyLoss)
            else None
        ),
        reduction="none",
    )
    loss.backward()

    group_names = ("conv stem", "recurrent", "attention", "classifier")
    group_modules = (
        bundle.model.stem,
        bundle.model.rnn,
        bundle.model.attention,
        bundle.model.head,
    )
    gradient_norms: list[float] = []
    for module in group_modules:
        squared = 0.0
        for parameter in module.parameters():
            if parameter.grad is not None:
                squared += float(parameter.grad.detach().pow(2).sum())
        gradient_norms.append(math.sqrt(squared))
    bundle.optimizer.zero_grad(set_to_none=True)

    print(f"Cross-entropy loss for this batch: {float(loss.item()):.4f}")
    print("Gradient norm per part of the network:")
    for name, value in zip(group_names, gradient_norms):
        print(f"  {name:<12s} {value:.6f}")
    print("Gradients were cleared after measuring them so training starts clean.")

    return LossResult(
        loss=float(loss.item()),
        per_sample=per_sample.detach().cpu().numpy(),
        targets=batch.targets.numpy(),
        label_map=batch.label_map,
        gradient_names=list(group_names),
        gradient_norms=gradient_norms,
    )


def _train_one_epoch(
    bundle: ModelBundle,
    loader: DataLoader,
) -> tuple[float, ClassificationMetrics]:
    bundle.model.train()
    total_loss = 0.0
    all_targets: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []

    for features, targets in loader:
        features = features.to(bundle.device)
        targets = targets.to(bundle.device)
        bundle.optimizer.zero_grad(set_to_none=True)
        logits = bundle.model(features)
        loss = bundle.criterion(logits, targets)
        loss.backward()
        bundle.optimizer.step()

        total_loss += float(loss.item()) * features.shape[0]
        all_predictions.append(
            logits.argmax(dim=1).detach().cpu().numpy().astype(np.int64)
        )
        all_targets.append(targets.detach().cpu().numpy().astype(np.int64))

    target_array = np.concatenate(all_targets)
    prediction_array = np.concatenate(all_predictions)
    metrics = classification_metrics(
        target_array, prediction_array, bundle.model.n_classes
    )
    return total_loss / len(loader.dataset), metrics


@torch.no_grad()
def _evaluate_loader(
    bundle: ModelBundle,
    loader: DataLoader,
) -> tuple[float, np.ndarray, np.ndarray, ClassificationMetrics]:
    bundle.model.eval()
    total_loss = 0.0
    all_targets: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []

    for features, targets in loader:
        features = features.to(bundle.device)
        targets = targets.to(bundle.device)
        logits = bundle.model(features)
        total_loss += float(bundle.criterion(logits, targets).item()) * features.shape[0]
        all_predictions.append(logits.argmax(dim=1).cpu().numpy().astype(np.int64))
        all_targets.append(targets.cpu().numpy().astype(np.int64))

    target_array = np.concatenate(all_targets)
    prediction_array = np.concatenate(all_predictions)
    metrics = classification_metrics(
        target_array, prediction_array, bundle.model.n_classes
    )
    return total_loss / len(loader.dataset), target_array, prediction_array, metrics


def train_model(
    bundle: ModelBundle,
    prepared: PreparedData,
    *,
    epochs: int,
) -> TrainingResult:
    """Section 6: the real training loop."""

    section(7, "TRAINING LOOP")
    print("One epoch means: look at every training example once.")
    print("For each batch: forward -> loss -> backward -> optimizer step.")
    print("After each epoch we measure the validation set.")
    print(f"Training for {epochs} epoch(s) on {bundle.device}.")

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "val_f1": [],
    }
    best_f1 = -1.0
    best_loss = float("inf")
    best_epoch = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        train_loss, train_metrics = _train_one_epoch(bundle, prepared.train_loader)
        val_loss, _, _, val_metrics = _evaluate_loader(bundle, prepared.val_loader)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_metrics.accuracy)
        history["val_accuracy"].append(val_metrics.accuracy)
        history["val_f1"].append(val_metrics.f1)

        print(
            f"epoch {epoch:02d} | "
            f"train loss {train_loss:.4f} acc {train_metrics.accuracy:.1%} | "
            f"val loss {val_loss:.4f} acc {val_metrics.accuracy:.1%} "
            f"F1 {val_metrics.f1:.3f}"
        )

        better_f1 = val_metrics.f1 > best_f1 + 1e-9
        tied_but_lower_loss = (
            abs(val_metrics.f1 - best_f1) <= 1e-9 and val_loss < best_loss
        )
        if better_f1 or tied_but_lower_loss:
            best_f1 = val_metrics.f1
            best_loss = val_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in bundle.model.state_dict().items()
            }

    if best_state is not None:
        bundle.model.load_state_dict(best_state)

    # Evaluate the best weights on the unaugmented training split and on the
    # validation split, just like the real training pipeline.
    train_eval_loader = DataLoader(
        prepared.train_dataset,
        batch_size=prepared.train_loader.batch_size or 32,
        shuffle=False,
    )
    train_loss, _, _, train_metrics = _evaluate_loader(bundle, train_eval_loader)
    val_loss, val_targets, val_predictions, val_metrics = _evaluate_loader(
        bundle, prepared.val_loader
    )

    print(f"\nBest validation epoch: {best_epoch} (macro-F1 {best_f1:.3f})")
    print(f"Training loss of the best model:   {train_loss:.4f}")
    print(f"Validation loss of the best model: {val_loss:.4f}")

    return TrainingResult(
        label_map=prepared.index.label_map,
        best_epoch=best_epoch,
        best_f1=best_f1,
        train_loss=train_loss,
        val_loss=val_loss,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        val_targets=val_targets,
        val_predictions=val_predictions,
        history=history,
    )


def evaluate_model(training: TrainingResult) -> EvaluationResult:
    """Section 7: build the confusion matrix and print the metric table."""

    section(8, "TRAINING RESULTS: CONFUSION MATRIX AND METRICS")
    print("The confusion matrix shows which words were confused with which.")
    print("Rows are the true word; columns are the model's prediction.")
    print("Sensitivity, specificity, precision and F1 are macro-averaged.")
    print("Error rate is 1 - accuracy.")

    matrix = confusion_matrix(
        training.val_targets,
        training.val_predictions,
        len(training.label_map),
    )
    print_metric_table(training.train_metrics, training.val_metrics)
    return EvaluationResult(
        label_map=training.label_map,
        train_metrics=training.train_metrics,
        val_metrics=training.val_metrics,
        matrix=matrix,
    )
