"""Loading, splitting, batching, and model setup for the walkthrough."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from emg_speech.config import ModelConfig
from emg_speech.constants import N_CHANNELS
from emg_speech.dataset import (
    DatasetIndex,
    EMGDataset,
    NormStats,
    class_weights,
    stratified_split,
)
from emg_speech.model import EMGSequenceClassifier
from emg_speech.train import resolve_device

from .reporting import section


@dataclass
class PreparedData:
    """Everything the later sections need after the split and normalisation."""

    index: DatasetIndex
    train_indices: list[int]
    val_indices: list[int]
    train_dataset: EMGDataset
    val_dataset: EMGDataset
    norm_stats: NormStats
    train_loader: DataLoader
    val_loader: DataLoader
    train_counts: np.ndarray
    val_counts: np.ndarray


@dataclass
class BatchData:
    """One mini-batch, ready for the model."""

    features: torch.Tensor
    targets: torch.Tensor
    label_map: list[str]
    mean: np.ndarray
    std: np.ndarray


@dataclass
class ModelBundle:
    """The model plus the training objects it needs."""

    model: EMGSequenceClassifier
    device: torch.device
    criterion: nn.Module
    optimizer: torch.optim.Optimizer
    class_weights: torch.Tensor


def load_database(dataset_path: str | Path) -> DatasetIndex:
    """Section 1: read the examples saved by the acquisition app."""

    section(1, "PRE-PROCESSING: THE SAVED DATABASE")
    root = Path(dataset_path).expanduser()
    if not root.exists():
        raise ValueError(
            f"Dataset folder not found: {root}\n"
            "Record examples with: python record.py\n"
            "Or try the synthetic demo: python pipeline_walkthrough.py "
            "--data dataset_demo"
        )

    index = DatasetIndex.build(root)
    if index.n_classes < 2:
        raise ValueError(
            "The walkthrough needs at least two word classes. "
            "Record another word first."
        )

    print("Acquisition has already detected each word and saved one .npy file.")
    print("This section is the starting point for everything that follows.")
    print(f"Dataset folder : {index.root}")
    print(f"Sample rate    : {index.meta.sample_rate:g} Hz")
    print(f"Sequence length: {index.meta.seq_len} time steps")
    print(f"Classes        : {', '.join(index.label_map)}")
    print(f"Total examples : {index.n_samples}")
    print(
        "Class counts   : "
        + ", ".join(f"{label}={count}" for label, count in index.counts().items())
    )
    for warning in index.warnings:
        print(f"Warning        : {warning}")
    return index


def prepare_data(
    index: DatasetIndex,
    *,
    val_fraction: float,
    seed: int,
    batch_size: int,
) -> PreparedData:
    """Section 2: split the examples and compute normalisation statistics."""

    section(2, "PRE-PROCESSING: SPLIT AND NORMALISE")
    print("The examples are split per class, so every word appears in both parts.")
    print("Normalisation statistics are computed from the training split only.")
    print("This prevents the validation split from leaking into training.")

    np.random.seed(seed)
    torch.manual_seed(seed)
    train_indices, val_indices = stratified_split(
        index.labels, val_fraction=val_fraction, seed=seed
    )
    train_dataset = EMGDataset(index, train_indices)
    val_dataset = EMGDataset(index, val_indices)
    norm_stats = NormStats.compute(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    train_counts = np.bincount(
        index.labels[train_indices], minlength=index.n_classes
    )
    val_counts = np.bincount(index.labels[val_indices], minlength=index.n_classes)
    print(f"Training examples  : {len(train_indices)}")
    print(f"Validation examples: {len(val_indices)}")
    print("Per-channel normalisation statistics:")
    for channel, name in enumerate(index.meta.channels):
        print(
            f"  {name:4s} mean={norm_stats.mean[channel]: .4f} "
            f"std={norm_stats.std[channel]:.4f}"
        )

    return PreparedData(
        index=index,
        train_indices=train_indices,
        val_indices=val_indices,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        norm_stats=norm_stats,
        train_loader=train_loader,
        val_loader=val_loader,
        train_counts=train_counts,
        val_counts=val_counts,
    )


def make_batch(prepared: PreparedData) -> BatchData:
    """Section 3: take one mini-batch from the training loader."""

    section(3, "BATCHING: THE MODEL INPUT")
    print("A DataLoader groups examples into mini-batches.")
    print("One batch has shape (batch, time, channels).")

    features, targets = next(iter(prepared.train_loader))
    print(f"features shape : {tuple(features.shape)}  dtype={features.dtype}")
    print(f"targets shape  : {tuple(targets.shape)}  dtype={targets.dtype}")
    print(f"targets        : {targets.tolist()}")

    return BatchData(
        features=features,
        targets=targets,
        label_map=prepared.index.label_map,
        mean=prepared.norm_stats.mean,
        std=prepared.norm_stats.std,
    )


def build_model(
    prepared: PreparedData,
    *,
    device_name: str,
    learning_rate: float,
) -> ModelBundle:
    """Section 4 setup: create the network and its training objects."""

    section(4, "THE MODEL: ARCHITECTURE")
    print("The model is: Conv1d stem -> BiLSTM -> attention pooling -> Linear.")
    print("The normalisation statistics are stored inside the model itself.")

    model = EMGSequenceClassifier(
        N_CHANNELS,
        prepared.index.n_classes,
        ModelConfig(
            conv_channels=16,
            kernel_size=5,
            hidden_size=32,
            num_layers=1,
            bidirectional=True,
            dropout=0.1,
            attention_dim=16,
        ),
    )
    model.set_norm_stats(prepared.norm_stats.mean, prepared.norm_stats.std)

    device = resolve_device(device_name)
    model.to(device)
    weights = class_weights(
        prepared.train_dataset.labels, prepared.index.n_classes
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Trainable parameters: {model.n_parameters:,}")
    print(f"Using device: {device}")
    print(
        "Class weights (inverse frequency): "
        + ", ".join(
            f"{label}={float(weight):.2f}"
            for label, weight in zip(prepared.index.label_map, weights.cpu())
        )
    )
    return ModelBundle(
        model=model,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        class_weights=weights,
    )
