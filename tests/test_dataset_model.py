from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from emg_speech.config import DetectionConfig, DspConfig, ModelConfig
from emg_speech.constants import N_CHANNELS
from emg_speech.dataset import (
    DatasetIndex,
    EMGDataset,
    NormStats,
    stratified_split,
)
from emg_speech.model import (
    EMGSequenceClassifier,
    load_checkpoint,
    predict_probabilities,
    read_checkpoint_summary,
    save_checkpoint,
)


def test_index_and_splits(demo_dataset: Path) -> None:
    index = DatasetIndex.build(demo_dataset)
    assert index.n_samples == 32
    assert index.n_classes == 4
    assert index.label_map == sorted(index.label_map)
    assert sum(index.counts().values()) == 32

    train, val = stratified_split(index.labels, val_fraction=0.25, seed=3)
    assert sorted(train + val) == list(range(32))
    assert len(set(train) & set(val)) == 0
    train_labels = set(index.labels[train])
    val_labels = set(index.labels[val])
    assert train_labels == val_labels == set(range(4))


def test_norm_stats_and_batching(demo_dataset: Path) -> None:
    index = DatasetIndex.build(demo_dataset)
    dataset = EMGDataset(index)
    stats = NormStats.compute(dataset)
    assert stats.mean.shape == (N_CHANNELS,)
    assert stats.std.shape == (N_CHANNELS,)
    assert np.all(np.isfinite(stats.mean))
    assert np.all(stats.std > 0)

    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    features, targets = next(iter(loader))
    assert features.shape == (8, index.meta.seq_len, N_CHANNELS)
    assert features.dtype == torch.float32
    assert targets.dtype == torch.int64
    assert torch.isfinite(features).all()


def test_augmentation_preserves_shape(demo_dataset: Path) -> None:
    index = DatasetIndex.build(demo_dataset)
    dataset = EMGDataset(index, augment=True, seed=1)
    sample, label = dataset[0]
    assert sample.shape == (index.meta.seq_len, N_CHANNELS)
    assert 0 <= label < index.n_classes


def test_model_forward_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    config = ModelConfig(hidden_size=16, conv_channels=8, attention_dim=8)
    model = EMGSequenceClassifier(N_CHANNELS, 3, config)
    model.set_norm_stats(np.zeros(N_CHANNELS), np.ones(N_CHANNELS) * 2)
    logits = model(torch.zeros(2, 128, N_CHANNELS))
    assert logits.shape == (2, 3)

    checkpoint = save_checkpoint(
        tmp_path / "model.pt",
        model,
        label_map=["alpha", "beta", "gamma"],
        dsp_config=DspConfig(),
        detection_config=DetectionConfig(),
        metrics={"val_accuracy": 0.9},
        dataset_root="dataset",
    )
    summary = read_checkpoint_summary(checkpoint)
    assert summary.label_map == ["alpha", "beta", "gamma"]
    assert summary.n_parameters > 0
    assert summary.metrics["val_accuracy"] == 0.9

    info = load_checkpoint(checkpoint, device="cpu")
    assert info.label_map == ["alpha", "beta", "gamma"]
    assert info.dsp_config.seq_len == DspConfig().seq_len
    probabilities = predict_probabilities(
        info.model, np.zeros((DspConfig().seq_len, N_CHANNELS), dtype=np.float32)
    )
    assert probabilities.shape == (3,)
    assert float(probabilities.sum()) == pytest.approx(1.0, abs=1e-5)


def test_checkpoint_missing_is_reported(tmp_path: Path) -> None:
    from emg_speech.model import CheckpointError

    with pytest.raises(CheckpointError):
        load_checkpoint(tmp_path / "missing.pt")


def test_classes_without_valid_samples_are_dropped(demo_dataset: Path, tmp_path: Path) -> None:
    import shutil

    root = tmp_path / "dataset"
    shutil.copytree(demo_dataset, root)
    ghost = root / "ghost"
    ghost.mkdir()
    (ghost / "ghost_broken.npy").write_bytes(b"not a numpy array")

    index = DatasetIndex.build(root)
    assert index.label_map == ["alpha", "beta", "delta", "gamma"]
    assert index.n_classes == 4
    assert int(index.labels.max()) == 3
    assert any("ghost" in warning for warning in index.warnings)
