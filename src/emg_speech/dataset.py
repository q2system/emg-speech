"""Training dataset indexing, normalisation statistics and PyTorch datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset, WeightedRandomSampler, get_worker_info

from .config import DatasetMeta, DspConfig
from .constants import N_CHANNELS
from .storage import scan_dataset

Sample = tuple[torch.Tensor, int]


@dataclass
class DatasetIndex:
    """A validated snapshot of ``dataset/<class>/*.npy``."""

    root: Path
    meta: DatasetMeta
    label_map: list[str]
    paths: list[Path]
    labels: NDArray[np.int64]
    warnings: list[str] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return len(self.paths)

    @property
    def n_classes(self) -> int:
        return len(self.label_map)

    def counts(self) -> dict[str, int]:
        values, counts = np.unique(self.labels, return_counts=True)
        return {self.label_map[int(v)]: int(c) for v, c in zip(values, counts)}

    @classmethod
    def build(cls, dataset_root: str | Path) -> "DatasetIndex":
        root = Path(dataset_root).expanduser().resolve()
        classes = scan_dataset(root)
        if not classes:
            raise ValueError(
                f"No labelled samples found in '{root}'. Record a dataset first."
            )

        label_map = sorted(classes.keys())
        paths: list[Path] = []
        labels: list[int] = []
        warnings: list[str] = []

        for label_index, label in enumerate(label_map):
            for path in classes[label]:
                try:
                    array = np.load(path, mmap_mode="r", allow_pickle=False)
                except (OSError, ValueError) as exc:
                    warnings.append(f"skipping unreadable file '{path.name}': {exc}")
                    continue
                if array.ndim != 2 or array.shape[1] != N_CHANNELS:
                    warnings.append(
                        f"skipping '{path.name}': shape {array.shape} is not "
                        f"(seq_len, {N_CHANNELS})"
                    )
                    continue
                paths.append(path)
                labels.append(label_index)

        if not paths:
            raise ValueError(f"All samples in '{root}' are invalid; cannot train.")

        meta = DatasetMeta.load(root)
        if meta is None:
            first = np.load(paths[0], mmap_mode="r", allow_pickle=False)
            seq_len = int(first.shape[0])
            meta = DatasetMeta(
                sample_rate=500.0,
                seq_len=seq_len,
                dsp=DspConfig(sample_rate=500.0, seq_len=seq_len).to_dict(),
            )
            warnings.append(
                "No dataset.json found; assuming 500 Hz. Sample rate only matters "
                "for the DSP event detector, not for the network input."
            )

        # Drop files whose length disagrees with the dataset contract.  Skipping
        # here (instead of inside __getitem__) keeps a training run from dying
        # halfway through an epoch.
        kept_paths: list[Path] = []
        kept_labels: list[int] = []
        mismatched = 0
        for path, label in zip(paths, labels):
            header = np.load(path, mmap_mode="r", allow_pickle=False)
            if header.shape[0] != meta.seq_len:
                mismatched += 1
                continue
            kept_paths.append(path)
            kept_labels.append(label)
        if mismatched:
            warnings.append(
                f"skipped {mismatched} sample(s) whose length differs from "
                f"seq_len={meta.seq_len}"
            )
        if not kept_paths:
            raise ValueError("No samples match the dataset sequence length")

        kept_label_array = np.asarray(kept_labels, dtype=np.int64)
        present = sorted({int(value) for value in kept_label_array})
        if len(present) < len(label_map):
            dropped = [label_map[index] for index in range(len(label_map)) if index not in set(present)]
            warnings.append(
                "dropped classes with no valid samples: " + ", ".join(dropped)
            )
            remap = {old: new for new, old in enumerate(present)}
            label_map = [label_map[old] for old in present]
            kept_label_array = np.asarray(
                [remap[int(value)] for value in kept_label_array], dtype=np.int64
            )

        return cls(
            root,
            meta,
            label_map,
            kept_paths,
            kept_label_array,
            warnings,
        )


class EMGDataset(Dataset[Sample]):
    """Reads fixed-length ``.npy`` samples and returns ``(x, y)`` tensors."""

    def __init__(
        self,
        index: DatasetIndex,
        indices: Sequence[int] | None = None,
        *,
        augment: bool = False,
        seed: int = 0,
    ) -> None:
        self.index = index
        self.indices = (
            list(range(index.n_samples)) if indices is None else [int(i) for i in indices]
        )
        self.augment = augment
        self._seed = seed
        self._rng: np.random.Generator | None = None
        self._warned: set[str] = set()

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def labels(self) -> NDArray[np.int64]:
        return self.index.labels[self.indices]

    def _generator(self) -> np.random.Generator:
        if self._rng is None:
            worker = get_worker_info()
            worker_id = worker.id if worker is not None else 0
            self._rng = np.random.default_rng(self._seed + 7919 * worker_id)
        return self._rng

    def __getitem__(self, item: int) -> Sample:
        source_index = self.indices[item]
        path = self.index.paths[source_index]
        array = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
        expected = (self.index.meta.seq_len, N_CHANNELS)
        if array.shape != expected:
            raise ValueError(
                f"Sample '{path}' has shape {array.shape}, expected {expected}. "
                "Delete it or re-record with the current settings."
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"Sample '{path}' contains NaN or infinite values")

        if self.augment:
            array = self._augment(array)
        return torch.from_numpy(array), int(self.index.labels[source_index])

    def _augment(self, array: NDArray[np.float32]) -> NDArray[np.float32]:
        rng = self._generator()
        output = array.copy()
        # Small temporal shift (+/- 5 %), implemented without wrap-around.
        shift = int(rng.integers(-max(1, len(output) // 20), max(2, len(output) // 20) + 1))
        if shift > 0:
            output[shift:] = array[:-shift]
            output[:shift] = array[0]
        elif shift < 0:
            output[:shift] = array[-shift:]
            output[shift:] = array[-1]
        # EMG gain jitter (the model should not depend on absolute effort).
        emg = output[:, :3]
        emg *= rng.uniform(0.85, 1.15, size=(1, 3)).astype(np.float32)
        # Sensor noise.
        output += rng.normal(0.0, 0.01, size=output.shape).astype(np.float32)
        return output

    def iter_arrays(self) -> Iterator[NDArray[np.float32]]:
        for source_index in self.indices:
            yield np.load(self.index.paths[source_index], allow_pickle=False).astype(
                np.float32, copy=False
            )


@dataclass(frozen=True)
class NormStats:
    mean: NDArray[np.float32]
    std: NDArray[np.float32]

    @classmethod
    def compute(cls, dataset: EMGDataset) -> "NormStats":
        channel_sum = np.zeros(N_CHANNELS, dtype=np.float64)
        channel_sq_sum = np.zeros(N_CHANNELS, dtype=np.float64)
        count = 0
        for array in dataset.iter_arrays():
            if array.shape != (dataset.index.meta.seq_len, N_CHANNELS):
                continue
            values = array.astype(np.float64)
            channel_sum += values.sum(axis=0)
            channel_sq_sum += np.square(values).sum(axis=0)
            count += values.shape[0]
        if count == 0:
            raise ValueError("Cannot compute normalisation statistics from empty data")
        mean = channel_sum / count
        variance = np.maximum(channel_sq_sum / count - mean**2, 1e-8)
        return cls(mean=mean.astype(np.float32), std=np.sqrt(variance).astype(np.float32))

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


def stratified_split(
    labels: NDArray[np.int64],
    *,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Per-class split so every class is represented in train and validation."""

    rng = np.random.default_rng(seed)
    train: list[int] = []
    val: list[int] = []
    for class_index in np.unique(labels):
        members = np.flatnonzero(labels == class_index)
        rng.shuffle(members)
        if members.size < 2:
            train.extend(int(i) for i in members)
            continue
        n_val = int(round(members.size * val_fraction))
        n_val = min(max(n_val, 1), members.size - 1)
        val.extend(int(i) for i in members[:n_val])
        train.extend(int(i) for i in members[n_val:])
    if not train:
        raise ValueError("Training split is empty")
    if not val:
        raise ValueError(
            "Validation split is empty: record at least two samples for one class"
        )
    rng.shuffle(train)
    return train, val


def class_weights(labels: NDArray[np.int64], n_classes: int) -> torch.Tensor:
    """Inverse-frequency weights for CrossEntropyLoss."""

    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    weights = np.zeros(n_classes, dtype=np.float64)
    present = counts > 0
    weights[present] = counts.sum() / (present.sum() * counts[present])
    return torch.tensor(weights, dtype=torch.float32)


def balanced_sampler(
    dataset: EMGDataset, *, seed: int = 0
) -> WeightedRandomSampler:
    labels = dataset.labels
    counts = np.bincount(labels).astype(np.float64)
    sample_weights = 1.0 / counts[labels]
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )


def load_array(path: str | Path) -> NDArray[np.float32]:
    array = np.load(Path(path), allow_pickle=False)
    if array.ndim != 2 or array.shape[1] != N_CHANNELS:
        raise ValueError(f"Invalid sample shape {array.shape} in '{path}'")
    return np.asarray(array, dtype=np.float32)
