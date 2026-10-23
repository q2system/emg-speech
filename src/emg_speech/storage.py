"""Dataset folders: class management and crash-safe sample storage.

Layout::

    dataset/
        dataset.json                 <- metadata (rate, seq_len, DSP contract)
        hello/
            hello_20261022-101500_ab12cd34.npy
        world/
            ...
"""

from __future__ import annotations

import os
import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .config import DatasetMeta
from .constants import CHANNELS, DATASET_META_FILENAME, N_CHANNELS, SAMPLE_SUFFIX

_CLASS_RE = re.compile(r"[^\w\-]+", flags=re.UNICODE)
_RESERVED_NAMES = {"dataset.json", "con", "prn", "aux", "nul"}


class DatasetError(RuntimeError):
    """Base error for dataset operations."""


class ClassNameError(ValueError, DatasetError):
    """Raised for empty or unsafe class names."""


class DatasetMismatchError(DatasetError):
    """Raised when a new sample does not match the dataset's metadata."""


class ClassNotEmptyError(DatasetError):
    """Raised when trying to delete a class folder that still contains data."""


@dataclass(frozen=True)
class ClassInfo:
    name: str
    path: Path
    count: int


def sanitize_class_name(raw: str) -> str:
    """Turn free text into a safe folder name (``"Hello World!" -> "hello_world"``)."""

    candidate = raw.strip().lower()
    if candidate.startswith(("_", ".")):
        raise ClassNameError("Class name may not start with '_' or '.'")
    name = candidate.replace(" ", "_")
    name = _CLASS_RE.sub("_", name)
    name = re.sub(r"_+", "_", name).strip("_-.")
    if not name:
        raise ClassNameError("Class name must contain at least one letter or digit")
    if name in _RESERVED_NAMES:
        raise ClassNameError(f"'{name}' is a reserved name")
    if len(name) > 48:
        raise ClassNameError("Class name is too long (max 48 characters)")
    return name


def ensure_dataset_root(dataset_root: str | Path) -> Path:
    root = Path(dataset_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def ensure_class(dataset_root: str | Path, class_name: str) -> Path:
    root = ensure_dataset_root(dataset_root)
    name = sanitize_class_name(class_name)
    path = root / name
    if path.exists() and not path.is_dir():
        raise DatasetError(f"'{path}' exists but is not a directory")
    path.mkdir(parents=True, exist_ok=True)
    return path


def count_samples(class_path: Path) -> int:
    try:
        return sum(1 for _ in class_path.glob(f"*{SAMPLE_SUFFIX}"))
    except OSError:
        return 0


def list_classes(dataset_root: str | Path) -> list[ClassInfo]:
    root = Path(dataset_root)
    if not root.exists():
        return []
    classes: list[ClassInfo] = []
    for entry in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if not entry.is_dir() or entry.name.startswith((".", "_")):
            continue
        classes.append(ClassInfo(entry.name, entry, count_samples(entry)))
    return classes


def scan_dataset(dataset_root: str | Path) -> dict[str, list[Path]]:
    """Return ``{class_name: [sample paths]}`` for classes that contain data."""

    root = Path(dataset_root)
    result: dict[str, list[Path]] = {}
    for info in list_classes(root):
        paths = sorted(info.path.glob(f"*{SAMPLE_SUFFIX}"))
        if paths:
            result[info.name] = paths
    return result


def class_counts(dataset_root: str | Path) -> dict[str, int]:
    return {info.name: info.count for info in list_classes(dataset_root)}


def _check_metadata(root: Path, meta: DatasetMeta) -> None:
    existing = DatasetMeta.load(root)
    if existing is None:
        meta.save(root)
        return
    if existing.seq_len != meta.seq_len:
        raise DatasetMismatchError(
            f"This dataset was created with seq_len={existing.seq_len}, "
            f"but the current setting is seq_len={meta.seq_len}. "
            "Start a new dataset folder instead of mixing shapes."
        )
    if abs(existing.sample_rate - meta.sample_rate) > 1e-6:
        raise DatasetMismatchError(
            f"This dataset was created at {existing.sample_rate:g} Hz, "
            f"but the current setting is {meta.sample_rate:g} Hz. "
            "Start a new dataset folder instead of mixing rates."
        )


def save_sample(
    dataset_root: str | Path,
    class_name: str,
    sample: NDArray[np.floating],
    meta: DatasetMeta,
) -> Path:
    """Validate and atomically write one fixed-length sample.

    Returns the path of the new ``.npy`` file.
    """

    array = np.asarray(sample, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != N_CHANNELS:
        raise DatasetMismatchError(
            f"Expected shape (seq_len, {N_CHANNELS}), got {array.shape}"
        )
    if array.shape[0] != meta.seq_len:
        raise DatasetMismatchError(
            f"Expected {meta.seq_len} time steps, got {array.shape[0]}"
        )
    if not np.all(np.isfinite(array)):
        raise DatasetMismatchError("Sample contains NaN or infinite values")

    root = ensure_dataset_root(dataset_root)
    name = sanitize_class_name(class_name)
    _check_metadata(root, meta)
    destination = ensure_class(root, name)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{name}_{stamp}_{uuid.uuid4().hex[:8]}{SAMPLE_SUFFIX}"
    target = destination / filename
    tmp = destination / f".{filename}.tmp"

    try:
        with tmp.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
        os.replace(tmp, target)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise DatasetError(f"Could not write sample '{target}': {exc}") from exc
    return target


def remove_sample(path: str | Path) -> None:
    Path(path).unlink(missing_ok=False)


def remove_class(dataset_root: str | Path, class_name: str) -> None:
    """Delete an *empty* class folder (never destroys recorded data)."""

    path = Path(dataset_root) / sanitize_class_name(class_name)
    if not path.exists():
        return
    if count_samples(path) > 0:
        raise ClassNotEmptyError(
            f"Class '{class_name}' still contains samples; delete them manually first."
        )
    try:
        shutil.rmtree(path)
    except OSError as exc:
        raise DatasetError(f"Could not delete '{path}': {exc}") from exc


def dataset_channels(dataset_root: str | Path) -> tuple[str, ...]:
    meta = DatasetMeta.load(dataset_root)
    return meta.channels if meta else CHANNELS


def dataset_meta_path(dataset_root: str | Path) -> Path:
    return Path(dataset_root) / DATASET_META_FILENAME
