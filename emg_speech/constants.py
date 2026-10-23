"""Project-wide constants.

The channel order is part of the hardware contract between the ESP32 firmware and
every Python component.  Never reorder this tuple without retraining the model.
"""

from __future__ import annotations

from pathlib import Path

CHANNELS: tuple[str, ...] = (
    "emg1",
    "emg2",
    "emg3",
    "gx",
    "gy",
    "gz",
    "ax",
    "ay",
    "az",
)

EMG_CHANNELS: tuple[int, ...] = (0, 1, 2)
IMU_CHANNELS: tuple[int, ...] = (3, 4, 5, 6, 7, 8)
N_CHANNELS: int = len(CHANNELS)

DATASET_META_FILENAME: str = "dataset.json"
DATASET_FORMAT_VERSION: int = 1
SAMPLE_SUFFIX: str = ".npy"

DEFAULT_DATASET_DIR: Path = Path("dataset")
DEFAULT_ARTIFACT_DIR: Path = Path("artifacts")
CHECKPOINT_FILENAME: str = "model.pt"
