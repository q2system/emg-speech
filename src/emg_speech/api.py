"""A small, friendly API for the EMG speech pipeline.

The four top-level scripts only use the functions below.  All of the difficult
work -- UART framing, filtering, event detection, PyTorch and the desktop
windows -- lives inside the rest of the ``emg_speech`` library.

Typical use::

    from emg_speech import record, train_model, recognize, simulate

    record("dataset")
    train_model("dataset", "artifacts/model.pt")
    recognize("artifacts/model.pt")
    simulate("--pty")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .config import DEFAULT_BAUDRATE, TrainConfig


def record(
    dataset: str | Path = "dataset",
    *,
    port: str | None = None,
    baudrate: int = DEFAULT_BAUDRATE,
) -> int:
    """Open the acquisition window and record labelled examples.

    Args:
        dataset: Folder where the class folders and ``.npy`` examples are stored.
        port: Serial port such as ``/dev/ttyUSB0`` or ``COM5``.  If omitted,
            the port can be typed into the window.
        baudrate: UART speed.  It must match the ESP32 firmware.

    Returns:
        The Qt application exit code.
    """

    from .ui.app import run_acquisition

    return run_acquisition(dataset, port=port, baudrate=baudrate)


def train_model(
    dataset: str | Path = "dataset",
    checkpoint: str | Path = "artifacts/model.pt",
    *,
    epochs: int = 40,
    device: str = "auto",
    on_epoch: Callable[[Any], None] | None = None,
) -> Any:
    """Train a classifier from a recorded dataset and save a checkpoint.

    Args:
        dataset: Folder produced by :func:`record`.
        checkpoint: Output ``.pt`` file.
        epochs: Maximum number of training epochs.
        device: ``"auto"``, ``"cpu"``, ``"cuda"``, ``"xpu"`` or ``"mps"``.
        on_epoch: Optional callback receiving one metrics object per epoch.

    Returns:
        A training report containing validation metrics and the label map.
    """

    from .train import train

    config = TrainConfig(epochs=epochs, device=device)
    return train(
        dataset,
        checkpoint,
        train_config=config,
        on_epoch=on_epoch,
    )


def simulate(*options: str) -> int:
    """Run the fake ESP32 stream.

    Args:
        *options: Command-line options forwarded to the simulator, for example
            ``"--pty"`` or ``"--port", "COM5"``.

    Returns:
        The simulator exit code.
    """

    from .apps.simulate import main as run_simulator

    return run_simulator(list(options))


def recognize(
    checkpoint: str | Path = "artifacts/model.pt",
    *,
    port: str | None = None,
    replay: str | Path | None = None,
    baudrate: int = DEFAULT_BAUDRATE,
    min_confidence: float = 0.5,
) -> int:
    """Open the real-time recognition window.

    Args:
        checkpoint: Model file produced by :func:`train_model`.
        port: Serial port.  If omitted, the port can be typed into the window.
        replay: Optional recorded dataset to play instead of a live serial port.
        baudrate: UART speed.  It must match the ESP32 firmware.
        min_confidence: Minimum probability required to type a word.

    Returns:
        The Qt application exit code.
    """

    from .ui.app import run_inference

    return run_inference(
        checkpoint=checkpoint,
        replay_root=replay,
        port=port,
        baudrate=baudrate,
        min_confidence=min_confidence,
    )
