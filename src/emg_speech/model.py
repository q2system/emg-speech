"""PyTorch model and checkpoint handling."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from .config import (
    DetectionConfig,
    DspConfig,
    ModelConfig,
    TrainConfig,
    utc_now_iso,
)
from .constants import N_CHANNELS

CHECKPOINT_FORMAT_VERSION = 1


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is missing, corrupt or incompatible."""


class EMGSequenceClassifier(nn.Module):
    """Conv stem -> (Bi)LSTM/GRU -> attention pooling -> linear classifier.

    The input normalisation buffers live inside the module and are saved in the
    ``state_dict``, so training and inference can never disagree about scaling.
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        config: ModelConfig | None = None,
    ) -> None:
        super().__init__()
        if n_channels < 1 or n_classes < 1:
            raise ValueError("n_channels and n_classes must be positive")
        self.n_channels = int(n_channels)
        self.n_classes = int(n_classes)
        self.config = config or ModelConfig()

        cfg = self.config
        groups = 8 if cfg.conv_channels % 8 == 0 else (4 if cfg.conv_channels % 4 == 0 else 1)
        self.stem = nn.Sequential(
            nn.Conv1d(
                n_channels,
                cfg.conv_channels,
                kernel_size=cfg.kernel_size,
                padding=cfg.kernel_size // 2,
            ),
            nn.GroupNorm(groups, cfg.conv_channels),
            nn.GELU(),
            nn.Dropout(cfg.conv_dropout),
            nn.Conv1d(
                cfg.conv_channels,
                cfg.conv_channels,
                kernel_size=cfg.kernel_size,
                stride=2,
                padding=cfg.kernel_size // 2,
            ),
            nn.GroupNorm(groups, cfg.conv_channels),
            nn.GELU(),
        )

        rnn_cls = nn.LSTM if cfg.rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=cfg.conv_channels,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            bidirectional=cfg.bidirectional,
        )
        directions = 2 if cfg.bidirectional else 1
        rnn_output = cfg.hidden_size * directions

        self.attention = nn.Sequential(
            nn.Linear(rnn_output, cfg.attention_dim),
            nn.Tanh(),
            nn.Linear(cfg.attention_dim, 1),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(rnn_output),
            nn.Dropout(cfg.dropout),
            nn.Linear(rnn_output, n_classes),
        )

        self.register_buffer("input_mean", torch.zeros(1, 1, n_channels))
        self.register_buffer("input_std", torch.ones(1, 1, n_channels))

    def set_norm_stats(
        self,
        mean: NDArray[np.floating] | Tensor,
        std: NDArray[np.floating] | Tensor,
    ) -> None:
        mean_t = torch.as_tensor(np.asarray(mean), dtype=torch.float32).reshape(1, 1, -1)
        std_t = torch.as_tensor(np.asarray(std), dtype=torch.float32).reshape(1, 1, -1)
        if mean_t.numel() != self.n_channels or std_t.numel() != self.n_channels:
            raise ValueError("normalisation statistics must have one value per channel")
        if not torch.all(torch.isfinite(mean_t)) or not torch.all(torch.isfinite(std_t)):
            raise ValueError("normalisation statistics contain non-finite values")
        with torch.no_grad():
            self.input_mean.copy_(mean_t)
            self.input_std.copy_(torch.clamp(std_t, min=1e-6))

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[2] != self.n_channels:
            raise ValueError(
                f"expected input (batch, time, {self.n_channels}), got "
                f"{tuple(x.shape)}"
            )
        if x.shape[1] < 2:
            raise ValueError("sequence must contain at least two time steps")
        x = (x - self.input_mean) / self.input_std
        x = x.transpose(1, 2)
        x = self.stem(x)
        x = x.transpose(1, 2)
        features, _ = self.rnn(x)
        weights = torch.softmax(self.attention(features), dim=1)
        pooled = torch.sum(features * weights, dim=1)
        return self.head(pooled)

    @property
    def n_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


@torch.inference_mode()
def predict_probabilities(
    model: EMGSequenceClassifier,
    sample: NDArray[np.floating] | Tensor,
    *,
    device: torch.device | str = "cpu",
) -> NDArray[np.float32]:
    """Return class probabilities for one ``(time, channels)`` sample."""

    tensor = torch.as_tensor(np.asarray(sample), dtype=torch.float32)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError("sample must have shape (time, channels)")
    model.eval()
    tensor = tensor.to(device)
    logits = model(tensor)
    return torch.softmax(logits, dim=1)[0].float().cpu().numpy()


@dataclass
class CheckpointInfo:
    model: EMGSequenceClassifier
    label_map: list[str]
    model_config: ModelConfig
    dsp_config: DspConfig
    detection_config: DetectionConfig
    metrics: dict[str, Any]
    trained_at: str
    dataset_root: str
    path: Path
    class_sentences: dict[str, str] = field(default_factory=dict)

    @property
    def n_classes(self) -> int:
        return len(self.label_map)


@dataclass(frozen=True)
class CheckpointSummary:
    """Lightweight checkpoint metadata (no model instantiation)."""

    path: Path
    label_map: list[str]
    model_config: ModelConfig
    dsp_config: DspConfig
    detection_config: DetectionConfig
    metrics: dict[str, Any]
    trained_at: str
    n_parameters: int
    class_sentences: dict[str, str] = field(default_factory=dict)


def _normalise_class_sentences(
    value: Any, label_map: list[str]
) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    labels = set(label_map)
    sentences: dict[str, str] = {}
    for raw_label, raw_sentence in value.items():
        label = str(raw_label)
        sentence = str(raw_sentence).strip()
        if label in labels and sentence:
            sentences[label] = sentence
    return sentences


def _read_payload(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        raise CheckpointError(f"Checkpoint not found: '{source}'")
    try:
        payload = torch.load(source, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise CheckpointError(f"Could not read checkpoint '{source}': {exc}") from exc
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise CheckpointError(f"'{source}' is not a valid EMG checkpoint")
    if not list(payload.get("label_map", [])):
        raise CheckpointError("Checkpoint has an empty label map")
    version = int(payload.get("format_version", 0))
    if version > CHECKPOINT_FORMAT_VERSION:
        raise CheckpointError(
            f"Checkpoint format v{version} is newer than this application "
            f"(v{CHECKPOINT_FORMAT_VERSION})"
        )
    return payload


def read_checkpoint_summary(path: str | Path) -> CheckpointSummary:
    """Read classes and metrics without building the network (used by the UI)."""

    payload = _read_payload(path)
    label_map = [str(label) for label in payload.get("label_map", [])]
    if not label_map:
        raise CheckpointError("Checkpoint has an empty label map")
    dsp_config = DspConfig.from_dict(payload.get("dsp_config", {}))
    model_config = ModelConfig.from_dict(payload.get("model_config", {}))
    detection_config = DetectionConfig.from_dict(payload.get("detection_config", {}))
    state_dict = payload["state_dict"]
    n_parameters = sum(
        int(value.numel())
        for key, value in state_dict.items()
        if isinstance(value, Tensor) and not key.endswith(("input_mean", "input_std"))
    )
    return CheckpointSummary(
        path=Path(path),
        label_map=label_map,
        model_config=model_config,
        dsp_config=dsp_config,
        detection_config=detection_config,
        metrics=dict(payload.get("metrics", {})),
        trained_at=str(payload.get("trained_at", "")),
        n_parameters=n_parameters,
        class_sentences=_normalise_class_sentences(
            payload.get("class_sentences", {}), label_map
        ),
    )


def save_checkpoint(
    path: str | Path,
    model: EMGSequenceClassifier,
    *,
    label_map: list[str],
    class_sentences: Mapping[str, str] | None = None,
    dsp_config: DspConfig,
    detection_config: DetectionConfig,
    metrics: Mapping[str, Any],
    train_config: TrainConfig | None = None,
    dataset_root: str | Path = "",
) -> Path:
    """Atomically save everything needed for inference in one file."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    labels = list(label_map)
    payload: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "label_map": labels,
        "class_sentences": _normalise_class_sentences(class_sentences or {}, labels),
        "model_config": model.config.to_dict(),
        "dsp_config": dsp_config.to_dict(),
        "detection_config": detection_config.to_dict(),
        "n_channels": model.n_channels,
        "metrics": dict(metrics),
        "trained_at": utc_now_iso(),
        "dataset_root": str(dataset_root),
    }
    if train_config is not None:
        payload["train_config"] = train_config.to_dict()

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.stem}-", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
        os.replace(tmp_name, target)
    except Exception as exc:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise CheckpointError(f"Could not write checkpoint '{target}': {exc}") from exc
    return target


def load_checkpoint(
    path: str | Path, *, device: str | torch.device = "cpu"
) -> CheckpointInfo:
    source = Path(path)
    payload = _read_payload(source)
    label_map = [str(label) for label in payload["label_map"]]
    if not label_map:
        raise CheckpointError("Checkpoint has an empty label map")
    model_config = ModelConfig.from_dict(payload.get("model_config", {}))
    dsp_config = DspConfig.from_dict(payload.get("dsp_config", {}))
    detection_config = DetectionConfig.from_dict(payload.get("detection_config", {}))
    n_channels = int(payload.get("n_channels", N_CHANNELS))

    model = EMGSequenceClassifier(n_channels, len(label_map), model_config)
    try:
        model.load_state_dict(payload["state_dict"], strict=True)
    except RuntimeError as exc:
        raise CheckpointError(
            f"Checkpoint weights are incompatible with this architecture: {exc}"
        ) from exc

    target_device = torch.device(device)
    model.to(target_device).eval()
    return CheckpointInfo(
        model=model,
        label_map=label_map,
        model_config=model_config,
        dsp_config=dsp_config,
        detection_config=detection_config,
        metrics=dict(payload.get("metrics", {})),
        trained_at=str(payload.get("trained_at", "")),
        dataset_root=str(payload.get("dataset_root", "")),
        path=source,
        class_sentences=_normalise_class_sentences(
            payload.get("class_sentences", {}), label_map
        ),
    )
