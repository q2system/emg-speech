"""Matplotlib setup and every figure used by the walkthrough."""

from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import torch

from emg_speech.constants import CHANNELS, EMG_CHANNELS, IMU_CHANNELS, N_CHANNELS
from emg_speech.dataset import DatasetIndex, load_array

from .data import BatchData, ModelBundle, PreparedData
from .training import EvaluationResult, ForwardResult, LossResult, TrainingResult


class PlotSetupError(RuntimeError):
    """Raised when matplotlib is unavailable or the plot backend cannot start."""


class Plotter:
    """Show a figure, or save it as a PNG when running without a display."""

    def __init__(self, plt: Any, save_dir: Path | None, show: bool) -> None:
        self.plt = plt
        self.save_dir = save_dir
        self.show = show

    def finish(self, figure: Any, filename: str) -> None:
        if self.save_dir is not None:
            path = self.save_dir / filename
            figure.savefig(path, dpi=140, bbox_inches="tight")
            print(f"  plot saved -> {path}")
        if self.show:
            self.plt.show()
        else:
            self.plt.close(figure)


def _has_display() -> bool:
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


def make_plotter(save_dir: Path | None, no_show: bool) -> Plotter:
    """Create the plotting backend and output folder for the walkthrough."""

    target = Path(save_dir).expanduser() if save_dir is not None else None
    show = not no_show and target is None
    if target is None and not _has_display():
        target = Path("artifacts") / "pipeline_plots"
        show = False
        print(f"No display detected; plots will be saved in '{target}'.")
    if target is not None:
        target.mkdir(parents=True, exist_ok=True)
    if no_show and target is None:
        print("--no-show used without --save-dir; the plots will not be saved.")

    try:
        import matplotlib

        if target is not None or not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise PlotSetupError(
            "This walkthrough needs matplotlib. Install it with:\n"
            "    pip install matplotlib"
        ) from exc

    if show:
        print("Close each figure window to continue to the next section.")
    return Plotter(plt, target, show)


def plot_database(index: DatasetIndex, plotter: Plotter) -> None:
    """Section 1 figure: saved signals and class counts."""

    first_by_class: dict[str, Path] = {}
    for class_index, label in enumerate(index.label_map):
        members = np.flatnonzero(index.labels == class_index)
        if members.size:
            first_by_class[label] = index.paths[int(members[0])]

    selected_label = next(iter(first_by_class))
    selected = load_array(first_by_class[selected_label])
    time_ms = np.arange(selected.shape[0]) / index.meta.sample_rate * 1000.0
    print(f"Selected example: {first_by_class[selected_label].name}")
    print(f"Array shape     : {selected.shape}  (time steps, channels)")

    plt = plotter.plt
    colors = plt.get_cmap("tab10")
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    figure.suptitle(
        "Section 1 - Starting point: one saved example and the whole database",
        fontsize=14,
        fontweight="bold",
    )

    ax = axes[0, 0]
    for channel in EMG_CHANNELS:
        ax.plot(time_ms, selected[:, channel], label=CHANNELS[channel])
    ax.set_title(f"EMG channels of one saved example ({selected_label})")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("saved amplitude")
    ax.grid(alpha=0.3)
    ax.legend(ncol=3, fontsize=8)

    ax = axes[0, 1]
    for channel in IMU_CHANNELS:
        ax.plot(time_ms, selected[:, channel], linewidth=0.9, label=CHANNELS[channel])
    ax.set_title("IMU channels (accelerometer z is close to 1 g)")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("saved amplitude")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=7)

    ax = axes[1, 0]
    for class_index, (label, path) in enumerate(first_by_class.items()):
        example = load_array(path)
        ax.plot(
            time_ms,
            example[:, EMG_CHANNELS[0]],
            linewidth=1.4,
            color=colors(class_index % 10),
            label=label,
        )
    ax.set_title("First saved example of each word (emg1 channel)")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("saved amplitude")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    labels = list(index.counts())
    values = list(index.counts().values())
    bars = ax.bar(
        np.arange(len(labels)),
        values,
        color=[colors(i % 10) for i in range(len(labels))],
    )
    ax.bar_label(bars, padding=2)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=30, ha="right")
    ax.set_title("How many examples are saved for each word?")
    ax.set_ylabel("number of examples")
    ax.grid(axis="y", alpha=0.3)

    plotter.finish(figure, "01_database.png")


def plot_preprocessing(prepared: PreparedData, plotter: Plotter) -> None:
    """Section 2 figure: split counts and normalisation."""

    index = prepared.index
    sample = load_array(index.paths[prepared.train_indices[0]])
    normalised = (sample - prepared.norm_stats.mean) / prepared.norm_stats.std
    time_ms = np.arange(sample.shape[0]) / index.meta.sample_rate * 1000.0

    plt = plotter.plt
    colors = plt.get_cmap("tab10")
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    figure.suptitle(
        "Section 2 - Preprocessing: split the data and normalise the input",
        fontsize=14,
        fontweight="bold",
    )

    ax = axes[0, 0]
    positions = np.arange(index.n_classes)
    width = 0.38
    ax.bar(
        positions - width / 2,
        prepared.train_counts,
        width,
        label="training",
        color=colors(0),
    )
    ax.bar(
        positions + width / 2,
        prepared.val_counts,
        width,
        label="validation",
        color=colors(1),
    )
    ax.set_xticks(positions, index.label_map, rotation=30, ha="right")
    ax.set_title("Stratified split: every class is in both parts")
    ax.set_ylabel("number of examples")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(time_ms, sample[:, EMG_CHANNELS[0]], color=colors(2))
    ax.set_title("Before normalisation: one training example (emg1)")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("saved amplitude")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(time_ms, normalised[:, EMG_CHANNELS[0]], color=colors(3))
    ax.set_title("After normalisation: the model's internal input")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("(value - mean) / std")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    channel_positions = np.arange(N_CHANNELS)
    ax.bar(
        channel_positions - 0.2,
        prepared.norm_stats.mean,
        0.4,
        label="mean",
        color=colors(0),
    )
    ax.bar(
        channel_positions + 0.2,
        prepared.norm_stats.std,
        0.4,
        label="std",
        color=colors(1),
    )
    ax.set_xticks(channel_positions, CHANNELS, rotation=45, ha="right")
    ax.set_title("Normalisation statistics per channel")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    plotter.finish(figure, "02_preprocessing.png")


def plot_batch(batch: BatchData, plotter: Plotter) -> None:
    """Section 3 figure: four normalised examples as heatmaps."""

    display = (batch.features - torch.as_tensor(batch.mean)) / torch.as_tensor(
        batch.std
    )
    plt = plotter.plt
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    figure.suptitle(
        "Section 3 - A mini-batch: four examples after normalisation",
        fontsize=14,
        fontweight="bold",
    )

    image = None
    for position, ax in enumerate(axes.flat):
        if position >= len(batch.features):
            ax.axis("off")
            continue
        image = ax.imshow(
            display[position].numpy().T,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-2.5,
            vmax=2.5,
        )
        ax.set_yticks(np.arange(N_CHANNELS), CHANNELS, fontsize=7)
        ax.set_xlabel("time step")
        label = batch.label_map[int(batch.targets[position])]
        ax.set_title(f"sample {position}: true word = '{label}'")

    if image is not None:
        figure.colorbar(
            image,
            ax=axes,
            shrink=0.7,
            pad=0.02,
            label="normalised value",
        )

    plotter.finish(figure, "03_batches.png")


def _parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def plot_model_architecture(
    bundle: ModelBundle,
    batch: BatchData,
    plotter: Plotter,
) -> None:
    """Section 4 figure: the network drawn as a labelled block diagram."""

    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    model = bundle.model
    cfg = model.config
    batch_size, seq_len, n_channels = batch.features.shape
    conv_time = (
        seq_len + 2 * (cfg.kernel_size // 2) - cfg.kernel_size
    ) // 2 + 1
    directions = 2 if cfg.bidirectional else 1
    rnn_output = cfg.hidden_size * directions
    rnn_name = ("Bi" if cfg.bidirectional else "") + (
        "LSTM" if cfg.rnn_type == "lstm" else "GRU"
    )
    layer_word = "layer" if cfg.num_layers == 1 else "layers"
    n_classes = len(batch.label_map)
    class_names = textwrap.fill(", ".join(batch.label_map), width=100)

    stem_parameters = _parameter_count(model.stem)
    rnn_parameters = _parameter_count(model.rnn)
    attention_parameters = _parameter_count(model.attention)
    head_parameters = _parameter_count(model.head)

    input_shape = f"({batch_size}, {seq_len}, {n_channels})"
    normalized_shape = f"({batch_size}, {n_channels}, {seq_len})"
    conv_shape = f"({batch_size}, {conv_time}, {cfg.conv_channels})"
    rnn_shape = f"({batch_size}, {conv_time}, {rnn_output})"
    pooled_shape = f"({batch_size}, {rnn_output})"
    output_shape = f"({batch_size}, {n_classes})"

    input_detail = (
        f"channels: {', '.join(CHANNELS)}\n"
        f"B = batch size, T = {seq_len} saved time steps"
    )
    normalize_detail = (
        "x = (x - input_mean) / input_std\n"
        "then transpose to (B, channels, T) for Conv1d"
    )
    stem_detail = (
        f"Conv1d({n_channels}→{cfg.conv_channels}, k={cfg.kernel_size}, "
        f"pad={cfg.kernel_size // 2}) → GroupNorm → GELU → "
        f"Dropout({cfg.conv_dropout:g})\n"
        f"Conv1d({cfg.conv_channels}→{cfg.conv_channels}, "
        f"k={cfg.kernel_size}, stride=2) → GroupNorm → GELU\n"
        "The second convolution halves the time axis."
    )
    rnn_detail = (
        f"{rnn_name}(input={cfg.conv_channels}, hidden={cfg.hidden_size}, "
        f"{cfg.num_layers} {layer_word}, bidirectional={cfg.bidirectional})\n"
        "One feature vector per time step."
    )
    attention_detail = (
        f"score = Linear({rnn_output}→{cfg.attention_dim}) → tanh → "
        f"Linear({cfg.attention_dim}→1)\n"
        "weights = softmax(score over time)\n"
        "pooled = Σ_t weights[t] · features[t]"
    )
    head_detail = (
        f"LayerNorm({rnn_output}) → Dropout({cfg.dropout:g}) → "
        f"Linear({rnn_output}→{n_classes})\n"
        "One logit per word."
    )
    output_detail = (
        "softmax(logits) → one probability per word\n"
        "the class with the largest probability wins"
    )

    plt = plotter.plt
    figure, ax = plt.subplots(figsize=(13, 10))
    figure.suptitle(
        "Section 4 - Model architecture: from a saved sequence to word probabilities",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlim(0, 12.8)
    ax.set_ylim(-0.9, 10.1)
    ax.axis("off")

    box_x = 0.35
    box_width = 6.9
    box_center = box_x + box_width / 2
    shape_x = 7.55
    params_x = 10.15

    ax.text(
        shape_x,
        9.95,
        "shape",
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
        color="#4A5568",
    )
    ax.text(
        params_x,
        9.95,
        "trainable params",
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
        color="#4A5568",
    )

    def draw_block(
        y: float,
        height: float,
        title: str,
        detail: str,
        shape: str,
        params: str,
        color: str,
    ) -> None:
        box = FancyBboxPatch(
            (box_x, y),
            box_width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=1.2,
            edgecolor="#4A5568",
            facecolor=color,
        )
        ax.add_patch(box)
        ax.text(
            box_x + 0.18,
            y + height - 0.13,
            title,
            ha="left",
            va="top",
            fontsize=10.5,
            fontweight="bold",
            color="#1A202C",
        )
        ax.text(
            box_x + 0.18,
            y + height - 0.42,
            detail,
            ha="left",
            va="top",
            fontsize=8.2,
            color="#2D3748",
            linespacing=1.35,
        )
        ax.text(
            shape_x,
            y + height / 2,
            shape,
            ha="left",
            va="center",
            fontsize=8.4,
            family="monospace",
            color="#2D3748",
        )
        ax.text(
            params_x,
            y + height / 2,
            params,
            ha="left",
            va="center",
            fontsize=8.4,
            color="#2D3748",
        )

    def connect(upper_y: float, lower_y: float) -> None:
        ax.add_patch(
            FancyArrowPatch(
                (box_center, upper_y),
                (box_center, lower_y),
                arrowstyle="-|>",
                mutation_scale=16,
                linewidth=1.4,
                color="#4A5568",
                shrinkA=0,
                shrinkB=0,
            )
        )

    input_y, input_h = 8.90, 0.85
    normalize_y, normalize_h = 7.65, 0.90
    stem_y, stem_h = 5.75, 1.55
    rnn_y, rnn_h = 4.25, 1.15
    attention_y, attention_h = 2.45, 1.45
    head_y, head_h = 0.95, 1.15
    output_y, output_h = -0.35, 0.85

    draw_block(
        input_y,
        input_h,
        "Input sequence",
        input_detail,
        input_shape,
        "—",
        "#E8F1FB",
    )
    draw_block(
        normalize_y,
        normalize_h,
        "Normalisation inside the model",
        normalize_detail,
        normalized_shape,
        "0\n(buffers)",
        "#EDF7ED",
    )
    draw_block(
        stem_y,
        stem_h,
        "Convolutional stem",
        stem_detail,
        conv_shape,
        f"{stem_parameters:,}",
        "#FFF4D6",
    )
    draw_block(
        rnn_y,
        rnn_h,
        f"{rnn_name} sequence encoder",
        rnn_detail,
        rnn_shape,
        f"{rnn_parameters:,}",
        "#FDE7E9",
    )
    draw_block(
        attention_y,
        attention_h,
        "Attention pooling",
        attention_detail,
        pooled_shape,
        f"{attention_parameters:,}",
        "#EDE7F6",
    )
    draw_block(
        head_y,
        head_h,
        "Classifier head",
        head_detail,
        output_shape,
        f"{head_parameters:,}",
        "#E3F2FD",
    )
    draw_block(
        output_y,
        output_h,
        "Output probabilities",
        output_detail,
        output_shape,
        "—",
        "#F1F8E9",
    )

    connect(input_y, normalize_y + normalize_h)
    connect(normalize_y, stem_y + stem_h)
    connect(stem_y, rnn_y + rnn_h)
    connect(rnn_y, attention_y + attention_h)
    connect(attention_y, head_y + head_h)
    connect(head_y, output_y + output_h)

    ax.text(
        0.35,
        -0.60,
        f"Classes: {class_names}\n"
        f"B = {batch_size}, T = {seq_len}, n_classes = {n_classes}  |  "
        f"total trainable parameters = {model.n_parameters:,}",
        ha="left",
        va="top",
        fontsize=8.2,
        color="#2D3748",
        linespacing=1.35,
    )

    plotter.finish(figure, "04_model_architecture.png")


def plot_forward(forward: ForwardResult, plotter: Plotter) -> None:
    """Section 4 figure: logits and probabilities for one example."""

    plt = plotter.plt
    positions = np.arange(len(forward.label_map))
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout="constrained")
    figure.suptitle(
        "Section 5 - What the network outputs for one example",
        fontsize=14,
        fontweight="bold",
    )

    axes[0].bar(positions, forward.logits[0].numpy(), color="tab:blue")
    axes[0].set_xticks(positions, forward.label_map, rotation=30, ha="right")
    axes[0].set_title("Raw logits (one score per class)")
    axes[0].set_ylabel("score")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(positions, forward.probabilities[0].numpy(), color="tab:orange")
    axes[1].set_xticks(positions, forward.label_map, rotation=30, ha="right")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Softmax probabilities (sum to 1)")
    axes[1].set_ylabel("probability")
    axes[1].grid(axis="y", alpha=0.3)

    plotter.finish(figure, "05_forward_pass.png")


def plot_loss(losses: LossResult, plotter: Plotter) -> None:
    """Section 5 figure: per-example loss and gradient norms."""

    plt = plotter.plt
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout="constrained")
    figure.suptitle(
        "Section 6 - Loss for each example and gradient strength",
        fontsize=14,
        fontweight="bold",
    )

    positions = np.arange(len(losses.targets))
    axes[0].bar(positions, losses.per_sample, color="tab:red")
    axes[0].set_xticks(
        positions,
        [
            f"{position}\n{losses.label_map[int(label)]}"
            for position, label in enumerate(losses.targets)
        ],
        fontsize=7,
    )
    axes[0].set_title("Per-example loss (true word under each bar)")
    axes[0].set_xlabel("example in the batch")
    axes[0].set_ylabel("loss")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(
        losses.gradient_names,
        losses.gradient_norms,
        color="tab:green",
    )
    axes[1].set_title("Gradient norm by network section")
    axes[1].set_ylabel("L2 norm")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", alpha=0.3)

    plotter.finish(figure, "06_loss_and_gradients.png")


def plot_training(training: TrainingResult, plotter: Plotter) -> None:
    """Section 6 figure: loss and accuracy curves."""

    history = training.history
    plt = plotter.plt
    epochs_axis = np.arange(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout="constrained")
    figure.suptitle(
        "Section 7 - Training curves (the model is learning)",
        fontsize=14,
        fontweight="bold",
    )

    axes[0].plot(epochs_axis, history["train_loss"], marker="o", label="training")
    axes[0].plot(epochs_axis, history["val_loss"], marker="o", label="validation")
    axes[0].axvline(
        training.best_epoch,
        color="0.6",
        linestyle="--",
        linewidth=1,
        label=f"best epoch ({training.best_epoch})",
    )
    axes[0].set_title("Loss per epoch")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross-entropy loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        epochs_axis, history["train_accuracy"], marker="o", label="train accuracy"
    )
    axes[1].plot(
        epochs_axis, history["val_accuracy"], marker="o", label="validation accuracy"
    )
    axes[1].plot(
        epochs_axis, history["val_f1"], marker="s", label="validation macro-F1"
    )
    axes[1].set_title("Scores per epoch")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

    if len(epochs_axis) <= 12:
        axes[0].set_xticks(epochs_axis)
        axes[1].set_xticks(epochs_axis)

    plotter.finish(figure, "07_training_curves.png")


def plot_evaluation(evaluation: EvaluationResult, plotter: Plotter) -> None:
    """Section 7 figure: confusion matrix and metric comparison."""

    plt = plotter.plt
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    figure.suptitle(
        "Section 8 - Evaluation: where the model is right and wrong",
        fontsize=14,
        fontweight="bold",
    )

    matrix = evaluation.matrix
    image = axes[0].imshow(matrix, cmap="Blues")
    axes[0].set_xticks(
        np.arange(len(evaluation.label_map)),
        evaluation.label_map,
        rotation=30,
        ha="right",
    )
    axes[0].set_yticks(np.arange(len(evaluation.label_map)), evaluation.label_map)
    axes[0].set_xlabel("predicted word")
    axes[0].set_ylabel("true word")
    axes[0].set_title("Validation confusion matrix")
    for row in range(len(evaluation.label_map)):
        for column in range(len(evaluation.label_map)):
            axes[0].text(
                column,
                row,
                str(int(matrix[row, column])),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > matrix.max() / 2 else "black",
            )
    figure.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)

    metric_names = [
        "Accuracy",
        "Error",
        "Sensitivity",
        "Specificity",
        "Precision",
        "F1",
    ]
    train_metrics = evaluation.train_metrics
    val_metrics = evaluation.val_metrics
    train_values = [
        train_metrics.accuracy,
        train_metrics.error,
        train_metrics.sensitivity,
        train_metrics.specificity,
        train_metrics.precision,
        train_metrics.f1,
    ]
    val_values = [
        val_metrics.accuracy,
        val_metrics.error,
        val_metrics.sensitivity,
        val_metrics.specificity,
        val_metrics.precision,
        val_metrics.f1,
    ]
    positions = np.arange(len(metric_names))
    width = 0.38
    train_bars = axes[1].bar(
        positions - width / 2,
        train_values,
        width,
        label="training",
        color="tab:blue",
    )
    val_bars = axes[1].bar(
        positions + width / 2,
        val_values,
        width,
        label="validation",
        color="tab:orange",
    )
    axes[1].bar_label(train_bars, fmt="%.2f", fontsize=7, padding=2)
    axes[1].bar_label(val_bars, fmt="%.2f", fontsize=7, padding=2)
    axes[1].set_xticks(positions, metric_names, rotation=25, ha="right")
    axes[1].set_ylim(0.0, 1.35)
    axes[1].set_title("Same metrics train.py prints after training")
    axes[1].set_ylabel("score")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend(loc="upper center", ncol=2, fontsize=8)

    plotter.finish(figure, "08_evaluation.png")
