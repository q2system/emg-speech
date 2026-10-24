"""Helper library for ``pipeline_walkthrough.py``.

The classroom pipeline file stays short by importing these functions.  The
longer setup code, Matplotlib backend selection, plotting details, and the
training loop live in this folder.
"""

from __future__ import annotations

from .cli import parse_args
from .data import (
    BatchData,
    ModelBundle,
    PreparedData,
    build_model,
    load_database,
    make_batch,
    prepare_data,
)
from .plotting import (
    PlotSetupError,
    Plotter,
    make_plotter,
    plot_batch,
    plot_database,
    plot_evaluation,
    plot_forward,
    plot_loss,
    plot_model_architecture,
    plot_preprocessing,
    plot_training,
)
from .reporting import goodbye, print_metric_table, section, welcome
from .training import (
    EvaluationResult,
    ForwardResult,
    LossResult,
    TrainingResult,
    compute_loss_and_gradients,
    evaluate_model,
    forward_pass,
    train_model,
)

__all__ = [
    "BatchData",
    "EvaluationResult",
    "ForwardResult",
    "LossResult",
    "ModelBundle",
    "PlotSetupError",
    "Plotter",
    "PreparedData",
    "TrainingResult",
    "build_model",
    "compute_loss_and_gradients",
    "evaluate_model",
    "forward_pass",
    "goodbye",
    "load_database",
    "make_batch",
    "make_plotter",
    "parse_args",
    "plot_batch",
    "plot_database",
    "plot_evaluation",
    "plot_forward",
    "plot_loss",
    "plot_model_architecture",
    "plot_preprocessing",
    "plot_training",
    "prepare_data",
    "print_metric_table",
    "section",
    "train_model",
    "welcome",
]
