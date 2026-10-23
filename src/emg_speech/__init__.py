"""EMG speech recognition library.

The public API is deliberately tiny.  Students only need::

    from emg_speech import ProjectConfig, record, train_model, recognize, simulate

``ProjectConfig`` reads/writes the optional ``emg_config.json`` file.  Everything
else in this package is the implementation that makes those steps work.
"""

from __future__ import annotations

from .api import record, recognize, simulate, train_model
from .config import ProjectConfig

__version__ = "1.0.0"

__all__ = [
    "ProjectConfig",
    "record",
    "recognize",
    "simulate",
    "train_model",
    "__version__",
]
