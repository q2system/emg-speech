"""EMG speech recognition library.

The public API is deliberately tiny.  Students only need::

    from emg_speech import record, train_model, recognize, simulate

Everything else in this package is the implementation that makes those steps
work.
"""

from __future__ import annotations

from .api import record, recognize, simulate, train_model

__version__ = "1.0.0"

__all__ = ["record", "recognize", "simulate", "train_model", "__version__"]
