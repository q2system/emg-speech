"""Package initialisation (kept import-light on purpose).

Import submodules explicitly, e.g. ``from emg_speech.dsp import extract_sample``.
Importing Qt or PyTorch is never required by the core DSP modules.
"""

from __future__ import annotations

__version__ = "1.0.0"
