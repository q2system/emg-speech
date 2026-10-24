"""Add the ``src/`` directory to ``sys.path`` for source-checkout scripts.

The package follows the standard ``src/`` layout.  After an editable or normal
install this helper is unnecessary, but it lets the entry scripts run directly
from a freshly cloned repository without setting ``PYTHONPATH``.
"""

from __future__ import annotations

import sys
from pathlib import Path


def add_src_to_path() -> None:
    """Make ``src/`` importable when running from a source checkout."""

    src = Path(__file__).resolve().parent / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
