"""Small, dependency-free JSON helpers with atomic writes."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


class JsonIOError(RuntimeError):
    """Raised when a JSON document cannot be read or written."""


def read_json(path: str | Path) -> Any:
    """Read a JSON document from ``path``."""

    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise JsonIOError(f"Cannot read JSON file '{path}': {exc}") from exc


def write_json(path: str | Path, payload: Any) -> None:
    """Write ``payload`` to ``path`` atomically (never leaves a half-written file)."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.stem}-", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=False)
            handle.write("\n")
        os.replace(tmp_name, path)
    except OSError as exc:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise JsonIOError(f"Cannot write JSON file '{path}': {exc}") from exc
