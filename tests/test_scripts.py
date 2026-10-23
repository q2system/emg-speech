from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

SCRIPTS = {
    "record.py": "Record labelled EMG examples",
    "train.py": "Train the EMG word classifier",
    "recognize.py": "Recognise words from EMG",
    "simulate.py": "simulated 9-channel signal",
}


@pytest.mark.parametrize("script, expected", SCRIPTS.items())
def test_entry_script_has_help(script: str, expected: str) -> None:
    """Each visible entry file can be run directly with --help."""

    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    env["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run(
        [sys.executable, script, "--help"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert expected.lower() in result.stdout.lower()
