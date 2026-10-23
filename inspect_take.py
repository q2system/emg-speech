"""Print one saved recording.

    python inspect_take.py
    python inspect_take.py dataset/hello/example.npy
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if len(sys.argv) > 1:
    path = Path(sys.argv[1])
else:
    files = sorted(Path("dataset").rglob("*.npy"))
    if not files:
        raise SystemExit("No recordings found. Run: python record.py")
    path = files[-1]

data = np.load(path)
print(f"File:  {path}")
print(f"Shape: {data.shape}  (time steps, channels)")
print(f"Type:  {data.dtype}")
print(data)
