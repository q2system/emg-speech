from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

ROOT = Path(__file__).resolve().parents[1]


def test_pipeline_stays_short_and_uses_the_helper_library() -> None:
    text = (ROOT / "pipeline_walkthrough.py").read_text(encoding="utf-8")

    assert len(text.splitlines()) < 160
    assert "helper_lib" in text
    assert "matplotlib" not in text
    for number in range(1, 9):
        assert f"SECTION {number}" in text
    assert (ROOT / "helper_lib" / "plotting.py").exists()
    assert (ROOT / "helper_lib" / "training.py").exists()


def test_walkthrough_saves_one_plot_per_section(
    demo_dataset: Path, tmp_path: Path
) -> None:
    output = tmp_path / "plots"
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "pipeline_walkthrough.py"),
            "--data",
            str(demo_dataset),
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--save-dir",
            str(output),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "SECTION 8" in result.stdout
    plots = sorted(output.glob("*.png"))
    assert [path.name for path in plots] == [
        "01_database.png",
        "02_preprocessing.png",
        "03_batches.png",
        "04_model_architecture.png",
        "05_forward_pass.png",
        "06_loss_and_gradients.png",
        "07_training_curves.png",
        "08_evaluation.png",
    ]
    assert all(path.stat().st_size > 0 for path in plots)
