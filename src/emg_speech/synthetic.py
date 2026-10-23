"""Synthetic biosignal generator used by tests and the no-hardware demo.

The generator does not save pre-baked arrays: it creates raw takes and runs them
through the real :func:`emg_speech.dsp.extract_sample` pipeline.  That exercises
event detection and normalisation end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .config import DatasetMeta, DetectionConfig, DspConfig
from .dsp import EventNotFound, extract_sample
from .storage import save_sample

SYNTHETIC_CLASSES: tuple[str, ...] = ("alpha", "beta", "gamma", "delta")

_CARRIERS_HZ = (40.0, 65.0, 90.0, 110.0)


@dataclass
class DemoReport:
    created: int = 0
    failed: int = 0
    paths: list[Path] = field(default_factory=list)


def _envelope(kind: int, t: NDArray[np.float64]) -> NDArray[np.float64]:
    if kind == 0:
        return np.sin(np.pi * t) ** 1.4
    if kind == 1:
        return np.abs(np.sin(2.0 * np.pi * t)) ** 2.0
    if kind == 2:
        return np.clip(t**1.2, 0.0, 1.0) * np.clip(2.0 - 1.5 * t, 0.0, 1.0)
    return np.clip(1.6 * np.sin(np.pi * t), 0.0, 1.0)


def _make_take(
    rng: np.random.Generator,
    class_index: int,
    *,
    sample_rate: float,
    prepend_seconds: float = 0.5,
    append_seconds: float = 0.5,
    idle_std: float = 0.02,
) -> NDArray[np.float32]:
    prepend = int(prepend_seconds * sample_rate)
    duration = rng.uniform(0.22, 0.40)
    body = int(duration * sample_rate)
    append = int(append_seconds * sample_rate)
    total = prepend + body + append
    time = np.arange(total, dtype=np.float64) / sample_rate

    emg = rng.normal(0.0, idle_std, size=(total, 3))
    t_norm = np.linspace(0.0, 1.0, body)
    t_seconds = np.arange(body, dtype=np.float64) / sample_rate
    pattern = _envelope(class_index, t_norm)
    carrier = _CARRIERS_HZ[class_index]
    for channel in range(3):
        frequency = carrier * (1.0 + 0.03 * channel) + rng.uniform(-2.0, 2.0)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        signal = np.sin(2.0 * np.pi * frequency * t_seconds + phase)
        signal += 0.25 * np.sin(
            2.0 * np.pi * frequency * 1.7 * t_seconds + phase / 2.0
        )
        emg[prepend : prepend + body, channel] += 1.1 * pattern * signal

    gyro = rng.normal(0.0, 0.003, size=(total, 3))
    accel = rng.normal(0.0, 0.003, size=(total, 3))
    accel[:, 2] += 0.98
    # Class-specific slow motion signature.
    motion = 0.04 * np.sin(2.0 * np.pi * (1.2 + 0.3 * class_index) * time + class_index)
    gyro[:, class_index % 3] += motion
    accel[:, (class_index + 1) % 3] += 0.02 * np.sin(
        2.0 * np.pi * (2.0 + 0.2 * class_index) * time
    )

    return np.concatenate(
        [emg, gyro, accel], axis=1
    ).astype(np.float32)


def generate_dataset(
    output_dir: str | Path,
    *,
    classes: tuple[str, ...] = SYNTHETIC_CLASSES,
    per_class: int = 20,
    sample_rate: float = 500.0,
    seq_len: int = 128,
    seed: int = 7,
    max_attempts: int = 4,
) -> DemoReport:
    """Create a small, balanced, model-ready dataset on disk."""

    dsp = DspConfig(sample_rate=sample_rate, seq_len=seq_len)
    detection = DetectionConfig()
    meta = DatasetMeta.from_configs(dsp, detection)
    report = DemoReport()
    rng = np.random.default_rng(seed)

    for class_index, label in enumerate(classes):
        for _ in range(per_class):
            for attempt in range(max_attempts):
                take = _make_take(rng, class_index, sample_rate=sample_rate)
                try:
                    extracted = extract_sample(take, dsp, detection)
                except EventNotFound:
                    if attempt == max_attempts - 1:
                        report.failed += 1
                    continue
                path = save_sample(output_dir, label, extracted.data, meta)
                report.paths.append(path)
                report.created += 1
                break
    return report


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="dataset_demo", help="output folder")
    parser.add_argument("--per-class", type=int, default=20)
    parser.add_argument("--sample-rate", type=float, default=500.0)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    report = generate_dataset(
        args.out,
        per_class=args.per_class,
        sample_rate=args.sample_rate,
        seq_len=args.seq_len,
        seed=args.seed,
    )
    print(
        f"Created {report.created} samples in '{args.out}' "
        f"({report.failed} failed)."
    )
    return 0 if report.created else 1


if __name__ == "__main__":
    raise SystemExit(main())
