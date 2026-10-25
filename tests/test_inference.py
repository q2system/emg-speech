from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from emg_speech.config import (
    DatasetMeta,
    DetectionConfig,
    DspConfig,
    ModelConfig,
    ProjectConfig,
    TrainConfig,
)
from emg_speech.dataset import DatasetIndex
from emg_speech.inference import RecognizerState, StreamingRecognizer
from emg_speech.model import EMGSequenceClassifier, load_checkpoint, save_checkpoint
from emg_speech.synthetic import _make_take
from emg_speech.train import (
    TrainingError,
    classification_metrics,
    load_training_history,
    train,
)


def test_classification_metrics_are_macro_averaged() -> None:
    targets = np.array([0, 0, 1, 1], dtype=np.int64)
    predictions = np.array([0, 1, 1, 1], dtype=np.int64)

    metrics = classification_metrics(targets, predictions, 2)

    assert metrics.accuracy == pytest.approx(0.75)
    assert metrics.error == pytest.approx(0.25)
    assert metrics.sensitivity == pytest.approx(0.75)
    assert metrics.specificity == pytest.approx(0.75)
    assert metrics.precision == pytest.approx(5 / 6)
    assert metrics.f1 == pytest.approx((2 / 3 + 0.8) / 2)


def test_training_produces_a_deployable_checkpoint(demo_dataset: Path, tmp_path: Path) -> None:
    import shutil

    dataset_root = tmp_path / "dataset"
    shutil.copytree(demo_dataset, dataset_root)
    labels = DatasetIndex.build(dataset_root).label_map
    meta = DatasetMeta.load(dataset_root)
    assert meta is not None
    linked = {labels[0]: "Please bring me some water"}
    meta.class_sentences = dict(linked)
    meta.save(dataset_root)

    checkpoint = tmp_path / "artifacts" / "model.pt"
    report = train(
        dataset_root,
        checkpoint,
        train_config=TrainConfig(
            epochs=4,
            batch_size=8,
            patience=3,
            device="cpu",
            val_fraction=0.25,
        ),
        model_config=ModelConfig(hidden_size=32, conv_channels=16, attention_dim=16),
    )
    assert checkpoint.exists()
    assert checkpoint.with_name("model_history.json").exists()
    assert report.best_epoch >= 1
    assert len(report.history) >= 1
    assert report.confusion.shape == (4, 4)
    assert int(report.confusion.sum()) == 8
    assert 0.0 <= report.best_val_accuracy <= 1.0
    assert 0.0 <= report.best_val_macro_f1 <= 1.0
    assert report.train_metrics.error == pytest.approx(
        1.0 - report.train_metrics.accuracy
    )
    assert report.val_metrics.error == pytest.approx(
        1.0 - report.val_metrics.accuracy
    )
    for metrics in (report.train_metrics, report.val_metrics):
        assert 0.0 <= metrics.sensitivity <= 1.0
        assert 0.0 <= metrics.specificity <= 1.0
        assert 0.0 <= metrics.precision <= 1.0
        assert 0.0 <= metrics.f1 <= 1.0
    assert "Sensitivity" in report.summary()
    assert "Specificity" in report.summary()
    assert "Precision" in report.summary()
    assert set(report.label_map) == set(labels)

    epoch = report.history[0]
    assert epoch.train_error == pytest.approx(1.0 - epoch.train_accuracy)
    assert epoch.val_error == pytest.approx(1.0 - epoch.val_accuracy)
    history = load_training_history(checkpoint)
    assert "train_sensitivity" in history["history"][0]
    assert "val_specificity" in history["history"][0]
    assert "val_precision" in history["history"][0]
    assert history["class_sentences"] == linked

    info = load_checkpoint(checkpoint, device="cpu")
    assert info.label_map == report.label_map
    assert info.class_sentences == linked
    assert info.metrics["val_accuracy"] == pytest.approx(report.best_val_accuracy)
    assert info.metrics["val_sensitivity"] == pytest.approx(
        report.val_metrics.sensitivity
    )
    assert info.metrics["val_specificity"] == pytest.approx(
        report.val_metrics.specificity
    )
    assert info.metrics["val_precision"] == pytest.approx(
        report.val_metrics.precision
    )


def test_single_class_dataset_is_rejected(tmp_path: Path) -> None:
    import shutil

    from emg_speech.synthetic import generate_dataset

    generated = generate_dataset(tmp_path / "gen", per_class=2, seed=4)
    root = tmp_path / "single"
    class_dir = root / "solo"
    class_dir.mkdir(parents=True)
    shutil.copy(generated.paths[0], class_dir / generated.paths[0].name)
    with pytest.raises(TrainingError, match="two classes"):
        train(
            root,
            tmp_path / "model.pt",
            train_config=TrainConfig(epochs=1, device="cpu"),
            model_config=ModelConfig(hidden_size=8, conv_channels=8, attention_dim=8),
        )


def test_training_rejects_project_config_mismatch(
    demo_dataset: Path, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="Project config uses"):
        train(
            demo_dataset,
            tmp_path / "model.pt",
            train_config=TrainConfig(epochs=1, device="cpu"),
            model_config=ModelConfig(hidden_size=8, conv_channels=8, attention_dim=8),
            project_config=ProjectConfig(
                dsp=DspConfig(sample_rate=1000.0, seq_len=256)
            ),
        )


def test_streaming_recognizer_end_to_end(tmp_path: Path) -> None:
    config = ModelConfig(hidden_size=16, conv_channels=8, attention_dim=8)
    model = EMGSequenceClassifier(9, 3, config)
    model.set_norm_stats(np.zeros(9), np.ones(9))
    dsp = DspConfig()
    detection = DetectionConfig(calibration_ms=1500)
    sentences = {
        "alpha": "Say alpha clearly",
        "beta": "Say beta clearly",
        "gamma": "Say gamma clearly",
    }
    checkpoint = save_checkpoint(
        tmp_path / "model.pt",
        model,
        label_map=["alpha", "beta", "gamma"],
        class_sentences=sentences,
        dsp_config=dsp,
        detection_config=detection,
        metrics={},
    )
    info = load_checkpoint(checkpoint, device="cpu")
    recognizer = StreamingRecognizer(info, min_confidence=0.0, device="cpu")
    assert recognizer.state is RecognizerState.CALIBRATING

    rng = np.random.default_rng(6)
    fs = dsp.sample_rate
    idle_frames = rng.normal(0.0, 0.02, size=(int(2.0 * fs), 9)).astype(np.float32)
    idle_frames[:, 8] += 0.98
    for frame in idle_frames:
        assert recognizer.push(frame) is None
    assert recognizer.state is RecognizerState.LISTENING
    assert recognizer.on_threshold is not None

    take = _make_take(rng, 1, sample_rate=fs)
    predictions = [p for p in (recognizer.push(frame) for frame in take) if p is not None]
    assert predictions, "a clear burst must produce a prediction"
    prediction = predictions[-1]
    assert prediction.label in info.label_map
    assert prediction.output_text == sentences[prediction.label]
    assert prediction.probabilities.shape == (3,)
    assert float(prediction.probabilities.sum()) == pytest.approx(1.0, abs=1e-4)
    assert prediction.event_ms >= detection.min_event_ms
    assert prediction.latency_ms >= 0.0
    assert prediction.accepted is True  # min_confidence == 0


def test_recognizer_can_be_reset(tmp_path: Path) -> None:
    model = EMGSequenceClassifier(9, 2, ModelConfig(hidden_size=8, conv_channels=8))
    checkpoint = save_checkpoint(
        tmp_path / "model.pt",
        model,
        label_map=["one", "two"],
        dsp_config=DspConfig(),
        detection_config=DetectionConfig(),
        metrics={},
    )
    recognizer = StreamingRecognizer(load_checkpoint(checkpoint), device="cpu")
    assert recognizer.last_filtered is None
    recognizer.push(np.zeros(9, dtype=np.float32))
    assert recognizer.last_filtered is not None
    assert recognizer.last_filtered.shape == (9,)
    recognizer.reset()
    assert recognizer.state is RecognizerState.CALIBRATING
    assert recognizer.frames_seen == 0
    assert recognizer.last_filtered is None


def test_replay_timeline_keeps_the_imu_continuous(demo_dataset: Path) -> None:
    from emg_speech.sources import ReplaySource

    source = ReplaySource(
        demo_dataset,
        loop=False,
        start_idle_seconds=0.2,
        idle_seconds=(0.1, 0.1),
        max_samples=4,
        seed=3,
    )
    timeline = source._build_timeline()
    source.close()
    # Concatenating samples must not create 1 g accelerometer steps that the
    # model was never trained on.
    steps = np.abs(np.diff(timeline[:, 3:], axis=0))
    assert float(steps.max()) < 0.1
