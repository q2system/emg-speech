from __future__ import annotations

import numpy as np

from emg_speech.inference import Prediction


class FakeSpeech:
    available = True
    error = ""

    def __init__(self) -> None:
        self.spoken: list[str] = []
        self.stop_calls = 0
        self.shutdown_calls = 0

    def say(self, text: str) -> bool:
        self.spoken.append(text)
        return True

    def stop(self) -> None:
        self.stop_calls += 1

    def shutdown(self) -> None:
        self.shutdown_calls += 1


def _prediction(
    label: str, *, accepted: bool = True, output_text: str = ""
) -> Prediction:
    return Prediction(
        label=label,
        confidence=0.9,
        probabilities=np.array([0.9, 0.1], dtype=np.float32),
        accepted=accepted,
        event_ms=250.0,
        latency_ms=12.0,
        timestamp=0.0,
        output_text=output_text,
    )


def test_inference_speech_checkbox_defaults_to_off(qt_app) -> None:
    from emg_speech.ui.inference_window import InferenceWindow

    window = InferenceWindow()
    try:
        assert window._speak_words.text() == "Speak words"
        assert not window._speak_words.isChecked()
    finally:
        window.close()


def test_inference_speaks_only_accepted_words_when_enabled(qt_app) -> None:
    from emg_speech.ui.inference_window import InferenceWindow

    window = InferenceWindow()
    try:
        speech = FakeSpeech()
        window._speech = speech
        window._speak_words.setEnabled(True)

        window._on_prediction(_prediction("alpha"))
        assert speech.spoken == []

        window._speak_words.setChecked(True)
        window._on_prediction(_prediction("alpha"))
        assert speech.spoken == ["alpha"]

        window._on_prediction(_prediction("beta", accepted=False))
        assert speech.spoken == ["alpha"]

        window._speak_words.setChecked(False)
        assert speech.stop_calls >= 1
        window._on_prediction(_prediction("gamma"))
        assert speech.spoken == ["alpha"]
    finally:
        window.close()


def test_inference_uses_linked_sentence_for_transcript_and_speech(qt_app) -> None:
    from emg_speech.ui.inference_window import InferenceWindow

    window = InferenceWindow()
    try:
        speech = FakeSpeech()
        window._speech = speech
        window._speak_words.setEnabled(True)
        window._speak_words.setChecked(True)

        window._on_prediction(
            _prediction("alpha", output_text="Please bring me some water")
        )

        assert window._transcript.toPlainText() == "Please bring me some water"
        assert speech.spoken == ["Please bring me some water"]
        assert "Please bring me some water" in window._last_label.text()
        assert "[alpha]" in window._last_label.text()
    finally:
        window.close()


def test_inference_disables_speech_when_the_engine_fails(qt_app) -> None:
    from emg_speech.ui.inference_window import InferenceWindow

    class FailingSpeech(FakeSpeech):
        error = "engine failed"

        def say(self, text: str) -> bool:
            return False

    window = InferenceWindow()
    try:
        window._speech = FailingSpeech()
        window._speak_words.setEnabled(True)
        window._speak_words.setChecked(True)

        window._on_prediction(_prediction("alpha"))

        assert not window._speak_words.isChecked()
        assert not window._speak_words.isEnabled()
        assert "engine failed" in window._speak_words.toolTip()
    finally:
        window.close()
