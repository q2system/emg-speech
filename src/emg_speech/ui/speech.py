"""Optional text-to-speech output for recognised words.

The inference window owns one :class:`SpeechOutput`.  The engine is created
lazily so a machine without speech support can still run the recognition UI;
the checkbox is simply disabled with a useful tooltip.
"""

from __future__ import annotations

from PyQt6.QtCore import QObject
from PyQt6.QtWidgets import QWidget

try:
    from PyQt6.QtTextToSpeech import QTextToSpeech
except ImportError:  # pragma: no cover - depends on the PyQt6 build
    QTextToSpeech = None  # type: ignore[assignment]


class SpeechOutput(QObject):
    """Small wrapper around Qt's system text-to-speech engine."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._engine = None
        self._engine_name: str | None = None
        self._error = ""

        if QTextToSpeech is None:
            self._error = "This PyQt6 build does not include QtTextToSpeech."
            return

        try:
            engines = [
                name
                for name in QTextToSpeech.availableEngines()
                if name.lower() != "mock"
            ]
        except Exception as exc:  # pragma: no cover - platform backend failure
            self._error = f"Could not query text-to-speech engines: {exc}"
            return

        if not engines:
            self._error = "No system text-to-speech engine is available."
            return
        self._engine_name = engines[0]

    @property
    def available(self) -> bool:
        return self._engine_name is not None and not self._error

    @property
    def error(self) -> str:
        return self._error

    def say(self, text: str) -> bool:
        """Speak ``text`` without blocking the GUI thread.

        Returns ``False`` when speech is unavailable or the engine fails to
        start.  The caller can then disable the checkbox and show the reason.
        """

        cleaned = text.strip()
        if not cleaned or not self.available:
            return False

        try:
            if self._engine is None:
                self._engine = QTextToSpeech(self._engine_name, self)
            self._engine.stop()
            self._engine.say(cleaned)
        except Exception as exc:  # pragma: no cover - platform backend failure
            self._error = f"Text-to-speech failed: {exc}"
            self._engine = None
            return False
        return True

    def stop(self) -> None:
        if self._engine is not None:
            self._engine.stop()

    def shutdown(self) -> None:
        if self._engine is not None:
            self._engine.stop()
            self._engine.deleteLater()
            self._engine = None
