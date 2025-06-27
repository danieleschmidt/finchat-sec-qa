"""Simple text-to-speech utilities."""

from __future__ import annotations

import importlib

__all__ = ["speak"]


def speak(text: str) -> None:
    """Speak the provided text using the default TTS engine."""
    if not text:
        raise ValueError("text must be provided")

    try:
        pyttsx3 = importlib.import_module("pyttsx3")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pyttsx3 is required for voice support; install via 'pip install pyttsx3'"
        ) from exc

    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

