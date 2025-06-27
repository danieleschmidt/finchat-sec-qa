import pytest

from finchat_sec_qa.voice_interface import speak
import importlib


class DummyEngine:
    def __init__(self) -> None:
        self.spoken = []

    def say(self, text: str) -> None:
        self.spoken.append(text)

    def runAndWait(self) -> None:  # noqa: N802  # method name from pyttsx3
        pass


class DummyModule:
    def __init__(self, engine: DummyEngine) -> None:
        self._engine = engine

    def init(self) -> DummyEngine:
        return self._engine


def test_success(monkeypatch):
    dummy = DummyEngine()
    monkeypatch.setattr(importlib, "import_module", lambda name: DummyModule(dummy))
    speak("hello")
    assert dummy.spoken == ["hello"]


def test_edge_case_invalid_input(monkeypatch):
    dummy = DummyEngine()
    monkeypatch.setattr(importlib, "import_module", lambda name: DummyModule(dummy))
    with pytest.raises(ValueError):
        speak("")


def test_missing_pyttsx3(monkeypatch):
    def fail(name: str):
        raise ModuleNotFoundError

    monkeypatch.setattr(importlib, "import_module", fail)
    with pytest.raises(RuntimeError):
        speak("hello")
