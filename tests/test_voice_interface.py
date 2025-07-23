import pytest
from unittest.mock import Mock

from finchat_sec_qa.voice_interface import speak
import importlib


def test_success(monkeypatch):
    # Create mock engine and module with proper Mock objects
    mock_engine = Mock()
    mock_engine.spoken = []
    mock_engine.say.side_effect = lambda text: mock_engine.spoken.append(text)
    mock_engine.runAndWait.return_value = None
    
    mock_module = Mock()
    mock_module.init.return_value = mock_engine
    
    monkeypatch.setattr(importlib, "import_module", lambda name: mock_module)
    speak("hello")
    assert mock_engine.spoken == ["hello"]


def test_edge_case_invalid_input(monkeypatch):
    # Create mock engine and module for invalid input test
    mock_engine = Mock()
    mock_engine.spoken = []
    mock_engine.say.side_effect = lambda text: mock_engine.spoken.append(text)
    mock_engine.runAndWait.return_value = None
    
    mock_module = Mock()
    mock_module.init.return_value = mock_engine
    
    monkeypatch.setattr(importlib, "import_module", lambda name: mock_module)
    with pytest.raises(ValueError):
        speak("")


def test_missing_pyttsx3(monkeypatch):
    def fail(name: str):
        raise ModuleNotFoundError

    monkeypatch.setattr(importlib, "import_module", fail)
    with pytest.raises(RuntimeError):
        speak("hello")
