import logging

from finchat_sec_qa.logging_utils import configure_logging
from finchat_sec_qa.edgar_client import EdgarClient


class DummySession:
    def __init__(self):
        self.headers = {}
        self.called = []
        self.adapters = {}

    def mount(self, prefix, adapter):
        self.adapters[prefix] = adapter

    def get(self, url, timeout=None):
        self.called.append(timeout)
        class Resp:
            def raise_for_status(self):
                pass
            def json(self):
                return {}
        return Resp()


def test_configure_logging_and_log_capture(caplog):
    configure_logging("INFO")
    logger = logging.getLogger("finchat_sec_qa.test")
    with caplog.at_level(logging.INFO):
        logger.info("hello")
    assert "hello" in caplog.text


def test_edgarclient_timeout_and_retry():
    session = DummySession()
    client = EdgarClient("ua", session=session, timeout=5, retries=2)
    client._get("https://example.com")
    assert session.called == [5]
    assert session.adapters["https://"].max_retries.total == 2


def test_cli_log_level(monkeypatch):
    calls: list[str] = []

    def fake_config(level):
        calls.append(str(level))

    from finchat_sec_qa import cli
    monkeypatch.setattr(cli, "configure_logging", fake_config)

    cli.main(["query", "question", __file__, "--log-level", "DEBUG"])
    assert "DEBUG" in calls
