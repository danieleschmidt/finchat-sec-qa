from datetime import date
from pathlib import Path


import pytest

from finchat.agent import FinChatAgent, QueryResult
from finchat_sec_qa.edgar_client import FilingMetadata


class DummyClient:
    def __init__(self) -> None:
        self.filings = [
            FilingMetadata(
                cik="0001",
                accession_no="0001-01",
                form_type="10-K",
                filing_date=date.today(),
                document_url="https://example.com",
            )
        ]
        self.downloaded = False

    def get_recent_filings(self, ticker: str, form_type: str = "10-K", limit: int = 1):
        return self.filings

    def download_filing(self, filing: FilingMetadata, dest: Path) -> Path:
        self.downloaded = True
        path = dest / "f.html"
        path.write_text("alpha beta")
        return path


class DummyEngine:
    def __init__(self) -> None:
        self.added = None

    def add_document(self, doc_id: str, text: str) -> None:
        self.added = (doc_id, text)

    def answer_with_citations(self, question: str):
        return ("answer", [])


def test_query(monkeypatch, tmp_path):
    agent = FinChatAgent("ua", download_dir=tmp_path)
    dummy_client = DummyClient()
    monkeypatch.setattr(agent, "client", dummy_client)
    monkeypatch.setattr(agent, "_make_engine", lambda: DummyEngine())

    result = agent.query("q", "AAPL")
    assert isinstance(result, QueryResult)
    assert dummy_client.downloaded
    assert result.answer == "answer"


def test_invalid_question(tmp_path):
    agent = FinChatAgent("ua", download_dir=tmp_path)
    with pytest.raises(ValueError):
        agent.query("", "AAPL")
