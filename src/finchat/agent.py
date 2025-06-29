from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging

from finchat_sec_qa.edgar_client import EdgarClient, FilingMetadata
from finchat_sec_qa.qa_engine import FinancialQAEngine


@dataclass
class QueryResult:
    """Answer returned for a single filing."""

    filing: FilingMetadata
    answer: str


class FinChatAgent:
    """High-level helper for answering questions about SEC filings."""

    logger = logging.getLogger(__name__)

    def __init__(
        self,
        user_agent: str,
        download_dir: Path | None = None,
        *,
        engine_store: Path | None = None,
    ) -> None:
        if not user_agent:
            raise ValueError("user_agent must be provided")
        self.client = EdgarClient(user_agent)
        self.download_dir = Path(download_dir or Path.home() / ".cache" / "finchat_sec_qa")
        self.engine_store = engine_store

    def _make_engine(self) -> FinancialQAEngine:  # pragma: no cover - overridable
        return FinancialQAEngine(storage_path=self.engine_store)

    def query(self, question: str, ticker: str, filing_type: str = "10-K") -> QueryResult:
        """Return an answer for the latest filing of the company."""
        if not question:
            raise ValueError("question must be provided")
        self.logger.info("Querying %s for %s", ticker, question)
        filings = self.client.get_recent_filings(ticker, form_type=filing_type, limit=1)
        if not filings:
            raise ValueError(f"No filings found for {ticker}")
        filing = filings[0]
        path = self.client.download_filing(filing, self.download_dir)
        text = Path(path).read_text()
        engine = self._make_engine()
        engine.add_document(filing.accession_no, text)
        answer, _ = engine.answer_with_citations(question)
        self.logger.info("Answer produced for %s", ticker)
        return QueryResult(filing=filing, answer=answer)
