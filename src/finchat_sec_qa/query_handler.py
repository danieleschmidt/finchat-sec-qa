"""Shared query processing logic for both Flask and FastAPI servers."""
from __future__ import annotations

import asyncio
import logging
from typing import List, Tuple

from .citation import Citation
from .edgar_client import AsyncEdgarClient, EdgarClient, FilingMetadata
from .file_security import safe_read_file
from .qa_engine import FinancialQAEngine

logger = logging.getLogger(__name__)


class QueryHandler:
    """Shared query processing logic for both Flask and FastAPI servers.
    
    This class encapsulates the common workflow for:
    1. Retrieving SEC filings
    2. Downloading and reading filing text
    3. Processing through QA engine
    4. Returning structured results with citations
    """

    def __init__(self, edgar_client: EdgarClient, qa_engine: FinancialQAEngine) -> None:
        """Initialize QueryHandler with required dependencies.
        
        Args:
            edgar_client: SEC EDGAR API client for filing retrieval
            qa_engine: QA engine for document processing and question answering
        """
        self.client = edgar_client
        self.engine = qa_engine

    def process_query(
        self,
        ticker: str,
        question: str,
        form_type: str = "10-K",
        limit: int = 1
    ) -> Tuple[str, List[Citation]]:
        """Process a query and return answer with citations.
        
        This method encapsulates the complete query workflow:
        1. Retrieve SEC filings for the ticker
        2. Download and read the filing text
        3. Process the text through the QA engine
        4. Return answer and citations
        
        Args:
            ticker: Stock ticker symbol
            question: Question to ask about the filing
            form_type: SEC form type to retrieve (default: "10-K")
            limit: Maximum number of filings to consider (default: 1)
            
        Returns:
            Tuple of (answer, citations) where answer is the response string
            and citations is a list of Citation objects
            
        Raises:
            ValueError: If no filings are found for the ticker
            FileNotFoundError: If filing download or reading fails
            Exception: If QA engine processing fails
        """
        logger.debug("Processing query for ticker=%s, question=%s", ticker, question[:50])

        # Get filings
        filings = self._get_filings(ticker, form_type, limit)

        # Process filing (use first filing)
        filing_text = self._download_and_read_filing(filings[0])

        # Get answer through QA engine
        self.engine.add_document(filings[0].accession_no, filing_text)
        answer, citations = self.engine.answer_with_citations(question)

        logger.debug("Query processed successfully, found %d citations", len(citations))
        return answer, citations

    def _get_filings(
        self,
        ticker: str,
        form_type: str,
        limit: int
    ) -> List[FilingMetadata]:
        """Retrieve SEC filings for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            form_type: SEC form type to retrieve
            limit: Maximum number of filings to retrieve
            
        Returns:
            List of FilingMetadata objects
            
        Raises:
            ValueError: If no filings are found for the ticker
        """
        logger.debug("Retrieving %s filings for ticker %s", form_type, ticker)

        filings = self.client.get_recent_filings(
            ticker, form_type=form_type, limit=limit
        )

        if not filings:
            logger.warning("No filings found for ticker: %s", ticker)
            raise ValueError(f"No filings found for ticker {ticker}")

        logger.debug("Found %d filing(s) for ticker %s", len(filings), ticker)
        return filings

    def _download_and_read_filing(self, filing: FilingMetadata) -> str:
        """Download and read filing text.
        
        Args:
            filing: FilingMetadata object representing the filing to download
            
        Returns:
            The text content of the filing
            
        Raises:
            FileNotFoundError: If filing download or reading fails
            ValueError: If file path validation fails (security)
        """
        logger.debug("Downloading filing: %s", filing.accession_no)

        path = self.client.download_filing(filing)
        filing_text = safe_read_file(path)

        logger.debug("Filing downloaded and read, length: %d chars", len(filing_text))
        return filing_text

    @staticmethod
    def serialize_citations(citations: List[Citation]) -> List[dict]:
        """Convert citations to dictionary format for JSON serialization.
        
        Args:
            citations: List of Citation objects
            
        Returns:
            List of dictionaries containing citation data
        """
        return [c.__dict__ for c in citations]


class AsyncQueryHandler:
    """Async version of QueryHandler for improved performance with concurrent requests."""

    def __init__(self, edgar_client: AsyncEdgarClient, qa_engine: FinancialQAEngine) -> None:
        """Initialize AsyncQueryHandler with required dependencies.
        
        Args:
            edgar_client: Async SEC EDGAR API client for filing retrieval
            qa_engine: QA engine for document processing and question answering
        """
        self.client = edgar_client
        self.engine = qa_engine

    async def process_query(
        self,
        ticker: str,
        question: str,
        form_type: str = "10-K",
        limit: int = 1
    ) -> Tuple[str, List[Citation]]:
        """Process a query asynchronously and return answer with citations.
        
        This method encapsulates the complete async query workflow:
        1. Retrieve SEC filings for the ticker
        2. Download and read the filing text
        3. Process the text through the QA engine
        4. Return answer and citations
        
        Args:
            ticker: Stock ticker symbol
            question: Question to ask about the filing
            form_type: SEC form type to retrieve (default: "10-K")
            limit: Maximum number of filings to consider (default: 1)
            
        Returns:
            Tuple of (answer, citations) where answer is the response string
            and citations is a list of Citation objects
            
        Raises:
            ValueError: If no filings are found for the ticker
            FileNotFoundError: If filing download or reading fails
            Exception: If QA engine processing fails
        """
        logger.debug("Processing async query for ticker=%s, question=%s", ticker, question[:50])

        # Get filings asynchronously
        filings = await self._get_filings(ticker, form_type, limit)

        # Process filing (use first filing)
        filing_text = await self._download_and_read_filing(filings[0])

        # Get answer through QA engine (this is CPU-bound, so we run it in executor)
        loop = asyncio.get_event_loop()
        def _process_document():
            self.engine.add_document(filings[0].accession_no, filing_text)
            return self.engine.answer_with_citations(question)

        answer, citations = await loop.run_in_executor(None, _process_document)

        logger.debug("Async query processed successfully, found %d citations", len(citations))
        return answer, citations

    async def _get_filings(
        self,
        ticker: str,
        form_type: str,
        limit: int
    ) -> List[FilingMetadata]:
        """Retrieve SEC filings for a ticker asynchronously.
        
        Args:
            ticker: Stock ticker symbol
            form_type: SEC form type to retrieve
            limit: Maximum number of filings to retrieve
            
        Returns:
            List of FilingMetadata objects
            
        Raises:
            ValueError: If no filings are found for the ticker
        """
        logger.debug("Retrieving %s filings for ticker %s (async)", form_type, ticker)

        filings = await self.client.get_recent_filings(
            ticker, form_type=form_type, limit=limit
        )

        if not filings:
            logger.warning("No filings found for ticker: %s", ticker)
            raise ValueError(f"No filings found for ticker {ticker}")

        logger.debug("Found %d filing(s) for ticker %s (async)", len(filings), ticker)
        return filings

    async def _download_and_read_filing(self, filing: FilingMetadata) -> str:
        """Download and read filing text asynchronously.
        
        Args:
            filing: FilingMetadata object representing the filing to download
            
        Returns:
            The text content of the filing
            
        Raises:
            FileNotFoundError: If filing download or reading fails
            ValueError: If file path validation fails (security)
        """
        logger.debug("Downloading filing: %s (async)", filing.accession_no)

        path = await self.client.download_filing(filing)

        # Read file asynchronously using asyncio with secure file operations
        loop = asyncio.get_event_loop()
        filing_text = await loop.run_in_executor(None, lambda: safe_read_file(path))

        logger.debug("Filing downloaded and read (async), length: %d chars", len(filing_text))
        return filing_text

    @staticmethod
    def serialize_citations(citations: List[Citation]) -> List[dict]:
        """Convert citations to dictionary format for JSON serialization.
        
        Args:
            citations: List of Citation objects
            
        Returns:
            List of dictionaries containing citation data
        """
        return [c.__dict__ for c in citations]
