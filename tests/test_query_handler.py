"""Tests for the shared query handling module."""
import pytest
from unittest.mock import Mock, patch, mock_open
from pathlib import Path
from typing import List

from finchat_sec_qa.edgar_client import FilingMetadata
from finchat_sec_qa.citation import Citation


class TestQueryHandler:
    """Test cases for the shared QueryHandler class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Mock dependencies
        self.mock_edgar_client = Mock()
        self.mock_qa_engine = Mock()
        
        # Import and create handler (import here to avoid circular imports during test discovery)
        from finchat_sec_qa.query_handler import QueryHandler
        self.handler = QueryHandler(self.mock_edgar_client, self.mock_qa_engine)

    def test_query_handler_initialization(self):
        """Test QueryHandler initializes with proper dependencies."""
        from finchat_sec_qa.query_handler import QueryHandler
        
        handler = QueryHandler(self.mock_edgar_client, self.mock_qa_engine)
        assert handler.client is self.mock_edgar_client
        assert handler.engine is self.mock_qa_engine

    def test_process_query_success(self):
        """Test successful query processing end-to-end."""
        # Setup mock data
        mock_filing = Mock(spec=FilingMetadata)
        mock_filing.accession_no = "0000123456-23-000001"
        
        self.mock_edgar_client.get_recent_filings.return_value = [mock_filing]
        self.mock_edgar_client.download_filing.return_value = "/path/to/filing.txt"
        
        mock_citation = Mock(spec=Citation)
        self.mock_qa_engine.answer_with_citations.return_value = (
            "Test answer", [mock_citation]
        )
        
        # Mock file reading
        with patch("pathlib.Path.read_text", return_value="Mock filing text"):
            answer, citations = self.handler.process_query(
                ticker="AAPL",
                question="What are the risks?",
                form_type="10-K"
            )
        
        # Verify interactions
        self.mock_edgar_client.get_recent_filings.assert_called_once_with(
            "AAPL", form_type="10-K", limit=1
        )
        self.mock_edgar_client.download_filing.assert_called_once_with(mock_filing)
        self.mock_qa_engine.add_document.assert_called_once_with(
            "0000123456-23-000001", "Mock filing text"
        )
        self.mock_qa_engine.answer_with_citations.assert_called_once_with(
            "What are the risks?"
        )
        
        # Verify results
        assert answer == "Test answer"
        assert citations == [mock_citation]

    def test_process_query_no_filings_found(self):
        """Test query processing when no filings are found."""
        self.mock_edgar_client.get_recent_filings.return_value = []
        
        with pytest.raises(ValueError, match="No filings found for ticker AAPL"):
            self.handler.process_query("AAPL", "What are the risks?")

    def test_process_query_with_default_parameters(self):
        """Test process_query uses correct default parameters."""
        mock_filing = Mock(spec=FilingMetadata)
        mock_filing.accession_no = "0000123456-23-000001"
        self.mock_edgar_client.get_recent_filings.return_value = [mock_filing]
        self.mock_edgar_client.download_filing.return_value = "/path/to/filing.txt"
        self.mock_qa_engine.answer_with_citations.return_value = ("answer", [])
        
        with patch("pathlib.Path.read_text", return_value="text"):
            self.handler.process_query("AAPL", "question")
        
        # Should use default form_type="10-K" and limit=1
        self.mock_edgar_client.get_recent_filings.assert_called_once_with(
            "AAPL", form_type="10-K", limit=1
        )

    def test_get_filings_success(self):
        """Test successful filing retrieval."""
        mock_filing = Mock(spec=FilingMetadata)
        self.mock_edgar_client.get_recent_filings.return_value = [mock_filing]
        
        filings = self.handler._get_filings("AAPL", "10-K", 1)
        
        assert filings == [mock_filing]
        self.mock_edgar_client.get_recent_filings.assert_called_once_with(
            "AAPL", form_type="10-K", limit=1
        )

    def test_get_filings_not_found(self):
        """Test filing retrieval when no filings exist."""
        self.mock_edgar_client.get_recent_filings.return_value = []
        
        with pytest.raises(ValueError, match="No filings found for ticker MSFT"):
            self.handler._get_filings("MSFT", "10-Q", 2)

    def test_download_and_read_filing_success(self):
        """Test successful filing download and read."""
        mock_filing = Mock(spec=FilingMetadata)
        mock_filing.accession_no = "0000123456-23-000001"
        self.mock_edgar_client.download_filing.return_value = "/path/to/filing.html"
        
        with patch("pathlib.Path.read_text", return_value="Filing content"):
            content = self.handler._download_and_read_filing(mock_filing)
        
        assert content == "Filing content"
        self.mock_edgar_client.download_filing.assert_called_once_with(mock_filing)

    def test_serialize_citations_empty_list(self):
        """Test citation serialization with empty list."""
        from finchat_sec_qa.query_handler import QueryHandler
        
        result = QueryHandler.serialize_citations([])
        assert result == []

    def test_serialize_citations_with_citations(self):
        """Test citation serialization with actual citations."""
        from finchat_sec_qa.query_handler import QueryHandler
        
        # Create mock citations with __dict__ attribute
        mock_citation1 = Mock()
        mock_citation1.__dict__ = {
            "doc_id": "doc1", 
            "text": "text1", 
            "start": 0, 
            "end": 10
        }
        
        mock_citation2 = Mock()
        mock_citation2.__dict__ = {
            "doc_id": "doc2", 
            "text": "text2", 
            "start": 20, 
            "end": 30
        }
        
        result = QueryHandler.serialize_citations([mock_citation1, mock_citation2])
        
        expected = [
            {"doc_id": "doc1", "text": "text1", "start": 0, "end": 10},
            {"doc_id": "doc2", "text": "text2", "start": 20, "end": 30}
        ]
        assert result == expected

    def test_process_query_file_read_error(self):
        """Test query processing when file reading fails."""
        mock_filing = Mock(spec=FilingMetadata)
        mock_filing.accession_no = "0000123456-23-000001"
        self.mock_edgar_client.get_recent_filings.return_value = [mock_filing]
        self.mock_edgar_client.download_filing.return_value = "/path/to/filing.txt"
        
        with patch("pathlib.Path.read_text", side_effect=FileNotFoundError("File not found")):
            with pytest.raises(FileNotFoundError):
                self.handler.process_query("AAPL", "What are the risks?")

    def test_process_query_qa_engine_error(self):
        """Test query processing when QA engine fails."""
        mock_filing = Mock(spec=FilingMetadata)
        mock_filing.accession_no = "0000123456-23-000001"
        self.mock_edgar_client.get_recent_filings.return_value = [mock_filing]
        self.mock_edgar_client.download_filing.return_value = "/path/to/filing.txt"
        self.mock_qa_engine.answer_with_citations.side_effect = Exception("QA engine failed")
        
        with patch("pathlib.Path.read_text", return_value="Mock filing text"):
            with pytest.raises(Exception, match="QA engine failed"):
                self.handler.process_query("AAPL", "What are the risks?")

    def test_process_query_multiple_filings_uses_first(self):
        """Test that process_query uses the first filing when multiple are returned."""
        mock_filing1 = Mock(spec=FilingMetadata)
        mock_filing1.accession_no = "filing1"
        mock_filing2 = Mock(spec=FilingMetadata)
        mock_filing2.accession_no = "filing2"
        
        self.mock_edgar_client.get_recent_filings.return_value = [mock_filing1, mock_filing2]
        self.mock_edgar_client.download_filing.return_value = "/path/to/filing.txt"
        self.mock_qa_engine.answer_with_citations.return_value = ("answer", [])
        
        with patch("pathlib.Path.read_text", return_value="text"):
            self.handler.process_query("AAPL", "question")
        
        # Should download and process only the first filing
        self.mock_edgar_client.download_filing.assert_called_once_with(mock_filing1)
        self.mock_qa_engine.add_document.assert_called_once_with("filing1", "text")