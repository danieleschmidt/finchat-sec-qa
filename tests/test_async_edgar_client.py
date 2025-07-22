import asyncio
import pytest
from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import httpx

from finchat_sec_qa.edgar_client import AsyncEdgarClient, FilingMetadata


class TestAsyncEdgarClient:
    """Test async EdgarClient functionality."""
    
    @pytest.fixture
    def client(self):
        """Create async EDGAR client for testing."""
        return AsyncEdgarClient("TestAgent/1.0")
    
    @pytest.mark.asyncio
    async def test_init_creates_async_client(self, client):
        """Test that initialization creates an async HTTP client."""
        assert hasattr(client, 'session')
        assert isinstance(client.session, httpx.AsyncClient)
        assert client.timeout == 10.0
        
    @pytest.mark.asyncio
    async def test_context_manager_closes_session(self):
        """Test that client properly closes session when used as context manager."""
        async with AsyncEdgarClient("TestAgent/1.0") as client:
            assert not client.session.is_closed
        assert client.session.is_closed
        
    @pytest.mark.asyncio
    async def test_get_method_makes_async_request(self, client):
        """Test that _get method makes async HTTP requests."""
        with patch.object(client.session, 'get', new_callable=AsyncMock) as mock_get:
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response
            
            result = await client._get("https://example.com")
            
            mock_get.assert_called_once_with("https://example.com", timeout=10.0)
            mock_response.raise_for_status.assert_called_once()
            assert result == mock_response
            
    @pytest.mark.asyncio
    async def test_ticker_to_cik_async(self, client):
        """Test async ticker to CIK resolution."""
        mock_response_data = {
            "0": {"ticker": "AAPL", "cik_str": "320193"},
            "1": {"ticker": "MSFT", "cik_str": "789019"}
        }
        
        with patch.object(client, '_get', new_callable=AsyncMock) as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_response_data
            mock_get.return_value = mock_response
            
            cik = await client.ticker_to_cik("AAPL")
            
            assert cik == "0000320193"
            mock_get.assert_called_once()
            
    @pytest.mark.asyncio
    async def test_get_recent_filings_async(self, client):
        """Test async recent filings retrieval."""
        mock_ticker_data = {
            "0": {"ticker": "AAPL", "cik_str": "320193"}
        }
        mock_filings_data = {
            "filings": {
                "recent": {
                    "accessionNumber": ["0000320193-23-000007"],
                    "form": ["10-K"],
                    "filingDate": ["2023-10-27"],
                    "primaryDocument": ["aapl-20230930.htm"]
                }
            }
        }
        
        with patch.object(client, '_get', new_callable=AsyncMock) as mock_get:
            # Mock ticker resolution
            mock_ticker_response = MagicMock()
            mock_ticker_response.json.return_value = mock_ticker_data
            
            # Mock filings response
            mock_filings_response = MagicMock()
            mock_filings_response.json.return_value = mock_filings_data
            
            mock_get.side_effect = [mock_ticker_response, mock_filings_response]
            
            filings = await client.get_recent_filings("AAPL", "10-K", 1)
            
            assert len(filings) == 1
            assert filings[0].cik == "0000320193"
            assert filings[0].form_type == "10-K"
            assert filings[0].filing_date == date(2023, 10, 27)
            assert mock_get.call_count == 2
            
    @pytest.mark.asyncio
    async def test_download_filing_async(self, client, tmp_path):
        """Test async filing download."""
        filing = FilingMetadata(
            cik="0000320193",
            accession_no="0000320193-23-000007",
            form_type="10-K",
            filing_date=date(2023, 10, 27),
            document_url="https://example.com/filing.html"
        )
        
        mock_content = b"<html>Mock filing content</html>"
        
        with patch.object(client, '_get', new_callable=AsyncMock) as mock_get:
            mock_response = MagicMock()
            mock_response.content = mock_content
            mock_get.return_value = mock_response
            
            result_path = await client.download_filing(filing, tmp_path)
            
            assert result_path.exists()
            assert result_path.read_bytes() == mock_content
            mock_get.assert_called_once_with("https://example.com/filing.html")
            
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, client):
        """Test that multiple async requests can run concurrently."""
        mock_response_data = {
            "0": {"ticker": "AAPL", "cik_str": "320193"},
            "1": {"ticker": "MSFT", "cik_str": "789019"}
        }
        
        with patch.object(client, '_get', new_callable=AsyncMock) as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_response_data
            mock_get.return_value = mock_response
            
            # Run multiple requests concurrently
            tasks = [
                client.ticker_to_cik("AAPL"),
                client.ticker_to_cik("MSFT")
            ]
            
            results = await asyncio.gather(*tasks)
            
            assert results[0] == "0000320193"
            assert results[1] == "0000789019"
            assert mock_get.call_count == 2
            
    @pytest.mark.asyncio
    async def test_error_handling_preserves_sync_behavior(self, client):
        """Test that async client preserves error handling from sync version."""
        with pytest.raises(ValueError, match="Ticker must be a non-empty string"):
            await client.ticker_to_cik("")
            
        with pytest.raises(ValueError, match="Invalid ticker format"):
            await client.ticker_to_cik("INVALID123")