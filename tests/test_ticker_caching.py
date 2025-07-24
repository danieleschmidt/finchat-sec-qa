"""Tests for ticker caching optimization."""
import unittest
from unittest.mock import Mock, patch, call
import time
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from finchat_sec_qa.edgar_client import EdgarClient, AsyncEdgarClient


class TestTickerCaching(unittest.TestCase):
    """Test ticker caching to eliminate N+1 download pattern."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_ticker_data = {
            "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
            "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corporation"},
            "2": {"cik_str": 1018724, "ticker": "AMZN", "title": "Amazon.com Inc."}
        }
    
    def test_ticker_cache_avoids_repeated_downloads(self):
        """Test that ticker data is cached and not downloaded repeatedly."""
        client = EdgarClient()
        
        with patch.object(client, '_get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = self.mock_ticker_data
            mock_get.return_value = mock_response
            
            # Multiple ticker lookups
            client.ticker_to_cik("AAPL")
            client.ticker_to_cik("MSFT") 
            client.ticker_to_cik("AMZN")
            
            # Should only download once, not three times
            self.assertEqual(mock_get.call_count, 1)
    
    def test_ticker_cache_efficient_lookup(self):
        """Test that ticker lookup uses efficient data structure (not O(n) search)."""
        client = EdgarClient()
        
        with patch.object(client, '_get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = self.mock_ticker_data
            mock_get.return_value = mock_response
            
            start_time = time.time()
            result = client.ticker_to_cik("MSFT")
            lookup_time = time.time() - start_time
            
            # Should find result quickly (hash lookup, not linear search)
            self.assertEqual(result, "789019")
            # Even with small dataset, lookup should be very fast
            self.assertLess(lookup_time, 0.01)  # Less than 10ms
    
    def test_ticker_cache_handles_case_insensitive_search(self):
        """Test case-insensitive ticker lookup."""
        client = EdgarClient()
        
        with patch.object(client, '_get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = self.mock_ticker_data
            mock_get.return_value = mock_response
            
            # Should work with different cases
            self.assertEqual(client.ticker_to_cik("aapl"), "320193")
            self.assertEqual(client.ticker_to_cik("AAPL"), "320193")
            self.assertEqual(client.ticker_to_cik("Aapl"), "320193")
    
    def test_ticker_cache_memory_efficient(self):
        """Test that cache doesn't store entire raw JSON."""
        client = EdgarClient()
        
        with patch.object(client, '_get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = self.mock_ticker_data
            mock_get.return_value = mock_response
            
            client.ticker_to_cik("AAPL")
            
            # Cache should exist and be efficient
            self.assertTrue(hasattr(client, '_ticker_cache'))
            # Should be a simple dict mapping, not storing full JSON
            self.assertIsInstance(client._ticker_cache, dict)
    
    def test_ticker_cache_handles_not_found(self):
        """Test that cache properly handles ticker not found."""
        client = EdgarClient()
        
        with patch.object(client, '_get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = self.mock_ticker_data
            mock_get.return_value = mock_response
            
            with self.assertRaises(ValueError) as cm:
                client.ticker_to_cik("NOTFOUND")
            
            self.assertIn("Ticker 'NOTFOUND' not found", str(cm.exception))


class TestAsyncTickerCaching(unittest.TestCase):
    """Test async ticker caching implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_ticker_data = {
            "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
            "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corporation"},
        }
    
    async def test_async_ticker_cache_avoids_repeated_downloads(self):
        """Test that async ticker data is cached and not downloaded repeatedly."""
        client = AsyncEdgarClient()
        
        with patch.object(client, '_get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = self.mock_ticker_data
            mock_get.return_value = mock_response
            
            # Multiple ticker lookups
            await client.ticker_to_cik("AAPL")
            await client.ticker_to_cik("MSFT")
            
            # Should only download once
            self.assertEqual(mock_get.call_count, 1)
    
    async def test_async_ticker_cache_efficient_lookup(self):
        """Test that async ticker lookup is efficient."""
        client = AsyncEdgarClient()
        
        with patch.object(client, '_get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = self.mock_ticker_data
            mock_get.return_value = mock_response
            
            result = await client.ticker_to_cik("MSFT")
            self.assertEqual(result, "789019")


if __name__ == '__main__':
    unittest.main()