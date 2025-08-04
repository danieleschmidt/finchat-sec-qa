"""
Comprehensive unit tests for the QA Engine.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List

from tests.helpers.assertions import (
    assert_valid_response_structure,
    assert_citation_accuracy,
    assert_performance_threshold
)


class TestFinancialQAEngineCore:
    """Test core QA engine functionality."""
    
    @pytest.fixture
    def qa_engine(self, mock_openai_client, test_config):
        """Create QA engine instance for testing."""
        from finchat_sec_qa.qa_engine import FinancialQAEngine
        return FinancialQAEngine()
    
    def test_query_processing_structure(self, qa_engine, sample_query, sample_filing_path):
        """Test that query processing returns proper structure."""
        result = qa_engine.process_query(sample_query, str(sample_filing_path))
        assert_valid_response_structure(result)
    
    def test_citation_extraction_accuracy(self, qa_engine, sample_filing_path):
        """Test citation extraction accuracy."""
        query = "What are the risk factors?"
        result = qa_engine.process_query(query, str(sample_filing_path))
        
        source_text = sample_filing_path.read_text()
        assert_citation_accuracy(result["citations"], source_text, min_accuracy=0.8)
    
    @pytest.mark.performance
    def test_query_performance(self, qa_engine, sample_query, sample_filing_path, performance_thresholds):
        """Test that queries complete within performance thresholds."""
        import time
        
        start_time = time.time()
        result = qa_engine.process_query(sample_query, str(sample_filing_path))
        elapsed_time = time.time() - start_time
        
        assert_performance_threshold(
            elapsed_time, 
            performance_thresholds["query_response_time"], 
            "Query processing"
        )
    
    def test_empty_query_handling(self, qa_engine, sample_filing_path):
        """Test handling of empty or invalid queries."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            qa_engine.process_query("", str(sample_filing_path))
    
    def test_missing_file_handling(self, qa_engine, sample_query):
        """Test handling of missing filing files."""
        with pytest.raises(FileNotFoundError):
            qa_engine.process_query(sample_query, "nonexistent_file.txt")
    
    @patch('finchat_sec_qa.qa_engine.FinancialQAEngine._generate_embeddings')
    def test_embedding_generation_called(self, mock_embeddings, qa_engine, sample_query, sample_filing_path):
        """Test that embedding generation is called appropriately."""
        mock_embeddings.return_value = [[0.1] * 1536]
        
        qa_engine.process_query(sample_query, str(sample_filing_path))
        
        mock_embeddings.assert_called()
    
    def test_confidence_scoring(self, qa_engine, sample_query, sample_filing_path):
        """Test that confidence scores are reasonable."""
        result = qa_engine.process_query(sample_query, str(sample_filing_path))
        
        confidence = result["confidence"]
        assert 0.0 <= confidence <= 1.0, "Confidence score must be between 0 and 1"
    
    def test_context_window_management(self, qa_engine, sample_filing_path):
        """Test handling of large documents that exceed context windows."""
        # Create a very long query to test context management
        long_query = "What are the details about " + "risk factors, " * 100
        
        result = qa_engine.process_query(long_query, str(sample_filing_path))
        assert_valid_response_structure(result)


class TestFinancialQAEngineEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def qa_engine(self, mock_openai_client, test_config):
        from finchat_sec_qa.qa_engine import FinancialQAEngine
        return FinancialQAEngine()
    
    def test_malformed_filing_content(self, qa_engine, sample_query, temp_cache_dir):
        """Test handling of malformed filing content."""
        malformed_file = temp_cache_dir / "malformed.txt"
        malformed_file.write_text("Invalid content with special chars: \x00\x01\x02")
        
        # Should not crash, but may return low confidence
        result = qa_engine.process_query(sample_query, str(malformed_file))
        assert_valid_response_structure(result)
    
    def test_very_short_filing(self, qa_engine, sample_query, temp_cache_dir):
        """Test handling of very short filings."""
        short_file = temp_cache_dir / "short.txt"
        short_file.write_text("Very short filing.")
        
        result = qa_engine.process_query(sample_query, str(short_file))
        assert_valid_response_structure(result)
        # Should indicate low confidence for insufficient context
        assert result["confidence"] < 0.5
    
    def test_unicode_handling(self, qa_engine, temp_cache_dir):
        """Test handling of Unicode characters in queries and filings."""
        unicode_file = temp_cache_dir / "unicode.txt"
        unicode_file.write_text("Filing with Unicode: café, naïve, résumé, 中文")
        
        unicode_query = "What about café and naïve policies?"
        result = qa_engine.process_query(unicode_query, str(unicode_file))
        assert_valid_response_structure(result)
    
    @patch('finchat_sec_qa.qa_engine.FinancialQAEngine._call_openai')
    def test_openai_api_failure_handling(self, mock_openai, qa_engine, sample_query, sample_filing_path):
        """Test handling of OpenAI API failures."""
        mock_openai.side_effect = Exception("API rate limit exceeded")
        
        with pytest.raises(Exception, match="API rate limit exceeded"):
            qa_engine.process_query(sample_query, str(sample_filing_path))
    
    def test_concurrent_query_processing(self, qa_engine, sample_filing_path):
        """Test concurrent query processing safety."""
        import threading
        from tests.helpers.assertions import assert_concurrent_safety
        
        queries = [
            "What are the risk factors?",
            "What is the revenue trend?",
            "Who are the key competitors?",
            "What are the main business segments?"
        ]
        
        results = []
        threads = []
        
        def process_query(query):
            try:
                result = qa_engine.process_query(query, str(sample_filing_path))
                results.append(result)
            except Exception as e:
                results.append(e)
        
        for query in queries:
            thread = threading.Thread(target=process_query, args=(query,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        assert_concurrent_safety(results, len(queries))


class TestFinancialQAEngineIntegration:
    """Integration tests with other components."""
    
    @pytest.fixture
    def qa_engine_with_cache(self, mock_openai_client, test_config, temp_cache_dir):
        """Create QA engine with caching enabled."""
        from finchat_sec_qa.qa_engine import FinancialFinancialQAEngine
        config = test_config.copy()
        config["cache"]["directory"] = str(temp_cache_dir)
        return FinancialQAEngine(config=config, openai_client=mock_openai_client)
    
    def test_caching_behavior(self, qa_engine_with_cache, sample_query, sample_filing_path):
        """Test that caching improves performance on repeated queries."""
        import time
        
        # First query - should be slower (cache miss)
        start_time = time.time()
        result1 = qa_engine_with_cache.process_query(sample_query, str(sample_filing_path))
        first_duration = time.time() - start_time
        
        # Second identical query - should be faster (cache hit)
        start_time = time.time()
        result2 = qa_engine_with_cache.process_query(sample_query, str(sample_filing_path))
        second_duration = time.time() - start_time
        
        # Results should be identical
        assert result1["answer"] == result2["answer"]
        
        # Second query should be significantly faster (allowing for some variation)
        assert second_duration < first_duration * 0.8, "Caching should improve performance"
    
    @patch('finchat_sec_qa.metrics.record_query_metrics')
    def test_metrics_recording(self, mock_metrics, qa_engine, sample_query, sample_filing_path):
        """Test that metrics are recorded during query processing."""
        qa_engine.process_query(sample_query, str(sample_filing_path))
        
        mock_metrics.assert_called()
        call_args = mock_metrics.call_args[1]
        assert "query_time" in call_args
        assert "confidence" in call_args
        assert "citations_count" in call_args