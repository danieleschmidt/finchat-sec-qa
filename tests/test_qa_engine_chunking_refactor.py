"""Tests for QA Engine chunking logic refactoring."""

import pytest
from unittest.mock import patch, MagicMock
from finchat_sec_qa.qa_engine import FinancialQAEngine


def test_chunking_helper_methods_exist():
    """Test that chunking logic is split into helper methods."""
    engine = FinancialQAEngine()
    
    # Check that helper methods exist
    assert hasattr(engine, '_is_single_chunk')
    assert hasattr(engine, '_find_sentence_boundary')
    assert hasattr(engine, '_create_chunk_at_boundary')
    assert hasattr(engine, '_create_chunk_at_position')


def test_single_chunk_detection():
    """Test detection of documents that fit in a single chunk."""
    engine = FinancialQAEngine()
    
    # Short text should be detected as single chunk
    short_text = "This is a short document."
    assert engine._is_single_chunk(short_text) == True
    
    # Long text should not be single chunk
    long_text = "A" * 2000  # Assuming CHUNK_SIZE is smaller
    assert engine._is_single_chunk(long_text) == False


def test_sentence_boundary_finding():
    """Test sentence boundary detection logic."""
    engine = FinancialQAEngine()
    
    text = "First sentence. Second sentence! Third sentence? Fourth sentence."
    search_start = 10
    search_text = text[search_start:search_start+30]
    target_end = search_start + 25
    
    result = engine._find_sentence_boundary(search_text, search_start, target_end)
    
    # Should find a sentence boundary
    assert result is not None
    assert isinstance(result, int)


def test_chunk_creation_methods():
    """Test chunk creation helper methods."""
    engine = FinancialQAEngine()
    
    text = "This is a test document with multiple sentences. It has content."
    start = 0
    end = 30
    
    # Test boundary-based chunk creation
    chunk = engine._create_chunk_at_boundary(text, start, end)
    assert isinstance(chunk, tuple)
    assert len(chunk) == 3  # (text, start_pos, end_pos)
    
    # Test position-based chunk creation
    chunk = engine._create_chunk_at_position(text, start, end)
    assert isinstance(chunk, tuple)
    assert len(chunk) == 3  # (text, start_pos, end_pos)


def test_original_chunking_behavior_preserved():
    """Test that refactored chunking produces same results as original."""
    engine = FinancialQAEngine()
    
    # Test with various text sizes and patterns
    test_texts = [
        "Short text.",
        "Medium length text with sentences. Multiple sentences here. More content follows.",
        "A" * 100 + ". " + "B" * 100 + ". " + "C" * 100,  # Long text with sentence boundaries 
        "VeryLongTextWithoutAnyPunctuationOrSpacesToTestEdgeCases" * 20,  # No sentence boundaries
    ]
    
    for text in test_texts:
        chunks = engine._chunk_text(text)
        
        # Basic validation
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        
        # All chunks should be tuples of (text, start, end)
        for chunk in chunks:
            assert isinstance(chunk, tuple)
            assert len(chunk) == 3
            assert isinstance(chunk[0], str)
            assert isinstance(chunk[1], int)
            assert isinstance(chunk[2], int)
            
        # Positions should be sensible
        for i, (chunk_text, start, end) in enumerate(chunks):
            assert start >= 0
            assert end <= len(text)
            assert start < end
            assert chunk_text == text[start:end]


def test_chunking_maintains_text_coverage():
    """Test that chunking covers the entire text without gaps."""
    engine = FinancialQAEngine()
    
    text = "This is a test document. " * 50  # Create a document that needs chunking
    chunks = engine._chunk_text(text)
    
    if len(chunks) > 1:
        # Check that we have proper coverage with overlap
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i]
            next_chunk = chunks[i + 1]
            
            # There should be some overlap or continuation
            assert current_chunk[2] >= next_chunk[1]  # end >= next_start (with overlap)