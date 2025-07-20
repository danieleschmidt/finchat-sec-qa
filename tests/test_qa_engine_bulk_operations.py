"""Tests for QA Engine bulk operations optimization."""

import pytest
from unittest.mock import patch, MagicMock
from finchat_sec_qa.qa_engine import FinancialQAEngine


def test_add_documents_bulk_no_rebuilds_until_commit(tmp_path):
    """Test that bulk operations don't rebuild index until commit."""
    store = tmp_path / "index.joblib"
    engine = FinancialQAEngine(storage_path=store)
    
    with patch.object(engine, '_rebuild_index') as mock_rebuild:
        # Add multiple documents in bulk mode
        with engine.bulk_operation():
            engine.add_document("doc1", "alpha beta gamma")
            engine.add_document("doc2", "delta epsilon zeta") 
            engine.add_document("doc3", "eta theta iota")
            # Index should not be rebuilt during bulk operations
            mock_rebuild.assert_not_called()
        
        # Index should be rebuilt exactly once after bulk operation ends
        mock_rebuild.assert_called_once()


def test_add_documents_bulk_performance(tmp_path):
    """Test that bulk operations provide significant performance improvement."""
    store = tmp_path / "index.joblib"
    engine = FinancialQAEngine(storage_path=store)
    
    # Count how many times _rebuild_index is called
    with patch.object(engine, '_rebuild_index', wraps=engine._rebuild_index) as mock_rebuild:
        # Individual operations - should rebuild each time
        engine.add_document("doc1", "individual operation one")
        engine.add_document("doc2", "individual operation two")
        individual_calls = mock_rebuild.call_count
        
        # Reset counter
        mock_rebuild.reset_mock()
        
        # Bulk operations - should rebuild only once
        with engine.bulk_operation():
            engine.add_document("doc3", "bulk operation one")
            engine.add_document("doc4", "bulk operation two")
            engine.add_document("doc5", "bulk operation three")
        bulk_calls = mock_rebuild.call_count
        
        # Bulk should be more efficient
        assert bulk_calls == 1
        assert individual_calls > bulk_calls


def test_bulk_operation_context_manager_exception_safety(tmp_path):
    """Test that bulk operations handle exceptions properly."""
    store = tmp_path / "index.joblib"
    engine = FinancialQAEngine(storage_path=store)
    
    with patch.object(engine, '_rebuild_index') as mock_rebuild:
        try:
            with engine.bulk_operation():
                engine.add_document("doc1", "test document")
                raise ValueError("Simulated error")
        except ValueError:
            pass
        
        # Index should still be rebuilt even if exception occurs
        mock_rebuild.assert_called_once()


def test_bulk_operation_queries_work_correctly(tmp_path):
    """Test that queries work correctly after bulk operations."""
    store = tmp_path / "index.joblib"
    engine = FinancialQAEngine(storage_path=store)
    
    # Add documents in bulk
    with engine.bulk_operation():
        engine.add_document("financial", "revenue increased by 20% this quarter")
        engine.add_document("technical", "cloud infrastructure scaling improvements")
        engine.add_document("legal", "regulatory compliance updates and changes")
    
    # Query should work correctly
    results = engine.query("revenue quarter", top_k=1)
    assert len(results) == 1
    assert results[0][0] == "financial"
    assert "revenue" in results[0][1]


def test_nested_bulk_operations_not_supported():
    """Test that nested bulk operations raise an error."""
    engine = FinancialQAEngine()
    
    with pytest.raises(RuntimeError, match="already in bulk operation mode"):
        with engine.bulk_operation():
            with engine.bulk_operation():
                pass


def test_add_documents_method_convenience():
    """Test the convenience add_documents method for bulk operations."""
    engine = FinancialQAEngine()
    
    documents = [
        ("doc1", "first document text"),
        ("doc2", "second document text"),
        ("doc3", "third document text")
    ]
    
    with patch.object(engine, '_rebuild_index') as mock_rebuild:
        engine.add_documents(documents)
        # Should rebuild only once for all documents
        mock_rebuild.assert_called_once()
    
    # Verify documents were added
    assert len(engine.chunks) == 3
    assert engine.chunks[0].doc_id == "doc1"
    assert engine.chunks[1].doc_id == "doc2"
    assert engine.chunks[2].doc_id == "doc3"