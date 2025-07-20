from finchat_sec_qa.multi_company import compare_question_across_filings
from unittest.mock import patch, MagicMock


def test_success():
    docs = {"a": "alpha beta", "b": "beta gamma"}
    results = compare_question_across_filings("beta", docs)
    assert len(results) == 2
    assert results[0].answer


def test_edge_case_invalid_input():
    docs = {"a": "alpha"}
    try:
        compare_question_across_filings("", docs)
    except ValueError:
        pass
    else:
        assert False
    try:
        compare_question_across_filings("beta", {})
    except ValueError:
        pass
    else:
        assert False


def test_performance_single_engine_instance():
    """Test that the optimized implementation uses a single QA engine instance."""
    docs = {"company1": "financial data 1", "company2": "financial data 2", "company3": "financial data 3"}
    
    with patch('finchat_sec_qa.multi_company.FinancialQAEngine') as mock_engine_class:
        mock_engine = MagicMock()
        mock_engine_class.return_value = mock_engine
        mock_engine.answer_with_citations.return_value = ("test answer", [])
        
        compare_question_across_filings("test question", docs)
        
        # Should create only ONE engine instance, not one per document
        assert mock_engine_class.call_count == 1
        # Should call add_documents with all documents at once
        mock_engine.add_documents.assert_called_once()
        # Should call answer_with_citations only once with the question
        assert mock_engine.answer_with_citations.call_count == len(docs)
