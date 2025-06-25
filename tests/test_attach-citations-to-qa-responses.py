import pytest
from finchat_sec_qa.qa_engine import FinancialQAEngine


def test_success():
    engine = FinancialQAEngine()
    engine.add_document("doc1", "alpha beta gamma")
    engine.add_document("doc2", "gamma delta epsilon")
    answer, citations = engine.answer_with_citations("gamma")
    assert answer
    assert len(citations) == 2
    doc_ids = {c.doc_id for c in citations}
    assert {"doc1", "doc2"} == doc_ids


def test_edge_case_invalid_input():
    engine = FinancialQAEngine()
    with pytest.raises(ValueError):
        engine.answer_with_citations("")
