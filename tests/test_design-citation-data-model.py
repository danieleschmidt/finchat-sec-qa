import pytest
from finchat_sec_qa.citation import Citation


def test_success():
    citation = Citation(
        doc_id="doc1", text="hello world", start=0, end=5, section="intro", page=1
    )
    assert citation.doc_id == "doc1"
    assert citation.section == "intro"
    assert citation.page == 1
    assert citation.text == "hello world"


def test_edge_case_invalid_input():
    with pytest.raises(ValueError):
        Citation(doc_id="doc1", text="bad", start=5, end=2, page=0)
