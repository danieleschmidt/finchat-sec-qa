import pytest
from finchat_sec_qa.citation import extract_citation_anchors


def test_success():
    html = (
        '<span data-section="Item 1" data-page="5">Business</span>'
        " text "
        '<span data-section="Item 2" data-page="10">Risk</span>'
    )
    citations = extract_citation_anchors("doc1", html)
    assert len(citations) == 2
    assert citations[0].section == "Item 1"
    assert citations[0].page == 5
    assert citations[0].text == "Business"


def test_edge_case_invalid_input():
    with pytest.raises(ValueError):
        extract_citation_anchors("", "<span></span>")
    with pytest.raises(ValueError):
        extract_citation_anchors("doc", "")
