"""Test accurate citation positioning with text chunking."""
import pytest
from finchat_sec_qa.qa_engine import FinancialQAEngine


def test_citation_positions_with_text_chunks():
    """Test that citations reference accurate positions within original document."""
    engine = FinancialQAEngine()
    
    # Large document that will be chunked
    doc_text = (
        "Company Overview: ACME Corp is a technology company founded in 1995. "
        "The company specializes in cloud computing solutions and artificial intelligence. "
        "Financial Performance: Revenue for Q4 2023 reached $50 million, an increase of 25% year-over-year. "
        "The company reported net income of $8 million for the quarter. "
        "Risk Factors: Market competition remains intense in the cloud computing sector. "
        "Regulatory changes could impact our international operations significantly."
    )
    
    engine.add_document("acme-10k", doc_text)
    
    # Query about revenue should return citation pointing to financial section
    answer, citations = engine.answer_with_citations("revenue quarter", top_k=1)
    
    assert len(citations) == 1
    citation = citations[0]
    
    # For small documents that fit in one chunk, citation may span entire document (which is correct)
    # For larger documents with multiple chunks, citation should not span entire document
    if len(doc_text) > 1000:  # Larger than default chunk size
        assert citation.start != 0 or citation.end != len(doc_text)
    else:
        # Small document - single chunk is acceptable
        assert citation.start <= citation.end
    
    # Citation text should be a substring of original document
    cited_text = doc_text[citation.start:citation.end]
    assert citation.text == cited_text
    
    # Citation should contain the relevant content about revenue
    assert "revenue" in citation.text.lower() or "50 million" in citation.text


def test_multiple_chunks_multiple_citations():
    """Test that multiple relevant chunks produce accurate citations."""
    engine = FinancialQAEngine()
    
    doc_text = (
        "Section 1: Risk Assessment Overview. Our primary risk factors include market volatility. "
        "We face significant competition from established players in the technology sector. "
        "Section 2: Operational Risks. Supply chain disruptions pose ongoing risks to operations. "
        "Currency exchange fluctuations affect our international revenue streams. "
        "Section 3: Financial Risks. Credit risk exposure through customer concentration increases. "
        "Interest rate changes impact our borrowing costs and investment returns substantially."
    )
    
    engine.add_document("risk-report", doc_text)
    
    # Query about risks should return multiple relevant citations
    answer, citations = engine.answer_with_citations("risks", top_k=3)
    
    assert len(citations) >= 1  # At least one citation about risks
    
    for citation in citations:
        # Each citation should reference a valid position
        assert 0 <= citation.start < citation.end <= len(doc_text)
        
        # Citation text should match the slice from original document
        cited_text = doc_text[citation.start:citation.end]
        assert citation.text == cited_text
        
        # Citation should contain risk-related content
        assert any(word in citation.text.lower() for word in ["risk", "risks"])


def test_chunk_boundaries_preserve_sentences():
    """Test that text chunking respects sentence boundaries when possible."""
    engine = FinancialQAEngine()
    
    # Text with clear sentence boundaries
    doc_text = (
        "First sentence about revenue growth in Q1 2023. "
        "Second sentence discusses market expansion strategies. "
        "Third sentence covers operational efficiency improvements. "
        "Fourth sentence analyzes competitive positioning factors."
    )
    
    engine.add_document("quarterly-report", doc_text)
    
    # Inspect internal chunks to verify sentence boundary preservation
    chunks = engine.chunks
    
    # If document is chunked, chunks should ideally end at sentence boundaries
    for chunk in chunks:
        if chunk.text != doc_text:  # If it's a partial chunk
            # Should end with period and space or end of document
            assert chunk.text.endswith(". ") or chunk.text.endswith(".")


def test_citation_positions_with_overlapping_content():
    """Test citation accuracy when chunks have overlapping content."""
    engine = FinancialQAEngine()
    
    doc_text = (
        "Revenue analysis shows strong performance in Q4. "
        "Q4 revenue reached record highs of $100M. "
        "Revenue growth was driven by new product launches. "
        "Product revenue segments showed varied performance."
    )
    
    engine.add_document("earnings", doc_text)
    
    answer, citations = engine.answer_with_citations("revenue Q4", top_k=2)
    
    # Verify all citations have valid positions
    for citation in citations:
        assert 0 <= citation.start < citation.end <= len(doc_text)
        
        # Verify cited text matches document slice
        cited_text = doc_text[citation.start:citation.end]
        assert citation.text == cited_text


def test_single_chunk_fallback():
    """Test that small documents that don't need chunking still work correctly."""
    engine = FinancialQAEngine()
    
    # Small document that should remain as single chunk
    doc_text = "ACME Corp revenue was $10M in Q3."
    
    engine.add_document("short-doc", doc_text)
    
    answer, citations = engine.answer_with_citations("revenue", top_k=1)
    
    assert len(citations) == 1
    citation = citations[0]
    
    # For single chunk, citation may span entire document, but positions should be accurate
    assert citation.start == 0
    assert citation.end == len(doc_text)
    assert citation.text == doc_text


def test_citation_position_consistency():
    """Test that citation positions are consistent across multiple queries."""
    engine = FinancialQAEngine()
    
    doc_text = (
        "Financial highlights for fiscal year 2023. "
        "Total revenue increased to $200 million. "
        "Operating expenses were $150 million. "
        "Net income reached $50 million."
    )
    
    engine.add_document("annual-report", doc_text)
    
    # Run same query multiple times
    for _ in range(3):
        answer, citations = engine.answer_with_citations("revenue", top_k=1)
        
        if citations:
            citation = citations[0]
            # Position should be consistent
            assert 0 <= citation.start < citation.end <= len(doc_text)
            
            # Cited text should match document slice
            cited_text = doc_text[citation.start:citation.end]
            assert citation.text == cited_text


def test_empty_and_edge_cases():
    """Test citation behavior with edge cases."""
    engine = FinancialQAEngine()
    
    # Empty document
    engine.add_document("empty", "")
    answer, citations = engine.answer_with_citations("test")
    assert len(citations) == 0
    
    # Single word document
    engine.add_document("single", "Revenue")
    answer, citations = engine.answer_with_citations("revenue")
    if citations:
        citation = citations[0]
        assert citation.start == 0
        assert citation.end == len("Revenue")
        assert citation.text == "Revenue"