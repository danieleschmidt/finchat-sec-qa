"""
Custom assertion helpers for FinChat-SEC-QA tests.
"""

from typing import Any, Dict, List, Optional
import json
import re


def assert_valid_response_structure(response: Dict[str, Any]):
    """Assert that a response has the expected structure."""
    required_fields = ["answer", "citations", "confidence", "processing_time"]
    
    for field in required_fields:
        assert field in response, f"Missing required field: {field}"
    
    assert isinstance(response["answer"], str), "Answer must be a string"
    assert isinstance(response["citations"], list), "Citations must be a list"
    assert isinstance(response["confidence"], (int, float)), "Confidence must be numeric"
    assert isinstance(response["processing_time"], (int, float)), "Processing time must be numeric"
    
    # Validate citations structure
    for citation in response["citations"]:
        assert isinstance(citation, dict), "Each citation must be a dictionary"
        assert "text" in citation, "Citation must have text"
        assert "source" in citation, "Citation must have source"


def assert_citation_accuracy(citations: List[Dict], source_text: str, min_accuracy: float = 0.9):
    """Assert that citations accurately reference the source text."""
    total_citations = len(citations)
    accurate_citations = 0
    
    for citation in citations:
        citation_text = citation.get("text", "")
        if citation_text.lower() in source_text.lower():
            accurate_citations += 1
    
    accuracy = accurate_citations / total_citations if total_citations > 0 else 0
    assert accuracy >= min_accuracy, f"Citation accuracy {accuracy:.2f} below threshold {min_accuracy}"


def assert_performance_threshold(actual_time: float, threshold: float, operation: str):
    """Assert that operation performance meets threshold."""
    assert actual_time <= threshold, f"{operation} took {actual_time:.2f}s, exceeding threshold {threshold}s"


def assert_security_compliance(response: Dict[str, Any]):
    """Assert that response doesn't contain sensitive information."""
    response_str = json.dumps(response).lower()
    
    # Check for common sensitive patterns
    sensitive_patterns = [
        r'api[_-]?key',
        r'password',
        r'secret',
        r'token',
        r'credential',
        r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',  # Credit card pattern
        r'\b\d{3}[- ]?\d{2}[- ]?\d{4}\b',  # SSN pattern
    ]
    
    for pattern in sensitive_patterns:
        matches = re.findall(pattern, response_str)
        assert not matches, f"Response contains sensitive information matching pattern: {pattern}"


def assert_risk_analysis_quality(risk_analysis: Dict[str, Any]):
    """Assert that risk analysis meets quality standards."""
    assert "sentiment_score" in risk_analysis, "Risk analysis must include sentiment score"
    assert "risk_categories" in risk_analysis, "Risk analysis must include risk categories"
    assert "severity" in risk_analysis, "Risk analysis must include severity assessment"
    
    sentiment = risk_analysis["sentiment_score"]
    assert -1.0 <= sentiment <= 1.0, "Sentiment score must be between -1.0 and 1.0"
    
    severity = risk_analysis["severity"]
    assert severity in ["low", "medium", "high", "critical"], "Invalid severity level"


def assert_edgar_compliance(request_headers: Dict[str, str], rate_limit_info: Dict[str, Any]):
    """Assert that EDGAR API usage complies with SEC requirements."""
    # Check User-Agent header
    assert "User-Agent" in request_headers, "User-Agent header required for EDGAR API"
    user_agent = request_headers["User-Agent"]
    assert "@" in user_agent, "User-Agent must include email address"
    
    # Check rate limiting
    assert "requests_per_second" in rate_limit_info, "Rate limit info missing"
    assert rate_limit_info["requests_per_second"] <= 10, "EDGAR rate limit exceeded"


def assert_multi_company_consistency(responses: List[Dict[str, Any]]):
    """Assert that multi-company analysis responses are consistent."""
    assert len(responses) >= 2, "Multi-company analysis requires at least 2 companies"
    
    # Check that all responses have the same structure
    first_response_keys = set(responses[0].keys())
    for response in responses[1:]:
        assert set(response.keys()) == first_response_keys, "Inconsistent response structure across companies"
    
    # Check that each response has a company identifier
    for response in responses:
        assert "company" in response or "ticker" in response, "Each response must identify the company"


def assert_voice_output_quality(audio_data: bytes, expected_duration_range: tuple = (1, 60)):
    """Assert that voice output meets quality standards."""
    assert len(audio_data) > 0, "Audio data cannot be empty"
    
    # Basic audio file validation (this would need actual audio library for full validation)
    min_duration, max_duration = expected_duration_range
    # Rough estimate: assume 16kHz, 16-bit audio = ~32KB per second
    estimated_duration = len(audio_data) / 32000
    
    assert min_duration <= estimated_duration <= max_duration, \
        f"Audio duration {estimated_duration:.2f}s outside expected range {expected_duration_range}"


def assert_cache_efficiency(cache_stats: Dict[str, Any], min_hit_rate: float = 0.7):
    """Assert that caching is working efficiently."""
    assert "hits" in cache_stats, "Cache stats must include hits"
    assert "misses" in cache_stats, "Cache stats must include misses"
    
    hits = cache_stats["hits"]
    misses = cache_stats["misses"]
    total_requests = hits + misses
    
    if total_requests > 0:
        hit_rate = hits / total_requests
        assert hit_rate >= min_hit_rate, f"Cache hit rate {hit_rate:.2f} below threshold {min_hit_rate}"


def assert_concurrent_safety(results: List[Any], expected_count: int):
    """Assert that concurrent operations completed safely."""
    assert len(results) == expected_count, f"Expected {expected_count} results, got {len(results)}"
    
    # Check that all results are valid (not None or errors)
    valid_results = [r for r in results if r is not None and not isinstance(r, Exception)]
    assert len(valid_results) == expected_count, "Some concurrent operations failed"


def assert_financial_data_accuracy(extracted_data: Dict[str, Any], source_filing: str):
    """Assert that extracted financial data is accurate."""
    # This would implement specific financial data validation logic
    # For now, basic structure validation
    financial_fields = ["revenue", "net_income", "total_assets", "total_debt"]
    
    for field in financial_fields:
        if field in extracted_data:
            value = extracted_data[field]
            assert isinstance(value, (int, float, str)), f"Financial field {field} must be numeric or string"
            
            # If numeric, should be reasonable (not negative revenue for most companies)
            if isinstance(value, (int, float)) and field == "revenue":
                assert value >= 0, "Revenue should not be negative for most companies"