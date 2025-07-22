"""Tests for Prometheus metrics integration."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from prometheus_client import REGISTRY, CollectorRegistry


def test_metrics_endpoint_exists():
    """Test that /metrics endpoint is available."""
    from src.finchat_sec_qa.server import app
    
    client = TestClient(app)
    response = client.get("/metrics")
    
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]


def test_metrics_endpoint_returns_prometheus_format():
    """Test that /metrics returns data in Prometheus format."""
    from src.finchat_sec_qa.server import app
    
    client = TestClient(app)
    response = client.get("/metrics")
    
    content = response.text
    
    # Should contain standard Prometheus metrics
    assert "# HELP" in content
    assert "# TYPE" in content
    

def test_request_metrics_collected():
    """Test that request metrics are properly collected."""
    from src.finchat_sec_qa.server import app
    
    client = TestClient(app)
    
    # Make a few requests to generate metrics
    client.get("/health")
    client.get("/health")
    
    response = client.get("/metrics")
    content = response.text
    
    # Should track HTTP request counts
    assert "http_requests_total" in content
    assert "http_request_duration_seconds" in content


def test_business_metrics_collected():
    """Test that business-specific metrics are collected."""
    from src.finchat_sec_qa.server import app
    
    client = TestClient(app)
    
    # Mock the dependencies to avoid actual API calls
    with patch.object(app.state, 'query_handler') as mock_handler:
        mock_handler.process_query.return_value = ("Test answer", [])
        mock_handler.serialize_citations.return_value = []
        
        # Make a query request
        response = client.post("/query", json={
            "question": "What is the revenue?",
            "ticker": "AAPL",
            "form_type": "10-K",
            "limit": 1
        })
        
    # Check metrics endpoint
    response = client.get("/metrics") 
    content = response.text
    
    # Should track query-specific metrics
    assert "qa_queries_total" in content or "finchat_queries_total" in content


def test_metrics_reset_between_tests():
    """Test that metrics don't leak between test runs."""
    # Use a separate registry for testing
    test_registry = CollectorRegistry()
    
    with patch('src.finchat_sec_qa.server.REGISTRY', test_registry):
        from src.finchat_sec_qa.server import app
        client = TestClient(app)
        
        response = client.get("/metrics")
        assert response.status_code == 200


@pytest.fixture(autouse=True)
def clear_metrics():
    """Clear metrics between tests."""
    # Clear the default registry 
    for collector in list(REGISTRY._collector_to_names.keys()):
        try:
            REGISTRY.unregister(collector)
        except KeyError:
            pass  # Already unregistered
    yield
    # Clear again after test
    for collector in list(REGISTRY._collector_to_names.keys()):
        try:
            REGISTRY.unregister(collector)
        except KeyError:
            pass