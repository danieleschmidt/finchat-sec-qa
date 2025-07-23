"""Tests for CORS configuration security."""

import pytest
from unittest.mock import patch, MagicMock
from flask import Flask
from fastapi.testclient import TestClient
from fastapi import FastAPI


def test_flask_cors_headers_present():
    """Test that Flask webapp includes proper CORS headers."""
    with patch('finchat_sec_qa.webapp.EdgarClient'), \
         patch('finchat_sec_qa.webapp.FinancialQAEngine'), \
         patch('finchat_sec_qa.webapp.QueryHandler'):
        
        from finchat_sec_qa.webapp import app
        client = app.test_client()
        
        # Test OPTIONS preflight request
        response = client.options('/qa')
        
        # Should have CORS headers
        assert 'Access-Control-Allow-Origin' in response.headers
        assert 'Access-Control-Allow-Methods' in response.headers
        assert 'Access-Control-Allow-Headers' in response.headers
        
        # Should NOT allow all origins (security risk)
        assert response.headers.get('Access-Control-Allow-Origin') != '*'


def test_flask_cors_origin_whitelist():
    """Test that Flask webapp enforces origin whitelist."""
    with patch('finchat_sec_qa.webapp.EdgarClient'), \
         patch('finchat_sec_qa.webapp.FinancialQAEngine'), \
         patch('finchat_sec_qa.webapp.QueryHandler'):
        
        from finchat_sec_qa.webapp import app
        client = app.test_client()
        
        # Test with malicious origin
        response = client.options('/qa', headers={'Origin': 'https://malicious-site.com'})
        
        # Should not include CORS headers for unauthorized origins
        cors_origin = response.headers.get('Access-Control-Allow-Origin')
        assert cors_origin is None or cors_origin != 'https://malicious-site.com'


def test_fastapi_cors_headers_present():
    """Test that FastAPI server includes proper CORS headers."""
    with patch('finchat_sec_qa.server.AsyncEdgarClient'), \
         patch('finchat_sec_qa.server.FinancialQAEngine'), \
         patch('finchat_sec_qa.server.AsyncQueryHandler'):
        
        from finchat_sec_qa.server import app
        client = TestClient(app)
        
        # Test OPTIONS preflight request
        response = client.options('/qa')
        
        # Should have CORS headers
        assert 'access-control-allow-origin' in response.headers
        assert 'access-control-allow-methods' in response.headers
        assert 'access-control-allow-headers' in response.headers
        
        # Should NOT allow all origins (security risk)
        assert response.headers.get('access-control-allow-origin') != '*'


def test_fastapi_cors_origin_whitelist():
    """Test that FastAPI server enforces origin whitelist."""
    with patch('finchat_sec_qa.server.AsyncEdgarClient'), \
         patch('finchat_sec_qa.server.FinancialQAEngine'), \
         patch('finchat_sec_qa.server.AsyncQueryHandler'):
        
        from finchat_sec_qa.server import app
        client = TestClient(app)
        
        # Test with malicious origin
        response = client.options('/qa', headers={'Origin': 'https://malicious-site.com'})
        
        # Should not include CORS headers for unauthorized origins
        cors_origin = response.headers.get('access-control-allow-origin')
        assert cors_origin is None or cors_origin != 'https://malicious-site.com'


def test_cors_credentials_not_allowed_with_wildcard():
    """Test that credentials are not allowed when using wildcard origins."""
    with patch('finchat_sec_qa.webapp.EdgarClient'), \
         patch('finchat_sec_qa.webapp.FinancialQAEngine'), \
         patch('finchat_sec_qa.webapp.QueryHandler'):
        
        from finchat_sec_qa.webapp import app
        client = app.test_client()
        
        response = client.options('/qa')
        
        # If wildcard origin is used, credentials should not be allowed
        cors_origin = response.headers.get('Access-Control-Allow-Origin')
        cors_credentials = response.headers.get('Access-Control-Allow-Credentials')
        
        if cors_origin == '*':
            assert cors_credentials != 'true'


def test_cors_allowed_methods_restrictive():
    """Test that CORS allowed methods are restrictive."""
    with patch('finchat_sec_qa.webapp.EdgarClient'), \
         patch('finchat_sec_qa.webapp.FinancialQAEngine'), \
         patch('finchat_sec_qa.webapp.QueryHandler'):
        
        from finchat_sec_qa.webapp import app
        client = app.test_client()
        
        response = client.options('/qa')
        
        allowed_methods = response.headers.get('Access-Control-Allow-Methods', '')
        
        # Should not include dangerous methods
        dangerous_methods = ['DELETE', 'PUT', 'PATCH']
        for method in dangerous_methods:
            assert method not in allowed_methods.upper()