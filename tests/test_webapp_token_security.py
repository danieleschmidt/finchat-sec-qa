"""Tests for webapp token authentication security."""
import pytest
from finchat_sec_qa.webapp import app


def test_query_parameter_token_rejected(monkeypatch):
    """Test that token authentication via query parameters is rejected for security."""
    from finchat_sec_qa import webapp
    monkeypatch.setattr(webapp, 'SECRET_TOKEN', 'test_token_123')
    
    with app.test_client() as client:
        # Query parameter token should be rejected
        resp = client.post('/query?token=test_token_123', 
                          json={'question': 'What is the revenue?', 'ticker': 'AAPL'})
        assert resp.status_code == 401
        assert 'Invalid authentication token' in resp.get_data(as_text=True)


def test_bearer_token_authentication_works(monkeypatch, tmp_path):
    """Test that Bearer token authentication still works."""
    from finchat_sec_qa import webapp
    from unittest.mock import Mock
    
    monkeypatch.setattr(webapp, 'SECRET_TOKEN', 'test_token_123')
    
    # Create mock client
    mock_client = Mock()
    mock_filing = Mock()
    mock_filing.accession_no = '1'
    mock_filing.document_url = str(tmp_path / 'f.html')
    mock_client.get_recent_filings.return_value = [mock_filing]
    
    def download_side_effect(filing):
        path = tmp_path / 'f.html'
        path.write_text('test content for revenue analysis')
        return path
    
    mock_client.download_filing.side_effect = download_side_effect
    monkeypatch.setattr(webapp, 'client', mock_client)
    
    # Clear engine chunks
    webapp.engine.chunks.clear()
    
    with app.test_client() as client:
        # Bearer token should work
        headers = {'Authorization': 'Bearer test_token_123'}
        resp = client.post('/query', 
                          json={'question': 'What is the revenue?', 'ticker': 'AAPL'},
                          headers=headers)
        assert resp.status_code == 200


def test_missing_authorization_header_rejected(monkeypatch):
    """Test that requests without authorization are rejected."""
    from finchat_sec_qa import webapp
    monkeypatch.setattr(webapp, 'SECRET_TOKEN', 'test_token_123')
    
    with app.test_client() as client:
        # No authorization should be rejected
        resp = client.post('/query', 
                          json={'question': 'What is the revenue?', 'ticker': 'AAPL'})
        assert resp.status_code == 401


def test_malformed_bearer_token_rejected(monkeypatch):
    """Test that malformed Bearer tokens are rejected."""
    from finchat_sec_qa import webapp
    monkeypatch.setattr(webapp, 'SECRET_TOKEN', 'test_token_123')
    
    with app.test_client() as client:
        # Malformed Bearer token should be rejected
        headers = {'Authorization': 'Bearer'}  # Missing actual token
        resp = client.post('/query', 
                          json={'question': 'What is the revenue?', 'ticker': 'AAPL'},
                          headers=headers)
        assert resp.status_code == 401


def test_wrong_bearer_token_rejected(monkeypatch):
    """Test that wrong Bearer tokens are rejected."""
    from finchat_sec_qa import webapp
    monkeypatch.setattr(webapp, 'SECRET_TOKEN', 'correct_token')
    
    with app.test_client() as client:
        # Wrong Bearer token should be rejected
        headers = {'Authorization': 'Bearer wrong_token'}
        resp = client.post('/query', 
                          json={'question': 'What is the revenue?', 'ticker': 'AAPL'},
                          headers=headers)
        assert resp.status_code == 401