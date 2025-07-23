"""Tests for request security limits and CSRF protection."""
import pytest
from unittest.mock import Mock, patch
from finchat_sec_qa.webapp import app
from finchat_sec_qa.server import app as fastapi_app
from fastapi.testclient import TestClient


class TestRequestSizeLimits:
    """Test request size limits for DoS prevention."""
    
    def test_flask_request_size_limit_enforced(self):
        """Test that Flask enforces request size limits."""
        from finchat_sec_qa import webapp
        
        # Large payload (>1MB should be rejected)
        large_payload = {'question': 'X' * (1024 * 1024 + 1), 'ticker': 'AAPL'}
        
        with app.test_client() as client:
            headers = {'Authorization': 'Bearer test_token'}
            with patch.object(webapp, 'SECRET_TOKEN', 'test_token'):
                resp = client.post('/query', json=large_payload, headers=headers)
                
                # Should be rejected due to size limit
                assert resp.status_code == 413  # Request Entity Too Large
    
    def test_fastapi_request_size_limit_enforced(self):
        """Test that FastAPI enforces request size limits."""
        # Large payload (>1MB should be rejected)
        large_payload = {'question': 'X' * (1024 * 1024 + 1), 'ticker': 'AAPL'}
        
        with TestClient(fastapi_app) as client:
            resp = client.post('/query', json=large_payload)
            
            # Should be rejected due to size limit
            assert resp.status_code == 413  # Request Entity Too Large
    
    def test_normal_request_size_allowed(self):
        """Test that normal-sized requests are still allowed."""
        from finchat_sec_qa import webapp
        
        # Normal payload (should be accepted)
        normal_payload = {'question': 'What is the revenue?', 'ticker': 'AAPL'}
        
        with app.test_client() as client:
            headers = {'Authorization': 'Bearer test_token'}
            with patch.object(webapp, 'SECRET_TOKEN', 'test_token'):
                with patch.object(webapp, 'client') as mock_client:
                    # Mock successful response
                    mock_filing = Mock()
                    mock_filing.accession_no = '1'
                    mock_filing.document_url = '/tmp/test.html'
                    mock_client.get_recent_filings.return_value = [mock_filing]
                    mock_client.download_filing.return_value = '/tmp/test.html'
                    
                    # Mock file content
                    with patch('pathlib.Path.read_text', return_value='test content'):
                        resp = client.post('/query', json=normal_payload, headers=headers)
                        
                        # Should be accepted (might fail for other reasons, but not size)
                        assert resp.status_code != 413
    
    def test_configurable_size_limit(self):
        """Test that request size limit is configurable."""
        from finchat_sec_qa.config import get_config
        
        config = get_config()
        
        # Should have a configurable max request size
        assert hasattr(config, 'MAX_REQUEST_SIZE_MB')
        assert isinstance(config.MAX_REQUEST_SIZE_MB, int)
        assert config.MAX_REQUEST_SIZE_MB > 0


class TestCSRFProtection:
    """Test CSRF protection for state-changing operations."""
    
    def test_csrf_token_required_for_post_requests(self):
        """Test that CSRF tokens are required for POST requests."""
        from finchat_sec_qa import webapp
        
        payload = {'question': 'What is the revenue?', 'ticker': 'AAPL'}
        
        with app.test_client() as client:
            headers = {'Authorization': 'Bearer test_token'}
            with patch.object(webapp, 'SECRET_TOKEN', 'test_token'):
                # Request without CSRF token should be rejected
                resp = client.post('/query', json=payload, headers=headers)
                
                # Should be rejected due to missing CSRF token
                assert resp.status_code == 403  # Forbidden
                assert 'CSRF' in resp.get_data(as_text=True)
    
    def test_csrf_token_validation(self):
        """Test that CSRF tokens are properly validated."""
        from finchat_sec_qa import webapp
        
        payload = {'question': 'What is the revenue?', 'ticker': 'AAPL'}
        
        with app.test_client() as client:
            headers = {
                'Authorization': 'Bearer test_token',
                'X-CSRF-Token': 'invalid_token'
            }
            with patch.object(webapp, 'SECRET_TOKEN', 'test_token'):
                # Request with invalid CSRF token should be rejected
                resp = client.post('/query', json=payload, headers=headers)
                
                assert resp.status_code == 403
                assert 'CSRF' in resp.get_data(as_text=True)
    
    def test_csrf_token_generation_endpoint(self):
        """Test that CSRF tokens can be generated."""
        from finchat_sec_qa import webapp
        
        with app.test_client() as client:
            headers = {'Authorization': 'Bearer test_token'}
            with patch.object(webapp, 'SECRET_TOKEN', 'test_token'):
                # Should have endpoint to get CSRF token
                resp = client.get('/csrf-token', headers=headers)
                
                assert resp.status_code == 200
                data = resp.get_json()
                assert 'csrf_token' in data
                assert len(data['csrf_token']) > 20  # Should be reasonably long
    
    def test_valid_csrf_token_allows_request(self):
        """Test that valid CSRF tokens allow requests to proceed."""
        from finchat_sec_qa import webapp
        
        with app.test_client() as client:
            headers = {'Authorization': 'Bearer test_token'}
            with patch.object(webapp, 'SECRET_TOKEN', 'test_token'):
                # First get a valid CSRF token
                csrf_resp = client.get('/csrf-token', headers=headers)
                csrf_token = csrf_resp.get_json()['csrf_token']
                
                # Use the token in a request
                headers['X-CSRF-Token'] = csrf_token
                payload = {'question': 'What is the revenue?', 'ticker': 'AAPL'}
                
                with patch.object(webapp, 'client') as mock_client:
                    mock_filing = Mock()
                    mock_filing.accession_no = '1'
                    mock_filing.document_url = '/tmp/test.html'
                    mock_client.get_recent_filings.return_value = [mock_filing]
                    mock_client.download_filing.return_value = '/tmp/test.html'
                    
                    with patch('pathlib.Path.read_text', return_value='test content'):
                        resp = client.post('/query', json=payload, headers=headers)
                        
                        # Should not be rejected for CSRF (might fail for other reasons)
                        assert resp.status_code != 403
    
    def test_csrf_exempt_endpoints(self):
        """Test that certain endpoints are exempt from CSRF protection."""
        # Health check and metrics endpoints should be exempt
        with TestClient(fastapi_app) as client:
            health_resp = client.get('/health')
            assert health_resp.status_code == 200  # Should not require CSRF
            
            metrics_resp = client.get('/metrics')
            assert metrics_resp.status_code == 200  # Should not require CSRF
    
    def test_csrf_token_expiration(self):
        """Test that CSRF tokens expire after reasonable time."""
        from finchat_sec_qa import webapp
        import time
        
        with app.test_client() as client:
            headers = {'Authorization': 'Bearer test_token'}
            with patch.object(webapp, 'SECRET_TOKEN', 'test_token'):
                # Get CSRF token
                csrf_resp = client.get('/csrf-token', headers=headers)
                csrf_token = csrf_resp.get_json()['csrf_token']
                
                # Fast-forward time (mock time.time)
                with patch('time.time', return_value=time.time() + 3600):  # 1 hour later
                    headers['X-CSRF-Token'] = csrf_token
                    payload = {'question': 'What is the revenue?', 'ticker': 'AAPL'}
                    
                    resp = client.post('/query', json=payload, headers=headers)
                    
                    # Expired token should be rejected
                    assert resp.status_code == 403
                    assert 'expired' in resp.get_data(as_text=True).lower()


class TestSecurityHeaders:
    """Test additional security headers."""
    
    def test_security_headers_present(self):
        """Test that security headers are present in responses."""
        with TestClient(fastapi_app) as client:
            resp = client.get('/health')
            
            # Should have security headers
            assert 'X-Content-Type-Options' in resp.headers
            assert resp.headers['X-Content-Type-Options'] == 'nosniff'
            
            assert 'X-Frame-Options' in resp.headers
            assert resp.headers['X-Frame-Options'] == 'DENY'
            
            assert 'Strict-Transport-Security' in resp.headers
    
    def test_content_security_policy_header(self):
        """Test that CSP header is present and restrictive."""
        from finchat_sec_qa import webapp
        
        with app.test_client() as client:
            headers = {'Authorization': 'Bearer test_token'}
            with patch.object(webapp, 'SECRET_TOKEN', 'test_token'):
                resp = client.get('/csrf-token', headers=headers)
                
                assert 'Content-Security-Policy' in resp.headers
                csp = resp.headers['Content-Security-Policy']
                assert 'default-src' in csp
                assert "'self'" in csp