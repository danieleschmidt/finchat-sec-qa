"""Tests for enhanced authentication security features."""

import pytest
import time
from unittest.mock import patch, MagicMock
from flask import Flask
from finchat_sec_qa.webapp import app


@pytest.fixture
def client():
    """Create test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_token_validation_with_header_auth(client):
    """Test token validation via Authorization header."""
    with patch.dict('os.environ', {'FINCHAT_TOKEN': 'test_token'}):
        # Valid token in header
        response = client.post('/query',
                              json={'question': 'test', 'ticker': 'AAPL'},
                              headers={'Authorization': 'Bearer test_token'})
        assert response.status_code != 401  # Should not be unauthorized
        
        # Invalid token in header
        response = client.post('/query',
                              json={'question': 'test', 'ticker': 'AAPL'},
                              headers={'Authorization': 'Bearer wrong_token'})
        assert response.status_code == 401


def test_token_validation_constant_time():
    """Test that token validation takes constant time to prevent timing attacks."""
    from finchat_sec_qa.webapp import _validate_token_constant_time
    
    correct_token = "super_secret_token_12345"
    
    # Time validation of correct token
    start_time = time.time()
    result1 = _validate_token_constant_time("super_secret_token_12345", correct_token)
    time1 = time.time() - start_time
    
    # Time validation of incorrect token of same length
    start_time = time.time()
    result2 = _validate_token_constant_time("wrong_secret_token_12345", correct_token)
    time2 = time.time() - start_time
    
    # Time validation of incorrect token of different length
    start_time = time.time()
    result3 = _validate_token_constant_time("short", correct_token)
    time3 = time.time() - start_time
    
    assert result1 is True
    assert result2 is False
    assert result3 is False
    
    # Times should be relatively similar (within 50% of each other)
    # to prevent timing attacks
    max_time = max(time1, time2, time3)
    min_time = min(time1, time2, time3)
    assert (max_time - min_time) / max_time < 0.5


def test_rate_limiting():
    """Test rate limiting functionality."""
    from finchat_sec_qa.webapp import RateLimiter
    
    limiter = RateLimiter(max_requests=3, window_seconds=60)
    client_ip = "192.168.1.1"
    
    # First 3 requests should succeed
    assert limiter.is_allowed(client_ip) is True
    assert limiter.is_allowed(client_ip) is True
    assert limiter.is_allowed(client_ip) is True
    
    # 4th request should be rate limited
    assert limiter.is_allowed(client_ip) is False
    
    # Different IP should still work
    assert limiter.is_allowed("192.168.1.2") is True


def test_rate_limiting_window_reset():
    """Test that rate limiting window resets properly."""
    from finchat_sec_qa.webapp import RateLimiter
    
    limiter = RateLimiter(max_requests=2, window_seconds=1)
    client_ip = "192.168.1.1"
    
    # Use up rate limit
    assert limiter.is_allowed(client_ip) is True
    assert limiter.is_allowed(client_ip) is True
    assert limiter.is_allowed(client_ip) is False
    
    # Wait for window to reset
    time.sleep(1.1)
    
    # Should be allowed again
    assert limiter.is_allowed(client_ip) is True


def test_brute_force_protection():
    """Test brute force protection with exponential backoff."""
    from finchat_sec_qa.webapp import BruteForceProtection
    
    protection = BruteForceProtection()
    client_ip = "192.168.1.1"
    
    # First few failed attempts should not trigger protection
    assert protection.is_blocked(client_ip) is False
    protection.record_failed_attempt(client_ip)
    assert protection.is_blocked(client_ip) is False
    
    protection.record_failed_attempt(client_ip)
    protection.record_failed_attempt(client_ip)
    
    # After multiple failures, should be blocked
    assert protection.is_blocked(client_ip) is True
    
    # Successful auth should reset
    protection.record_successful_attempt(client_ip)
    assert protection.is_blocked(client_ip) is False


def test_token_strength_validation():
    """Test token strength validation."""
    from finchat_sec_qa.webapp import validate_token_strength
    
    # Strong tokens should pass
    assert validate_token_strength("StrongToken123!@#$abcd") is True
    
    # Weak tokens should fail
    assert validate_token_strength("weak") is False
    assert validate_token_strength("12345678") is False
    assert validate_token_strength("password") is False
    assert validate_token_strength("") is False
    assert validate_token_strength(None) is False


def test_security_headers_middleware():
    """Test that security headers are added to responses."""
    with patch.dict('os.environ', {'FINCHAT_TOKEN': 'test_token'}):
        response = app.test_client().get('/',
                                       headers={'Authorization': 'Bearer test_token'})
        
        # Check for security headers
        assert 'X-Content-Type-Options' in response.headers
        assert response.headers['X-Content-Type-Options'] == 'nosniff'
        assert 'X-Frame-Options' in response.headers
        assert response.headers['X-Frame-Options'] == 'DENY'
        assert 'X-XSS-Protection' in response.headers
        assert response.headers['X-XSS-Protection'] == '1; mode=block'


def test_auth_without_token_configured():
    """Test that auth is bypassed when no token is configured."""
    with patch.dict('os.environ', {}, clear=True):
        client = app.test_client()
        # Should not require auth when no token is set
        response = client.post('/query', json={'question': 'test', 'ticker': 'AAPL'})
        assert response.status_code != 401  # Should not be unauthorized due to missing auth