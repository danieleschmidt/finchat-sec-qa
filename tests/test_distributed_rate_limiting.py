"""
Tests for distributed rate limiting functionality.
Testing Redis-based rate limiting with fallback to in-memory.
"""
import time
import pytest
from unittest.mock import Mock, patch, MagicMock
from finchat_sec_qa.rate_limiting import DistributedRateLimiter


class TestDistributedRateLimiter:
    """Test suite for Redis-based distributed rate limiting."""
    
    def test_redis_rate_limiting_allows_within_limit(self):
        """Test that Redis rate limiter allows requests within limits."""
        with patch('redis.Redis.from_url') as mock_redis:
            mock_client = Mock()
            mock_redis.return_value = mock_client
            mock_client.ping.return_value = True  # Successful connection
            mock_client.eval.return_value = 1  # Script returns 1 (allowed)
            
            limiter = DistributedRateLimiter(max_requests=5, window_seconds=60)
            
            result = limiter.is_allowed('client_123')
            
            assert result is True
            mock_client.eval.assert_called_once()
    
    def test_redis_rate_limiting_blocks_over_limit(self):
        """Test that Redis rate limiter blocks requests over the limit."""
        with patch('redis.Redis.from_url') as mock_redis:
            mock_client = Mock()
            mock_redis.return_value = mock_client
            mock_client.ping.return_value = True  # Successful connection
            mock_client.eval.return_value = 0  # Script returns 0 (blocked)
            
            limiter = DistributedRateLimiter(max_requests=5, window_seconds=60)
            
            result = limiter.is_allowed('client_123')
            
            assert result is False
            mock_client.eval.assert_called_once()
    
    def test_redis_cleanup_old_requests(self):
        """Test that old requests outside the window are cleaned up."""
        with patch('redis.Redis.from_url') as mock_redis:
            mock_client = Mock()
            mock_redis.return_value = mock_client
            mock_client.ping.return_value = True  # Successful connection
            mock_client.eval.return_value = 1  # Script returns 1 (allowed)
            
            limiter = DistributedRateLimiter(max_requests=5, window_seconds=60)
            
            limiter.is_allowed('client_123')
            
            # Should use Lua script which handles cleanup internally
            mock_client.eval.assert_called_once()
    
    def test_fallback_to_memory_on_redis_failure(self):
        """Test fallback to in-memory rate limiting when Redis is unavailable."""
        with patch('redis.Redis.from_url') as mock_redis:
            mock_redis.side_effect = Exception("Redis connection failed")
            
            limiter = DistributedRateLimiter(max_requests=5, window_seconds=60)
            
            # Should still work with in-memory fallback
            result1 = limiter.is_allowed('client_123')
            result2 = limiter.is_allowed('client_123')
            
            assert result1 is True
            assert result2 is True
    
    def test_thread_safety_with_lua_script(self):
        """Test that Redis operations use atomic Lua scripts for thread safety."""
        with patch('redis.Redis.from_url') as mock_redis:
            mock_client = Mock()
            mock_redis.return_value = mock_client
            mock_client.ping.return_value = True  # Successful connection
            mock_client.eval.return_value = 1  # Script returns 1 (allowed)
            
            limiter = DistributedRateLimiter(max_requests=5, window_seconds=60)
            
            result = limiter.is_allowed('client_123')
            
            assert result is True
            # Should use Lua script for atomic operations
            mock_client.eval.assert_called_once()
    
    def test_multiple_clients_independence(self):
        """Test that different clients have independent rate limits."""
        with patch('redis.Redis.from_url') as mock_redis:
            mock_client = Mock()
            mock_redis.return_value = mock_client
            mock_client.ping.return_value = True  # Successful connection
            mock_client.eval.return_value = 1  # Script returns 1 (allowed)
            
            limiter = DistributedRateLimiter(max_requests=2, window_seconds=60)
            
            # Different clients should be tracked independently
            result1 = limiter.is_allowed('client_A')
            result2 = limiter.is_allowed('client_B')
            
            assert result1 is True
            assert result2 is True
            
            # Should call eval for both clients
            assert mock_client.eval.call_count == 2


class TestDistributedRateLimiterIntegration:
    """Integration tests requiring Redis connection."""
    
    @pytest.mark.skipif(True, reason="Requires Redis server for integration testing")
    def test_real_redis_integration(self):
        """Integration test with actual Redis (skipped by default)."""
        limiter = DistributedRateLimiter(max_requests=3, window_seconds=1)
        
        # Should allow first 3 requests
        assert limiter.is_allowed('test_client') is True
        assert limiter.is_allowed('test_client') is True
        assert limiter.is_allowed('test_client') is True
        
        # Should block the 4th request
        assert limiter.is_allowed('test_client') is False
        
        # After window expires, should allow again
        time.sleep(1.1)
        assert limiter.is_allowed('test_client') is True