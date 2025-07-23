"""Tests for Redis connection pooling implementation."""
import unittest
from unittest.mock import patch, Mock, MagicMock
from finchat_sec_qa.rate_limiting import DistributedRateLimiter


class TestRedisConnectionPooling(unittest.TestCase):
    """Test Redis connection pooling for better performance."""
    
    def test_redis_connection_pool_created(self):
        """Test that Redis connection pool is created instead of direct connection."""
        with patch('redis.ConnectionPool.from_url') as mock_pool_from_url, \
             patch('redis.Redis') as mock_redis:
            
            mock_pool = Mock()
            mock_pool_from_url.return_value = mock_pool
            mock_redis_instance = Mock()
            mock_redis.return_value = mock_redis_instance
            mock_redis_instance.ping.return_value = True
            
            limiter = DistributedRateLimiter()
            
            # Should create connection pool with proper settings
            mock_pool_from_url.assert_called_once()
            args, kwargs = mock_pool_from_url.call_args
            
            # Verify connection pool parameters
            self.assertIn('max_connections', kwargs)
            self.assertIn('decode_responses', kwargs)
            self.assertIn('socket_connect_timeout', kwargs)
            self.assertIn('socket_timeout', kwargs)
            self.assertTrue(kwargs['decode_responses'])
            self.assertGreater(kwargs['max_connections'], 0)
            
            # Should use connection pool for Redis client
            mock_redis.assert_called_once_with(connection_pool=mock_pool)
    
    def test_redis_connection_pool_configuration(self):
        """Test that connection pool uses appropriate configuration."""
        with patch('redis.ConnectionPool.from_url') as mock_pool_from_url, \
             patch('redis.Redis') as mock_redis:
            
            mock_pool = Mock()
            mock_pool_from_url.return_value = mock_pool
            mock_redis_instance = Mock()
            mock_redis.return_value = mock_redis_instance
            mock_redis_instance.ping.return_value = True
            
            limiter = DistributedRateLimiter()
            
            args, kwargs = mock_pool_from_url.call_args
            
            # Should have appropriate pool size for rate limiting workload
            self.assertGreaterEqual(kwargs['max_connections'], 5)  # Minimum for production
            self.assertLessEqual(kwargs['max_connections'], 50)  # Reasonable upper bound
            
            # Should have proper timeouts
            self.assertEqual(kwargs['socket_connect_timeout'], 1)
            self.assertEqual(kwargs['socket_timeout'], 1)
            
            # Should enable response decoding
            self.assertTrue(kwargs['decode_responses'])
    
    def test_redis_pool_reuses_connections(self):
        """Test that multiple operations reuse connections from pool."""
        with patch('redis.ConnectionPool.from_url') as mock_pool_from_url, \
             patch('redis.Redis') as mock_redis:
            
            mock_pool = Mock()
            mock_pool_from_url.return_value = mock_pool
            mock_redis_instance = Mock()
            mock_redis.return_value = mock_redis_instance
            mock_redis_instance.ping.return_value = True
            mock_redis_instance.eval.return_value = 1
            
            limiter = DistributedRateLimiter()
            
            # Perform multiple operations
            limiter.is_allowed('client1')
            limiter.is_allowed('client2')
            limiter.is_allowed('client3')
            
            # Should create pool only once, not per operation
            mock_pool_from_url.assert_called_once()
            mock_redis.assert_called_once()
            
            # Should reuse the same Redis client for all operations
            self.assertEqual(mock_redis_instance.eval.call_count, 3)
    
    def test_redis_pool_fallback_on_failure(self):
        """Test fallback to in-memory when Redis pool fails."""
        with patch('redis.ConnectionPool.from_url') as mock_pool_from_url:
            # Simulate connection pool creation failure
            mock_pool_from_url.side_effect = Exception("Redis connection failed")
            
            limiter = DistributedRateLimiter()
            
            # Should fall back to in-memory storage
            self.assertIsNone(limiter.redis_client)
            
            # Should still work with fallback storage
            result = limiter.is_allowed('test_client')
            self.assertTrue(result)  # First request should be allowed
    
    def test_redis_pool_handles_connection_errors(self):
        """Test handling of Redis connection errors during operations."""
        with patch('redis.ConnectionPool.from_url') as mock_pool_from_url, \
             patch('redis.Redis') as mock_redis:
            
            mock_pool = Mock()
            mock_pool_from_url.return_value = mock_pool
            mock_redis_instance = Mock()
            mock_redis.return_value = mock_redis_instance
            mock_redis_instance.ping.return_value = True
            
            # Simulate operation failure
            mock_redis_instance.eval.side_effect = Exception("Connection lost")
            
            limiter = DistributedRateLimiter()
            
            # Should fallback to in-memory on operation failure
            result = limiter.is_allowed('test_client')
            self.assertTrue(result)  # Should work via fallback
            
            # Redis client should be set to None after failure
            self.assertIsNone(limiter.redis_client)
    
    def test_connection_pool_resource_cleanup(self):
        """Test that connection pool properly manages resources."""
        with patch('redis.ConnectionPool.from_url') as mock_pool_from_url, \
             patch('redis.Redis') as mock_redis:
            
            mock_pool = Mock()
            mock_pool_from_url.return_value = mock_pool
            mock_redis_instance = Mock()
            mock_redis.return_value = mock_redis_instance
            mock_redis_instance.ping.return_value = True
            
            limiter = DistributedRateLimiter()
            
            # Should have method to properly close connections
            self.assertTrue(hasattr(limiter, 'close') or hasattr(limiter, '__del__'))
    
    def test_configurable_pool_size(self):
        """Test that connection pool size is configurable."""
        with patch('finchat_sec_qa.rate_limiting.get_config') as mock_config, \
             patch('redis.ConnectionPool.from_url') as mock_pool_from_url, \
             patch('redis.Redis') as mock_redis:
            
            # Mock configuration with custom pool size
            mock_config.return_value.RATE_LIMIT_MAX_REQUESTS = 10
            mock_config.return_value.RATE_LIMIT_WINDOW_SECONDS = 60
            mock_config.return_value.REDIS_POOL_MAX_CONNECTIONS = 20
            
            mock_pool = Mock()
            mock_pool_from_url.return_value = mock_pool
            mock_redis_instance = Mock()
            mock_redis.return_value = mock_redis_instance
            mock_redis_instance.ping.return_value = True
            
            limiter = DistributedRateLimiter()
            
            args, kwargs = mock_pool_from_url.call_args
            self.assertEqual(kwargs['max_connections'], 20)
    
    def test_pool_health_monitoring(self):
        """Test connection pool health can be monitored."""
        with patch('redis.ConnectionPool.from_url') as mock_pool_from_url, \
             patch('redis.Redis') as mock_redis:
            
            mock_pool = Mock()
            mock_pool.created_connections = 5
            mock_pool.available_connections = 3
            mock_pool_from_url.return_value = mock_pool
            
            mock_redis_instance = Mock()
            mock_redis.return_value = mock_redis_instance
            mock_redis_instance.ping.return_value = True
            
            limiter = DistributedRateLimiter()
            
            # Should be able to get pool statistics
            pool_stats = limiter.get_redis_pool_stats()
            self.assertIn('created_connections', pool_stats)
            self.assertIn('available_connections', pool_stats)


if __name__ == '__main__':
    unittest.main()