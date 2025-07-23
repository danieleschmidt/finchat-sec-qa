"""Tests for bounded cache implementations to prevent memory leaks."""
import pytest
import time
from unittest.mock import patch
from finchat_sec_qa.webapp import CSRFProtection
from finchat_sec_qa.rate_limiting import DistributedRateLimiter


class TestBoundedCSRFCache:
    """Test CSRF token cache memory bounds."""
    
    def test_csrf_cache_respects_size_limit(self):
        """Test that CSRF cache evicts old tokens when size limit exceeded."""
        csrf = CSRFProtection()
        
        # Should have a configurable maximum cache size
        assert hasattr(csrf, 'max_cache_size')
        assert csrf.max_cache_size > 0
        
        # Generate tokens up to the limit
        tokens = []
        for i in range(csrf.max_cache_size):
            token = csrf.generate_token()
            tokens.append(token)
        
        # All tokens should be valid
        for token in tokens:
            assert csrf.validate_token(token) is True
        
        # Generate one more token (should trigger eviction)
        extra_token = csrf.generate_token()
        
        # Cache should not exceed size limit
        assert len(csrf.tokens) <= csrf.max_cache_size
        
        # First token should be evicted (LRU policy)
        assert csrf.validate_token(tokens[0]) is False
        assert csrf.validate_token(extra_token) is True
    
    def test_csrf_cache_lru_eviction_order(self):
        """Test that CSRF cache evicts least recently used tokens."""
        csrf = CSRFProtection()
        
        # Generate tokens to fill cache
        tokens = []
        for i in range(csrf.max_cache_size):
            token = csrf.generate_token()
            tokens.append(token)
        
        # Access first token to make it recently used
        csrf.validate_token(tokens[0])
        
        # Generate new token (should evict second token, not first)
        new_token = csrf.generate_token()
        
        # First token should still be valid (was recently accessed)
        assert csrf.validate_token(tokens[0]) is True
        # Second token should be evicted
        assert csrf.validate_token(tokens[1]) is False
        # New token should be valid
        assert csrf.validate_token(new_token) is True
    
    def test_csrf_cache_size_configurable(self):
        """Test that CSRF cache size is configurable via environment."""
        with patch('finchat_sec_qa.webapp.get_config') as mock_config:
            mock_config.return_value.CSRF_TOKEN_EXPIRY_SECONDS = 1800
            mock_config.return_value.CSRF_MAX_CACHE_SIZE = 100
            
            csrf = CSRFProtection()
            assert csrf.max_cache_size == 100


class TestBoundedRateLimitCache:
    """Test rate limiter fallback cache memory bounds."""
    
    def test_rate_limiter_fallback_cache_bounded(self):
        """Test that rate limiter fallback storage respects size limits."""
        # Force fallback mode (no Redis)
        limiter = DistributedRateLimiter()
        limiter.redis_client = None  # Force fallback
        
        # Should have bounded cache
        assert hasattr(limiter, 'max_fallback_cache_size')
        assert limiter.max_fallback_cache_size > 0
        
        # Generate requests from many clients
        clients = [f"client_{i}" for i in range(limiter.max_fallback_cache_size + 10)]
        
        for client_id in clients:
            limiter.is_allowed(client_id)
        
        # Cache should not exceed size limit
        assert len(limiter.fallback_storage) <= limiter.max_fallback_cache_size
    
    def test_rate_limiter_lru_eviction(self):
        """Test that rate limiter evicts least recently used clients."""
        limiter = DistributedRateLimiter()
        limiter.redis_client = None  # Force fallback
        
        # Fill cache to capacity
        clients = [f"client_{i}" for i in range(limiter.max_fallback_cache_size)]
        for client_id in clients:
            limiter.is_allowed(client_id)
        
        # Access first client to make it recently used
        limiter.is_allowed(clients[0])
        
        # Add new client (should evict second client)
        new_client = "new_client"
        limiter.is_allowed(new_client)
        
        # First client should still be in cache (recently accessed)
        assert clients[0] in limiter.fallback_storage
        # Second client should be evicted
        assert clients[1] not in limiter.fallback_storage
        # New client should be in cache
        assert new_client in limiter.fallback_storage
    
    def test_rate_limiter_cache_size_configurable(self):
        """Test that rate limiter cache size is configurable."""
        with patch('finchat_sec_qa.rate_limiting.get_config') as mock_config:
            mock_config.return_value.RATE_LIMIT_MAX_REQUESTS = 10
            mock_config.return_value.RATE_LIMIT_WINDOW_SECONDS = 60
            mock_config.return_value.RATE_LIMIT_MAX_FALLBACK_CACHE_SIZE = 500
            
            limiter = DistributedRateLimiter()
            assert limiter.max_fallback_cache_size == 500


class TestBoundedCacheUtility:
    """Test the bounded cache utility class."""
    
    def test_bounded_cache_creation(self):
        """Test that BoundedCache can be created and configured."""
        from finchat_sec_qa.utils import BoundedCache
        
        cache = BoundedCache(max_size=100)
        assert cache.max_size == 100
        assert len(cache) == 0
    
    def test_bounded_cache_lru_behavior(self):
        """Test LRU eviction behavior of BoundedCache."""
        from finchat_sec_qa.utils import BoundedCache
        
        cache = BoundedCache(max_size=3)
        
        # Add items
        cache.set('a', 1)
        cache.set('b', 2)
        cache.set('c', 3)
        
        # Access 'a' to make it recently used
        cache.get('a')
        
        # Add 'd' (should evict 'b', the least recently used)
        cache.set('d', 4)
        
        assert cache.get('a') == 1  # Still there
        assert cache.get('b') is None  # Evicted
        assert cache.get('c') == 3  # Still there
        assert cache.get('d') == 4  # Newly added
    
    def test_bounded_cache_memory_efficiency(self):
        """Test that cache doesn't grow beyond bounds under stress."""
        from finchat_sec_qa.utils import BoundedCache
        
        cache = BoundedCache(max_size=100)
        
        # Add many items
        for i in range(1000):
            cache.set(f'key_{i}', i)
        
        # Should never exceed max size
        assert len(cache) <= 100
        
        # Should contain most recent items
        assert cache.get('key_999') == 999
        assert cache.get('key_0') is None  # Should be evicted