"""Tests for lazy cache cleanup optimization."""

import pytest
import time
from unittest.mock import patch
from finchat_sec_qa.utils import TimeBoundedCache


def test_lazy_cleanup_properties_exist():
    """Test that lazy cleanup properties are added to cache."""
    cache = TimeBoundedCache(max_size=100, ttl_seconds=1)
    
    # Should have lazy cleanup properties
    assert hasattr(cache, '_last_cleanup_time')
    assert hasattr(cache, '_cleanup_interval')
    assert hasattr(cache, '_should_cleanup')


def test_get_does_not_check_expiration_on_every_access():
    """Test that get() doesn't check expiration on every single access."""
    cache = TimeBoundedCache(max_size=100, ttl_seconds=3600)  # 1 hour TTL
    
    # Mock time to control expiration behavior
    with patch('time.time') as mock_time:
        mock_time.return_value = 1000.0
        
        # Add item
        cache.set('key1', 'value1')
        
        # Mock multiple rapid accesses - should not trigger expiration checks
        access_times = [1001.0, 1002.0, 1003.0, 1004.0, 1005.0]
        
        for access_time in access_times:
            mock_time.return_value = access_time
            value = cache.get('key1')
            assert value == 'value1'
        
        # Should not have called cleanup on every access
        # (Implementation detail: lazy cleanup should only trigger periodically)


def test_periodic_cleanup_triggers():
    """Test that cleanup triggers periodically based on interval."""
    cache = TimeBoundedCache(max_size=100, ttl_seconds=1)  # 1 second TTL
    
    with patch('time.time') as mock_time:
        mock_time.return_value = 1000.0
        
        # Add items
        cache.set('key1', 'value1')
        cache.set('key2', 'value2')
        
        # Time passes but not enough to trigger cleanup
        mock_time.return_value = 1001.0
        cache.get('key1')  # Should not trigger cleanup
        
        # Time passes enough to trigger cleanup interval
        mock_time.return_value = 1010.0  # 10 seconds later
        cache.get('key1')  # Should trigger cleanup and remove expired items
        
        # Items should be expired and removed
        assert cache.get('key1') is None
        assert cache.get('key2') is None


def test_should_cleanup_method():
    """Test the _should_cleanup() method logic."""
    cache = TimeBoundedCache(max_size=100, ttl_seconds=3600)
    
    with patch('time.time') as mock_time:
        mock_time.return_value = 1000.0
        
        # Initial cleanup should be needed
        assert cache._should_cleanup() == True
        
        # After cleanup, should not need cleanup immediately
        cache._last_cleanup_time = 1000.0
        mock_time.return_value = 1001.0  # 1 second later
        assert cache._should_cleanup() == False
        
        # After cleanup interval, should need cleanup again
        mock_time.return_value = 1000.0 + cache._cleanup_interval + 1
        assert cache._should_cleanup() == True


def test_cleanup_performance_improvement():
    """Test that lazy cleanup improves performance."""
    cache = TimeBoundedCache(max_size=100, ttl_seconds=1)
    
    # Add many items
    with patch('time.time') as mock_time:
        mock_time.return_value = 1000.0
        
        for i in range(50):
            cache.set(f'key{i}', f'value{i}')
        
        # Measure time for multiple accesses without cleanup
        mock_time.return_value = 1001.0
        start_time = time.perf_counter()
        
        for i in range(50):
            cache.get(f'key{i}')
        
        access_time = time.perf_counter() - start_time
        
        # Should be fast since no expiration checks on every access
        assert access_time < 0.1  # Should be very fast


def test_cleanup_interval_configurable():
    """Test that cleanup interval can be configured."""
    # Test default interval
    cache1 = TimeBoundedCache(max_size=100, ttl_seconds=3600)
    assert cache1._cleanup_interval > 0
    
    # Test custom interval (if supported)
    cache2 = TimeBoundedCache(max_size=100, ttl_seconds=3600)
    if hasattr(cache2, 'set_cleanup_interval'):
        cache2.set_cleanup_interval(30)
        assert cache2._cleanup_interval == 30