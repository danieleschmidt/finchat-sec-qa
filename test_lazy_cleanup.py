#!/usr/bin/env python3
"""Test lazy cache cleanup optimization functionality."""

import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_lazy_cleanup_properties():
    """Test that lazy cleanup properties exist."""
    try:
        from finchat_sec_qa.utils import TimeBoundedCache
        
        cache = TimeBoundedCache(max_size=100, ttl_seconds=60)
        
        # Check properties exist
        assert hasattr(cache, '_last_cleanup_time')
        assert hasattr(cache, '_cleanup_interval')
        assert hasattr(cache, '_should_cleanup')
        assert hasattr(cache, '_perform_lazy_cleanup')
        
        print("✅ Lazy cleanup properties exist")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def test_cleanup_interval_logic():
    """Test cleanup interval calculation."""
    try:
        from finchat_sec_qa.utils import TimeBoundedCache
        
        # Test different TTL values
        cache1 = TimeBoundedCache(max_size=100, ttl_seconds=600)  # 10 minutes
        assert cache1._cleanup_interval == 60  # Should be 60 (minimum)
        
        cache2 = TimeBoundedCache(max_size=100, ttl_seconds=3600)  # 1 hour
        assert cache2._cleanup_interval == 360  # Should be 360 (10% of 3600)
        
        print("✅ Cleanup interval calculation works correctly")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def test_should_cleanup_logic():
    """Test the _should_cleanup method."""
    try:
        from finchat_sec_qa.utils import TimeBoundedCache
        
        cache = TimeBoundedCache(max_size=100, ttl_seconds=60)
        
        # Initially should need cleanup
        assert cache._should_cleanup() == True
        
        # After setting last cleanup time to now, should not need cleanup
        cache._last_cleanup_time = time.time()
        assert cache._should_cleanup() == False
        
        # After setting last cleanup time to past, should need cleanup
        cache._last_cleanup_time = time.time() - cache._cleanup_interval - 1
        assert cache._should_cleanup() == True
        
        print("✅ _should_cleanup logic works correctly")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def test_basic_cache_functionality_preserved():
    """Test that basic cache functionality still works."""
    try:
        from finchat_sec_qa.utils import TimeBoundedCache
        
        cache = TimeBoundedCache(max_size=100, ttl_seconds=60)
        
        # Test basic operations
        cache.set('key1', 'value1')  
        assert cache.get('key1') == 'value1'
        
        cache.set('key2', 'value2')
        assert cache.get('key2') == 'value2'
        
        # Test non-existent key
        assert cache.get('nonexistent') is None
        assert cache.get('nonexistent', 'default') == 'default'
        
        print("✅ Basic cache functionality preserved")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def test_performance_improvement_indication():
    """Test indication that performance should be improved."""
    try:
        from finchat_sec_qa.utils import TimeBoundedCache
        
        cache = TimeBoundedCache(max_size=100, ttl_seconds=3600)  # 1 hour TTL
        
        # Add items
        for i in range(20):
            cache.set(f'key{i}', f'value{i}')
        
        # Access items multiple times - should be fast since no expiration check on every access
        start_time = time.perf_counter()
        
        for _ in range(10):  # Multiple rounds of access
            for i in range(20):
                cache.get(f'key{i}')
        
        access_time = time.perf_counter() - start_time
        
        # Should be very fast (no specific threshold, just ensuring it doesn't hang)
        assert access_time < 1.0  # Should complete in under 1 second
        
        print(f"✅ Performance test completed in {access_time:.4f}s (should be fast)")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

if __name__ == "__main__":
    print("Testing lazy cache cleanup optimization...")
    
    test1 = test_lazy_cleanup_properties()
    test2 = test_cleanup_interval_logic()
    test3 = test_should_cleanup_logic()
    test4 = test_basic_cache_functionality_preserved()
    test5 = test_performance_improvement_indication()
    
    if test1 and test2 and test3 and test4 and test5:
        print("\n🎉 All lazy cleanup optimization tests passed! Performance should be improved.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed.")
        sys.exit(1)