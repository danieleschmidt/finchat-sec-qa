"""Utility classes and functions for FinChat SEC QA."""
from typing import Any, Optional, Generic, TypeVar, Dict
from collections import OrderedDict

K = TypeVar('K')
V = TypeVar('V')


class BoundedCache(Generic[K, V]):
    """
    A bounded LRU cache that prevents memory leaks by evicting least recently used items.
    
    Thread-safe for single-threaded use. For multi-threaded use, wrap with locks.
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize bounded cache.
        
        Args:
            max_size: Maximum number of items to store before eviction
        """
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        
        self.max_size = max_size
        self._cache: OrderedDict[K, V] = OrderedDict()
    
    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """
        Get value by key, marking it as recently used.
        
        Args:
            key: Key to look up
            default: Default value if key not found
            
        Returns:
            Value if found, default otherwise
        """
        if key in self._cache:
            # Move to end (mark as recently used)
            self._cache.move_to_end(key)
            return self._cache[key]
        return default
    
    def set(self, key: K, value: V) -> None:
        """
        Set key-value pair, evicting LRU items if necessary.
        
        Args:
            key: Key to set
            value: Value to store
        """
        if key in self._cache:
            # Update existing key
            self._cache[key] = value
            self._cache.move_to_end(key)
        else:
            # Add new key
            self._cache[key] = value
            # Evict LRU item if over capacity
            if len(self._cache) > self.max_size:
                self._cache.popitem(last=False)  # Remove least recently used
    
    def delete(self, key: K) -> bool:
        """
        Delete key from cache.
        
        Args:
            key: Key to delete
            
        Returns:
            True if key existed and was deleted, False otherwise
        """
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all items from cache."""
        self._cache.clear()
    
    def __len__(self) -> int:
        """Return number of items in cache."""
        return len(self._cache)
    
    def __contains__(self, key: K) -> bool:
        """Check if key exists in cache without updating LRU order."""
        return key in self._cache
    
    def keys(self):
        """Return cache keys in LRU order (oldest first)."""
        return self._cache.keys()
    
    def values(self):
        """Return cache values in LRU order (oldest first)."""
        return self._cache.values()
    
    def items(self):
        """Return cache items in LRU order (oldest first)."""
        return self._cache.items()


class TimeBoundedCache(BoundedCache[K, V]):
    """
    A bounded cache that also supports time-based expiration.
    
    Combines LRU eviction with TTL expiration for memory-safe caching
    with automatic cleanup of stale entries.
    """
    
    def __init__(self, max_size: int = 1000, ttl_seconds: float = 3600):
        """
        Initialize time-bounded cache.
        
        Args:
            max_size: Maximum number of items before LRU eviction
            ttl_seconds: Time-to-live for cache entries in seconds
        """
        super().__init__(max_size)
        self.ttl_seconds = ttl_seconds
        self._timestamps: OrderedDict[K, float] = OrderedDict()
    
    def set(self, key: K, value: V) -> None:
        """Set key-value with current timestamp."""
        import time
        
        super().set(key, value)
        self._timestamps[key] = time.time()
        
        # Clean up timestamps for evicted items
        cache_keys = set(self._cache.keys())
        timestamp_keys = set(self._timestamps.keys())
        for expired_key in timestamp_keys - cache_keys:
            del self._timestamps[expired_key]
    
    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """Get value, checking expiration first."""
        import time
        
        if key in self._cache:
            # Check if expired
            if key in self._timestamps:
                age = time.time() - self._timestamps[key]
                if age > self.ttl_seconds:
                    # Expired - remove from cache
                    self.delete(key)
                    return default
            
            # Not expired - update LRU and return
            self._timestamps.move_to_end(key)
            return super().get(key, default)
        
        return default
    
    def delete(self, key: K) -> bool:
        """Delete key and its timestamp."""
        result = super().delete(key)
        if key in self._timestamps:
            del self._timestamps[key]
        return result
    
    def clear(self) -> None:
        """Clear cache and timestamps."""
        super().clear()
        self._timestamps.clear()
    
    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.
        
        Returns:
            Number of entries removed
        """
        import time
        
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self._timestamps.items()
            if current_time - timestamp > self.ttl_seconds
        ]
        
        for key in expired_keys:
            self.delete(key)
        
        return len(expired_keys)