"""
Distributed rate limiting module with Redis backend and in-memory fallback.
Provides thread-safe, atomic rate limiting across multiple instances.
"""
import time
import logging
from typing import Dict, List, Optional, Union, Any
from collections import defaultdict
from .config import get_config
from .utils import BoundedCache

try:
    import redis
    from redis import Redis as RedisClient
except ImportError:
    class RedisClient:  # type: ignore
        pass

logger = logging.getLogger(__name__)


class DistributedRateLimiter:
    """
    Redis-based distributed rate limiter with in-memory fallback.
    
    Uses Redis sorted sets for efficient sliding window rate limiting
    with atomic Lua script operations for thread safety.
    """
    
    # Lua script for atomic rate limiting operations
    RATE_LIMIT_SCRIPT = """
    local key = KEYS[1]
    local window_start = ARGV[1]
    local current_time = ARGV[2]
    local max_requests = tonumber(ARGV[3])
    
    -- Remove expired entries
    redis.call('ZREMRANGEBYSCORE', key, '-inf', window_start)
    
    -- Count current requests
    local current_count = redis.call('ZCARD', key)
    
    -- Check if under limit
    if current_count < max_requests then
        -- Add current request
        redis.call('ZADD', key, current_time, current_time)
        -- Set expiration for cleanup
        redis.call('EXPIRE', key, tonumber(ARGV[4]))
        return 1
    else
        return 0
    end
    """
    
    def __init__(self, max_requests: Optional[int] = None, window_seconds: Optional[int] = None, redis_url: Optional[str] = None):
        """Initialize distributed rate limiter with bounded fallback cache."""
        config = get_config()
        self.max_requests = max_requests or config.RATE_LIMIT_MAX_REQUESTS
        self.window_seconds = window_seconds or config.RATE_LIMIT_WINDOW_SECONDS
        self.redis_url = redis_url or getattr(config, 'REDIS_URL', 'redis://localhost:6379/0')
        self.max_fallback_cache_size = config.RATE_LIMIT_MAX_FALLBACK_CACHE_SIZE
        self.pool_max_connections = config.REDIS_POOL_MAX_CONNECTIONS
        
        # Initialize Redis client with connection pool and bounded fallback
        self.redis_client: Optional[RedisClient] = None
        self.redis_pool: Optional[Any] = None
        self.fallback_storage = BoundedCache[str, List[float]](max_size=self.max_fallback_cache_size)
        
        self._init_redis()
    
    def _init_redis(self) -> None:
        """Initialize Redis connection pool with error handling."""
        try:
            import redis
            
            # Create connection pool for better performance
            self.redis_pool = redis.ConnectionPool.from_url(
                self.redis_url,
                max_connections=self.pool_max_connections,
                decode_responses=True,
                socket_connect_timeout=1,
                socket_timeout=1
            )
            
            # Create Redis client using the connection pool
            client = redis.Redis(connection_pool=self.redis_pool)
            
            # Test connection
            client.ping()
            self.redis_client = client
            logger.info("Connected to Redis with connection pool for distributed rate limiting (pool_size=%d)", self.pool_max_connections)
        except Exception as e:
            logger.warning(f"Failed to connect to Redis, falling back to in-memory: {e}")
            self.redis_client = None
            self.redis_pool = None
    
    def is_allowed(self, client_id: str) -> bool:
        """
        Check if client is within rate limits.
        
        Args:
            client_id: Unique identifier for the client
            
        Returns:
            True if request is allowed, False if rate limited
        """
        if self.redis_client:
            return self._redis_is_allowed(client_id)
        else:
            return self._memory_is_allowed(client_id)
    
    def _redis_is_allowed(self, client_id: str) -> bool:
        """Redis-based rate limiting with atomic Lua script."""
        if self.redis_client is None:
            return self._memory_is_allowed(client_id)
            
        try:
            key = f"rate_limit:{client_id}"
            current_time = time.time()
            window_start = current_time - self.window_seconds
            
            # Execute atomic Lua script
            result = self.redis_client.eval(
                self.RATE_LIMIT_SCRIPT,
                1,  # Number of keys
                key,
                str(window_start),
                str(current_time),
                str(self.max_requests),
                str(self.window_seconds * 2)  # Expiration buffer
            )
            
            return bool(result)
            
        except Exception as e:
            logger.warning(f"Redis rate limiting failed, falling back to memory: {e}")
            # Fallback to in-memory on Redis failure
            self.redis_client = None
            return self._memory_is_allowed(client_id)
    
    def _memory_is_allowed(self, client_id: str) -> bool:
        """In-memory fallback rate limiting with bounded cache."""
        now = time.time()
        
        # Get current requests for client (or empty list)
        client_requests = self.fallback_storage.get(client_id, [])
        
        # Clean old requests outside the window
        client_requests = [
            req_time for req_time in client_requests
            if now - req_time < self.window_seconds
        ]
        
        # Check if under limit
        if len(client_requests) < self.max_requests:
            client_requests.append(now)
            self.fallback_storage.set(client_id, client_requests)
            return True
        else:
            # Update cache with cleaned requests even if over limit
            self.fallback_storage.set(client_id, client_requests)
            return False
    
    def reset_client(self, client_id: str) -> None:
        """Reset rate limit for a specific client (for testing/admin)."""
        if self.redis_client:
            try:
                key = f"rate_limit:{client_id}"
                self.redis_client.delete(key)
            except Exception as e:
                logger.warning(f"Failed to reset Redis rate limit for {client_id}: {e}")
        
        # Also reset in-memory storage
        self.fallback_storage.delete(client_id)
    
    def get_remaining_requests(self, client_id: str) -> int:
        """Get number of remaining requests for client."""
        if self.redis_client:
            try:
                key = f"rate_limit:{client_id}"
                current_time = time.time()
                window_start = current_time - self.window_seconds
                
                # Clean expired entries and count
                self.redis_client.zremrangebyscore(key, '-inf', window_start)
                current_count = int(self.redis_client.zcard(key) or 0)  # type: ignore
                
                return max(0, self.max_requests - current_count)
            except Exception:
                pass
        
        # Fallback to memory count
        now = time.time()
        client_requests = self.fallback_storage.get(client_id, [])
        current_requests = [
            req_time for req_time in client_requests
            if now - req_time < self.window_seconds
        ]
        return max(0, self.max_requests - len(current_requests))
    
    def get_redis_pool_stats(self) -> Dict[str, Any]:
        """Get Redis connection pool statistics for monitoring."""
        if not self.redis_pool:
            return {"error": "Redis pool not initialized"}
        
        try:
            return {
                "created_connections": getattr(self.redis_pool, "created_connections", 0),
                "available_connections": len(getattr(self.redis_pool, "_available_connections", [])),
                "in_use_connections": len(getattr(self.redis_pool, "_in_use_connections", [])),
                "max_connections": self.redis_pool.max_connections,
                "pool_enabled": True
            }
        except Exception as e:
            return {"error": f"Failed to get pool stats: {e}"}


# Backward compatibility: Update existing webapp to use distributed limiter
class RateLimiter(DistributedRateLimiter):
    """Legacy class name for backward compatibility."""
    pass