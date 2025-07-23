# Distributed Rate Limiting

This document describes the distributed rate limiting implementation that replaces the previous in-memory solution.

## Overview

The distributed rate limiting system provides:
- **Redis-based storage** for rate limiting across multiple instances
- **Automatic fallback** to in-memory storage when Redis is unavailable
- **Thread-safe atomic operations** using Lua scripts
- **Backward compatibility** with existing rate limiting interface

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Instance A    │    │   Instance B    │    │   Instance C    │
│                 │    │                 │    │                 │
│ RateLimiter     │    │ RateLimiter     │    │ RateLimiter     │
│      │          │    │      │          │    │      │          │
└──────┼──────────┘    └──────┼──────────┘    └──────┼──────────┘
       │                      │                      │
       └──────────────────────┼──────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │                   │
                    │   Redis Server    │
                    │                   │
                    │ rate_limit:client │
                    │   [timestamps]    │
                    └───────────────────┘
```

## Configuration

### Environment Variables

```bash
# Redis connection URL (optional, defaults to localhost)
FINCHAT_REDIS_URL=redis://localhost:6379/0

# Rate limiting configuration (existing)
FINCHAT_RATE_LIMIT_MAX_REQUESTS=100
FINCHAT_RATE_LIMIT_WINDOW_SECONDS=3600
```

### Redis Setup

For production deployments:

```bash
# Install Redis
apt-get install redis-server

# Basic Redis configuration
redis-cli config set maxmemory 256mb
redis-cli config set maxmemory-policy allkeys-lru

# For high availability, consider Redis Cluster or Sentinel
```

## Implementation Details

### Data Structure

Rate limiting data is stored in Redis as **sorted sets** with the following schema:

```
Key: rate_limit:{client_id}
Type: Sorted Set (ZSET)
Members: timestamp values
Scores: same timestamp values (for efficient range queries)
```

### Lua Script

The rate limiting uses an atomic Lua script to ensure thread safety:

```lua
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
    redis.call('EXPIRE', key, tonumber(ARGV[4]))
    return 1
else
    return 0
end
```

## Usage

### Basic Usage (Backward Compatible)

```python
from finchat_sec_qa.webapp import RateLimiter

# Create rate limiter (automatically uses Redis if available)
limiter = RateLimiter(max_requests=100, window_seconds=3600)

# Check if client is allowed
if limiter.is_allowed('client_123'):
    # Process request
    pass
else:
    # Rate limited
    return 429
```

### Advanced Usage

```python
from finchat_sec_qa.rate_limiting import DistributedRateLimiter

# Custom Redis configuration
limiter = DistributedRateLimiter(
    max_requests=50,
    window_seconds=60,
    redis_url='redis://redis-cluster:6379/1'
)

# Check remaining requests
remaining = limiter.get_remaining_requests('client_123')

# Reset client (admin operation)
limiter.reset_client('client_123')
```

## Fallback Behavior

When Redis is unavailable:
1. **Initialization**: If Redis connection fails during startup, falls back to in-memory
2. **Runtime**: If Redis operations fail during runtime, switches to in-memory for that client
3. **Logging**: All fallback events are logged with warning level

## Performance Characteristics

### Redis Mode
- **Latency**: ~1-2ms per rate limit check
- **Memory**: O(R) where R = max_requests per client
- **Scalability**: Supports unlimited instances
- **Persistence**: Survives application restarts

### Fallback Mode  
- **Latency**: ~0.1ms per rate limit check
- **Memory**: O(C × R) where C = active clients, R = max_requests
- **Scalability**: Single instance only
- **Persistence**: Lost on application restart

## Monitoring

### Metrics

The rate limiter provides these metrics:
- Redis connection status
- Fallback events
- Rate limit violations
- Request counts per client

### Logging

```
INFO: Connected to Redis for distributed rate limiting
WARN: Failed to connect to Redis, falling back to in-memory: Connection refused  
WARN: Redis rate limiting failed, falling back to memory: Connection timeout
```

## Security Considerations

1. **Redis Security**: Use Redis AUTH and network isolation
2. **Client ID Validation**: Ensure client IDs are sanitized
3. **Resource Limits**: Set appropriate Redis memory limits
4. **DOS Protection**: The rate limiter itself provides DOS protection

## Migration from In-Memory

The migration is **zero-downtime** and **backward compatible**:

1. **Before**: `RateLimiter` used in-memory storage
2. **After**: `RateLimiter` uses Redis with in-memory fallback
3. **API**: No changes to existing code required

## Testing

### Unit Tests

```bash
# Run distributed rate limiting tests
pytest tests/test_distributed_rate_limiting.py -v

# Run integration tests (requires Redis)
REDIS_AVAILABLE=1 pytest tests/test_distributed_rate_limiting.py::TestDistributedRateLimiterIntegration -v
```

### Load Testing 

```bash
# Test rate limiting under load
python scripts/load_test_rate_limiting.py
```

## Deployment

### Docker Compose

```yaml
version: '3.8'
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    
  finchat-api:
    image: finchat-sec-qa
    environment:
      - FINCHAT_REDIS_URL=redis://redis:6379/0
    depends_on:
      - redis

volumes:
  redis_data:
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: finchat-api
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: api
        image: finchat-sec-qa
        env:
        - name: FINCHAT_REDIS_URL
          value: "redis://redis-service:6379/0"
```

## Troubleshooting

### Common Issues

1. **Redis Connection Failed**
   - Check Redis server status
   - Verify network connectivity
   - Check Redis AUTH configuration

2. **High Memory Usage in Redis**
   - Set appropriate `maxmemory` policy
   - Tune window_seconds to reduce data retention
   - Monitor client patterns

3. **Rate Limiting Not Working**
   - Check Redis key expiration
   - Verify client ID consistency
   - Monitor fallback mode activation

### Debugging

```python
import logging
logging.getLogger('finchat_sec_qa.rate_limiting').setLevel(logging.DEBUG)
```