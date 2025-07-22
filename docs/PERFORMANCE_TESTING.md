# Performance Testing and Benchmarking

This document describes the performance testing tools and methodologies for the FinChat SEC QA service.

## Overview

The performance testing suite includes:
- **Load Testing**: Measure current API performance under various load conditions
- **Benchmarking**: Track performance improvements over time
- **Regression Testing**: Ensure performance doesn't degrade with new changes

## Quick Start

### Installation

Install performance testing dependencies:

```bash
pip install -e .[performance]
```

### Running Load Tests

Basic load test against a running server:

```bash
# Test localhost (default)
python scripts/load_test.py

# Test specific URL
python scripts/load_test.py http://your-api-server.com:8000
```

### Running Benchmarks

Track performance over time:

```bash
# Run benchmark for current version
python scripts/benchmark.py 1.4.3 "http://localhost:8000" "Added Prometheus metrics"

# Set baseline for comparison
python scripts/benchmark.py set-baseline 1.4.3

# View latest report
python scripts/benchmark.py report
```

## Performance Metrics

### HTTP Metrics
- **Requests per Second (RPS)**: Total throughput
- **Response Time**: Average, median, 95th, 99th percentiles
- **Error Rate**: Percentage of failed requests
- **Concurrency**: Number of simultaneous connections

### Business Metrics
- **Query Processing Time**: Time to answer financial questions
- **Risk Analysis Speed**: Performance of sentiment analysis
- **Health Check Latency**: Service availability response time

## Expected Performance Targets

Based on the async implementation and metrics collection:

### Health Endpoint (`/health`)
- **Target RPS**: > 1000 requests/second
- **Target Latency**: < 10ms average
- **Max P95**: < 50ms

### Metrics Endpoint (`/metrics`)
- **Target RPS**: > 500 requests/second  
- **Target Latency**: < 20ms average
- **Max P95**: < 100ms

### Risk Analysis (`/risk`)
- **Target RPS**: > 100 requests/second
- **Target Latency**: < 100ms average
- **Max P95**: < 500ms

### Query Endpoint (`/query`)
- **Target RPS**: > 10 requests/second (limited by external API calls)
- **Target Latency**: < 2000ms average (depends on SEC API)
- **Max P95**: < 10000ms

## Load Testing Scenarios

### Scenario 1: Health Check Load
```bash
# Simulate monitoring systems
# 1000 requests, 50 concurrent connections
Health Endpoint: High frequency, low latency
```

### Scenario 2: Metrics Scraping
```bash  
# Simulate Prometheus scraping
# 200 requests, 10 concurrent connections
Metrics Endpoint: Regular intervals, consistent performance
```

### Scenario 3: Risk Analysis Burst
```bash
# Simulate batch processing
# 500 requests, 25 concurrent connections  
Risk Endpoint: Sustained load, CPU intensive
```

### Scenario 4: Query Processing
```bash
# Simulate user queries
# 20 requests, 4 concurrent connections (respectful of SEC API limits)
Query Endpoint: Real-world usage patterns
```

## Performance Optimizations Implemented

### 1. Async I/O Implementation (WSJF: 4.40)
- **Before**: Synchronous requests blocking threads
- **After**: httpx/asyncio for concurrent processing
- **Expected Improvement**: 2-5x throughput increase

### 2. Metrics Collection (WSJF: 6.00)
- **Added**: Prometheus metrics with minimal overhead
- **Monitoring**: Request timing, error rates, service health
- **Expected Impact**: < 1% performance overhead

### 3. Bulk Operations (WSJF: 4.50)
- **Before**: Document-by-document index updates
- **After**: Batch processing with context manager
- **Expected Improvement**: 3-10x faster for multi-document operations

## Benchmarking Methodology

### Test Environment
- **Consistent Hardware**: Same machine/container for comparisons
- **Controlled Load**: Fixed request patterns for repeatability  
- **Multiple Runs**: Average of 3-5 test runs
- **Baseline Comparison**: Track improvements vs. established baseline

### Metrics Collection
- **Response Time Distribution**: Not just averages
- **Error Categorization**: Different failure modes tracked separately
- **Resource Usage**: CPU, memory, connection pool metrics
- **Business Logic Performance**: Domain-specific timing

### Reporting
- **Trend Analysis**: Performance over multiple versions
- **Regression Detection**: Automatic alerts for performance degradation
- **Optimization ROI**: Cost/benefit analysis of improvements

## CI/CD Integration

### Pre-deployment Checks
```bash
# Run performance regression tests
python scripts/benchmark.py ${BUILD_VERSION} ${STAGING_URL}

# Compare to baseline (exit non-zero if regression > 20%)  
# This would be implemented in CI pipeline
```

### Production Monitoring
- Prometheus metrics scraped every 15 seconds
- Grafana dashboards for real-time performance
- Alerting on SLA violations (p95 > thresholds)

## Troubleshooting Performance Issues

### High Latency
1. Check async operation effectiveness  
2. Verify database/external API response times
3. Review CPU usage during load
4. Examine connection pool exhaustion

### Low Throughput
1. Validate async endpoint implementation
2. Check for blocking operations in async context
3. Review concurrent connection limits
4. Examine resource contention

### Memory Usage
1. Monitor for memory leaks in long-running tests
2. Check object pool reuse (database connections, HTTP sessions)
3. Review metrics collection overhead
4. Validate cleanup in async contexts

## Advanced Testing

### Chaos Engineering
- Network latency injection
- External service failures
- Resource exhaustion simulation
- Recovery time measurement

### Stress Testing
- Progressive load increase to find breaking points
- Sustained load for endurance testing
- Spike testing for traffic bursts
- Resource limit testing

### Performance Profiling
- cProfile integration for function-level timing
- Memory profiling with tracemalloc
- Async operation bottleneck identification
- Database query performance analysis

## Contributing Performance Tests

When adding new endpoints or features:

1. **Add Load Tests**: Include endpoint in load_test.py
2. **Update Benchmarks**: Add business metrics for new functionality  
3. **Set Performance Targets**: Define acceptable performance thresholds
4. **Document Changes**: Update this guide with new testing scenarios

### Example: Adding New Endpoint Test

```python
async def test_new_endpoint(self, requests: int = 100, concurrency: int = 10):
    """Load test the new endpoint."""
    payload = {"param": "value"}
    
    start_time = time.time()
    results = await self.run_concurrent_requests(
        "POST", "/new-endpoint", requests, concurrency,
        json=payload, headers={"Content-Type": "application/json"}
    )
    duration = time.time() - start_time
    return self.analyze_results("/new-endpoint", results, duration)
```

---

**Performance is a feature** - these tools ensure the FinChat SEC QA service delivers consistent, predictable performance as it scales.