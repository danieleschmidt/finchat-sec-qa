# K6 Performance Testing Suite

This directory contains K6 performance tests for the FinChat SEC Q&A application.

## Test Types

### 1. Smoke Test (`smoke-test.js`)
- **Purpose**: Verify basic functionality with minimal load
- **Load**: 1 virtual user for 1 minute
- **Use Case**: Quick verification that the system is working
- **Run**: `k6 run smoke-test.js`

### 2. Load Test (`load-test.js`)
- **Purpose**: Test normal expected load conditions
- **Load**: Gradually increases from 10 to 100 users over 8 minutes
- **Use Case**: Verify performance under normal operating conditions
- **Run**: `k6 run load-test.js`

### 3. Stress Test (`stress-test.js`)
- **Purpose**: Push system beyond normal capacity to find breaking points
- **Load**: Up to 1000 concurrent users with intensive queries
- **Use Case**: Understand system limits and failure modes
- **Run**: `k6 run stress-test.js`

## Environment Variables

Set these environment variables before running tests:

```bash
export BASE_URL=http://localhost:8000
export API_TOKEN=your-test-token
```

## Prerequisites

1. Install K6:
   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install k6
   
   # macOS
   brew install k6
   
   # Docker
   docker run --rm -i grafana/k6:latest run - <script.js
   ```

2. Ensure the application is running:
   ```bash
   # Start the FastAPI server
   python -m finchat_sec_qa.server
   
   # Or using Docker
   docker-compose up
   ```

## Running Tests

### Individual Tests
```bash
# Smoke test
k6 run smoke-test.js

# Load test with custom parameters
BASE_URL=http://localhost:8000 k6 run load-test.js

# Stress test with results output
k6 run --out json=results.json stress-test.js
```

### Test Suite
```bash
# Run all tests in sequence
./run-performance-tests.sh
```

## Test Results and Metrics

K6 provides detailed metrics including:

- **http_req_duration**: Response time statistics
- **http_req_failed**: Failure rate
- **http_reqs**: Total number of requests
- **vus**: Number of virtual users
- **Custom metrics**: Error rates, response times

### Key Performance Indicators (KPIs)

- **Response Time**: p95 < 500ms for normal load, < 2s for stress
- **Error Rate**: < 5% for normal load, < 10% for stress
- **Throughput**: Target RPS based on system capacity
- **Availability**: System remains responsive under load

## Integration with CI/CD

These tests can be integrated into your CI/CD pipeline:

```yaml
# Example GitHub Actions step
- name: Run Performance Tests
  run: |
    docker-compose up -d
    sleep 30  # Wait for services to start
    k6 run tests/performance/k6/smoke-test.js
    k6 run tests/performance/k6/load-test.js
```

## Monitoring and Alerting

For production monitoring, consider:

1. **Prometheus Integration**: Export K6 metrics to Prometheus
2. **Grafana Dashboards**: Visualize performance trends
3. **Alert Rules**: Set up alerts for performance degradation

```bash
# Export to Prometheus
k6 run --out experimental-prometheus-rw load-test.js
```

## Best Practices

1. **Baseline Testing**: Establish performance baselines before changes
2. **Regular Testing**: Run performance tests on every major release
3. **Environment Parity**: Test in production-like environments
4. **Gradual Load**: Start with smoke tests before heavy load testing
5. **Recovery Testing**: Verify system recovery after stress tests

## Troubleshooting

### Common Issues

1. **Connection Refused**: Ensure the application is running and accessible
2. **Timeout Errors**: Adjust timeout values in test configuration
3. **High Error Rates**: Check application logs for underlying issues
4. **Resource Limits**: Monitor system resources during tests

### Debug Mode

Run tests with debug output:
```bash
k6 run --http-debug smoke-test.js
```

## Contributing

When adding new performance tests:

1. Follow the existing naming convention
2. Include appropriate thresholds and checks
3. Document the test purpose and expected load
4. Update this README with new test information