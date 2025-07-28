# Performance Monitoring Runbook

## Overview

This runbook covers performance monitoring, alerting, and optimization procedures for FinChat-SEC-QA.

## Key Performance Indicators (KPIs)

### Response Time Metrics
- **API Response Time**: P95 < 2.5s, P99 < 5s
- **Query Processing**: Average < 3s for standard queries
- **EDGAR API Calls**: P95 < 10s (external dependency)
- **Cache Hit Rate**: > 80% for frequently accessed data

### Throughput Metrics
- **Queries per Second**: Target 50 QPS sustained
- **Concurrent Users**: Support 100+ concurrent users
- **API Requests**: Handle 500+ requests per minute

### Resource Utilization
- **CPU Usage**: Average < 70%, Peak < 90%
- **Memory Usage**: Average < 80%, No memory leaks
- **Disk I/O**: Monitor for bottlenecks in file operations
- **Network**: Monitor SEC API rate limits

## Monitoring Setup

### Prometheus Metrics

```yaml
# Key metrics to monitor
metrics:
  - name: finchat_query_duration_seconds
    type: histogram
    description: Time spent processing queries
    
  - name: finchat_cache_hit_rate
    type: gauge
    description: Cache hit rate percentage
    
  - name: finchat_edgar_api_calls_total
    type: counter
    description: Total SEC EDGAR API calls
    
  - name: finchat_active_sessions
    type: gauge
    description: Number of active user sessions
```

### Grafana Dashboards

#### System Overview Dashboard
- Response time trends
- Error rate monitoring
- Resource utilization
- Cache performance

#### Application Performance Dashboard
- Query processing times
- EDGAR API performance
- User session metrics
- Database performance

## Alert Rules

### Critical Alerts (P0)
```yaml
alerts:
  - name: HighErrorRate
    condition: error_rate > 5%
    duration: 2m
    action: Page on-call team
    
  - name: ServiceDown
    condition: up == 0
    duration: 1m
    action: Immediate escalation
    
  - name: HighResponseTime
    condition: p95_response_time > 10s
    duration: 5m
    action: Page on-call team
```

### Warning Alerts (P1)
```yaml
alerts:
  - name: HighCPUUsage
    condition: cpu_usage > 80%
    duration: 10m
    action: Notify team
    
  - name: LowCacheHitRate
    condition: cache_hit_rate < 70%
    duration: 15m
    action: Investigate caching
    
  - name: EDGARAPILatency
    condition: edgar_api_p95 > 30s
    duration: 5m
    action: Check external service
```

## Performance Investigation Procedures

### Step 1: Initial Assessment
```bash
# Check overall system health
curl -f http://localhost:8000/health

# Check Prometheus metrics
curl http://localhost:8000/metrics | grep finchat

# Review application logs
docker logs finchat-api --tail 100
```

### Step 2: Identify Bottlenecks

#### Database Performance
```bash
# Check query performance
python scripts/benchmark.py --component database

# Monitor slow queries
grep "slow query" logs/app.log
```

#### Cache Performance
```bash
# Check cache hit rates
curl http://localhost:8000/metrics | grep cache_hit

# Monitor cache size and evictions
redis-cli info memory
```

#### EDGAR API Performance
```bash
# Test EDGAR API directly
curl -H "User-Agent: YourApp contact@domain.com" \
     https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json

# Check rate limiting status
grep "rate.limit" logs/app.log
```

### Step 3: Performance Optimization

#### Query Optimization
1. **Analyze slow queries**
   ```python
   # Enable query profiling
   import cProfile
   cProfile.run('process_query(question)', 'query_profile.prof')
   ```

2. **Optimize vector search**
   - Review embedding dimensions
   - Tune similarity thresholds
   - Optimize chunk sizes

3. **Improve caching strategy**
   - Increase cache TTL for stable data
   - Implement intelligent prefetching
   - Use cache warming strategies

#### Resource Optimization
1. **Memory optimization**
   ```bash
   # Monitor memory usage
   docker stats finchat-api
   
   # Check for memory leaks
   python scripts/memory_profile.py
   ```

2. **CPU optimization**
   - Profile CPU-intensive operations
   - Implement async processing where possible
   - Optimize serialization/deserialization

## Load Testing Procedures

### Baseline Performance Testing
```bash
# Run smoke test
cd tests/performance/k6
k6 run smoke-test.js

# Run load test
k6 run load-test.js

# Run stress test
k6 run stress-test.js
```

### Custom Load Testing
```bash
# Test specific endpoints
k6 run --env BASE_URL=http://localhost:8000 \
       --env ENDPOINT=/qa \
       --env VUS=50 \
       --env DURATION=5m \
       custom-load-test.js
```

## Performance Tuning Checklist

### Application Level
- [ ] Optimize database queries and indexing
- [ ] Implement effective caching strategy
- [ ] Use connection pooling
- [ ] Optimize JSON serialization
- [ ] Implement request compression
- [ ] Use async/await patterns

### Infrastructure Level
- [ ] Scale horizontally with load balancing
- [ ] Optimize container resource allocation
- [ ] Implement CDN for static assets
- [ ] Use Redis for session management
- [ ] Optimize network configuration

### Monitoring Level
- [ ] Set up comprehensive dashboards
- [ ] Configure alerting thresholds
- [ ] Implement SLO monitoring
- [ ] Set up log aggregation
- [ ] Monitor business metrics

## Common Performance Issues

### Issue: Slow Query Response Times
**Symptoms:** P95 response time > 5s
**Investigation:**
1. Check database query performance
2. Review vector search efficiency
3. Analyze EDGAR API response times
4. Check cache hit rates

**Resolution:**
- Optimize database indexes
- Tune vector search parameters
- Implement query result caching
- Use async processing for heavy operations

### Issue: High Memory Usage
**Symptoms:** Memory usage > 90%, potential OOM
**Investigation:**
1. Profile memory allocation
2. Check for memory leaks
3. Review cache size and eviction policies
4. Analyze object lifecycle

**Resolution:**
- Implement object pooling
- Optimize cache eviction policies
- Fix memory leaks in long-running operations
- Increase container memory limits

### Issue: Low Cache Hit Rate
**Symptoms:** Cache hit rate < 70%
**Investigation:**
1. Analyze cache key patterns
2. Review TTL settings
3. Check cache eviction logs
4. Monitor cache size vs. hit rate

**Resolution:**
- Optimize cache key strategy
- Implement cache warming
- Adjust TTL based on data volatility
- Increase cache size if needed

## SLA and SLO Monitoring

### Service Level Objectives (SLOs)
- **Availability**: 99.5% uptime
- **Response Time**: 95% of requests < 2.5s
- **Error Rate**: < 1% error rate
- **Cache Performance**: > 80% hit rate

### Error Budget Management
- Monthly error budget: 0.5% downtime (≈ 3.5 hours)
- Weekly performance review
- Quarterly SLO review and adjustment

## Escalation Procedures

### Performance Degradation (P1)
1. **Immediate actions (0-15 minutes)**
   - Check system health endpoints
   - Review recent deployments
   - Check external dependencies

2. **Investigation (15-60 minutes)**
   - Run performance benchmarks
   - Analyze system metrics
   - Check resource utilization

3. **Resolution (1-4 hours)**
   - Apply immediate fixes
   - Scale resources if needed
   - Implement workarounds

### Contact Information
- **On-call Engineer**: [Phone/Slack]
- **Performance Team**: [Email/Slack channel]
- **Infrastructure Team**: [Contact details]

## Documentation and Reporting

### Performance Reports
- Daily performance summary
- Weekly trend analysis
- Monthly capacity planning report
- Quarterly performance review

### Post-Incident Analysis
- Root cause analysis
- Performance impact assessment
- Prevention measures
- Documentation updates

---

**Last Updated:** 2025-01-27  
**Next Review:** 2025-04-27  
**Owner:** Platform Team