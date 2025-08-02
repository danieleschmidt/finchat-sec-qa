# Observability Guide for FinChat-SEC-QA

This guide provides comprehensive information about monitoring, logging, alerting, and observability practices for FinChat-SEC-QA.

## Overview

Our observability strategy follows the three pillars of observability:
- **Metrics**: Quantitative measurements of system behavior
- **Logs**: Detailed records of system events
- **Traces**: Request flow through distributed components

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Application   │───▶│   Prometheus    │───▶│    Grafana      │
│                 │    │   (Metrics)     │    │ (Visualization) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │
         ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Structured     │───▶│      ELK        │───▶│   Kibana        │
│    Logging      │    │ (Log Analysis)  │    │  (Log Search)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │
         ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│    Alerting     │───▶│  Alertmanager   │───▶│ Notification    │
│     Rules       │    │   (Routing)     │    │   Channels      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Metrics

### Application Metrics

#### Business Metrics
- **Query Volume**: `finchat_queries_total`
- **Query Latency**: `finchat_query_duration_seconds`
- **Cache Hit Rate**: `finchat_cache_hits_total / finchat_cache_requests_total`
- **EDGAR API Calls**: `finchat_edgar_requests_total`

#### Technical Metrics
- **HTTP Request Rate**: `http_requests_total`
- **HTTP Request Duration**: `http_request_duration_seconds`
- **Error Rate**: `http_requests_total{code=~"5.."}`
- **Authentication Failures**: `finchat_auth_failures_total`

#### System Metrics
- **CPU Usage**: `system_cpu_usage_percent`
- **Memory Usage**: `system_memory_usage_bytes`
- **Disk Usage**: `node_filesystem_usage_bytes`
- **Network I/O**: `node_network_receive_bytes_total`

### Custom Metrics Implementation

```python
from prometheus_client import Counter, Histogram, Gauge

# Business metrics
QUERY_COUNTER = Counter('finchat_queries_total', 'Total queries processed', ['status'])
QUERY_DURATION = Histogram('finchat_query_duration_seconds', 'Query processing time')
CACHE_HITS = Counter('finchat_cache_hits_total', 'Cache hits')
CACHE_MISSES = Counter('finchat_cache_misses_total', 'Cache misses')

# Usage in application
def process_query(query):
    start_time = time.time()
    try:
        result = execute_query(query)
        QUERY_COUNTER.labels(status='success').inc()
        return result
    except Exception as e:
        QUERY_COUNTER.labels(status='error').inc()
        raise
    finally:
        QUERY_DURATION.observe(time.time() - start_time)
```

## Logging

### Structured Logging

All logs use structured JSON format for better parsing and analysis:

```json
{
  "timestamp": "2024-01-15T10:30:00.000Z",
  "level": "INFO",
  "logger": "finchat_sec_qa.qa_engine",
  "message": "Query processed successfully",
  "correlation_id": "req-123456",
  "user_id": "user-789",
  "query_type": "risk_analysis",
  "processing_time": 2.34,
  "cache_hit": true
}
```

### Log Levels

- **DEBUG**: Detailed diagnostic information
- **INFO**: General operational information
- **WARNING**: Something unexpected happened
- **ERROR**: Error occurred but application continues
- **CRITICAL**: Serious error, application may not continue

### Log Categories

#### Application Logs
- Query processing
- Cache operations
- Authentication events
- Configuration changes

#### Security Logs
- Authentication failures
- Authorization attempts
- Suspicious activity
- Rate limit violations

#### Performance Logs
- Slow queries (>5 seconds)
- High memory usage
- CPU spikes
- External API timeouts

### Log Configuration

```python
import logging
import json
from datetime import datetime

class StructuredFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add correlation ID if available
        if hasattr(record, 'correlation_id'):
            log_data['correlation_id'] = record.correlation_id
            
        return json.dumps(log_data)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)

for handler in logging.root.handlers:
    handler.setFormatter(StructuredFormatter())
```

## Health Checks

### Endpoint Types

#### Liveness Probe (`/health/live`)
- Checks if application is running
- Always returns 200 if app is alive
- Used by Kubernetes for restart decisions

#### Readiness Probe (`/health/ready`)
- Checks if application is ready to serve traffic
- Validates dependencies (database, cache, etc.)
- Used by Kubernetes for traffic routing

#### Deep Health Check (`/health`)
- Comprehensive health assessment
- Includes external service checks
- Used for detailed diagnostics

### Health Check Response

```json
{
  "service": "finchat-sec-qa",
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "version": "1.4.9",
  "uptime_seconds": 86400,
  "checks": {
    "database": {
      "status": "healthy",
      "response_time_ms": 12.34
    },
    "cache": {
      "status": "healthy",
      "file_count": 42,
      "total_size_mb": 128.5
    },
    "external_services": {
      "sec_edgar": {
        "status": "healthy",
        "response_time_ms": 234.5
      }
    }
  },
  "metrics": {
    "process_memory_mb": 256.7,
    "process_cpu_percent": 15.2,
    "open_files": 23,
    "threads": 8
  }
}
```

## Alerting

### Alert Severity Levels

- **Critical**: Immediate action required (page on-call)
- **Warning**: Action required within hours
- **Info**: Informational, no immediate action needed

### Alert Categories

#### Service Availability
- Service down
- High error rate (>5%)
- High response time (p95 >10s)

#### Resource Utilization
- High CPU usage (>80%)
- High memory usage (>85%)
- Low disk space (>85% used)

#### Business Metrics
- Unusual query volume
- High cache miss rate
- EDGAR API issues

#### Security
- High authentication failures
- Suspicious activity detected
- Rate limit violations

### Alert Configuration

Alerts are defined in `docs/monitoring/alert_rules.yml` and include:
- Expression for triggering condition
- Duration before firing
- Severity and routing labels
- Human-readable descriptions
- Runbook links for response procedures

## Dashboards

### Grafana Dashboards

#### Application Overview
- Request rate and latency
- Error rates
- Cache performance
- Query volume trends

#### System Resources
- CPU and memory usage
- Disk space and I/O
- Network traffic
- Container metrics

#### Business Metrics
- User activity patterns
- Popular query types
- Performance benchmarks
- Cost analysis

### Key Performance Indicators (KPIs)

#### Availability Metrics
- **Uptime**: Target 99.9%
- **Error Rate**: Target <1%
- **Response Time**: p95 <500ms

#### Performance Metrics
- **Query Processing Time**: p95 <5s
- **Cache Hit Rate**: Target >80%
- **Throughput**: Queries per second

#### Business Metrics
- **User Satisfaction**: Response quality
- **Cost Efficiency**: API usage optimization
- **Data Freshness**: EDGAR update frequency

## Distributed Tracing

### Trace Implementation

```python
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

jaeger_exporter = JaegerExporter(
    agent_host_name="localhost",
    agent_port=6831,
)

span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Usage in application
@tracer.start_as_current_span("process_query")
def process_query(query):
    span = trace.get_current_span()
    span.set_attribute("query.type", query.type)
    span.set_attribute("query.length", len(query.text))
    
    with tracer.start_as_current_span("fetch_documents"):
        documents = fetch_documents(query)
    
    with tracer.start_as_current_span("generate_response"):
        response = generate_response(query, documents)
    
    return response
```

### Trace Analysis

- **Request Flow**: Track requests across services
- **Performance Bottlenecks**: Identify slow components
- **Error Attribution**: Pinpoint failure sources
- **Dependency Mapping**: Understand service interactions

## Monitoring Best Practices

### Metrics Design
1. **Use consistent naming**: Follow Prometheus conventions
2. **Include relevant labels**: For filtering and grouping
3. **Avoid high cardinality**: Prevent metric explosion
4. **Monitor business KPIs**: Not just technical metrics

### Alert Design
1. **Alert on symptoms**: What users experience
2. **Include context**: Enough info to understand impact
3. **Provide runbooks**: Clear response procedures
4. **Test alerting**: Regularly verify alert paths

### Dashboard Design
1. **Start with overview**: High-level system health
2. **Enable drill-down**: From summary to details
3. **Use consistent time ranges**: For comparison
4. **Include SLA tracking**: Business objectives

## Runbooks

### Service Down Response
1. Check service status in monitoring
2. Review recent deployments
3. Check resource utilization
4. Examine application logs
5. Escalate if needed

### Performance Degradation
1. Identify affected components
2. Check resource constraints
3. Review recent changes
4. Analyze query patterns
5. Implement temporary mitigations

### Security Incident Response
1. Assess threat severity
2. Contain potential breach
3. Preserve evidence
4. Notify stakeholders
5. Document and learn

## Tool Configuration

### Prometheus Configuration

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'finchat-api'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'finchat-webapp'
    static_configs:
      - targets: ['localhost:5000']
    metrics_path: '/metrics'
    scrape_interval: 10s

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### Grafana Data Sources

```json
{
  "name": "Prometheus",
  "type": "prometheus",
  "url": "http://prometheus:9090",
  "access": "proxy",
  "isDefault": true
}
```

## Performance Monitoring

### Application Performance Monitoring (APM)
- Request tracing
- Database query analysis
- External API monitoring
- Code-level insights

### Infrastructure Monitoring
- Container metrics
- Host system metrics
- Network performance
- Storage I/O

### Real User Monitoring (RUM)
- Page load times
- User interactions
- Error rates
- Performance perception

## Cost Monitoring

### Cloud Resource Costs
- Compute instance usage
- Storage costs
- Network transfer
- External API costs

### Optimization Opportunities
- Right-sizing instances
- Cache effectiveness
- Query optimization
- Resource scheduling

## Compliance and Audit

### Audit Logging
- All user actions
- Configuration changes
- Access patterns
- Data modifications

### Compliance Reporting
- Uptime reports
- Security incident logs
- Performance SLA tracking
- Data retention policies

## References

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [OpenTelemetry](https://opentelemetry.io/)
- [SRE Workbook](https://sre.google/workbook/table-of-contents/)