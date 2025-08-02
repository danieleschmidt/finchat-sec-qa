# Monitoring Setup Guide

This guide provides step-by-step instructions for setting up comprehensive monitoring for FinChat-SEC-QA.

## Prerequisites

- Docker and Docker Compose installed
- Access to the application source code
- Network access for external monitoring services

## Quick Start

### 1. Start Monitoring Stack

```bash
# Start the complete monitoring stack
docker-compose --profile monitoring up -d

# Or start individual services
docker-compose up -d prometheus grafana alertmanager
```

### 2. Configure Prometheus

Prometheus configuration is automatically loaded from `docker/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "/etc/prometheus/alert_rules.yml"

scrape_configs:
  - job_name: 'finchat-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'finchat-webapp'
    static_configs:
      - targets: ['webapp:5000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### 3. Import Grafana Dashboard

1. Access Grafana at http://localhost:3000
2. Login with admin/admin (change password on first login)
3. Go to "+" → Import
4. Upload `docs/monitoring/grafana-dashboard.json`
5. Configure Prometheus data source: http://prometheus:9090

### 4. Configure Alertmanager

Create `alertmanager.yml` configuration:

```yaml
global:
  smtp_smarthost: 'smtp.example.com:587'
  smtp_from: 'alerts@example.com'

route:
  group_by: ['alertname', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'default'
  routes:
  - match:
      severity: critical
    receiver: 'critical-alerts'
  - match:
      severity: warning
    receiver: 'warning-alerts'

receivers:
- name: 'default'
  slack_configs:
  - api_url: 'YOUR_SLACK_WEBHOOK_URL'
    channel: '#monitoring'
    title: 'FinChat Alert'
    text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'

- name: 'critical-alerts'
  pagerduty_configs:
  - routing_key: 'YOUR_PAGERDUTY_KEY'
    description: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
  slack_configs:
  - api_url: 'YOUR_SLACK_WEBHOOK_URL'
    channel: '#critical-alerts'
    title: '🚨 CRITICAL: FinChat Alert'
    text: '{{ range .Alerts }}{{ .Annotations.summary }}\n{{ .Annotations.description }}{{ end }}'

- name: 'warning-alerts'
  slack_configs:
  - api_url: 'YOUR_SLACK_WEBHOOK_URL'
    channel: '#alerts'
    title: '⚠️ WARNING: FinChat Alert'
    text: '{{ range .Alerts }}{{ .Annotations.summary }}\n{{ .Annotations.description }}{{ end }}'

inhibit_rules:
- source_match:
    severity: 'critical'
  target_match:
    severity: 'warning'
  equal: ['alertname', 'service']
```

## Detailed Setup

### Prometheus Configuration

#### Service Discovery

For dynamic environments, configure service discovery:

```yaml
scrape_configs:
  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
    - role: pod
    relabel_configs:
    - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
      action: keep
      regex: true
    - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
      action: replace
      target_label: __metrics_path__
      regex: (.+)
```

#### Recording Rules

Add performance recording rules:

```yaml
groups:
  - name: finchat-performance
    interval: 30s
    rules:
      - record: finchat:request_rate
        expr: sum(rate(http_requests_total{job="finchat-api"}[5m]))
      
      - record: finchat:error_rate
        expr: sum(rate(http_requests_total{job="finchat-api",code=~"5.."}[5m])) / sum(rate(http_requests_total{job="finchat-api"}[5m]))
      
      - record: finchat:response_time_p95
        expr: histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{job="finchat-api"}[5m])) by (le))
```

### Grafana Setup

#### Data Source Configuration

```json
{
  "name": "Prometheus",
  "type": "prometheus",
  "url": "http://prometheus:9090",
  "access": "proxy",
  "isDefault": true,
  "jsonData": {
    "timeInterval": "15s",
    "httpMethod": "POST"
  }
}
```

#### Dashboard Variables

Create template variables for dynamic dashboards:

```json
{
  "name": "environment",
  "type": "query",
  "query": "label_values(environment)",
  "refresh": 1,
  "includeAll": false
}
```

#### Alert Rules in Grafana

Configure Grafana alerts for real-time notifications:

```json
{
  "alert": {
    "name": "High Error Rate",
    "message": "Error rate is above 5%",
    "frequency": "10s",
    "conditions": [
      {
        "query": {
          "refId": "A",
          "queryType": "",
          "model": {
            "expr": "finchat:error_rate * 100 > 5"
          }
        },
        "reducer": {
          "type": "last"
        },
        "evaluator": {
          "params": [5],
          "type": "gt"
        }
      }
    ]
  }
}
```

### Log Aggregation Setup

#### ELK Stack Configuration

**Elasticsearch:**
```yaml
version: '3.8'
services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.5.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
    ports:
      - "9200:9200"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data
```

**Logstash:**
```ruby
input {
  beats {
    port => 5044
  }
}

filter {
  if [fields][service] == "finchat" {
    json {
      source => "message"
    }
    
    date {
      match => [ "timestamp", "ISO8601" ]
    }
    
    mutate {
      add_field => { "service" => "finchat-sec-qa" }
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "finchat-logs-%{+YYYY.MM.dd}"
  }
}
```

**Filebeat:**
```yaml
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /var/log/finchat/*.log
  fields:
    service: finchat
  json.keys_under_root: true

output.logstash:
  hosts: ["logstash:5044"]

logging.level: info
```

#### Kibana Dashboards

Create log analysis dashboards:

1. **Error Analysis Dashboard**
   - Error count over time
   - Top error messages
   - Error rate by endpoint

2. **Performance Dashboard**
   - Request duration distribution
   - Slow query analysis
   - Cache performance metrics

3. **Security Dashboard**
   - Authentication failures
   - Suspicious IP addresses
   - Rate limit violations

### Distributed Tracing Setup

#### Jaeger Configuration

```yaml
version: '3.8'
services:
  jaeger:
    image: jaegertracing/all-in-one:latest
    environment:
      - COLLECTOR_ZIPKIN_HTTP_PORT=9411
    ports:
      - "16686:16686"
      - "14268:14268"
      - "9411:9411"
```

#### Application Instrumentation

```python
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor

# Initialize tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Configure Jaeger exporter
jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger",
    agent_port=6831,
)

span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Auto-instrument FastAPI and requests
FastAPIInstrumentor.instrument_app(app)
RequestsInstrumentor().instrument()
```

## Production Deployment

### High Availability Setup

#### Prometheus HA

```yaml
# prometheus-1.yml
global:
  external_labels:
    replica: 1

# prometheus-2.yml
global:
  external_labels:
    replica: 2
```

#### Grafana HA with Load Balancer

```yaml
version: '3.8'
services:
  grafana-1:
    image: grafana/grafana:latest
    environment:
      - GF_DATABASE_TYPE=postgres
      - GF_DATABASE_HOST=postgres:5432
      - GF_DATABASE_NAME=grafana
  
  grafana-2:
    image: grafana/grafana:latest
    environment:
      - GF_DATABASE_TYPE=postgres
      - GF_DATABASE_HOST=postgres:5432
      - GF_DATABASE_NAME=grafana
  
  nginx:
    image: nginx:alpine
    ports:
      - "3000:80"
    depends_on:
      - grafana-1
      - grafana-2
```

### Security Configuration

#### TLS Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

tls_config:
  cert_file: /etc/prometheus/certs/prometheus.crt
  key_file: /etc/prometheus/certs/prometheus.key
  ca_file: /etc/prometheus/certs/ca.crt

scrape_configs:
  - job_name: 'finchat-api'
    scheme: https
    tls_config:
      cert_file: /etc/prometheus/certs/client.crt
      key_file: /etc/prometheus/certs/client.key
      ca_file: /etc/prometheus/certs/ca.crt
```

#### Authentication

```yaml
# grafana.ini
[auth.basic]
enabled = true

[auth.ldap]
enabled = true
config_file = /etc/grafana/ldap.toml

[security]
admin_user = admin
admin_password = $__file{/etc/grafana/admin_password}
```

### Backup and Disaster Recovery

#### Prometheus Data Backup

```bash
#!/bin/bash
# backup-prometheus.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backup/prometheus"
PROMETHEUS_DATA="/prometheus-data"

# Create snapshot
curl -XPOST http://localhost:9090/api/v1/admin/tsdb/snapshot

# Find latest snapshot
SNAPSHOT=$(ls -t ${PROMETHEUS_DATA}/snapshots | head -1)

# Compress and backup
tar -czf ${BACKUP_DIR}/prometheus_${DATE}.tar.gz \
    -C ${PROMETHEUS_DATA}/snapshots ${SNAPSHOT}

# Cleanup old snapshots
find ${PROMETHEUS_DATA}/snapshots -type d -mtime +7 -exec rm -rf {} \;
```

#### Grafana Configuration Backup

```bash
#!/bin/bash
# backup-grafana.sh

GRAFANA_URL="http://localhost:3000"
API_KEY="your-api-key"
BACKUP_DIR="/backup/grafana"
DATE=$(date +%Y%m%d_%H%M%S)

# Export dashboards
curl -H "Authorization: Bearer ${API_KEY}" \
     "${GRAFANA_URL}/api/search?type=dash-db" | \
     jq -r '.[].uri' | \
     while read uri; do
       dashboard_uid=$(echo $uri | cut -d'/' -f2)
       curl -H "Authorization: Bearer ${API_KEY}" \
            "${GRAFANA_URL}/api/dashboards/uid/${dashboard_uid}" \
            > "${BACKUP_DIR}/dashboard_${dashboard_uid}_${DATE}.json"
     done
```

## Troubleshooting

### Common Issues

#### Prometheus Not Scraping Targets

1. Check network connectivity:
   ```bash
   docker exec prometheus wget -qO- http://api:8000/metrics
   ```

2. Verify service discovery:
   ```bash
   curl http://localhost:9090/api/v1/targets
   ```

3. Check configuration:
   ```bash
   docker exec prometheus promtool check config /etc/prometheus/prometheus.yml
   ```

#### Grafana Dashboard Not Loading

1. Verify data source connectivity
2. Check query syntax in Prometheus
3. Examine Grafana logs:
   ```bash
   docker logs grafana
   ```

#### High Memory Usage

1. Reduce retention period:
   ```yaml
   # prometheus.yml
   global:
     storage.tsdb.retention.time: 15d
   ```

2. Increase memory limits:
   ```yaml
   # docker-compose.yml
   prometheus:
     deploy:
       resources:
         limits:
           memory: 2G
   ```

### Performance Tuning

#### Prometheus Optimization

```yaml
global:
  scrape_interval: 30s  # Reduce frequency
  evaluation_interval: 30s

storage:
  tsdb:
    retention.time: 30d
    retention.size: 10GB
    min-block-duration: 2h
    max-block-duration: 25h
```

#### Grafana Optimization

```ini
[database]
type = postgres
host = postgres:5432
name = grafana
user = grafana
password = password

[server]
protocol = http
http_port = 3000
domain = localhost
enforce_domain = false
root_url = http://localhost:3000/

[security]
admin_user = admin
admin_password = admin
secret_key = SW2YcwTIb9zpOOhoPsMm

[analytics]
reporting_enabled = false
check_for_updates = false

[log]
mode = console
level = info
```

## Maintenance

### Regular Tasks

1. **Weekly:**
   - Review alert configurations
   - Check disk space usage
   - Analyze slow queries

2. **Monthly:**
   - Update monitoring stack versions
   - Review and archive old data
   - Audit user access

3. **Quarterly:**
   - Performance optimization review
   - Security assessment
   - Disaster recovery testing

### Monitoring the Monitoring

Set up meta-monitoring to ensure monitoring stack health:

```yaml
# Meta-monitoring alerts
groups:
  - name: meta-monitoring
    rules:
      - alert: PrometheusDown
        expr: up{job="prometheus"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Prometheus server is down"
      
      - alert: GrafanaDown
        expr: up{job="grafana"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Grafana server is down"
```

## References

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Alertmanager Documentation](https://prometheus.io/docs/alerting/latest/alertmanager/)
- [Best Practices for Monitoring](https://prometheus.io/docs/practices/monitoring/)