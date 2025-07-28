# Operational Procedures Runbook

## Overview

This runbook covers standard operational procedures for maintaining and operating the FinChat-SEC-QA system.

## Daily Operations

### Morning Health Check
```bash
#!/bin/bash
# Daily health check script

echo "=== FinChat-SEC-QA Daily Health Check ==="
echo "Date: $(date)"
echo

# Check service status
echo "1. Service Health:"
curl -sf http://localhost:8000/health || echo "❌ API service unhealthy"
curl -sf http://localhost:5000/ || echo "❌ WebApp service unhealthy"
echo "✅ Services checked"
echo

# Check metrics endpoint
echo "2. Metrics Collection:"
curl -sf http://localhost:8000/metrics > /dev/null && echo "✅ Metrics available" || echo "❌ Metrics unavailable"
echo

# Check disk space
echo "3. System Resources:"
df -h | grep -E '(Filesystem|/dev/)' | awk '{print $5 " " $6}' | while read output; do
  if [[ ${output%% *} > "90%" ]]; then
    echo "❌ High disk usage: $output"
  fi
done
echo "✅ Disk space checked"
echo

# Check log errors
echo "4. Recent Errors:"
error_count=$(docker logs finchat-api --since 24h 2>&1 | grep -c ERROR || echo 0)
if [ "$error_count" -gt 10 ]; then
  echo "⚠️  High error count: $error_count errors in last 24h"
else
  echo "✅ Error count acceptable: $error_count errors"
fi
echo

echo "=== Health Check Complete ==="
```

### Log Rotation and Cleanup
```bash
#!/bin/bash
# Log cleanup script

# Rotate application logs
docker exec finchat-api logrotate /etc/logrotate.d/app

# Clean old log files (keep 30 days)
find /var/log/finchat -name "*.log" -mtime +30 -delete

# Clean old docker logs
docker system prune -f --filter "until=72h"

# Clean cache files (keep 7 days)
find ~/.cache/finchat_sec_qa -name "*.joblib" -mtime +7 -delete
```

## Weekly Operations

### Performance Review
1. **Review metrics dashboard**
   - Check response time trends
   - Monitor error rates
   - Analyze resource utilization

2. **Capacity planning**
   - Review growth trends
   - Check resource limits
   - Plan scaling if needed

3. **Security review**
   - Check for security alerts
   - Review access logs
   - Update security configs if needed

### Dependency Updates
```bash
#!/bin/bash
# Weekly dependency check

# Check for Python package updates
pip list --outdated

# Check for security vulnerabilities
safety check

# Check Docker base image updates
docker pull python:3.11-slim

# Run dependency audit
pip-audit
```

## Monthly Operations

### Backup Procedures
```bash
#!/bin/bash
# Monthly backup script

BACKUP_DATE=$(date +%Y%m%d)
BACKUP_DIR="/backups/finchat-$BACKUP_DATE"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Backup configuration
cp -r /app/config "$BACKUP_DIR/"

# Backup cached data
tar -czf "$BACKUP_DIR/cache.tar.gz" ~/.cache/finchat_sec_qa/

# Backup logs
tar -czf "$BACKUP_DIR/logs.tar.gz" /var/log/finchat/

# Backup database if applicable
if [ -f "/app/data/finchat.db" ]; then
  cp /app/data/finchat.db "$BACKUP_DIR/"
fi

# Upload to backup storage
# aws s3 sync "$BACKUP_DIR" s3://finchat-backups/

echo "Backup completed: $BACKUP_DIR"
```

### Security Audit
1. **Access review**
   - Review user access logs
   - Check for suspicious activity
   - Audit API key usage

2. **Vulnerability scanning**
   ```bash
   # Run security scans
   bandit -r src/
   safety check
   docker scout cves finchat-sec-qa:latest
   ```

3. **Compliance check**
   - Review SEC API usage compliance
   - Check data retention policies
   - Verify logging compliance

## Deployment Procedures

### Pre-deployment Checklist
- [ ] All tests passing
- [ ] Security scans clean
- [ ] Performance tests acceptable
- [ ] Documentation updated
- [ ] Rollback plan prepared
- [ ] Stakeholders notified

### Deployment Steps
```bash
#!/bin/bash
# Deployment script

set -euo pipefail

VERSION=${1:-latest}
ENVIRONMENT=${2:-production}

echo "Deploying FinChat-SEC-QA version $VERSION to $ENVIRONMENT"

# Pre-deployment health check
curl -sf http://localhost:8000/health || { echo "Current service unhealthy"; exit 1; }

# Pull new images
docker pull finchat-sec-qa-api:$VERSION
docker pull finchat-sec-qa-webapp:$VERSION

# Rolling update with health checks
docker-compose up -d --no-deps api
sleep 30
curl -sf http://localhost:8000/health || { echo "New API service unhealthy"; exit 1; }

docker-compose up -d --no-deps webapp
sleep 30
curl -sf http://localhost:5000/ || { echo "New WebApp service unhealthy"; exit 1; }

echo "Deployment successful"
```

### Post-deployment Verification
```bash
#!/bin/bash
# Post-deployment verification

# Health checks
curl -sf http://localhost:8000/health
curl -sf http://localhost:5000/

# Functional tests
pytest tests/integration/ -v

# Performance smoke test
cd tests/performance/k6
k6 run smoke-test.js

# Check metrics
curl -sf http://localhost:8000/metrics | grep -q "finchat_"

echo "Post-deployment verification complete"
```

### Rollback Procedures
```bash
#!/bin/bash
# Emergency rollback

PREVIOUS_VERSION=${1:?"Previous version required"}

echo "Rolling back to version $PREVIOUS_VERSION"

# Stop current services
docker-compose down

# Deploy previous version
DOCKER_TAG=$PREVIOUS_VERSION docker-compose up -d

# Wait and verify
sleep 60
curl -sf http://localhost:8000/health || { echo "Rollback failed"; exit 1; }

echo "Rollback to $PREVIOUS_VERSION successful"
```

## Maintenance Windows

### Scheduled Maintenance
- **Frequency**: Monthly, first Saturday 2-4 AM EST
- **Duration**: 2 hours maximum
- **Notification**: 48 hours advance notice

### Maintenance Tasks
1. **System updates**
   - OS security patches
   - Docker base image updates
   - Dependency updates

2. **Performance optimization**
   - Database maintenance
   - Cache optimization
   - Log cleanup

3. **Security updates**
   - Certificate renewal
   - Security patch application
   - Access review

## Monitoring and Alerting

### Critical Alerts (Immediate Response)
- Service down (5-minute detection)
- High error rate > 5% (2-minute duration)
- Response time > 10s (5-minute duration)
- Security incidents

### Warning Alerts (1-hour Response)
- High CPU/Memory usage > 80%
- Low cache hit rate < 70%
- External API latency > 30s
- Disk space > 85%

### Information Alerts (Next Business Day)
- Daily summary reports
- Performance trends
- Capacity planning alerts
- Non-critical security events

## Disaster Recovery

### Data Recovery
1. **Cache reconstruction**
   ```bash
   # Rebuild cache from source
   python scripts/rebuild_cache.py --source edgar --days 30
   ```

2. **Configuration restoration**
   ```bash
   # Restore from backup
   cp /backups/latest/config/* /app/config/
   docker-compose restart
   ```

### Service Recovery
1. **Container recovery**
   ```bash
   # Restart all services
   docker-compose down
   docker-compose up -d
   ```

2. **Database recovery**
   ```bash
   # Restore database if applicable
   cp /backups/latest/finchat.db /app/data/
   ```

## Troubleshooting Common Issues

### Issue: Service Won't Start
**Symptoms:** Container exits immediately
**Investigation:**
```bash
# Check container logs
docker logs finchat-api

# Check configuration
docker exec finchat-api cat /app/config/app.conf

# Check file permissions
docker exec finchat-api ls -la /app/
```

### Issue: High Memory Usage
**Symptoms:** OOM kills, slow performance
**Investigation:**
```bash
# Monitor memory usage
docker stats

# Check for memory leaks
python scripts/memory_profile.py

# Review cache size
redis-cli info memory
```

### Issue: SEC API Rate Limiting
**Symptoms:** 429 errors, slow data retrieval
**Investigation:**
```bash
# Check rate limit status
grep "rate.limit" /var/log/finchat/app.log

# Verify user agent
grep "user.agent" /var/log/finchat/app.log

# Check request frequency
grep "edgar.api" /var/log/finchat/app.log | tail -100
```

## Contact Information

### On-Call Rotation
- **Primary**: [On-call engineer contact]
- **Secondary**: [Backup engineer contact]
- **Escalation**: [Team lead contact]

### Team Contacts
- **Platform Team**: platform-team@company.com
- **Security Team**: security@company.com
- **Infrastructure**: infrastructure@company.com

### External Contacts
- **Cloud Provider**: [Support contact]
- **Monitoring Service**: [Support contact]
- **Security Vendor**: [Support contact]

---

**Document Version**: 1.0  
**Last Updated**: 2025-01-27  
**Next Review**: 2025-04-27  
**Owner**: Platform Operations Team