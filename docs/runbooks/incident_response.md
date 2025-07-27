# FinChat-SEC-QA Incident Response Runbook

## Overview

This runbook provides step-by-step procedures for responding to incidents in the FinChat-SEC-QA system. Use this guide to ensure consistent, effective incident response.

## Incident Classification

### Severity Levels

| Level | Description | Response Time | Examples |
|-------|-------------|---------------|----------|
| **Critical (P0)** | System down, data breach, security incident | 15 minutes | Complete service outage, data loss, security breach |
| **High (P1)** | Major feature broken, performance degraded | 1 hour | API errors, slow responses, authentication issues |
| **Medium (P2)** | Minor feature issues, workarounds available | 4 hours | Specific query failures, UI bugs |
| **Low (P3)** | Cosmetic issues, enhancement requests | 24 hours | Documentation errors, minor UI inconsistencies |

## Incident Response Process

### 1. Detection & Initial Response (0-15 minutes)

#### Immediate Actions
1. **Acknowledge the incident**
   - Respond to alerts within 5 minutes
   - Update incident status in monitoring system
   - Notify on-call team via Slack/PagerDuty

2. **Initial triage**
   ```bash
   # Check system health
   curl -f http://localhost:8000/health
   curl -f http://localhost:5000/health
   
   # Check metrics
   curl -s http://localhost:8000/metrics | grep -E "(error|failure)"
   
   # Check logs
   docker logs finchat-api --tail 100
   docker logs finchat-webapp --tail 100
   ```

3. **Establish incident channel**
   - Create Slack channel: `#incident-YYYY-MM-DD-HHXX`
   - Invite relevant team members
   - Post initial status update

#### Initial Assessment Template
```
🚨 INCIDENT ALERT 🚨

**Incident ID:** INC-YYYY-MM-DD-XXX
**Severity:** [P0/P1/P2/P3]
**Status:** Investigating
**Started:** [timestamp]
**Reporter:** [name]

**Summary:** Brief description of the issue

**Impact:** 
- Services affected: [list]
- Users affected: [estimate]
- Financial impact: [if applicable]

**Current Status:** [what we know so far]

**Actions Taken:**
- [ ] Acknowledged alert
- [ ] Initial triage completed
- [ ] Team notified

**Next Steps:**
- [ ] Detailed investigation
- [ ] Implement mitigation
- [ ] Monitor and verify fix
```

### 2. Investigation & Diagnosis (15 minutes - 1 hour)

#### System Health Checks
```bash
# FastAPI service health
curl -v http://localhost:8000/health
curl -v http://localhost:8000/ready

# Flask webapp health  
curl -v http://localhost:5000/health
curl -v http://localhost:5000/ready

# Docker container status
docker ps
docker stats --no-stream

# System resources
top -b -n1 | head -20
df -h
free -h
```

#### Log Analysis
```bash
# Application logs
tail -f logs/app.log | grep -i error
journalctl -u finchat-api --since "1 hour ago"

# Web server logs
tail -f /var/log/nginx/error.log
tail -f /var/log/nginx/access.log | grep -E "(4[0-9][0-9]|5[0-9][0-9])"

# Database/cache logs
tail -f ~/.cache/finchat_sec_qa/debug.log
```

#### Performance Analysis
```bash
# API response times
curl -w "@curl-format.txt" -s -o /dev/null http://localhost:8000/api/v1/query

# Memory usage
ps aux | grep -E "(finchat|python)" | awk '{print $2, $4, $11}'

# Network connectivity
ping -c 3 api.openai.com
ping -c 3 data.sec.gov
```

### 3. Mitigation & Resolution (1-4 hours)

#### Quick Fixes

**High Memory Usage**
```bash
# Restart services
docker-compose restart api webapp

# Clear cache
rm -rf ~/.cache/finchat_sec_qa/*.tmp
```

**API Rate Limiting Issues**
```bash
# Check rate limit status
grep -i "rate.limit" logs/app.log | tail -10

# Temporary rate limit increase (if allowed)
export EDGAR_RATE_LIMIT=5  # Reduce from 10
```

**Database/Cache Corruption**
```bash
# Backup current cache
cp -r ~/.cache/finchat_sec_qa ~/.cache/finchat_sec_qa.backup.$(date +%Y%m%d_%H%M)

# Clear and reinitialize cache
rm -rf ~/.cache/finchat_sec_qa/*.joblib
python -c "from finchat_sec_qa.cache import initialize_cache; initialize_cache()"
```

**External Service Outages**
```bash
# Switch to fallback/cached responses
export ENABLE_FALLBACK_MODE=true
export CACHE_ONLY_MODE=true

# Restart with fallback configuration
docker-compose restart
```

#### Rollback Procedures

**Code Rollback**
```bash
# Rollback to previous Docker image
docker tag finchat-api:latest finchat-api:broken
docker tag finchat-api:previous finchat-api:latest
docker-compose up -d

# Rollback using git
git log --oneline -10  # Find last good commit
git revert <commit-hash>
```

**Configuration Rollback**
```bash
# Restore previous configuration
git checkout HEAD~1 -- config/
docker-compose restart
```

### 4. Monitoring & Verification (Ongoing)

#### Health Monitoring
```bash
# Continuous health check
watch -n 30 'curl -s http://localhost:8000/health | jq .status'

# Monitor key metrics
watch -n 60 'curl -s http://localhost:8000/metrics | grep -E "(response_time|error_rate|memory)"'
```

#### User Impact Assessment
```bash
# Check error rates
grep -i error logs/app.log | wc -l

# Check successful requests
grep "200" logs/access.log | wc -l

# Response time analysis
awk '/response_time/ {sum+=$NF; count++} END {print "Avg:", sum/count "ms"}' logs/app.log
```

## Common Incident Scenarios

### API Service Down

**Symptoms:**
- Health check returns 500/503
- No response from API endpoints
- High error rate in logs

**Investigation:**
```bash
# Check container status
docker ps | grep finchat-api

# Check logs for crashes
docker logs finchat-api --tail 50

# Check resource usage
docker stats finchat-api --no-stream
```

**Resolution:**
```bash
# Restart service
docker-compose restart api

# If persistent, rollback
docker tag finchat-api:previous finchat-api:latest
docker-compose up -d api
```

### High Memory Usage

**Symptoms:**
- Slow response times
- Out of memory errors
- System becoming unresponsive

**Investigation:**
```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head -10

# Check for memory leaks
valgrind --tool=memcheck python -m finchat_sec_qa.server
```

**Resolution:**
```bash
# Restart services
docker-compose restart

# Implement memory limits
echo "memory: 1g" >> docker-compose.yml

# Clear caches
rm -rf ~/.cache/finchat_sec_qa/temp*
```

### External API Failures

**Symptoms:**
- OpenAI API errors
- SEC EDGAR timeouts
- Authentication failures

**Investigation:**
```bash
# Test external connectivity
curl -I https://api.openai.com
curl -I https://data.sec.gov

# Check API key validity
curl -H "Authorization: Bearer $OPENAI_API_KEY" https://api.openai.com/v1/models
```

**Resolution:**
```bash
# Enable fallback mode
export FALLBACK_MODE=true

# Use cached responses only
export CACHE_ONLY=true

# Implement circuit breaker
export CIRCUIT_BREAKER_THRESHOLD=5
```

## Post-Incident Activities

### 1. Incident Report Template

```markdown
# Incident Report: INC-YYYY-MM-DD-XXX

## Summary
Brief description of what happened.

## Timeline
- **YYYY-MM-DD HH:MM** - Incident detected
- **YYYY-MM-DD HH:MM** - Investigation started
- **YYYY-MM-DD HH:MM** - Root cause identified
- **YYYY-MM-DD HH:MM** - Fix implemented
- **YYYY-MM-DD HH:MM** - Incident resolved

## Root Cause
Detailed explanation of what caused the incident.

## Impact
- Duration: X hours Y minutes
- Services affected: [list]
- Users affected: [number/percentage]
- Revenue impact: $[amount]

## Resolution
What was done to fix the issue.

## Lessons Learned
- What went well
- What could be improved
- Action items for prevention

## Action Items
- [ ] Item 1 - Assigned to: [name] - Due: [date]
- [ ] Item 2 - Assigned to: [name] - Due: [date]
```

### 2. Follow-up Actions

1. **Schedule post-mortem meeting** within 48 hours
2. **Update runbooks** with new learnings
3. **Implement preventive measures**
4. **Update monitoring and alerting**
5. **Communicate with stakeholders**

## Emergency Contacts

| Role | Name | Email | Phone | Slack |
|------|------|-------|-------|-------|
| On-Call Engineer | [Name] | engineer@company.com | +1-555-0123 | @engineer |
| DevOps Lead | [Name] | devops@company.com | +1-555-0124 | @devops |
| Security Team | [Name] | security@company.com | +1-555-0125 | @security |
| Product Manager | [Name] | pm@company.com | +1-555-0126 | @pm |

## Communication Templates

### Internal Update
```
📊 INCIDENT UPDATE - INC-YYYY-MM-DD-XXX

**Status:** [Investigating/Mitigating/Resolved]
**Severity:** P[0-3]
**Duration:** [time since start]

**Current Situation:** [brief update]

**Actions Taken:**
- [action 1]
- [action 2]

**Next Steps:**
- [next step 1]
- [next step 2]

**ETA for Resolution:** [estimate]
```

### Customer Communication
```
We are currently experiencing an issue with [service] that may impact [functionality]. 

Our team is actively working on a resolution. We will provide updates every [frequency] until resolved.

We apologize for any inconvenience and appreciate your patience.

Status page: [URL]
```

## Useful Commands & Scripts

### Quick Diagnostics
```bash
#!/bin/bash
# quick-diag.sh - Run quick system diagnostics

echo "=== System Health Check ==="
curl -s http://localhost:8000/health | jq .
curl -s http://localhost:5000/health | jq .

echo "=== Resource Usage ==="
free -h
df -h /

echo "=== Recent Errors ==="
tail -20 logs/app.log | grep -i error

echo "=== Service Status ==="
docker-compose ps
```

### Log Analysis
```bash
#!/bin/bash
# analyze-logs.sh - Analyze recent logs for patterns

echo "=== Error Summary ==="
grep -i error logs/app.log | awk '{print $3}' | sort | uniq -c | sort -nr

echo "=== Response Time Analysis ==="
grep "response_time" logs/app.log | awk '{sum+=$NF; count++} END {print "Average:", sum/count "ms"}'

echo "=== Top Error Messages ==="
grep -i error logs/app.log | cut -d' ' -f4- | sort | uniq -c | sort -nr | head -10
```

## Continuous Improvement

1. **Regular runbook reviews** - Monthly updates
2. **Incident retrospectives** - After each P0/P1 incident
3. **Training exercises** - Quarterly disaster recovery drills
4. **Tool improvements** - Invest in better monitoring and automation
5. **Knowledge sharing** - Document all learnings and share with team