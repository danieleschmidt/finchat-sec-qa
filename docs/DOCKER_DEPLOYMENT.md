# Docker Deployment Guide

This guide covers containerized deployment of the FinChat SEC QA service using Docker and docker-compose.

## Quick Start

### Prerequisites

- Docker (v20.10+)
- Docker Compose (v2.0+)
- At least 2GB of available RAM
- Internet connection for downloading SEC filings

### Local Development Setup

1. **Clone and prepare environment**:
   ```bash
   git clone <repository-url>
   cd finchat-sec-qa
   
   # Copy environment template
   cp .env.example .env
   
   # Edit .env file with your configuration
   nano .env
   ```

2. **Start services**:
   ```bash
   # Production-like environment
   docker-compose up -d
   
   # Development environment with hot reload
   docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
   ```

3. **Verify deployment**:
   ```bash
   # Check service health
   curl http://localhost:8000/health
   curl http://localhost:5000/
   
   # View logs
   docker-compose logs -f api
   docker-compose logs -f webapp
   ```

4. **Stop services**:
   ```bash
   docker-compose down
   ```

## Service Architecture

### Services Overview

| Service | Purpose | Port | Health Check |
|---------|---------|------|-------------|
| `api` | FastAPI server for API endpoints | 8000 | `/health` |
| `webapp` | Flask web interface | 5000 | `/` |
| `prometheus` | Metrics collection (optional) | 9090 | Built-in |

### Network Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Host Network                                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │    8000     │  │    5000     │  │    9090     │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
              │              │              │
┌─────────────────────────────────────────────────────────────┐
│ finchat_network (Bridge)                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   api:8000  │  │webapp:5000  │  │prometheus:  │         │
│  │   (FastAPI) │  │   (Flask)   │  │    9090     │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

## Configuration

### Environment Variables

All configuration is managed through environment variables following the Twelve-Factor App principles. See `.env.example` for complete configuration options.

#### Core Configuration

```bash
# Server Configuration
FINCHAT_HOST=0.0.0.0
FINCHAT_PORT=8000
FINCHAT_WEBAPP_PORT=5000
FINCHAT_LOG_LEVEL=INFO

# Security Configuration
FINCHAT_RATE_LIMIT_MAX_REQUESTS=100
FINCHAT_MIN_TOKEN_LENGTH=16
FINCHAT_FAILED_ATTEMPTS_LOCKOUT_THRESHOLD=3
```

#### Docker-Specific Configuration

```bash
# Cache directory (inside container)
FINCHAT_CACHE_DIR=/app/.cache/finchat_sec_qa

# Python configuration
PYTHONUNBUFFERED=1
PYTHONDONTWRITEBYTECODE=1
```

### Volume Management

#### Persistent Cache Storage

The `finchat_cache` volume stores:
- Downloaded SEC filings
- Processed document indexes  
- ML model caches

```bash
# View cache usage
docker-compose exec api du -sh /app/.cache

# Clear cache (will re-download filings)
docker-compose down
docker volume rm finchat_cache
docker-compose up -d
```

#### Development Volume Mounts

For development with hot reload:

```bash
# Enable source code mounting
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

This mounts:
- `./src:/app/src:ro` - Source code (read-only)
- `./tests:/app/tests:ro` - Test files (dev-tools profile)

## Building Images

### Manual Build

```bash
# Build API server image
docker build -f docker/Dockerfile.api -t finchat-api:latest .

# Build webapp image  
docker build -f docker/Dockerfile.webapp -t finchat-webapp:latest .
```

### Build Arguments

Both Dockerfiles support build-time customization:

```bash
# Build with specific Python version
docker build --build-arg PYTHON_VERSION=3.11 -f docker/Dockerfile.api .
```

### Multi-Platform Builds

For deployment across different architectures:

```bash
# Build for multiple platforms
docker buildx build --platform linux/amd64,linux/arm64 \
                    -f docker/Dockerfile.api \
                    -t finchat-api:latest .
```

## Production Deployment

### Security Hardening

1. **Non-root user**: Both containers run as `finchat` user
2. **Minimal base images**: Using `python:3.11-slim`
3. **Multi-stage builds**: Separate build and runtime stages
4. **Health checks**: Configured for container orchestration

### Resource Limits

Add resource constraints for production:

```yaml
# docker-compose.prod.yml
services:
  api:
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
        reservations:
          memory: 512M
          cpus: '0.25'
  
  webapp:
    deploy:
      resources:
        limits:
          memory: 512M
          cpus: '0.25'
        reservations:
          memory: 256M
          cpus: '0.125'
```

### High Availability

For production scaling:

```yaml
services:
  api:
    deploy:
      replicas: 3
      restart_policy:
        condition: on-failure
        max_attempts: 3
    depends_on:
      - redis  # Add Redis for session storage
      - postgres  # Add PostgreSQL for persistent data
```

### Monitoring Integration

Enable Prometheus monitoring:

```bash
# Start with monitoring
docker-compose --profile monitoring up -d

# Access Prometheus UI
open http://localhost:9090
```

Available metrics:
- HTTP request counts and latencies
- Business metrics (QA queries, risk analyses)
- Service health status
- System resource usage

## Troubleshooting

### Common Issues

#### Container Won't Start

```bash
# Check logs
docker-compose logs api
docker-compose logs webapp

# Common fixes:
# 1. Port already in use
sudo lsof -i :8000
sudo lsof -i :5000

# 2. Permission issues
sudo chown -R $(id -u):$(id -g) .docker_cache/

# 3. Out of disk space
docker system prune
```

#### Health Check Failures

```bash
# Check health status
docker-compose ps

# Test health endpoints manually
curl -v http://localhost:8000/health
curl -v http://localhost:5000/

# Check container health
docker inspect finchat-api | jq '.[].State.Health'
```

#### Performance Issues

```bash
# Monitor resource usage
docker stats

# Check cache performance
docker-compose exec api ls -la /app/.cache/finchat_sec_qa/

# Review configuration
docker-compose exec api env | grep FINCHAT_
```

### Debugging

#### Interactive Debugging

```bash
# Get shell in running container
docker-compose exec api bash
docker-compose exec webapp bash

# Run tests inside container
docker-compose exec api python -m pytest tests/

# Check Python imports
docker-compose exec api python -c "import finchat_sec_qa; print('OK')"
```

#### Development Tools

```bash
# Start dev tools container
docker-compose --profile dev-tools up -d dev-tools

# Use dev tools
docker-compose exec dev-tools python scripts/load_test.py http://api:8000
docker-compose exec dev-tools python scripts/benchmark.py 1.4.4 http://api:8000
```

### Log Management

```bash
# View real-time logs
docker-compose logs -f --tail=100

# Export logs
docker-compose logs api > api.log
docker-compose logs webapp > webapp.log

# Rotate logs (production)
docker-compose down
docker system prune --volumes
docker-compose up -d
```

## CI/CD Integration

### Docker Image Registry

```bash
# Tag for registry
docker tag finchat-api:latest registry.company.com/finchat-api:v1.4.4
docker tag finchat-webapp:latest registry.company.com/finchat-webapp:v1.4.4

# Push to registry
docker push registry.company.com/finchat-api:v1.4.4
docker push registry.company.com/finchat-webapp:v1.4.4
```

### Automated Testing

```bash
# Test Docker builds
docker build -f docker/Dockerfile.api -t finchat-api:test .
docker build -f docker/Dockerfile.webapp -t finchat-webapp:test .

# Test docker-compose configuration
docker-compose -f docker-compose.yml config

# Integration tests
docker-compose -f docker-compose.yml -f docker-compose.test.yml up --abort-on-container-exit
```

### Environment Promotion

```bash
# Development → Staging → Production
docker-compose -f docker-compose.yml -f docker-compose.staging.yml up -d
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## Best Practices

### Development Workflow

1. **Use dev compose**: `docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d`
2. **Hot reload enabled**: Source code changes reflect immediately
3. **Debug logging**: Full debug output for development
4. **Test locally**: Run tests inside dev-tools container

### Production Deployment

1. **Use specific versions**: Never deploy `:latest` tags
2. **Resource limits**: Set memory and CPU constraints
3. **Health checks**: Configure appropriate health check intervals
4. **Monitoring**: Enable Prometheus metrics collection
5. **Backup strategy**: Regular cache and data backups
6. **Security scanning**: Scan images for vulnerabilities

### Performance Optimization

1. **Multi-stage builds**: Minimize final image size
2. **Layer caching**: Optimize Dockerfile layer ordering
3. **Resource sharing**: Use shared volumes for cache
4. **Connection pooling**: Configure appropriate worker counts
5. **Monitoring**: Use metrics to identify bottlenecks

---

**Need help?** Check the logs first, then review this guide. For issues not covered here, create an issue in the repository.