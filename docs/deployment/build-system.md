# Build System Documentation

This document describes the build system, containerization, and deployment procedures for FinChat-SEC-QA.

## Build Architecture

### Multi-Stage Docker Builds

Our Docker builds use multi-stage builds to optimize image size and security:

1. **Builder Stage**: Compiles dependencies and builds the application
2. **Production Stage**: Creates minimal runtime image with only necessary components

### Container Images

#### API Server (`docker/Dockerfile.api`)
- **Base Image**: `python:3.11-slim`
- **Purpose**: FastAPI server for REST API endpoints
- **Port**: 8000
- **Health Check**: `/health` endpoint
- **Security**: Runs as non-root user `finchat`

#### Web Application (`docker/Dockerfile.webapp`)
- **Base Image**: `python:3.11-slim`
- **Purpose**: Flask web interface
- **Port**: 5000
- **Health Check**: `/` endpoint
- **Security**: Runs as non-root user `finchat`

## Build Commands

### Local Development

```bash
# Install dependencies
make install-dev

# Build Python package
make build

# Build Docker images
make docker-build

# Run containers locally
make docker-run
```

### CI/CD Pipeline

```bash
# Full build pipeline
make release-check

# Security scanning
make security-scan

# Generate SBOM
python scripts/generate_sbom.py
```

## Docker Compose Configuration

### Production (`docker-compose.yml`)
- API service on port 8000
- Web application on port 5000
- Prometheus monitoring on port 9090 (optional)
- Shared cache volume
- Health checks for all services

### Development (`docker-compose.dev.yml`)
- Hot reloading enabled
- Source code mounted as volumes
- Debug logging enabled
- Development environment variables

## Build Optimization

### Docker Layer Caching
- Requirements copied before source code for better caching
- Multi-stage builds to minimize final image size
- `.dockerignore` excludes unnecessary files

### Security Best Practices
- Non-root user execution
- Minimal base images
- No secrets in build context
- Regular security scanning

## Build Artifacts

### Software Bill of Materials (SBOM)
Generated automatically during builds:
- **SPDX format**: `sbom.spdx.json`
- **CycloneDX format**: `sbom.cyclonedx.json`

```bash
# Generate SBOM manually
python scripts/generate_sbom.py
```

### Package Distribution
- Source distribution (sdist)
- Wheel distribution (bdist_wheel)
- Container images for deployment

## Environment Configuration

### Build-time Variables
```dockerfile
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV FINCHAT_LOG_LEVEL=INFO
```

### Runtime Configuration
All runtime configuration via environment variables:
- See `.env.example` for complete list
- Configuration validation on startup
- Secrets management support

## Deployment Targets

### Staging Environment
```bash
make deploy-staging
```
- Automated testing environment
- Pre-production validation
- Performance testing

### Production Environment
```bash
make deploy-prod
```
- Full release validation required
- Blue-green deployment strategy
- Monitoring and alerting

## Build Pipeline Integration

### GitHub Actions
```yaml
name: Build and Test
on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build Docker images
        run: make docker-build
      - name: Run tests
        run: make test
      - name: Security scan
        run: make security-scan
```

### Container Registry
- Images pushed to container registry
- Tagged with commit SHA and version
- Latest tag for main branch
- Automated cleanup of old images

## Performance Optimization

### Build Performance
- Parallel layer building
- Dependency caching
- Minimal context transfer

### Runtime Performance
- Optimized Python packages
- Minimal runtime dependencies
- Efficient container startup

## Monitoring and Observability

### Build Metrics
- Build duration tracking
- Image size monitoring
- Dependency vulnerability counts
- SBOM generation success rates

### Runtime Metrics
- Container health status
- Resource utilization
- Application performance metrics
- Error rates and logging

## Security Considerations

### Build Security
- Dependency vulnerability scanning
- Static code analysis
- Secrets detection
- Container image scanning

### Runtime Security
- Non-root execution
- Read-only file systems where possible
- Network security policies
- Resource limits

## Troubleshooting

### Common Build Issues

#### Docker Build Failures
```bash
# Check build context size
du -sh .

# Verify .dockerignore is working
docker build --no-cache -f docker/Dockerfile.api .
```

#### Dependency Issues
```bash
# Clear pip cache
pip cache purge

# Rebuild with fresh dependencies
make clean && make install-dev
```

#### Container Runtime Issues
```bash
# Check container logs
docker-compose logs api

# Debug container startup
docker run -it finchat-sec-qa:api /bin/bash
```

### Performance Issues

#### Slow Build Times
- Check Docker layer caching
- Optimize .dockerignore
- Use multi-stage builds effectively

#### Large Image Sizes
- Review installed packages
- Use alpine base images if appropriate
- Clean up build artifacts

## Maintenance

### Regular Tasks
- Update base images monthly
- Review and update dependencies
- Clean up old build artifacts
- Update build documentation

### Security Updates
- Monitor security advisories
- Automated dependency updates
- Regular vulnerability scans
- SBOM review and analysis

## Best Practices

### Development
1. Test builds locally before pushing
2. Use consistent environment variables
3. Follow semantic versioning
4. Document configuration changes

### Production
1. Always use tagged versions
2. Validate SBOM before deployment
3. Monitor resource usage
4. Maintain rollback capabilities

### Security
1. Regular security scanning
2. Minimal attack surface
3. Secrets management best practices
4. Regular access reviews

## References

- [Docker Best Practices](https://docs.docker.com/develop/best-practices/)
- [Multi-stage Builds](https://docs.docker.com/develop/images/multistage-build/)
- [SBOM Guide](https://www.cisa.gov/sbom)
- [Container Security](https://kubernetes.io/docs/concepts/security/)