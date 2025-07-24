# Setup and Configuration Guide

This guide provides detailed instructions for setting up FinChat SEC QA with all required API keys and configuration options.

## Prerequisites

- Python 3.8 or higher
- pip or poetry for package management
- Internet connection for API access

## API Keys and External Services

### EDGAR API Configuration

The SEC EDGAR database requires a User-Agent header to identify your application:

1. **User Agent Setup**:
   ```bash
   # Set your application identifier
   export FINCHAT_USER_AGENT="YourCompany YourApp your.email@company.com"
   ```

2. **Rate Limiting**:
   - SEC allows 10 requests per second maximum
   - The application automatically handles rate limiting
   - Configure rate limits if needed:
   ```bash
   export FINCHAT_RATE_LIMIT_MAX_REQUESTS=10
   export FINCHAT_RATE_LIMIT_WINDOW_SECONDS=1
   ```

3. **EDGAR Best Practices**:
   - Always include a meaningful User-Agent
   - Respect rate limits to avoid being blocked
   - Cache responses when possible (automatically handled)

### OpenAI API Configuration (Optional)

For enhanced Q&A capabilities, you can configure OpenAI integration:

1. **Get OpenAI API Key**:
   - Visit [OpenAI API Dashboard](https://platform.openai.com/api-keys)
   - Create a new API key
   - Copy the key for configuration

2. **Set OpenAI Configuration**:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key-here"
   export FINCHAT_OPENAI_MODEL="gpt-3.5-turbo"  # or gpt-4
   ```

## Environment Variables

### Required Configuration

```bash
# Application Authentication
FINCHAT_TOKEN="your-secure-token-here"

# EDGAR API Configuration
FINCHAT_USER_AGENT="YourCompany YourApp your.email@company.com"

# Logging Configuration
FINCHAT_LOG_LEVEL="INFO"  # DEBUG, INFO, WARNING, ERROR
```

### Optional Configuration

```bash
# Rate Limiting
FINCHAT_RATE_LIMIT_MAX_REQUESTS=10
FINCHAT_RATE_LIMIT_WINDOW_SECONDS=1

# Redis Configuration (for distributed rate limiting)
FINCHAT_REDIS_URL="redis://localhost:6379/0"
FINCHAT_REDIS_POOL_MAX_CONNECTIONS=20

# Performance Tuning
FINCHAT_CHUNK_SIZE=1000
FINCHAT_CHUNK_OVERLAP=200

# Security Configuration
FINCHAT_SECRETS_PROVIDER="local"  # local, aws, vault
FINCHAT_CORS_ORIGINS="http://localhost:3000,https://yourapp.com"

# Metrics and Monitoring
FINCHAT_METRICS_ENABLED="true"
FINCHAT_PROMETHEUS_PORT=8000
```

## Configuration Methods

### Method 1: Environment File (.env)

1. **Create Environment File**:
   ```bash
   cp .env.example .env
   ```

2. **Edit Configuration**:
   ```bash
   nano .env  # or use your preferred editor
   ```

3. **Set Required Values**:
   ```env
   FINCHAT_TOKEN="generate-secure-token-here"
   FINCHAT_USER_AGENT="YourCompany FinChatApp your.email@company.com"
   FINCHAT_LOG_LEVEL="INFO"
   ```

### Method 2: Direct Environment Variables

```bash
export FINCHAT_TOKEN="your-secure-token"
export FINCHAT_USER_AGENT="YourCompany FinChatApp your.email@company.com"
export FINCHAT_LOG_LEVEL="INFO"
```

### Method 3: Docker Environment

```yaml
# docker-compose.yml
environment:
  - FINCHAT_TOKEN=your-secure-token
  - FINCHAT_USER_AGENT=YourCompany FinChatApp your.email@company.com
  - FINCHAT_LOG_LEVEL=INFO
```

## Generating Secure Tokens

For production deployments, generate secure authentication tokens:

```bash
# Generate a secure random token
python3 -c "import secrets; print(secrets.token_urlsafe(32))"

# Or using openssl
openssl rand -base64 32
```

## Validation and Testing

### Test Configuration

```bash
# Test basic functionality
python3 -c "
from finchat_sec_qa.config import get_config
config = get_config()
print(f'Token configured: {bool(config.FINCHAT_TOKEN)}')
print(f'User Agent: {config.USER_AGENT}')
print(f'Log Level: {config.LOG_LEVEL}')
"
```

### Test EDGAR API Access

```bash
# Test EDGAR client
python3 -c "
from finchat_sec_qa.edgar_client import EdgarClient
client = EdgarClient('YourCompany FinChatApp your.email@company.com')
print('EDGAR client created successfully')
"
```

## Troubleshooting

### Common Issues

1. **"User-Agent required" Error**:
   - Ensure FINCHAT_USER_AGENT is set
   - Format: "Company Application email@domain.com"
   - Example: "ACME FinChat support@acme.com"

2. **Rate Limiting Issues**:
   - SEC enforces 10 requests/second limit
   - Application automatically handles this
   - Reduce concurrent requests if issues persist

3. **Authentication Errors**:
   - Verify FINCHAT_TOKEN is set correctly
   - Ensure token is kept secure and not logged
   - Generate new token if compromised

4. **Redis Connection Issues**:
   - Check Redis server is running
   - Verify FINCHAT_REDIS_URL is correct
   - Application falls back to in-memory if Redis unavailable

### Getting Help

- Check logs for detailed error messages
- Verify all environment variables are set
- Ensure external services (Redis, etc.) are accessible
- Review [CONTRIBUTING.md](../CONTRIBUTING.md) for development setup

## Production Deployment

### Security Checklist

- [ ] Use strong, randomly generated FINCHAT_TOKEN
- [ ] Set appropriate CORS origins
- [ ] Enable HTTPS/TLS
- [ ] Configure proper logging level (INFO or WARNING)
- [ ] Use external Redis for rate limiting
- [ ] Set up monitoring and metrics collection
- [ ] Regular security updates

### Performance Optimization

- [ ] Configure Redis connection pooling
- [ ] Tune chunk size for document processing
- [ ] Set appropriate worker counts for parallel processing
- [ ] Monitor metrics and adjust configuration
- [ ] Use CDN for static assets if applicable

For more deployment options, see [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md).