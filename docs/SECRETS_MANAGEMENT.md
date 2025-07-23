# Secrets Management

This document describes the enterprise-grade secrets management system that replaces plain environment variable storage.

## Overview

The secrets management system provides:
- **Multiple provider support** (Environment Variables, Local Encrypted, AWS Secrets Manager, HashiCorp Vault)
- **Automatic fallback chain** with graceful degradation
- **Secret rotation and versioning** capabilities 
- **Encrypted local storage** with configurable encryption keys
- **Audit logging** for security compliance
- **Caching with TTL** for performance optimization
- **Timing attack prevention** for secure verification
- **Backward compatibility** with existing FINCHAT_TOKEN usage

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Secrets Manager                         │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │    Cache    │  │  Audit Log  │  │ Validation  │        │
│  │   (5min)    │  │  (Access)   │  │ (Timing)    │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                             │
│  Provider Chain (with fallback):                           │
│  1. AWS Secrets Manager  ──┐                               │
│  2. HashiCorp Vault       ──┼─► Secret Value               │
│  3. Local Encrypted       ──┘                               │
│  4. Environment Variables                                   │
└─────────────────────────────────────────────────────────────┘
```

## Configuration

### Environment Variables

```bash
# Provider Configuration
FINCHAT_SECRETS_PROVIDER=env|local|aws|vault     # Primary provider (default: env)
FINCHAT_SECRETS_FALLBACKS=vault,local,env        # Fallback chain (default: env)
FINCHAT_SECRETS_CACHE_TTL=300                    # Cache TTL in seconds (default: 300)

# Encryption (for local provider)
FINCHAT_ENCRYPTION_KEY=your_32_byte_encryption_key_here!

# AWS Secrets Manager
FINCHAT_AWS_REGION=us-east-1                     # AWS region (default: us-east-1)

# HashiCorp Vault  
FINCHAT_VAULT_URL=https://vault.example.com      # Vault server URL
FINCHAT_VAULT_TOKEN=vault_auth_token             # Vault authentication token

# Backward Compatibility
FINCHAT_TOKEN=your_authentication_token          # Still supported for backward compatibility
```

### Provider Setup

#### AWS Secrets Manager

```bash
# Install AWS CLI and configure credentials
aws configure set aws_access_key_id YOUR_ACCESS_KEY
aws configure set aws_secret_access_key YOUR_SECRET_KEY  
aws configure set default.region us-east-1

# Create secret
aws secretsmanager create-secret \
    --name "prod/finchat/token" \
    --description "FinChat authentication token" \
    --secret-string "your_secret_token_here"

# Set environment
export FINCHAT_SECRETS_PROVIDER=aws
export FINCHAT_AWS_REGION=us-east-1
```

#### HashiCorp Vault

```bash
# Start Vault server (development)
vault server -dev -dev-root-token-id="dev-token"

# Create secret
vault kv put secret/finchat/token token="your_secret_token_here"

# Set environment
export FINCHAT_SECRETS_PROVIDER=vault
export FINCHAT_VAULT_URL=http://localhost:8200
export FINCHAT_VAULT_TOKEN=dev-token
```

#### Local Encrypted Storage

```bash
# Generate strong encryption key (32+ bytes)
export FINCHAT_ENCRYPTION_KEY=$(openssl rand -base64 32)
export FINCHAT_SECRETS_PROVIDER=local

# Secrets will be stored encrypted in memory
# Use secret rotation API to populate
```

## Usage

### Automatic Integration (Recommended)

The secrets manager is automatically integrated into the configuration system. No code changes required:

```python
from finchat_sec_qa.config import get_config

config = get_config()
token = config.SECRET_TOKEN  # Automatically uses secrets manager
```

### Manual Usage

```python
from finchat_sec_qa.secrets_manager import SecretsManager

# Basic usage with environment fallback
manager = SecretsManager()
token = manager.get_secret('FINCHAT_TOKEN')

# Advanced configuration  
manager = SecretsManager(
    provider='aws',
    fallback_providers=['vault', 'local', 'env'],
    cache_ttl=600,
    region='us-west-2'
)

# Secret rotation
manager.rotate_secret('api_key', 'new_secret_value_v2', version=2)

# Get specific version
old_key = manager.get_secret('api_key', version=1)
current_key = manager.get_secret('api_key')  # Latest version

# Secure verification (timing-attack resistant)
is_valid = manager.verify_secret('FINCHAT_TOKEN', user_provided_token)
```

## Secret Rotation

### Manual Rotation

```python
from finchat_sec_qa.secrets_manager import SecretsManager

manager = SecretsManager(provider='local', encryption_key='your_key_here')

# Rotate to new version
manager.rotate_secret('database_password', 'new_secure_password_123', version=2)

# Verify new version is active
current = manager.get_secret('database_password')
assert current == 'new_secure_password_123'

# Old version still accessible for gradual migration
old = manager.get_secret('database_password', version=1)
```

### Automated Rotation (AWS)

```bash
# Enable automatic rotation in AWS Secrets Manager
aws secretsmanager rotate-secret \
    --secret-id "prod/finchat/token" \
    --rotation-lambda-arn "arn:aws:lambda:us-east-1:123456789:function:rotate-secret"
```

## Security Features

### Encryption

- **Local Storage**: AES-like encryption with SHA256 key derivation
- **In Transit**: HTTPS for external providers (AWS, Vault)
- **At Rest**: Provider-specific encryption (AWS KMS, Vault Transit)

### Access Control

```python
# Audit logging automatically tracks access
# Logs: timestamp, secret name, provider, user, access pattern

# Example log entry:
# INFO: Secret access: name=FINCHAT_TOKEN, source=aws, timestamp=1640995200, user=webapp
```

### Timing Attack Prevention

```python
# Constant-time comparison prevents timing attacks
is_valid = manager.verify_secret('token', user_input)  # Always takes ~same time
```

### Memory Management

```python
# Automatic cleanup for sensitive data
secret = manager.get_secret('sensitive_key', auto_cleanup=True)
# Secret cleared from cache after use
```

## Migration Guide

### From Environment Variables

**Before:**
```python
import os
token = os.getenv('FINCHAT_TOKEN')
```

**After (Automatic):**
```python
from finchat_sec_qa.config import get_config
config = get_config()
token = config.SECRET_TOKEN  # Uses secrets manager automatically
```

**After (Manual):**
```python
from finchat_sec_qa.secrets_manager import SecretsManager
manager = SecretsManager()
token = manager.get_secret('FINCHAT_TOKEN')
```

### Migration Steps

1. **Phase 1**: Deploy secrets manager with `env` provider (no change)
2. **Phase 2**: Configure external provider (AWS/Vault) as primary
3. **Phase 3**: Migrate secrets to external provider
4. **Phase 4**: Remove environment variables
5. **Phase 5**: Enable encryption for local fallback

## Monitoring & Troubleshooting

### Health Checks

```python
from finchat_sec_qa.secrets_manager import SecretsManager

manager = SecretsManager()

# Test provider connectivity
try:
    test_secret = manager.get_secret('health_check_secret')
    print("✅ Secrets manager healthy")
except Exception as e:
    print(f"❌ Secrets manager error: {e}")
```

### Metrics & Logging

```python
import logging

# Enable debug logging
logging.getLogger('finchat_sec_qa.secrets_manager').setLevel(logging.DEBUG)

# Monitor cache performance
print(f"Cache hit ratio: {manager.cache_stats()}")
```

### Common Issues

#### 1. Provider Connection Failed

```
ERROR: Failed to initialize AWS client: No credentials found
```

**Solution**: Configure AWS credentials or use fallback chain

#### 2. Encryption Key Too Short

```
ValueError: Encryption key must be at least 32 bytes long
```  

**Solution**: Generate proper key:
```bash
export FINCHAT_ENCRYPTION_KEY=$(openssl rand -base64 32)
```

#### 3. Secret Not Found

```
SecretNotFoundError: Secret 'api_key' not found in any provider
```

**Solution**: Verify secret exists in configured providers

#### 4. Vault Authentication Failed

```
WARN: Failed to initialize Vault session: 403 Forbidden
```

**Solution**: Check Vault token and permissions

## Performance Characteristics

| Provider | Latency | Scalability | Persistence |
|----------|---------|-------------|-------------|  
| Environment | ~0.1ms | ✅ Unlimited | ❌ Process-bound |
| Local Encrypted | ~1ms | ✅ Unlimited | ❌ Process-bound |
| AWS Secrets Manager | ~50ms | ✅ Unlimited | ✅ Persistent |
| HashiCorp Vault | ~20ms | ✅ Unlimited | ✅ Persistent |

### Caching Impact

- **Cache Hit**: ~0.1ms (any provider)
- **Cache Miss**: Provider latency + cache update
- **Default TTL**: 5 minutes (configurable)

## Security Best Practices

1. **Use External Providers**: Prefer AWS/Vault over environment variables
2. **Enable Encryption**: Use encryption keys for local storage
3. **Rotate Regularly**: Implement automated secret rotation
4. **Monitor Access**: Enable audit logging for compliance
5. **Limit Scope**: Use principle of least privilege for provider access
6. **Secure Transport**: Always use HTTPS/TLS for external providers

## Compliance Features

### SOC 2 Type II
- ✅ Audit logging with timestamps
- ✅ Access control and authentication
- ✅ Encryption at rest and in transit
- ✅ Secret rotation capabilities

### PCI DSS
- ✅ Encrypted storage of sensitive data
- ✅ Access logging and monitoring
- ✅ Regular secret rotation
- ✅ Secure transmission protocols

### HIPAA
- ✅ Administrative safeguards (access control)
- ✅ Physical safeguards (encrypted storage)
- ✅ Technical safeguards (audit logs, encryption)

## API Reference

### SecretsManager Class

```python
class SecretsManager:
    def __init__(
        self,
        provider: str = 'env',
        encryption_key: Optional[str] = None,
        fallback_providers: Optional[List[str]] = None,
        cache_ttl: int = 300,
        **provider_config
    ) -> None:
        """Initialize secrets manager."""
    
    def get_secret(
        self,
        secret_name: str,
        version: Optional[int] = None,
        field: Optional[str] = None,
        auto_cleanup: bool = False
    ) -> str:
        """Retrieve secret with fallback chain."""
    
    def store_secret(
        self,
        secret_name: str,
        secret_value: str,
        version: int = 1,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Store secret in local encrypted storage."""
    
    def rotate_secret(
        self,
        secret_name: str,
        new_value: str,
        version: int
    ) -> None:
        """Rotate secret to new version."""
    
    def verify_secret(
        self,
        secret_name: str,
        provided_value: str
    ) -> bool:
        """Verify provided value (timing-attack resistant)."""
```

### Exceptions

```python
class SecretNotFoundError(Exception):
    """Raised when secret not found in any provider."""
    pass
```