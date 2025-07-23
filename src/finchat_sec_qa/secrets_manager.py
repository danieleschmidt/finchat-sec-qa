"""
Secure secrets management module with encryption, external providers, and audit logging.
Provides enterprise-grade secret storage with fallback mechanisms and security features.
"""
import os
import time
import hmac
import hashlib
import logging
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field
import json
import base64

logger = logging.getLogger(__name__)


class SecretNotFoundError(Exception):
    """Raised when a requested secret is not found in any provider."""
    pass


@dataclass
class SecretMetadata:
    """Metadata for stored secrets."""
    version: int = 1
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    tags: Dict[str, str] = field(default_factory=dict)


class SecretsManager:
    """
    Enterprise secrets management with multiple providers and security features.
    
    Supports:
    - Environment variables (backward compatibility)
    - Encrypted local storage 
    - AWS Secrets Manager
    - HashiCorp Vault
    - Secret rotation and versioning
    - Audit logging and caching
    """
    
    def __init__(
        self,
        provider: str = 'env',
        encryption_key: Optional[str] = None,
        fallback_providers: Optional[List[str]] = None,
        cache_ttl: int = 300,
        **provider_config
    ):
        """Initialize secrets manager with specified provider and configuration."""
        self.provider = provider
        self.fallback_providers = fallback_providers or ['env']
        self.cache_ttl = cache_ttl
        self.provider_config = provider_config
        
        # Secret cache with TTL
        self._cache: Dict[str, Dict[str, Any]] = {}
        
        # Local encrypted storage
        self._local_storage: Dict[str, Dict[str, Any]] = {}
        
        # Encryption setup
        if encryption_key:
            self._validate_encryption_key(encryption_key)
            self.encryption_key = encryption_key.encode('utf-8')
        else:
            self.encryption_key = None
        
        # Initialize provider clients
        self._init_providers()
    
    def _validate_encryption_key(self, key: str) -> None:
        """Validate encryption key strength."""
        if len(key.encode('utf-8')) < 32:
            raise ValueError("Encryption key must be at least 32 bytes long")
    
    def _init_providers(self) -> None:
        """Initialize external provider clients."""
        self._aws_client = None
        self._vault_session = None
        
        if self.provider == 'aws' or 'aws' in self.fallback_providers:
            self._init_aws()
        
        if self.provider == 'vault' or 'vault' in self.fallback_providers:
            self._init_vault()
    
    def _init_aws(self) -> None:
        """Initialize AWS Secrets Manager client."""
        try:
            import boto3
            self._aws_client = boto3.client(
                'secretsmanager',
                region_name=self.provider_config.get('region', 'us-east-1')
            )
            logger.info("Initialized AWS Secrets Manager client")
        except ImportError:
            logger.warning("boto3 not available, AWS Secrets Manager disabled")
        except Exception as e:
            logger.warning(f"Failed to initialize AWS client: {e}")
    
    def _init_vault(self) -> None:
        """Initialize HashiCorp Vault session."""
        try:
            import requests
            self._vault_session = requests.Session()
            
            vault_token = self.provider_config.get('vault_token')
            if vault_token:
                self._vault_session.headers.update({
                    'X-Vault-Token': vault_token
                })
            
            logger.info("Initialized HashiCorp Vault session")
        except ImportError:
            logger.warning("requests not available, Vault integration disabled")
        except Exception as e:
            logger.warning(f"Failed to initialize Vault session: {e}")
    
    def get_secret(
        self,
        secret_name: str,
        version: Optional[int] = None,
        field: Optional[str] = None,
        auto_cleanup: bool = False
    ) -> str:
        """
        Retrieve secret from configured provider with fallback chain.
        
        Args:
            secret_name: Name/ID of the secret
            version: Specific version to retrieve (if supported)
            field: Specific field within secret (for structured secrets)
            auto_cleanup: Whether to clear from cache after retrieval
            
        Returns:
            Secret value as string
            
        Raises:
            SecretNotFoundError: If secret not found in any provider
        """
        # Check cache first
        cache_key = f"{secret_name}:{version or 'latest'}"
        if cache_key in self._cache:
            cached_entry = self._cache[cache_key]
            if time.time() - cached_entry['timestamp'] < self.cache_ttl:
                self._audit_secret_access(secret_name, 'cache_hit')
                cached_entry['metadata'].last_accessed = time.time()
                cached_entry['metadata'].access_count += 1
                
                secret_value = cached_entry['value']
                if field and isinstance(secret_value, dict):
                    secret_value = secret_value.get(field, '')
                
                if auto_cleanup:
                    del self._cache[cache_key]
                
                return secret_value
        
        # Try providers in order
        providers_to_try = [self.provider] + self.fallback_providers
        
        for provider in providers_to_try:
            try:
                secret_value = self._get_secret_from_provider(
                    provider, secret_name, version, field
                )
                
                # Cache the result
                metadata = SecretMetadata(version=version or 1)
                self._cache[cache_key] = {
                    'value': secret_value,
                    'timestamp': time.time(),
                    'metadata': metadata
                }
                
                self._audit_secret_access(secret_name, provider)
                
                if auto_cleanup:
                    # Schedule cleanup (in production, use proper cleanup mechanism)
                    pass
                
                return secret_value
                
            except Exception as e:
                logger.debug(f"Provider {provider} failed for {secret_name}: {e}")
                continue
        
        # No provider succeeded
        raise SecretNotFoundError(f"Secret '{secret_name}' not found in any provider")
    
    def _get_secret_from_provider(
        self,
        provider: str,
        secret_name: str,
        version: Optional[int] = None,
        field: Optional[str] = None
    ) -> str:
        """Get secret from specific provider."""
        if provider == 'env':
            return self._get_from_environment(secret_name)
        elif provider == 'local':
            return self._get_from_local_storage(secret_name, version)
        elif provider == 'aws':
            return self._get_from_aws(secret_name, version, field)
        elif provider == 'vault':
            return self._get_from_vault(secret_name, field)
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def _get_from_environment(self, secret_name: str) -> str:
        """Get secret from environment variables."""
        value = os.getenv(secret_name)
        if value is None:
            raise SecretNotFoundError(f"Environment variable {secret_name} not found")
        return value
    
    def _get_from_local_storage(self, secret_name: str, version: Optional[int] = None) -> str:
        """Get secret from encrypted local storage."""
        if secret_name not in self._local_storage:
            raise SecretNotFoundError(f"Local secret {secret_name} not found")
        
        secret_data = self._local_storage[secret_name]
        
        if version:
            version_key = f"v{version}"
            if version_key not in secret_data:
                raise SecretNotFoundError(f"Version {version} of {secret_name} not found")
            encrypted_value = secret_data[version_key]
        else:
            # Get latest version
            latest_version = max(
                int(k[1:]) for k in secret_data.keys() if k.startswith('v')
            )
            encrypted_value = secret_data[f"v{latest_version}"]
        
        return self._decrypt_value(encrypted_value)
    
    def _get_from_aws(self, secret_name: str, version: Optional[int] = None, field: Optional[str] = None) -> str:
        """Get secret from AWS Secrets Manager."""
        if not self._aws_client:
            raise Exception("AWS client not initialized")
        
        kwargs = {'SecretId': secret_name}
        if version:
            kwargs['VersionStage'] = f'AWSPENDING{version}'
        
        response = self._aws_client.get_secret_value(**kwargs)
        secret_string = response['SecretString']
        
        if field:
            secret_data = json.loads(secret_string)
            return secret_data.get(field, '')
        
        return secret_string
    
    def _get_from_vault(self, secret_name: str, field: Optional[str] = None) -> str:
        """Get secret from HashiCorp Vault."""
        if not self._vault_session:
            raise Exception("Vault session not initialized")
        
        vault_url = self.provider_config.get('vault_url', 'http://localhost:8200')
        url = f"{vault_url}/v1/{secret_name}"
        
        response = self._vault_session.get(url)
        response.raise_for_status()
        
        data = response.json()
        secret_data = data['data']['data']
        
        if field:
            return secret_data.get(field, '')
        
        # Return first value if no field specified
        return list(secret_data.values())[0] if secret_data else ''
    
    def store_secret(
        self,
        secret_name: str,
        secret_value: str,
        version: int = 1,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Store secret in local encrypted storage."""
        if not self.encryption_key:
            raise ValueError("Encryption key required for storing secrets")
        
        encrypted_value = self._encrypt_value(secret_value)
        
        if secret_name not in self._local_storage:
            self._local_storage[secret_name] = {}
        
        version_key = f"v{version}"
        self._local_storage[secret_name][version_key] = encrypted_value
        
        # Store metadata
        metadata = SecretMetadata(version=version, tags=tags or {})
        self._local_storage[secret_name]['metadata'] = metadata
        
        self._audit_secret_access(secret_name, 'stored')
    
    def rotate_secret(self, secret_name: str, new_value: str, version: int) -> None:
        """Rotate secret to new version."""
        self.store_secret(secret_name, new_value, version)
        
        # Invalidate cache
        cache_keys_to_remove = [
            key for key in self._cache.keys() 
            if key.startswith(f"{secret_name}:")
        ]
        for key in cache_keys_to_remove:
            del self._cache[key]
        
        logger.info(f"Rotated secret {secret_name} to version {version}")
    
    def verify_secret(self, secret_name: str, provided_value: str) -> bool:
        """Verify provided value against stored secret (constant-time comparison)."""
        try:
            actual_value = self.get_secret(secret_name)
            # Use constant-time comparison to prevent timing attacks
            return hmac.compare_digest(actual_value.encode('utf-8'), provided_value.encode('utf-8'))
        except SecretNotFoundError:
            # Still perform comparison to maintain constant time
            hmac.compare_digest(b'dummy_value', provided_value.encode('utf-8'))
            return False
    
    def _encrypt_value(self, value: str) -> str:
        """Encrypt secret value using AES-GCM authenticated encryption."""
        if not self.encryption_key:
            raise ValueError("No encryption key configured")
        
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
            import os
            
            # Generate random 12-byte IV for AES-GCM
            iv = os.urandom(12)
            
            # Derive AES key from provided key using PBKDF2
            # Use a fixed salt for backward compatibility (in production, use per-secret salt)
            salt = b'finchat_secrets_salt_v1'
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,  # AES-256
                salt=salt,
                iterations=100000,  # Recommended minimum
            )
            derived_key = kdf.derive(self.encryption_key)
            
            # Encrypt with AES-GCM
            aesgcm = AESGCM(derived_key)
            value_bytes = value.encode('utf-8')
            
            # Additional authenticated data (version info for future upgrades)
            aad = b'finchat_secrets_v2'
            ciphertext = aesgcm.encrypt(iv, value_bytes, aad)
            
            # Format: version(2) + IV(12) + ciphertext_with_tag(variable)
            version_prefix = b'v2'
            encrypted_data = version_prefix + iv + ciphertext
            
            return base64.b64encode(encrypted_data).decode('ascii')
            
        except ImportError:
            # Fallback to legacy XOR encryption if cryptography not available
            logger.warning("Cryptography library not available, using legacy XOR encryption")
            return self._encrypt_value_legacy(value)
    
    def _decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt secret value with format detection and backward compatibility."""
        if not self.encryption_key:
            raise ValueError("No encryption key configured")
        
        try:
            encrypted_bytes = base64.b64decode(encrypted_value.encode('ascii'))
            
            # Check for version prefix to determine format
            if encrypted_bytes.startswith(b'v2'):
                return self._decrypt_value_v2(encrypted_bytes)
            else:
                # Legacy XOR-encrypted data (no version prefix)
                return self._decrypt_value_legacy(encrypted_bytes)
                
        except Exception as e:
            # Constant-time error handling to prevent timing attacks
            import time
            time.sleep(0.001)  # Small constant delay
            raise ValueError("Failed to decrypt secret value") from e
    
    def _decrypt_value_v2(self, encrypted_bytes: bytes) -> str:
        """Decrypt AES-GCM encrypted value (v2 format)."""
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
            
            # Extract components: version(2) + IV(12) + ciphertext_with_tag
            if len(encrypted_bytes) < 2 + 12 + 16:  # version + IV + min ciphertext + tag
                raise ValueError("Invalid encrypted data format")
            
            version_prefix = encrypted_bytes[:2]
            iv = encrypted_bytes[2:14]
            ciphertext = encrypted_bytes[14:]
            
            # Derive AES key using same parameters as encryption
            salt = b'finchat_secrets_salt_v1'
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,  # AES-256
                salt=salt,
                iterations=100000,
            )
            derived_key = kdf.derive(self.encryption_key)
            
            # Decrypt with AES-GCM
            aesgcm = AESGCM(derived_key)
            aad = b'finchat_secrets_v2'
            decrypted_bytes = aesgcm.decrypt(iv, ciphertext, aad)
            
            return decrypted_bytes.decode('utf-8')
            
        except ImportError:
            raise ValueError("Cryptography library required for v2 format decryption")
    
    def _decrypt_value_legacy(self, encrypted_bytes: bytes) -> str:
        """Decrypt legacy XOR-encrypted value for backward compatibility."""
        key_hash = hashlib.sha256(self.encryption_key).digest()
        decrypted_bytes = bytes(a ^ b for a, b in zip(encrypted_bytes, key_hash * (len(encrypted_bytes) // len(key_hash) + 1)))
        return decrypted_bytes.decode('utf-8')
    
    def _encrypt_value_legacy(self, value: str) -> str:
        """Legacy XOR encryption fallback."""
        key_hash = hashlib.sha256(self.encryption_key).digest()
        value_bytes = value.encode('utf-8')
        encrypted_bytes = bytes(a ^ b for a, b in zip(value_bytes, key_hash * (len(value_bytes) // len(key_hash) + 1)))
        return base64.b64encode(encrypted_bytes).decode('ascii')
    
    def _audit_secret_access(self, secret_name: str, source: str) -> None:
        """Log secret access for audit purposes (without revealing the value)."""
        logger.info(
            f"Secret access: name={secret_name}, source={source}, "
            f"timestamp={time.time()}, user={os.getenv('USER', 'unknown')}"
        )


# Backward compatibility helper
def get_authentication_token() -> str:
    """Get authentication token using secrets manager with fallback."""
    manager = SecretsManager()
    try:
        return manager.get_secret('FINCHAT_TOKEN')
    except SecretNotFoundError:
        logger.error("FINCHAT_TOKEN not found in any secrets provider")
        return ''