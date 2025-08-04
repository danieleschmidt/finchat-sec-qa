"""Centralized configuration management for FinChat SEC QA.

This module provides a centralized way to manage all configuration values,
supporting environment variable overrides and validation.
"""
import logging
import os
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from .secrets_manager import SecretsManager

logger = logging.getLogger(__name__)


class Config:
    """Central configuration class with environment variable support.
    
    All configuration values can be overridden via environment variables
    with the FINCHAT_ prefix. This follows the Twelve-Factor App principle
    of storing config in the environment.
    """

    def __init__(self) -> None:
        """Initialize configuration with defaults and environment overrides."""
        # Rate limiting configuration
        self.RATE_LIMIT_MAX_REQUESTS = self._get_int_env(
            'FINCHAT_RATE_LIMIT_MAX_REQUESTS', 100
        )
        self.RATE_LIMIT_WINDOW_SECONDS = self._get_int_env(
            'FINCHAT_RATE_LIMIT_WINDOW_SECONDS', 3600
        )

        # Authentication configuration
        self.MIN_TOKEN_LENGTH = self._get_int_env(
            'FINCHAT_MIN_TOKEN_LENGTH', 16
        )
        self.MIN_PASSWORD_CRITERIA = self._get_int_env(
            'FINCHAT_MIN_PASSWORD_CRITERIA', 3
        )
        self.FAILED_ATTEMPTS_LOCKOUT_THRESHOLD = self._get_int_env(
            'FINCHAT_FAILED_ATTEMPTS_LOCKOUT_THRESHOLD', 3
        )
        self.LOCKOUT_DURATION_SECONDS = self._get_int_env(
            'FINCHAT_LOCKOUT_DURATION_SECONDS', 3600
        )

        # API validation limits
        self.MAX_QUESTION_LENGTH = self._get_int_env(
            'FINCHAT_MAX_QUESTION_LENGTH', 1000
        )
        self.MAX_TICKER_LENGTH = self._get_int_env(
            'FINCHAT_MAX_TICKER_LENGTH', 5
        )
        self.MAX_FORM_TYPE_LENGTH = self._get_int_env(
            'FINCHAT_MAX_FORM_TYPE_LENGTH', 10
        )
        self.MAX_TEXT_INPUT_LENGTH = self._get_int_env(
            'FINCHAT_MAX_TEXT_INPUT_LENGTH', 50000
        )

        # Text chunking configuration
        self.CHUNK_SIZE = self._get_int_env(
            'FINCHAT_CHUNK_SIZE', 1000  # Characters per chunk
        )
        self.CHUNK_OVERLAP = self._get_int_env(
            'FINCHAT_CHUNK_OVERLAP', 200  # Overlap between chunks
        )

        # Security headers
        self.HSTS_MAX_AGE = self._get_int_env(
            'FINCHAT_HSTS_MAX_AGE', 31536000  # 1 year
        )
        self.XSS_PROTECTION_MODE = os.getenv(
            'FINCHAT_XSS_PROTECTION_MODE', '1; mode=block'
        )

        # Exponential backoff configuration
        self.EXPONENTIAL_BACKOFF_BASE = self._get_int_env(
            'FINCHAT_EXPONENTIAL_BACKOFF_BASE', 2
        )
        self.EXPONENTIAL_BACKOFF_UNIT_SECONDS = self._get_int_env(
            'FINCHAT_EXPONENTIAL_BACKOFF_UNIT_SECONDS', 60
        )

        # Authentication token (with secrets manager support)
        self.secrets_manager: Optional[SecretsManager] = None
        self._init_secrets_manager()
        self.SECRET_TOKEN = self._get_secure_token()

        # Redis configuration for distributed rate limiting
        self.REDIS_URL = os.getenv('FINCHAT_REDIS_URL', 'redis://localhost:6379/0')
        self.REDIS_POOL_MAX_CONNECTIONS = self._get_int_env(
            'FINCHAT_REDIS_POOL_MAX_CONNECTIONS', 20  # Connection pool size
        )

        # CORS configuration
        self.CORS_ALLOWED_ORIGINS = self._get_list_env(
            'FINCHAT_CORS_ALLOWED_ORIGINS',
            ['http://localhost:3000', 'http://localhost:8080']  # Default for development
        )
        self.CORS_ALLOW_CREDENTIALS = self._get_bool_env(
            'FINCHAT_CORS_ALLOW_CREDENTIALS', False
        )
        self.CORS_MAX_AGE = self._get_int_env(
            'FINCHAT_CORS_MAX_AGE', 86400  # 24 hours
        )

        # Security configuration
        self.MAX_REQUEST_SIZE_MB = self._get_int_env(
            'FINCHAT_MAX_REQUEST_SIZE_MB', 1  # 1MB default
        )
        self.CSRF_TOKEN_EXPIRY_SECONDS = self._get_int_env(
            'FINCHAT_CSRF_TOKEN_EXPIRY_SECONDS', 1800  # 30 minutes
        )

        # Cache size limits to prevent memory leaks
        self.CSRF_MAX_CACHE_SIZE = self._get_int_env(
            'FINCHAT_CSRF_MAX_CACHE_SIZE', 1000  # Max CSRF tokens in memory
        )
        self.RATE_LIMIT_MAX_FALLBACK_CACHE_SIZE = self._get_int_env(
            'FINCHAT_RATE_LIMIT_MAX_FALLBACK_CACHE_SIZE', 10000  # Max fallback clients
        )

        # Validate configuration
        self._validate()

    def _get_int_env(self, key: str, default: int) -> int:
        """Get integer value from environment variable with default."""
        value = os.getenv(key)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            raise ValueError(f"Environment variable {key} must be a valid integer, got: {value}")

    def _get_bool_env(self, key: str, default: bool) -> bool:
        """Get boolean value from environment variable with default."""
        value = os.getenv(key)
        if value is None:
            return default
        return value.lower() in ('true', '1', 'yes', 'on')

    def _get_list_env(self, key: str, default: List[str]) -> List[str]:
        """Get list value from environment variable with default."""
        value = os.getenv(key)
        if value is None:
            return default
        # Split by comma and strip whitespace
        return [item.strip() for item in value.split(',') if item.strip()]

    def _init_secrets_manager(self) -> None:
        """Initialize secrets manager based on environment configuration."""
        from .secrets_manager import SecretsManager

        # Determine provider from environment
        provider = os.getenv('FINCHAT_SECRETS_PROVIDER', 'env')

        # Provider-specific configuration
        provider_config = {}
        if provider == 'aws':
            provider_config['region'] = os.getenv('FINCHAT_AWS_REGION', 'us-east-1')
        elif provider == 'vault':
            provider_config['vault_url'] = os.getenv('FINCHAT_VAULT_URL', 'http://localhost:8200')
            provider_config['vault_token'] = os.getenv('FINCHAT_VAULT_TOKEN')

        # Encryption key for local storage
        encryption_key = os.getenv('FINCHAT_ENCRYPTION_KEY')

        # Fallback chain
        fallback_providers = os.getenv('FINCHAT_SECRETS_FALLBACKS', 'env').split(',')

        self.secrets_manager = SecretsManager(
            provider=provider,
            encryption_key=encryption_key,
            fallback_providers=fallback_providers,
            cache_ttl=int(os.getenv('FINCHAT_SECRETS_CACHE_TTL', '300')),
            **provider_config
        )

    def _get_secure_token(self) -> Optional[str]:
        """Get authentication token using secrets manager."""
        try:
            return self.secrets_manager.get_secret('FINCHAT_TOKEN')
        except (KeyError, ValueError) as e:
            # Fallback to direct environment variable for backward compatibility
            logger.debug(f"Secret not found in secrets manager, using env var: {e}")
            return os.getenv('FINCHAT_TOKEN')
        except Exception as e:
            # Unexpected errors should be logged but still fallback
            logger.warning(f"Unexpected error accessing secrets manager: {e}")
            return os.getenv('FINCHAT_TOKEN')

    def _validate(self) -> None:
        """Validate configuration values."""
        if self.RATE_LIMIT_MAX_REQUESTS <= 0:
            raise ValueError("RATE_LIMIT_MAX_REQUESTS must be positive")

        if self.RATE_LIMIT_WINDOW_SECONDS <= 0:
            raise ValueError("RATE_LIMIT_WINDOW_SECONDS must be positive")

        if self.MIN_TOKEN_LENGTH <= 0:
            raise ValueError("MIN_TOKEN_LENGTH must be positive")

        if self.MIN_PASSWORD_CRITERIA <= 0:
            raise ValueError("MIN_PASSWORD_CRITERIA must be positive")

        if self.FAILED_ATTEMPTS_LOCKOUT_THRESHOLD <= 0:
            raise ValueError("FAILED_ATTEMPTS_LOCKOUT_THRESHOLD must be positive")

        if self.LOCKOUT_DURATION_SECONDS <= 0:
            raise ValueError("LOCKOUT_DURATION_SECONDS must be positive")

        if self.MAX_QUESTION_LENGTH <= 0:
            raise ValueError("MAX_QUESTION_LENGTH must be positive")

        if self.MAX_TICKER_LENGTH <= 0:
            raise ValueError("MAX_TICKER_LENGTH must be positive")

        if self.MAX_FORM_TYPE_LENGTH <= 0:
            raise ValueError("MAX_FORM_TYPE_LENGTH must be positive")

        if self.MAX_TEXT_INPUT_LENGTH <= 0:
            raise ValueError("MAX_TEXT_INPUT_LENGTH must be positive")

        if self.CHUNK_SIZE <= 0:
            raise ValueError("CHUNK_SIZE must be positive")

        if self.CHUNK_OVERLAP < 0:
            raise ValueError("CHUNK_OVERLAP must be non-negative")

        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError("CHUNK_OVERLAP must be less than CHUNK_SIZE")

        # CORS validation
        if not isinstance(self.CORS_ALLOWED_ORIGINS, list):
            raise ValueError("CORS_ALLOWED_ORIGINS must be a list")

        if self.CORS_MAX_AGE < 0:
            raise ValueError("CORS_MAX_AGE must be non-negative")

        # Security check: warn if wildcard with credentials
        if '*' in self.CORS_ALLOWED_ORIGINS and self.CORS_ALLOW_CREDENTIALS:
            raise ValueError(
                "Security risk: CORS_ALLOW_CREDENTIALS cannot be True when wildcard '*' is in allowed origins"
            )


# Singleton instance for consistent configuration across the application
_config_instance: Optional[Config] = None


def get_config() -> Config:
    """Get the singleton configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance


def reset_config() -> None:
    """Reset the singleton instance (mainly for testing)."""
    global _config_instance
    _config_instance = None
