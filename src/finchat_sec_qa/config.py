"""Centralized configuration management for FinChat SEC QA.

This module provides a centralized way to manage all configuration values,
supporting environment variable overrides and validation.
"""
import os
from typing import Optional


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
        
        # Authentication token
        self.SECRET_TOKEN = os.getenv('FINCHAT_TOKEN')
        
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