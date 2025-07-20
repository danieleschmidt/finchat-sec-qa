"""Tests for centralized configuration system."""
import os
import pytest
from unittest.mock import patch


def test_config_module_imports():
    """Test that config module can be imported."""
    from finchat_sec_qa.config import Config
    assert Config is not None


def test_default_config_values():
    """Test that default configuration values are set correctly."""
    from finchat_sec_qa.config import Config
    
    config = Config()
    
    # Rate limiting defaults
    assert config.RATE_LIMIT_MAX_REQUESTS == 100
    assert config.RATE_LIMIT_WINDOW_SECONDS == 3600
    
    # Authentication defaults
    assert config.MIN_TOKEN_LENGTH == 16
    assert config.MIN_PASSWORD_CRITERIA == 3
    assert config.FAILED_ATTEMPTS_LOCKOUT_THRESHOLD == 3
    assert config.LOCKOUT_DURATION_SECONDS == 3600
    
    # API validation defaults
    assert config.MAX_QUESTION_LENGTH == 1000
    assert config.MAX_TICKER_LENGTH == 5
    assert config.MAX_FORM_TYPE_LENGTH == 10
    assert config.MAX_TEXT_INPUT_LENGTH == 50000


def test_environment_variable_override():
    """Test that environment variables override default values."""
    from finchat_sec_qa.config import Config
    
    with patch.dict(os.environ, {
        'FINCHAT_RATE_LIMIT_MAX_REQUESTS': '200',
        'FINCHAT_RATE_LIMIT_WINDOW_SECONDS': '7200',
        'FINCHAT_MIN_TOKEN_LENGTH': '32'
    }):
        config = Config()
        assert config.RATE_LIMIT_MAX_REQUESTS == 200
        assert config.RATE_LIMIT_WINDOW_SECONDS == 7200
        assert config.MIN_TOKEN_LENGTH == 32


def test_config_validation():
    """Test that invalid configuration values raise appropriate errors."""
    from finchat_sec_qa.config import Config
    
    # Test negative values
    with patch.dict(os.environ, {'FINCHAT_RATE_LIMIT_MAX_REQUESTS': '-1'}):
        with pytest.raises(ValueError, match="RATE_LIMIT_MAX_REQUESTS must be positive"):
            Config()
    
    # Test zero values where not allowed
    with patch.dict(os.environ, {'FINCHAT_RATE_LIMIT_WINDOW_SECONDS': '0'}):
        with pytest.raises(ValueError, match="RATE_LIMIT_WINDOW_SECONDS must be positive"):
            Config()


def test_config_singleton_pattern():
    """Test that Config follows singleton pattern for consistency."""
    from finchat_sec_qa.config import get_config
    
    config1 = get_config()
    config2 = get_config()
    
    assert config1 is config2  # Should be the same instance


def test_security_config_values():
    """Test security-related configuration values."""
    from finchat_sec_qa.config import Config
    
    config = Config()
    
    # Security headers
    assert config.HSTS_MAX_AGE == 31536000  # 1 year
    assert config.XSS_PROTECTION_MODE == '1; mode=block'
    
    # Backoff settings
    assert config.EXPONENTIAL_BACKOFF_BASE == 2
    assert config.EXPONENTIAL_BACKOFF_UNIT_SECONDS == 60


def test_config_environment_prefix():
    """Test that all environment variables use consistent FINCHAT_ prefix."""
    from finchat_sec_qa.config import Config
    
    # Test with complete environment override
    env_vars = {
        'FINCHAT_RATE_LIMIT_MAX_REQUESTS': '50',
        'FINCHAT_RATE_LIMIT_WINDOW_SECONDS': '1800',
        'FINCHAT_MIN_TOKEN_LENGTH': '20',
        'FINCHAT_TOKEN': 'test-token-value-123456'
    }
    
    with patch.dict(os.environ, env_vars):
        config = Config()
        assert config.RATE_LIMIT_MAX_REQUESTS == 50
        assert config.RATE_LIMIT_WINDOW_SECONDS == 1800
        assert config.MIN_TOKEN_LENGTH == 20
        assert config.SECRET_TOKEN == 'test-token-value-123456'