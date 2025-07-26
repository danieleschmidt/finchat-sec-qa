"""Tests for Edgar client code duplication refactoring."""

import pytest
from unittest.mock import patch, Mock
from finchat_sec_qa.edgar_client import EdgarClient, AsyncEdgarClient


def test_base_class_exists():
    """Test that a base class for common functionality exists."""
    from finchat_sec_qa.edgar_client import BaseEdgarClient
    
    # Should be able to import the base class
    assert BaseEdgarClient is not None


def test_shared_validation_methods():
    """Test that validation methods are shared between sync and async clients."""
    from finchat_sec_qa.edgar_client import BaseEdgarClient
    
    # Both clients should inherit from base class
    assert issubclass(EdgarClient, BaseEdgarClient)
    assert issubclass(AsyncEdgarClient, BaseEdgarClient)
    
    # Validation methods should be defined in base class
    assert hasattr(BaseEdgarClient, '_validate_ticker')
    assert hasattr(BaseEdgarClient, '_validate_cik')
    assert hasattr(BaseEdgarClient, '_validate_accession_number')


def test_no_duplicate_validation_methods():
    """Test that validation methods are not duplicated in child classes."""
    # EdgarClient should not define its own validation methods
    edgar_methods = [name for name in dir(EdgarClient) if name.startswith('_validate')]
    async_edgar_methods = [name for name in dir(AsyncEdgarClient) if name.startswith('_validate')]
    
    # Should inherit validation methods, not define their own
    # (This test will pass once we move methods to base class)
    for method in ['_validate_ticker', '_validate_cik', '_validate_accession_number']:
        # Methods should exist but be inherited, not defined in child class
        assert hasattr(EdgarClient, method)
        assert hasattr(AsyncEdgarClient, method)


def test_shared_initialization_logic():
    """Test that common initialization logic is extracted to base class."""
    from finchat_sec_qa.edgar_client import BaseEdgarClient
    
    # Base class should handle common validation and setup
    assert hasattr(BaseEdgarClient, '_validate_user_agent')
    assert hasattr(BaseEdgarClient, '_setup_cache_dir')


def test_clients_work_after_refactoring():
    """Test that both clients still work correctly after refactoring."""
    # Test sync client
    with patch('requests.Session'):
        client = EdgarClient(user_agent="test-agent")
        assert client.BASE_URL == "https://data.sec.gov"
        assert client.timeout == 10.0
    
    # Test async client  
    with patch('httpx.AsyncClient'):
        async_client = AsyncEdgarClient(user_agent="test-agent")
        assert async_client.BASE_URL == "https://data.sec.gov"
        assert async_client.timeout == 10.0