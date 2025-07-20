"""Tests for resource management improvements."""
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


def test_webapp_resource_cleanup():
    """Test that webapp properly cleans up resources on shutdown."""
    from finchat_sec_qa.webapp import app
    
    with app.app_context():
        # Test that teardown handlers are registered
        teardown_funcs = app.teardown_appcontext_funcs
        assert len(teardown_funcs) > 0, "No teardown handlers registered"


def test_engine_save_on_teardown():
    """Test that QA engine is saved during application teardown."""
    from finchat_sec_qa.webapp import app
    
    with patch('finchat_sec_qa.webapp.engine') as mock_engine:
        mock_engine.save = MagicMock()
        
        with app.app_context():
            # Trigger teardown
            pass
        
        # Verify save was called during teardown
        mock_engine.save.assert_called()


def test_no_atexit_usage():
    """Test that atexit is no longer used for resource cleanup."""
    # Simply check that atexit.register is not called in webapp
    import inspect
    import finchat_sec_qa.webapp as webapp_module
    
    # Get the source code of the webapp module
    source = inspect.getsource(webapp_module)
    
    # Should not contain atexit.register calls
    assert 'atexit.register' not in source, "webapp still uses atexit.register"


def test_flask_teardown_handler_registration():
    """Test that Flask teardown handlers are properly registered."""
    from finchat_sec_qa.webapp import app
    
    # Check that teardown handlers are registered
    assert hasattr(app, 'teardown_appcontext_funcs')
    assert len(app.teardown_appcontext_funcs) > 0
    
    # Verify our cleanup function is registered
    teardown_func_names = [func.__name__ for func in app.teardown_appcontext_funcs if hasattr(func, '__name__')]
    assert 'cleanup_resources' in teardown_func_names or any('cleanup' in name for name in teardown_func_names)


def test_resource_cleanup_error_handling():
    """Test that resource cleanup handles errors gracefully."""
    from finchat_sec_qa.webapp import app
    
    with patch('finchat_sec_qa.webapp.engine') as mock_engine:
        # Make save() raise an exception
        mock_engine.save.side_effect = Exception("Save failed")
        
        # Should not raise exception during teardown
        with app.app_context():
            pass  # Teardown happens automatically
        
        # Verify save was attempted despite error
        mock_engine.save.assert_called()


def test_webapp_context_manager_pattern():
    """Test that webapp can be used as a context manager for testing."""
    from finchat_sec_qa.webapp import app
    
    # Test the app context manager pattern
    with app.app_context():
        assert app._get_current_object() is not None
    
    # Context should be cleaned up after exiting