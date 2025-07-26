#!/usr/bin/env python3
"""Test Edgar client refactoring functionality."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_base_class_functionality():
    """Test that base class functionality works correctly."""
    try:
        from finchat_sec_qa.edgar_client import BaseEdgarClient, EdgarClient, AsyncEdgarClient
        
        # Test base class exists
        print("✅ BaseEdgarClient class exists")
        
        # Test inheritance
        assert issubclass(EdgarClient, BaseEdgarClient)
        assert issubclass(AsyncEdgarClient, BaseEdgarClient)
        print("✅ Both clients inherit from BaseEdgarClient")
        
        # Test shared validation methods exist
        base_methods = ['_validate_ticker', '_validate_cik', '_validate_accession_number', '_validate_user_agent', '_setup_cache_dir']
        for method in base_methods:
            assert hasattr(BaseEdgarClient, method)
        print("✅ All validation methods exist in BaseEdgarClient")
        
        # Test clients have access to validation methods
        from unittest.mock import patch
        
        with patch('requests.Session'):
            client = EdgarClient(user_agent="test-agent")
            assert hasattr(client, '_validate_ticker')
            assert hasattr(client, '_validate_cik')
            print("✅ EdgarClient has access to inherited validation methods")
        
        with patch('httpx.AsyncClient'):
            async_client = AsyncEdgarClient(user_agent="test-agent")
            assert hasattr(async_client, '_validate_ticker')
            assert hasattr(async_client, '_validate_cik')
            print("✅ AsyncEdgarClient has access to inherited validation methods")
            
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def test_validation_functionality():
    """Test that validation methods work correctly."""
    try:
        from finchat_sec_qa.edgar_client import BaseEdgarClient
        from unittest.mock import patch
        
        base = BaseEdgarClient()
        
        # Test ticker validation (should work with inherited edgar_validation functions)
        try:
            result = base._validate_ticker("AAPL")
            assert result == "AAPL"
            print("✅ Ticker validation works correctly")
        except Exception as e:
            print(f"❌ Ticker validation failed: {e}")
            return False
            
        # Test user agent validation
        try:
            base._validate_user_agent("test-agent")  # Should not raise
            print("✅ User agent validation works correctly")
        except Exception as e:
            print(f"❌ User agent validation failed: {e}")
            return False
            
        # Test empty user agent should fail
        try:
            base._validate_user_agent("")
            print("❌ Empty user agent should have failed")
            return False
        except ValueError:
            print("✅ Empty user agent correctly rejected")
            
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

if __name__ == "__main__":
    print("Testing Edgar client refactoring...")
    
    test1 = test_base_class_functionality()
    test2 = test_validation_functionality()
    
    if test1 and test2:
        print("\n🎉 All refactoring tests passed! Code duplication successfully eliminated.")
        sys.exit(0)  
    else:
        print("\n❌ Some tests failed.")
        sys.exit(1)