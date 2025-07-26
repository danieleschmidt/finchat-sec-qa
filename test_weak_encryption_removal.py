#!/usr/bin/env python3
"""Test script to verify weak encryption fallback removal."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_no_weak_encryption_fallback():
    """Test that there's no fallback to weak encryption."""
    from unittest.mock import patch
    
    # Mock the AES import to fail
    with patch('finchat_sec_qa.secrets_manager.AESGCM') as mock_aes:
        mock_aes.side_effect = ImportError("No module named 'cryptography'")
        
        try:
            from finchat_sec_qa.secrets_manager import SecretsManager
            
            key_32_bytes = 'a' * 32
            manager = SecretsManager(provider='local', encryption_key=key_32_bytes)
            
            # This should raise ImportError instead of falling back
            try:
                manager.store_secret('test_secret', 'test_value')
                print("❌ FAIL: Should have raised ImportError for missing cryptography")
                return False
            except ImportError as e:
                if "AES-GCM encryption requires the cryptography library" in str(e):
                    print("✅ PASS: Correctly raises ImportError when cryptography is missing")
                    return True
                else:
                    print(f"❌ FAIL: Wrong ImportError message: {e}")
                    return False
            except Exception as e:
                print(f"❌ FAIL: Unexpected exception: {e}")
                return False
                
        except ImportError as e:
            print(f"❌ Cannot import module for testing: {e}")
            return False

def test_legacy_methods_removed():
    """Test that legacy methods are no longer available."""
    try:
        from finchat_sec_qa.secrets_manager import SecretsManager
        
        manager = SecretsManager()
        
        # Check that legacy methods don't exist
        legacy_methods = ['_encrypt_value_legacy', '_decrypt_value_legacy']
        
        for method in legacy_methods:
            if hasattr(manager, method):
                print(f"❌ FAIL: Legacy method {method} still exists")
                return False
        
        print("✅ PASS: All legacy encryption methods have been removed")
        return True
        
    except ImportError as e:
        print(f"❌ Cannot import module for testing: {e}")
        return False

def test_only_v2_format_supported():
    """Test that only v2 encryption format is supported for decryption."""
    import base64
    
    try:
        from finchat_sec_qa.secrets_manager import SecretsManager
        
        key_32_bytes = 'a' * 32
        manager = SecretsManager(provider='local', encryption_key=key_32_bytes)
        
        # Create fake legacy-format encrypted data (without v2 prefix)
        fake_legacy_data = base64.b64encode(b'fake_legacy_encrypted_data').decode('ascii')
        
        try:
            manager._decrypt_value(fake_legacy_data)
            print("❌ FAIL: Should not decrypt legacy format data")
            return False
        except ValueError as e:
            if "Unsupported encryption format" in str(e):
                print("✅ PASS: Correctly rejects legacy encryption format")
                return True
            else:
                print(f"❌ FAIL: Wrong error message: {e}")
                return False
        except Exception as e:
            print(f"❌ FAIL: Unexpected exception: {e}")
            return False
            
    except ImportError as e:
        print(f"❌ Cannot import module for testing: {e}")
        return False

if __name__ == "__main__":
    print("Testing weak encryption fallback removal...")
    
    test1 = test_no_weak_encryption_fallback()
    test2 = test_legacy_methods_removed()  
    test3 = test_only_v2_format_supported()
    
    if test1 and test2 and test3:
        print("\n🎉 All tests passed! Weak encryption fallback successfully removed.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed.")
        sys.exit(1)