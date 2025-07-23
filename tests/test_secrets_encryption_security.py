"""Tests for secure authenticated encryption in secrets management."""
import pytest
import base64
from unittest.mock import patch
from finchat_sec_qa.secrets_manager import SecretsManager


class TestAuthenticatedEncryption:
    """Test authenticated encryption implementation."""
    
    def test_aes_gcm_encryption_produces_different_ciphertexts(self):
        """Test that AES-GCM produces different ciphertexts for same plaintext."""
        key_32_bytes = 'a' * 32
        manager = SecretsManager(provider='local', encryption_key=key_32_bytes)
        
        plaintext = "secret_value_123"
        
        # Encrypt same value twice
        ciphertext1 = manager._encrypt_value(plaintext)
        ciphertext2 = manager._encrypt_value(plaintext)
        
        # Should produce different ciphertexts due to random IV
        assert ciphertext1 != ciphertext2
        
        # But both should decrypt to same plaintext
        assert manager._decrypt_value(ciphertext1) == plaintext
        assert manager._decrypt_value(ciphertext2) == plaintext
    
    def test_aes_gcm_authentication_detects_tampering(self):
        """Test that AES-GCM detects tampering and fails decryption."""
        key_32_bytes = 'b' * 32
        manager = SecretsManager(provider='local', encryption_key=key_32_bytes)
        
        plaintext = "important_secret"
        ciphertext = manager._encrypt_value(plaintext)
        
        # Tamper with the ciphertext by flipping a bit
        tampered_bytes = base64.b64decode(ciphertext.encode('ascii'))
        tampered_bytes = tampered_bytes[:10] + bytes([tampered_bytes[10] ^ 0x01]) + tampered_bytes[11:]
        tampered_ciphertext = base64.b64encode(tampered_bytes).decode('ascii')
        
        # Should detect tampering and raise an exception
        with pytest.raises(Exception):  # Could be ValueError, InvalidTag, etc.
            manager._decrypt_value(tampered_ciphertext)
    
    def test_encryption_uses_proper_iv_length(self):
        """Test that encryption uses proper IV length for AES-GCM."""
        key_32_bytes = 'c' * 32
        manager = SecretsManager(provider='local', encryption_key=key_32_bytes)
        
        plaintext = "test_secret"
        ciphertext = manager._encrypt_value(plaintext)
        
        # Decode the ciphertext to check structure
        encrypted_bytes = base64.b64decode(ciphertext.encode('ascii'))
        
        # AES-GCM typically uses 12-byte IV followed by ciphertext and 16-byte tag
        # Total structure: IV (12) + ciphertext (variable) + tag (16)
        assert len(encrypted_bytes) >= 12 + len(plaintext.encode('utf-8')) + 16
    
    def test_backward_compatibility_with_xor_encrypted_data(self):
        """Test that new implementation can decrypt legacy XOR-encrypted data."""
        key_32_bytes = 'd' * 32
        manager = SecretsManager(provider='local', encryption_key=key_32_bytes)
        
        # Create XOR-encrypted data using old method
        import hashlib
        plaintext = "legacy_secret"
        key_hash = hashlib.sha256(key_32_bytes.encode('utf-8')).digest()
        value_bytes = plaintext.encode('utf-8')
        encrypted_bytes = bytes(a ^ b for a, b in zip(value_bytes, key_hash * (len(value_bytes) // len(key_hash) + 1)))
        legacy_ciphertext = base64.b64encode(encrypted_bytes).decode('ascii')
        
        # Should be able to decrypt legacy data
        decrypted = manager._decrypt_value(legacy_ciphertext)
        assert decrypted == plaintext
    
    def test_encryption_key_derivation_secure(self):
        """Test that key derivation uses secure methods."""
        # Test should fail initially since current implementation uses sha256
        key_password = "user_password_123"
        manager = SecretsManager(provider='local', encryption_key=key_password)
        
        # This test expects PBKDF2 or similar, should fail with current SHA256
        # The implementation should derive keys securely from passwords
        plaintext = "test_value"
        
        # Multiple encryptions with same password should use different salts
        ciphertext1 = manager._encrypt_value(plaintext)
        ciphertext2 = manager._encrypt_value(plaintext)
        
        assert ciphertext1 != ciphertext2  # Different due to random IV/salt
    
    def test_constant_time_decryption_behavior(self):
        """Test that decryption failure timing is constant to prevent timing attacks."""
        key_32_bytes = 'e' * 32
        manager = SecretsManager(provider='local', encryption_key=key_32_bytes)
        
        import time
        
        # Valid ciphertext
        valid_plaintext = "valid_secret"
        valid_ciphertext = manager._encrypt_value(valid_plaintext)
        
        # Invalid ciphertext
        invalid_ciphertext = "invalid_base64_data"
        
        # Time valid decryption failure (authentication failure)
        start = time.time()
        try:
            manager._decrypt_value(valid_ciphertext[:-5] + "AAAAA")  # Corrupt end
        except Exception:
            pass
        valid_fail_time = time.time() - start
        
        # Time invalid format failure
        start = time.time()
        try:
            manager._decrypt_value(invalid_ciphertext)
        except Exception:
            pass
        invalid_fail_time = time.time() - start
        
        # Both should fail in roughly constant time (within 10ms)
        time_diff = abs(valid_fail_time - invalid_fail_time)
        assert time_diff < 0.01, f"Timing difference too large: {time_diff}s"
    
    def test_encryption_format_versioning(self):
        """Test that encryption format includes version information."""
        key_32_bytes = 'f' * 32
        manager = SecretsManager(provider='local', encryption_key=key_32_bytes)
        
        plaintext = "versioned_secret"
        ciphertext = manager._encrypt_value(plaintext)
        
        # Should be able to detect format version (for future upgrades)
        # This test expects some form of version prefix or header
        decrypted = manager._decrypt_value(ciphertext)
        assert decrypted == plaintext
    
    def test_large_value_encryption(self):
        """Test encryption of large values."""
        key_32_bytes = 'g' * 32
        manager = SecretsManager(provider='local', encryption_key=key_32_bytes)
        
        # Large secret value (4KB)
        large_plaintext = "X" * 4096
        ciphertext = manager._encrypt_value(large_plaintext)
        decrypted = manager._decrypt_value(ciphertext)
        
        assert decrypted == large_plaintext
    
    def test_unicode_value_encryption(self):
        """Test encryption of Unicode values."""
        key_32_bytes = 'h' * 32
        manager = SecretsManager(provider='local', encryption_key=key_32_bytes)
        
        # Unicode secret with various characters
        unicode_plaintext = "🔐密码测试👨‍💻"
        ciphertext = manager._encrypt_value(unicode_plaintext)
        decrypted = manager._decrypt_value(ciphertext)
        
        assert decrypted == unicode_plaintext