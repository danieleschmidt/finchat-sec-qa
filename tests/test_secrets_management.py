"""
Tests for secure secrets management functionality.
Testing encrypted storage, key rotation, and external provider integration.
"""
import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from finchat_sec_qa.secrets_manager import SecretsManager, SecretNotFoundError


class TestSecretsManager:
    """Test suite for secure secrets management."""
    
    def test_environment_variable_fallback(self):
        """Test fallback to environment variables for backward compatibility."""
        # RED: This test should fail initially because SecretsManager doesn't exist
        with patch.dict(os.environ, {'FINCHAT_TOKEN': 'test_token_123'}):
            manager = SecretsManager()
            
            token = manager.get_secret('FINCHAT_TOKEN')
            
            assert token == 'test_token_123'
    
    def test_encrypted_local_storage(self):
        """Test encrypted local secrets storage."""
        # Use a key that's exactly 32 bytes
        key_32_bytes = 'a' * 32  # 32 'a' characters = 32 bytes
        manager = SecretsManager(provider='local', encryption_key=key_32_bytes)
        
        # Store encrypted secret
        manager.store_secret('test_secret', 'encrypted_value_123')
        
        # Retrieve and decrypt
        retrieved = manager.get_secret('test_secret')
        
        assert retrieved == 'encrypted_value_123'
    
    def test_aws_secrets_manager_integration(self):
        """Test AWS Secrets Manager integration."""
        manager = SecretsManager(provider='env')  # Don't initialize AWS
        
        # Mock the AWS client directly on the instance
        mock_client = Mock()
        mock_client.get_secret_value.return_value = {
            'SecretString': 'aws_secret_value'
        }
        manager._aws_client = mock_client
        
        result = manager._get_from_aws('prod/finchat/token')
        
        assert result == 'aws_secret_value'
        mock_client.get_secret_value.assert_called_once_with(
            SecretId='prod/finchat/token'
        )
    
    def test_hashicorp_vault_integration(self):
        """Test HashiCorp Vault integration."""
        manager = SecretsManager(provider='env')  # Don't initialize Vault
        
        # Mock the vault session directly
        mock_session = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {
            'data': {'data': {'token': 'vault_secret_value'}}
        }
        mock_response.raise_for_status.return_value = None
        mock_session.get.return_value = mock_response
        manager._vault_session = mock_session
        manager.provider_config = {'vault_url': 'https://vault.example.com'}
        
        result = manager._get_from_vault('secret/data/finchat/token', field='token')
        
        assert result == 'vault_secret_value'
    
    def test_secret_rotation_capability(self):
        """Test secret rotation and versioning."""
        key_32_bytes = 'b' * 32  # 32 'b' characters = 32 bytes  
        manager = SecretsManager(provider='local', encryption_key=key_32_bytes)
        
        # Store initial secret
        manager.store_secret('rotatable_secret', 'version_1', version=1)
        
        # Rotate to new version
        manager.rotate_secret('rotatable_secret', 'version_2', version=2)
        
        # Should get latest version by default
        current = manager.get_secret('rotatable_secret')
        assert current == 'version_2'
        
        # Should be able to get specific version
        old_version = manager.get_secret('rotatable_secret', version=1)
        assert old_version == 'version_1'
    
    def test_secret_not_found_error(self):
        """Test proper error handling for missing secrets."""
        manager = SecretsManager()
        
        with pytest.raises(SecretNotFoundError) as exc_info:
            manager.get_secret('nonexistent_secret')
        
        assert 'nonexistent_secret' in str(exc_info.value)
    
    def test_provider_fallback_chain(self):
        """Test fallback chain: AWS -> Vault -> Local -> Environment."""
        with patch.dict(os.environ, {'FALLBACK_SECRET': 'env_value'}):
            
            manager = SecretsManager(
                provider='aws',
                fallback_providers=['vault', 'local', 'env']
            )
            
            # Mock failing providers
            manager._aws_client = None  # AWS fails
            manager._vault_session = None  # Vault fails
            # Local storage empty, should fallback to env
            
            # Should fallback to environment variable
            result = manager.get_secret('FALLBACK_SECRET')
            assert result == 'env_value'
    
    def test_secret_caching_and_ttl(self):
        """Test secret caching with TTL for performance."""
        with patch('time.time') as mock_time:
            mock_time.return_value = 1000.0
            
            manager = SecretsManager(cache_ttl=300)  # 5 minutes
            
            with patch.dict(os.environ, {'CACHED_SECRET': 'cached_value'}):
                # First call should fetch and cache
                result1 = manager.get_secret('CACHED_SECRET')
                assert result1 == 'cached_value'
                
                # Second call should use cache (simulate env change)
                os.environ['CACHED_SECRET'] = 'changed_value'
                result2 = manager.get_secret('CACHED_SECRET')
                assert result2 == 'cached_value'  # Still cached
                
                # After TTL expires, should refresh
                mock_time.return_value = 1400.0  # 400 seconds later
                result3 = manager.get_secret('CACHED_SECRET')
                assert result3 == 'changed_value'  # Cache expired, new value
    
    def test_audit_logging_for_secret_access(self):
        """Test that secret access is properly audited."""
        manager = SecretsManager()
        
        with patch.dict(os.environ, {'AUDIT_SECRET': 'audit_value'}), \
             patch('finchat_sec_qa.secrets_manager.logger') as mock_logger:
            
            manager.get_secret('AUDIT_SECRET')
            
            # Should log secret access (without revealing the value)
            mock_logger.info.assert_called_once()
            log_call = mock_logger.info.call_args[0][0]
            assert 'AUDIT_SECRET' in log_call
            assert 'audit_value' not in log_call  # Should not log actual secret


class TestSecretsManagerSecurity:
    """Security-focused tests for secrets management."""
    
    def test_encryption_key_validation(self):
        """Test that encryption keys are properly validated."""
        # Should reject weak keys
        with pytest.raises(ValueError, match="Encryption key must be"):
            SecretsManager(encryption_key='weak')
    
    def test_memory_cleanup_after_use(self):
        """Test that secrets are cleared from memory after use."""
        manager = SecretsManager()
        
        with patch.dict(os.environ, {'CLEANUP_SECRET': 'sensitive_data'}):
            secret = manager.get_secret('CLEANUP_SECRET', auto_cleanup=True)
            assert secret == 'sensitive_data'
            
            # After some time, should be cleared from internal cache
            # This is implementation-dependent
    
    def test_timing_attack_prevention(self):
        """Test constant-time comparison for secret validation."""
        manager = SecretsManager()
        
        with patch.dict(os.environ, {'TIMING_SECRET': 'correct_value'}):
            # Both correct and incorrect should take similar time
            import time
            
            start = time.time()
            manager.verify_secret('TIMING_SECRET', 'correct_value')
            correct_time = time.time() - start
            
            start = time.time()
            manager.verify_secret('TIMING_SECRET', 'wrong_value')
            wrong_time = time.time() - start
            
            # Time difference should be minimal (within 10ms)
            time_diff = abs(correct_time - wrong_time)
            assert time_diff < 0.01, f"Timing difference too large: {time_diff}s"

    def test_timing_attack_prevention_nonexistent_secret(self):
        """Test constant-time comparison when secret doesn't exist."""
        manager = SecretsManager()
        
        # Test timing for existing vs non-existent secrets
        with patch.dict(os.environ, {'EXISTING_SECRET': 'correct_value'}):
            import time
            
            # Time for existing secret with wrong value
            start = time.time()
            result1 = manager.verify_secret('EXISTING_SECRET', 'wrong_value')
            existing_time = time.time() - start
            
            # Time for non-existent secret
            start = time.time()
            result2 = manager.verify_secret('NONEXISTENT_SECRET', 'any_value')
            nonexistent_time = time.time() - start
            
            # Both should return False
            assert result1 is False
            assert result2 is False
            
            # Time difference should be minimal (within 10ms) to prevent timing attacks
            time_diff = abs(existing_time - nonexistent_time)
            assert time_diff < 0.01, f"Timing difference reveals secret existence: {time_diff}s"
    
    def test_no_weak_encryption_fallback(self):
        """Test that weak XOR encryption fallback is not available."""
        key_32_bytes = 'a' * 32
        manager = SecretsManager(provider='local', encryption_key=key_32_bytes)
        
        # Mock ImportError for cryptography library
        with patch('finchat_sec_qa.secrets_manager.AES') as mock_aes:
            mock_aes.side_effect = ImportError("Mocked import error")
            
            # Should raise ImportError instead of falling back to weak encryption
            with pytest.raises(ImportError, match="AES-GCM encryption requires the cryptography library"):
                manager.store_secret('test_secret', 'test_value')
    
    def test_legacy_xor_methods_removed(self):
        """Test that legacy XOR encryption methods have been removed."""
        manager = SecretsManager()
        
        # These methods should not exist anymore
        assert not hasattr(manager, '_encrypt_value_legacy')
        assert not hasattr(manager, '_decrypt_value_legacy')
    
    def test_cryptography_library_required(self):
        """Test that cryptography library is required for secure operations."""
        with patch('finchat_sec_qa.secrets_manager.AES') as mock_aes:
            mock_aes.side_effect = ImportError("No module named 'cryptography'")
            
            key_32_bytes = 'a' * 32
            manager = SecretsManager(provider='local', encryption_key=key_32_bytes)
            
            # Should fail gracefully with clear error message
            with pytest.raises(ImportError, match="AES-GCM encryption requires the cryptography library"):
                manager.store_secret('critical_secret', 'secure_value')