# Encryption Upgrade Documentation

## Overview

The secrets management system has been upgraded from insecure XOR encryption to industry-standard AES-GCM authenticated encryption.

## Security Improvements

### Before (Vulnerable)
- **XOR encryption** with SHA256-derived key
- **No authentication** - data integrity not verified
- **Deterministic** - same plaintext always produces same ciphertext
- **Vulnerable to known-plaintext attacks**

### After (Secure)
- **AES-GCM authenticated encryption** (industry standard)
- **PBKDF2 key derivation** with 100,000 iterations
- **Random IV per encryption** - same plaintext produces different ciphertext
- **Authenticated encryption** - detects tampering automatically
- **Version-prefixed format** for future upgrades

## Backward Compatibility

The system maintains full backward compatibility:
- Legacy XOR-encrypted secrets are automatically detected and decrypted
- New secrets are encrypted with AES-GCM (v2 format)
- No manual migration required

## Format Specification

### V2 Format (AES-GCM)
```
Base64( version_prefix(2) + IV(12) + AES-GCM-ciphertext-with-tag(variable) )
```

- **version_prefix**: `b'v2'` (2 bytes)
- **IV**: Random 12-byte initialization vector
- **ciphertext**: AES-GCM encrypted data with 16-byte authentication tag
- **AAD**: `b'finchat_secrets_v2'` for additional authentication

### Legacy Format (XOR)
```
Base64( XOR-encrypted-bytes )
```

## Key Derivation

```python
PBKDF2-HMAC-SHA256(
    password=user_provided_key,
    salt=b'finchat_secrets_salt_v1',
    iterations=100000,
    key_length=32  # AES-256
)
```

## Migration Strategy

1. **Automatic Detection**: Format detected by version prefix
2. **Lazy Migration**: Secrets upgraded on next write operation
3. **Zero Downtime**: No service interruption required
4. **Rollback Safe**: Legacy format still supported if needed

## Security Properties

- ✅ **Confidentiality**: AES-256 encryption
- ✅ **Integrity**: GCM authentication tag
- ✅ **Authenticity**: Prevents tampering
- ✅ **Semantic Security**: Random IV prevents pattern analysis
- ✅ **Forward Security**: Key derivation with high iteration count
- ✅ **Timing Attack Resistance**: Constant-time error handling

## Dependencies

- **cryptography**: Industry-standard Python cryptography library
- **Fallback**: Graceful degradation to legacy XOR if library unavailable

## Performance Impact

- **Encryption**: ~2-5ms additional overhead per operation
- **Key Derivation**: ~50-100ms (cached for session)
- **Memory**: Minimal additional memory usage

## Testing

Comprehensive test suite covers:
- Different ciphertexts for same plaintext
- Tampering detection
- Legacy format compatibility
- Unicode handling
- Large value encryption
- Timing attack resistance

## Rollback Plan

If issues arise:
1. All legacy secrets continue working without changes
2. Remove `cryptography` dependency to force XOR fallback
3. New secrets will use legacy format until issue resolved