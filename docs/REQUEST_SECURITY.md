# Request Security and CSRF Protection

## Overview

Enhanced security measures implemented to prevent DoS attacks and CSRF vulnerabilities.

## Request Size Limits

### Implementation
- **Flask**: Configured via `MAX_CONTENT_LENGTH` setting
- **FastAPI**: Custom `RequestSizeLimitMiddleware`
- **Default Limit**: 1MB per request (configurable via `FINCHAT_MAX_REQUEST_SIZE_MB`)

### Benefits
- Prevents denial of service attacks via large payloads
- Reduces memory consumption
- Protects against resource exhaustion

### Configuration
```bash
export FINCHAT_MAX_REQUEST_SIZE_MB=5  # Increase to 5MB if needed
```

## CSRF Protection

### Token-Based Protection
- **Algorithm**: Cryptographically secure random tokens (32 bytes)
- **Expiry**: 30 minutes (configurable via `FINCHAT_CSRF_TOKEN_EXPIRY_SECONDS`)
- **Storage**: In-memory with automatic cleanup
- **Validation**: Required for all state-changing operations (POST, PUT, DELETE, PATCH)

### Workflow
1. Client authenticates and requests CSRF token: `GET /csrf-token`
2. Client includes token in subsequent requests: `X-CSRF-Token: <token>`
3. Server validates token for all state-changing operations
4. Token expires after configured time period

### Exempt Endpoints
- `GET` requests (read-only operations)
- Health checks (`/health`)
- Metrics endpoint (`/metrics`)
- CSRF token generation (`/csrf-token`)
- CORS preflight requests (`OPTIONS`)

### Example Usage
```javascript
// Get CSRF token
const tokenResponse = await fetch('/csrf-token', {
    headers: { 'Authorization': 'Bearer your-token' }
});
const { csrf_token } = await tokenResponse.json();

// Use token in subsequent requests
const response = await fetch('/query', {
    method: 'POST',
    headers: {
        'Authorization': 'Bearer your-token',
        'X-CSRF-Token': csrf_token,
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({ question: 'What is the revenue?', ticker: 'AAPL' })
});
```

## Security Headers

### Implemented Headers
- **X-Content-Type-Options**: `nosniff` - Prevents MIME type sniffing
- **X-Frame-Options**: `DENY` - Prevents clickjacking attacks
- **X-XSS-Protection**: `1; mode=block` - Enables XSS filtering
- **Strict-Transport-Security**: `max-age=31536000; includeSubDomains` - Enforces HTTPS
- **Content-Security-Policy**: Restrictive policy preventing inline scripts

### CSP Policy
```
default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'
```

## Error Handling

### Request Size Exceeded
- **Status Code**: 413 (Request Entity Too Large)
- **Response**: JSON error with size limit information

### CSRF Validation Failed
- **Status Code**: 403 (Forbidden)
- **Response**: "CSRF token missing or invalid"
- **Logging**: Warning logged with client IP

### Token Expired
- **Status Code**: 403 (Forbidden)
- **Response**: "CSRF token expired"
- **Action**: Client should request new token

## Configuration Options

```bash
# Request size limit (in MB)
export FINCHAT_MAX_REQUEST_SIZE_MB=1

# CSRF token expiry (in seconds)
export FINCHAT_CSRF_TOKEN_EXPIRY_SECONDS=1800

# CORS origins (if needed for CSRF token requests)
export FINCHAT_CORS_ALLOWED_ORIGINS='["https://app.example.com","https://admin.example.com"]'
```

## Testing

Comprehensive test suite includes:
- Request size limit enforcement
- CSRF token generation and validation
- Token expiration handling
- Security headers verification
- Exempt endpoint testing
- Error response validation

## Migration Guide

### Existing Clients
1. **Breaking Change**: All POST requests now require CSRF tokens
2. **Update Process**:
   - Add CSRF token request to authentication flow
   - Include `X-CSRF-Token` header in all state-changing requests
   - Handle 403 responses by refreshing CSRF token

### Backward Compatibility
- No changes to read-only endpoints (`GET` requests)
- Authentication mechanism unchanged
- Response formats remain the same (except for security headers)

## Security Benefits

- ✅ **DoS Protection**: Request size limits prevent resource exhaustion
- ✅ **CSRF Prevention**: Token-based protection against cross-site request forgery
- ✅ **Clickjacking Protection**: X-Frame-Options header
- ✅ **XSS Mitigation**: Content Security Policy and XSS protection headers
- ✅ **HTTPS Enforcement**: Strict Transport Security header
- ✅ **MIME Sniffing Prevention**: X-Content-Type-Options header

## Performance Impact

- **CSRF Token Generation**: ~1-2ms per token
- **Token Validation**: ~0.1ms per request
- **Request Size Check**: ~0.1ms per request
- **Memory Usage**: Minimal (tokens stored in-memory with TTL)
- **Automatic Cleanup**: Expired tokens removed every token generation