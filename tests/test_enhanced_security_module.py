"""
Tests for Enhanced Security Module - Generation 2 Quality Gates
TERRAGON SDLC v4.0 - Comprehensive Security Testing
"""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from finchat_sec_qa.enhanced_security_module import (
    EnhancedSecurityModule,
    SecurityThreatLevel,
    SecurityEventType,
    SecurityEvent,
    RateLimitBucket,
    security_module,
    secure_input,
    check_rate_limit,
    authenticate
)


class TestEnhancedSecurityModule:
    """Test enhanced security module functionality."""
    
    @pytest.fixture
    def security_instance(self):
        """Create security module instance for testing."""
        return EnhancedSecurityModule()
    
    def test_security_module_initialization(self, security_instance):
        """Test security module initialization."""
        module = security_instance
        
        assert module.rate_limit_requests == 100
        assert module.rate_limit_window == 3600
        assert len(module.rate_limit_buckets) == 0
        assert len(module.security_events) == 0
        assert len(module.blocked_ips) == 0
        assert len(module.compiled_sql_patterns) > 0
        assert len(module.compiled_xss_patterns) > 0
        assert len(module.compiled_path_patterns) > 0
    
    def test_encryption_decryption(self, security_instance):
        """Test data encryption and decryption."""
        module = security_instance
        
        original_data = "sensitive information"
        
        # Encrypt data
        encrypted = module.encrypt_sensitive_data(original_data)
        assert encrypted != original_data
        assert len(encrypted) > 0
        
        # Decrypt data
        decrypted = module.decrypt_sensitive_data(encrypted)
        assert decrypted == original_data
    
    def test_sql_injection_detection(self, security_instance):
        """Test SQL injection detection."""
        module = security_instance
        
        # Test malicious SQL inputs
        sql_attacks = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin' UNION SELECT * FROM passwords--",
            "'; INSERT INTO users VALUES ('hacker', 'pass'); --",
            "1; EXEC sp_configure 'show advanced options', 1; --"
        ]
        
        for attack in sql_attacks:
            result = module.validate_and_sanitize_input(attack, source_ip="192.168.1.100")
            
            assert 'sql_pattern' in str(result['threats_detected'])
            assert result['severity'] == SecurityThreatLevel.HIGH
            assert result['sanitized_data'] != attack  # Should be sanitized
    
    def test_xss_detection(self, security_instance):
        """Test XSS attack detection."""
        module = security_instance
        
        # Test XSS attacks
        xss_attacks = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "javascript:alert('XSS')",
            "<iframe src='javascript:alert(\"XSS\")'></iframe>",
            "<body onload=alert('XSS')>"
        ]
        
        for attack in xss_attacks:
            result = module.validate_and_sanitize_input(attack, source_ip="192.168.1.100")
            
            assert 'xss_pattern' in str(result['threats_detected'])
            assert result['severity'] == SecurityThreatLevel.HIGH
            assert '<script>' not in result['sanitized_data'].lower()
    
    def test_path_traversal_detection(self, security_instance):
        """Test path traversal attack detection."""
        module = security_instance
        
        # Test path traversal attacks
        path_attacks = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "....//....//....//etc//passwd"
        ]
        
        for attack in path_attacks:
            result = module.validate_and_sanitize_input(attack, source_ip="192.168.1.100")
            
            assert 'path_pattern' in str(result['threats_detected'])
            assert result['severity'] == SecurityThreatLevel.HIGH
            assert '..' not in result['sanitized_data']
    
    def test_input_validation_types(self, security_instance):
        """Test input validation for different types."""
        module = security_instance
        
        # Valid email
        result = module.validate_and_sanitize_input("user@example.com", input_type="email")
        assert result['valid'] is True
        
        # Invalid email
        result = module.validate_and_sanitize_input("invalid-email", input_type="email")
        assert result['valid'] is False
        assert 'invalid_email_format' in result['threats_detected']
        
        # Valid URL
        result = module.validate_and_sanitize_input("https://example.com", input_type="url")
        assert result['valid'] is True
        
        # Invalid URL
        result = module.validate_and_sanitize_input("not-a-url", input_type="url")
        assert result['valid'] is False
        assert 'invalid_url_format' in result['threats_detected']
        
        # Valid ticker
        result = module.validate_and_sanitize_input("AAPL", input_type="ticker")
        assert result['valid'] is True
        
        # Invalid ticker
        result = module.validate_and_sanitize_input("invalid123", input_type="ticker")
        assert result['valid'] is False
        assert 'invalid_ticker_format' in result['threats_detected']
    
    def test_input_length_validation(self, security_instance):
        """Test input length validation."""
        module = security_instance
        
        # Normal length input
        result = module.validate_and_sanitize_input("normal input", max_length=100)
        assert result['valid'] is True
        assert result['sanitized_data'] == "normal input"
        
        # Too long input
        long_input = "x" * 200
        result = module.validate_and_sanitize_input(long_input, max_length=100)
        assert 'input_too_long' in result['threats_detected']
        assert len(result['sanitized_data']) == 100
    
    def test_rate_limiting_basic(self, security_instance):
        """Test basic rate limiting functionality."""
        module = security_instance
        module.rate_limit_requests = 5  # Low limit for testing
        
        identifier = "test_user"
        
        # First few requests should be allowed
        for i in range(5):
            result = module.check_rate_limit(identifier)
            assert result['allowed'] is True
            assert result['remaining_requests'] == 5 - i - 1
        
        # Sixth request should be blocked
        result = module.check_rate_limit(identifier)
        assert result['allowed'] is False
        assert result['blocked'] is True
        assert result['remaining_requests'] == 0
    
    def test_rate_limiting_window_reset(self, security_instance):
        """Test rate limiting window reset."""
        module = security_instance
        module.rate_limit_requests = 2
        module.rate_limit_window = 1  # 1 second window for testing
        
        identifier = "test_user"
        
        # Use up the limit
        result1 = module.check_rate_limit(identifier)
        result2 = module.check_rate_limit(identifier)
        assert result1['allowed'] is True
        assert result2['allowed'] is True
        
        # Third request should be blocked
        result3 = module.check_rate_limit(identifier)
        assert result3['allowed'] is False
        
        # Wait for window to reset
        time.sleep(1.1)
        
        # Should be allowed again
        result4 = module.check_rate_limit(identifier)
        assert result4['allowed'] is True
    
    def test_rate_limiting_different_identifiers(self, security_instance):
        """Test rate limiting with different identifiers."""
        module = security_instance
        module.rate_limit_requests = 2
        
        # Two different users should have separate limits
        result1 = module.check_rate_limit("user1")
        result2 = module.check_rate_limit("user2")
        
        assert result1['allowed'] is True
        assert result2['allowed'] is True
        
        # Each should have their own remaining count
        assert result1['remaining_requests'] == 1
        assert result2['remaining_requests'] == 1
    
    def test_authentication_valid_token(self, security_instance):
        """Test authentication with valid token."""
        module = security_instance
        
        # Create a valid token
        user_id = "test_user"
        timestamp = str(int(time.time()))
        signature = module._generate_token_signature(user_id, timestamp)
        token = f"{user_id}:{timestamp}:{signature}"
        
        result = module.authenticate_request(token, source_ip="192.168.1.100")
        
        assert result['authenticated'] is True
        assert result['token_valid'] is True
        assert result['user_id'] == user_id
        assert 'read' in result['permissions']
    
    def test_authentication_invalid_token_format(self, security_instance):
        """Test authentication with invalid token format."""
        module = security_instance
        
        invalid_tokens = [
            "invalid",
            "too:few",
            "",
            "user:timestamp",  # Missing signature
            "user:notanumber:sig"  # Invalid timestamp
        ]
        
        for token in invalid_tokens:
            result = module.authenticate_request(token, source_ip="192.168.1.100")
            
            assert result['authenticated'] is False
            assert result['token_valid'] is False
            assert result['reason'] is not None
    
    def test_authentication_expired_token(self, security_instance):
        """Test authentication with expired token."""
        module = security_instance
        
        # Create an expired token (25 hours ago)
        user_id = "test_user"
        old_timestamp = str(int(time.time()) - 90000)  # 25 hours ago
        signature = module._generate_token_signature(user_id, old_timestamp)
        token = f"{user_id}:{old_timestamp}:{signature}"
        
        result = module.authenticate_request(token, source_ip="192.168.1.100")
        
        assert result['authenticated'] is False
        assert result['reason'] == 'token_expired'
    
    def test_authentication_invalid_signature(self, security_instance):
        """Test authentication with invalid signature."""
        module = security_instance
        
        # Create token with wrong signature
        user_id = "test_user"
        timestamp = str(int(time.time()))
        token = f"{user_id}:{timestamp}:wrong_signature"
        
        result = module.authenticate_request(token, source_ip="192.168.1.100")
        
        assert result['authenticated'] is False
        assert result['reason'] == 'invalid_signature'
    
    def test_user_permissions(self, security_instance):
        """Test user permission assignment."""
        module = security_instance
        
        # Regular user
        permissions = module._get_user_permissions("regular_user")
        assert 'read' in permissions
        assert 'query' in permissions
        assert 'admin' not in permissions
        
        # Admin user
        admin_permissions = module._get_user_permissions("admin_user")
        assert 'read' in admin_permissions
        assert 'query' in admin_permissions
        assert 'write' in admin_permissions
        assert 'admin' in admin_permissions
    
    def test_security_event_logging(self, security_instance):
        """Test security event logging."""
        module = security_instance
        
        initial_events = len(module.security_events)
        
        # Trigger a security event
        module._log_security_event(
            SecurityEventType.SQL_INJECTION_ATTEMPT,
            SecurityThreatLevel.HIGH,
            "192.168.1.100",
            details={'input': "'; DROP TABLE users; --"}
        )
        
        assert len(module.security_events) == initial_events + 1
        
        event = module.security_events[-1]
        assert event.event_type == SecurityEventType.SQL_INJECTION_ATTEMPT
        assert event.threat_level == SecurityThreatLevel.HIGH
        assert event.source_ip == "192.168.1.100"
        assert 'input' in event.details
    
    def test_automated_response_critical_threat(self, security_instance):
        """Test automated response to critical threats."""
        module = security_instance
        
        ip = "192.168.1.100"
        assert not module.is_ip_blocked(ip)
        
        # Create critical security event
        event = SecurityEvent(
            event_id="test_event",
            timestamp=datetime.now(),
            event_type=SecurityEventType.SQL_INJECTION_ATTEMPT,
            threat_level=SecurityThreatLevel.CRITICAL,
            source_ip=ip,
            user_agent=None,
            user_id=None
        )
        
        # Apply automated response
        module._apply_automated_response(event)
        
        # IP should be blocked
        assert module.is_ip_blocked(ip)
    
    def test_automated_response_high_threat_pattern(self, security_instance):
        """Test automated response to high threat patterns."""
        module = security_instance
        
        ip = "192.168.1.100"
        
        # Create multiple high-severity events
        for i in range(3):
            event = SecurityEvent(
                event_id=f"test_event_{i}",
                timestamp=datetime.now(),
                event_type=SecurityEventType.BRUTE_FORCE_ATTEMPT,
                threat_level=SecurityThreatLevel.HIGH,
                source_ip=ip,
                user_agent=None,
                user_id=None
            )
            module._apply_automated_response(event)
        
        # IP should be blocked after 3 high-severity events
        assert module.is_ip_blocked(ip)
    
    def test_security_analytics(self, security_instance):
        """Test security analytics generation."""
        module = security_instance
        
        # Add some test events
        events = [
            SecurityEvent(
                event_id="event1",
                timestamp=datetime.now(),
                event_type=SecurityEventType.SQL_INJECTION_ATTEMPT,
                threat_level=SecurityThreatLevel.HIGH,
                source_ip="192.168.1.100",
                user_agent=None,
                user_id=None
            ),
            SecurityEvent(
                event_id="event2",
                timestamp=datetime.now(),
                event_type=SecurityEventType.XSS_ATTEMPT,
                threat_level=SecurityThreatLevel.MEDIUM,
                source_ip="192.168.1.101",
                user_agent=None,
                user_id=None
            )
        ]
        
        module.security_events.extend(events)
        module.blocked_ips.add("192.168.1.100")
        
        analytics = module.get_security_analytics()
        
        assert analytics['total_events'] == 2
        assert 'sql_injection' in analytics['event_type_distribution']
        assert 'xss_attempt' in analytics['event_type_distribution']
        assert 'high' in analytics['threat_level_distribution']
        assert 'medium' in analytics['threat_level_distribution']
        assert '192.168.1.100' in analytics['blocked_ips']
        assert '192.168.1.100' in analytics['top_source_ips']
    
    def test_security_sanitization_effectiveness(self, security_instance):
        """Test effectiveness of security sanitization."""
        module = security_instance
        
        # Test SQL injection sanitization
        sql_input = "admin'; DROP TABLE users; SELECT * FROM passwords WHERE '1'='1"
        result = module.validate_and_sanitize_input(sql_input)
        
        sanitized = result['sanitized_data']
        dangerous_keywords = ['DROP', 'SELECT', 'INSERT', 'UPDATE', 'DELETE', '--', ';', "'"]
        
        for keyword in dangerous_keywords:
            assert keyword.lower() not in sanitized.lower()
        
        # Test XSS sanitization
        xss_input = "<script>alert('XSS')</script><img src=x onerror=alert('XSS')>"
        result = module.validate_and_sanitize_input(xss_input)
        
        sanitized = result['sanitized_data']
        assert '<script>' not in sanitized.lower()
        assert 'onerror=' not in sanitized.lower()
        assert 'javascript:' not in sanitized.lower()


class TestGlobalFunctions:
    """Test global convenience functions."""
    
    def test_secure_input_function(self):
        """Test global secure_input function."""
        result = secure_input("test input", input_type="general")
        
        assert 'valid' in result
        assert 'sanitized_data' in result
        assert 'threats_detected' in result
        assert 'severity' in result
    
    def test_check_rate_limit_function(self):
        """Test global check_rate_limit function."""
        result = check_rate_limit("test_identifier")
        
        assert 'allowed' in result
        assert 'blocked' in result
        assert 'remaining_requests' in result
    
    def test_authenticate_function(self):
        """Test global authenticate function."""
        # Test with invalid token
        result = authenticate("invalid_token")
        
        assert 'authenticated' in result
        assert 'token_valid' in result
        assert result['authenticated'] is False


class TestSecurityIntegration:
    """Integration tests for security module."""
    
    def test_comprehensive_attack_detection(self):
        """Test detection of combined attacks."""
        module = EnhancedSecurityModule()
        
        # Combined SQL injection and XSS attack
        malicious_input = "'; DROP TABLE users; --<script>alert('XSS')</script>"
        
        result = module.validate_and_sanitize_input(malicious_input, source_ip="192.168.1.100")
        
        # Should detect both types of attacks
        threats = result['threats_detected']
        assert any('sql_pattern' in str(threat) for threat in threats)
        assert any('xss_pattern' in str(threat) for threat in threats)
        assert result['severity'] == SecurityThreatLevel.HIGH
        
        # Should sanitize both types of attacks
        sanitized = result['sanitized_data']
        assert 'DROP' not in sanitized
        assert '<script>' not in sanitized.lower()
    
    def test_rate_limiting_with_security_events(self):
        """Test rate limiting integration with security event logging."""
        module = EnhancedSecurityModule(rate_limit_requests=2)
        
        identifier = "attacker"
        initial_events = len(module.security_events)
        
        # Make requests up to the limit
        module.check_rate_limit(identifier)
        module.check_rate_limit(identifier)
        
        # This should trigger rate limit and log security event
        result = module.check_rate_limit(identifier, source_ip="192.168.1.100")
        
        assert result['allowed'] is False
        assert len(module.security_events) > initial_events
        
        # Find the rate limit event
        rate_limit_events = [
            event for event in module.security_events
            if event.event_type == SecurityEventType.RATE_LIMIT_EXCEEDED
        ]
        assert len(rate_limit_events) > 0
    
    def test_escalating_security_response(self):
        """Test escalating security response to repeated attacks."""
        module = EnhancedSecurityModule()
        
        attacker_ip = "192.168.1.100"
        
        # Simulate escalating attacks
        # First: Medium threat
        result1 = module.validate_and_sanitize_input(
            "slightly suspicious input",
            source_ip=attacker_ip
        )
        
        # Second: High threat (XSS)
        result2 = module.validate_and_sanitize_input(
            "<script>alert('XSS')</script>",
            source_ip=attacker_ip
        )
        
        # Third: Critical threat (complex SQL injection)
        result3 = module.validate_and_sanitize_input(
            "'; DROP DATABASE production; EXEC sp_configure 'show advanced options', 1; --",
            source_ip=attacker_ip
        )
        
        # IP should be blocked after critical threat
        assert module.is_ip_blocked(attacker_ip)
        
        # Should have multiple security events
        attacker_events = [
            event for event in module.security_events
            if event.source_ip == attacker_ip
        ]
        assert len(attacker_events) >= 2  # At least XSS and SQL injection
    
    def test_performance_under_attack(self):
        """Test security module performance under simulated attack."""
        module = EnhancedSecurityModule()
        
        import time
        start_time = time.time()
        
        # Simulate rapid attacks
        attacks = [
            "'; DROP TABLE users; --",
            "<script>alert('XSS')</script>",
            "../../../etc/passwd",
            "admin' OR '1'='1",
            "<img src=x onerror=alert('XSS')>"
        ] * 20  # 100 total attacks
        
        for i, attack in enumerate(attacks):
            module.validate_and_sanitize_input(attack, source_ip=f"192.168.1.{i % 255}")
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should process 100 attacks in reasonable time (less than 5 seconds)
        assert processing_time < 5.0
        
        # Should have detected all attacks
        assert len(module.security_events) == len(attacks)
        
        # Should have blocked some IPs
        assert len(module.blocked_ips) > 0