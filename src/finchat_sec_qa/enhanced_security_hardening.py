"""
Enhanced Security Hardening System
Enterprise-grade security with zero-trust architecture
"""

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Set
from pathlib import Path

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class SecurityLevel(Enum):
    """Security levels for different operations"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatLevel(Enum):
    """Threat assessment levels"""
    MINIMAL = "minimal"
    LOW = "low" 
    MODERATE = "moderate"
    HIGH = "high"
    SEVERE = "severe"


@dataclass
class SecurityContext:
    """Security context for operations"""
    user_id: Optional[str]
    session_id: str
    ip_address: str
    user_agent: str
    timestamp: datetime
    security_level: SecurityLevel
    permissions: Set[str]
    risk_score: float


@dataclass
class ThreatAssessment:
    """Threat assessment result"""
    threat_level: ThreatLevel
    risk_score: float
    threat_indicators: List[str]
    mitigation_actions: List[str]
    confidence: float


@dataclass
class SecurityAuditEvent:
    """Security audit event"""
    event_id: str
    timestamp: datetime
    event_type: str
    user_id: Optional[str]
    resource: str
    action: str
    result: str
    risk_score: float
    metadata: Dict[str, Any]


class EnhancedSecurityHardening:
    """
    Enhanced Security Hardening System
    
    Features:
    - Zero-trust security model
    - Advanced threat detection
    - Multi-layer encryption
    - Real-time security monitoring
    - Automated incident response
    - Comprehensive audit logging
    - Behavioral anomaly detection
    - Adaptive security policies
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or self._default_config()
        
        # Initialize encryption systems
        self.master_key = self._initialize_master_key()
        self.fernet_cipher = Fernet(self.master_key)
        self.rsa_private_key = self._generate_rsa_keypair()
        
        # Security state tracking
        self.active_sessions: Dict[str, SecurityContext] = {}
        self.threat_intelligence: Dict[str, ThreatAssessment] = {}
        self.security_events: List[SecurityAuditEvent] = []
        self.anomaly_baselines: Dict[str, Dict[str, float]] = {}
        
        # Rate limiting and blocking
        self.rate_limits: Dict[str, List[float]] = {}
        self.blocked_ips: Set[str] = set()
        self.suspicious_patterns: Dict[str, int] = {}
        
        # Initialize security components
        self._initialize_threat_detection()
        self._initialize_anomaly_detection()
        
        self.logger.info("Enhanced Security Hardening System initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default security configuration"""
        return {
            "encryption": {
                "algorithm": "AES-256-GCM",
                "key_rotation_interval": 86400,  # 24 hours
                "salt_length": 32
            },
            "authentication": {
                "session_timeout": 3600,  # 1 hour
                "max_login_attempts": 5,
                "lockout_duration": 900,  # 15 minutes
                "require_mfa": True
            },
            "rate_limiting": {
                "requests_per_minute": 100,
                "burst_limit": 200,
                "window_size": 60
            },
            "monitoring": {
                "anomaly_threshold": 0.8,
                "threat_score_threshold": 0.7,
                "audit_retention_days": 90
            },
            "policies": {
                "min_password_length": 12,
                "require_special_chars": True,
                "password_rotation_days": 90,
                "session_ip_binding": True
            }
        }
    
    def _initialize_master_key(self) -> bytes:
        """Initialize or load master encryption key"""
        key_file = Path(".security_key")
        
        if key_file.exists():
            with open(key_file, "rb") as f:
                return f.read()
        else:
            # Generate new master key
            key = Fernet.generate_key()
            with open(key_file, "wb") as f:
                f.write(key)
            key_file.chmod(0o600)  # Restrict permissions
            return key
    
    def _generate_rsa_keypair(self) -> rsa.RSAPrivateKey:
        """Generate RSA key pair for asymmetric encryption"""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096
        )
        return private_key
    
    def _initialize_threat_detection(self) -> None:
        """Initialize threat detection patterns"""
        self.threat_patterns = {
            "sql_injection": [
                r"union\s+select", r"drop\s+table", r"insert\s+into",
                r"delete\s+from", r"update\s+set", r"--", r"/*", r"*/"
            ],
            "xss": [
                r"<script", r"javascript:", r"onload=", r"onerror=",
                r"onclick=", r"alert\(", r"eval\(", r"document\."
            ],
            "path_traversal": [
                r"\.\./", r"\.\.\\", r"%2e%2e%2f", r"%2e%2e\\",
                r"/etc/passwd", r"c:\\windows"
            ],
            "command_injection": [
                r";\s*rm\s", r";\s*cat\s", r";\s*ls\s", r"&&", r"\|\|",
                r"`", r"\$\(", r"nc\s+", r"curl\s+"
            ]
        }
    
    def _initialize_anomaly_detection(self) -> None:
        """Initialize behavioral anomaly detection"""
        self.anomaly_detectors = {
            "request_frequency": self._detect_frequency_anomaly,
            "access_pattern": self._detect_access_pattern_anomaly,
            "data_volume": self._detect_data_volume_anomaly,
            "geographic": self._detect_geographic_anomaly,
            "temporal": self._detect_temporal_anomaly
        }
    
    async def authenticate_request(
        self,
        credentials: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Tuple[bool, Optional[SecurityContext]]:
        """
        Authenticate request with comprehensive security checks
        
        Args:
            credentials: User credentials
            context: Request context (IP, user agent, etc.)
            
        Returns:
            Tuple of (authenticated, security_context)
        """
        start_time = time.time()
        
        try:
            # Extract context information
            ip_address = context.get("ip_address", "unknown")
            user_agent = context.get("user_agent", "unknown")
            
            # Check if IP is blocked
            if ip_address in self.blocked_ips:
                await self._log_security_event(
                    "authentication_blocked",
                    None,
                    "authentication",
                    "blocked",
                    ip_address,
                    {"reason": "blocked_ip"}
                )
                return False, None
            
            # Rate limiting check
            if not await self._check_rate_limit(ip_address):
                await self._log_security_event(
                    "rate_limit_exceeded",
                    None,
                    "authentication",
                    "blocked",
                    ip_address,
                    {"reason": "rate_limit"}
                )
                return False, None
            
            # Threat assessment
            threat_assessment = await self._assess_threat(credentials, context)
            if threat_assessment.threat_level in [ThreatLevel.HIGH, ThreatLevel.SEVERE]:
                await self._handle_threat(threat_assessment, context)
                return False, None
            
            # Validate credentials
            user_id = await self._validate_credentials(credentials)
            if not user_id:
                await self._log_security_event(
                    "authentication_failed",
                    None,
                    "authentication",
                    "failed",
                    ip_address,
                    {"reason": "invalid_credentials"}
                )
                return False, None
            
            # Create security context
            session_id = secrets.token_urlsafe(32)
            security_context = SecurityContext(
                user_id=user_id,
                session_id=session_id,
                ip_address=ip_address,
                user_agent=user_agent,
                timestamp=datetime.utcnow(),
                security_level=self._determine_security_level(threat_assessment),
                permissions=await self._get_user_permissions(user_id),
                risk_score=threat_assessment.risk_score
            )
            
            # Store active session
            self.active_sessions[session_id] = security_context
            
            # Log successful authentication
            await self._log_security_event(
                "authentication_success",
                user_id,
                "authentication",
                "success",
                ip_address,
                {
                    "session_id": session_id,
                    "risk_score": threat_assessment.risk_score,
                    "auth_time": time.time() - start_time
                }
            )
            
            return True, security_context
            
        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            await self._log_security_event(
                "authentication_error",
                None,
                "authentication",
                "error",
                context.get("ip_address", "unknown"),
                {"error": str(e)}
            )
            return False, None
    
    async def authorize_request(
        self,
        security_context: SecurityContext,
        resource: str,
        action: str
    ) -> bool:
        """
        Authorize request based on security context and policies
        
        Args:
            security_context: Security context from authentication
            resource: Resource being accessed
            action: Action being performed
            
        Returns:
            Authorization result
        """
        try:
            # Check session validity
            if not await self._validate_session(security_context):
                return False
            
            # Check permissions
            required_permission = f"{resource}:{action}"
            if required_permission not in security_context.permissions and "admin:*" not in security_context.permissions:
                await self._log_security_event(
                    "authorization_denied",
                    security_context.user_id,
                    resource,
                    action,
                    security_context.ip_address,
                    {
                        "required_permission": required_permission,
                        "user_permissions": list(security_context.permissions)
                    }
                )
                return False
            
            # Adaptive authorization based on risk score
            if security_context.risk_score > 0.8 and security_context.security_level != SecurityLevel.CRITICAL:
                # Require additional verification for high-risk requests
                if not await self._require_additional_verification(security_context):
                    return False
            
            # Log successful authorization
            await self._log_security_event(
                "authorization_granted",
                security_context.user_id,
                resource,
                action,
                security_context.ip_address,
                {"risk_score": security_context.risk_score}
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Authorization error: {e}")
            return False
    
    async def encrypt_data(self, data: bytes, security_level: SecurityLevel = SecurityLevel.MEDIUM) -> bytes:
        """
        Encrypt data with appropriate security level
        
        Args:
            data: Data to encrypt
            security_level: Security level for encryption
            
        Returns:
            Encrypted data
        """
        try:
            if security_level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
                # Use hybrid encryption for high security
                return await self._hybrid_encrypt(data)
            else:
                # Use symmetric encryption for standard security
                return self.fernet_cipher.encrypt(data)
                
        except Exception as e:
            self.logger.error(f"Encryption error: {e}")
            raise
    
    async def decrypt_data(self, encrypted_data: bytes, security_level: SecurityLevel = SecurityLevel.MEDIUM) -> bytes:
        """
        Decrypt data with appropriate security level
        
        Args:
            encrypted_data: Data to decrypt
            security_level: Security level for decryption
            
        Returns:
            Decrypted data
        """
        try:
            if security_level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
                # Use hybrid decryption for high security
                return await self._hybrid_decrypt(encrypted_data)
            else:
                # Use symmetric decryption for standard security
                return self.fernet_cipher.decrypt(encrypted_data)
                
        except Exception as e:
            self.logger.error(f"Decryption error: {e}")
            raise
    
    async def scan_for_threats(self, input_data: str, context: Dict[str, Any]) -> ThreatAssessment:
        """
        Scan input data for security threats
        
        Args:
            input_data: Data to scan
            context: Context information
            
        Returns:
            Threat assessment
        """
        threat_indicators = []
        risk_score = 0.0
        
        try:
            # Check for known attack patterns
            for threat_type, patterns in self.threat_patterns.items():
                for pattern in patterns:
                    import re
                    if re.search(pattern, input_data, re.IGNORECASE):
                        threat_indicators.append(f"{threat_type}:{pattern}")
                        risk_score += 0.2
            
            # Entropy analysis for encoded payloads
            entropy = self._calculate_entropy(input_data)
            if entropy > 7.0:  # High entropy indicates possible encoding
                threat_indicators.append("high_entropy_content")
                risk_score += 0.3
            
            # Length-based anomaly detection
            if len(input_data) > 10000:  # Unusually long input
                threat_indicators.append("unusual_length")
                risk_score += 0.1
            
            # Character frequency analysis
            if self._detect_suspicious_chars(input_data):
                threat_indicators.append("suspicious_characters")
                risk_score += 0.2
            
            # Determine threat level
            threat_level = ThreatLevel.MINIMAL
            if risk_score >= 0.8:
                threat_level = ThreatLevel.SEVERE
            elif risk_score >= 0.6:
                threat_level = ThreatLevel.HIGH
            elif risk_score >= 0.4:
                threat_level = ThreatLevel.MODERATE
            elif risk_score >= 0.2:
                threat_level = ThreatLevel.LOW
            
            # Generate mitigation actions
            mitigation_actions = []
            if threat_level in [ThreatLevel.HIGH, ThreatLevel.SEVERE]:
                mitigation_actions.extend([
                    "block_request",
                    "alert_security_team",
                    "increase_monitoring"
                ])
            elif threat_level == ThreatLevel.MODERATE:
                mitigation_actions.extend([
                    "sanitize_input",
                    "log_for_analysis"
                ])
            
            return ThreatAssessment(
                threat_level=threat_level,
                risk_score=min(risk_score, 1.0),
                threat_indicators=threat_indicators,
                mitigation_actions=mitigation_actions,
                confidence=0.85
            )
            
        except Exception as e:
            self.logger.error(f"Threat scanning error: {e}")
            return ThreatAssessment(
                threat_level=ThreatLevel.MODERATE,
                risk_score=0.5,
                threat_indicators=["scan_error"],
                mitigation_actions=["manual_review"],
                confidence=0.1
            )
    
    async def detect_anomalies(self, security_context: SecurityContext, request_data: Dict[str, Any]) -> List[str]:
        """
        Detect behavioral anomalies
        
        Args:
            security_context: Security context
            request_data: Request data for analysis
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        try:
            for detector_name, detector_func in self.anomaly_detectors.items():
                try:
                    is_anomaly = await detector_func(security_context, request_data)
                    if is_anomaly:
                        anomalies.append(detector_name)
                except Exception as e:
                    self.logger.warning(f"Anomaly detector {detector_name} failed: {e}")
            
            if anomalies:
                await self._log_security_event(
                    "anomaly_detected",
                    security_context.user_id,
                    "behavioral_analysis",
                    "anomaly",
                    security_context.ip_address,
                    {"anomalies": anomalies}
                )
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Anomaly detection error: {e}")
            return ["detection_error"]
    
    async def generate_security_report(self, time_range: timedelta = timedelta(hours=24)) -> Dict[str, Any]:
        """
        Generate comprehensive security report
        
        Args:
            time_range: Time range for report
            
        Returns:
            Security report
        """
        cutoff_time = datetime.utcnow() - time_range
        recent_events = [e for e in self.security_events if e.timestamp > cutoff_time]
        
        # Categorize events
        event_categories = {}
        for event in recent_events:
            category = event.event_type
            if category not in event_categories:
                event_categories[category] = 0
            event_categories[category] += 1
        
        # Calculate risk metrics
        avg_risk_score = sum(e.risk_score for e in recent_events) / len(recent_events) if recent_events else 0
        high_risk_events = [e for e in recent_events if e.risk_score > 0.7]
        
        # Active threats
        active_threats = [t for t in self.threat_intelligence.values() 
                         if t.threat_level in [ThreatLevel.HIGH, ThreatLevel.SEVERE]]
        
        # Session statistics
        active_session_count = len(self.active_sessions)
        unique_users = len(set(ctx.user_id for ctx in self.active_sessions.values() if ctx.user_id))
        
        return {
            "report_generated": datetime.utcnow().isoformat(),
            "time_range_hours": time_range.total_seconds() / 3600,
            "summary": {
                "total_events": len(recent_events),
                "event_categories": event_categories,
                "average_risk_score": avg_risk_score,
                "high_risk_events": len(high_risk_events),
                "active_threats": len(active_threats),
                "active_sessions": active_session_count,
                "unique_users": unique_users,
                "blocked_ips": len(self.blocked_ips)
            },
            "threat_intelligence": [
                {
                    "threat_level": t.threat_level.value,
                    "risk_score": t.risk_score,
                    "indicators": t.threat_indicators,
                    "confidence": t.confidence
                }
                for t in active_threats
            ],
            "top_risk_events": [
                {
                    "event_id": e.event_id,
                    "timestamp": e.timestamp.isoformat(),
                    "event_type": e.event_type,
                    "risk_score": e.risk_score,
                    "resource": e.resource
                }
                for e in sorted(high_risk_events, key=lambda x: x.risk_score, reverse=True)[:10]
            ],
            "recommendations": self._generate_security_recommendations(recent_events)
        }
    
    # Private helper methods
    async def _assess_threat(self, credentials: Dict[str, Any], context: Dict[str, Any]) -> ThreatAssessment:
        """Assess threat level for authentication request"""
        risk_score = 0.0
        threat_indicators = []
        
        # Check IP reputation
        ip_address = context.get("ip_address", "")
        if ip_address in self.suspicious_patterns:
            risk_score += 0.3
            threat_indicators.append("suspicious_ip")
        
        # Check for brute force patterns
        if self._detect_brute_force_pattern(ip_address):
            risk_score += 0.4
            threat_indicators.append("brute_force_pattern")
        
        # Geographic anomaly
        if await self._detect_geographic_anomaly_auth(context):
            risk_score += 0.2
            threat_indicators.append("geographic_anomaly")
        
        # Time-based anomaly
        if await self._detect_temporal_anomaly_auth(context):
            risk_score += 0.1
            threat_indicators.append("temporal_anomaly")
        
        threat_level = ThreatLevel.MINIMAL
        if risk_score >= 0.8:
            threat_level = ThreatLevel.SEVERE
        elif risk_score >= 0.6:
            threat_level = ThreatLevel.HIGH
        elif risk_score >= 0.4:
            threat_level = ThreatLevel.MODERATE
        elif risk_score >= 0.2:
            threat_level = ThreatLevel.LOW
        
        return ThreatAssessment(
            threat_level=threat_level,
            risk_score=min(risk_score, 1.0),
            threat_indicators=threat_indicators,
            mitigation_actions=["monitor", "log"] if threat_level == ThreatLevel.MINIMAL else ["block", "alert"],
            confidence=0.8
        )
    
    async def _validate_credentials(self, credentials: Dict[str, Any]) -> Optional[str]:
        """Validate user credentials (placeholder)"""
        # In real implementation, this would check against user database
        username = credentials.get("username")
        password = credentials.get("password")
        
        if username and password:
            # Simulate password validation
            return username
        return None
    
    async def _get_user_permissions(self, user_id: str) -> Set[str]:
        """Get user permissions (placeholder)"""
        # In real implementation, this would fetch from permission system
        return {"read:documents", "write:documents", "read:analytics"}
    
    def _determine_security_level(self, threat_assessment: ThreatAssessment) -> SecurityLevel:
        """Determine security level based on threat assessment"""
        if threat_assessment.risk_score >= 0.8:
            return SecurityLevel.CRITICAL
        elif threat_assessment.risk_score >= 0.6:
            return SecurityLevel.HIGH
        elif threat_assessment.risk_score >= 0.4:
            return SecurityLevel.MEDIUM
        else:
            return SecurityLevel.LOW
    
    async def _validate_session(self, security_context: SecurityContext) -> bool:
        """Validate session is still active and valid"""
        session_timeout = self.config["authentication"]["session_timeout"]
        
        if datetime.utcnow() - security_context.timestamp > timedelta(seconds=session_timeout):
            # Session expired
            if security_context.session_id in self.active_sessions:
                del self.active_sessions[security_context.session_id]
            return False
        
        # IP binding check
        if self.config["policies"]["session_ip_binding"]:
            stored_context = self.active_sessions.get(security_context.session_id)
            if stored_context and stored_context.ip_address != security_context.ip_address:
                return False
        
        return True
    
    async def _require_additional_verification(self, security_context: SecurityContext) -> bool:
        """Require additional verification for high-risk requests"""
        # In real implementation, this would prompt for MFA or additional verification
        return security_context.security_level == SecurityLevel.CRITICAL
    
    async def _hybrid_encrypt(self, data: bytes) -> bytes:
        """Hybrid encryption using RSA + AES"""
        # Generate random AES key
        aes_key = secrets.token_bytes(32)
        
        # Encrypt data with AES
        f = Fernet(base64.urlsafe_b64encode(aes_key))
        encrypted_data = f.encrypt(data)
        
        # Encrypt AES key with RSA
        public_key = self.rsa_private_key.public_key()
        encrypted_key = public_key.encrypt(
            aes_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        # Combine encrypted key and data
        return len(encrypted_key).to_bytes(4, 'big') + encrypted_key + encrypted_data
    
    async def _hybrid_decrypt(self, encrypted_data: bytes) -> bytes:
        """Hybrid decryption using RSA + AES"""
        # Extract encrypted key length
        key_length = int.from_bytes(encrypted_data[:4], 'big')
        
        # Extract encrypted key and data
        encrypted_key = encrypted_data[4:4 + key_length]
        encrypted_content = encrypted_data[4 + key_length:]
        
        # Decrypt AES key with RSA
        aes_key = self.rsa_private_key.decrypt(
            encrypted_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        # Decrypt data with AES
        f = Fernet(base64.urlsafe_b64encode(aes_key))
        return f.decrypt(encrypted_content)
    
    def _calculate_entropy(self, data: str) -> float:
        """Calculate Shannon entropy of data"""
        if not data:
            return 0.0
        
        # Count character frequencies
        frequencies = {}
        for char in data:
            frequencies[char] = frequencies.get(char, 0) + 1
        
        # Calculate entropy
        entropy = 0.0
        data_length = len(data)
        for count in frequencies.values():
            probability = count / data_length
            if probability > 0:
                entropy -= probability * (probability).bit_length()
        
        return entropy
    
    def _detect_suspicious_chars(self, data: str) -> bool:
        """Detect suspicious character patterns"""
        suspicious_chars = set(['<', '>', '"', "'", '&', ';', '(', ')', '{', '}', '[', ']'])
        char_count = sum(1 for char in data if char in suspicious_chars)
        return char_count / len(data) > 0.1 if data else False
    
    def _detect_brute_force_pattern(self, ip_address: str) -> bool:
        """Detect brute force attack pattern"""
        if ip_address not in self.rate_limits:
            return False
        
        recent_requests = [t for t in self.rate_limits[ip_address] if time.time() - t < 300]  # 5 minutes
        return len(recent_requests) > 20  # More than 20 requests in 5 minutes
    
    async def _check_rate_limit(self, ip_address: str) -> bool:
        """Check rate limit for IP address"""
        current_time = time.time()
        window_size = self.config["rate_limiting"]["window_size"]
        requests_per_minute = self.config["rate_limiting"]["requests_per_minute"]
        
        if ip_address not in self.rate_limits:
            self.rate_limits[ip_address] = []
        
        # Clean old requests
        self.rate_limits[ip_address] = [
            t for t in self.rate_limits[ip_address] 
            if current_time - t < window_size
        ]
        
        # Check limit
        if len(self.rate_limits[ip_address]) >= requests_per_minute:
            return False
        
        # Add current request
        self.rate_limits[ip_address].append(current_time)
        return True
    
    async def _handle_threat(self, threat_assessment: ThreatAssessment, context: Dict[str, Any]) -> None:
        """Handle detected threat"""
        ip_address = context.get("ip_address", "unknown")
        
        if threat_assessment.threat_level == ThreatLevel.SEVERE:
            self.blocked_ips.add(ip_address)
            await self._send_security_alert(
                "SEVERE_THREAT_DETECTED",
                f"IP {ip_address} blocked due to severe threat",
                threat_assessment
            )
        elif threat_assessment.threat_level == ThreatLevel.HIGH:
            # Increase monitoring for this IP
            self.suspicious_patterns[ip_address] = self.suspicious_patterns.get(ip_address, 0) + 1
    
    async def _log_security_event(
        self,
        event_type: str,
        user_id: Optional[str],
        resource: str,
        action: str,
        ip_address: str,
        metadata: Dict[str, Any]
    ) -> None:
        """Log security event"""
        event = SecurityAuditEvent(
            event_id=secrets.token_hex(8),
            timestamp=datetime.utcnow(),
            event_type=event_type,
            user_id=user_id,
            resource=resource,
            action=action,
            result=metadata.get("result", "unknown"),
            risk_score=metadata.get("risk_score", 0.0),
            metadata=metadata
        )
        
        self.security_events.append(event)
        
        # Trim old events to prevent memory bloat
        if len(self.security_events) > 10000:
            self.security_events = self.security_events[-5000:]  # Keep last 5000
        
        # Log to file or external system
        self.logger.info(f"Security Event: {json.dumps({
            'event_id': event.event_id,
            'type': event.event_type,
            'user': event.user_id,
            'resource': event.resource,
            'action': event.action,
            'ip': ip_address,
            'risk_score': event.risk_score
        })}")
    
    # Anomaly detection methods
    async def _detect_frequency_anomaly(self, security_context: SecurityContext, request_data: Dict[str, Any]) -> bool:
        """Detect request frequency anomalies"""
        user_id = security_context.user_id
        current_time = time.time()
        
        if user_id not in self.anomaly_baselines:
            self.anomaly_baselines[user_id] = {"request_times": []}
        
        user_baseline = self.anomaly_baselines[user_id]
        user_baseline["request_times"].append(current_time)
        
        # Keep only last hour of requests
        user_baseline["request_times"] = [
            t for t in user_baseline["request_times"] 
            if current_time - t < 3600
        ]
        
        # Check if frequency is anomalous (more than 10x normal)
        recent_requests = len(user_baseline["request_times"])
        avg_requests_per_hour = user_baseline.get("avg_requests_per_hour", 10)
        
        return recent_requests > avg_requests_per_hour * 10
    
    async def _detect_access_pattern_anomaly(self, security_context: SecurityContext, request_data: Dict[str, Any]) -> bool:
        """Detect access pattern anomalies"""
        # Placeholder for access pattern analysis
        return False
    
    async def _detect_data_volume_anomaly(self, security_context: SecurityContext, request_data: Dict[str, Any]) -> bool:
        """Detect data volume anomalies"""
        # Placeholder for data volume analysis
        return False
    
    async def _detect_geographic_anomaly(self, security_context: SecurityContext, request_data: Dict[str, Any]) -> bool:
        """Detect geographic anomalies"""
        # Placeholder for geographic analysis
        return False
    
    async def _detect_temporal_anomaly(self, security_context: SecurityContext, request_data: Dict[str, Any]) -> bool:
        """Detect temporal anomalies"""
        # Placeholder for temporal analysis
        return False
    
    async def _detect_geographic_anomaly_auth(self, context: Dict[str, Any]) -> bool:
        """Detect geographic anomaly in authentication"""
        # Placeholder for geographic authentication analysis
        return False
    
    async def _detect_temporal_anomaly_auth(self, context: Dict[str, Any]) -> bool:
        """Detect temporal anomaly in authentication"""
        current_hour = datetime.utcnow().hour
        # Unusual login hours (between 2 AM and 6 AM)
        return 2 <= current_hour <= 6
    
    async def _send_security_alert(self, alert_type: str, message: str, threat_assessment: ThreatAssessment) -> None:
        """Send security alert"""
        alert = {
            "type": alert_type,
            "message": message,
            "threat_level": threat_assessment.threat_level.value,
            "risk_score": threat_assessment.risk_score,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.logger.critical(f"SECURITY ALERT: {json.dumps(alert)}")
    
    def _generate_security_recommendations(self, recent_events: List[SecurityAuditEvent]) -> List[str]:
        """Generate security recommendations based on recent events"""
        recommendations = []
        
        # Check for high risk events
        high_risk_count = len([e for e in recent_events if e.risk_score > 0.7])
        if high_risk_count > 10:
            recommendations.append("Consider implementing stricter access controls")
        
        # Check for failed authentications
        auth_failures = len([e for e in recent_events if e.event_type == "authentication_failed"])
        if auth_failures > 50:
            recommendations.append("Review authentication mechanisms and implement account lockout")
        
        # Check for blocked IPs
        if len(self.blocked_ips) > 20:
            recommendations.append("Review IP blocking policies and consider network-level filtering")
        
        return recommendations


# Global security hardening instance
_global_security_hardening: Optional[EnhancedSecurityHardening] = None


def get_security_hardening() -> EnhancedSecurityHardening:
    """Get global security hardening instance"""
    global _global_security_hardening
    if _global_security_hardening is None:
        _global_security_hardening = EnhancedSecurityHardening()
    return _global_security_hardening