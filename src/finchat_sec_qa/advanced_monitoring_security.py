"""
Advanced Monitoring & Security - Generation 2: MAKE IT ROBUST
TERRAGON SDLC v4.0 - Autonomous Execution Phase

Features:
- Real-time security monitoring and threat detection
- Comprehensive audit logging
- Performance anomaly detection
- Automated incident response
- Compliance monitoring (GDPR, CCPA, PDPA)
- Advanced metrics collection and alerting
"""

from __future__ import annotations

import logging
import time
import asyncio
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable
from enum import Enum
import json
import hashlib
import ipaddress
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class SecurityEventType(Enum):
    """Types of security events."""
    AUTHENTICATION_FAILURE = "auth_failure"
    AUTHORIZATION_VIOLATION = "authz_violation"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    MALICIOUS_INPUT = "malicious_input"
    DATA_BREACH_ATTEMPT = "data_breach_attempt"
    SYSTEM_INTRUSION = "system_intrusion"


class ComplianceRegulation(Enum):
    """Compliance regulations."""
    GDPR = "gdpr"
    CCPA = "ccpa"
    PDPA = "pdpa"
    SOX = "sox"
    PCI_DSS = "pci_dss"


@dataclass
class SecurityEvent:
    """Security event data structure."""
    event_id: str
    event_type: SecurityEventType
    severity: AlertSeverity
    timestamp: datetime
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    response_actions: List[str] = field(default_factory=list)


@dataclass
class PerformanceMetric:
    """Performance metric data structure."""
    metric_name: str
    value: float
    timestamp: datetime
    unit: str = ""
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class AuditLogEntry:
    """Audit log entry for compliance."""
    log_id: str
    timestamp: datetime
    user_id: Optional[str]
    action: str
    resource: str
    outcome: str
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    compliance_relevant: bool = True
    retention_period_days: int = 2555  # 7 years default
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedMonitoringSecurity:
    """
    Generation 2: Advanced monitoring and security system.
    
    Features:
    - Real-time security monitoring
    - Automated threat detection and response
    - Comprehensive audit logging
    - Performance anomaly detection
    - Compliance monitoring
    - Advanced alerting system
    """
    
    def __init__(self):
        # Security monitoring
        self.security_events: deque = deque(maxlen=10000)
        self.threat_indicators: Dict[str, Any] = {}
        self.blocked_ips: set = set()
        self.rate_limiters: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Performance monitoring
        self.performance_metrics: deque = deque(maxlen=50000)
        self.baseline_metrics: Dict[str, float] = {}
        self.anomaly_thresholds: Dict[str, float] = {}
        
        # Audit logging
        self.audit_logs: deque = deque(maxlen=100000)
        self.compliance_configs: Dict[ComplianceRegulation, Dict[str, Any]] = {}
        
        # Alerting
        self.alert_handlers: Dict[AlertSeverity, List[Callable]] = defaultdict(list)
        self.alert_history: deque = deque(maxlen=5000)
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        self._initialize_security_baselines()
        self._initialize_compliance_configs()
        
        logger.info("Advanced monitoring and security system initialized")
    
    def _initialize_security_baselines(self):
        """Initialize security monitoring baselines."""
        
        # Rate limiting configurations
        self.rate_limit_configs = {
            'api_requests': {'limit': 100, 'window': 60},  # 100 requests per minute
            'login_attempts': {'limit': 5, 'window': 300},  # 5 attempts per 5 minutes
            'query_requests': {'limit': 50, 'window': 60}   # 50 queries per minute
        }
        
        # Threat detection thresholds
        self.threat_thresholds = {
            'failed_auth_attempts': 5,
            'suspicious_query_patterns': 3,
            'unusual_access_patterns': 10,
            'data_exfiltration_indicators': 1
        }
        
        # Performance baselines (will be updated dynamically)
        self.baseline_metrics = {
            'response_time_ms': 1000.0,
            'memory_usage_mb': 512.0,
            'cpu_usage_percent': 50.0,
            'error_rate_percent': 1.0
        }
        
        # Anomaly detection thresholds (multiples of baseline)
        self.anomaly_thresholds = {
            'response_time_ms': 3.0,  # 3x baseline
            'memory_usage_mb': 2.0,   # 2x baseline
            'cpu_usage_percent': 1.5,  # 1.5x baseline
            'error_rate_percent': 5.0  # 5x baseline
        }
    
    def _initialize_compliance_configs(self):
        """Initialize compliance monitoring configurations."""
        
        self.compliance_configs = {
            ComplianceRegulation.GDPR: {
                'data_retention_days': 2555,  # 7 years
                'required_audit_fields': ['user_id', 'action', 'timestamp', 'ip_address'],
                'sensitive_data_patterns': [r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'],  # Credit card
                'breach_notification_hours': 72
            },
            ComplianceRegulation.CCPA: {
                'data_retention_days': 1095,  # 3 years
                'required_audit_fields': ['user_id', 'action', 'timestamp', 'data_category'],
                'consumer_rights': ['access', 'delete', 'portability', 'opt_out'],
                'breach_notification_hours': 72
            },
            ComplianceRegulation.PDPA: {
                'data_retention_days': 1825,  # 5 years
                'required_audit_fields': ['user_id', 'action', 'timestamp', 'consent_status'],
                'consent_requirements': ['explicit', 'informed', 'specific'],
                'breach_notification_hours': 72
            }
        }
    
    def start_monitoring(self):
        """Start comprehensive monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Advanced monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("Advanced monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Security monitoring
                self._perform_security_checks()
                
                # Performance monitoring
                self._perform_performance_checks()
                
                # Compliance monitoring
                self._perform_compliance_checks()
                
                # Cleanup old data
                self._cleanup_old_data()
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(60)  # Wait longer on error
    
    def record_security_event(self, 
                            event_type: SecurityEventType,
                            severity: AlertSeverity = AlertSeverity.WARNING,
                            source_ip: Optional[str] = None,
                            user_id: Optional[str] = None,
                            description: str = "",
                            metadata: Optional[Dict[str, Any]] = None) -> str:
        """Record a security event."""
        
        event_id = f"sec_{int(time.time())}_{len(self.security_events)}"
        
        event = SecurityEvent(
            event_id=event_id,
            event_type=event_type,
            severity=severity,
            timestamp=datetime.now(),
            source_ip=source_ip,
            user_id=user_id,
            description=description,
            metadata=metadata or {}
        )
        
        self.security_events.append(event)
        
        # Automated response
        self._handle_security_event(event)
        
        # Alert if necessary
        if severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL]:
            self._send_alert(f"Security Event: {description}", severity)
        
        logger.warning(f"Security event recorded: {event_type.value} - {description}")
        return event_id
    
    def _handle_security_event(self, event: SecurityEvent):
        """Automated security event handling."""
        
        # Rate limiting violations
        if event.event_type == SecurityEventType.RATE_LIMIT_EXCEEDED:
            if event.source_ip:
                self._apply_temporary_ip_block(event.source_ip, duration_minutes=15)
                event.response_actions.append(f"Temporarily blocked IP: {event.source_ip}")
        
        # Authentication failures
        elif event.event_type == SecurityEventType.AUTHENTICATION_FAILURE:
            if event.source_ip:
                # Increment failure count for IP
                ip_key = f"auth_failures_{event.source_ip}"
                current_failures = self.threat_indicators.get(ip_key, 0) + 1
                self.threat_indicators[ip_key] = current_failures
                
                if current_failures >= self.threat_thresholds['failed_auth_attempts']:
                    self._apply_temporary_ip_block(event.source_ip, duration_minutes=60)
                    event.response_actions.append(f"Blocked IP after {current_failures} auth failures")
        
        # Malicious input detected
        elif event.event_type == SecurityEventType.MALICIOUS_INPUT:
            if event.source_ip:
                self._apply_temporary_ip_block(event.source_ip, duration_minutes=30)
                event.response_actions.append("Blocked IP due to malicious input")
        
        # Data breach attempts
        elif event.event_type == SecurityEventType.DATA_BREACH_ATTEMPT:
            if event.source_ip:
                self._apply_permanent_ip_block(event.source_ip)
                event.response_actions.append("Permanently blocked IP due to data breach attempt")
            
            # Trigger incident response
            self._trigger_incident_response(event)
    
    def _apply_temporary_ip_block(self, ip: str, duration_minutes: int):
        """Apply temporary IP block."""
        self.blocked_ips.add(ip)
        
        # Schedule unblock
        def unblock_ip():
            time.sleep(duration_minutes * 60)
            if ip in self.blocked_ips:
                self.blocked_ips.remove(ip)
                logger.info(f"Temporary IP block expired: {ip}")
        
        threading.Thread(target=unblock_ip, daemon=True).start()
        logger.warning(f"Temporarily blocked IP: {ip} for {duration_minutes} minutes")
    
    def _apply_permanent_ip_block(self, ip: str):
        """Apply permanent IP block."""
        self.blocked_ips.add(ip)
        logger.critical(f"Permanently blocked IP: {ip}")
    
    def _trigger_incident_response(self, event: SecurityEvent):
        """Trigger automated incident response."""
        incident_id = f"inc_{int(time.time())}"
        
        # Log incident
        self.log_audit_event(
            user_id=event.user_id,
            action="SECURITY_INCIDENT",
            resource="SYSTEM",
            outcome="DETECTED",
            ip_address=event.source_ip,
            metadata={
                'incident_id': incident_id,
                'event_id': event.event_id,
                'event_type': event.event_type.value,
                'severity': event.severity.value
            }
        )
        
        # Send critical alert
        self._send_alert(f"SECURITY INCIDENT: {incident_id} - {event.description}", AlertSeverity.CRITICAL)
        
        logger.critical(f"Security incident triggered: {incident_id}")
    
    def check_rate_limit(self, identifier: str, limit_type: str = 'api_requests') -> bool:
        """Check if request is within rate limits."""
        if limit_type not in self.rate_limit_configs:
            return True
        
        config = self.rate_limit_configs[limit_type]
        limit = config['limit']
        window = config['window']
        
        now = time.time()
        
        # Initialize if not exists
        if identifier not in self.rate_limiters:
            self.rate_limiters[identifier] = {'requests': [], 'blocked_until': 0}
        
        limiter = self.rate_limiters[identifier]
        
        # Check if currently blocked
        if now < limiter['blocked_until']:
            return False
        
        # Clean old requests outside window
        limiter['requests'] = [req_time for req_time in limiter['requests'] if now - req_time < window]
        
        # Check if over limit
        if len(limiter['requests']) >= limit:
            # Block for window duration
            limiter['blocked_until'] = now + window
            
            # Record security event
            self.record_security_event(
                SecurityEventType.RATE_LIMIT_EXCEEDED,
                AlertSeverity.WARNING,
                source_ip=identifier if self._is_ip_address(identifier) else None,
                description=f"Rate limit exceeded for {limit_type}: {identifier}"
            )
            
            return False
        
        # Add current request
        limiter['requests'].append(now)
        return True
    
    def _is_ip_address(self, value: str) -> bool:
        """Check if value is a valid IP address."""
        try:
            ipaddress.ip_address(value)
            return True
        except ValueError:
            return False
    
    def record_performance_metric(self, 
                                metric_name: str, 
                                value: float, 
                                unit: str = "",
                                labels: Optional[Dict[str, str]] = None):
        """Record performance metric."""
        
        metric = PerformanceMetric(
            metric_name=metric_name,
            value=value,
            timestamp=datetime.now(),
            unit=unit,
            labels=labels or {}
        )
        
        self.performance_metrics.append(metric)
        
        # Update baseline if enough data
        self._update_performance_baseline(metric_name, value)
        
        # Check for anomalies
        self._check_performance_anomaly(metric_name, value)
    
    def _update_performance_baseline(self, metric_name: str, value: float):
        """Update performance baseline with exponential moving average."""
        if metric_name in self.baseline_metrics:
            # Exponential moving average with alpha=0.1
            current_baseline = self.baseline_metrics[metric_name]
            self.baseline_metrics[metric_name] = 0.9 * current_baseline + 0.1 * value
        else:
            self.baseline_metrics[metric_name] = value
    
    def _check_performance_anomaly(self, metric_name: str, value: float):
        """Check for performance anomalies."""
        if metric_name not in self.baseline_metrics or metric_name not in self.anomaly_thresholds:
            return
        
        baseline = self.baseline_metrics[metric_name]
        threshold_multiplier = self.anomaly_thresholds[metric_name]
        threshold = baseline * threshold_multiplier
        
        if value > threshold:
            severity = AlertSeverity.WARNING
            if value > threshold * 2:
                severity = AlertSeverity.ERROR
            if value > threshold * 5:
                severity = AlertSeverity.CRITICAL
            
            self._send_alert(
                f"Performance anomaly detected: {metric_name} = {value} (baseline: {baseline:.2f})",
                severity
            )
    
    def log_audit_event(self,
                       user_id: Optional[str],
                       action: str,
                       resource: str,
                       outcome: str,
                       ip_address: Optional[str] = None,
                       user_agent: Optional[str] = None,
                       metadata: Optional[Dict[str, Any]] = None):
        """Log audit event for compliance."""
        
        log_id = f"audit_{int(time.time())}_{len(self.audit_logs)}"
        
        audit_entry = AuditLogEntry(
            log_id=log_id,
            timestamp=datetime.now(),
            user_id=user_id,
            action=action,
            resource=resource,
            outcome=outcome,
            ip_address=ip_address,
            user_agent=user_agent,
            metadata=metadata or {}
        )
        
        self.audit_logs.append(audit_entry)
        
        # Check compliance requirements
        self._check_compliance_requirements(audit_entry)
        
        logger.info(f"Audit event logged: {action} on {resource} by {user_id} - {outcome}")
    
    def _check_compliance_requirements(self, audit_entry: AuditLogEntry):
        """Check audit entry against compliance requirements."""
        
        for regulation, config in self.compliance_configs.items():
            required_fields = config.get('required_audit_fields', [])
            
            # Check if all required fields are present
            missing_fields = []
            for field in required_fields:
                if not getattr(audit_entry, field, None):
                    missing_fields.append(field)
            
            if missing_fields:
                logger.warning(f"Compliance violation ({regulation.value}): Missing audit fields: {missing_fields}")
                
                self.record_security_event(
                    SecurityEventType.AUTHORIZATION_VIOLATION,
                    AlertSeverity.ERROR,
                    description=f"Compliance violation: Missing audit fields for {regulation.value}"
                )
    
    def _perform_security_checks(self):
        """Perform periodic security checks."""
        
        # Check for suspicious patterns in recent events
        recent_events = [e for e in self.security_events if (datetime.now() - e.timestamp).seconds < 300]
        
        # Detect unusual activity patterns
        ip_activity = defaultdict(int)
        user_activity = defaultdict(int)
        
        for event in recent_events:
            if event.source_ip:
                ip_activity[event.source_ip] += 1
            if event.user_id:
                user_activity[event.user_id] += 1
        
        # Alert on unusual activity
        for ip, count in ip_activity.items():
            if count > self.threat_thresholds['unusual_access_patterns']:
                self.record_security_event(
                    SecurityEventType.SUSPICIOUS_ACTIVITY,
                    AlertSeverity.WARNING,
                    source_ip=ip,
                    description=f"Unusual activity pattern: {count} events from IP {ip} in 5 minutes"
                )
    
    def _perform_performance_checks(self):
        """Perform periodic performance checks."""
        
        # Check recent metrics for trends
        recent_metrics = [m for m in self.performance_metrics if (datetime.now() - m.timestamp).seconds < 300]
        
        # Group by metric name
        metric_groups = defaultdict(list)
        for metric in recent_metrics:
            metric_groups[metric.metric_name].append(metric.value)
        
        # Check for sustained anomalies
        for metric_name, values in metric_groups.items():
            if len(values) >= 5:  # At least 5 data points
                avg_value = sum(values) / len(values)
                self._check_performance_anomaly(metric_name, avg_value)
    
    def _perform_compliance_checks(self):
        """Perform periodic compliance checks."""
        
        # Check data retention policies
        now = datetime.now()
        
        for regulation, config in self.compliance_configs.items():
            retention_days = config.get('data_retention_days', 2555)
            cutoff_date = now - timedelta(days=retention_days)
            
            # Count old audit logs that should be archived/deleted
            old_logs = [log for log in self.audit_logs if log.timestamp < cutoff_date]
            
            if old_logs:
                logger.info(f"Found {len(old_logs)} audit logs past retention period for {regulation.value}")
                # In a real system, these would be archived or deleted according to policy
    
    def _cleanup_old_data(self):
        """Clean up old monitoring data."""
        
        # Clean up old threat indicators
        now = time.time()
        expired_indicators = []
        
        for key, timestamp in self.threat_indicators.items():
            if isinstance(timestamp, (int, float)) and now - timestamp > 3600:  # 1 hour
                expired_indicators.append(key)
        
        for key in expired_indicators:
            del self.threat_indicators[key]
        
        # Clean up old rate limiter data
        for identifier, limiter in list(self.rate_limiters.items()):
            if now > limiter.get('blocked_until', 0) and not limiter.get('requests'):
                del self.rate_limiters[identifier]
    
    def _send_alert(self, message: str, severity: AlertSeverity):
        """Send alert through configured handlers."""
        
        alert_data = {
            'message': message,
            'severity': severity.value,
            'timestamp': datetime.now().isoformat(),
            'alert_id': f"alert_{int(time.time())}"
        }
        
        self.alert_history.append(alert_data)
        
        # Call registered alert handlers
        for handler in self.alert_handlers[severity]:
            try:
                handler(alert_data)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
        
        # Log the alert
        logger.log(
            logging.CRITICAL if severity == AlertSeverity.CRITICAL else
            logging.ERROR if severity == AlertSeverity.ERROR else
            logging.WARNING if severity == AlertSeverity.WARNING else
            logging.INFO,
            f"ALERT [{severity.value.upper()}]: {message}"
        )
    
    def add_alert_handler(self, severity: AlertSeverity, handler: Callable[[Dict[str, Any]], None]):
        """Add custom alert handler."""
        self.alert_handlers[severity].append(handler)
        logger.info(f"Added alert handler for {severity.value}")
    
    def is_ip_blocked(self, ip: str) -> bool:
        """Check if IP is blocked."""
        return ip in self.blocked_ips
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get comprehensive security summary."""
        
        recent_events = [e for e in self.security_events if (datetime.now() - e.timestamp).hours < 24]
        
        event_counts = defaultdict(int)
        for event in recent_events:
            event_counts[event.event_type.value] += 1
        
        return {
            'total_security_events_24h': len(recent_events),
            'event_breakdown': dict(event_counts),
            'blocked_ips_count': len(self.blocked_ips),
            'active_rate_limiters': len(self.rate_limiters),
            'threat_indicators_count': len(self.threat_indicators),
            'alerts_24h': len([a for a in self.alert_history if 
                             (datetime.now() - datetime.fromisoformat(a['timestamp'])).hours < 24]),
            'monitoring_active': self.monitoring_active,
            'timestamp': datetime.now().isoformat()
        }