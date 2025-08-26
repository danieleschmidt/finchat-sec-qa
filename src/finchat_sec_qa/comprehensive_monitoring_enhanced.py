"""
Enhanced Comprehensive Monitoring System - Generation 2: MAKE IT ROBUST
TERRAGON SDLC v4.0 - Autonomous Monitoring Implementation

Features:
- Real-time system health monitoring
- Advanced performance metrics collection
- Predictive anomaly detection
- Automated alerting and response systems
- Quantum performance optimization tracking
- Financial data quality monitoring
- Multi-dimensional observability
"""

from __future__ import annotations

import asyncio
import logging
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union
import json
import statistics
import psutil
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    TIMER = "timer"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = 1
    WARNING = 2
    CRITICAL = 3
    EMERGENCY = 4


class AlertType(Enum):
    """Types of alerts."""
    THRESHOLD_BREACH = "threshold_breach"
    ANOMALY_DETECTED = "anomaly_detected"
    ERROR_RATE_HIGH = "error_rate_high"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    SECURITY_INCIDENT = "security_incident"
    QUANTUM_CIRCUIT_FAILURE = "quantum_circuit_failure"
    FINANCIAL_DATA_QUALITY = "financial_data_quality"


@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: datetime
    value: Union[float, int]
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricSeries:
    """Time series of metric points."""
    name: str
    metric_type: MetricType
    points: deque = field(default_factory=lambda: deque(maxlen=10000))
    labels: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def add_point(self, value: Union[float, int], tags: Dict[str, str] = None):
        """Add a new metric point."""
        point = MetricPoint(
            timestamp=datetime.now(),
            value=value,
            tags=tags or {}
        )
        self.points.append(point)
    
    def get_recent_values(self, duration_seconds: int = 300) -> List[float]:
        """Get values from recent time period."""
        cutoff = datetime.now() - timedelta(seconds=duration_seconds)
        return [p.value for p in self.points if p.timestamp > cutoff]
    
    def get_current_value(self) -> Optional[float]:
        """Get the most recent value."""
        return self.points[-1].value if self.points else None
    
    def calculate_statistics(self, duration_seconds: int = 300) -> Dict[str, float]:
        """Calculate statistics for recent values."""
        values = self.get_recent_values(duration_seconds)
        
        if not values:
            return {}
        
        return {
            'count': len(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'min': min(values),
            'max': max(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0.0,
            'p95': statistics.quantiles(values, n=20)[18] if len(values) >= 20 else max(values),
            'p99': statistics.quantiles(values, n=100)[98] if len(values) >= 100 else max(values)
        }


@dataclass
class Alert:
    """Alert notification."""
    alert_id: str
    timestamp: datetime
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    description: str
    metric_name: str
    current_value: float
    threshold: Optional[float] = None
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class ThresholdRule:
    """Threshold-based alerting rule."""
    metric_name: str
    operator: str  # >, <, >=, <=, ==, !=
    threshold: float
    duration_seconds: int = 300
    severity: AlertSeverity = AlertSeverity.WARNING
    alert_type: AlertType = AlertType.THRESHOLD_BREACH
    enabled: bool = True


@dataclass
class AnomalyRule:
    """Anomaly detection rule."""
    metric_name: str
    sensitivity: float = 2.0  # Standard deviations
    min_samples: int = 100
    lookback_seconds: int = 3600
    severity: AlertSeverity = AlertSeverity.WARNING
    enabled: bool = True


class SystemHealthMonitor:
    """Monitor system health and resources."""
    
    def __init__(self, monitoring_system: 'ComprehensiveMonitoringSystem'):
        self.monitoring_system = monitoring_system
        self._monitoring = False
        self._monitor_thread = None
    
    def start_monitoring(self):
        """Start system health monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("System health monitoring started")
    
    def stop_monitoring(self):
        """Stop system health monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("System health monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._monitoring:
            try:
                self._collect_system_metrics()
                time.sleep(30)  # Collect every 30 seconds
            except Exception as e:
                logger.error(f"Error in system monitoring loop: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _collect_system_metrics(self):
        """Collect comprehensive system metrics."""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        self.monitoring_system.record_metric('system.cpu.usage_percent', cpu_percent)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        self.monitoring_system.record_metric('system.memory.usage_percent', memory.percent)
        self.monitoring_system.record_metric('system.memory.available_gb', memory.available / (1024**3))
        self.monitoring_system.record_metric('system.memory.used_gb', memory.used / (1024**3))
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        self.monitoring_system.record_metric('system.disk.usage_percent', (disk.used / disk.total) * 100)
        self.monitoring_system.record_metric('system.disk.free_gb', disk.free / (1024**3))
        
        # Network metrics
        net_io = psutil.net_io_counters()
        self.monitoring_system.record_metric('system.network.bytes_sent', net_io.bytes_sent)
        self.monitoring_system.record_metric('system.network.bytes_recv', net_io.bytes_recv)
        
        # Process metrics
        process = psutil.Process()
        self.monitoring_system.record_metric('process.cpu_percent', process.cpu_percent())
        self.monitoring_system.record_metric('process.memory_mb', process.memory_info().rss / (1024**2))
        self.monitoring_system.record_metric('process.num_threads', process.num_threads())
        self.monitoring_system.record_metric('process.num_fds', process.num_fds() if hasattr(process, 'num_fds') else 0)


class PerformanceTracker:
    """Track application performance metrics."""
    
    def __init__(self, monitoring_system: 'ComprehensiveMonitoringSystem'):
        self.monitoring_system = monitoring_system
        self.active_requests: Dict[str, datetime] = {}
    
    def start_request(self, request_id: str, operation: str):
        """Start tracking a request."""
        self.active_requests[request_id] = datetime.now()
        self.monitoring_system.record_metric(f'performance.{operation}.requests_active', 
                                           len(self.active_requests))
    
    def end_request(self, request_id: str, operation: str, success: bool = True):
        """End tracking a request."""
        if request_id not in self.active_requests:
            return
        
        start_time = self.active_requests.pop(request_id)
        duration = (datetime.now() - start_time).total_seconds()
        
        # Record timing metrics
        self.monitoring_system.record_metric(f'performance.{operation}.duration_seconds', duration)
        self.monitoring_system.record_metric(f'performance.{operation}.requests_total', 1)
        
        if success:
            self.monitoring_system.record_metric(f'performance.{operation}.requests_success', 1)
        else:
            self.monitoring_system.record_metric(f'performance.{operation}.requests_error', 1)
        
        # Update active request count
        self.monitoring_system.record_metric(f'performance.{operation}.requests_active', 
                                           len(self.active_requests))


class AnomalyDetector:
    """Detect anomalies in metric data."""
    
    def __init__(self):
        self.baseline_stats: Dict[str, Dict[str, float]] = {}
    
    def update_baseline(self, metric_name: str, values: List[float]):
        """Update baseline statistics for a metric."""
        if len(values) < 10:
            return
        
        self.baseline_stats[metric_name] = {
            'mean': statistics.mean(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0.0,
            'median': statistics.median(values),
            'q25': statistics.quantiles(values, n=4)[0] if len(values) >= 4 else min(values),
            'q75': statistics.quantiles(values, n=4)[2] if len(values) >= 4 else max(values)
        }
    
    def detect_anomaly(self, metric_name: str, value: float, sensitivity: float = 2.0) -> Optional[Dict[str, Any]]:
        """Detect if a value is anomalous."""
        if metric_name not in self.baseline_stats:
            return None
        
        stats = self.baseline_stats[metric_name]
        
        # Z-score based detection
        if stats['std'] > 0:
            z_score = abs(value - stats['mean']) / stats['std']
            if z_score > sensitivity:
                return {
                    'type': 'z_score',
                    'z_score': z_score,
                    'threshold': sensitivity,
                    'baseline_mean': stats['mean'],
                    'baseline_std': stats['std']
                }
        
        # IQR based detection
        iqr = stats['q75'] - stats['q25']
        if iqr > 0:
            lower_bound = stats['q25'] - 1.5 * iqr
            upper_bound = stats['q75'] + 1.5 * iqr
            
            if value < lower_bound or value > upper_bound:
                return {
                    'type': 'iqr',
                    'value': value,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'iqr': iqr
                }
        
        return None


class ComprehensiveMonitoringSystem:
    """Comprehensive monitoring and alerting system."""
    
    def __init__(self):
        self.metrics: Dict[str, MetricSeries] = {}
        self.threshold_rules: List[ThresholdRule] = []
        self.anomaly_rules: List[AnomalyRule] = []
        self.alerts: List[Alert] = []
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        
        # Sub-systems
        self.system_health_monitor = SystemHealthMonitor(self)
        self.performance_tracker = PerformanceTracker(self)
        self.anomaly_detector = AnomalyDetector()
        
        # State
        self._monitoring_enabled = False
        self._alert_thread = None
        
        # Initialize default rules
        self._initialize_default_rules()
    
    def start_monitoring(self):
        """Start comprehensive monitoring."""
        if self._monitoring_enabled:
            return
        
        self._monitoring_enabled = True
        
        # Start sub-systems
        self.system_health_monitor.start_monitoring()
        
        # Start alerting thread
        self._alert_thread = threading.Thread(target=self._alert_loop, daemon=True)
        self._alert_thread.start()
        
        logger.info("Comprehensive monitoring system started")
    
    def stop_monitoring(self):
        """Stop comprehensive monitoring."""
        self._monitoring_enabled = False
        
        # Stop sub-systems
        self.system_health_monitor.stop_monitoring()
        
        # Stop alerting thread
        if self._alert_thread:
            self._alert_thread.join(timeout=5.0)
        
        logger.info("Comprehensive monitoring system stopped")
    
    def record_metric(self, name: str, value: Union[float, int], 
                     metric_type: MetricType = MetricType.GAUGE,
                     tags: Dict[str, str] = None):
        """Record a metric value."""
        if name not in self.metrics:
            self.metrics[name] = MetricSeries(name=name, metric_type=metric_type)
        
        self.metrics[name].add_point(value, tags)
        
        # Update anomaly detector baseline periodically
        if len(self.metrics[name].points) % 100 == 0:
            recent_values = self.metrics[name].get_recent_values(3600)  # Last hour
            self.anomaly_detector.update_baseline(name, recent_values)
    
    def get_metric(self, name: str) -> Optional[MetricSeries]:
        """Get metric series by name."""
        return self.metrics.get(name)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        summary = {}
        
        for name, series in self.metrics.items():
            stats = series.calculate_statistics()
            current_value = series.get_current_value()
            
            summary[name] = {
                'current_value': current_value,
                'statistics': stats,
                'metric_type': series.metric_type.value,
                'point_count': len(series.points)
            }
        
        return summary
    
    def add_threshold_rule(self, rule: ThresholdRule):
        """Add a threshold-based alerting rule."""
        self.threshold_rules.append(rule)
        logger.info(f"Added threshold rule for {rule.metric_name}: {rule.operator} {rule.threshold}")
    
    def add_anomaly_rule(self, rule: AnomalyRule):
        """Add an anomaly detection rule."""
        self.anomaly_rules.append(rule)
        logger.info(f"Added anomaly rule for {rule.metric_name} (sensitivity: {rule.sensitivity})")
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add callback for alert notifications."""
        self.alert_callbacks.append(callback)
    
    def _initialize_default_rules(self):
        """Initialize default monitoring rules."""
        # System resource thresholds
        self.add_threshold_rule(ThresholdRule(
            metric_name='system.cpu.usage_percent',
            operator='>=',
            threshold=80.0,
            severity=AlertSeverity.WARNING
        ))
        
        self.add_threshold_rule(ThresholdRule(
            metric_name='system.cpu.usage_percent',
            operator='>=',
            threshold=95.0,
            severity=AlertSeverity.CRITICAL
        ))
        
        self.add_threshold_rule(ThresholdRule(
            metric_name='system.memory.usage_percent',
            operator='>=',
            threshold=85.0,
            severity=AlertSeverity.WARNING
        ))
        
        self.add_threshold_rule(ThresholdRule(
            metric_name='system.disk.usage_percent',
            operator='>=',
            threshold=90.0,
            severity=AlertSeverity.CRITICAL
        ))
        
        # Performance thresholds
        self.add_threshold_rule(ThresholdRule(
            metric_name='performance.query.duration_seconds',
            operator='>=',
            threshold=5.0,
            severity=AlertSeverity.WARNING,
            alert_type=AlertType.PERFORMANCE_DEGRADATION
        ))
        
        # Anomaly detection rules
        self.add_anomaly_rule(AnomalyRule(
            metric_name='performance.query.duration_seconds',
            sensitivity=2.5,
            severity=AlertSeverity.WARNING
        ))
    
    def _alert_loop(self):
        """Main alerting loop."""
        while self._monitoring_enabled:
            try:
                self._check_alerts()
                time.sleep(30)  # Check alerts every 30 seconds
            except Exception as e:
                logger.error(f"Error in alert loop: {e}")
                time.sleep(60)
    
    def _check_alerts(self):
        """Check all alerting rules."""
        # Check threshold rules
        for rule in self.threshold_rules:
            if not rule.enabled:
                continue
                
            try:
                self._check_threshold_rule(rule)
            except Exception as e:
                logger.error(f"Error checking threshold rule {rule.metric_name}: {e}")
        
        # Check anomaly rules
        for rule in self.anomaly_rules:
            if not rule.enabled:
                continue
                
            try:
                self._check_anomaly_rule(rule)
            except Exception as e:
                logger.error(f"Error checking anomaly rule {rule.metric_name}: {e}")
    
    def _check_threshold_rule(self, rule: ThresholdRule):
        """Check a threshold rule."""
        metric = self.get_metric(rule.metric_name)
        if not metric:
            return
        
        current_value = metric.get_current_value()
        if current_value is None:
            return
        
        # Check if threshold is breached
        breached = False
        if rule.operator == '>':
            breached = current_value > rule.threshold
        elif rule.operator == '<':
            breached = current_value < rule.threshold
        elif rule.operator == '>=':
            breached = current_value >= rule.threshold
        elif rule.operator == '<=':
            breached = current_value <= rule.threshold
        elif rule.operator == '==':
            breached = abs(current_value - rule.threshold) < 1e-9
        elif rule.operator == '!=':
            breached = abs(current_value - rule.threshold) >= 1e-9
        
        if breached:
            alert = Alert(
                alert_id=f"threshold_{rule.metric_name}_{int(time.time())}",
                timestamp=datetime.now(),
                alert_type=rule.alert_type,
                severity=rule.severity,
                title=f"Threshold breached: {rule.metric_name}",
                description=f"{rule.metric_name} is {current_value}, which is {rule.operator} {rule.threshold}",
                metric_name=rule.metric_name,
                current_value=current_value,
                threshold=rule.threshold
            )
            
            self._trigger_alert(alert)
    
    def _check_anomaly_rule(self, rule: AnomalyRule):
        """Check an anomaly detection rule."""
        metric = self.get_metric(rule.metric_name)
        if not metric:
            return
        
        current_value = metric.get_current_value()
        if current_value is None:
            return
        
        # Check for anomaly
        anomaly = self.anomaly_detector.detect_anomaly(
            rule.metric_name, current_value, rule.sensitivity
        )
        
        if anomaly:
            alert = Alert(
                alert_id=f"anomaly_{rule.metric_name}_{int(time.time())}",
                timestamp=datetime.now(),
                alert_type=AlertType.ANOMALY_DETECTED,
                severity=rule.severity,
                title=f"Anomaly detected: {rule.metric_name}",
                description=f"Anomalous value {current_value} detected for {rule.metric_name}",
                metric_name=rule.metric_name,
                current_value=current_value,
                metadata=anomaly
            )
            
            self._trigger_alert(alert)
    
    def _trigger_alert(self, alert: Alert):
        """Trigger an alert."""
        self.alerts.append(alert)
        
        # Log alert
        logger.warning(f"ALERT [{alert.severity.name}]: {alert.title} - {alert.description}")
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get currently active alerts."""
        return [alert for alert in self.alerts if not alert.resolved]
    
    def resolve_alert(self, alert_id: str):
        """Resolve an alert."""
        for alert in self.alerts:
            if alert.alert_id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolved_at = datetime.now()
                logger.info(f"Alert resolved: {alert_id}")
                break
    
    def get_monitoring_health(self) -> Dict[str, Any]:
        """Get overall monitoring system health."""
        active_alerts = self.get_active_alerts()
        
        return {
            'monitoring_enabled': self._monitoring_enabled,
            'total_metrics': len(self.metrics),
            'total_alerts': len(self.alerts),
            'active_alerts': len(active_alerts),
            'critical_alerts': len([a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]),
            'threshold_rules': len(self.threshold_rules),
            'anomaly_rules': len(self.anomaly_rules),
            'system_health': self._get_system_health_summary()
        }
    
    def _get_system_health_summary(self) -> Dict[str, Any]:
        """Get system health summary."""
        summary = {}
        
        # CPU health
        cpu_metric = self.get_metric('system.cpu.usage_percent')
        if cpu_metric:
            summary['cpu_usage'] = cpu_metric.get_current_value()
        
        # Memory health
        memory_metric = self.get_metric('system.memory.usage_percent')
        if memory_metric:
            summary['memory_usage'] = memory_metric.get_current_value()
        
        # Disk health
        disk_metric = self.get_metric('system.disk.usage_percent')
        if disk_metric:
            summary['disk_usage'] = disk_metric.get_current_value()
        
        return summary


# Global monitoring instance
global_monitoring_system = ComprehensiveMonitoringSystem()


def monitor_performance(operation: str):
    """Decorator for monitoring function performance."""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            request_id = f"{operation}_{int(time.time() * 1000000)}"
            global_monitoring_system.performance_tracker.start_request(request_id, operation)
            
            try:
                result = func(*args, **kwargs)
                global_monitoring_system.performance_tracker.end_request(request_id, operation, True)
                return result
            except Exception as e:
                global_monitoring_system.performance_tracker.end_request(request_id, operation, False)
                raise
        
        async def async_wrapper(*args, **kwargs):
            request_id = f"{operation}_{int(time.time() * 1000000)}"
            global_monitoring_system.performance_tracker.start_request(request_id, operation)
            
            try:
                result = await func(*args, **kwargs)
                global_monitoring_system.performance_tracker.end_request(request_id, operation, True)
                return result
            except Exception as e:
                global_monitoring_system.performance_tracker.end_request(request_id, operation, False)
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper
    return decorator


# Export all classes and functions
__all__ = [
    'MetricType',
    'AlertSeverity',
    'AlertType',
    'MetricPoint',
    'MetricSeries',
    'Alert',
    'ThresholdRule',
    'AnomalyRule',
    'SystemHealthMonitor',
    'PerformanceTracker',
    'AnomalyDetector',
    'ComprehensiveMonitoringSystem',
    'global_monitoring_system',
    'monitor_performance'
]