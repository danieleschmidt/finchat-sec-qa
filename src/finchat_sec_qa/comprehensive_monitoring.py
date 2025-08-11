"""
Comprehensive System Monitoring and Observability.

This module provides enterprise-grade monitoring, alerting, and observability
features for the financial analysis platform.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from collections import defaultdict, deque
import psutil
import os

import numpy as np

from .config import get_config
from .logging_utils import configure_logging
from .metrics import get_business_tracker

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class Alert:
    """System alert data structure."""
    id: str
    timestamp: datetime
    level: AlertLevel
    component: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_time: Optional[datetime] = None


@dataclass
class MetricPoint:
    """Single metric data point."""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class HealthCheck:
    """Health check result."""
    component: str
    status: HealthStatus
    response_time: float
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class ComprehensiveMonitoring:
    """
    Comprehensive monitoring and observability system providing
    metrics collection, alerting, health checks, and system insights.
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path.home() / ".cache" / "finchat_sec_qa" / "monitoring"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Metrics storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.metric_aggregations: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Alerting
        self.alerts: deque = deque(maxlen=1000)
        self.alert_rules: List[Dict[str, Any]] = []
        self.notification_channels: List[Dict[str, Any]] = []
        
        # Health checks
        self.health_checks: Dict[str, Callable] = {}
        self.health_status: Dict[str, HealthCheck] = {}
        
        # System metrics
        self.system_metrics_enabled = True
        self.business_metrics: Dict[str, Any] = {}
        
        # Performance tracking
        self.performance_baselines: Dict[str, Dict[str, float]] = {}
        self.anomaly_thresholds: Dict[str, Dict[str, float]] = {}
        
        # Background monitoring
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        self._initialize_default_health_checks()
        self._initialize_default_alert_rules()
        configure_logging()

    def start_monitoring(self) -> None:
        """Start the comprehensive monitoring system."""
        if self._running:
            return
            
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop, 
            daemon=True
        )
        self._monitor_thread.start()
        logger.info("Comprehensive monitoring started")

    def stop_monitoring(self) -> None:
        """Stop the monitoring system."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Comprehensive monitoring stopped")

    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Run health checks
                self._run_health_checks()
                
                # Check alert conditions
                self._check_alert_conditions()
                
                # Update performance baselines
                self._update_performance_baselines()
                
                # Detect anomalies
                self._detect_anomalies()
                
                # Cleanup old data
                self._cleanup_old_data()
                
                # Sleep for 30 seconds
                time.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(10)  # Back off on error

    def record_metric(self, name: str, value: float, 
                     metric_type: MetricType = MetricType.GAUGE,
                     tags: Optional[Dict[str, str]] = None) -> None:
        """Record a metric point."""
        metric_point = MetricPoint(
            name=name,
            value=value,
            timestamp=datetime.now(),
            tags=tags or {},
            metric_type=metric_type
        )
        
        self.metrics[name].append(metric_point)
        
        # Update aggregations
        self._update_metric_aggregations(name, value)

    def record_timer(self, name: str, duration: float,
                    tags: Optional[Dict[str, str]] = None) -> None:
        """Record a timer metric."""
        self.record_metric(name, duration, MetricType.TIMER, tags)

    def increment_counter(self, name: str, value: float = 1.0,
                         tags: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric."""
        current_value = self.get_latest_metric_value(name, 0)
        self.record_metric(name, current_value + value, MetricType.COUNTER, tags)

    @contextmanager
    def timer_context(self, metric_name: str, 
                     tags: Optional[Dict[str, str]] = None):
        """Context manager for timing operations."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.record_timer(metric_name, duration, tags)

    def get_latest_metric_value(self, name: str, default: float = 0.0) -> float:
        """Get the latest value for a metric."""
        if name in self.metrics and self.metrics[name]:
            return self.metrics[name][-1].value
        return default

    def get_metric_statistics(self, name: str, 
                            time_window: Optional[timedelta] = None) -> Dict[str, float]:
        """Get statistics for a metric over a time window."""
        if name not in self.metrics:
            return {}
        
        metrics = list(self.metrics[name])
        
        # Filter by time window if specified
        if time_window:
            cutoff_time = datetime.now() - time_window
            metrics = [m for m in metrics if m.timestamp >= cutoff_time]
        
        if not metrics:
            return {}
        
        values = [m.value for m in metrics]
        
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'p95': np.percentile(values, 95),
            'p99': np.percentile(values, 99)
        }

    def register_health_check(self, name: str, check_func: Callable) -> None:
        """Register a health check function."""
        self.health_checks[name] = check_func

    def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status."""
        overall_status = HealthStatus.HEALTHY
        component_statuses = {}
        
        for component, health_check in self.health_status.items():
            component_statuses[component] = {
                'status': health_check.status.value,
                'response_time': health_check.response_time,
                'last_check': health_check.timestamp.isoformat(),
                'details': health_check.details
            }
            
            # Determine overall status
            if health_check.status == HealthStatus.CRITICAL:
                overall_status = HealthStatus.CRITICAL
            elif health_check.status == HealthStatus.UNHEALTHY and overall_status != HealthStatus.CRITICAL:
                overall_status = HealthStatus.UNHEALTHY
            elif health_check.status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                overall_status = HealthStatus.DEGRADED
        
        return {
            'overall_status': overall_status.value,
            'components': component_statuses,
            'timestamp': datetime.now().isoformat()
        }

    def create_alert(self, level: AlertLevel, component: str, 
                    message: str, details: Optional[Dict[str, Any]] = None) -> Alert:
        """Create a new alert."""
        alert = Alert(
            id=f"alert_{int(time.time() * 1000000)}",
            timestamp=datetime.now(),
            level=level,
            component=component,
            message=message,
            details=details or {}
        )
        
        self.alerts.append(alert)
        
        # Log alert
        log_message = f"Alert [{level.value.upper()}] {component}: {message}"
        if level == AlertLevel.CRITICAL:
            logger.critical(log_message)
        elif level == AlertLevel.ERROR:
            logger.error(log_message)
        elif level == AlertLevel.WARNING:
            logger.warning(log_message)
        else:
            logger.info(log_message)
        
        # Send notifications
        self._send_alert_notifications(alert)
        
        return alert

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert by ID."""
        for alert in self.alerts:
            if alert.id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolution_time = datetime.now()
                logger.info(f"Alert resolved: {alert_id}")
                return True
        return False

    def add_alert_rule(self, name: str, condition: str, 
                      alert_level: AlertLevel, component: str,
                      message: str) -> None:
        """Add an alert rule."""
        rule = {
            'name': name,
            'condition': condition,
            'alert_level': alert_level,
            'component': component,
            'message': message,
            'enabled': True
        }
        self.alert_rules.append(rule)

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active (unresolved) alerts."""
        return [
            {
                'id': alert.id,
                'timestamp': alert.timestamp.isoformat(),
                'level': alert.level.value,
                'component': alert.component,
                'message': alert.message,
                'details': alert.details
            }
            for alert in self.alerts if not alert.resolved
        ]

    def get_system_overview(self) -> Dict[str, Any]:
        """Get comprehensive system overview."""
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Application metrics
        recent_metrics = {}
        for name, points in self.metrics.items():
            if points:
                recent_metrics[name] = {
                    'current': points[-1].value,
                    'timestamp': points[-1].timestamp.isoformat()
                }
        
        # Performance indicators
        performance_score = self._calculate_performance_score()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'system': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'disk_percent': (disk.used / disk.total) * 100,
                'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
            },
            'application': recent_metrics,
            'health': self.get_health_status(),
            'alerts': {
                'active_count': len([a for a in self.alerts if not a.resolved]),
                'critical_count': len([a for a in self.alerts if not a.resolved and a.level == AlertLevel.CRITICAL])
            },
            'performance': {
                'score': performance_score,
                'baselines': self.performance_baselines
            }
        }

    def export_metrics(self, format: str = 'prometheus', 
                      time_window: Optional[timedelta] = None) -> str:
        """Export metrics in specified format."""
        if format.lower() == 'prometheus':
            return self._export_prometheus_format(time_window)
        elif format.lower() == 'json':
            return self._export_json_format(time_window)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    # Private methods
    def _collect_system_metrics(self) -> None:
        """Collect system-level metrics."""
        try:
            # CPU metrics
            self.record_metric('system.cpu.percent', psutil.cpu_percent())
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.record_metric('system.memory.percent', memory.percent)
            self.record_metric('system.memory.available_gb', memory.available / (1024**3))
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            self.record_metric('system.disk.percent', (disk.used / disk.total) * 100)
            self.record_metric('system.disk.free_gb', disk.free / (1024**3))
            
            # Process metrics
            process = psutil.Process()
            self.record_metric('process.cpu.percent', process.cpu_percent())
            self.record_metric('process.memory.rss_mb', process.memory_info().rss / (1024**2))
            self.record_metric('process.threads.count', process.num_threads())
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")

    def _initialize_default_health_checks(self) -> None:
        """Initialize default health checks."""
        self.register_health_check('system', self._system_health_check)
        self.register_health_check('database', self._database_health_check)
        self.register_health_check('external_apis', self._external_apis_health_check)
        self.register_health_check('quantum_system', self._quantum_system_health_check)

    def _system_health_check(self) -> HealthCheck:
        """System health check."""
        start_time = time.time()
        
        try:
            # Check system resources
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_percent = psutil.virtual_memory().percent
            disk_percent = (psutil.disk_usage('/').used / psutil.disk_usage('/').total) * 100
            
            # Determine status
            if cpu_percent > 90 or memory_percent > 90 or disk_percent > 95:
                status = HealthStatus.CRITICAL
            elif cpu_percent > 80 or memory_percent > 80 or disk_percent > 90:
                status = HealthStatus.UNHEALTHY
            elif cpu_percent > 70 or memory_percent > 70 or disk_percent > 85:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY
            
            return HealthCheck(
                component='system',
                status=status,
                response_time=time.time() - start_time,
                details={
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_percent,
                    'disk_percent': disk_percent
                }
            )
            
        except Exception as e:
            return HealthCheck(
                component='system',
                status=HealthStatus.CRITICAL,
                response_time=time.time() - start_time,
                details={'error': str(e)}
            )

    def _database_health_check(self) -> HealthCheck:
        """Database health check."""
        start_time = time.time()
        
        # Simulate database health check
        try:
            # In practice, this would check database connectivity
            response_time = time.time() - start_time
            
            if response_time > 5.0:
                status = HealthStatus.UNHEALTHY
            elif response_time > 2.0:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY
            
            return HealthCheck(
                component='database',
                status=status,
                response_time=response_time,
                details={'connection_pool_size': 10, 'active_connections': 3}
            )
            
        except Exception as e:
            return HealthCheck(
                component='database',
                status=HealthStatus.CRITICAL,
                response_time=time.time() - start_time,
                details={'error': str(e)}
            )

    def _external_apis_health_check(self) -> HealthCheck:
        """External APIs health check."""
        start_time = time.time()
        
        # Simulate external API health checks
        api_statuses = {
            'edgar_api': HealthStatus.HEALTHY,
            'openai_api': HealthStatus.HEALTHY,
            'quantum_api': HealthStatus.DEGRADED
        }
        
        # Determine overall status
        if any(status == HealthStatus.CRITICAL for status in api_statuses.values()):
            overall_status = HealthStatus.CRITICAL
        elif any(status == HealthStatus.UNHEALTHY for status in api_statuses.values()):
            overall_status = HealthStatus.UNHEALTHY
        elif any(status == HealthStatus.DEGRADED for status in api_statuses.values()):
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY
        
        return HealthCheck(
            component='external_apis',
            status=overall_status,
            response_time=time.time() - start_time,
            details={'api_statuses': {name: status.value for name, status in api_statuses.items()}}
        )

    def _quantum_system_health_check(self) -> HealthCheck:
        """Quantum system health check."""
        start_time = time.time()
        
        try:
            # Simulate quantum system check
            quantum_available = True  # Would check actual quantum system
            circuit_fidelity = 0.95   # Would get from quantum system
            
            if not quantum_available:
                status = HealthStatus.CRITICAL
            elif circuit_fidelity < 0.8:
                status = HealthStatus.UNHEALTHY
            elif circuit_fidelity < 0.9:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY
            
            return HealthCheck(
                component='quantum_system',
                status=status,
                response_time=time.time() - start_time,
                details={
                    'available': quantum_available,
                    'circuit_fidelity': circuit_fidelity,
                    'active_qubits': 32
                }
            )
            
        except Exception as e:
            return HealthCheck(
                component='quantum_system',
                status=HealthStatus.CRITICAL,
                response_time=time.time() - start_time,
                details={'error': str(e)}
            )

    def _run_health_checks(self) -> None:
        """Run all registered health checks."""
        for name, check_func in self.health_checks.items():
            try:
                health_result = check_func()
                self.health_status[name] = health_result
                
                # Record health check metrics
                self.record_metric(
                    f'healthcheck.{name}.response_time', 
                    health_result.response_time
                )
                
                # Create alerts for unhealthy components
                if health_result.status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
                    alert_level = AlertLevel.CRITICAL if health_result.status == HealthStatus.CRITICAL else AlertLevel.ERROR
                    self.create_alert(
                        alert_level,
                        name,
                        f"Health check failed: {health_result.status.value}",
                        health_result.details
                    )
                    
            except Exception as e:
                logger.error(f"Error running health check {name}: {e}")
                self.health_status[name] = HealthCheck(
                    component=name,
                    status=HealthStatus.CRITICAL,
                    response_time=0.0,
                    details={'error': str(e)}
                )

    def _initialize_default_alert_rules(self) -> None:
        """Initialize default alert rules."""
        self.add_alert_rule(
            'high_cpu_usage',
            'system.cpu.percent > 80',
            AlertLevel.WARNING,
            'system',
            'High CPU usage detected'
        )
        
        self.add_alert_rule(
            'critical_cpu_usage',
            'system.cpu.percent > 95',
            AlertLevel.CRITICAL,
            'system',
            'Critical CPU usage detected'
        )
        
        self.add_alert_rule(
            'high_memory_usage',
            'system.memory.percent > 85',
            AlertLevel.WARNING,
            'system',
            'High memory usage detected'
        )
        
        self.add_alert_rule(
            'low_disk_space',
            'system.disk.percent > 90',
            AlertLevel.ERROR,
            'system',
            'Low disk space detected'
        )

    def _check_alert_conditions(self) -> None:
        """Check all alert rule conditions."""
        for rule in self.alert_rules:
            if not rule['enabled']:
                continue
                
            try:
                condition = rule['condition']
                
                # Parse condition (simplified parser)
                if self._evaluate_condition(condition):
                    # Check if we already have an active alert for this rule
                    existing_alert = any(
                        not alert.resolved and 
                        alert.component == rule['component'] and
                        rule['name'] in alert.message
                        for alert in self.alerts
                    )
                    
                    if not existing_alert:
                        self.create_alert(
                            rule['alert_level'],
                            rule['component'],
                            f"{rule['name']}: {rule['message']}",
                            {'rule': rule['name']}
                        )
                        
            except Exception as e:
                logger.error(f"Error evaluating alert rule {rule['name']}: {e}")

    def _evaluate_condition(self, condition: str) -> bool:
        """Evaluate an alert condition."""
        try:
            # Simplified condition evaluation
            # In practice, this would be a more robust expression parser
            
            # Handle basic conditions like "metric_name > threshold"
            parts = condition.split()
            if len(parts) == 3:
                metric_name, operator, threshold_str = parts
                threshold = float(threshold_str)
                current_value = self.get_latest_metric_value(metric_name)
                
                if operator == '>':
                    return current_value > threshold
                elif operator == '<':
                    return current_value < threshold
                elif operator == '>=':
                    return current_value >= threshold
                elif operator == '<=':
                    return current_value <= threshold
                elif operator == '==':
                    return current_value == threshold
            
        except Exception as e:
            logger.error(f"Error evaluating condition '{condition}': {e}")
        
        return False

    def _update_metric_aggregations(self, name: str, value: float) -> None:
        """Update metric aggregations."""
        if name not in self.metric_aggregations:
            self.metric_aggregations[name] = {
                'count': 0,
                'sum': 0.0,
                'min': float('inf'),
                'max': float('-inf')
            }
        
        agg = self.metric_aggregations[name]
        agg['count'] += 1
        agg['sum'] += value
        agg['min'] = min(agg['min'], value)
        agg['max'] = max(agg['max'], value)
        agg['mean'] = agg['sum'] / agg['count']

    def _update_performance_baselines(self) -> None:
        """Update performance baselines based on recent metrics."""
        for metric_name, points in self.metrics.items():
            if len(points) >= 100:  # Need sufficient data
                recent_values = [p.value for p in list(points)[-100:]]
                
                if metric_name not in self.performance_baselines:
                    self.performance_baselines[metric_name] = {}
                
                baseline = self.performance_baselines[metric_name]
                baseline['mean'] = np.mean(recent_values)
                baseline['std'] = np.std(recent_values)
                baseline['p95'] = np.percentile(recent_values, 95)
                baseline['p99'] = np.percentile(recent_values, 99)

    def _detect_anomalies(self) -> None:
        """Detect anomalies in metrics."""
        for metric_name, baseline in self.performance_baselines.items():
            if metric_name in self.metrics and self.metrics[metric_name]:
                current_value = self.metrics[metric_name][-1].value
                
                # Simple anomaly detection based on standard deviations
                mean = baseline['mean']
                std = baseline['std']
                
                if abs(current_value - mean) > 3 * std:  # 3-sigma rule
                    self.create_alert(
                        AlertLevel.WARNING,
                        'anomaly_detection',
                        f'Anomaly detected in {metric_name}',
                        {
                            'current_value': current_value,
                            'baseline_mean': mean,
                            'deviation': abs(current_value - mean) / std
                        }
                    )

    def _calculate_performance_score(self) -> float:
        """Calculate overall performance score."""
        scores = []
        
        # CPU score (lower is better)
        cpu_percent = self.get_latest_metric_value('system.cpu.percent', 50)
        cpu_score = max(0, 1 - (cpu_percent / 100))
        scores.append(cpu_score)
        
        # Memory score (lower is better)
        memory_percent = self.get_latest_metric_value('system.memory.percent', 50)
        memory_score = max(0, 1 - (memory_percent / 100))
        scores.append(memory_score)
        
        # Response time score (lower is better)
        response_time = self.get_latest_metric_value('response_time', 1.0)
        response_score = max(0, 1 - min(response_time / 5.0, 1.0))  # Cap at 5 seconds
        scores.append(response_score)
        
        # Error rate score (lower is better)
        error_rate = self.get_latest_metric_value('error_rate', 0.01)
        error_score = max(0, 1 - min(error_rate * 10, 1.0))  # Cap at 10%
        scores.append(error_score)
        
        return np.mean(scores) * 100  # Scale to 0-100

    def _send_alert_notifications(self, alert: Alert) -> None:
        """Send alert notifications."""
        # In practice, this would send to configured notification channels
        logger.info(f"Alert notification: {alert.level.value} - {alert.message}")

    def _cleanup_old_data(self) -> None:
        """Clean up old monitoring data."""
        # Alerts older than 30 days
        cutoff_time = datetime.now() - timedelta(days=30)
        self.alerts = deque(
            [alert for alert in self.alerts if alert.timestamp >= cutoff_time],
            maxlen=1000
        )

    def _export_prometheus_format(self, time_window: Optional[timedelta]) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        for metric_name, points in self.metrics.items():
            if not points:
                continue
            
            # Filter by time window
            filtered_points = points
            if time_window:
                cutoff_time = datetime.now() - time_window
                filtered_points = [p for p in points if p.timestamp >= cutoff_time]
            
            if not filtered_points:
                continue
            
            # Use latest value for gauge metrics
            latest_point = filtered_points[-1]
            
            # Format metric name for Prometheus
            prom_name = metric_name.replace('.', '_')
            
            # Add help and type
            lines.append(f'# HELP {prom_name} {metric_name}')
            lines.append(f'# TYPE {prom_name} gauge')
            
            # Add metric with tags
            tag_str = ''
            if latest_point.tags:
                tag_pairs = [f'{k}="{v}"' for k, v in latest_point.tags.items()]
                tag_str = '{' + ','.join(tag_pairs) + '}'
            
            lines.append(f'{prom_name}{tag_str} {latest_point.value}')
        
        return '\n'.join(lines)

    def _export_json_format(self, time_window: Optional[timedelta]) -> str:
        """Export metrics in JSON format."""
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'metrics': {}
        }
        
        for metric_name, points in self.metrics.items():
            if not points:
                continue
            
            filtered_points = points
            if time_window:
                cutoff_time = datetime.now() - time_window
                filtered_points = [p for p in points if p.timestamp >= cutoff_time]
            
            if not filtered_points:
                continue
            
            export_data['metrics'][metric_name] = [
                {
                    'value': p.value,
                    'timestamp': p.timestamp.isoformat(),
                    'tags': p.tags
                }
                for p in filtered_points
            ]
        
        return json.dumps(export_data, indent=2)


# Global monitoring instance
_global_monitoring: Optional[ComprehensiveMonitoring] = None


def get_monitoring() -> ComprehensiveMonitoring:
    """Get the global monitoring instance."""
    global _global_monitoring
    if _global_monitoring is None:
        _global_monitoring = ComprehensiveMonitoring()
        _global_monitoring.start_monitoring()
    return _global_monitoring