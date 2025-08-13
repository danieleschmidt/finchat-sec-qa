"""
Advanced Monitoring and Observability for Quantum Financial Algorithms.

This module provides comprehensive monitoring, logging, metrics collection,
and observability features for quantum financial algorithms in production
environments.

PRODUCTION MONITORING - Enterprise-Grade Observability
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
import time
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
import threading
import warnings

import numpy as np

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)


class MetricType(Enum):
    """Types of metrics to collect."""
    
    COUNTER = "counter"              # Monotonic increasing counter
    GAUGE = "gauge"                  # Current value metric
    HISTOGRAM = "histogram"          # Distribution of values
    TIMER = "timer"                  # Time duration measurements
    QUANTUM_FIDELITY = "quantum_fidelity"    # Quantum-specific metrics
    QUANTUM_ADVANTAGE = "quantum_advantage"  # Quantum advantage measurements


class AlertSeverity(Enum):
    """Alert severity levels."""
    
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Metric:
    """Individual metric measurement."""
    
    name: str
    metric_type: MetricType
    value: Union[float, int]
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    unit: str = ""
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary for serialization."""
        return {
            'name': self.name,
            'type': self.metric_type.value,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'labels': self.labels,
            'unit': self.unit,
            'description': self.description
        }


@dataclass
class Alert:
    """System alert."""
    
    alert_id: str
    severity: AlertSeverity
    title: str
    description: str
    timestamp: datetime
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            'alert_id': self.alert_id,
            'severity': self.severity.value,
            'title': self.title,
            'description': self.description,
            'timestamp': self.timestamp.isoformat(),
            'metric_name': self.metric_name,
            'metric_value': self.metric_value,
            'threshold': self.threshold,
            'resolved': self.resolved,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None
        }


@dataclass
class PerformanceTrace:
    """Performance tracing information."""
    
    trace_id: str
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    quantum_algorithm: Optional[str] = None
    quantum_advantage: Optional[float] = None
    circuit_depth: Optional[int] = None
    fidelity: Optional[float] = None
    error_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def finish(self):
        """Mark trace as finished and calculate duration."""
        self.end_time = datetime.now()
        if self.start_time:
            self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trace to dictionary."""
        return {
            'trace_id': self.trace_id,
            'operation_name': self.operation_name,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_ms': self.duration_ms,
            'quantum_algorithm': self.quantum_algorithm,
            'quantum_advantage': self.quantum_advantage,
            'circuit_depth': self.circuit_depth,
            'fidelity': self.fidelity,
            'error_count': self.error_count,
            'metadata': self.metadata
        }


class MetricsCollector:
    """
    Collects and aggregates metrics from quantum financial algorithms.
    
    Provides thread-safe metrics collection with automatic aggregation
    and export capabilities for monitoring systems.
    """
    
    def __init__(self, buffer_size: int = 10000):
        """Initialize metrics collector."""
        self.buffer_size = buffer_size
        self.metrics_buffer: List[Metric] = []
        self.metric_aggregates: Dict[str, Dict[str, float]] = {}
        self.lock = threading.Lock()
        
        # Performance counters
        self.counters: Dict[str, int] = {}
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = {}
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info("Initialized metrics collector")
    
    def increment_counter(self, name: str, value: int = 1, labels: Dict[str, str] = None):
        """Increment a counter metric."""
        with self.lock:
            self.counters[name] = self.counters.get(name, 0) + value
            
        metric = Metric(
            name=name,
            metric_type=MetricType.COUNTER,
            value=self.counters[name],
            timestamp=datetime.now(),
            labels=labels or {},
            description=f"Counter for {name}"
        )
        
        self._add_metric(metric)
    
    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set a gauge metric."""
        with self.lock:
            self.gauges[name] = value
            
        metric = Metric(
            name=name,
            metric_type=MetricType.GAUGE,
            value=value,
            timestamp=datetime.now(),
            labels=labels or {},
            description=f"Gauge for {name}"
        )
        
        self._add_metric(metric)
    
    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a histogram metric."""
        with self.lock:
            if name not in self.histograms:
                self.histograms[name] = []
            self.histograms[name].append(value)
            
            # Keep only recent values to prevent unlimited growth
            if len(self.histograms[name]) > 1000:
                self.histograms[name] = self.histograms[name][-1000:]
        
        metric = Metric(
            name=name,
            metric_type=MetricType.HISTOGRAM,
            value=value,
            timestamp=datetime.now(),
            labels=labels or {},
            description=f"Histogram for {name}"
        )
        
        self._add_metric(metric)
    
    def record_timer(self, name: str, duration_ms: float, labels: Dict[str, str] = None):
        """Record a timer metric."""
        metric = Metric(
            name=name,
            metric_type=MetricType.TIMER,
            value=duration_ms,
            timestamp=datetime.now(),
            labels=labels or {},
            unit="ms",
            description=f"Timer for {name}"
        )
        
        self._add_metric(metric)
        
        # Also record as histogram for statistical analysis
        self.record_histogram(f"{name}_histogram", duration_ms, labels)
    
    def record_quantum_fidelity(self, name: str, fidelity: float, labels: Dict[str, str] = None):
        """Record quantum fidelity metric."""
        metric = Metric(
            name=name,
            metric_type=MetricType.QUANTUM_FIDELITY,
            value=fidelity,
            timestamp=datetime.now(),
            labels=labels or {},
            unit="fidelity",
            description=f"Quantum fidelity for {name}"
        )
        
        self._add_metric(metric)
    
    def record_quantum_advantage(self, name: str, advantage: float, labels: Dict[str, str] = None):
        """Record quantum advantage metric."""
        metric = Metric(
            name=name,
            metric_type=MetricType.QUANTUM_ADVANTAGE,
            value=advantage,
            timestamp=datetime.now(),
            labels=labels or {},
            unit="ratio",
            description=f"Quantum advantage for {name}"
        )
        
        self._add_metric(metric)
    
    def _add_metric(self, metric: Metric):
        """Add metric to buffer."""
        with self.lock:
            self.metrics_buffer.append(metric)
            
            # Prevent buffer overflow
            if len(self.metrics_buffer) > self.buffer_size:
                self.metrics_buffer = self.metrics_buffer[-self.buffer_size//2:]
    
    def get_metric_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics."""
        with self.lock:
            summary = {
                'counters': dict(self.counters),
                'gauges': dict(self.gauges),
                'histogram_stats': {},
                'buffer_size': len(self.metrics_buffer),
                'collection_timestamp': datetime.now().isoformat()
            }
            
            # Calculate histogram statistics
            for name, values in self.histograms.items():
                if values:
                    summary['histogram_stats'][name] = {
                        'count': len(values),
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'p50': np.percentile(values, 50),
                        'p95': np.percentile(values, 95),
                        'p99': np.percentile(values, 99)
                    }
        
        return summary
    
    def export_metrics(self, format_type: str = "json") -> str:
        """Export metrics in specified format."""
        if format_type == "json":
            return self._export_json()
        elif format_type == "prometheus":
            return self._export_prometheus()
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def _export_json(self) -> str:
        """Export metrics as JSON."""
        with self.lock:
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'metrics': [metric.to_dict() for metric in self.metrics_buffer[-1000:]],  # Last 1000
                'summary': self.get_metric_summary()
            }
        
        return json.dumps(export_data, indent=2)
    
    def _export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        with self.lock:
            # Export counters
            for name, value in self.counters.items():
                lines.append(f"# HELP {name} Counter metric")
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{name} {value}")
                lines.append("")
            
            # Export gauges
            for name, value in self.gauges.items():
                lines.append(f"# HELP {name} Gauge metric")
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name} {value}")
                lines.append("")
            
            # Export histogram summaries
            for name, values in self.histograms.items():
                if values:
                    lines.append(f"# HELP {name} Histogram metric")
                    lines.append(f"# TYPE {name} histogram")
                    lines.append(f"{name}_count {len(values)}")
                    lines.append(f"{name}_sum {sum(values)}")
                    
                    # Percentiles as separate metrics
                    lines.append(f"{name}_p50 {np.percentile(values, 50)}")
                    lines.append(f"{name}_p95 {np.percentile(values, 95)}")
                    lines.append(f"{name}_p99 {np.percentile(values, 99)}")
                    lines.append("")
        
        return "\\n".join(lines)


class AlertManager:
    """
    Manages alerts and notifications for quantum algorithms.
    
    Provides configurable alerting with thresholds and notification
    capabilities for production monitoring.
    """
    
    def __init__(self):
        """Initialize alert manager."""
        self.alerts: List[Alert] = []
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.notification_handlers: List[Callable[[Alert], None]] = []
        self.lock = threading.Lock()
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info("Initialized alert manager")
    
    def add_alert_rule(self, 
                      metric_name: str, 
                      threshold: float,
                      condition: str = "greater_than",
                      severity: AlertSeverity = AlertSeverity.WARNING,
                      description: str = ""):
        """Add alert rule for metric monitoring."""
        rule = {
            'threshold': threshold,
            'condition': condition,
            'severity': severity,
            'description': description or f"Alert for {metric_name}",
            'enabled': True
        }
        
        with self.lock:
            self.alert_rules[metric_name] = rule
        
        self.logger.info(f"Added alert rule for {metric_name}: {condition} {threshold}")
    
    def check_metric(self, metric: Metric):
        """Check metric against alert rules."""
        if metric.name not in self.alert_rules:
            return
        
        rule = self.alert_rules[metric.name]
        if not rule['enabled']:
            return
        
        threshold = rule['threshold']
        condition = rule['condition']
        triggered = False
        
        if condition == "greater_than" and metric.value > threshold:
            triggered = True
        elif condition == "less_than" and metric.value < threshold:
            triggered = True
        elif condition == "equals" and abs(metric.value - threshold) < 1e-6:
            triggered = True
        elif condition == "not_equals" and abs(metric.value - threshold) > 1e-6:
            triggered = True
        
        if triggered:
            self._create_alert(metric, rule, threshold)
    
    def _create_alert(self, metric: Metric, rule: Dict[str, Any], threshold: float):
        """Create new alert."""
        alert_id = f"alert_{metric.name}_{int(time.time())}"
        
        alert = Alert(
            alert_id=alert_id,
            severity=rule['severity'],
            title=f"Metric Alert: {metric.name}",
            description=f"{rule['description']}. Current value: {metric.value}, Threshold: {threshold}",
            timestamp=datetime.now(),
            metric_name=metric.name,
            metric_value=metric.value,
            threshold=threshold
        )
        
        with self.lock:
            self.alerts.append(alert)
            
            # Keep only recent alerts
            if len(self.alerts) > 1000:
                self.alerts = self.alerts[-500:]
        
        # Send notifications
        for handler in self.notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Error in notification handler: {e}")
        
        self.logger.warning(f"Alert created: {alert.title}")
    
    def add_notification_handler(self, handler: Callable[[Alert], None]):
        """Add notification handler for alerts."""
        self.notification_handlers.append(handler)
        self.logger.info("Added notification handler")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get list of active (unresolved) alerts."""
        with self.lock:
            return [alert for alert in self.alerts if not alert.resolved]
    
    def resolve_alert(self, alert_id: str):
        """Mark alert as resolved."""
        with self.lock:
            for alert in self.alerts:
                if alert.alert_id == alert_id and not alert.resolved:
                    alert.resolved = True
                    alert.resolved_at = datetime.now()
                    self.logger.info(f"Alert resolved: {alert_id}")
                    return True
        return False
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alert status."""
        with self.lock:
            active_alerts = [a for a in self.alerts if not a.resolved]
            
            summary = {
                'total_alerts': len(self.alerts),
                'active_alerts': len(active_alerts),
                'resolved_alerts': len(self.alerts) - len(active_alerts),
                'alerts_by_severity': {},
                'alert_rules': len(self.alert_rules)
            }
            
            # Count by severity
            for severity in AlertSeverity:
                count = len([a for a in active_alerts if a.severity == severity])
                summary['alerts_by_severity'][severity.value] = count
        
        return summary


class PerformanceTracer:
    """
    Performance tracing for quantum algorithm execution.
    
    Provides detailed tracing of quantum algorithm performance with
    quantum-specific metrics like fidelity and circuit depth.
    """
    
    def __init__(self):
        """Initialize performance tracer."""
        self.active_traces: Dict[str, PerformanceTrace] = {}
        self.completed_traces: List[PerformanceTrace] = []
        self.lock = threading.Lock()
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info("Initialized performance tracer")
    
    def start_trace(self, operation_name: str, trace_id: str = None) -> str:
        """Start a new performance trace."""
        if trace_id is None:
            trace_id = f"trace_{operation_name}_{int(time.time() * 1000000)}"
        
        trace = PerformanceTrace(
            trace_id=trace_id,
            operation_name=operation_name,
            start_time=datetime.now()
        )
        
        with self.lock:
            self.active_traces[trace_id] = trace
        
        return trace_id
    
    def finish_trace(self, trace_id: str, 
                    quantum_algorithm: str = None,
                    quantum_advantage: float = None,
                    circuit_depth: int = None,
                    fidelity: float = None,
                    error_count: int = 0,
                    metadata: Dict[str, Any] = None):
        """Finish a performance trace."""
        with self.lock:
            if trace_id not in self.active_traces:
                self.logger.warning(f"Trace not found: {trace_id}")
                return
            
            trace = self.active_traces.pop(trace_id)
            
        # Update trace with completion data
        trace.finish()
        trace.quantum_algorithm = quantum_algorithm
        trace.quantum_advantage = quantum_advantage
        trace.circuit_depth = circuit_depth
        trace.fidelity = fidelity
        trace.error_count = error_count
        trace.metadata = metadata or {}
        
        with self.lock:
            self.completed_traces.append(trace)
            
            # Keep only recent traces
            if len(self.completed_traces) > 1000:
                self.completed_traces = self.completed_traces[-500:]
        
        self.logger.debug(f"Trace completed: {trace_id} ({trace.duration_ms:.1f}ms)")
    
    def get_trace_statistics(self) -> Dict[str, Any]:
        """Get statistics from completed traces."""
        with self.lock:
            if not self.completed_traces:
                return {'message': 'No completed traces'}
            
            # Calculate statistics
            durations = [t.duration_ms for t in self.completed_traces if t.duration_ms]
            quantum_advantages = [t.quantum_advantage for t in self.completed_traces if t.quantum_advantage]
            fidelities = [t.fidelity for t in self.completed_traces if t.fidelity]
            
            stats = {
                'total_traces': len(self.completed_traces),
                'active_traces': len(self.active_traces),
                'duration_stats': {
                    'count': len(durations),
                    'mean_ms': np.mean(durations) if durations else 0,
                    'std_ms': np.std(durations) if durations else 0,
                    'min_ms': np.min(durations) if durations else 0,
                    'max_ms': np.max(durations) if durations else 0,
                    'p95_ms': np.percentile(durations, 95) if durations else 0
                }
            }
            
            if quantum_advantages:
                stats['quantum_advantage_stats'] = {
                    'count': len(quantum_advantages),
                    'mean': np.mean(quantum_advantages),
                    'std': np.std(quantum_advantages),
                    'min': np.min(quantum_advantages),
                    'max': np.max(quantum_advantages)
                }
            
            if fidelities:
                stats['fidelity_stats'] = {
                    'count': len(fidelities),
                    'mean': np.mean(fidelities),
                    'std': np.std(fidelities),
                    'min': np.min(fidelities),
                    'max': np.max(fidelities)
                }
            
            # Operation breakdown
            operation_counts = {}
            for trace in self.completed_traces:
                op = trace.operation_name
                operation_counts[op] = operation_counts.get(op, 0) + 1
            
            stats['operations'] = operation_counts
        
        return stats


class QuantumMonitor:
    """
    Comprehensive monitoring system for quantum financial algorithms.
    
    Integrates metrics collection, alerting, and performance tracing
    into a unified monitoring solution.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize quantum monitor."""
        self.config = config or {}
        
        # Initialize components
        self.metrics_collector = MetricsCollector(
            buffer_size=self.config.get('metrics_buffer_size', 10000)
        )
        self.alert_manager = AlertManager()
        self.performance_tracer = PerformanceTracer()
        
        # Setup default alert rules
        self._setup_default_alerts()
        
        # Setup default notification handlers
        self._setup_default_notifications()
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info("Initialized quantum monitor")
    
    def _setup_default_alerts(self):
        """Setup default alert rules."""
        # Quantum advantage alerts
        self.alert_manager.add_alert_rule(
            "quantum_advantage",
            threshold=1.0,
            condition="less_than",
            severity=AlertSeverity.WARNING,
            description="Quantum advantage below 1.0 - classical may be better"
        )
        
        # Fidelity alerts
        self.alert_manager.add_alert_rule(
            "quantum_fidelity",
            threshold=0.8,
            condition="less_than",
            severity=AlertSeverity.ERROR,
            description="Quantum fidelity below 80% - high noise/errors"
        )
        
        # Execution time alerts
        self.alert_manager.add_alert_rule(
            "execution_time",
            threshold=5000,  # 5 seconds
            condition="greater_than",
            severity=AlertSeverity.WARNING,
            description="Execution time exceeded 5 seconds"
        )
        
        # Error rate alerts
        self.alert_manager.add_alert_rule(
            "error_rate",
            threshold=0.05,  # 5%
            condition="greater_than",
            severity=AlertSeverity.ERROR,
            description="Error rate exceeded 5%"
        )
    
    def _setup_default_notifications(self):
        """Setup default notification handlers."""
        # Console logging handler
        def console_handler(alert: Alert):
            level = logging.ERROR if alert.severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL] else logging.WARNING
            self.logger.log(level, f"ALERT: {alert.title} - {alert.description}")
        
        self.alert_manager.add_notification_handler(console_handler)
        
        # File logging handler (if configured)
        if self.config.get('alert_log_file'):
            def file_handler(alert: Alert):
                try:
                    with open(self.config['alert_log_file'], 'a') as f:
                        f.write(f"{datetime.now().isoformat()}: {alert.to_dict()}\\n")
                except Exception as e:
                    self.logger.error(f"Error writing alert to file: {e}")
            
            self.alert_manager.add_notification_handler(file_handler)
    
    def monitor_algorithm_execution(self, 
                                  algorithm_name: str,
                                  execution_time_ms: float,
                                  quantum_advantage: float = None,
                                  fidelity: float = None,
                                  circuit_depth: int = None,
                                  success: bool = True,
                                  error_details: str = None):
        """Monitor quantum algorithm execution."""
        
        # Record metrics
        self.metrics_collector.record_timer(
            f"{algorithm_name}_execution_time",
            execution_time_ms,
            labels={'algorithm': algorithm_name, 'success': str(success)}
        )
        
        if quantum_advantage is not None:
            self.metrics_collector.record_quantum_advantage(
                f"{algorithm_name}_quantum_advantage",
                quantum_advantage,
                labels={'algorithm': algorithm_name}
            )
        
        if fidelity is not None:
            self.metrics_collector.record_quantum_fidelity(
                f"{algorithm_name}_fidelity",
                fidelity,
                labels={'algorithm': algorithm_name}
            )
        
        if circuit_depth is not None:
            self.metrics_collector.set_gauge(
                f"{algorithm_name}_circuit_depth",
                circuit_depth,
                labels={'algorithm': algorithm_name}
            )
        
        # Update success/error counters
        if success:
            self.metrics_collector.increment_counter(
                f"{algorithm_name}_success_count",
                labels={'algorithm': algorithm_name}
            )
        else:
            self.metrics_collector.increment_counter(
                f"{algorithm_name}_error_count",
                labels={'algorithm': algorithm_name, 'error': error_details or 'unknown'}
            )
        
        # Calculate and monitor error rate
        success_count = self.metrics_collector.counters.get(f"{algorithm_name}_success_count", 0)
        error_count = self.metrics_collector.counters.get(f"{algorithm_name}_error_count", 0)
        total_count = success_count + error_count
        
        if total_count > 0:
            error_rate = error_count / total_count
            self.metrics_collector.set_gauge(
                f"{algorithm_name}_error_rate",
                error_rate,
                labels={'algorithm': algorithm_name}
            )
    
    def start_operation_trace(self, operation_name: str) -> str:
        """Start tracing an operation."""
        return self.performance_tracer.start_trace(operation_name)
    
    def finish_operation_trace(self, trace_id: str, **kwargs):
        """Finish tracing an operation."""
        self.performance_tracer.finish_trace(trace_id, **kwargs)
    
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data."""
        dashboard = {
            'timestamp': datetime.now().isoformat(),
            'metrics_summary': self.metrics_collector.get_metric_summary(),
            'alert_summary': self.alert_manager.get_alert_summary(),
            'trace_statistics': self.performance_tracer.get_trace_statistics(),
            'active_alerts': [alert.to_dict() for alert in self.alert_manager.get_active_alerts()]
        }
        
        return dashboard
    
    def export_metrics(self, format_type: str = "json") -> str:
        """Export metrics in specified format."""
        return self.metrics_collector.export_metrics(format_type)
    
    def health_check(self) -> Dict[str, Any]:
        """Perform system health check."""
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'components': {
                'metrics_collector': 'healthy',
                'alert_manager': 'healthy',
                'performance_tracer': 'healthy'
            },
            'statistics': {}
        }
        
        # Check component health
        try:
            metrics_summary = self.metrics_collector.get_metric_summary()
            health_status['statistics']['metrics_count'] = metrics_summary['buffer_size']
        except Exception as e:
            health_status['components']['metrics_collector'] = f'error: {e}'
            health_status['status'] = 'degraded'
        
        try:
            alert_summary = self.alert_manager.get_alert_summary()
            health_status['statistics']['active_alerts'] = alert_summary['active_alerts']
            
            # Mark as unhealthy if critical alerts
            critical_alerts = alert_summary['alerts_by_severity'].get('critical', 0)
            if critical_alerts > 0:
                health_status['status'] = 'unhealthy'
                
        except Exception as e:
            health_status['components']['alert_manager'] = f'error: {e}'
            health_status['status'] = 'degraded'
        
        try:
            trace_stats = self.performance_tracer.get_trace_statistics()
            health_status['statistics']['completed_traces'] = trace_stats.get('total_traces', 0)
        except Exception as e:
            health_status['components']['performance_tracer'] = f'error: {e}'
            health_status['status'] = 'degraded'
        
        return health_status


# Monitoring decorator for easy integration
def monitor_quantum_algorithm(monitor: QuantumMonitor, algorithm_name: str):
    """Decorator to automatically monitor quantum algorithm execution."""
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Start tracing
            trace_id = monitor.start_operation_trace(f"{algorithm_name}_execution")
            
            start_time = time.time()
            success = True
            error_details = None
            result = None
            
            try:
                result = func(*args, **kwargs)
                
                # Extract quantum metrics from result if available
                quantum_advantage = getattr(result, 'quantum_advantage_score', None) or getattr(result, 'quantum_advantage', None)
                fidelity = getattr(result, 'fidelity', None) or getattr(result, 'photonic_fidelity', None)
                circuit_depth = getattr(result, 'circuit_depth', None)
                
                # Finish trace with quantum metrics
                monitor.finish_operation_trace(
                    trace_id,
                    quantum_algorithm=algorithm_name,
                    quantum_advantage=quantum_advantage,
                    circuit_depth=circuit_depth,
                    fidelity=fidelity
                )
                
            except Exception as e:
                success = False
                error_details = str(e)
                monitor.finish_operation_trace(
                    trace_id,
                    quantum_algorithm=algorithm_name,
                    error_count=1
                )
                raise
            
            finally:
                # Record execution metrics
                execution_time_ms = (time.time() - start_time) * 1000
                
                quantum_advantage = None
                fidelity = None
                circuit_depth = None
                
                if result:
                    quantum_advantage = getattr(result, 'quantum_advantage_score', None) or getattr(result, 'quantum_advantage', None)
                    fidelity = getattr(result, 'fidelity', None) or getattr(result, 'photonic_fidelity', None)
                    circuit_depth = getattr(result, 'circuit_depth', None)
                
                monitor.monitor_algorithm_execution(
                    algorithm_name=algorithm_name,
                    execution_time_ms=execution_time_ms,
                    quantum_advantage=quantum_advantage,
                    fidelity=fidelity,
                    circuit_depth=circuit_depth,
                    success=success,
                    error_details=error_details
                )
            
            return result
        
        return wrapper
    return decorator


# Alias for backward compatibility
QuantumMonitoringService = QuantumMonitor

# Export main classes and functions
__all__ = [
    'MetricType',
    'AlertSeverity',
    'Metric',
    'Alert',
    'PerformanceTrace',
    'MetricsCollector',
    'AlertManager',
    'PerformanceTracer',
    'QuantumMonitor',
    'QuantumMonitoringService',
    'monitor_quantum_algorithm'
]