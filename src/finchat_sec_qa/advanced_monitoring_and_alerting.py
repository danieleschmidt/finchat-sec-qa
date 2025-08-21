"""
Advanced Monitoring and Alerting System v4.0
Comprehensive monitoring, metrics collection, and intelligent alerting.
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import logging
import threading
import statistics
import functools

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertCondition(Enum):
    """Alert condition types"""
    THRESHOLD = "threshold"
    RATE_OF_CHANGE = "rate_of_change"
    ANOMALY = "anomaly"
    COMPOSITE = "composite"


@dataclass
class MetricValue:
    """Individual metric measurement"""
    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class AlertRule:
    """Alert rule configuration"""
    id: str
    name: str
    metric_name: str
    condition: AlertCondition
    threshold: float
    severity: AlertSeverity
    duration: int = 60  # seconds
    evaluation_interval: int = 30  # seconds
    labels: Dict[str, str] = field(default_factory=dict)
    message_template: str = "Alert triggered for {metric_name}"
    active: bool = True
    last_evaluation: float = 0
    last_triggered: float = 0
    consecutive_triggers: int = 0


@dataclass
class Alert:
    """Active alert"""
    id: str
    rule_id: str
    metric_name: str
    severity: AlertSeverity
    message: str
    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    resolved: bool = False
    resolved_timestamp: Optional[float] = None


class Metric:
    """Metric collection and storage"""
    
    def __init__(self, name: str, metric_type: MetricType, description: str = ""):
        self.name = name
        self.type = metric_type
        self.description = description
        self.values: deque = deque(maxlen=10000)  # Keep last 10k values
        self.labels: Dict[str, str] = {}
        self.lock = threading.Lock()
    
    def add_value(self, value: float, labels: Dict[str, str] = None):
        """Add a metric value"""
        with self.lock:
            metric_value = MetricValue(
                timestamp=time.time(),
                value=value,
                labels=labels or {}
            )
            self.values.append(metric_value)
    
    def get_latest_value(self) -> Optional[MetricValue]:
        """Get the latest metric value"""
        with self.lock:
            return self.values[-1] if self.values else None
    
    def get_values_in_range(self, start_time: float, end_time: float) -> List[MetricValue]:
        """Get values within time range"""
        with self.lock:
            return [
                v for v in self.values 
                if start_time <= v.timestamp <= end_time
            ]
    
    def get_statistics(self, duration: int = 3600) -> Dict[str, float]:
        """Get statistical summary for the last duration seconds"""
        end_time = time.time()
        start_time = end_time - duration
        values = self.get_values_in_range(start_time, end_time)
        
        if not values:
            return {}
        
        numeric_values = [v.value for v in values]
        
        return {
            "count": len(numeric_values),
            "min": min(numeric_values),
            "max": max(numeric_values),
            "mean": statistics.mean(numeric_values),
            "median": statistics.median(numeric_values),
            "stddev": statistics.stdev(numeric_values) if len(numeric_values) > 1 else 0,
            "rate": len(numeric_values) / duration  # values per second
        }


class AlertEngine:
    """Alert evaluation and notification engine"""
    
    def __init__(self):
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=10000)
        self.notification_handlers: List[Callable] = []
        self.evaluation_running = False
    
    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule"""
        self.alert_rules[rule.id] = rule
        logger.info(f"🚨 Alert rule added: {rule.name}")
    
    def remove_alert_rule(self, rule_id: str):
        """Remove an alert rule"""
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            logger.info(f"🚨 Alert rule removed: {rule_id}")
    
    def add_notification_handler(self, handler: Callable):
        """Add notification handler"""
        self.notification_handlers.append(handler)
    
    async def evaluate_rules(self, metrics: Dict[str, Metric]):
        """Evaluate all alert rules against current metrics"""
        current_time = time.time()
        
        for rule in self.alert_rules.values():
            if not rule.active:
                continue
            
            # Check if it's time to evaluate this rule
            if current_time - rule.last_evaluation < rule.evaluation_interval:
                continue
            
            rule.last_evaluation = current_time
            
            metric = metrics.get(rule.metric_name)
            if not metric:
                continue
            
            try:
                should_trigger = await self._evaluate_rule(rule, metric)
                
                if should_trigger:
                    await self._trigger_alert(rule, metric)
                else:
                    await self._resolve_alert(rule.id)
                    
            except Exception as e:
                logger.error(f"❌ Error evaluating alert rule {rule.name}: {str(e)}")
    
    async def _evaluate_rule(self, rule: AlertRule, metric: Metric) -> bool:
        """Evaluate individual alert rule"""
        latest_value = metric.get_latest_value()
        if not latest_value:
            return False
        
        if rule.condition == AlertCondition.THRESHOLD:
            return self._evaluate_threshold(rule, latest_value.value)
        elif rule.condition == AlertCondition.RATE_OF_CHANGE:
            return self._evaluate_rate_of_change(rule, metric)
        elif rule.condition == AlertCondition.ANOMALY:
            return self._evaluate_anomaly(rule, metric)
        else:
            return False
    
    def _evaluate_threshold(self, rule: AlertRule, value: float) -> bool:
        """Evaluate threshold condition"""
        if rule.severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL]:
            return value >= rule.threshold  # Alert if value exceeds threshold
        else:
            return value <= rule.threshold  # Alert if value drops below threshold
    
    def _evaluate_rate_of_change(self, rule: AlertRule, metric: Metric) -> bool:
        """Evaluate rate of change condition"""
        current_time = time.time()
        start_time = current_time - rule.duration
        
        values = metric.get_values_in_range(start_time, current_time)
        if len(values) < 2:
            return False
        
        # Calculate rate of change
        time_diff = values[-1].timestamp - values[0].timestamp
        value_diff = values[-1].value - values[0].value
        
        if time_diff == 0:
            return False
        
        rate = abs(value_diff / time_diff)
        return rate >= rule.threshold
    
    def _evaluate_anomaly(self, rule: AlertRule, metric: Metric) -> bool:
        """Evaluate anomaly detection"""
        stats = metric.get_statistics(duration=3600)  # Last hour
        if not stats or "mean" not in stats:
            return False
        
        latest_value = metric.get_latest_value()
        if not latest_value:
            return False
        
        # Simple anomaly detection: value is more than N standard deviations from mean
        threshold_stdevs = rule.threshold
        deviation = abs(latest_value.value - stats["mean"])
        
        if stats["stddev"] == 0:
            return False
        
        return deviation > (threshold_stdevs * stats["stddev"])
    
    async def _trigger_alert(self, rule: AlertRule, metric: Metric):
        """Trigger an alert"""
        latest_value = metric.get_latest_value()
        if not latest_value:
            return
        
        alert_id = f"{rule.id}_{int(time.time())}"
        
        # Check if this alert is already active (avoid spam)
        existing_alert = None
        for alert in self.active_alerts.values():
            if alert.rule_id == rule.id and not alert.resolved:
                existing_alert = alert
                break
        
        if existing_alert:
            # Update existing alert
            existing_alert.timestamp = time.time()
            existing_alert.value = latest_value.value
            rule.consecutive_triggers += 1
        else:
            # Create new alert
            message = rule.message_template.format(
                metric_name=rule.metric_name,
                value=latest_value.value,
                threshold=rule.threshold
            )
            
            alert = Alert(
                id=alert_id,
                rule_id=rule.id,
                metric_name=rule.metric_name,
                severity=rule.severity,
                message=message,
                timestamp=time.time(),
                value=latest_value.value,
                labels=latest_value.labels
            )
            
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            rule.last_triggered = time.time()
            rule.consecutive_triggers = 1
            
            # Send notifications
            await self._send_notifications(alert)
            
            logger.warning(f"🚨 Alert triggered: {alert.message}")
    
    async def _resolve_alert(self, rule_id: str):
        """Resolve active alerts for a rule"""
        for alert in list(self.active_alerts.values()):
            if alert.rule_id == rule_id and not alert.resolved:
                alert.resolved = True
                alert.resolved_timestamp = time.time()
                del self.active_alerts[alert.id]
                
                # Reset consecutive triggers
                if rule_id in self.alert_rules:
                    self.alert_rules[rule_id].consecutive_triggers = 0
                
                logger.info(f"✅ Alert resolved: {alert.message}")
    
    async def _send_notifications(self, alert: Alert):
        """Send alert notifications"""
        for handler in self.notification_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                logger.error(f"❌ Notification handler failed: {str(e)}")
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary"""
        active_by_severity = defaultdict(int)
        for alert in self.active_alerts.values():
            active_by_severity[alert.severity.value] += 1
        
        recent_alerts = [
            a for a in self.alert_history 
            if time.time() - a.timestamp < 3600  # Last hour
        ]
        
        return {
            "active_alerts": len(self.active_alerts),
            "active_by_severity": dict(active_by_severity),
            "recent_alerts": len(recent_alerts),
            "total_rules": len(self.alert_rules),
            "active_rules": sum(1 for r in self.alert_rules.values() if r.active)
        }


class MetricsCollector:
    """Centralized metrics collection"""
    
    def __init__(self):
        self.metrics: Dict[str, Metric] = {}
        self.collection_enabled = True
        self.auto_collection_tasks: List[asyncio.Task] = []
    
    def create_metric(self, name: str, metric_type: MetricType, description: str = "") -> Metric:
        """Create a new metric"""
        metric = Metric(name, metric_type, description)
        self.metrics[name] = metric
        logger.info(f"📊 Metric created: {name} ({metric_type.value})")
        return metric
    
    def get_metric(self, name: str) -> Optional[Metric]:
        """Get metric by name"""
        return self.metrics.get(name)
    
    def record_counter(self, name: str, value: float = 1, labels: Dict[str, str] = None):
        """Record counter metric"""
        metric = self.metrics.get(name)
        if not metric:
            metric = self.create_metric(name, MetricType.COUNTER)
        
        if metric.type != MetricType.COUNTER:
            logger.error(f"❌ Metric {name} is not a counter")
            return
        
        # For counters, add to the latest value
        latest = metric.get_latest_value()
        current_value = (latest.value if latest else 0) + value
        metric.add_value(current_value, labels)
    
    def record_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record gauge metric"""
        metric = self.metrics.get(name)
        if not metric:
            metric = self.create_metric(name, MetricType.GAUGE)
        
        if metric.type != MetricType.GAUGE:
            logger.error(f"❌ Metric {name} is not a gauge")
            return
        
        metric.add_value(value, labels)
    
    def record_timer(self, name: str, duration: float, labels: Dict[str, str] = None):
        """Record timer metric"""
        metric = self.metrics.get(name)
        if not metric:
            metric = self.create_metric(name, MetricType.TIMER)
        
        if metric.type != MetricType.TIMER:
            logger.error(f"❌ Metric {name} is not a timer")
            return
        
        metric.add_value(duration, labels)
    
    def start_auto_collection(self, interval: int = 30):
        """Start automatic system metrics collection"""
        async def collect_system_metrics():
            while self.collection_enabled:
                try:
                    # Collect basic system metrics
                    import psutil
                    
                    # CPU metrics
                    cpu_percent = psutil.cpu_percent(interval=1)
                    self.record_gauge("system_cpu_percent", cpu_percent)
                    
                    # Memory metrics
                    memory = psutil.virtual_memory()
                    self.record_gauge("system_memory_percent", memory.percent)
                    self.record_gauge("system_memory_available_mb", memory.available / (1024 * 1024))
                    
                    # Disk metrics
                    disk = psutil.disk_usage('/')
                    disk_percent = (disk.used / disk.total) * 100
                    self.record_gauge("system_disk_percent", disk_percent)
                    
                    # Process metrics
                    process = psutil.Process()
                    process_memory = process.memory_info()
                    self.record_gauge("process_memory_rss_mb", process_memory.rss / (1024 * 1024))
                    self.record_gauge("process_cpu_percent", process.cpu_percent())
                    
                except ImportError:
                    logger.warning("psutil not available, skipping system metrics")
                    break
                except Exception as e:
                    logger.error(f"❌ Error collecting system metrics: {str(e)}")
                
                await asyncio.sleep(interval)
        
        task = asyncio.create_task(collect_system_metrics())
        self.auto_collection_tasks.append(task)
        logger.info("📊 Auto metrics collection started")
    
    def stop_auto_collection(self):
        """Stop automatic metrics collection"""
        self.collection_enabled = False
        for task in self.auto_collection_tasks:
            if not task.done():
                task.cancel()
        self.auto_collection_tasks.clear()
        logger.info("📊 Auto metrics collection stopped")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        summary = {
            "total_metrics": len(self.metrics),
            "metrics_by_type": defaultdict(int),
            "metrics": {}
        }
        
        for name, metric in self.metrics.items():
            summary["metrics_by_type"][metric.type.value] += 1
            
            latest_value = metric.get_latest_value()
            stats = metric.get_statistics(3600)  # Last hour
            
            summary["metrics"][name] = {
                "type": metric.type.value,
                "description": metric.description,
                "latest_value": latest_value.value if latest_value else None,
                "latest_timestamp": latest_value.timestamp if latest_value else None,
                "stats": stats
            }
        
        summary["metrics_by_type"] = dict(summary["metrics_by_type"])
        return summary


class AdvancedMonitoringSystem:
    """
    Advanced Monitoring and Alerting System
    Combines metrics collection, alerting, and monitoring
    """
    
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.metrics_collector = MetricsCollector()
        self.alert_engine = AlertEngine()
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Setup default metrics and alerts
        self._setup_default_metrics()
        self._setup_default_alerts()
        self._setup_default_notifications()
    
    def _setup_default_metrics(self):
        """Setup default metrics"""
        # Performance metrics
        self.metrics_collector.create_metric("response_time_ms", MetricType.TIMER, "API response time in milliseconds")
        self.metrics_collector.create_metric("requests_total", MetricType.COUNTER, "Total number of requests")
        self.metrics_collector.create_metric("errors_total", MetricType.COUNTER, "Total number of errors")
        
        # System metrics
        self.metrics_collector.create_metric("cpu_usage_percent", MetricType.GAUGE, "CPU usage percentage")
        self.metrics_collector.create_metric("memory_usage_percent", MetricType.GAUGE, "Memory usage percentage")
        
        # Application metrics
        self.metrics_collector.create_metric("active_connections", MetricType.GAUGE, "Number of active connections")
        self.metrics_collector.create_metric("cache_hit_rate", MetricType.GAUGE, "Cache hit rate percentage")
    
    def _setup_default_alerts(self):
        """Setup default alert rules"""
        # High response time alert
        self.alert_engine.add_alert_rule(AlertRule(
            id="high_response_time",
            name="High Response Time",
            metric_name="response_time_ms",
            condition=AlertCondition.THRESHOLD,
            threshold=1000.0,  # 1 second
            severity=AlertSeverity.WARNING,
            duration=60,
            message_template="Response time is high: {value:.2f}ms (threshold: {threshold}ms)"
        ))
        
        # High error rate alert
        self.alert_engine.add_alert_rule(AlertRule(
            id="high_error_rate",
            name="High Error Rate",
            metric_name="errors_total",
            condition=AlertCondition.RATE_OF_CHANGE,
            threshold=10.0,  # 10 errors per second
            severity=AlertSeverity.ERROR,
            duration=300,
            message_template="High error rate detected: {value:.2f} errors/sec"
        ))
        
        # High CPU usage alert
        self.alert_engine.add_alert_rule(AlertRule(
            id="high_cpu_usage",
            name="High CPU Usage",
            metric_name="cpu_usage_percent",
            condition=AlertCondition.THRESHOLD,
            threshold=80.0,
            severity=AlertSeverity.WARNING,
            duration=120,
            message_template="CPU usage is high: {value:.1f}% (threshold: {threshold}%)"
        ))
        
        # High memory usage alert
        self.alert_engine.add_alert_rule(AlertRule(
            id="high_memory_usage",
            name="High Memory Usage",
            metric_name="memory_usage_percent",
            condition=AlertCondition.THRESHOLD,
            threshold=85.0,
            severity=AlertSeverity.CRITICAL,
            duration=60,
            message_template="Memory usage is critical: {value:.1f}% (threshold: {threshold}%)"
        ))
    
    def _setup_default_notifications(self):
        """Setup default notification handlers"""
        
        async def log_notification(alert: Alert):
            """Log alert notifications"""
            severity_emoji = {
                AlertSeverity.INFO: "ℹ️",
                AlertSeverity.WARNING: "⚠️",
                AlertSeverity.ERROR: "❌",
                AlertSeverity.CRITICAL: "🚨"
            }
            
            emoji = severity_emoji.get(alert.severity, "📢")
            logger.info(f"{emoji} ALERT: {alert.message}")
        
        def console_notification(alert: Alert):
            """Console alert notifications"""
            print(f"\n🚨 ALERT [{alert.severity.value.upper()}] - {alert.message}")
            print(f"   Metric: {alert.metric_name}")
            print(f"   Value: {alert.value}")
            print(f"   Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert.timestamp))}")
            print()
        
        self.alert_engine.add_notification_handler(log_notification)
        self.alert_engine.add_notification_handler(console_notification)
    
    async def start(self):
        """Start the monitoring system"""
        logger.info(f"📊 Starting Advanced Monitoring System for {self.project_name}")
        
        self.monitoring_active = True
        
        # Start auto metrics collection
        self.metrics_collector.start_auto_collection()
        
        # Start alert evaluation loop
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("📊 Advanced monitoring system started")
    
    async def stop(self):
        """Stop the monitoring system"""
        logger.info("📊 Stopping Advanced Monitoring System")
        
        self.monitoring_active = False
        
        # Stop auto metrics collection
        self.metrics_collector.stop_auto_collection()
        
        # Stop monitoring loop
        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("📊 Advanced monitoring system stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Evaluate alert rules
                await self.alert_engine.evaluate_rules(self.metrics_collector.metrics)
                
                # Wait before next evaluation
                await asyncio.sleep(10)  # Evaluate every 10 seconds
                
            except Exception as e:
                logger.error(f"❌ Error in monitoring loop: {str(e)}")
                await asyncio.sleep(5)  # Back off on error
    
    def record_metric(self, name: str, value: float, metric_type: str = "gauge", labels: Dict[str, str] = None):
        """Record a metric value"""
        if metric_type == "counter":
            self.metrics_collector.record_counter(name, value, labels)
        elif metric_type == "gauge":
            self.metrics_collector.record_gauge(name, value, labels)
        elif metric_type == "timer":
            self.metrics_collector.record_timer(name, value, labels)
        else:
            logger.error(f"❌ Unknown metric type: {metric_type}")
    
    def add_custom_alert(self, rule: AlertRule):
        """Add custom alert rule"""
        self.alert_engine.add_alert_rule(rule)
    
    def add_notification_handler(self, handler: Callable):
        """Add custom notification handler"""
        self.alert_engine.add_notification_handler(handler)
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get complete monitoring status"""
        return {
            "project_name": self.project_name,
            "monitoring_active": self.monitoring_active,
            "metrics_summary": self.metrics_collector.get_metrics_summary(),
            "alert_summary": self.alert_engine.get_alert_summary(),
            "active_alerts": [asdict(alert) for alert in self.alert_engine.active_alerts.values()]
        }


# Decorator for automatic performance monitoring
def monitor_performance(metric_name: str = None):
    """Decorator for automatic performance monitoring"""
    def decorator(func):
        actual_metric_name = metric_name or f"{func.__name__}_duration_ms"
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Get or create monitoring system
            monitoring = getattr(wrapper, '_monitoring', None)
            if monitoring is None:
                monitoring = AdvancedMonitoringSystem("default")
                wrapper._monitoring = monitoring
            
            start_time = time.time()
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Record success
                monitoring.record_metric("requests_total", 1, "counter", {"function": func.__name__, "status": "success"})
                return result
                
            except Exception as e:
                # Record error
                monitoring.record_metric("errors_total", 1, "counter", {"function": func.__name__, "error_type": type(e).__name__})
                raise
            finally:
                # Record duration
                duration_ms = (time.time() - start_time) * 1000
                monitoring.record_metric(actual_metric_name, duration_ms, "timer", {"function": func.__name__})
        
        return wrapper
    return decorator


# Factory function
def create_monitoring_system(project_name: str) -> AdvancedMonitoringSystem:
    """Create advanced monitoring system"""
    return AdvancedMonitoringSystem(project_name)


# Example usage
async def demonstrate_monitoring():
    """Demonstrate monitoring system"""
    monitoring = create_monitoring_system("FinChat-SEC-QA")
    
    try:
        await monitoring.start()
        
        # Simulate some metrics
        for i in range(10):
            # Simulate API requests
            response_time = 50 + (i * 10)  # Increasing response time
            monitoring.record_metric("response_time_ms", response_time, "timer")
            monitoring.record_metric("requests_total", 1, "counter")
            
            # Simulate occasional errors
            if i % 3 == 0:
                monitoring.record_metric("errors_total", 1, "counter")
            
            # Simulate increasing CPU usage
            cpu_usage = 20 + (i * 8)  # Will trigger alert at 80%
            monitoring.record_metric("cpu_usage_percent", cpu_usage, "gauge")
            
            await asyncio.sleep(2)
        
        # Let alerts trigger and show status
        await asyncio.sleep(5)
        
        status = monitoring.get_monitoring_status()
        logger.info(f"📊 Monitoring status: {status}")
        
    finally:
        await monitoring.stop()


if __name__ == "__main__":
    # Example usage
    async def main():
        await demonstrate_monitoring()
    
    asyncio.run(main())