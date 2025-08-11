"""
Comprehensive tests for the Monitoring and Observability System.

This test suite ensures the monitoring system correctly tracks metrics,
generates alerts, and provides system observability.
"""

import asyncio
import pytest
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from finchat_sec_qa.comprehensive_monitoring import (
    ComprehensiveMonitoring,
    Alert,
    AlertLevel,
    MetricPoint,
    MetricType,
    HealthCheck,
    HealthStatus,
    get_monitoring
)


@pytest.fixture
def temp_storage_path(tmp_path):
    """Create temporary storage path for testing."""
    return tmp_path / "test_monitoring"


@pytest.fixture
def monitoring(temp_storage_path):
    """Create a monitoring instance for testing."""
    monitor = ComprehensiveMonitoring(storage_path=temp_storage_path)
    yield monitor
    monitor.stop_monitoring()


@pytest.fixture
def sample_metric():
    """Sample metric point for testing."""
    return MetricPoint(
        name="test.metric",
        value=42.0,
        timestamp=datetime.now(),
        tags={"component": "test", "env": "development"},
        metric_type=MetricType.GAUGE
    )


@pytest.fixture
def sample_alert():
    """Sample alert for testing."""
    return Alert(
        id="test_alert_001",
        timestamp=datetime.now(),
        level=AlertLevel.WARNING,
        component="test_component",
        message="Test alert message",
        details={"metric_value": 85.0, "threshold": 80.0}
    )


class TestComprehensiveMonitoring:
    """Test cases for the Comprehensive Monitoring system."""

    def test_initialization(self, monitoring):
        """Test monitoring system initialization."""
        assert monitoring.storage_path.exists()
        assert len(monitoring.alerts) == 0
        assert len(monitoring.health_checks) > 0  # Should have default health checks
        assert not monitoring._running

    def test_start_stop_monitoring(self, monitoring):
        """Test starting and stopping the monitoring system."""
        # Start monitoring
        monitoring.start_monitoring()
        assert monitoring._running
        assert monitoring._monitor_thread is not None
        
        # Stop monitoring
        monitoring.stop_monitoring()
        assert not monitoring._running

    def test_metric_recording(self, monitoring):
        """Test recording of metrics."""
        # Record a gauge metric
        monitoring.record_metric("cpu.usage", 65.0, MetricType.GAUGE, {"host": "test"})
        
        assert "cpu.usage" in monitoring.metrics
        assert len(monitoring.metrics["cpu.usage"]) == 1
        
        latest_metric = monitoring.metrics["cpu.usage"][-1]
        assert latest_metric.value == 65.0
        assert latest_metric.tags["host"] == "test"
        assert latest_metric.metric_type == MetricType.GAUGE

    def test_counter_increment(self, monitoring):
        """Test counter metric increment."""
        # Record initial counter value
        monitoring.record_metric("requests.total", 10.0, MetricType.COUNTER)
        
        # Increment counter
        monitoring.increment_counter("requests.total", 5.0)
        
        latest_value = monitoring.get_latest_metric_value("requests.total")
        assert latest_value == 15.0

    def test_timer_recording(self, monitoring):
        """Test timer metric recording."""
        # Record timer metric
        monitoring.record_timer("request.duration", 0.250, {"endpoint": "/api/query"})
        
        assert "request.duration" in monitoring.metrics
        latest_metric = monitoring.metrics["request.duration"][-1]
        assert latest_metric.value == 0.250
        assert latest_metric.metric_type == MetricType.TIMER
        assert latest_metric.tags["endpoint"] == "/api/query"

    def test_timer_context_manager(self, monitoring):
        """Test timer context manager functionality."""
        with monitoring.timer_context("operation.test"):
            time.sleep(0.1)  # Simulate operation
        
        assert "operation.test" in monitoring.metrics
        duration = monitoring.get_latest_metric_value("operation.test")
        assert duration >= 0.1  # Should be at least 100ms

    def test_metric_statistics(self, monitoring):
        """Test metric statistics calculation."""
        # Add multiple data points
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        for value in values:
            monitoring.record_metric("test.metric", value)
        
        stats = monitoring.get_metric_statistics("test.metric")
        
        assert stats['count'] == 5
        assert stats['min'] == 10.0
        assert stats['max'] == 50.0
        assert stats['mean'] == 30.0
        assert stats['median'] == 30.0

    def test_metric_statistics_with_time_window(self, monitoring):
        """Test metric statistics with time window filter."""
        # Add old metric
        old_time = datetime.now() - timedelta(hours=2)
        old_metric = MetricPoint(
            name="test.windowed",
            value=100.0,
            timestamp=old_time,
            metric_type=MetricType.GAUGE
        )
        monitoring.metrics["test.windowed"].append(old_metric)
        
        # Add recent metrics
        monitoring.record_metric("test.windowed", 10.0)
        monitoring.record_metric("test.windowed", 20.0)
        
        # Get statistics for last hour only
        recent_stats = monitoring.get_metric_statistics(
            "test.windowed", 
            time_window=timedelta(hours=1)
        )
        
        assert recent_stats['count'] == 2  # Should exclude old metric
        assert recent_stats['mean'] == 15.0

    def test_health_check_registration(self, monitoring):
        """Test health check registration."""
        def custom_health_check():
            return HealthCheck(
                component="custom_service",
                status=HealthStatus.HEALTHY,
                response_time=0.1,
                details={"version": "1.0.0"}
            )
        
        monitoring.register_health_check("custom_service", custom_health_check)
        
        assert "custom_service" in monitoring.health_checks

    def test_system_health_check(self, monitoring):
        """Test system health check functionality."""
        # Mock psutil for consistent testing
        with patch('finchat_sec_qa.comprehensive_monitoring.psutil') as mock_psutil:
            mock_psutil.cpu_percent.return_value = 45.0
            mock_memory = Mock()
            mock_memory.percent = 60.0
            mock_psutil.virtual_memory.return_value = mock_memory
            
            mock_disk_usage = Mock()
            mock_disk_usage.used = 500 * (1024**3)  # 500GB
            mock_disk_usage.total = 1000 * (1024**3)  # 1TB
            mock_psutil.disk_usage.return_value = mock_disk_usage
            
            health_check = monitoring._system_health_check()
            
            assert health_check.component == "system"
            assert health_check.status == HealthStatus.HEALTHY
            assert health_check.details["cpu_percent"] == 45.0
            assert health_check.details["memory_percent"] == 60.0

    def test_database_health_check(self, monitoring):
        """Test database health check."""
        health_check = monitoring._database_health_check()
        
        assert health_check.component == "database"
        assert health_check.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNHEALTHY]
        assert health_check.response_time >= 0

    def test_external_apis_health_check(self, monitoring):
        """Test external APIs health check."""
        health_check = monitoring._external_apis_health_check()
        
        assert health_check.component == "external_apis"
        assert isinstance(health_check.details["api_statuses"], dict)

    def test_quantum_system_health_check(self, monitoring):
        """Test quantum system health check."""
        health_check = monitoring._quantum_system_health_check()
        
        assert health_check.component == "quantum_system"
        assert "circuit_fidelity" in health_check.details
        assert "active_qubits" in health_check.details

    def test_overall_health_status(self, monitoring):
        """Test overall health status calculation."""
        # Mock health check results
        monitoring.health_status = {
            "system": HealthCheck(
                component="system",
                status=HealthStatus.HEALTHY,
                response_time=0.1
            ),
            "database": HealthCheck(
                component="database",
                status=HealthStatus.DEGRADED,
                response_time=2.0
            ),
            "external_apis": HealthCheck(
                component="external_apis",
                status=HealthStatus.HEALTHY,
                response_time=0.5
            )
        }
        
        health_status = monitoring.get_health_status()
        
        assert health_status["overall_status"] == HealthStatus.DEGRADED.value
        assert "components" in health_status
        assert len(health_status["components"]) == 3

    def test_alert_creation(self, monitoring):
        """Test alert creation and management."""
        alert = monitoring.create_alert(
            AlertLevel.ERROR,
            "test_component",
            "Test error message",
            {"error_code": 500}
        )
        
        assert alert.level == AlertLevel.ERROR
        assert alert.component == "test_component"
        assert alert.message == "Test error message"
        assert not alert.resolved
        assert len(monitoring.alerts) == 1

    def test_alert_resolution(self, monitoring, sample_alert):
        """Test alert resolution."""
        # Add alert to monitoring system
        monitoring.alerts.append(sample_alert)
        
        # Resolve the alert
        success = monitoring.resolve_alert(sample_alert.id)
        
        assert success
        assert sample_alert.resolved
        assert sample_alert.resolution_time is not None

    def test_alert_rule_management(self, monitoring):
        """Test alert rule management."""
        # Add alert rule
        monitoring.add_alert_rule(
            name="high_cpu_test",
            condition="cpu.usage > 90",
            alert_level=AlertLevel.CRITICAL,
            component="system",
            message="CPU usage is critically high"
        )
        
        assert len(monitoring.alert_rules) > 0
        
        # Find the added rule
        cpu_rules = [rule for rule in monitoring.alert_rules if rule['name'] == 'high_cpu_test']
        assert len(cpu_rules) == 1
        assert cpu_rules[0]['condition'] == "cpu.usage > 90"

    def test_condition_evaluation(self, monitoring):
        """Test alert condition evaluation."""
        # Add test metric
        monitoring.record_metric("test.cpu.usage", 95.0)
        
        # Test various conditions
        assert monitoring._evaluate_condition("test.cpu.usage > 90")
        assert not monitoring._evaluate_condition("test.cpu.usage < 90")
        assert monitoring._evaluate_condition("test.cpu.usage >= 95")
        assert monitoring._evaluate_condition("test.cpu.usage <= 95")
        assert monitoring._evaluate_condition("test.cpu.usage == 95")

    def test_alert_condition_checking(self, monitoring):
        """Test automated alert condition checking."""
        # Add metric that should trigger alert
        monitoring.record_metric("system.cpu.percent", 95.0)
        
        # Check alert conditions (should trigger default high CPU rule)
        initial_alert_count = len(monitoring.alerts)
        monitoring._check_alert_conditions()
        
        # Should have created new alert
        assert len(monitoring.alerts) > initial_alert_count

    def test_active_alerts_retrieval(self, monitoring):
        """Test retrieval of active alerts."""
        # Create resolved and unresolved alerts
        resolved_alert = Alert(
            id="resolved_001",
            timestamp=datetime.now(),
            level=AlertLevel.WARNING,
            component="test",
            message="Resolved alert",
            resolved=True
        )
        
        active_alert = Alert(
            id="active_001",
            timestamp=datetime.now(),
            level=AlertLevel.ERROR,
            component="test",
            message="Active alert",
            resolved=False
        )
        
        monitoring.alerts.extend([resolved_alert, active_alert])
        
        active_alerts = monitoring.get_active_alerts()
        
        assert len(active_alerts) == 1
        assert active_alerts[0]["id"] == "active_001"

    def test_system_overview(self, monitoring):
        """Test system overview generation."""
        # Add some test data
        monitoring.record_metric("response_time", 0.5)
        monitoring.record_metric("error_rate", 0.02)
        
        # Mock psutil for system metrics
        with patch('finchat_sec_qa.comprehensive_monitoring.psutil') as mock_psutil:
            mock_psutil.cpu_percent.return_value = 65.0
            mock_memory = Mock()
            mock_memory.percent = 70.0
            mock_psutil.virtual_memory.return_value = mock_memory
            
            mock_disk_usage = Mock()
            mock_disk_usage.used = 400 * (1024**3)
            mock_disk_usage.total = 1000 * (1024**3)
            mock_psutil.disk_usage.return_value = mock_disk_usage
            
            overview = monitoring.get_system_overview()
            
            assert "timestamp" in overview
            assert "system" in overview
            assert "application" in overview
            assert "health" in overview
            assert "alerts" in overview
            assert "performance" in overview
            
            assert overview["system"]["cpu_percent"] == 65.0
            assert overview["system"]["memory_percent"] == 70.0

    def test_performance_score_calculation(self, monitoring):
        """Test performance score calculation."""
        # Add performance metrics
        monitoring.record_metric("system.cpu.percent", 50.0)
        monitoring.record_metric("system.memory.percent", 60.0)
        monitoring.record_metric("response_time", 0.8)
        monitoring.record_metric("error_rate", 0.01)
        
        score = monitoring._calculate_performance_score()
        
        assert 0 <= score <= 100
        assert isinstance(score, float)

    def test_metrics_export_prometheus(self, monitoring):
        """Test Prometheus format metrics export."""
        # Add test metrics
        monitoring.record_metric("http_requests_total", 1500.0, MetricType.COUNTER, {"method": "GET"})
        monitoring.record_metric("response_time_seconds", 0.250, MetricType.GAUGE)
        
        prometheus_output = monitoring.export_metrics("prometheus")
        
        assert isinstance(prometheus_output, str)
        assert "http_requests_total" in prometheus_output
        assert "response_time_seconds" in prometheus_output
        assert "# HELP" in prometheus_output
        assert "# TYPE" in prometheus_output

    def test_metrics_export_json(self, monitoring):
        """Test JSON format metrics export."""
        # Add test metrics
        monitoring.record_metric("cpu_usage", 65.0)
        monitoring.record_metric("memory_usage", 70.0)
        
        json_output = monitoring.export_metrics("json")
        
        assert isinstance(json_output, str)
        
        # Parse JSON to verify structure
        data = json.loads(json_output)
        assert "timestamp" in data
        assert "metrics" in data
        assert "cpu_usage" in data["metrics"]
        assert "memory_usage" in data["metrics"]

    def test_metrics_export_with_time_window(self, monitoring):
        """Test metrics export with time window filter."""
        # Add old metric
        old_time = datetime.now() - timedelta(hours=2)
        old_metric = MetricPoint(
            name="old.metric",
            value=100.0,
            timestamp=old_time,
            metric_type=MetricType.GAUGE
        )
        monitoring.metrics["old.metric"].append(old_metric)
        
        # Add recent metric
        monitoring.record_metric("recent.metric", 50.0)
        
        # Export with 1-hour window
        json_output = monitoring.export_metrics("json", timedelta(hours=1))
        data = json.loads(json_output)
        
        # Should only include recent metric
        assert "recent.metric" in data["metrics"]
        assert "old.metric" not in data["metrics"]

    @patch('finchat_sec_qa.comprehensive_monitoring.time.sleep')
    def test_monitoring_loop_execution(self, mock_sleep, monitoring):
        """Test execution of the monitoring loop."""
        # Mock monitoring methods
        monitoring._collect_system_metrics = Mock()
        monitoring._run_health_checks = Mock()
        monitoring._check_alert_conditions = Mock()
        monitoring._update_performance_baselines = Mock()
        monitoring._detect_anomalies = Mock()
        monitoring._cleanup_old_data = Mock()
        
        # Set up for single loop iteration
        mock_sleep.side_effect = [None, KeyboardInterrupt()]
        monitoring._running = True
        
        # Run monitoring loop
        with pytest.raises(KeyboardInterrupt):
            monitoring._monitoring_loop()
        
        # Verify methods were called
        monitoring._collect_system_metrics.assert_called_once()
        monitoring._run_health_checks.assert_called_once()
        monitoring._check_alert_conditions.assert_called_once()

    def test_system_metrics_collection(self, monitoring):
        """Test system metrics collection."""
        # Mock psutil for consistent testing
        with patch('finchat_sec_qa.comprehensive_monitoring.psutil') as mock_psutil:
            mock_psutil.cpu_percent.return_value = 45.0
            
            mock_memory = Mock()
            mock_memory.percent = 55.0
            mock_memory.available = 8 * (1024**3)  # 8GB
            mock_psutil.virtual_memory.return_value = mock_memory
            
            mock_disk_usage = Mock()
            mock_disk_usage.used = 300 * (1024**3)
            mock_disk_usage.total = 1000 * (1024**3)
            mock_disk_usage.free = 700 * (1024**3)
            mock_psutil.disk_usage.return_value = mock_disk_usage
            
            mock_process = Mock()
            mock_process.cpu_percent.return_value = 5.0
            mock_memory_info = Mock()
            mock_memory_info.rss = 256 * (1024**2)  # 256MB
            mock_process.memory_info.return_value = mock_memory_info
            mock_process.num_threads.return_value = 8
            mock_psutil.Process.return_value = mock_process
            
            monitoring._collect_system_metrics()
            
            # Verify metrics were recorded
            assert monitoring.get_latest_metric_value("system.cpu.percent") == 45.0
            assert monitoring.get_latest_metric_value("system.memory.percent") == 55.0
            assert monitoring.get_latest_metric_value("process.cpu.percent") == 5.0
            assert monitoring.get_latest_metric_value("process.threads.count") == 8

    def test_health_checks_execution(self, monitoring):
        """Test execution of health checks."""
        # Add custom health check
        def failing_health_check():
            return HealthCheck(
                component="failing_service",
                status=HealthStatus.UNHEALTHY,
                response_time=5.0,
                details={"error": "Service unavailable"}
            )
        
        monitoring.register_health_check("failing_service", failing_health_check)
        
        # Run health checks
        monitoring._run_health_checks()
        
        # Verify health check was executed
        assert "failing_service" in monitoring.health_status
        assert monitoring.health_status["failing_service"].status == HealthStatus.UNHEALTHY

    def test_anomaly_detection(self, monitoring):
        """Test anomaly detection functionality."""
        # Add baseline metrics
        baseline_values = [10.0, 11.0, 9.0, 10.5, 9.5] * 20  # 100 values around 10
        for value in baseline_values:
            monitoring.record_metric("test.anomaly", value)
        
        # Update performance baselines
        monitoring._update_performance_baselines()
        
        # Add anomalous value
        monitoring.record_metric("test.anomaly", 50.0)  # Significant outlier
        
        # Run anomaly detection
        initial_alerts = len(monitoring.alerts)
        monitoring._detect_anomalies()
        
        # Should detect anomaly and create alert
        assert len(monitoring.alerts) > initial_alerts

    def test_data_cleanup(self, monitoring):
        """Test cleanup of old data."""
        # Add old alerts
        old_time = datetime.now() - timedelta(days=35)
        for i in range(50):
            old_alert = Alert(
                id=f"old_alert_{i}",
                timestamp=old_time,
                level=AlertLevel.INFO,
                component="test",
                message="Old alert"
            )
            monitoring.alerts.append(old_alert)
        
        # Add recent alerts
        for i in range(10):
            recent_alert = Alert(
                id=f"recent_alert_{i}",
                timestamp=datetime.now(),
                level=AlertLevel.INFO,
                component="test",
                message="Recent alert"
            )
            monitoring.alerts.append(recent_alert)
        
        # Run cleanup
        monitoring._cleanup_old_data()
        
        # Should have removed old alerts
        assert len(monitoring.alerts) <= 10

    def test_error_handling_in_monitoring(self, monitoring):
        """Test error handling in monitoring operations."""
        # Mock method that raises exception
        monitoring._collect_system_metrics = Mock(side_effect=Exception("Test error"))
        
        # Should handle errors gracefully
        try:
            monitoring._monitoring_loop()
        except KeyboardInterrupt:
            pass  # Expected from loop termination
        except Exception:
            pytest.fail("Monitoring should handle errors gracefully")

    def test_metric_aggregations(self, monitoring):
        """Test metric aggregations functionality."""
        # Add metrics for aggregation
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        for value in values:
            monitoring.record_metric("aggregation.test", value)
            monitoring._update_metric_aggregations("aggregation.test", value)
        
        # Check aggregations
        agg = monitoring.metric_aggregations["aggregation.test"]
        assert agg["count"] == 5
        assert agg["sum"] == 150.0
        assert agg["min"] == 10.0
        assert agg["max"] == 50.0
        assert agg["mean"] == 30.0

    def test_performance_baselines_update(self, monitoring):
        """Test performance baselines update."""
        # Add sufficient data for baseline calculation
        values = list(range(100, 200))  # 100 values from 100 to 199
        for value in values:
            monitoring.record_metric("baseline.test", float(value))
        
        # Update baselines
        monitoring._update_performance_baselines()
        
        # Check baseline was calculated
        assert "baseline.test" in monitoring.performance_baselines
        baseline = monitoring.performance_baselines["baseline.test"]
        assert "mean" in baseline
        assert "std" in baseline
        assert "p95" in baseline
        assert "p99" in baseline


class TestGlobalMonitoring:
    """Test cases for global monitoring functions."""

    def test_get_monitoring_singleton(self):
        """Test that get_monitoring returns singleton instance."""
        monitor1 = get_monitoring()
        monitor2 = get_monitoring()
        
        assert monitor1 is monitor2
        assert monitor1._running  # Should be auto-started
        
        # Cleanup
        monitor1.stop_monitoring()


@pytest.mark.integration
class TestMonitoringIntegration:
    """Integration tests for the monitoring system."""

    def test_full_monitoring_cycle(self, monitoring):
        """Test complete monitoring cycle."""
        # Start monitoring
        monitoring.start_monitoring()
        
        # Add metrics that should trigger alerts
        monitoring.record_metric("system.cpu.percent", 95.0)
        monitoring.record_metric("system.memory.percent", 92.0)
        
        # Give monitoring system time to process
        time.sleep(0.5)
        
        # Should have created alerts
        active_alerts = monitoring.get_active_alerts()
        assert len(active_alerts) > 0
        
        # Stop monitoring
        monitoring.stop_monitoring()

    def test_health_status_integration(self, monitoring):
        """Test health status integration."""
        # Run health checks
        monitoring._run_health_checks()
        
        # Get overall health status
        health_status = monitoring.get_health_status()
        
        # Should have component health statuses
        assert "components" in health_status
        assert len(health_status["components"]) > 0
        assert "overall_status" in health_status

    def test_metrics_and_alerts_integration(self, monitoring):
        """Test integration between metrics and alerts."""
        # Record metric that should trigger alert
        monitoring.record_metric("system.cpu.percent", 96.0)
        
        # Check alert conditions
        monitoring._check_alert_conditions()
        
        # Should have created critical CPU alert
        critical_alerts = [
            alert for alert in monitoring.alerts 
            if alert.level == AlertLevel.CRITICAL and not alert.resolved
        ]
        assert len(critical_alerts) > 0


@pytest.mark.performance
class TestMonitoringPerformance:
    """Performance tests for the monitoring system."""

    def test_metric_recording_performance(self, monitoring):
        """Test performance of metric recording."""
        # Measure recording time for many metrics
        start_time = time.time()
        
        for i in range(1000):
            monitoring.record_metric(f"performance.test.{i % 10}", float(i))
        
        recording_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert recording_time < 2.0  # 2 seconds for 1000 metrics

    def test_alert_checking_performance(self, monitoring):
        """Test performance of alert condition checking."""
        # Add many metrics
        for i in range(100):
            monitoring.record_metric(f"test.metric.{i}", float(i))
        
        # Add many alert rules
        for i in range(50):
            monitoring.add_alert_rule(
                name=f"test_rule_{i}",
                condition=f"test.metric.{i % 10} > 50",
                alert_level=AlertLevel.WARNING,
                component="test",
                message=f"Test rule {i} triggered"
            )
        
        # Measure alert checking time
        start_time = time.time()
        monitoring._check_alert_conditions()
        checking_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert checking_time < 5.0  # 5 seconds for 50 rules

    def test_export_performance(self, monitoring):
        """Test performance of metrics export."""
        # Add many metrics
        for i in range(1000):
            monitoring.record_metric(f"export.test.{i}", float(i))
        
        # Measure export time
        start_time = time.time()
        prometheus_output = monitoring.export_metrics("prometheus")
        export_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert export_time < 3.0  # 3 seconds for 1000 metrics
        assert len(prometheus_output) > 0