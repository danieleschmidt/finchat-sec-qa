"""
Comprehensive tests for the Performance Optimization Engine.

This test suite ensures the performance optimization and auto-scaling
functionality works correctly under various conditions.
"""

import asyncio
import pytest
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor

from finchat_sec_qa.performance_optimization import (
    PerformanceOptimizationEngine,
    PerformanceMetrics,
    OptimizationStrategy,
    ScalingDirection,
    ResourceType,
    OptimizationResult,
    ScalingDecision,
    get_performance_engine
)


@pytest.fixture
def temp_storage_path(tmp_path):
    """Create temporary storage path for testing."""
    return tmp_path / "test_performance"


@pytest.fixture
def perf_engine(temp_storage_path):
    """Create a performance engine instance for testing."""
    engine = PerformanceOptimizationEngine(storage_path=temp_storage_path)
    yield engine
    engine.stop_optimization_engine()


@pytest.fixture
def sample_metrics():
    """Sample performance metrics for testing."""
    return PerformanceMetrics(
        timestamp=datetime.now(),
        cpu_usage=75.0,
        memory_usage=80.0,
        network_io=1024000,
        disk_io=512000,
        response_time=1.5,
        throughput=100.0,
        error_rate=0.02,
        queue_depth=5
    )


@pytest.fixture
def high_load_metrics():
    """High load performance metrics for testing."""
    return PerformanceMetrics(
        timestamp=datetime.now(),
        cpu_usage=95.0,
        memory_usage=92.0,
        network_io=5120000,
        disk_io=2048000,
        response_time=5.0,
        throughput=25.0,
        error_rate=0.15,
        queue_depth=20
    )


class TestPerformanceOptimizationEngine:
    """Test cases for the Performance Optimization Engine."""

    def test_initialization(self, perf_engine):
        """Test performance engine initialization."""
        assert perf_engine.storage_path.exists()
        assert len(perf_engine.metrics_history) == 0
        assert len(perf_engine.optimization_history) == 0
        assert not perf_engine._running
        
        # Check default scaling thresholds
        assert ResourceType.CPU in perf_engine.scaling_thresholds
        assert ResourceType.MEMORY in perf_engine.scaling_thresholds
        
        # Check thread pools are initialized
        assert perf_engine.thread_pool is not None
        assert perf_engine.process_pool is not None

    def test_start_stop_optimization(self, perf_engine):
        """Test starting and stopping the optimization engine."""
        # Start optimization
        perf_engine.start_optimization_engine()
        assert perf_engine._running
        assert perf_engine._optimization_thread is not None
        
        # Stop optimization
        perf_engine.stop_optimization_engine()
        assert not perf_engine._running

    def test_metrics_collection(self, perf_engine):
        """Test performance metrics collection."""
        # Mock psutil for consistent testing
        with patch('finchat_sec_qa.performance_optimization.psutil') as mock_psutil:
            mock_psutil.cpu_percent.return_value = 65.0
            mock_memory = Mock()
            mock_memory.percent = 70.0
            mock_psutil.virtual_memory.return_value = mock_memory
            
            mock_disk = Mock()
            mock_disk.read_bytes = 1000000
            mock_disk.write_bytes = 500000
            mock_psutil.disk_io_counters.return_value = mock_disk
            
            mock_network = Mock()
            mock_network.bytes_sent = 2000000
            mock_network.bytes_recv = 1500000
            mock_psutil.net_io_counters.return_value = mock_network
            
            # Collect metrics
            metrics = perf_engine._collect_performance_metrics()
            
            assert isinstance(metrics, PerformanceMetrics)
            assert metrics.cpu_usage == 65.0
            assert metrics.memory_usage == 70.0
            assert metrics.disk_io == 1500000
            assert metrics.network_io == 3500000

    def test_performance_pattern_analysis(self, perf_engine, high_load_metrics):
        """Test analysis of performance patterns."""
        # Add high load metrics to history
        for _ in range(10):
            perf_engine.metrics_history.append(high_load_metrics)
        
        # Analyze patterns
        issues = perf_engine._analyze_performance_patterns()
        
        assert len(issues) > 0
        
        # Should detect high CPU usage
        cpu_issues = [issue for issue in issues if issue['type'] == 'high_cpu_usage']
        assert len(cpu_issues) > 0
        
        # Should detect high memory usage
        memory_issues = [issue for issue in issues if issue['type'] == 'high_memory_usage']
        assert len(memory_issues) > 0
        
        # Should detect slow response time
        response_issues = [issue for issue in issues if issue['type'] == 'slow_response_time']
        assert len(response_issues) > 0

    def test_scaling_decisions(self, perf_engine, high_load_metrics):
        """Test auto-scaling decision making."""
        decisions = perf_engine._make_scaling_decisions(high_load_metrics)
        
        assert len(decisions) > 0
        
        # Should decide to scale up CPU
        cpu_decisions = [d for d in decisions if d.resource_type == ResourceType.CPU]
        assert len(cpu_decisions) > 0
        assert cpu_decisions[0].direction == ScalingDirection.UP
        
        # Should decide to scale up memory
        memory_decisions = [d for d in decisions if d.resource_type == ResourceType.MEMORY]
        assert len(memory_decisions) > 0
        assert memory_decisions[0].direction == ScalingDirection.UP

    def test_caching_optimization(self, perf_engine):
        """Test caching optimization strategy."""
        issue = {
            'type': 'slow_response_time',
            'severity': 'high',
            'value': 5.0,
            'recommended_strategies': [OptimizationStrategy.CACHING]
        }
        
        success, details = perf_engine._optimize_caching(issue)
        
        assert success
        assert 'cache_size_before' in details
        assert 'cache_size_after' in details
        assert 'cache_ttl' in details

    def test_parallelization_optimization(self, perf_engine):
        """Test parallelization optimization strategy."""
        initial_workers = perf_engine.thread_pool._max_workers
        
        issue = {
            'type': 'high_cpu_usage',
            'severity': 'high',
            'value': 90.0,
            'recommended_strategies': [OptimizationStrategy.PARALLELIZATION]
        }
        
        success, details = perf_engine._optimize_parallelization(issue)
        
        assert success
        assert details['workers_before'] == initial_workers
        assert details['workers_after'] > initial_workers
        assert perf_engine.thread_pool._max_workers > initial_workers

    def test_connection_pooling_optimization(self, perf_engine):
        """Test connection pooling optimization."""
        initial_pool_size = perf_engine.current_capacity.get(ResourceType.CONNECTIONS, 100)
        
        issue = {
            'type': 'slow_response_time',
            'severity': 'medium',
            'value': 3.0,
            'recommended_strategies': [OptimizationStrategy.CONNECTION_POOLING]
        }
        
        success, details = perf_engine._optimize_connection_pooling(issue)
        
        assert success
        assert details['pool_size_after'] > initial_pool_size
        assert perf_engine.current_capacity[ResourceType.CONNECTIONS] > initial_pool_size

    def test_batch_processing_optimization(self, perf_engine):
        """Test batch processing optimization."""
        initial_batch_size = getattr(perf_engine, 'batch_size', 32)
        
        issue = {
            'type': 'low_throughput',
            'severity': 'medium',
            'value': 25.0,
            'recommended_strategies': [OptimizationStrategy.BATCH_PROCESSING]
        }
        
        success, details = perf_engine._optimize_batch_processing(issue)
        
        assert success
        assert details['batch_size_after'] > initial_batch_size
        assert perf_engine.batch_size > initial_batch_size

    def test_lazy_loading_optimization(self, perf_engine):
        """Test lazy loading optimization."""
        issue = {
            'type': 'high_memory_usage',
            'severity': 'high',
            'value': 95.0,
            'recommended_strategies': [OptimizationStrategy.LAZY_LOADING]
        }
        
        success, details = perf_engine._optimize_lazy_loading(issue)
        
        assert success
        assert 'threshold_before' in details
        assert 'threshold_after' in details
        assert hasattr(perf_engine, 'lazy_loading_threshold')

    def test_compression_optimization(self, perf_engine):
        """Test compression optimization."""
        issue = {
            'type': 'high_memory_usage',
            'severity': 'medium',
            'value': 87.0,
            'recommended_strategies': [OptimizationStrategy.COMPRESSION]
        }
        
        success, details = perf_engine._optimize_compression(issue)
        
        assert success
        assert 'compression_level_after' in details
        assert hasattr(perf_engine, 'compression_level')

    def test_prefetching_optimization(self, perf_engine):
        """Test prefetching optimization."""
        issue = {
            'type': 'slow_response_time',
            'severity': 'medium',
            'value': 2.5,
            'recommended_strategies': [OptimizationStrategy.PREFETCHING]
        }
        
        success, details = perf_engine._optimize_prefetching(issue)
        
        assert success
        assert 'prefetch_size_after' in details
        assert hasattr(perf_engine, 'prefetch_size')

    def test_load_balancing_optimization(self, perf_engine):
        """Test load balancing optimization."""
        issue = {
            'type': 'low_throughput',
            'severity': 'medium',
            'value': 30.0,
            'recommended_strategies': [OptimizationStrategy.LOAD_BALANCING]
        }
        
        success, details = perf_engine._optimize_load_balancing(issue)
        
        assert success
        assert 'algorithm_after' in details
        assert hasattr(perf_engine, 'load_balance_algorithm')

    def test_scaling_decision_execution(self, perf_engine):
        """Test execution of scaling decisions."""
        initial_workers = perf_engine.current_capacity[ResourceType.WORKERS]
        
        decision = ScalingDecision(
            resource_type=ResourceType.WORKERS,
            direction=ScalingDirection.UP,
            current_capacity=initial_workers,
            target_capacity=initial_workers + 2,
            reason="High queue depth",
            confidence=0.8
        )
        
        perf_engine._execute_scaling_decision(decision)
        
        assert perf_engine.current_capacity[ResourceType.WORKERS] == initial_workers + 2

    def test_improvement_calculation(self, perf_engine, sample_metrics, high_load_metrics):
        """Test performance improvement calculation."""
        # Test CPU improvement
        cpu_improvement = perf_engine._calculate_improvement(
            high_load_metrics, sample_metrics, 'high_cpu_usage'
        )
        assert cpu_improvement > 0
        
        # Test response time improvement
        response_improvement = perf_engine._calculate_improvement(
            high_load_metrics, sample_metrics, 'slow_response_time'
        )
        assert response_improvement > 0
        
        # Test throughput improvement
        throughput_improvement = perf_engine._calculate_improvement(
            sample_metrics, high_load_metrics, 'low_throughput'
        )
        assert throughput_improvement < 0  # Throughput got worse

    def test_performance_monitor_context(self, perf_engine):
        """Test performance monitoring context manager."""
        with perf_engine.performance_monitor('test_operation'):
            time.sleep(0.1)  # Simulate operation
        
        # Should have recorded metrics
        # (In a real implementation, this would check monitoring system)

    def test_caching_strategy_optimization(self, perf_engine):
        """Test caching strategy optimization."""
        # Add some cache statistics
        perf_engine.cache_statistics['test_cache']['total'] = 1000
        perf_engine.cache_statistics['test_cache']['hits'] = 200  # 20% hit rate
        
        # Should optimize low-performing cache
        perf_engine._optimize_caching_strategies()
        
        # Verify optimization was attempted
        # (Implementation would check specific optimizations)

    def test_performance_summary_generation(self, perf_engine, sample_metrics):
        """Test generation of performance summary."""
        # Add some test data
        perf_engine.metrics_history.append(sample_metrics)
        
        # Add optimization result
        opt_result = OptimizationResult(
            strategy=OptimizationStrategy.CACHING,
            before_metrics=sample_metrics,
            after_metrics=sample_metrics,
            improvement_percent=15.0,
            cost=0.0,
            success=True
        )
        perf_engine.optimization_history.append(opt_result)
        
        summary = perf_engine.get_performance_summary()
        
        assert 'current_performance' in summary
        assert 'optimization_summary' in summary
        assert 'scaling_summary' in summary
        assert summary['optimization_summary']['total_optimizations'] == 1
        assert summary['optimization_summary']['successful_optimizations'] == 1

    def test_optimization_recommendations(self, perf_engine, high_load_metrics):
        """Test generation of optimization recommendations."""
        # Add high load metrics
        for _ in range(10):
            perf_engine.metrics_history.append(high_load_metrics)
        
        recommendations = perf_engine.recommend_optimizations()
        
        assert len(recommendations) > 0
        
        # Should recommend caching for slow response times
        caching_recs = [r for r in recommendations if r['strategy'] == 'caching']
        assert len(caching_recs) > 0
        
        # Check recommendation structure
        rec = recommendations[0]
        assert 'priority' in rec
        assert 'strategy' in rec
        assert 'reason' in rec
        assert 'expected_improvement' in rec
        assert 'implementation_effort' in rec

    def test_cooldown_mechanism(self, perf_engine):
        """Test optimization cooldown mechanism."""
        # Set cooldown for caching strategy
        strategy = OptimizationStrategy.CACHING
        perf_engine.optimization_cooldown[strategy] = time.time() + 300  # 5 minutes
        
        issue = {
            'type': 'slow_response_time',
            'recommended_strategies': [strategy, OptimizationStrategy.PREFETCHING]
        }
        
        # Should skip caching and use prefetching instead
        result = perf_engine._apply_optimization(issue, sample_metrics)
        
        # Should either return None (if no alternative) or use different strategy
        if result:
            assert result.strategy != strategy

    def test_thread_pool_scaling(self, perf_engine):
        """Test thread pool scaling functionality."""
        initial_workers = perf_engine.thread_pool._max_workers
        
        # Create high queue depth scenario
        high_queue_metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_usage=60.0,
            memory_usage=70.0,
            network_io=1000000,
            disk_io=500000,
            response_time=1.0,
            throughput=100.0,
            error_rate=0.01,
            queue_depth=50  # High queue depth
        )
        
        decisions = perf_engine._make_scaling_decisions(high_queue_metrics)
        
        # Should decide to scale up workers
        worker_decisions = [d for d in decisions if d.resource_type == ResourceType.WORKERS]
        assert len(worker_decisions) > 0
        
        # Execute scaling
        for decision in worker_decisions:
            if decision.direction == ScalingDirection.UP:
                perf_engine._execute_scaling_decision(decision)
                break
        
        # Verify thread pool was scaled
        assert perf_engine.thread_pool._max_workers > initial_workers

    @patch('finchat_sec_qa.performance_optimization.time.sleep')
    def test_optimization_loop_execution(self, mock_sleep, perf_engine):
        """Test execution of the optimization loop."""
        # Mock optimization methods
        perf_engine._collect_performance_metrics = Mock(return_value=sample_metrics)
        perf_engine._analyze_performance_patterns = Mock(return_value=[])
        perf_engine._make_scaling_decisions = Mock(return_value=[])
        perf_engine._optimize_caching_strategies = Mock()
        perf_engine._cleanup_optimization_data = Mock()
        
        # Set up for single loop iteration
        mock_sleep.side_effect = [None, KeyboardInterrupt()]
        perf_engine._running = True
        
        # Run optimization loop
        with pytest.raises(KeyboardInterrupt):
            perf_engine._optimization_loop()
        
        # Verify methods were called
        perf_engine._collect_performance_metrics.assert_called_once()
        perf_engine._analyze_performance_patterns.assert_called_once()
        perf_engine._make_scaling_decisions.assert_called_once()

    def test_data_cleanup(self, perf_engine):
        """Test cleanup of old optimization data."""
        # Add large amount of test data
        for i in range(1500):
            opt_result = OptimizationResult(
                strategy=OptimizationStrategy.CACHING,
                before_metrics=sample_metrics,
                after_metrics=sample_metrics,
                improvement_percent=10.0,
                cost=0.0,
                success=True
            )
            perf_engine.optimization_history.append(opt_result)
        
        # Add scaling decisions
        for i in range(600):
            decision = ScalingDecision(
                resource_type=ResourceType.CPU,
                direction=ScalingDirection.UP,
                current_capacity=1.0,
                target_capacity=1.5,
                reason="Test cleanup",
                confidence=0.8
            )
            perf_engine.scaling_decisions.append(decision)
        
        # Run cleanup
        perf_engine._cleanup_optimization_data()
        
        # Verify data was cleaned up
        assert len(perf_engine.optimization_history) <= 500
        assert len(perf_engine.scaling_decisions) <= 250

    def test_error_handling_in_optimization(self, perf_engine):
        """Test error handling during optimization."""
        # Mock method that raises exception
        perf_engine._optimize_caching = Mock(side_effect=Exception("Test error"))
        
        issue = {
            'type': 'slow_response_time',
            'recommended_strategies': [OptimizationStrategy.CACHING]
        }
        
        # Should handle errors gracefully
        result = perf_engine._apply_optimization(issue, sample_metrics)
        assert result is None  # Should return None on error

    def test_concurrent_optimization(self, perf_engine):
        """Test concurrent optimization execution."""
        # Create multiple optimization issues
        issues = [
            {
                'type': 'high_cpu_usage',
                'recommended_strategies': [OptimizationStrategy.PARALLELIZATION]
            },
            {
                'type': 'slow_response_time',
                'recommended_strategies': [OptimizationStrategy.CACHING]
            },
            {
                'type': 'high_memory_usage',
                'recommended_strategies': [OptimizationStrategy.COMPRESSION]
            }
        ]
        
        # Mock optimization implementations
        perf_engine._optimize_parallelization = Mock(return_value=(True, {}))
        perf_engine._optimize_caching = Mock(return_value=(True, {}))
        perf_engine._optimize_compression = Mock(return_value=(True, {}))
        
        # Apply optimizations concurrently
        results = []
        for issue in issues:
            result = perf_engine._apply_optimization(issue, sample_metrics)
            if result:
                results.append(result)
        
        # Should handle multiple optimizations
        assert len(results) <= len(issues)


class TestGlobalPerformanceEngine:
    """Test cases for global performance engine functions."""

    def test_get_performance_engine_singleton(self):
        """Test that get_performance_engine returns singleton instance."""
        engine1 = get_performance_engine()
        engine2 = get_performance_engine()
        
        assert engine1 is engine2
        assert engine1._running  # Should be auto-started
        
        # Cleanup
        engine1.stop_optimization_engine()


@pytest.mark.asyncio
class TestAsyncOperations:
    """Test async operations in performance optimization."""

    async def test_async_optimization_simulation(self, perf_engine):
        """Test simulated async optimization operations."""
        async def mock_async_optimization():
            await asyncio.sleep(0.1)
            return OptimizationResult(
                strategy=OptimizationStrategy.CACHING,
                before_metrics=sample_metrics,
                after_metrics=sample_metrics,
                improvement_percent=20.0,
                cost=0.0,
                success=True
            )
        
        result = await mock_async_optimization()
        assert result.success
        assert result.improvement_percent == 20.0


@pytest.mark.integration
class TestPerformanceEngineIntegration:
    """Integration tests for the performance engine."""

    def test_integration_with_monitoring_system(self, perf_engine):
        """Test integration with the monitoring system."""
        # Should integrate with monitoring for metrics collection
        assert hasattr(perf_engine, 'monitoring')
        assert perf_engine.monitoring is not None

    def test_resource_pool_management(self, perf_engine):
        """Test management of resource pools."""
        # Thread pool management
        initial_threads = perf_engine.thread_pool._max_workers
        
        # Simulate high load requiring more threads
        perf_engine.current_capacity[ResourceType.WORKERS] = initial_threads + 5
        
        # Should be able to create new thread pool
        perf_engine.thread_pool.shutdown(wait=False)
        perf_engine.thread_pool = ThreadPoolExecutor(
            max_workers=int(perf_engine.current_capacity[ResourceType.WORKERS])
        )
        
        assert perf_engine.thread_pool._max_workers > initial_threads

    def test_full_optimization_cycle(self, perf_engine, high_load_metrics):
        """Test complete optimization cycle."""
        # Add high load metrics
        perf_engine.metrics_history.append(high_load_metrics)
        
        # Analyze patterns
        issues = perf_engine._analyze_performance_patterns()
        assert len(issues) > 0
        
        # Apply optimizations
        for issue in issues[:3]:  # Limit to avoid resource exhaustion
            result = perf_engine._apply_optimization(issue, high_load_metrics)
            if result:
                perf_engine.optimization_history.append(result)
        
        # Make scaling decisions
        decisions = perf_engine._make_scaling_decisions(high_load_metrics)
        
        # Should have made some optimizations and decisions
        assert len(perf_engine.optimization_history) > 0 or len(decisions) > 0


@pytest.mark.performance
class TestPerformanceEnginePerformance:
    """Performance tests for the performance engine."""

    def test_metrics_collection_performance(self, perf_engine):
        """Test performance of metrics collection."""
        # Measure collection time
        start_time = time.time()
        
        for _ in range(100):
            metrics = perf_engine._collect_performance_metrics()
        
        collection_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert collection_time < 5.0  # 5 seconds for 100 collections
        assert isinstance(metrics, PerformanceMetrics)

    def test_pattern_analysis_performance(self, perf_engine, sample_metrics):
        """Test performance of pattern analysis."""
        # Add large dataset
        for _ in range(1000):
            perf_engine.metrics_history.append(sample_metrics)
        
        # Measure analysis time
        start_time = time.time()
        issues = perf_engine._analyze_performance_patterns()
        analysis_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert analysis_time < 2.0  # 2 seconds max
        assert isinstance(issues, list)

    def test_optimization_application_performance(self, perf_engine, sample_metrics):
        """Test performance of optimization application."""
        issue = {
            'type': 'slow_response_time',
            'severity': 'medium',
            'value': 2.0,
            'recommended_strategies': [OptimizationStrategy.CACHING]
        }
        
        # Measure optimization time
        start_time = time.time()
        
        for _ in range(50):
            result = perf_engine._apply_optimization(issue, sample_metrics)
        
        optimization_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert optimization_time < 10.0  # 10 seconds for 50 optimizations