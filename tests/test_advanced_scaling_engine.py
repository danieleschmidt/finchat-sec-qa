"""
Tests for Advanced Scaling Engine - Generation 3 Quality Gates
TERRAGON SDLC v4.0 - Comprehensive Testing
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from finchat_sec_qa.advanced_scaling_engine import (
    AdvancedScalingEngine,
    AdvancedConnectionPool,
    IntelligentCache,
    ResourceMetrics,
    ScalingDecision,
    ScalingAction,
    ResourceType,
    WorkerNode,
    scaling_engine,
    get_connection_pool,
    get_intelligent_cache
)


class TestAdvancedConnectionPool:
    """Test advanced connection pool functionality."""
    
    def test_pool_initialization(self):
        """Test connection pool initialization."""
        pool = AdvancedConnectionPool(max_connections=50, min_connections=5)
        
        assert pool.max_connections == 50
        assert pool.min_connections == 5
        assert pool.total_created == 0
        assert pool.total_closed == 0
    
    @pytest.mark.asyncio
    async def test_connection_acquire_release(self):
        """Test connection acquisition and release."""
        pool = AdvancedConnectionPool(max_connections=2, min_connections=1)
        
        # Acquire connection
        conn1 = await pool.acquire()
        assert conn1 is not None
        assert len(pool._in_use) == 1
        
        # Acquire second connection
        conn2 = await pool.acquire()
        assert conn2 is not None
        assert len(pool._in_use) == 2
        
        # Release connections
        await pool.release(conn1)
        assert len(pool._in_use) == 1
        assert len(pool._connections) == 1
        
        await pool.release(conn2)
        assert len(pool._in_use) == 0
        assert len(pool._connections) == 2
    
    @pytest.mark.asyncio
    async def test_connection_pool_timeout(self):
        """Test connection pool timeout behavior."""
        pool = AdvancedConnectionPool(max_connections=1, acquire_timeout=0.1)
        
        # Acquire the only connection
        conn1 = await pool.acquire()
        
        # Try to acquire another connection - should timeout
        with pytest.raises(TimeoutError):
            await pool.acquire()
        
        # Release connection
        await pool.release(conn1)
    
    def test_pool_stats(self):
        """Test connection pool statistics."""
        pool = AdvancedConnectionPool(max_connections=10, min_connections=2)
        
        stats = pool.get_pool_stats()
        
        assert 'available_connections' in stats
        assert 'in_use_connections' in stats
        assert 'max_connections' in stats
        assert 'total_created' in stats
        assert 'total_closed' in stats
        assert 'avg_wait_time' in stats
        assert stats['max_connections'] == 10


class TestIntelligentCache:
    """Test intelligent cache functionality."""
    
    def test_cache_initialization(self):
        """Test cache initialization."""
        cache = IntelligentCache(max_size=1000, ttl_seconds=300)
        
        assert cache.max_size == 1000
        assert cache.ttl_seconds == 300
        assert cache.enable_prediction is True
    
    def test_cache_set_get(self):
        """Test basic cache set and get operations."""
        cache = IntelligentCache(max_size=10, ttl_seconds=3600)
        
        # Set value
        result = cache.set("test_key", "test_value")
        assert result is True
        
        # Get value
        value = cache.get("test_key")
        assert value == "test_value"
        
        # Get non-existent key
        value = cache.get("non_existent")
        assert value is None
    
    def test_cache_ttl_expiration(self):
        """Test cache TTL expiration."""
        cache = IntelligentCache(max_size=10, ttl_seconds=1)  # 1 second TTL
        
        cache.set("test_key", "test_value")
        
        # Should be available immediately
        assert cache.get("test_key") == "test_value"
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should be expired
        assert cache.get("test_key") is None
    
    def test_cache_eviction(self):
        """Test intelligent cache eviction."""
        cache = IntelligentCache(max_size=2, ttl_seconds=3600)
        
        # Fill cache to capacity
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        # Access key1 to make it more recently used
        cache.get("key1")
        
        # Add third key - should evict key2
        cache.set("key3", "value3")
        
        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None  # Should be evicted
        assert cache.get("key3") == "value3"
    
    def test_cache_prediction(self):
        """Test cache prediction functionality."""
        cache = IntelligentCache(max_size=10, ttl_seconds=3600, enable_prediction=True)
        
        # First set the key so it exists in cache
        cache.set("pattern_key", "pattern_value")
        
        # Simulate access pattern
        for i in range(5):
            cache.get("pattern_key")  # This should build access pattern
            time.sleep(0.1)
        
        # Should have built some access pattern
        assert "pattern_key" in cache._access_patterns
        assert len(cache._access_patterns["pattern_key"]) > 0
        
        # Test prediction
        probability = cache._predict_access_probability("pattern_key")
        assert 0 <= probability <= 1
    
    def test_cache_stats(self):
        """Test cache statistics."""
        cache = IntelligentCache(max_size=10, ttl_seconds=3600)
        
        # Add some data
        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss
        
        stats = cache.get_cache_stats()
        
        assert 'cache_size' in stats
        assert 'hit_rate' in stats
        assert 'total_hits' in stats
        assert 'total_misses' in stats
        assert stats['cache_size'] == 1
        assert stats['total_hits'] == 1
        assert stats['total_misses'] == 1


class TestAdvancedScalingEngine:
    """Test advanced scaling engine functionality."""
    
    @pytest.fixture
    def scaling_engine_instance(self):
        """Create scaling engine instance for testing."""
        return AdvancedScalingEngine()
    
    def test_scaling_engine_initialization(self, scaling_engine_instance):
        """Test scaling engine initialization."""
        engine = scaling_engine_instance
        
        assert engine.cpu_scale_up_threshold == 70.0
        assert engine.cpu_scale_down_threshold == 30.0
        assert engine.memory_scale_up_threshold == 80.0
        assert engine.max_workers == 20
        assert engine.min_workers == 2
        assert engine.current_workers == 2
        assert len(engine.worker_nodes) == 0
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self, scaling_engine_instance):
        """Test resource metrics collection."""
        engine = scaling_engine_instance
        
        with patch('psutil.cpu_percent', return_value=50.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_io_counters') as mock_disk, \
             patch('psutil.net_io_counters') as mock_net:
            
            mock_memory.return_value.percent = 60.0
            mock_disk.return_value.read_bytes = 1000
            mock_disk.return_value.write_bytes = 500
            mock_net.return_value.bytes_sent = 2000
            mock_net.return_value.bytes_recv = 1500
            
            metrics = await engine._collect_metrics()
            
            assert isinstance(metrics, ResourceMetrics)
            assert metrics.cpu_percent == 50.0
            assert metrics.memory_percent == 60.0
            assert metrics.disk_io_read == 1000
            assert metrics.disk_io_write == 500
    
    @pytest.mark.asyncio
    async def test_scaling_decision_cpu_scale_up(self, scaling_engine_instance):
        """Test CPU-based scale up decision."""
        engine = scaling_engine_instance
        
        # Create metrics that should trigger scale up
        metrics = ResourceMetrics(
            timestamp=datetime.now(),
            cpu_percent=80.0,  # Above threshold
            memory_percent=50.0,
            disk_io_read=0,
            disk_io_write=0,
            network_bytes_sent=0,
            network_bytes_recv=0,
            active_connections=0,
            cache_hit_rate=0.8,
            avg_response_time=1.0,
            requests_per_second=10.0
        )
        
        decision = await engine._analyze_and_decide(metrics)
        
        assert decision is not None
        assert decision.action == ScalingAction.SCALE_UP
        assert decision.resource_type == ResourceType.CPU
        assert decision.current_value == 80.0
        assert decision.confidence > 0
    
    @pytest.mark.asyncio
    async def test_scaling_decision_memory_scale_up(self, scaling_engine_instance):
        """Test memory-based scale up decision."""
        engine = scaling_engine_instance
        
        metrics = ResourceMetrics(
            timestamp=datetime.now(),
            cpu_percent=50.0,
            memory_percent=85.0,  # Above threshold
            disk_io_read=0,
            disk_io_write=0,
            network_bytes_sent=0,
            network_bytes_recv=0,
            active_connections=0,
            cache_hit_rate=0.8,
            avg_response_time=1.0,
            requests_per_second=10.0
        )
        
        decision = await engine._analyze_and_decide(metrics)
        
        assert decision is not None
        assert decision.action == ScalingAction.SCALE_UP
        assert decision.resource_type == ResourceType.MEMORY
        assert decision.current_value == 85.0
    
    @pytest.mark.asyncio
    async def test_scaling_decision_no_action_needed(self, scaling_engine_instance):
        """Test when no scaling action is needed."""
        engine = scaling_engine_instance
        
        metrics = ResourceMetrics(
            timestamp=datetime.now(),
            cpu_percent=50.0,  # Normal range
            memory_percent=50.0,  # Normal range
            disk_io_read=0,
            disk_io_write=0,
            network_bytes_sent=0,
            network_bytes_recv=0,
            active_connections=0,
            cache_hit_rate=0.8,
            avg_response_time=1.0,  # Normal
            requests_per_second=10.0
        )
        
        decision = await engine._analyze_and_decide(metrics)
        assert decision is None
    
    @pytest.mark.asyncio
    async def test_worker_management(self, scaling_engine_instance):
        """Test worker node management."""
        engine = scaling_engine_instance
        initial_workers = engine.current_workers
        
        # Add worker
        await engine._add_worker()
        assert engine.current_workers == initial_workers + 1
        assert len(engine.worker_nodes) == 1
        
        # Remove worker
        await engine._remove_worker()
        assert engine.current_workers == initial_workers
        assert len(engine.worker_nodes) == 0
    
    @pytest.mark.asyncio
    async def test_health_check_workers(self, scaling_engine_instance):
        """Test worker health checking."""
        engine = scaling_engine_instance
        
        # Add a worker with high error rate
        worker = WorkerNode(
            node_id="test_worker",
            status="active",
            error_rate=0.15,  # 15% error rate
            last_health_check=datetime.now() - timedelta(minutes=10)
        )
        engine.worker_nodes["test_worker"] = worker
        engine.current_workers = 1
        
        await engine._health_check_workers()
        
        # Worker should be marked as failed and replaced
        assert "test_worker" not in engine.worker_nodes
        # Should have added replacement, so we expect 2 workers now (1 original + 1 replacement)
        assert engine.current_workers == 2
    
    def test_load_prediction(self, scaling_engine_instance):
        """Test future load prediction."""
        engine = scaling_engine_instance
        
        # Add some historical metrics
        for i in range(10):
            metrics = ResourceMetrics(
                timestamp=datetime.now(),
                cpu_percent=50.0 + i,  # Increasing trend
                memory_percent=40.0,
                disk_io_read=0,
                disk_io_write=0,
                network_bytes_sent=0,
                network_bytes_recv=0,
                active_connections=0,
                cache_hit_rate=0.8,
                avg_response_time=1.0,
                requests_per_second=10.0
            )
            engine.metrics_history.append(metrics)
        
        # Predict CPU load
        predicted_cpu = engine._predict_future_load(ResourceType.CPU)
        assert predicted_cpu > 50.0  # Should predict increase
    
    def test_connection_pool_management(self, scaling_engine_instance):
        """Test connection pool management."""
        engine = scaling_engine_instance
        
        # Get connection pool
        pool = engine.get_connection_pool("test_pool")
        assert isinstance(pool, AdvancedConnectionPool)
        assert "test_pool" in engine.connection_pools
        
        # Get same pool again
        pool2 = engine.get_connection_pool("test_pool")
        assert pool is pool2  # Should return same instance
    
    def test_pool_utilization_calculation(self, scaling_engine_instance):
        """Test connection pool utilization calculation."""
        engine = scaling_engine_instance
        
        # Create mock pools
        pool1 = Mock()
        pool1.get_pool_stats.return_value = {
            'in_use_connections': 5,
            'max_connections': 10
        }
        
        pool2 = Mock()
        pool2.get_pool_stats.return_value = {
            'in_use_connections': 8,
            'max_connections': 10
        }
        
        engine.connection_pools = {
            'pool1': pool1,
            'pool2': pool2
        }
        
        utilization = engine._calculate_pool_utilization()
        expected = (0.5 + 0.8) / 2  # Average of 50% and 80%
        assert utilization == expected
    
    @pytest.mark.asyncio
    async def test_monitoring_lifecycle(self, scaling_engine_instance):
        """Test monitoring start and stop."""
        engine = scaling_engine_instance
        
        assert not engine._monitoring_active
        
        # Start monitoring
        await engine.start_monitoring()
        assert engine._monitoring_active
        assert engine._monitoring_task is not None
        
        # Wait a brief moment for task to start
        await asyncio.sleep(0.1)
        
        # Stop monitoring
        await engine.stop_monitoring()
        assert not engine._monitoring_active
    
    def test_scaling_analytics(self, scaling_engine_instance):
        """Test scaling analytics generation."""
        engine = scaling_engine_instance
        
        # Add some test data
        worker = WorkerNode(
            node_id="test_worker",
            status="active",
            cpu_usage=50.0,
            memory_usage=60.0,
            request_count=100,
            error_rate=0.05
        )
        engine.worker_nodes["test_worker"] = worker
        
        decision = ScalingDecision(
            timestamp=datetime.now(),
            action=ScalingAction.SCALE_UP,
            resource_type=ResourceType.CPU,
            current_value=80.0,
            threshold=70.0,
            confidence=0.8,
            predicted_load=85.0,
            reasoning="Test scaling decision"
        )
        engine.scaling_decisions.append(decision)
        
        analytics = engine.get_scaling_analytics()
        
        assert 'current_workers' in analytics
        assert 'worker_nodes' in analytics
        assert 'avg_metrics' in analytics
        assert 'recent_decisions' in analytics
        assert 'cache_stats' in analytics
        assert analytics['current_workers'] == engine.current_workers
        assert 'test_worker' in analytics['worker_nodes']
        assert analytics['recent_decisions']['scale_up'] == 1


class TestGlobalFunctions:
    """Test global convenience functions."""
    
    @pytest.mark.asyncio
    async def test_get_connection_pool(self):
        """Test global get_connection_pool function."""
        pool = await get_connection_pool("test_global_pool")
        assert isinstance(pool, AdvancedConnectionPool)
    
    def test_get_intelligent_cache(self):
        """Test global get_intelligent_cache function."""
        cache = get_intelligent_cache()
        assert isinstance(cache, IntelligentCache)


class TestIntegration:
    """Integration tests for scaling engine components."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_scaling_scenario(self):
        """Test complete scaling scenario."""
        engine = AdvancedScalingEngine()
        
        # Simulate high CPU usage scenario
        with patch('psutil.cpu_percent', return_value=85.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_io_counters', return_value=None), \
             patch('psutil.net_io_counters', return_value=None):
            
            mock_memory.return_value.percent = 60.0
            
            # Collect metrics
            metrics = await engine._collect_metrics()
            assert metrics.cpu_percent == 85.0
            
            # Make scaling decision
            decision = await engine._analyze_and_decide(metrics)
            assert decision.action == ScalingAction.SCALE_UP
            
            # Execute scaling decision
            initial_workers = engine.current_workers
            await engine._execute_scaling_decision(decision)
            assert engine.current_workers == initial_workers + 1
    
    @pytest.mark.asyncio
    async def test_cache_integration_with_scaling(self):
        """Test cache integration with scaling decisions."""
        engine = AdvancedScalingEngine()
        cache = engine.intelligent_cache
        
        # Set some cache data
        cache.set("test_key", "test_value")
        
        # Get cache stats
        stats = cache.get_cache_stats()
        assert stats['cache_size'] == 1
        
        # Get scaling analytics (includes cache stats)
        analytics = engine.get_scaling_analytics()
        assert 'cache_stats' in analytics
        assert analytics['cache_stats']['cache_size'] == 1
    
    @pytest.mark.asyncio
    async def test_performance_optimization_cycle(self):
        """Test performance optimization cycle."""
        engine = AdvancedScalingEngine()
        
        # Add some metrics history
        metrics = ResourceMetrics(
            timestamp=datetime.now(),
            cpu_percent=50.0,
            memory_percent=60.0,
            disk_io_read=0,
            disk_io_write=0,
            network_bytes_sent=0,
            network_bytes_recv=0,
            active_connections=0,
            cache_hit_rate=0.5,  # Low hit rate should trigger optimization
            avg_response_time=1.0,
            requests_per_second=10.0
        )
        engine.metrics_history.append(metrics)
        
        # Run optimization
        await engine._optimize_performance()
        
        # Should have recorded optimization
        assert len(engine.optimization_history) > 0
        
        optimization = engine.optimization_history[-1]
        assert 'cache_hit_rate' in optimization
        assert 'timestamp' in optimization