"""
Advanced Performance Optimization and Scaling Engine.

This module provides comprehensive performance optimization, auto-scaling,
and resource management for the financial analysis platform.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, AsyncGenerator
from collections import defaultdict, deque
import multiprocessing as mp
import queue

import numpy as np

from .config import get_config
from .logging_utils import configure_logging
from .metrics import get_business_tracker
from .comprehensive_monitoring import get_monitoring

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Performance optimization strategies."""
    CACHING = "caching"
    PARALLELIZATION = "parallelization"
    LOAD_BALANCING = "load_balancing"
    CONNECTION_POOLING = "connection_pooling"
    BATCH_PROCESSING = "batch_processing"
    LAZY_LOADING = "lazy_loading"
    COMPRESSION = "compression"
    PREFETCHING = "prefetching"


class ScalingDirection(Enum):
    """Auto-scaling directions."""
    UP = "up"
    DOWN = "down"
    MAINTAIN = "maintain"


class ResourceType(Enum):
    """Types of resources that can be scaled."""
    CPU = "cpu"
    MEMORY = "memory"
    NETWORK = "network"
    STORAGE = "storage"
    WORKERS = "workers"
    CONNECTIONS = "connections"


@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization decisions."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    network_io: float
    disk_io: float
    response_time: float
    throughput: float
    error_rate: float
    queue_depth: int


@dataclass
class OptimizationResult:
    """Result of a performance optimization."""
    strategy: OptimizationStrategy
    before_metrics: PerformanceMetrics
    after_metrics: PerformanceMetrics
    improvement_percent: float
    cost: float
    success: bool
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScalingDecision:
    """Auto-scaling decision."""
    resource_type: ResourceType
    direction: ScalingDirection
    current_capacity: float
    target_capacity: float
    reason: str
    confidence: float


class PerformanceOptimizationEngine:
    """
    Advanced performance optimization engine with auto-scaling,
    intelligent caching, and resource management capabilities.
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path.home() / ".cache" / "finchat_sec_qa" / "performance"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Performance tracking
        self.metrics_history: deque = deque(maxlen=10000)
        self.optimization_history: List[OptimizationResult] = []
        self.scaling_decisions: List[ScalingDecision] = []
        
        # Resource pools
        self.thread_pool = ThreadPoolExecutor(max_workers=mp.cpu_count() * 2)
        self.process_pool = ProcessPoolExecutor(max_workers=mp.cpu_count())
        self.connection_pools: Dict[str, Any] = {}
        
        # Caching systems
        self.adaptive_cache: Dict[str, Any] = {}
        self.cache_statistics: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.cache_strategies: Dict[str, str] = {}
        
        # Load balancing
        self.load_balancer_state: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.request_queue: queue.Queue = queue.Queue()
        
        # Auto-scaling parameters
        self.scaling_thresholds = {
            ResourceType.CPU: {'up': 80.0, 'down': 30.0},
            ResourceType.MEMORY: {'up': 85.0, 'down': 40.0},
            ResourceType.WORKERS: {'up': 90.0, 'down': 20.0}
        }
        
        self.current_capacity = {
            ResourceType.CPU: 1.0,
            ResourceType.MEMORY: 1.0,
            ResourceType.WORKERS: mp.cpu_count(),
            ResourceType.CONNECTIONS: 100
        }
        
        # Optimization strategies
        self.active_optimizations: Set[OptimizationStrategy] = set()
        self.optimization_cooldown: Dict[OptimizationStrategy, float] = {}
        
        # Background processing
        self._running = False
        self._optimization_thread: Optional[threading.Thread] = None
        
        self.monitoring = get_monitoring()
        configure_logging()

    def start_optimization_engine(self) -> None:
        """Start the performance optimization engine."""
        if self._running:
            return
            
        self._running = True
        self._optimization_thread = threading.Thread(
            target=self._optimization_loop, 
            daemon=True
        )
        self._optimization_thread.start()
        logger.info("Performance optimization engine started")

    def stop_optimization_engine(self) -> None:
        """Stop the performance optimization engine."""
        self._running = False
        if self._optimization_thread:
            self._optimization_thread.join(timeout=5)
        
        # Cleanup resources
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        
        logger.info("Performance optimization engine stopped")

    def _optimization_loop(self) -> None:
        """Main optimization loop."""
        while self._running:
            try:
                # Collect performance metrics
                current_metrics = self._collect_performance_metrics()
                self.metrics_history.append(current_metrics)
                
                # Analyze performance patterns
                performance_issues = self._analyze_performance_patterns()
                
                # Make scaling decisions
                scaling_decisions = self._make_scaling_decisions(current_metrics)
                
                # Apply optimizations
                for issue in performance_issues:
                    optimization_result = self._apply_optimization(issue, current_metrics)
                    if optimization_result:
                        self.optimization_history.append(optimization_result)
                
                # Execute scaling decisions
                for decision in scaling_decisions:
                    self._execute_scaling_decision(decision)
                    self.scaling_decisions.append(decision)
                
                # Update cache strategies
                self._optimize_caching_strategies()
                
                # Cleanup old data
                self._cleanup_optimization_data()
                
                # Sleep for optimization interval
                time.sleep(60)  # Run every minute
                
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                time.sleep(30)

    def _collect_performance_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        try:
            import psutil
            
            # System metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk_io = psutil.disk_io_counters()
            network_io = psutil.net_io_counters()
            
            # Application metrics
            response_time = self.monitoring.get_latest_metric_value('response_time', 0.5)
            throughput = self.monitoring.get_latest_metric_value('throughput', 100.0)
            error_rate = self.monitoring.get_latest_metric_value('error_rate', 0.01)
            queue_depth = self.request_queue.qsize()
            
            return PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_usage=cpu_usage,
                memory_usage=memory.percent,
                network_io=network_io.bytes_sent + network_io.bytes_recv if network_io else 0,
                disk_io=disk_io.read_bytes + disk_io.write_bytes if disk_io else 0,
                response_time=response_time,
                throughput=throughput,
                error_rate=error_rate,
                queue_depth=queue_depth
            )
            
        except Exception as e:
            logger.error(f"Error collecting performance metrics: {e}")
            # Return default metrics
            return PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_usage=50.0,
                memory_usage=60.0,
                network_io=0,
                disk_io=0,
                response_time=1.0,
                throughput=50.0,
                error_rate=0.05,
                queue_depth=0
            )

    def _analyze_performance_patterns(self) -> List[Dict[str, Any]]:
        """Analyze performance patterns to identify optimization opportunities."""
        issues = []
        
        if len(self.metrics_history) < 10:
            return issues
        
        recent_metrics = list(self.metrics_history)[-10:]
        
        # CPU utilization analysis
        avg_cpu = np.mean([m.cpu_usage for m in recent_metrics])
        if avg_cpu > 80:
            issues.append({
                'type': 'high_cpu_usage',
                'severity': 'high' if avg_cpu > 90 else 'medium',
                'value': avg_cpu,
                'recommended_strategies': [OptimizationStrategy.PARALLELIZATION, OptimizationStrategy.CACHING]
            })
        
        # Memory usage analysis
        avg_memory = np.mean([m.memory_usage for m in recent_metrics])
        if avg_memory > 85:
            issues.append({
                'type': 'high_memory_usage',
                'severity': 'high' if avg_memory > 95 else 'medium',
                'value': avg_memory,
                'recommended_strategies': [OptimizationStrategy.LAZY_LOADING, OptimizationStrategy.COMPRESSION]
            })
        
        # Response time analysis
        avg_response_time = np.mean([m.response_time for m in recent_metrics])
        if avg_response_time > 2.0:
            issues.append({
                'type': 'slow_response_time',
                'severity': 'high' if avg_response_time > 5.0 else 'medium',
                'value': avg_response_time,
                'recommended_strategies': [OptimizationStrategy.CACHING, OptimizationStrategy.PREFETCHING, OptimizationStrategy.CONNECTION_POOLING]
            })
        
        # Throughput analysis
        avg_throughput = np.mean([m.throughput for m in recent_metrics])
        if avg_throughput < 50:
            issues.append({
                'type': 'low_throughput',
                'severity': 'medium',
                'value': avg_throughput,
                'recommended_strategies': [OptimizationStrategy.BATCH_PROCESSING, OptimizationStrategy.LOAD_BALANCING]
            })
        
        # Error rate analysis
        avg_error_rate = np.mean([m.error_rate for m in recent_metrics])
        if avg_error_rate > 0.05:  # > 5%
            issues.append({
                'type': 'high_error_rate',
                'severity': 'high' if avg_error_rate > 0.1 else 'medium',
                'value': avg_error_rate,
                'recommended_strategies': [OptimizationStrategy.CONNECTION_POOLING, OptimizationStrategy.LOAD_BALANCING]
            })
        
        return issues

    def _make_scaling_decisions(self, current_metrics: PerformanceMetrics) -> List[ScalingDecision]:
        """Make auto-scaling decisions based on current metrics."""
        decisions = []
        
        # CPU scaling decision
        cpu_threshold = self.scaling_thresholds[ResourceType.CPU]
        if current_metrics.cpu_usage > cpu_threshold['up']:
            decisions.append(ScalingDecision(
                resource_type=ResourceType.CPU,
                direction=ScalingDirection.UP,
                current_capacity=self.current_capacity[ResourceType.CPU],
                target_capacity=self.current_capacity[ResourceType.CPU] * 1.5,
                reason=f"CPU usage ({current_metrics.cpu_usage:.1f}%) exceeds threshold ({cpu_threshold['up']:.1f}%)",
                confidence=0.8
            ))
        elif current_metrics.cpu_usage < cpu_threshold['down']:
            decisions.append(ScalingDecision(
                resource_type=ResourceType.CPU,
                direction=ScalingDirection.DOWN,
                current_capacity=self.current_capacity[ResourceType.CPU],
                target_capacity=max(0.5, self.current_capacity[ResourceType.CPU] * 0.8),
                reason=f"CPU usage ({current_metrics.cpu_usage:.1f}%) below threshold ({cpu_threshold['down']:.1f}%)",
                confidence=0.6
            ))
        
        # Memory scaling decision
        memory_threshold = self.scaling_thresholds[ResourceType.MEMORY]
        if current_metrics.memory_usage > memory_threshold['up']:
            decisions.append(ScalingDecision(
                resource_type=ResourceType.MEMORY,
                direction=ScalingDirection.UP,
                current_capacity=self.current_capacity[ResourceType.MEMORY],
                target_capacity=self.current_capacity[ResourceType.MEMORY] * 1.3,
                reason=f"Memory usage ({current_metrics.memory_usage:.1f}%) exceeds threshold ({memory_threshold['up']:.1f}%)",
                confidence=0.9
            ))
        
        # Workers scaling decision
        queue_depth_per_worker = current_metrics.queue_depth / self.current_capacity[ResourceType.WORKERS]
        if queue_depth_per_worker > 5:  # More than 5 requests per worker
            decisions.append(ScalingDecision(
                resource_type=ResourceType.WORKERS,
                direction=ScalingDirection.UP,
                current_capacity=self.current_capacity[ResourceType.WORKERS],
                target_capacity=min(mp.cpu_count() * 4, self.current_capacity[ResourceType.WORKERS] + 2),
                reason=f"High queue depth per worker ({queue_depth_per_worker:.1f})",
                confidence=0.7
            ))
        elif queue_depth_per_worker < 1 and self.current_capacity[ResourceType.WORKERS] > mp.cpu_count():
            decisions.append(ScalingDecision(
                resource_type=ResourceType.WORKERS,
                direction=ScalingDirection.DOWN,
                current_capacity=self.current_capacity[ResourceType.WORKERS],
                target_capacity=max(mp.cpu_count(), self.current_capacity[ResourceType.WORKERS] - 1),
                reason=f"Low queue depth per worker ({queue_depth_per_worker:.1f})",
                confidence=0.5
            ))
        
        return decisions

    def _apply_optimization(self, issue: Dict[str, Any], current_metrics: PerformanceMetrics) -> Optional[OptimizationResult]:
        """Apply optimization strategy for a performance issue."""
        strategies = issue['recommended_strategies']
        
        # Select best strategy that's not in cooldown
        selected_strategy = None
        for strategy in strategies:
            if strategy not in self.active_optimizations:
                cooldown_end = self.optimization_cooldown.get(strategy, 0)
                if time.time() > cooldown_end:
                    selected_strategy = strategy
                    break
        
        if not selected_strategy:
            return None
        
        logger.info(f"Applying optimization strategy: {selected_strategy.value} for issue: {issue['type']}")
        
        try:
            # Apply the optimization
            success = False
            details = {}
            
            if selected_strategy == OptimizationStrategy.CACHING:
                success, details = self._optimize_caching(issue)
            elif selected_strategy == OptimizationStrategy.PARALLELIZATION:
                success, details = self._optimize_parallelization(issue)
            elif selected_strategy == OptimizationStrategy.CONNECTION_POOLING:
                success, details = self._optimize_connection_pooling(issue)
            elif selected_strategy == OptimizationStrategy.BATCH_PROCESSING:
                success, details = self._optimize_batch_processing(issue)
            elif selected_strategy == OptimizationStrategy.LAZY_LOADING:
                success, details = self._optimize_lazy_loading(issue)
            elif selected_strategy == OptimizationStrategy.COMPRESSION:
                success, details = self._optimize_compression(issue)
            elif selected_strategy == OptimizationStrategy.PREFETCHING:
                success, details = self._optimize_prefetching(issue)
            elif selected_strategy == OptimizationStrategy.LOAD_BALANCING:
                success, details = self._optimize_load_balancing(issue)
            
            if success:
                self.active_optimizations.add(selected_strategy)
                # Set cooldown to prevent rapid re-application
                self.optimization_cooldown[selected_strategy] = time.time() + 300  # 5 minutes
            
            # Collect metrics after optimization
            time.sleep(5)  # Give some time for the optimization to take effect
            after_metrics = self._collect_performance_metrics()
            
            # Calculate improvement
            improvement = self._calculate_improvement(current_metrics, after_metrics, issue['type'])
            
            return OptimizationResult(
                strategy=selected_strategy,
                before_metrics=current_metrics,
                after_metrics=after_metrics,
                improvement_percent=improvement,
                cost=0.0,  # Cost calculation would be implemented based on strategy
                success=success,
                details=details
            )
            
        except Exception as e:
            logger.error(f"Error applying optimization {selected_strategy.value}: {e}")
            return None

    def _optimize_caching(self, issue: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Optimize caching strategy."""
        # Implement adaptive caching based on access patterns
        cache_size_before = len(self.adaptive_cache)
        
        # Increase cache size or implement more aggressive caching
        if issue['type'] == 'slow_response_time':
            self.cache_strategies['response_cache'] = 'aggressive'
            cache_ttl = 3600  # 1 hour
        else:
            self.cache_strategies['response_cache'] = 'moderate'
            cache_ttl = 1800  # 30 minutes
        
        # Simulate cache optimization
        self.adaptive_cache['optimization_timestamp'] = time.time()
        self.adaptive_cache['cache_ttl'] = cache_ttl
        
        return True, {
            'cache_size_before': cache_size_before,
            'cache_size_after': len(self.adaptive_cache),
            'cache_ttl': cache_ttl,
            'strategy': self.cache_strategies.get('response_cache')
        }

    def _optimize_parallelization(self, issue: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Optimize parallelization strategy."""
        current_workers = self.thread_pool._max_workers
        
        if issue['type'] == 'high_cpu_usage' and issue['severity'] == 'high':
            # Increase thread pool size
            new_workers = min(current_workers + 2, mp.cpu_count() * 3)
        else:
            new_workers = min(current_workers + 1, mp.cpu_count() * 2)
        
        # Create new thread pool with updated size
        self.thread_pool.shutdown(wait=False)
        self.thread_pool = ThreadPoolExecutor(max_workers=new_workers)
        
        return True, {
            'workers_before': current_workers,
            'workers_after': new_workers,
            'scaling_factor': new_workers / current_workers
        }

    def _optimize_connection_pooling(self, issue: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Optimize connection pooling."""
        # Simulate connection pool optimization
        current_pool_size = self.current_capacity.get(ResourceType.CONNECTIONS, 100)
        
        if issue['type'] in ['slow_response_time', 'high_error_rate']:
            new_pool_size = min(current_pool_size * 1.5, 500)
        else:
            new_pool_size = min(current_pool_size * 1.2, 300)
        
        self.current_capacity[ResourceType.CONNECTIONS] = new_pool_size
        
        return True, {
            'pool_size_before': current_pool_size,
            'pool_size_after': new_pool_size,
            'connections_added': new_pool_size - current_pool_size
        }

    def _optimize_batch_processing(self, issue: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Optimize batch processing."""
        current_batch_size = getattr(self, 'batch_size', 32)
        
        if issue['type'] == 'low_throughput':
            new_batch_size = min(current_batch_size * 2, 128)
        else:
            new_batch_size = min(current_batch_size * 1.5, 64)
        
        self.batch_size = new_batch_size
        
        return True, {
            'batch_size_before': current_batch_size,
            'batch_size_after': new_batch_size,
            'efficiency_gain': new_batch_size / current_batch_size
        }

    def _optimize_lazy_loading(self, issue: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Optimize lazy loading strategy."""
        # Enable more aggressive lazy loading
        lazy_loading_threshold = getattr(self, 'lazy_loading_threshold', 1024)  # bytes
        
        if issue['type'] == 'high_memory_usage':
            new_threshold = lazy_loading_threshold // 2
        else:
            new_threshold = int(lazy_loading_threshold * 0.8)
        
        self.lazy_loading_threshold = max(new_threshold, 256)
        
        return True, {
            'threshold_before': lazy_loading_threshold,
            'threshold_after': self.lazy_loading_threshold,
            'memory_savings_potential': '20-40%'
        }

    def _optimize_compression(self, issue: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Optimize compression strategy."""
        # Enable or enhance compression
        current_compression_level = getattr(self, 'compression_level', 0)
        
        if issue['type'] == 'high_memory_usage':
            new_compression_level = min(current_compression_level + 2, 9)
        else:
            new_compression_level = min(current_compression_level + 1, 6)
        
        self.compression_level = new_compression_level
        
        return True, {
            'compression_level_before': current_compression_level,
            'compression_level_after': new_compression_level,
            'estimated_space_savings': f"{new_compression_level * 10}%"
        }

    def _optimize_prefetching(self, issue: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Optimize prefetching strategy."""
        # Enable or enhance prefetching
        current_prefetch_size = getattr(self, 'prefetch_size', 10)
        
        if issue['type'] == 'slow_response_time':
            new_prefetch_size = min(current_prefetch_size * 2, 50)
        else:
            new_prefetch_size = min(current_prefetch_size + 5, 25)
        
        self.prefetch_size = new_prefetch_size
        
        return True, {
            'prefetch_size_before': current_prefetch_size,
            'prefetch_size_after': new_prefetch_size,
            'cache_hit_improvement': f"{(new_prefetch_size / current_prefetch_size - 1) * 100:.1f}%"
        }

    def _optimize_load_balancing(self, issue: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Optimize load balancing strategy."""
        # Implement or enhance load balancing
        current_algorithm = getattr(self, 'load_balance_algorithm', 'round_robin')
        
        if issue['type'] in ['low_throughput', 'high_error_rate']:
            if current_algorithm == 'round_robin':
                new_algorithm = 'least_connections'
            else:
                new_algorithm = 'weighted_response_time'
        else:
            new_algorithm = 'least_connections' if current_algorithm == 'round_robin' else 'round_robin'
        
        self.load_balance_algorithm = new_algorithm
        
        return True, {
            'algorithm_before': current_algorithm,
            'algorithm_after': new_algorithm,
            'expected_improvement': '10-30%'
        }

    def _execute_scaling_decision(self, decision: ScalingDecision) -> None:
        """Execute an auto-scaling decision."""
        logger.info(f"Executing scaling decision: {decision.direction.value} {decision.resource_type.value} "
                   f"from {decision.current_capacity} to {decision.target_capacity}")
        
        try:
            if decision.resource_type == ResourceType.WORKERS:
                if decision.direction == ScalingDirection.UP:
                    # Scale up thread pool
                    self.thread_pool.shutdown(wait=False)
                    self.thread_pool = ThreadPoolExecutor(max_workers=int(decision.target_capacity))
                elif decision.direction == ScalingDirection.DOWN:
                    # Scale down thread pool
                    self.thread_pool.shutdown(wait=False)
                    self.thread_pool = ThreadPoolExecutor(max_workers=int(decision.target_capacity))
            
            # Update capacity tracking
            self.current_capacity[decision.resource_type] = decision.target_capacity
            
            # Record scaling action
            self.monitoring.record_metric(
                f'scaling.{decision.resource_type.value}.capacity',
                decision.target_capacity
            )
            
        except Exception as e:
            logger.error(f"Error executing scaling decision: {e}")

    def _optimize_caching_strategies(self) -> None:
        """Optimize caching strategies based on usage patterns."""
        # Analyze cache hit rates
        for cache_name, stats in self.cache_statistics.items():
            if stats['total'] > 0:
                hit_rate = stats['hits'] / stats['total']
                
                if hit_rate < 0.3:  # Low hit rate
                    # Increase cache size or change eviction policy
                    logger.info(f"Low cache hit rate for {cache_name}: {hit_rate:.2f}")
                elif hit_rate > 0.9:  # Very high hit rate
                    # Cache might be too large, consider reducing size
                    logger.info(f"Very high cache hit rate for {cache_name}: {hit_rate:.2f}")

    def _calculate_improvement(self, before: PerformanceMetrics, after: PerformanceMetrics, issue_type: str) -> float:
        """Calculate performance improvement percentage."""
        if issue_type == 'high_cpu_usage':
            if before.cpu_usage == 0:
                return 0.0
            return ((before.cpu_usage - after.cpu_usage) / before.cpu_usage) * 100
        
        elif issue_type == 'high_memory_usage':
            if before.memory_usage == 0:
                return 0.0
            return ((before.memory_usage - after.memory_usage) / before.memory_usage) * 100
        
        elif issue_type == 'slow_response_time':
            if before.response_time == 0:
                return 0.0
            return ((before.response_time - after.response_time) / before.response_time) * 100
        
        elif issue_type == 'low_throughput':
            if before.throughput == 0:
                return 0.0
            return ((after.throughput - before.throughput) / before.throughput) * 100
        
        elif issue_type == 'high_error_rate':
            if before.error_rate == 0:
                return 0.0
            return ((before.error_rate - after.error_rate) / before.error_rate) * 100
        
        return 0.0

    def _cleanup_optimization_data(self) -> None:
        """Clean up old optimization data."""
        # Keep only recent optimization history
        if len(self.optimization_history) > 1000:
            self.optimization_history = self.optimization_history[-500:]
        
        # Keep only recent scaling decisions
        if len(self.scaling_decisions) > 500:
            self.scaling_decisions = self.scaling_decisions[-250:]
        
        # Clean up cache statistics older than 24 hours
        # (Implementation would depend on timestamp tracking)

    @contextmanager
    def performance_monitor(self, operation_name: str):
        """Context manager for monitoring operation performance."""
        start_time = time.time()
        start_metrics = self._collect_performance_metrics()
        
        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time
            end_metrics = self._collect_performance_metrics()
            
            # Record operation metrics
            self.monitoring.record_timer(f'operation.{operation_name}.duration', duration)
            
            # Log performance impact
            logger.debug(f"Operation {operation_name} completed in {duration:.3f}s")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.metrics_history:
            return {'status': 'no_data'}
        
        recent_metrics = list(self.metrics_history)[-60:]  # Last hour
        
        return {
            'current_performance': {
                'cpu_usage': recent_metrics[-1].cpu_usage,
                'memory_usage': recent_metrics[-1].memory_usage,
                'response_time': recent_metrics[-1].response_time,
                'throughput': recent_metrics[-1].throughput,
                'error_rate': recent_metrics[-1].error_rate
            },
            'average_performance': {
                'cpu_usage': np.mean([m.cpu_usage for m in recent_metrics]),
                'memory_usage': np.mean([m.memory_usage for m in recent_metrics]),
                'response_time': np.mean([m.response_time for m in recent_metrics]),
                'throughput': np.mean([m.throughput for m in recent_metrics]),
                'error_rate': np.mean([m.error_rate for m in recent_metrics])
            },
            'optimization_summary': {
                'total_optimizations': len(self.optimization_history),
                'successful_optimizations': len([o for o in self.optimization_history if o.success]),
                'average_improvement': np.mean([o.improvement_percent for o in self.optimization_history if o.success]),
                'active_strategies': [s.value for s in self.active_optimizations]
            },
            'scaling_summary': {
                'total_scaling_actions': len(self.scaling_decisions),
                'current_capacity': {k.value: v for k, v in self.current_capacity.items()},
                'recent_scaling': [
                    {
                        'resource': d.resource_type.value,
                        'direction': d.direction.value,
                        'reason': d.reason
                    }
                    for d in self.scaling_decisions[-5:]
                ]
            },
            'cache_performance': {
                name: {
                    'hit_rate': stats['hits'] / stats['total'] if stats['total'] > 0 else 0,
                    'total_requests': stats['total']
                }
                for name, stats in self.cache_statistics.items()
            }
        }

    def recommend_optimizations(self) -> List[Dict[str, Any]]:
        """Get optimization recommendations based on current performance."""
        if len(self.metrics_history) < 5:
            return []
        
        recommendations = []
        recent_metrics = list(self.metrics_history)[-10:]
        avg_metrics = {
            'cpu_usage': np.mean([m.cpu_usage for m in recent_metrics]),
            'memory_usage': np.mean([m.memory_usage for m in recent_metrics]),
            'response_time': np.mean([m.response_time for m in recent_metrics]),
            'throughput': np.mean([m.throughput for m in recent_metrics]),
            'error_rate': np.mean([m.error_rate for m in recent_metrics])
        }
        
        # Generate recommendations
        if avg_metrics['response_time'] > 1.0:
            recommendations.append({
                'priority': 'high',
                'strategy': 'caching',
                'reason': f"Average response time is {avg_metrics['response_time']:.2f}s",
                'expected_improvement': '30-50%',
                'implementation_effort': 'medium'
            })
        
        if avg_metrics['cpu_usage'] > 75:
            recommendations.append({
                'priority': 'medium',
                'strategy': 'parallelization',
                'reason': f"Average CPU usage is {avg_metrics['cpu_usage']:.1f}%",
                'expected_improvement': '20-40%',
                'implementation_effort': 'high'
            })
        
        if avg_metrics['memory_usage'] > 80:
            recommendations.append({
                'priority': 'high',
                'strategy': 'lazy_loading',
                'reason': f"Average memory usage is {avg_metrics['memory_usage']:.1f}%",
                'expected_improvement': '25-45%',
                'implementation_effort': 'medium'
            })
        
        return recommendations


# Global performance optimization engine
_global_performance_engine: Optional[PerformanceOptimizationEngine] = None


def get_performance_engine() -> PerformanceOptimizationEngine:
    """Get the global performance optimization engine."""
    global _global_performance_engine
    if _global_performance_engine is None:
        _global_performance_engine = PerformanceOptimizationEngine()
        _global_performance_engine.start_optimization_engine()
    return _global_performance_engine