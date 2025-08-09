"""Adaptive auto-scaling and resource management for optimal performance."""
import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from .metrics import (
    system_memory_usage_bytes,
    system_cpu_usage_percent,
    get_adaptive_collector,
    alert_threshold_breaches_total
)

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of resources that can be scaled."""
    THREAD_POOL = "thread_pool"
    PROCESS_POOL = "process_pool"
    CONNECTION_POOL = "connection_pool"
    CACHE_SIZE = "cache_size"
    WORKER_PROCESSES = "worker_processes"


class ScalingTrigger(Enum):
    """Triggers for scaling decisions."""
    CPU_THRESHOLD = "cpu_threshold"
    MEMORY_THRESHOLD = "memory_threshold"
    QUEUE_LENGTH = "queue_length"
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"
    CUSTOM_METRIC = "custom_metric"


@dataclass
class ScalingRule:
    """Configuration for a scaling rule."""
    resource_type: ResourceType
    trigger: ScalingTrigger
    scale_up_threshold: float
    scale_down_threshold: float
    min_instances: int
    max_instances: int
    scale_up_increment: int = 1
    scale_down_increment: int = 1
    cooldown_seconds: float = 300  # 5 minutes
    evaluation_period_seconds: float = 60  # 1 minute
    custom_metric_name: Optional[str] = None
    enabled: bool = True


@dataclass
class ResourcePool:
    """Managed resource pool with auto-scaling."""
    name: str
    resource_type: ResourceType
    current_size: int
    min_size: int
    max_size: int
    executor: Optional[Any] = None
    last_scaled: float = field(default_factory=time.time)
    scaling_history: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


class AutoScaler:
    """Adaptive auto-scaling system with intelligent resource management."""
    
    def __init__(self):
        self.pools: Dict[str, ResourcePool] = {}
        self.scaling_rules: Dict[str, ScalingRule] = {}
        self.metrics_history: List[Dict[str, Any]] = []
        self.max_history_size = 1000
        self.is_running = False
        self._lock = asyncio.Lock()
        
        # Performance baselines for adaptive scaling
        self.performance_baselines = {
            'cpu_target': 70.0,
            'memory_target': 80.0,
            'response_time_target': 2.0,
            'error_rate_target': 0.05
        }
        
        # Machine learning-like adaptive parameters
        self.learning_rate = 0.1
        self.prediction_window = 10  # minutes
        
        self._initialize_default_rules()
        logger.info("AutoScaler initialized with adaptive scaling capabilities")
    
    def _initialize_default_rules(self):
        """Initialize default scaling rules."""
        # Thread pool scaling based on CPU
        self.add_scaling_rule("thread_pool_cpu", ScalingRule(
            resource_type=ResourceType.THREAD_POOL,
            trigger=ScalingTrigger.CPU_THRESHOLD,
            scale_up_threshold=75.0,
            scale_down_threshold=40.0,
            min_instances=2,
            max_instances=min(32, (os.cpu_count() or 4) * 2),
            scale_up_increment=2,
            scale_down_increment=1,
            cooldown_seconds=180
        ))
        
        # Process pool scaling based on memory
        self.add_scaling_rule("process_pool_memory", ScalingRule(
            resource_type=ResourceType.PROCESS_POOL,
            trigger=ScalingTrigger.MEMORY_THRESHOLD,
            scale_up_threshold=85.0,
            scale_down_threshold=50.0,
            min_instances=1,
            max_instances=min(8, os.cpu_count() or 4),
            cooldown_seconds=300
        ))
        
        # Cache size scaling based on memory usage
        self.add_scaling_rule("cache_memory", ScalingRule(
            resource_type=ResourceType.CACHE_SIZE,
            trigger=ScalingTrigger.MEMORY_THRESHOLD,
            scale_up_threshold=60.0,
            scale_down_threshold=40.0,
            min_instances=100,  # MB
            max_instances=2000,  # MB
            scale_up_increment=100,
            scale_down_increment=50,
            cooldown_seconds=120
        ))
    
    def add_scaling_rule(self, name: str, rule: ScalingRule):
        """Add a new scaling rule."""
        self.scaling_rules[name] = rule
        logger.info(f"Added scaling rule '{name}': {rule.resource_type.value} "
                   f"triggered by {rule.trigger.value}")
    
    def create_resource_pool(self, name: str, resource_type: ResourceType, 
                           initial_size: int, min_size: int, max_size: int) -> ResourcePool:
        """Create a new managed resource pool."""
        pool = ResourcePool(
            name=name,
            resource_type=resource_type,
            current_size=initial_size,
            min_size=min_size,
            max_size=max_size
        )
        
        # Initialize the actual executor/resource
        if resource_type == ResourceType.THREAD_POOL:
            pool.executor = ThreadPoolExecutor(max_workers=initial_size, thread_name_prefix=f"{name}_")
        elif resource_type == ResourceType.PROCESS_POOL:
            pool.executor = ProcessPoolExecutor(max_workers=initial_size)
        
        self.pools[name] = pool
        logger.info(f"Created resource pool '{name}' ({resource_type.value}) "
                   f"with {initial_size} initial resources")
        return pool
    
    async def start_auto_scaling(self):
        """Start the auto-scaling monitoring loop."""
        if self.is_running:
            logger.warning("AutoScaler is already running")
            return
        
        self.is_running = True
        logger.info("Starting auto-scaling monitoring")
        
        try:
            while self.is_running:
                await self._scaling_evaluation_cycle()
                await asyncio.sleep(30)  # Evaluate every 30 seconds
        except Exception as e:
            logger.error(f"Auto-scaling monitoring error: {e}")
        finally:
            self.is_running = False
    
    async def stop_auto_scaling(self):
        """Stop the auto-scaling monitoring."""
        self.is_running = False
        logger.info("Stopped auto-scaling monitoring")
    
    async def _scaling_evaluation_cycle(self):
        """Perform one cycle of scaling evaluation."""
        try:
            current_metrics = await self._collect_current_metrics()
            
            # Store metrics history for trend analysis
            self.metrics_history.append({
                'timestamp': time.time(),
                **current_metrics
            })
            
            # Trim history
            if len(self.metrics_history) > self.max_history_size:
                self.metrics_history = self.metrics_history[-self.max_history_size:]
            
            # Evaluate each scaling rule
            for rule_name, rule in self.scaling_rules.items():
                if not rule.enabled:
                    continue
                
                try:
                    await self._evaluate_scaling_rule(rule_name, rule, current_metrics)
                except Exception as e:
                    logger.error(f"Error evaluating scaling rule '{rule_name}': {e}")
            
            # Adaptive baseline adjustment
            self._adjust_performance_baselines(current_metrics)
            
        except Exception as e:
            logger.error(f"Scaling evaluation cycle error: {e}")
    
    async def _collect_current_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics for scaling decisions."""
        metrics = {}
        
        try:
            import psutil
            
            # System metrics
            memory = psutil.virtual_memory()
            metrics['memory_percent'] = memory.percent
            metrics['memory_available_gb'] = memory.available / (1024**3)
            
            cpu_percent = psutil.cpu_percent(interval=0.1)
            metrics['cpu_percent'] = cpu_percent
            
            # Load average (Unix only)
            if hasattr(os, 'getloadavg'):
                load_avg = os.getloadavg()
                metrics['load_avg_1min'] = load_avg[0]
                metrics['load_avg_normalized'] = load_avg[0] / (os.cpu_count() or 1)
            
            # Pool metrics
            for pool_name, pool in self.pools.items():
                if pool.executor and hasattr(pool.executor, '_threads'):
                    # Thread pool metrics
                    metrics[f'{pool_name}_active_threads'] = len(pool.executor._threads)
                    metrics[f'{pool_name}_queue_size'] = pool.executor._work_queue.qsize()
                elif pool.executor and hasattr(pool.executor, '_processes'):
                    # Process pool metrics
                    metrics[f'{pool_name}_active_processes'] = len([p for p in pool.executor._processes.values() if p.is_alive()])
            
        except ImportError:
            logger.warning("psutil not available for detailed system metrics")
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
        
        return metrics
    
    async def _evaluate_scaling_rule(self, rule_name: str, rule: ScalingRule, 
                                   current_metrics: Dict[str, Any]):
        """Evaluate a single scaling rule and take action if needed."""
        # Get the metric value for this rule's trigger
        metric_value = self._get_metric_for_trigger(rule.trigger, current_metrics, rule)
        
        if metric_value is None:
            return
        
        # Find relevant pools for this rule
        relevant_pools = [pool for pool in self.pools.values() 
                         if pool.resource_type == rule.resource_type]
        
        if not relevant_pools:
            return
        
        # Check if we're in cooldown period
        current_time = time.time()
        for pool in relevant_pools:
            if current_time - pool.last_scaled < rule.cooldown_seconds:
                return
        
        # Predictive scaling based on trends
        predicted_value = self._predict_metric_trend(rule.trigger, metric_value)
        decision_value = max(metric_value, predicted_value) if predicted_value else metric_value
        
        # Make scaling decisions
        if decision_value >= rule.scale_up_threshold:
            for pool in relevant_pools:
                await self._scale_up_pool(pool, rule, metric_value, decision_value)
        elif decision_value <= rule.scale_down_threshold:
            for pool in relevant_pools:
                await self._scale_down_pool(pool, rule, metric_value, decision_value)
    
    def _get_metric_for_trigger(self, trigger: ScalingTrigger, 
                              current_metrics: Dict[str, Any], 
                              rule: ScalingRule) -> Optional[float]:
        """Get the metric value for a specific trigger."""
        if trigger == ScalingTrigger.CPU_THRESHOLD:
            return current_metrics.get('cpu_percent')
        elif trigger == ScalingTrigger.MEMORY_THRESHOLD:
            return current_metrics.get('memory_percent')
        elif trigger == ScalingTrigger.QUEUE_LENGTH:
            # Find queue size metrics for this rule's resource type
            for key, value in current_metrics.items():
                if 'queue_size' in key:
                    return value
        elif trigger == ScalingTrigger.CUSTOM_METRIC and rule.custom_metric_name:
            return current_metrics.get(rule.custom_metric_name)
        
        return None
    
    def _predict_metric_trend(self, trigger: ScalingTrigger, current_value: float) -> Optional[float]:
        """Predict future metric value based on historical trends."""
        if len(self.metrics_history) < 5:
            return None
        
        # Get historical values for this trigger
        trigger_key = self._get_trigger_key(trigger)
        if not trigger_key:
            return None
        
        recent_values = []
        for record in self.metrics_history[-10:]:  # Last 10 records
            if trigger_key in record:
                recent_values.append(record[trigger_key])
        
        if len(recent_values) < 3:
            return None
        
        # Simple linear trend prediction
        time_points = list(range(len(recent_values)))
        
        # Calculate slope (simple linear regression)
        n = len(recent_values)
        sum_x = sum(time_points)
        sum_y = sum(recent_values)
        sum_xy = sum(x * y for x, y in zip(time_points, recent_values))
        sum_x2 = sum(x * x for x in time_points)
        
        if n * sum_x2 - sum_x * sum_x == 0:
            return None
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        intercept = (sum_y - slope * sum_x) / n
        
        # Predict value for next time point (conservative: +2 periods ahead)
        predicted_value = slope * (n + 2) + intercept
        
        # Only return prediction if trend is significant
        if abs(slope) > 0.1:  # Minimum meaningful slope
            return max(0, predicted_value)  # Ensure non-negative
        
        return None
    
    def _get_trigger_key(self, trigger: ScalingTrigger) -> Optional[str]:
        """Get the metrics history key for a trigger."""
        if trigger == ScalingTrigger.CPU_THRESHOLD:
            return 'cpu_percent'
        elif trigger == ScalingTrigger.MEMORY_THRESHOLD:
            return 'memory_percent'
        return None
    
    async def _scale_up_pool(self, pool: ResourcePool, rule: ScalingRule, 
                           current_value: float, decision_value: float):
        """Scale up a resource pool."""
        if pool.current_size >= pool.max_size:
            logger.debug(f"Pool '{pool.name}' already at maximum size ({pool.max_size})")
            return
        
        new_size = min(pool.current_size + rule.scale_up_increment, pool.max_size)
        
        async with self._lock:
            old_size = pool.current_size
            
            try:
                # Actually scale the resource
                if pool.resource_type == ResourceType.THREAD_POOL and pool.executor:
                    # For ThreadPoolExecutor, we need to create a new one
                    pool.executor.shutdown(wait=False)
                    pool.executor = ThreadPoolExecutor(
                        max_workers=new_size, 
                        thread_name_prefix=f"{pool.name}_"
                    )
                elif pool.resource_type == ResourceType.PROCESS_POOL and pool.executor:
                    # For ProcessPoolExecutor, similar approach
                    pool.executor.shutdown(wait=False)
                    pool.executor = ProcessPoolExecutor(max_workers=new_size)
                
                pool.current_size = new_size
                pool.last_scaled = time.time()
                
                # Record scaling event
                scaling_event = {
                    'timestamp': time.time(),
                    'action': 'scale_up',
                    'old_size': old_size,
                    'new_size': new_size,
                    'trigger_value': current_value,
                    'decision_value': decision_value,
                    'rule': rule.trigger.value
                }
                pool.scaling_history.append(scaling_event)
                
                # Trim scaling history
                if len(pool.scaling_history) > 100:
                    pool.scaling_history = pool.scaling_history[-100:]
                
                logger.info(f"Scaled UP pool '{pool.name}' from {old_size} to {new_size} "
                           f"(trigger: {rule.trigger.value}={current_value:.1f}, "
                           f"threshold: {rule.scale_up_threshold})")
                
            except Exception as e:
                logger.error(f"Failed to scale up pool '{pool.name}': {e}")
                pool.current_size = old_size  # Revert
    
    async def _scale_down_pool(self, pool: ResourcePool, rule: ScalingRule,
                             current_value: float, decision_value: float):
        """Scale down a resource pool."""
        if pool.current_size <= pool.min_size:
            logger.debug(f"Pool '{pool.name}' already at minimum size ({pool.min_size})")
            return
        
        new_size = max(pool.current_size - rule.scale_down_increment, pool.min_size)
        
        async with self._lock:
            old_size = pool.current_size
            
            try:
                # Actually scale the resource
                if pool.resource_type == ResourceType.THREAD_POOL and pool.executor:
                    pool.executor.shutdown(wait=False)
                    pool.executor = ThreadPoolExecutor(
                        max_workers=new_size,
                        thread_name_prefix=f"{pool.name}_"
                    )
                elif pool.resource_type == ResourceType.PROCESS_POOL and pool.executor:
                    pool.executor.shutdown(wait=False)
                    pool.executor = ProcessPoolExecutor(max_workers=new_size)
                
                pool.current_size = new_size
                pool.last_scaled = time.time()
                
                # Record scaling event
                scaling_event = {
                    'timestamp': time.time(),
                    'action': 'scale_down',
                    'old_size': old_size,
                    'new_size': new_size,
                    'trigger_value': current_value,
                    'decision_value': decision_value,
                    'rule': rule.trigger.value
                }
                pool.scaling_history.append(scaling_event)
                
                logger.info(f"Scaled DOWN pool '{pool.name}' from {old_size} to {new_size} "
                           f"(trigger: {rule.trigger.value}={current_value:.1f}, "
                           f"threshold: {rule.scale_down_threshold})")
                
            except Exception as e:
                logger.error(f"Failed to scale down pool '{pool.name}': {e}")
                pool.current_size = old_size  # Revert
    
    def _adjust_performance_baselines(self, current_metrics: Dict[str, Any]):
        """Adaptively adjust performance baselines based on system behavior."""
        # Adjust CPU target based on recent performance
        cpu_percent = current_metrics.get('cpu_percent')
        if cpu_percent is not None:
            # If system is stable at current CPU levels, adjust target
            recent_cpu = [m.get('cpu_percent', 0) for m in self.metrics_history[-10:]]
            if recent_cpu:
                avg_cpu = sum(recent_cpu) / len(recent_cpu)
                if len(recent_cpu) == 10 and max(recent_cpu) - min(recent_cpu) < 10:  # Stable
                    # Gradually adjust target toward current stable level
                    self.performance_baselines['cpu_target'] += self.learning_rate * (avg_cpu - self.performance_baselines['cpu_target'])
                    self.performance_baselines['cpu_target'] = max(50.0, min(90.0, self.performance_baselines['cpu_target']))
        
        # Similar adjustment for memory
        memory_percent = current_metrics.get('memory_percent')
        if memory_percent is not None:
            recent_memory = [m.get('memory_percent', 0) for m in self.metrics_history[-10:]]
            if recent_memory and len(recent_memory) == 10:
                avg_memory = sum(recent_memory) / len(recent_memory)
                if max(recent_memory) - min(recent_memory) < 5:  # Stable
                    self.performance_baselines['memory_target'] += self.learning_rate * (avg_memory - self.performance_baselines['memory_target'])
                    self.performance_baselines['memory_target'] = max(60.0, min(95.0, self.performance_baselines['memory_target']))
    
    def get_pool_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all managed resource pools."""
        status = {}
        for name, pool in self.pools.items():
            status[name] = {
                'name': pool.name,
                'resource_type': pool.resource_type.value,
                'current_size': pool.current_size,
                'min_size': pool.min_size,
                'max_size': pool.max_size,
                'last_scaled': pool.last_scaled,
                'scaling_events_count': len(pool.scaling_history),
                'recent_scaling': pool.scaling_history[-5:] if pool.scaling_history else []
            }
        return status
    
    def get_scaling_rules_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all scaling rules."""
        status = {}
        for name, rule in self.scaling_rules.items():
            status[name] = {
                'name': name,
                'resource_type': rule.resource_type.value,
                'trigger': rule.trigger.value,
                'scale_up_threshold': rule.scale_up_threshold,
                'scale_down_threshold': rule.scale_down_threshold,
                'enabled': rule.enabled,
                'cooldown_seconds': rule.cooldown_seconds
            }
        return status
    
    def get_performance_baselines(self) -> Dict[str, float]:
        """Get current adaptive performance baselines."""
        return self.performance_baselines.copy()
    
    async def manual_scale_pool(self, pool_name: str, new_size: int) -> bool:
        """Manually scale a resource pool."""
        if pool_name not in self.pools:
            return False
        
        pool = self.pools[pool_name]
        if new_size < pool.min_size or new_size > pool.max_size:
            logger.error(f"Cannot scale pool '{pool_name}' to {new_size}: "
                        f"must be between {pool.min_size} and {pool.max_size}")
            return False
        
        async with self._lock:
            old_size = pool.current_size
            
            try:
                if pool.resource_type == ResourceType.THREAD_POOL and pool.executor:
                    pool.executor.shutdown(wait=False)
                    pool.executor = ThreadPoolExecutor(
                        max_workers=new_size,
                        thread_name_prefix=f"{pool.name}_"
                    )
                elif pool.resource_type == ResourceType.PROCESS_POOL and pool.executor:
                    pool.executor.shutdown(wait=False)
                    pool.executor = ProcessPoolExecutor(max_workers=new_size)
                
                pool.current_size = new_size
                pool.last_scaled = time.time()
                
                # Record manual scaling
                scaling_event = {
                    'timestamp': time.time(),
                    'action': 'manual_scale',
                    'old_size': old_size,
                    'new_size': new_size,
                    'trigger_value': None,
                    'decision_value': None,
                    'rule': 'manual'
                }
                pool.scaling_history.append(scaling_event)
                
                logger.info(f"Manually scaled pool '{pool_name}' from {old_size} to {new_size}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to manually scale pool '{pool_name}': {e}")
                pool.current_size = old_size
                return False


# Global auto-scaler instance
_auto_scaler = AutoScaler()


def get_auto_scaler() -> AutoScaler:
    """Get the global auto-scaler instance."""
    return _auto_scaler


async def start_auto_scaling():
    """Start the global auto-scaling system."""
    await _auto_scaler.start_auto_scaling()


async def stop_auto_scaling():
    """Stop the global auto-scaling system."""
    await _auto_scaler.stop_auto_scaling()


def create_managed_thread_pool(name: str, initial_size: int = 4, 
                              min_size: int = 2, max_size: int = 16) -> ThreadPoolExecutor:
    """Create a managed thread pool with auto-scaling."""
    pool = _auto_scaler.create_resource_pool(name, ResourceType.THREAD_POOL, 
                                            initial_size, min_size, max_size)
    return pool.executor


def create_managed_process_pool(name: str, initial_size: int = 2, 
                               min_size: int = 1, max_size: int = 8) -> ProcessPoolExecutor:
    """Create a managed process pool with auto-scaling."""
    pool = _auto_scaler.create_resource_pool(name, ResourceType.PROCESS_POOL,
                                            initial_size, min_size, max_size)
    return pool.executor


# Convenience functions for monitoring
async def get_scaling_status() -> Dict[str, Any]:
    """Get comprehensive auto-scaling status."""
    return {
        'is_running': _auto_scaler.is_running,
        'pools': _auto_scaler.get_pool_status(),
        'rules': _auto_scaler.get_scaling_rules_status(),
        'performance_baselines': _auto_scaler.get_performance_baselines(),
        'metrics_history_size': len(_auto_scaler.metrics_history)
    }