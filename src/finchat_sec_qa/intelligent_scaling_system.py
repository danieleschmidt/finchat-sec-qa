"""
Intelligent Scaling System v4.0
Advanced auto-scaling, load balancing, and performance optimization for autonomous SDLC.
"""

import asyncio
import json
import time
import math
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import logging
import threading
import statistics
import functools
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

logger = logging.getLogger(__name__)


class ScalingStrategy(Enum):
    """Scaling strategies"""
    REACTIVE = "reactive"
    PREDICTIVE = "predictive"
    HYBRID = "hybrid"
    MACHINE_LEARNING = "ml_based"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_RESPONSE_TIME = "weighted_response_time"
    RESOURCE_BASED = "resource_based"
    AI_OPTIMIZED = "ai_optimized"


class ResourceType(Enum):
    """Types of resources that can be scaled"""
    CPU_CORES = "cpu_cores"
    MEMORY_GB = "memory_gb"
    WORKER_THREADS = "worker_threads"
    WORKER_PROCESSES = "worker_processes"
    CONNECTION_POOLS = "connection_pools"
    CACHE_SIZE = "cache_size"


@dataclass
class ScalingMetrics:
    """Metrics used for scaling decisions"""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    active_connections: int
    queue_length: int
    response_time_avg: float
    throughput_rps: float
    error_rate: float
    resource_utilization: Dict[str, float] = field(default_factory=dict)


@dataclass
class ScalingAction:
    """A scaling action to be performed"""
    resource_type: ResourceType
    action: str  # "scale_up", "scale_down", "maintain"
    current_value: int
    target_value: int
    confidence: float
    reason: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class WorkerNode:
    """Individual worker node in the scaling system"""
    id: str
    cpu_cores: int
    memory_gb: float
    active_tasks: int
    max_tasks: int
    response_time_avg: float
    health_score: float
    last_activity: float
    status: str = "active"  # active, busy, unhealthy, draining


class IntelligentResourcePool:
    """Intelligent resource pool with auto-scaling"""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or (multiprocessing.cpu_count() * 2)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=multiprocessing.cpu_count())
        self.worker_nodes: Dict[str, WorkerNode] = {}
        self.task_queue: deque = deque()
        self.metrics_history: deque = deque(maxlen=1000)
        self.scaling_history: deque = deque(maxlen=500)
        self.current_load = 0.0
        self.predicted_load = 0.0
        
        # Initialize worker nodes
        self._initialize_worker_nodes()
    
    def _initialize_worker_nodes(self):
        """Initialize worker nodes"""
        for i in range(min(4, self.max_workers)):  # Start with 4 workers
            node_id = f"worker_{i}"
            self.worker_nodes[node_id] = WorkerNode(
                id=node_id,
                cpu_cores=1,
                memory_gb=1.0,
                active_tasks=0,
                max_tasks=10,
                response_time_avg=0.0,
                health_score=1.0,
                last_activity=time.time()
            )
    
    async def submit_task(self, func: Callable, *args, **kwargs) -> Any:
        """Submit task to resource pool"""
        # Choose best worker node
        worker = self._select_optimal_worker()
        
        if worker:
            worker.active_tasks += 1
            worker.last_activity = time.time()
            
            try:
                start_time = time.time()
                
                # Execute task based on type
                if kwargs.get('use_process_pool', False):
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(self.process_pool, func, *args)
                else:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(self.thread_pool, func, *args)
                
                execution_time = time.time() - start_time
                
                # Update worker metrics
                worker.response_time_avg = (worker.response_time_avg * 0.8) + (execution_time * 0.2)
                worker.active_tasks = max(0, worker.active_tasks - 1)
                
                return result
                
            except Exception as e:
                worker.active_tasks = max(0, worker.active_tasks - 1)
                worker.health_score *= 0.9  # Reduce health on errors
                raise
        else:
            raise Exception("No available workers")
    
    def _select_optimal_worker(self) -> Optional[WorkerNode]:
        """Select optimal worker based on current load"""
        available_workers = [
            w for w in self.worker_nodes.values() 
            if w.status == "active" and w.active_tasks < w.max_tasks
        ]
        
        if not available_workers:
            return None
        
        # Select worker with best score (lowest load + highest health)
        best_worker = min(available_workers, key=lambda w: (
            w.active_tasks / w.max_tasks * 0.6 +
            w.response_time_avg * 0.3 +
            (1.0 - w.health_score) * 0.1
        ))
        
        return best_worker
    
    def get_pool_metrics(self) -> Dict[str, Any]:
        """Get current pool metrics"""
        total_tasks = sum(w.active_tasks for w in self.worker_nodes.values())
        total_capacity = sum(w.max_tasks for w in self.worker_nodes.values())
        avg_response_time = statistics.mean([
            w.response_time_avg for w in self.worker_nodes.values() 
            if w.response_time_avg > 0
        ]) if any(w.response_time_avg > 0 for w in self.worker_nodes.values()) else 0
        
        utilization = (total_tasks / total_capacity) * 100 if total_capacity > 0 else 0
        
        return {
            "active_workers": len([w for w in self.worker_nodes.values() if w.status == "active"]),
            "total_workers": len(self.worker_nodes),
            "active_tasks": total_tasks,
            "total_capacity": total_capacity,
            "utilization_percent": utilization,
            "avg_response_time": avg_response_time,
            "queue_length": len(self.task_queue)
        }


class PredictiveScaler:
    """Predictive scaling based on historical patterns"""
    
    def __init__(self):
        self.load_history: deque = deque(maxlen=1000)
        self.pattern_window = 60  # minutes
        self.prediction_horizon = 15  # minutes
        
    def record_load(self, metrics: ScalingMetrics):
        """Record load metrics for pattern analysis"""
        self.load_history.append({
            "timestamp": metrics.timestamp,
            "cpu_usage": metrics.cpu_usage,
            "memory_usage": metrics.memory_usage,
            "throughput": metrics.throughput_rps,
            "response_time": metrics.response_time_avg
        })
    
    def predict_future_load(self, horizon_minutes: int = None) -> Dict[str, float]:
        """Predict future load based on historical patterns"""
        if len(self.load_history) < 10:
            return {"cpu_usage": 50.0, "memory_usage": 50.0, "throughput": 100.0}
        
        horizon = horizon_minutes or self.prediction_horizon
        
        # Simple trend analysis and seasonal pattern detection
        recent_data = list(self.load_history)[-60:]  # Last 60 data points
        
        # Calculate trends
        timestamps = [d["timestamp"] for d in recent_data]
        cpu_values = [d["cpu_usage"] for d in recent_data]
        throughput_values = [d["throughput"] for d in recent_data]
        
        # Linear trend calculation
        cpu_trend = self._calculate_trend(timestamps, cpu_values)
        throughput_trend = self._calculate_trend(timestamps, throughput_values)
        
        # Predict values
        current_time = time.time()
        future_time = current_time + (horizon * 60)
        
        latest_cpu = cpu_values[-1] if cpu_values else 50.0
        latest_throughput = throughput_values[-1] if throughput_values else 100.0
        
        predicted_cpu = max(0, min(100, latest_cpu + (cpu_trend * horizon)))
        predicted_throughput = max(0, latest_throughput + (throughput_trend * horizon))
        
        # Apply seasonal adjustments (simplified)
        hour_of_day = time.localtime(future_time).tm_hour
        seasonal_factor = self._get_seasonal_factor(hour_of_day)
        
        return {
            "cpu_usage": predicted_cpu * seasonal_factor,
            "memory_usage": predicted_cpu * 0.8 * seasonal_factor,  # Assume memory follows CPU
            "throughput": predicted_throughput * seasonal_factor,
            "confidence": min(len(recent_data) / 60.0, 1.0)  # Confidence based on data availability
        }
    
    def _calculate_trend(self, x_values: List[float], y_values: List[float]) -> float:
        """Calculate linear trend slope"""
        if len(x_values) < 2:
            return 0.0
        
        n = len(x_values)
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_x2 = sum(x * x for x in x_values)
        
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope
    
    def _get_seasonal_factor(self, hour: int) -> float:
        """Get seasonal adjustment factor based on hour of day"""
        # Simple seasonal pattern: higher load during business hours
        if 9 <= hour <= 17:  # Business hours
            return 1.2
        elif 6 <= hour <= 9 or 17 <= hour <= 22:  # Moderate hours
            return 1.0
        else:  # Low hours
            return 0.7


class AutoScaler:
    """Intelligent auto-scaling system"""
    
    def __init__(self, strategy: ScalingStrategy = ScalingStrategy.HYBRID):
        self.strategy = strategy
        self.predictor = PredictiveScaler()
        self.scaling_thresholds = {
            "cpu_scale_up": 70.0,
            "cpu_scale_down": 30.0,
            "memory_scale_up": 80.0,
            "memory_scale_down": 40.0,
            "response_time_scale_up": 1000.0,  # 1 second
            "queue_length_scale_up": 50
        }
        self.cooldown_period = 300  # 5 minutes
        self.last_scaling_action = 0
        self.scaling_decisions: deque = deque(maxlen=100)
    
    async def evaluate_scaling_decision(self, metrics: ScalingMetrics) -> List[ScalingAction]:
        """Evaluate and return scaling decisions"""
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self.last_scaling_action < self.cooldown_period:
            return []
        
        # Record metrics for prediction
        self.predictor.record_load(metrics)
        
        scaling_actions = []
        
        if self.strategy in [ScalingStrategy.REACTIVE, ScalingStrategy.HYBRID]:
            reactive_actions = self._evaluate_reactive_scaling(metrics)
            scaling_actions.extend(reactive_actions)
        
        if self.strategy in [ScalingStrategy.PREDICTIVE, ScalingStrategy.HYBRID]:
            predictive_actions = self._evaluate_predictive_scaling(metrics)
            scaling_actions.extend(predictive_actions)
        
        # Filter and prioritize actions
        final_actions = self._prioritize_actions(scaling_actions)
        
        if final_actions:
            self.last_scaling_action = current_time
            for action in final_actions:
                self.scaling_decisions.append(action)
        
        return final_actions
    
    def _evaluate_reactive_scaling(self, metrics: ScalingMetrics) -> List[ScalingAction]:
        """Evaluate reactive scaling based on current metrics"""
        actions = []
        
        # CPU-based scaling
        if metrics.cpu_usage > self.scaling_thresholds["cpu_scale_up"]:
            actions.append(ScalingAction(
                resource_type=ResourceType.WORKER_THREADS,
                action="scale_up",
                current_value=metrics.active_connections,
                target_value=int(metrics.active_connections * 1.5),
                confidence=0.8,
                reason=f"High CPU usage: {metrics.cpu_usage:.1f}%"
            ))
        elif metrics.cpu_usage < self.scaling_thresholds["cpu_scale_down"]:
            actions.append(ScalingAction(
                resource_type=ResourceType.WORKER_THREADS,
                action="scale_down",
                current_value=metrics.active_connections,
                target_value=max(1, int(metrics.active_connections * 0.8)),
                confidence=0.7,
                reason=f"Low CPU usage: {metrics.cpu_usage:.1f}%"
            ))
        
        # Memory-based scaling
        if metrics.memory_usage > self.scaling_thresholds["memory_scale_up"]:
            actions.append(ScalingAction(
                resource_type=ResourceType.CACHE_SIZE,
                action="scale_up",
                current_value=1024,  # Current cache size MB
                target_value=1536,   # Increase cache
                confidence=0.75,
                reason=f"High memory usage: {metrics.memory_usage:.1f}%"
            ))
        
        # Response time-based scaling
        if metrics.response_time_avg > self.scaling_thresholds["response_time_scale_up"]:
            actions.append(ScalingAction(
                resource_type=ResourceType.WORKER_PROCESSES,
                action="scale_up",
                current_value=4,  # Current processes
                target_value=6,   # Add more processes
                confidence=0.85,
                reason=f"High response time: {metrics.response_time_avg:.1f}ms"
            ))
        
        # Queue length-based scaling
        if metrics.queue_length > self.scaling_thresholds["queue_length_scale_up"]:
            actions.append(ScalingAction(
                resource_type=ResourceType.WORKER_THREADS,
                action="scale_up",
                current_value=metrics.active_connections,
                target_value=int(metrics.active_connections * 1.3),
                confidence=0.9,
                reason=f"High queue length: {metrics.queue_length}"
            ))
        
        return actions
    
    def _evaluate_predictive_scaling(self, metrics: ScalingMetrics) -> List[ScalingAction]:
        """Evaluate predictive scaling based on forecasted load"""
        predictions = self.predictor.predict_future_load()
        actions = []
        
        if predictions["confidence"] < 0.5:
            return actions  # Not enough data for reliable prediction
        
        predicted_cpu = predictions["cpu_usage"]
        predicted_throughput = predictions["throughput"]
        
        # Predictive CPU scaling
        if predicted_cpu > self.scaling_thresholds["cpu_scale_up"]:
            actions.append(ScalingAction(
                resource_type=ResourceType.WORKER_THREADS,
                action="scale_up",
                current_value=metrics.active_connections,
                target_value=int(metrics.active_connections * 1.2),
                confidence=predictions["confidence"] * 0.8,
                reason=f"Predicted high CPU usage: {predicted_cpu:.1f}%"
            ))
        
        # Predictive throughput scaling
        current_throughput = metrics.throughput_rps
        if predicted_throughput > current_throughput * 1.5:
            actions.append(ScalingAction(
                resource_type=ResourceType.CONNECTION_POOLS,
                action="scale_up",
                current_value=20,  # Current pool size
                target_value=30,   # Increase pool
                confidence=predictions["confidence"] * 0.7,
                reason=f"Predicted high throughput: {predicted_throughput:.1f} RPS"
            ))
        
        return actions
    
    def _prioritize_actions(self, actions: List[ScalingAction]) -> List[ScalingAction]:
        """Prioritize and filter scaling actions"""
        if not actions:
            return []
        
        # Sort by confidence and severity
        sorted_actions = sorted(actions, key=lambda a: (
            a.confidence * (1.0 if a.action == "scale_up" else 0.8),  # Prefer scale-up
            1.0 if a.resource_type in [ResourceType.CPU_CORES, ResourceType.MEMORY_GB] else 0.8
        ), reverse=True)
        
        # Take top 3 actions to avoid over-scaling
        return sorted_actions[:3]
    
    def get_scaling_summary(self) -> Dict[str, Any]:
        """Get scaling summary and statistics"""
        recent_decisions = [d for d in self.scaling_decisions if time.time() - d.timestamp < 3600]
        
        action_counts = defaultdict(int)
        resource_counts = defaultdict(int)
        
        for decision in recent_decisions:
            action_counts[decision.action] += 1
            resource_counts[decision.resource_type.value] += 1
        
        return {
            "strategy": self.strategy.value,
            "total_decisions": len(self.scaling_decisions),
            "recent_decisions": len(recent_decisions),
            "action_distribution": dict(action_counts),
            "resource_distribution": dict(resource_counts),
            "last_scaling_action": self.last_scaling_action,
            "cooldown_remaining": max(0, self.cooldown_period - (time.time() - self.last_scaling_action))
        }


class LoadBalancer:
    """Intelligent load balancer"""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.AI_OPTIMIZED):
        self.strategy = strategy
        self.backend_nodes: List[Dict[str, Any]] = []
        self.request_counter = 0
        self.response_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.node_weights: Dict[str, float] = {}
        
        # Initialize some backend nodes
        self._initialize_backends()
    
    def _initialize_backends(self):
        """Initialize backend nodes"""
        for i in range(3):  # Start with 3 backend nodes
            node = {
                "id": f"backend_{i}",
                "host": f"localhost",
                "port": 8000 + i,
                "active_connections": 0,
                "max_connections": 100,
                "health_score": 1.0,
                "cpu_usage": 20.0,
                "memory_usage": 30.0,
                "status": "healthy"
            }
            self.backend_nodes.append(node)
            self.node_weights[node["id"]] = 1.0
    
    def select_backend(self) -> Optional[Dict[str, Any]]:
        """Select optimal backend based on strategy"""
        healthy_nodes = [node for node in self.backend_nodes if node["status"] == "healthy"]
        
        if not healthy_nodes:
            return None
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_selection(healthy_nodes)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_selection(healthy_nodes)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_RESPONSE_TIME:
            return self._weighted_response_time_selection(healthy_nodes)
        elif self.strategy == LoadBalancingStrategy.RESOURCE_BASED:
            return self._resource_based_selection(healthy_nodes)
        elif self.strategy == LoadBalancingStrategy.AI_OPTIMIZED:
            return self._ai_optimized_selection(healthy_nodes)
        else:
            return healthy_nodes[0]  # Default to first healthy node
    
    def _round_robin_selection(self, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Round-robin selection"""
        self.request_counter += 1
        return nodes[self.request_counter % len(nodes)]
    
    def _least_connections_selection(self, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select node with least active connections"""
        return min(nodes, key=lambda n: n["active_connections"])
    
    def _weighted_response_time_selection(self, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select based on weighted response times"""
        scores = []
        for node in nodes:
            node_id = node["id"]
            recent_times = list(self.response_times[node_id])
            
            if recent_times:
                avg_response_time = statistics.mean(recent_times)
                # Lower response time = higher score
                score = 1.0 / max(avg_response_time, 0.001)
            else:
                score = 1.0  # Default score for new nodes
            
            scores.append((node, score))
        
        # Select based on weighted random selection
        total_score = sum(score for _, score in scores)
        if total_score == 0:
            return nodes[0]
        
        import random
        threshold = random.uniform(0, total_score)
        current_sum = 0
        
        for node, score in scores:
            current_sum += score
            if current_sum >= threshold:
                return node
        
        return nodes[-1]  # Fallback
    
    def _resource_based_selection(self, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select based on resource utilization"""
        def node_score(node):
            cpu_factor = 1.0 - (node["cpu_usage"] / 100.0)
            memory_factor = 1.0 - (node["memory_usage"] / 100.0)
            connection_factor = 1.0 - (node["active_connections"] / node["max_connections"])
            
            return (cpu_factor * 0.4 + memory_factor * 0.3 + connection_factor * 0.3) * node["health_score"]
        
        return max(nodes, key=node_score)
    
    def _ai_optimized_selection(self, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """AI-optimized selection combining multiple factors"""
        def calculate_node_score(node):
            node_id = node["id"]
            
            # Performance factors
            recent_times = list(self.response_times[node_id])
            avg_response_time = statistics.mean(recent_times) if recent_times else 100.0
            
            # Resource factors
            cpu_availability = 1.0 - (node["cpu_usage"] / 100.0)
            memory_availability = 1.0 - (node["memory_usage"] / 100.0)
            connection_availability = 1.0 - (node["active_connections"] / node["max_connections"])
            
            # Health factor
            health_score = node["health_score"]
            
            # Weight calculation (AI-like scoring)
            performance_score = 1.0 / max(avg_response_time, 1.0)  # Inverse of response time
            resource_score = (cpu_availability * 0.4 + memory_availability * 0.3 + connection_availability * 0.3)
            
            # Combined score with learning weights
            node_weight = self.node_weights.get(node_id, 1.0)
            final_score = (
                performance_score * 0.35 +
                resource_score * 0.35 +
                health_score * 0.2 +
                node_weight * 0.1
            )
            
            return final_score
        
        return max(nodes, key=calculate_node_score)
    
    def record_response_time(self, node_id: str, response_time: float):
        """Record response time for a node"""
        self.response_times[node_id].append(response_time)
        
        # Update node weights based on performance (simple learning)
        if len(self.response_times[node_id]) >= 10:
            avg_time = statistics.mean(self.response_times[node_id])
            # Better performance increases weight
            self.node_weights[node_id] = max(0.1, 2.0 - (avg_time / 500.0))
    
    def update_node_health(self, node_id: str, health_metrics: Dict[str, Any]):
        """Update node health metrics"""
        for node in self.backend_nodes:
            if node["id"] == node_id:
                node.update(health_metrics)
                break
    
    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics"""
        total_connections = sum(node["active_connections"] for node in self.backend_nodes)
        healthy_nodes = [node for node in self.backend_nodes if node["status"] == "healthy"]
        
        avg_response_times = {}
        for node_id, times in self.response_times.items():
            if times:
                avg_response_times[node_id] = statistics.mean(times)
        
        return {
            "strategy": self.strategy.value,
            "total_nodes": len(self.backend_nodes),
            "healthy_nodes": len(healthy_nodes),
            "total_active_connections": total_connections,
            "node_weights": dict(self.node_weights),
            "avg_response_times": avg_response_times,
            "request_counter": self.request_counter
        }


class IntelligentScalingSystem:
    """
    Comprehensive Intelligent Scaling System
    Combines auto-scaling, load balancing, and resource management
    """
    
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.resource_pool = IntelligentResourcePool()
        self.auto_scaler = AutoScaler(ScalingStrategy.HYBRID)
        self.load_balancer = LoadBalancer(LoadBalancingStrategy.AI_OPTIMIZED)
        self.scaling_active = False
        self.scaling_task: Optional[asyncio.Task] = None
        self.performance_metrics: deque = deque(maxlen=1000)
    
    async def start(self):
        """Start the intelligent scaling system"""
        logger.info(f"⚡ Starting Intelligent Scaling System for {self.project_name}")
        
        self.scaling_active = True
        self.scaling_task = asyncio.create_task(self._scaling_loop())
        
        logger.info("⚡ Intelligent scaling system started")
    
    async def stop(self):
        """Stop the intelligent scaling system"""
        logger.info("⚡ Stopping Intelligent Scaling System")
        
        self.scaling_active = False
        
        if self.scaling_task and not self.scaling_task.done():
            self.scaling_task.cancel()
            try:
                await self.scaling_task
            except asyncio.CancelledError:
                pass
        
        # Shutdown resource pools
        self.resource_pool.thread_pool.shutdown(wait=True)
        self.resource_pool.process_pool.shutdown(wait=True)
        
        logger.info("⚡ Intelligent scaling system stopped")
    
    async def _scaling_loop(self):
        """Main scaling evaluation loop"""
        while self.scaling_active:
            try:
                # Collect current metrics
                metrics = await self._collect_system_metrics()
                self.performance_metrics.append(metrics)
                
                # Evaluate scaling decisions
                scaling_actions = await self.auto_scaler.evaluate_scaling_decision(metrics)
                
                # Execute scaling actions
                for action in scaling_actions:
                    await self._execute_scaling_action(action)
                
                # Update load balancer with current metrics
                await self._update_load_balancer_metrics()
                
                # Wait before next evaluation
                await asyncio.sleep(30)  # Evaluate every 30 seconds
                
            except Exception as e:
                logger.error(f"❌ Error in scaling loop: {str(e)}")
                await asyncio.sleep(10)  # Back off on error
    
    async def _collect_system_metrics(self) -> ScalingMetrics:
        """Collect current system metrics"""
        pool_metrics = self.resource_pool.get_pool_metrics()
        
        # Simulate system metrics (in real implementation, use actual system monitoring)
        import random
        
        metrics = ScalingMetrics(
            timestamp=time.time(),
            cpu_usage=random.uniform(20, 80),  # Simulated CPU usage
            memory_usage=random.uniform(30, 70),  # Simulated memory usage
            active_connections=pool_metrics["active_tasks"],
            queue_length=pool_metrics["queue_length"],
            response_time_avg=pool_metrics["avg_response_time"] * 1000,  # Convert to ms
            throughput_rps=random.uniform(50, 200),  # Simulated throughput
            error_rate=random.uniform(0, 5),  # Simulated error rate
            resource_utilization={
                "thread_pool": pool_metrics["utilization_percent"],
                "worker_nodes": len(self.resource_pool.worker_nodes)
            }
        )
        
        return metrics
    
    async def _execute_scaling_action(self, action: ScalingAction):
        """Execute a scaling action"""
        logger.info(f"⚡ Executing scaling action: {action.action} {action.resource_type.value}")
        logger.info(f"   Reason: {action.reason}")
        logger.info(f"   Target: {action.current_value} → {action.target_value}")
        
        try:
            if action.resource_type == ResourceType.WORKER_THREADS:
                await self._scale_worker_threads(action)
            elif action.resource_type == ResourceType.WORKER_PROCESSES:
                await self._scale_worker_processes(action)
            elif action.resource_type == ResourceType.CONNECTION_POOLS:
                await self._scale_connection_pools(action)
            elif action.resource_type == ResourceType.CACHE_SIZE:
                await self._scale_cache_size(action)
            else:
                logger.warning(f"⚠️ Unknown resource type: {action.resource_type}")
                
        except Exception as e:
            logger.error(f"❌ Failed to execute scaling action: {str(e)}")
    
    async def _scale_worker_threads(self, action: ScalingAction):
        """Scale worker threads"""
        if action.action == "scale_up":
            # Add more worker nodes
            current_workers = len(self.resource_pool.worker_nodes)
            for i in range(min(2, action.target_value - current_workers)):
                node_id = f"worker_{current_workers + i}"
                self.resource_pool.worker_nodes[node_id] = WorkerNode(
                    id=node_id,
                    cpu_cores=1,
                    memory_gb=1.0,
                    active_tasks=0,
                    max_tasks=10,
                    response_time_avg=0.0,
                    health_score=1.0,
                    last_activity=time.time()
                )
        elif action.action == "scale_down":
            # Remove excess worker nodes
            workers_to_remove = len(self.resource_pool.worker_nodes) - action.target_value
            if workers_to_remove > 0:
                # Remove least active workers
                sorted_workers = sorted(
                    self.resource_pool.worker_nodes.items(),
                    key=lambda x: x[1].active_tasks
                )
                for i in range(min(workers_to_remove, len(sorted_workers) - 1)):  # Keep at least 1
                    worker_id = sorted_workers[i][0]
                    if self.resource_pool.worker_nodes[worker_id].active_tasks == 0:
                        del self.resource_pool.worker_nodes[worker_id]
    
    async def _scale_worker_processes(self, action: ScalingAction):
        """Scale worker processes"""
        # In a real implementation, this would adjust the ProcessPoolExecutor
        logger.info(f"📈 Process scaling: {action.action} to {action.target_value} processes")
    
    async def _scale_connection_pools(self, action: ScalingAction):
        """Scale connection pools"""
        logger.info(f"🔗 Connection pool scaling: {action.action} to {action.target_value} connections")
    
    async def _scale_cache_size(self, action: ScalingAction):
        """Scale cache size"""
        logger.info(f"💾 Cache scaling: {action.action} to {action.target_value} MB")
    
    async def _update_load_balancer_metrics(self):
        """Update load balancer with current node metrics"""
        for node in self.load_balancer.backend_nodes:
            # Simulate node health updates
            import random
            health_metrics = {
                "cpu_usage": random.uniform(20, 80),
                "memory_usage": random.uniform(30, 70),
                "active_connections": random.randint(5, 50),
                "health_score": random.uniform(0.8, 1.0)
            }
            self.load_balancer.update_node_health(node["id"], health_metrics)
    
    async def process_request(self, func: Callable, *args, **kwargs) -> Any:
        """Process request through the scaling system"""
        # Select backend through load balancer
        backend = self.load_balancer.select_backend()
        if not backend:
            raise Exception("No healthy backends available")
        
        start_time = time.time()
        
        try:
            # Execute through resource pool
            result = await self.resource_pool.submit_task(func, *args, **kwargs)
            
            # Record performance metrics
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            self.load_balancer.record_response_time(backend["id"], response_time)
            
            return result
            
        except Exception as e:
            # Update backend health on error
            backend["health_score"] *= 0.9
            raise
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get complete scaling system status"""
        latest_metrics = self.performance_metrics[-1] if self.performance_metrics else None
        
        return {
            "project_name": self.project_name,
            "scaling_active": self.scaling_active,
            "resource_pool_metrics": self.resource_pool.get_pool_metrics(),
            "scaling_summary": self.auto_scaler.get_scaling_summary(),
            "load_balancer_stats": self.load_balancer.get_load_balancer_stats(),
            "latest_metrics": asdict(latest_metrics) if latest_metrics else None,
            "metrics_history_size": len(self.performance_metrics)
        }


# Factory function
def create_scaling_system(project_name: str) -> IntelligentScalingSystem:
    """Create intelligent scaling system"""
    return IntelligentScalingSystem(project_name)


# Example usage
async def demonstrate_scaling():
    """Demonstrate intelligent scaling system"""
    scaling_system = create_scaling_system("FinChat-SEC-QA")
    
    try:
        await scaling_system.start()
        
        # Simulate some workload
        async def sample_task(workload_id: int):
            # Simulate some work
            await asyncio.sleep(0.1 + (workload_id % 3) * 0.05)
            return f"Task {workload_id} completed"
        
        # Submit multiple tasks to trigger scaling
        tasks = []
        for i in range(20):
            task = scaling_system.process_request(sample_task, i)
            tasks.append(task)
            
            if i % 5 == 0:
                await asyncio.sleep(1)  # Simulate bursts
        
        # Wait for tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info(f"✅ Completed {len([r for r in results if not isinstance(r, Exception)])} tasks")
        
        # Let scaling system run for a bit
        await asyncio.sleep(10)
        
        # Get status
        status = scaling_system.get_scaling_status()
        logger.info(f"⚡ Scaling system status: {status}")
        
    finally:
        await scaling_system.stop()


if __name__ == "__main__":
    # Example usage
    async def main():
        await demonstrate_scaling()
    
    asyncio.run(main())