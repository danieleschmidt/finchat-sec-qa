"""
Photonic Quantum Computing Cache and Optimization Module.

This module provides advanced caching, optimization, and scaling capabilities 
for the photonic quantum computing bridge.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from .photonic_mlir import FinancialQueryType, PhotonicCircuit, QuantumFinancialResult

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cache entry for quantum computations."""

    key: str
    value: Any
    created_at: datetime
    accessed_at: datetime
    access_count: int
    expiry_time: Optional[datetime]
    size_bytes: int
    metadata: Dict[str, Any]

    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        if self.expiry_time is None:
            return False
        return datetime.now() > self.expiry_time

    def touch(self) -> None:
        """Update access time and count."""
        self.accessed_at = datetime.now()
        self.access_count += 1


@dataclass
class CircuitOptimizationResult:
    """Result of circuit optimization analysis."""

    original_depth: int
    optimized_depth: int
    original_gate_count: int
    optimized_gate_count: int
    optimization_techniques_applied: List[str]
    estimated_speedup: float
    fidelity_impact: float
    optimization_time_ms: float


class QuantumCircuitCache:
    """
    Advanced cache for quantum circuits with LRU eviction and optimization.
    """

    def __init__(
        self,
        max_size_mb: int = 100,
        default_ttl_hours: int = 24,
        enable_compression: bool = True
    ):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.default_ttl = timedelta(hours=default_ttl_hours)
        self.enable_compression = enable_compression

        self._cache: Dict[str, CacheEntry] = {}
        self._size_tracker = 0
        self._lock = asyncio.Lock()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Performance metrics
        self._hit_count = 0
        self._miss_count = 0
        self._eviction_count = 0

        self.logger.info(f"Initialized quantum circuit cache: {max_size_mb}MB max, {default_ttl_hours}h TTL")

    def _generate_key(self, query: str, query_type: FinancialQueryType, data_hash: str) -> str:
        """Generate cache key from query parameters."""
        key_data = f"{query}:{query_type.value}:{data_hash}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]

    def _calculate_size(self, value: Any) -> int:
        """Estimate size of cached value in bytes."""
        if isinstance(value, PhotonicCircuit):
            # Estimate circuit size based on operations and metadata
            base_size = 1024  # Base circuit overhead
            ops_size = len(value.operations) * 256  # ~256 bytes per operation
            context_size = len(str(value.financial_context)) * 2  # String size estimate
            return base_size + ops_size + context_size
        elif isinstance(value, QuantumFinancialResult):
            # Estimate result size
            base_size = 512
            result_size = len(json.dumps(value.quantum_result, default=str)) * 2
            metadata_size = len(json.dumps(value.metadata, default=str)) * 2
            return base_size + result_size + metadata_size
        else:
            # Fallback estimation
            return len(str(value)) * 2

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        async with self._lock:
            if key not in self._cache:
                self._miss_count += 1
                return None

            entry = self._cache[key]

            # Check expiry
            if entry.is_expired():
                del self._cache[key]
                self._size_tracker -= entry.size_bytes
                self._miss_count += 1
                return None

            # Update access info
            entry.touch()
            self._hit_count += 1

            self.logger.debug(f"Cache hit for key: {key}")
            return entry.value

    async def put(
        self,
        key: str,
        value: Any,
        ttl: Optional[timedelta] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Put value in cache."""
        async with self._lock:
            size_bytes = self._calculate_size(value)

            # Check if we need to evict entries
            await self._evict_if_needed(size_bytes)

            expiry_time = None
            if ttl or self.default_ttl:
                expiry_time = datetime.now() + (ttl or self.default_ttl)

            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                accessed_at=datetime.now(),
                access_count=1,
                expiry_time=expiry_time,
                size_bytes=size_bytes,
                metadata=metadata or {}
            )

            self._cache[key] = entry
            self._size_tracker += size_bytes

            self.logger.debug(f"Cached entry: {key} ({size_bytes} bytes)")

    async def _evict_if_needed(self, incoming_size: int) -> None:
        """Evict entries using LRU policy if needed."""
        target_size = self.max_size_bytes - incoming_size

        if self._size_tracker <= target_size:
            return

        # Sort by access time (LRU first)
        entries_by_access = sorted(
            self._cache.items(),
            key=lambda x: x[1].accessed_at
        )

        for key, entry in entries_by_access:
            if self._size_tracker <= target_size:
                break

            del self._cache[key]
            self._size_tracker -= entry.size_bytes
            self._eviction_count += 1

            self.logger.debug(f"Evicted cache entry: {key}")

    async def clear_expired(self) -> int:
        """Clear expired entries and return count cleared."""
        async with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]

            for key in expired_keys:
                entry = self._cache[key]
                del self._cache[key]
                self._size_tracker -= entry.size_bytes

            if expired_keys:
                self.logger.info(f"Cleared {len(expired_keys)} expired cache entries")

            return len(expired_keys)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self._hit_count + self._miss_count
        hit_rate = self._hit_count / total_requests if total_requests > 0 else 0.0

        return {
            "entries": len(self._cache),
            "size_bytes": self._size_tracker,
            "size_mb": self._size_tracker / (1024 * 1024),
            "max_size_mb": self.max_size_bytes / (1024 * 1024),
            "utilization": self._size_tracker / self.max_size_bytes,
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": hit_rate,
            "eviction_count": self._eviction_count
        }


class QuantumOptimizer:
    """
    Advanced quantum circuit optimizer with multiple optimization techniques.
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.optimization_cache: Dict[str, CircuitOptimizationResult] = {}

        # Optimization techniques
        self.optimizers = {
            'gate_fusion': self._optimize_gate_fusion,
            'redundancy_elimination': self._optimize_redundancy_elimination,
            'commutation_optimization': self._optimize_commutation,
            'depth_reduction': self._optimize_depth_reduction,
            'noise_aware': self._optimize_noise_aware
        }

    def optimize_circuit(
        self,
        circuit: PhotonicCircuit,
        optimization_level: int = 2,
        target_metrics: Optional[List[str]] = None
    ) -> Tuple[PhotonicCircuit, CircuitOptimizationResult]:
        """
        Optimize quantum circuit with specified optimization level.
        
        Args:
            circuit: Circuit to optimize
            optimization_level: 0=none, 1=basic, 2=advanced, 3=aggressive
            target_metrics: Target metrics for optimization (depth, gates, fidelity)
            
        Returns:
            Tuple of optimized circuit and optimization result
        """
        start_time = time.time()

        # Create circuit fingerprint for caching
        circuit_hash = self._hash_circuit(circuit)
        cache_key = f"{circuit_hash}:{optimization_level}:{target_metrics}"

        if cache_key in self.optimization_cache:
            self.logger.debug(f"Using cached optimization for circuit {circuit.circuit_id}")
            cached_result = self.optimization_cache[cache_key]
            # Return new circuit with cached optimization applied
            return self._apply_cached_optimization(circuit, cached_result), cached_result

        self.logger.info(f"Optimizing circuit {circuit.circuit_id} (level {optimization_level})")

        # Track original metrics
        original_depth = len(circuit.operations)
        original_gate_count = len(circuit.operations)

        # Apply optimizations based on level
        optimized_circuit = circuit
        techniques_applied = []

        if optimization_level >= 1:
            # Basic optimizations
            optimized_circuit, applied = self._apply_basic_optimizations(optimized_circuit)
            techniques_applied.extend(applied)

        if optimization_level >= 2:
            # Advanced optimizations
            optimized_circuit, applied = self._apply_advanced_optimizations(optimized_circuit)
            techniques_applied.extend(applied)

        if optimization_level >= 3:
            # Aggressive optimizations
            optimized_circuit, applied = self._apply_aggressive_optimizations(optimized_circuit)
            techniques_applied.extend(applied)

        # Calculate optimization results
        optimized_depth = len(optimized_circuit.operations)
        optimized_gate_count = len(optimized_circuit.operations)

        speedup = original_depth / optimized_depth if optimized_depth > 0 else 1.0
        fidelity_impact = self._estimate_fidelity_impact(techniques_applied)
        optimization_time = (time.time() - start_time) * 1000  # ms

        result = CircuitOptimizationResult(
            original_depth=original_depth,
            optimized_depth=optimized_depth,
            original_gate_count=original_gate_count,
            optimized_gate_count=optimized_gate_count,
            optimization_techniques_applied=techniques_applied,
            estimated_speedup=speedup,
            fidelity_impact=fidelity_impact,
            optimization_time_ms=optimization_time
        )

        # Cache result
        self.optimization_cache[cache_key] = result

        self.logger.info(
            f"Circuit optimization completed: {original_depth}->{optimized_depth} depth "
            f"({speedup:.1f}x speedup) in {optimization_time:.1f}ms"
        )

        return optimized_circuit, result

    def _hash_circuit(self, circuit: PhotonicCircuit) -> str:
        """Generate hash for circuit structure."""
        circuit_data = {
            'operations': [
                (op.gate_type.value, op.qubits, list(op.parameters.keys()))
                for op in circuit.operations
            ],
            'input_qubits': circuit.input_qubits,
            'output_qubits': circuit.output_qubits
        }
        return hashlib.md5(json.dumps(circuit_data, sort_keys=True).encode(), usedforsecurity=False).hexdigest()[:12]

    def _apply_basic_optimizations(self, circuit: PhotonicCircuit) -> Tuple[PhotonicCircuit, List[str]]:
        """Apply basic optimization techniques."""
        techniques = []
        optimized_ops = circuit.operations.copy()

        # Gate fusion optimization
        optimized_ops, fused = self._optimize_gate_fusion(optimized_ops)
        if fused:
            techniques.append('gate_fusion')

        # Redundancy elimination
        optimized_ops, eliminated = self._optimize_redundancy_elimination(optimized_ops)
        if eliminated:
            techniques.append('redundancy_elimination')

        # Create optimized circuit
        optimized_circuit = PhotonicCircuit(
            circuit_id=f"{circuit.circuit_id}_basic_opt",
            operations=optimized_ops,
            input_qubits=circuit.input_qubits,
            output_qubits=circuit.output_qubits,
            financial_context=circuit.financial_context,
            created_at=datetime.now()
        )

        return optimized_circuit, techniques

    def _apply_advanced_optimizations(self, circuit: PhotonicCircuit) -> Tuple[PhotonicCircuit, List[str]]:
        """Apply advanced optimization techniques."""
        techniques = []
        optimized_ops = circuit.operations.copy()

        # Commutation optimization
        optimized_ops, commuted = self._optimize_commutation(optimized_ops)
        if commuted:
            techniques.append('commutation_optimization')

        # Depth reduction
        optimized_ops, reduced = self._optimize_depth_reduction(optimized_ops)
        if reduced:
            techniques.append('depth_reduction')

        optimized_circuit = PhotonicCircuit(
            circuit_id=f"{circuit.circuit_id}_advanced_opt",
            operations=optimized_ops,
            input_qubits=circuit.input_qubits,
            output_qubits=circuit.output_qubits,
            financial_context=circuit.financial_context,
            created_at=datetime.now()
        )

        return optimized_circuit, techniques

    def _apply_aggressive_optimizations(self, circuit: PhotonicCircuit) -> Tuple[PhotonicCircuit, List[str]]:
        """Apply aggressive optimization techniques."""
        techniques = []
        optimized_ops = circuit.operations.copy()

        # Noise-aware optimization
        optimized_ops, noise_optimized = self._optimize_noise_aware(optimized_ops)
        if noise_optimized:
            techniques.append('noise_aware')

        optimized_circuit = PhotonicCircuit(
            circuit_id=f"{circuit.circuit_id}_aggressive_opt",
            operations=optimized_ops,
            input_qubits=circuit.input_qubits,
            output_qubits=circuit.output_qubits,
            financial_context=circuit.financial_context,
            created_at=datetime.now()
        )

        return optimized_circuit, techniques

    def _optimize_gate_fusion(self, operations) -> Tuple[List, bool]:
        """Fuse consecutive gates on same qubits."""
        # Simplified gate fusion - in practice would be more sophisticated
        fused_ops = []
        i = 0
        fused = False

        while i < len(operations):
            current = operations[i]

            # Look for consecutive rotation gates on same qubit
            if (i + 1 < len(operations) and
                current.gate_type.value.startswith('rotation') and
                operations[i + 1].gate_type == current.gate_type and
                current.qubits == operations[i + 1].qubits):

                # Fuse the operations
                from .photonic_mlir import MLIRQuantumOperation
                fused_params = current.parameters.copy()
                for key, value in operations[i + 1].parameters.items():
                    fused_params[key] = fused_params.get(key, 0) + value

                fused_op = MLIRQuantumOperation(
                    operation_type=f"fused_{current.operation_type}",
                    gate_type=current.gate_type,
                    qubits=current.qubits,
                    parameters=fused_params,
                    metadata={"fused": True, "original_count": 2}
                )
                fused_ops.append(fused_op)
                i += 2
                fused = True
            else:
                fused_ops.append(current)
                i += 1

        return fused_ops, fused

    def _optimize_redundancy_elimination(self, operations) -> Tuple[List, bool]:
        """Remove redundant operations."""
        filtered_ops = []
        eliminated = False

        for op in operations:
            # Remove operations with effectively zero parameters
            if op.gate_type.value in ['phase_shifter', 'rotation_x', 'rotation_y', 'rotation_z']:
                param_sum = sum(abs(v) for v in op.parameters.values())
                if param_sum < 1e-10:
                    eliminated = True
                    continue

            # Remove duplicate consecutive operations that cancel out
            if (filtered_ops and
                op.gate_type == filtered_ops[-1].gate_type and
                op.qubits == filtered_ops[-1].qubits and
                op.gate_type.value == 'hadamard'):  # Hadamard is self-inverse
                filtered_ops.pop()  # Remove the previous operation
                eliminated = True
                continue

            filtered_ops.append(op)

        return filtered_ops, eliminated

    def _optimize_commutation(self, operations) -> Tuple[List, bool]:
        """Optimize using gate commutation rules."""
        # Simplified commutation optimization
        # In practice, would implement sophisticated commutation analysis
        optimized_ops = operations.copy()
        commuted = False

        # Example: Move single-qubit gates before two-qubit gates when possible
        for i in range(len(optimized_ops) - 1):
            current = optimized_ops[i]
            next_op = optimized_ops[i + 1]

            # If current is 2-qubit and next is 1-qubit on different qubits, swap
            if (len(current.qubits) == 2 and len(next_op.qubits) == 1 and
                next_op.qubits[0] not in current.qubits):
                optimized_ops[i], optimized_ops[i + 1] = next_op, current
                commuted = True

        return optimized_ops, commuted

    def _optimize_depth_reduction(self, operations) -> Tuple[List, bool]:
        """Reduce circuit depth through parallelization."""
        # Simplified depth reduction
        # Group operations that can run in parallel
        parallel_groups = []
        current_group = []
        used_qubits = set()
        reduced = False

        for op in operations:
            op_qubits = set(op.qubits)

            # If operation uses qubits already in use, start new group
            if op_qubits & used_qubits:
                if current_group:
                    parallel_groups.append(current_group)
                current_group = [op]
                used_qubits = op_qubits
            else:
                current_group.append(op)
                used_qubits.update(op_qubits)

        if current_group:
            parallel_groups.append(current_group)

        # If we found opportunities for parallelization
        if len(parallel_groups) < len(operations):
            reduced = True

        # Flatten back to sequential for now (would need parallel execution support)
        flattened = [op for group in parallel_groups for op in group]

        return flattened, reduced

    def _optimize_noise_aware(self, operations) -> Tuple[List, bool]:
        """Apply noise-aware optimizations."""
        # Simplified noise-aware optimization
        # Prioritize gates with higher fidelity, minimize gate count
        optimized_ops = operations.copy()
        noise_optimized = False

        # Sort operations to minimize noise accumulation
        # Single-qubit gates typically have higher fidelity
        single_qubit_ops = [op for op in optimized_ops if len(op.qubits) == 1]
        two_qubit_ops = [op for op in optimized_ops if len(op.qubits) == 2]

        if len(single_qubit_ops) > 0 and len(two_qubit_ops) > 0:
            # Reorder to group single-qubit operations
            optimized_ops = single_qubit_ops + two_qubit_ops
            noise_optimized = True

        return optimized_ops, noise_optimized

    def _estimate_fidelity_impact(self, techniques: List[str]) -> float:
        """Estimate fidelity impact of optimization techniques."""
        fidelity_impacts = {
            'gate_fusion': 0.002,  # Small improvement
            'redundancy_elimination': 0.005,  # Medium improvement
            'commutation_optimization': 0.001,  # Small impact
            'depth_reduction': 0.003,  # Small-medium improvement
            'noise_aware': 0.004  # Medium improvement
        }

        total_impact = sum(fidelity_impacts.get(technique, 0) for technique in techniques)
        return min(0.02, total_impact)  # Cap at 2% improvement

    def _apply_cached_optimization(self, circuit: PhotonicCircuit, cached_result: CircuitOptimizationResult) -> PhotonicCircuit:
        """Apply cached optimization result to a circuit."""
        # In practice, would store and apply actual optimization transformations
        # For now, return circuit with updated metadata
        return PhotonicCircuit(
            circuit_id=f"{circuit.circuit_id}_cached_opt",
            operations=circuit.operations,  # Would apply actual optimizations
            input_qubits=circuit.input_qubits,
            output_qubits=circuit.output_qubits,
            financial_context=circuit.financial_context,
            created_at=datetime.now()
        )


class ParallelQuantumProcessor:
    """
    Parallel processor for handling multiple quantum computations concurrently.
    """

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Load balancing
        self._active_tasks = 0
        self._task_queue: List[Callable] = []

        self.logger.info(f"Initialized parallel quantum processor with {max_workers} workers")

    async def process_batch(
        self,
        queries: List[Dict[str, Any]],
        processor_func: Callable,
        batch_size: int = None
    ) -> List[QuantumFinancialResult]:
        """
        Process multiple quantum queries in parallel.
        
        Args:
            queries: List of query dictionaries
            processor_func: Function to process individual queries
            batch_size: Size of processing batches
            
        Returns:
            List of quantum financial results
        """
        if not queries:
            return []

        batch_size = batch_size or min(self.max_workers, len(queries))
        results = []

        self.logger.info(f"Processing {len(queries)} queries in batches of {batch_size}")

        # Process queries in batches
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]

            # Submit batch to thread pool
            futures = {
                self.executor.submit(processor_func, **query): idx
                for idx, query in enumerate(batch)
            }

            # Collect results as they complete
            batch_results = [None] * len(batch)
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result(timeout=30)  # 30 second timeout per query
                    batch_results[idx] = result
                except Exception as e:
                    self.logger.error(f"Query processing failed: {e}")
                    # Create error result
                    batch_results[idx] = self._create_error_result(str(e))

            results.extend(batch_results)

            # Small delay between batches to prevent overwhelming
            if i + batch_size < len(queries):
                await asyncio.sleep(0.1)

        self.logger.info(f"Completed batch processing: {len(results)} results")
        return results

    def _create_error_result(self, error_message: str) -> QuantumFinancialResult:
        """Create error result for failed processing."""
        from .photonic_mlir import FinancialQueryType, QuantumFinancialResult

        return QuantumFinancialResult(
            query_id=f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            query_type=FinancialQueryType.RISK_ASSESSMENT,
            quantum_advantage=1.0,
            classical_result={},
            quantum_result={"error": error_message},
            confidence_score=0.0,
            processing_time_ms=0.0,
            circuit_depth=0,
            metadata={"error": True}
        )

    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "max_workers": self.max_workers,
            "active_tasks": self._active_tasks,
            "queued_tasks": len(self._task_queue),
            "utilization": self._active_tasks / self.max_workers
        }

    def shutdown(self):
        """Shutdown the parallel processor."""
        self.logger.info("Shutting down parallel quantum processor")
        self.executor.shutdown(wait=True)


class PerformanceProfiler:
    """
    Performance profiler for quantum operations.
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.profiles: Dict[str, List[Dict[str, Any]]] = {}
        self._enabled = True

    def profile_operation(self, operation_name: str):
        """Decorator for profiling operations."""
        def decorator(func):
            async def async_wrapper(*args, **kwargs):
                if not self._enabled:
                    return await func(*args, **kwargs)

                start_time = time.perf_counter()
                start_memory = self._get_memory_usage()

                try:
                    result = await func(*args, **kwargs)
                    success = True
                    error = None
                except Exception as e:
                    success = False
                    error = str(e)
                    raise
                finally:
                    end_time = time.perf_counter()
                    end_memory = self._get_memory_usage()

                    profile_data = {
                        'timestamp': datetime.now().isoformat(),
                        'duration_ms': (end_time - start_time) * 1000,
                        'memory_delta_mb': (end_memory - start_memory) / (1024 * 1024),
                        'success': success,
                        'error': error,
                        'args_count': len(args),
                        'kwargs_count': len(kwargs)
                    }

                    if operation_name not in self.profiles:
                        self.profiles[operation_name] = []

                    self.profiles[operation_name].append(profile_data)

                    # Keep only last 100 profiles per operation
                    if len(self.profiles[operation_name]) > 100:
                        self.profiles[operation_name] = self.profiles[operation_name][-100:]

                return result

            def sync_wrapper(*args, **kwargs):
                if not self._enabled:
                    return func(*args, **kwargs)

                start_time = time.perf_counter()
                start_memory = self._get_memory_usage()

                try:
                    result = func(*args, **kwargs)
                    success = True
                    error = None
                except Exception as e:
                    success = False
                    error = str(e)
                    raise
                finally:
                    end_time = time.perf_counter()
                    end_memory = self._get_memory_usage()

                    profile_data = {
                        'timestamp': datetime.now().isoformat(),
                        'duration_ms': (end_time - start_time) * 1000,
                        'memory_delta_mb': (end_memory - start_memory) / (1024 * 1024),
                        'success': success,
                        'error': error,
                        'args_count': len(args),
                        'kwargs_count': len(kwargs)
                    }

                    if operation_name not in self.profiles:
                        self.profiles[operation_name] = []

                    self.profiles[operation_name].append(profile_data)

                    # Keep only last 100 profiles per operation
                    if len(self.profiles[operation_name]) > 100:
                        self.profiles[operation_name] = self.profiles[operation_name][-100:]

                return result

            # Return appropriate wrapper based on function type
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return decorator

    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        try:
            import os

            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss
        except ImportError:
            # Fallback if psutil not available
            return 0

    def get_profile_summary(self, operation_name: str) -> Dict[str, Any]:
        """Get profile summary for an operation."""
        if operation_name not in self.profiles:
            return {"error": "No profiles found for operation"}

        profiles = self.profiles[operation_name]
        durations = [p['duration_ms'] for p in profiles if p['success']]
        memory_deltas = [p['memory_delta_mb'] for p in profiles]

        if not durations:
            return {"error": "No successful operations to profile"}

        return {
            "operation_name": operation_name,
            "total_calls": len(profiles),
            "successful_calls": len(durations),
            "error_rate": 1 - (len(durations) / len(profiles)),
            "avg_duration_ms": sum(durations) / len(durations),
            "min_duration_ms": min(durations),
            "max_duration_ms": max(durations),
            "avg_memory_delta_mb": sum(memory_deltas) / len(memory_deltas) if memory_deltas else 0,
            "last_call": profiles[-1]['timestamp']
        }

    def get_all_profiles_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all operation profiles."""
        return {
            operation: self.get_profile_summary(operation)
            for operation in self.profiles.keys()
        }

    def enable_profiling(self, enabled: bool = True):
        """Enable or disable profiling."""
        self._enabled = enabled
        self.logger.info(f"Performance profiling {'enabled' if enabled else 'disabled'}")

    def clear_profiles(self):
        """Clear all collected profiles."""
        self.profiles.clear()
        self.logger.info("Cleared all performance profiles")


# Global instances for easy access
_circuit_cache = None
_quantum_optimizer = None
_parallel_processor = None
_performance_profiler = None


def get_circuit_cache() -> QuantumCircuitCache:
    """Get global circuit cache instance."""
    global _circuit_cache
    if _circuit_cache is None:
        _circuit_cache = QuantumCircuitCache()
    return _circuit_cache


def get_quantum_optimizer() -> QuantumOptimizer:
    """Get global quantum optimizer instance."""
    global _quantum_optimizer
    if _quantum_optimizer is None:
        _quantum_optimizer = QuantumOptimizer()
    return _quantum_optimizer


def get_parallel_processor() -> ParallelQuantumProcessor:
    """Get global parallel processor instance."""
    global _parallel_processor
    if _parallel_processor is None:
        _parallel_processor = ParallelQuantumProcessor()
    return _parallel_processor


def get_performance_profiler() -> PerformanceProfiler:
    """Get global performance profiler instance."""
    global _performance_profiler
    if _performance_profiler is None:
        _performance_profiler = PerformanceProfiler()
    return _performance_profiler


# Export main classes and functions
__all__ = [
    "QuantumCircuitCache",
    "QuantumOptimizer",
    "ParallelQuantumProcessor",
    "PerformanceProfiler",
    "CacheEntry",
    "CircuitOptimizationResult",
    "get_circuit_cache",
    "get_quantum_optimizer",
    "get_parallel_processor",
    "get_performance_profiler"
]
