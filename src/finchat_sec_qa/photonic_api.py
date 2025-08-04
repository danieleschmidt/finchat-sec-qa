"""
Enhanced FastAPI endpoints for Photonic Quantum Computing integration.

This module extends the existing REST API with quantum-enhanced financial analysis endpoints.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from .photonic_bridge import PhotonicBridge
from .photonic_cache import (
    get_circuit_cache,
    get_parallel_processor,
    get_performance_profiler,
    get_quantum_optimizer,
)
from .photonic_mlir import FinancialQueryType
from .qa_engine import FinancialQAEngine

logger = logging.getLogger(__name__)

# Pydantic models for API
class QuantumQueryRequest(BaseModel):
    """Request model for quantum-enhanced queries."""

    query: str = Field(..., description="Financial query text", min_length=1, max_length=1000)
    document_path: Optional[str] = Field(None, description="Path to financial document")
    document_content: Optional[str] = Field(None, description="Direct document content")
    query_type: Optional[FinancialQueryType] = Field(None, description="Type of financial query")
    enable_quantum: bool = Field(True, description="Enable quantum enhancement")
    quantum_threshold: float = Field(0.7, description="Quantum enhancement threshold", ge=0.0, le=1.0)
    optimization_level: int = Field(2, description="Circuit optimization level", ge=0, le=3)

    @validator('document_path', 'document_content')
    def validate_document_source(cls, v, values):
        """Ensure either document_path or document_content is provided."""
        if not v and not values.get('document_content') and not values.get('document_path'):
            raise ValueError("Either document_path or document_content must be provided")
        return v


class QuantumBatchRequest(BaseModel):
    """Request model for batch quantum processing."""

    queries: List[QuantumQueryRequest] = Field(..., description="List of quantum queries", min_items=1, max_items=10)
    parallel_processing: bool = Field(True, description="Enable parallel processing")
    batch_size: Optional[int] = Field(None, description="Batch processing size", ge=1, le=5)


class QuantumBenchmarkRequest(BaseModel):
    """Request model for quantum benchmarking."""

    test_queries: List[str] = Field(..., description="Test queries for benchmarking", min_items=1, max_items=20)
    document_paths: List[str] = Field(..., description="Document paths for testing", min_items=1, max_items=10)
    iterations: int = Field(1, description="Number of benchmark iterations", ge=1, le=10)


class QuantumCapabilitiesResponse(BaseModel):
    """Response model for quantum capabilities."""

    quantum_enabled: bool
    available_query_types: List[str]
    max_qubits: int
    coherence_time_ms: float
    gate_fidelity: float
    quantum_volume: int
    supported_algorithms: List[str]
    quantum_gates_supported: List[str]


class QuantumResultResponse(BaseModel):
    """Response model for quantum query results."""

    query_id: str
    classical_answer: str
    quantum_enhanced_answer: str
    quantum_advantage: float
    confidence_score: float
    processing_time_ms: float
    quantum_metadata: Dict[str, Any]
    citations: List[Dict[str, Any]]
    processing_metadata: Dict[str, Any]
    created_at: str


class CacheStatsResponse(BaseModel):
    """Response model for cache statistics."""

    entries: int
    size_mb: float
    max_size_mb: float
    utilization: float
    hit_rate: float
    hit_count: int
    miss_count: int
    eviction_count: int


class PerformanceProfileResponse(BaseModel):
    """Response model for performance profile data."""

    operation_name: str
    total_calls: int
    successful_calls: int
    error_rate: float
    avg_duration_ms: float
    min_duration_ms: float
    max_duration_ms: float
    avg_memory_delta_mb: float
    last_call: str


# Create API router
router = APIRouter(prefix="/quantum", tags=["Quantum Computing"])

# Global instances
_photonic_bridge: Optional[PhotonicBridge] = None
_qa_engine: Optional[FinancialQAEngine] = None


def get_photonic_bridge() -> PhotonicBridge:
    """Get or create photonic bridge instance."""
    global _photonic_bridge, _qa_engine

    if _photonic_bridge is None:
        if _qa_engine is None:
            _qa_engine = FinancialQAEngine(enable_quantum=True)
        _photonic_bridge = PhotonicBridge(qa_engine=_qa_engine)

    return _photonic_bridge


@router.get("/capabilities", response_model=QuantumCapabilitiesResponse)
async def get_quantum_capabilities():
    """Get quantum computing capabilities and system status."""
    try:
        bridge = get_photonic_bridge()
        capabilities = bridge.get_quantum_capabilities()

        return QuantumCapabilitiesResponse(**capabilities)

    except Exception as e:
        logger.error(f"Failed to get quantum capabilities: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get capabilities: {str(e)}")


@router.post("/query", response_model=QuantumResultResponse)
async def process_quantum_query(request: QuantumQueryRequest):
    """Process a single quantum-enhanced financial query."""
    try:
        bridge = get_photonic_bridge()

        # Prepare document content
        if request.document_content:
            # Create temporary file for document content
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(request.document_content)
                document_path = f.name
        else:
            document_path = request.document_path

        if not document_path:
            raise HTTPException(status_code=400, detail="Document path or content required")

        # Process enhanced query
        result = await bridge.process_enhanced_query_async(
            query=request.query,
            document_path=document_path,
            enable_quantum=request.enable_quantum,
            quantum_threshold=request.quantum_threshold
        )

        # Convert to response model
        result_dict = result.to_dict()
        return QuantumResultResponse(**result_dict)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Quantum query processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@router.post("/batch", response_model=List[QuantumResultResponse])
async def process_quantum_batch(request: QuantumBatchRequest, background_tasks: BackgroundTasks):
    """Process multiple quantum queries in batch."""
    try:
        bridge = get_photonic_bridge()
        results = []

        if request.parallel_processing:
            # Use parallel processing
            parallel_processor = get_parallel_processor()

            # Prepare query functions
            async def process_single_query(query_req: QuantumQueryRequest):
                # Prepare document
                if query_req.document_content:
                    import tempfile
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                        f.write(query_req.document_content)
                        document_path = f.name
                else:
                    document_path = query_req.document_path

                return await bridge.process_enhanced_query_async(
                    query=query_req.query,
                    document_path=document_path,
                    enable_quantum=query_req.enable_quantum,
                    quantum_threshold=query_req.quantum_threshold
                )

            # Process all queries
            tasks = [process_single_query(query_req) for query_req in request.queries]
            enhanced_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Convert results
            for i, result in enumerate(enhanced_results):
                if isinstance(result, Exception):
                    logger.error(f"Batch query {i} failed: {result}")
                    # Create error response
                    error_response = QuantumResultResponse(
                        query_id=f"batch_error_{i}",
                        classical_answer=f"Error: {str(result)}",
                        quantum_enhanced_answer=f"Error: {str(result)}",
                        quantum_advantage=1.0,
                        confidence_score=0.0,
                        processing_time_ms=0.0,
                        quantum_metadata={"error": True},
                        citations=[],
                        processing_metadata={"error": True},
                        created_at=datetime.now().isoformat()
                    )
                    results.append(error_response)
                else:
                    result_dict = result.to_dict()
                    results.append(QuantumResultResponse(**result_dict))

        else:
            # Sequential processing
            for query_req in request.queries:
                try:
                    # Prepare document
                    if query_req.document_content:
                        import tempfile
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                            f.write(query_req.document_content)
                            document_path = f.name
                    else:
                        document_path = query_req.document_path

                    result = await bridge.process_enhanced_query_async(
                        query=query_req.query,
                        document_path=document_path,
                        enable_quantum=query_req.enable_quantum,
                        quantum_threshold=query_req.quantum_threshold
                    )

                    result_dict = result.to_dict()
                    results.append(QuantumResultResponse(**result_dict))

                except Exception as e:
                    logger.error(f"Batch query failed: {e}")
                    error_response = QuantumResultResponse(
                        query_id=f"seq_error_{len(results)}",
                        classical_answer=f"Error: {str(e)}",
                        quantum_enhanced_answer=f"Error: {str(e)}",
                        quantum_advantage=1.0,
                        confidence_score=0.0,
                        processing_time_ms=0.0,
                        quantum_metadata={"error": True},
                        citations=[],
                        processing_metadata={"error": True},
                        created_at=datetime.now().isoformat()
                    )
                    results.append(error_response)

        return results

    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")


@router.post("/benchmark")
async def benchmark_quantum_performance(request: QuantumBenchmarkRequest):
    """Benchmark quantum vs classical performance."""
    try:
        bridge = get_photonic_bridge()

        all_results = []

        for iteration in range(request.iterations):
            logger.info(f"Running benchmark iteration {iteration + 1}/{request.iterations}")

            benchmark_result = bridge.benchmark_quantum_advantage(
                queries=request.test_queries,
                document_paths=request.document_paths
            )

            # Add iteration info
            benchmark_result["iteration"] = iteration + 1
            all_results.append(benchmark_result)

        # Calculate aggregate across iterations if multiple
        if request.iterations > 1:
            # Combine results across iterations
            combined_results = {
                "iterations": request.iterations,
                "individual_iterations": all_results,
                "aggregate_across_iterations": _calculate_benchmark_aggregate(all_results)
            }
            return combined_results
        else:
            return all_results[0]

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(e)}")


@router.get("/cache/stats", response_model=CacheStatsResponse)
async def get_cache_statistics():
    """Get quantum circuit cache statistics."""
    try:
        cache = get_circuit_cache()
        stats = cache.get_stats()

        return CacheStatsResponse(**stats)

    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get cache stats: {str(e)}")


@router.post("/cache/clear")
async def clear_cache():
    """Clear quantum circuit cache."""
    try:
        cache = get_circuit_cache()
        cleared_count = await cache.clear_expired()

        return {
            "message": "Cache cleared successfully",
            "expired_entries_cleared": cleared_count,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")


@router.get("/performance/profiles")
async def get_performance_profiles(
    operation: Optional[str] = Query(None, description="Specific operation to profile")
):
    """Get performance profiling data."""
    try:
        profiler = get_performance_profiler()

        if operation:
            profile_data = profiler.get_profile_summary(operation)
            if "error" in profile_data:
                raise HTTPException(status_code=404, detail=profile_data["error"])
            return PerformanceProfileResponse(**profile_data)
        else:
            all_profiles = profiler.get_all_profiles_summary()
            return {
                "operations": list(all_profiles.keys()),
                "profiles": all_profiles,
                "timestamp": datetime.now().isoformat()
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get performance profiles: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get profiles: {str(e)}")


@router.post("/performance/profiles/clear")
async def clear_performance_profiles():
    """Clear all performance profiling data."""
    try:
        profiler = get_performance_profiler()
        profiler.clear_profiles()

        return {
            "message": "Performance profiles cleared successfully",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to clear profiles: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear profiles: {str(e)}")


@router.get("/health")
async def quantum_health_check():
    """Health check for quantum computing components."""
    try:
        bridge = get_photonic_bridge()
        capabilities = bridge.get_quantum_capabilities()

        cache = get_circuit_cache()
        cache_stats = cache.get_stats()

        optimizer = get_quantum_optimizer()

        parallel_processor = get_parallel_processor()
        processor_stats = await parallel_processor.get_processing_stats()

        return {
            "status": "healthy",
            "quantum_enabled": capabilities.get("quantum_enabled", False),
            "components": {
                "photonic_bridge": "operational",
                "circuit_cache": f"{cache_stats['entries']} entries, {cache_stats['hit_rate']:.1%} hit rate",
                "quantum_optimizer": f"{len(optimizer.optimization_cache)} cached optimizations",
                "parallel_processor": f"{processor_stats['active_tasks']}/{processor_stats['max_workers']} workers"
            },
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


def _calculate_benchmark_aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate aggregate metrics across multiple benchmark iterations."""
    if not results:
        return {}

    # Extract aggregate metrics from each iteration
    all_aggregates = [result.get("aggregate_metrics", {}) for result in results]

    if not any(all_aggregates):
        return {}

    # Calculate means across iterations
    metrics = ["average_quantum_advantage", "average_confidence_improvement", "overall_speedup"]
    aggregate = {}

    for metric in metrics:
        values = [agg.get(metric, 0) for agg in all_aggregates if agg.get(metric) is not None]
        if values:
            aggregate[f"mean_{metric}"] = sum(values) / len(values)
            aggregate[f"min_{metric}"] = min(values)
            aggregate[f"max_{metric}"] = max(values)

    # Sum timing metrics
    timing_metrics = ["total_classical_time_ms", "total_quantum_time_ms"]
    for metric in timing_metrics:
        values = [agg.get(metric, 0) for agg in all_aggregates if agg.get(metric) is not None]
        if values:
            aggregate[f"total_{metric}"] = sum(values)

    return aggregate


# Export the router
__all__ = ["router"]
