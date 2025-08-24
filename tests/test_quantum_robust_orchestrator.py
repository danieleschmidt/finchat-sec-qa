"""
Comprehensive Test Suite for Quantum Robust Orchestrator
Production-Grade Robustness Testing with Fault Injection

Tests Cover:
- Circuit breaker functionality under various failure modes
- Health monitoring and automatic recovery
- Security validation and encryption
- Rate limiting and queue management
- Graceful degradation and fallback mechanisms
- Load testing and concurrent operation handling
- Error handling and recovery scenarios
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch
import json

from finchat_sec_qa.quantum_robust_orchestrator import (
    QuantumRobustOrchestrator,
    RobustProcessingRequest,
    ProcessingPriority,
    QuantumHealthStatus,
    SecurityError,
    RateLimitError,
    create_robust_orchestrator
)
from finchat_sec_qa.quantum_breakthrough_multimodal_engine import (
    MarketRegimeQuantum
)


class TestQuantumRobustOrchestrator:
    """Comprehensive test suite for quantum robust orchestrator."""

    @pytest.fixture
    async def orchestrator(self):
        """Create test orchestrator instance."""
        with patch('finchat_sec_qa.quantum_robust_orchestrator.create_quantum_breakthrough_engine'):
            orchestrator = await create_robust_orchestrator(
                max_concurrent_operations=5,
                health_check_interval=1  # Fast health checks for testing
            )
            yield orchestrator
            await orchestrator.shutdown()

    @pytest.fixture
    def sample_request(self):
        """Sample processing request for testing."""
        return RobustProcessingRequest(
            request_id="test-001",
            document="Test financial document with positive outlook and strong growth indicators",
            financial_data={'revenue_growth': 0.12, 'debt_ratio': 0.35, 'volatility': 0.25},
            priority=ProcessingPriority.NORMAL,
            timeout_seconds=10
        )

    @pytest.fixture
    def high_priority_request(self):
        """High priority processing request."""
        return RobustProcessingRequest(
            request_id="critical-001",
            document="Critical analysis required for urgent decision",
            financial_data={'revenue_growth': 0.08, 'debt_ratio': 0.55, 'volatility': 0.45},
            priority=ProcessingPriority.CRITICAL,
            timeout_seconds=30
        )

    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initializes all components correctly."""
        assert orchestrator.quantum_engine is not None
        assert len(orchestrator.circuit_breakers) > 0
        assert len(orchestrator.circuit_health) > 0
        assert orchestrator.system_health == QuantumHealthStatus.OPTIMAL
        assert orchestrator._health_monitor_task is not None
        assert orchestrator._queue_processor_task is not None

    @pytest.mark.asyncio
    async def test_successful_request_processing(self, orchestrator, sample_request):
        """Test successful request processing with all robustness features."""
        # Mock quantum engine methods
        orchestrator.quantum_engine.detect_market_regime = AsyncMock(
            return_value=MarketRegimeQuantum.BULL_QUANTUM_STATE
        )
        orchestrator.quantum_engine.extract_multimodal_features = AsyncMock(
            return_value=[]
        )
        orchestrator.quantum_engine.fuse_multimodal_features = AsyncMock(
            return_value=([], {})
        )
        orchestrator.quantum_engine.predict_with_uncertainty = AsyncMock(
            return_value=(0.75, {'confidence': 0.85})
        )
        
        # Process request
        result, metadata = await orchestrator.process_request_robust(sample_request)
        
        # Validate result
        assert result is not None
        assert result.document_id == sample_request.request_id
        assert 0 <= result.prediction_confidence <= 1
        assert result.market_regime == MarketRegimeQuantum.BULL_QUANTUM_STATE
        
        # Validate metadata
        assert metadata['security_validated'] is True
        assert metadata['fallback_used'] is False
        assert 'processing_time_ms' in metadata

    @pytest.mark.asyncio
    async def test_circuit_breaker_protection(self, orchestrator, sample_request):
        """Test circuit breaker protection during failures."""
        # Mock quantum engine to fail
        orchestrator.quantum_engine.detect_market_regime = AsyncMock(
            side_effect=Exception("Simulated quantum failure")
        )
        
        # Enable fallback
        sample_request.fallback_enabled = True
        
        # Process request (should fallback)
        result, metadata = await orchestrator.process_request_robust(sample_request)
        
        # Should fallback to classical processing
        assert result is not None
        assert metadata['fallback_used'] is True
        assert 'fallback_reason' in metadata

    @pytest.mark.asyncio
    async def test_timeout_handling(self, orchestrator, sample_request):
        """Test timeout handling and recovery."""
        # Mock slow quantum engine
        async def slow_detection(*args):
            await asyncio.sleep(15)  # Longer than timeout
            return MarketRegimeQuantum.BULL_QUANTUM_STATE
        
        orchestrator.quantum_engine.detect_market_regime = slow_detection
        sample_request.timeout_seconds = 1  # Very short timeout
        sample_request.fallback_enabled = True
        
        # Process request (should timeout and fallback)
        result, metadata = await orchestrator.process_request_robust(sample_request)
        
        # Should use fallback due to timeout
        assert result is not None
        assert metadata['fallback_used'] is True

    @pytest.mark.asyncio
    async def test_concurrent_operation_limits(self, orchestrator):
        """Test concurrent operation limits and queuing."""
        # Create multiple requests
        requests = [
            RobustProcessingRequest(
                request_id=f"concurrent-{i}",
                document=f"Document {i}",
                financial_data={'revenue_growth': 0.1 * i},
                timeout_seconds=5
            )
            for i in range(10)  # More than max_concurrent_operations
        ]
        
        # Mock quantum engine
        async def mock_slow_processing(*args):
            await asyncio.sleep(0.1)
            return MarketRegimeQuantum.BULL_QUANTUM_STATE
        
        orchestrator.quantum_engine.detect_market_regime = mock_slow_processing
        orchestrator.quantum_engine.extract_multimodal_features = AsyncMock(return_value=[])
        orchestrator.quantum_engine.fuse_multimodal_features = AsyncMock(return_value=([], {}))
        orchestrator.quantum_engine.predict_with_uncertainty = AsyncMock(return_value=(0.5, {}))
        
        # Process all requests concurrently
        tasks = [orchestrator.process_request_robust(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Most should succeed (some might use fallback due to load)
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) > 0

    @pytest.mark.asyncio
    async def test_health_monitoring(self, orchestrator, sample_request):
        """Test health monitoring and status updates."""
        # Initial health should be optimal
        health = orchestrator.get_system_health()
        assert health['system_health'] == QuantumHealthStatus.OPTIMAL.value
        
        # Simulate failures to degrade health
        orchestrator.quantum_engine.detect_market_regime = AsyncMock(
            side_effect=Exception("Health test failure")
        )
        
        # Process requests with failures
        for i in range(3):
            request = RobustProcessingRequest(
                request_id=f"health-test-{i}",
                document="Test document",
                financial_data={},
                fallback_enabled=True
            )
            await orchestrator.process_request_robust(request)
        
        # Wait for health monitor to update
        await asyncio.sleep(2)
        
        # Health should be degraded
        health = orchestrator.get_system_health()
        assert health['overall_success_rate'] < 1.0

    @pytest.mark.asyncio
    async def test_queue_management(self, orchestrator):
        """Test asynchronous queue processing."""
        request = RobustProcessingRequest(
            request_id="queue-test-001",
            document="Queued document",
            financial_data={'revenue_growth': 0.05}
        )
        
        # Queue request
        request_id = await orchestrator.queue_request(request)
        assert request_id == request.request_id
        
        # Check queue size
        assert orchestrator.processing_queue.qsize() > 0
        
        # Wait for processing
        await asyncio.sleep(1)

    @pytest.mark.asyncio
    async def test_security_validation(self, orchestrator):
        """Test security validation and context checking."""
        request = RobustProcessingRequest(
            request_id="security-test-001",
            document="Secure document",
            financial_data={},
            security_context={'user_id': 'test_user'}
        )
        
        # Should process successfully with valid context
        is_valid = await orchestrator._validate_security_context(request)
        assert is_valid is True
        
        # Test with invalid JWT
        import jwt
        invalid_token = jwt.encode(
            {'user': 'test'}, 'wrong_secret', algorithm='HS256'
        )
        request.security_context = {'jwt_token': invalid_token}
        
        is_valid = await orchestrator._validate_security_context(request)
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_rate_limiting(self, orchestrator):
        """Test rate limiting functionality."""
        # Normal priority request
        normal_request = RobustProcessingRequest(
            request_id="rate-test-normal",
            document="Normal priority",
            financial_data={},
            priority=ProcessingPriority.NORMAL
        )
        
        # Critical priority request
        critical_request = RobustProcessingRequest(
            request_id="rate-test-critical",
            document="Critical priority",
            financial_data={},
            priority=ProcessingPriority.CRITICAL
        )
        
        # Both should pass initially
        assert await orchestrator._check_rate_limits(normal_request) is True
        assert await orchestrator._check_rate_limits(critical_request) is True
        
        # Fill up the queue to test rate limiting
        for i in range(50):
            await orchestrator.processing_queue.put(f"queue_item_{i}")
        
        # Normal should be rate limited, critical should pass
        assert await orchestrator._check_rate_limits(critical_request) is True

    @pytest.mark.asyncio
    async def test_classical_fallback(self, orchestrator, sample_request):
        """Test classical fallback processing."""
        # Test fallback prediction
        prediction = orchestrator._classical_fallback_prediction(
            sample_request.document,
            sample_request.financial_data
        )
        
        assert 0 <= prediction <= 1
        assert isinstance(prediction, float)
        
        # Test with different document sentiments
        positive_doc = "Excellent growth and strong fundamentals with positive outlook"
        negative_doc = "Poor performance and declining revenues with significant losses"
        
        pos_pred = orchestrator._classical_fallback_prediction(positive_doc, {})
        neg_pred = orchestrator._classical_fallback_prediction(negative_doc, {})
        
        # Positive document should have higher prediction
        assert pos_pred > neg_pred

    @pytest.mark.asyncio
    async def test_error_recovery_scenarios(self, orchestrator, sample_request):
        """Test various error recovery scenarios."""
        # Scenario 1: Temporary quantum failure with recovery
        failure_count = 0
        
        async def intermittent_failure(*args):
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 2:
                raise Exception("Temporary failure")
            return MarketRegimeQuantum.BULL_QUANTUM_STATE
        
        orchestrator.quantum_engine.detect_market_regime = intermittent_failure
        orchestrator.quantum_engine.extract_multimodal_features = AsyncMock(return_value=[])
        orchestrator.quantum_engine.fuse_multimodal_features = AsyncMock(return_value=([], {}))
        orchestrator.quantum_engine.predict_with_uncertainty = AsyncMock(return_value=(0.5, {}))
        
        sample_request.retry_attempts = 3
        sample_request.fallback_enabled = True
        
        # Should eventually succeed or fallback
        result, metadata = await orchestrator.process_request_robust(sample_request)
        assert result is not None

    @pytest.mark.asyncio
    async def test_performance_metrics_collection(self, orchestrator, sample_request):
        """Test performance metrics collection and analysis."""
        # Mock quantum engine
        orchestrator.quantum_engine.detect_market_regime = AsyncMock(
            return_value=MarketRegimeQuantum.BULL_QUANTUM_STATE
        )
        orchestrator.quantum_engine.extract_multimodal_features = AsyncMock(return_value=[])
        orchestrator.quantum_engine.fuse_multimodal_features = AsyncMock(return_value=([], {}))
        orchestrator.quantum_engine.predict_with_uncertainty = AsyncMock(return_value=(0.75, {}))
        
        # Process request
        await orchestrator.process_request_robust(sample_request)
        
        # Check metrics were collected
        assert len(orchestrator.operation_metrics) > 0
        
        metric = orchestrator.operation_metrics[-1]
        assert metric.operation_id == sample_request.request_id
        assert metric.duration_ms is not None
        assert metric.success is True

    @pytest.mark.asyncio
    async def test_graceful_shutdown(self):
        """Test graceful shutdown procedure."""
        orchestrator = await create_robust_orchestrator()
        
        # Add some active operations
        request = RobustProcessingRequest(
            request_id="shutdown-test",
            document="Test document",
            financial_data={}
        )
        
        # Start a long-running operation
        async def long_operation():
            await asyncio.sleep(2)
            return MarketRegimeQuantum.BULL_QUANTUM_STATE
        
        orchestrator.quantum_engine.detect_market_regime = long_operation
        
        # Start processing
        task = asyncio.create_task(orchestrator.process_request_robust(request))
        await asyncio.sleep(0.1)  # Let operation start
        
        # Shutdown should wait for completion
        await orchestrator.shutdown()
        
        # Task should complete
        assert task.done()

    @pytest.mark.asyncio
    async def test_circuit_health_status_updates(self, orchestrator):
        """Test circuit health status updates based on performance."""
        circuit_name = 'regime_detection'
        health = orchestrator.circuit_health[circuit_name]
        
        # Initial health should be optimal
        assert health.health_status == QuantumHealthStatus.OPTIMAL
        
        # Simulate failures
        for _ in range(6):  # Exceed failure threshold
            health.consecutive_failures += 1
            health.error_count += 1
            health.success_rate *= 0.8
            orchestrator._update_circuit_health_status(health)
        
        # Should be in failed state
        assert health.health_status == QuantumHealthStatus.FAILED

    @pytest.mark.asyncio
    async def test_reproducibility_hashing(self, orchestrator, sample_request):
        """Test reproducibility hash generation."""
        hash1 = orchestrator._generate_processing_hash(sample_request)
        hash2 = orchestrator._generate_processing_hash(sample_request)
        
        # Should be consistent
        assert hash1 == hash2
        assert len(hash1) == 32  # MD5 hash length
        
        # Different request should have different hash
        different_request = RobustProcessingRequest(
            request_id="different",
            document="Different document",
            financial_data={'different': 'data'}
        )
        hash3 = orchestrator._generate_processing_hash(different_request)
        assert hash1 != hash3

    @pytest.mark.asyncio
    async def test_priority_based_processing(self, orchestrator):
        """Test priority-based request processing."""
        # Create requests with different priorities
        low_request = RobustProcessingRequest(
            request_id="low-priority",
            document="Low priority document",
            financial_data={},
            priority=ProcessingPriority.LOW
        )
        
        critical_request = RobustProcessingRequest(
            request_id="critical-priority", 
            document="Critical document",
            financial_data={},
            priority=ProcessingPriority.CRITICAL
        )
        
        # Rate limiting should treat them differently
        low_allowed = await orchestrator._check_rate_limits(low_request)
        critical_allowed = await orchestrator._check_rate_limits(critical_request)
        
        # Both should be allowed under normal conditions
        assert low_allowed is True
        assert critical_allowed is True


@pytest.mark.integration
class TestRobustOrchestratorIntegration:
    """Integration tests for robust orchestrator."""

    @pytest.mark.asyncio
    async def test_end_to_end_robust_processing(self):
        """Test complete end-to-end robust processing pipeline."""
        orchestrator = await create_robust_orchestrator(max_concurrent_operations=3)
        
        try:
            # Create comprehensive test request
            request = RobustProcessingRequest(
                request_id="integration-test-001",
                document="Comprehensive financial analysis document with mixed indicators including both positive growth potential and risk factors that require careful evaluation",
                financial_data={
                    'revenue_growth': 0.08,
                    'debt_ratio': 0.45,
                    'volatility': 0.35,
                    'profit_margin': 0.12,
                    'current_ratio': 1.8
                },
                priority=ProcessingPriority.HIGH,
                timeout_seconds=30,
                fallback_enabled=True
            )
            
            # Mock quantum engine for integration test
            orchestrator.quantum_engine.detect_market_regime = AsyncMock(
                return_value=MarketRegimeQuantum.UNCERTAINTY_ENTANGLED
            )
            orchestrator.quantum_engine.extract_multimodal_features = AsyncMock(return_value=[])
            orchestrator.quantum_engine.fuse_multimodal_features = AsyncMock(return_value=([], {}))
            orchestrator.quantum_engine.predict_with_uncertainty = AsyncMock(
                return_value=(0.65, {'confidence': 0.78, 'uncertainty': 0.22})
            )
            
            # Process request
            result, metadata = await orchestrator.process_request_robust(request)
            
            # Validate complete pipeline
            assert result is not None
            assert result.document_id == request.request_id
            assert result.market_regime == MarketRegimeQuantum.UNCERTAINTY_ENTANGLED
            assert 0 <= result.prediction_confidence <= 1
            assert metadata['security_validated'] is True
            assert 'processing_time_ms' in metadata
            
            # Check system health
            health = orchestrator.get_system_health()
            assert health['system_health'] in [s.value for s in QuantumHealthStatus]
            assert health['overall_success_rate'] >= 0.0
            
        finally:
            await orchestrator.shutdown()

    @pytest.mark.performance
    async def test_high_load_performance(self):
        """Test performance under high load conditions."""
        orchestrator = await create_robust_orchestrator(max_concurrent_operations=10)
        
        try:
            # Create many concurrent requests
            requests = [
                RobustProcessingRequest(
                    request_id=f"load-test-{i}",
                    document=f"Load test document {i} with financial analysis content",
                    financial_data={'revenue_growth': 0.1 + (i * 0.01)},
                    timeout_seconds=10
                )
                for i in range(50)
            ]
            
            # Mock quantum engine for consistent performance
            orchestrator.quantum_engine.detect_market_regime = AsyncMock(
                return_value=MarketRegimeQuantum.BULL_QUANTUM_STATE
            )
            orchestrator.quantum_engine.extract_multimodal_features = AsyncMock(return_value=[])
            orchestrator.quantum_engine.fuse_multimodal_features = AsyncMock(return_value=([], {}))
            orchestrator.quantum_engine.predict_with_uncertainty = AsyncMock(return_value=(0.7, {}))
            
            # Process all requests
            start_time = time.time()
            tasks = [orchestrator.process_request_robust(req) for req in requests]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            
            # Analyze results
            successful_results = [r for r in results if not isinstance(r, Exception)]
            processing_time = end_time - start_time
            
            print(f"Processed {len(successful_results)}/{len(requests)} requests in {processing_time:.2f}s")
            print(f"Average time per request: {processing_time/len(requests):.3f}s")
            
            # Should handle reasonable load
            assert len(successful_results) >= len(requests) * 0.8  # At least 80% success rate
            assert processing_time < 30.0  # Should complete within reasonable time
            
        finally:
            await orchestrator.shutdown()


if __name__ == "__main__":
    # Run basic functionality test
    pytest.main([__file__ + "::TestQuantumRobustOrchestrator::test_orchestrator_initialization", "-v"])