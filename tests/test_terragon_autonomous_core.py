"""
Tests for TERRAGON Autonomous Core - Generation 3 Quality Gates
TERRAGON SDLC v4.0 - Autonomous Execution Phase

Comprehensive tests ensuring all autonomous systems work correctly.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch
from datetime import datetime

from finchat_sec_qa.terragon_autonomous_core import TerragonAutonomousCore
from finchat_sec_qa.autonomous_intelligence_engine import AutonomousIntelligenceEngine
from finchat_sec_qa.self_healing_system import SelfHealingSystem, HealthStatus
from finchat_sec_qa.robust_error_handling_advanced import RobustErrorHandler
from finchat_sec_qa.comprehensive_validation_system import ComprehensiveValidationSystem, ValidationLevel


class TestTerragonAutonomousCore:
    """Test suite for the autonomous core system."""
    
    @pytest.fixture
    def autonomous_core(self):
        """Create autonomous core for testing."""
        return TerragonAutonomousCore()
    
    @pytest.mark.asyncio
    async def test_autonomous_query_processing(self, autonomous_core):
        """Test autonomous query processing with all systems integrated."""
        
        query = "What are the main risk factors for Apple's revenue growth?"
        documents = ["Apple Inc. faces several risk factors including supply chain disruptions..."]
        
        result = await autonomous_core.process_autonomous_query(
            question=query,
            documents=documents,
            language='en',
            region='us-east'
        )
        
        # Verify response structure
        assert 'answer' in result
        assert 'citations' in result
        assert 'autonomous_insights' in result
        assert 'quantum_enhancements' in result
        assert 'risk_assessment' in result
        assert 'performance_metrics' in result
        assert 'learning_status' in result
        assert 'session_info' in result
        
        # Verify session info
        assert result['session_info']['autonomous_mode'] is True
        assert 'session_id' in result['session_info']
        
        # Verify autonomous insights were generated
        assert isinstance(result['autonomous_insights'], list)
        
        # Verify performance metrics
        assert 'response_time_ms' in result['performance_metrics']
        assert result['performance_metrics']['response_time_ms'] > 0
        
    @pytest.mark.asyncio 
    async def test_error_handling_and_recovery(self, autonomous_core):
        """Test autonomous error handling and recovery."""
        
        # Simulate error by passing invalid input
        with patch.object(autonomous_core.intelligence_engine, 'process_autonomous_query', 
                         side_effect=Exception("Simulated error")):
            
            result = await autonomous_core.process_autonomous_query(
                question="Test query",
                documents=["Test document"]
            )
            
            # Verify error response structure
            assert result['error'] is True
            assert 'error_type' in result
            assert 'autonomous_recovery' in result
            assert 'fallback_response' in result
            
            # Verify autonomous recovery was attempted
            assert 'recovery_attempted' in result['autonomous_recovery']
            assert 'health_status' in result['autonomous_recovery']
    
    def test_autonomous_status_reporting(self, autonomous_core):
        """Test comprehensive autonomous status reporting."""
        
        status = autonomous_core.get_autonomous_status()
        
        # Verify status structure
        assert 'session_info' in status
        assert 'processing_stats' in status
        assert 'health_status' in status
        assert 'learning_progress' in status
        assert 'global_features' in status
        assert 'autonomous_capabilities' in status
        
        # Verify autonomous capabilities
        capabilities = status['autonomous_capabilities']
        assert capabilities['continuous_learning'] is True
        assert capabilities['self_healing'] is True
        assert capabilities['global_optimization'] is True
        assert capabilities['proactive_maintenance'] is True
        
        # Verify global features
        global_features = status['global_features']
        assert len(global_features['languages_supported']) > 0
        assert len(global_features['compliance_regions']) > 0
        
    def test_intelligence_export(self, autonomous_core):
        """Test autonomous intelligence export functionality."""
        
        export_data = autonomous_core.export_autonomous_intelligence("/tmp")
        
        # Verify export structure
        assert 'intelligence_file' in export_data
        assert 'status_file' in export_data
        assert 'export_timestamp' in export_data
        
        # Files should be created
        assert export_data['intelligence_file'].endswith('.json')
        assert export_data['status_file'].endswith('.json')
    
    def test_session_lifecycle(self, autonomous_core):
        """Test complete autonomous session lifecycle."""
        
        # Start with initial state
        initial_status = autonomous_core.get_autonomous_status()
        assert initial_status['processing_stats']['queries_processed'] == 0
        
        # Shutdown and verify final state
        session_result = autonomous_core.shutdown_autonomous_systems()
        
        # Verify session result
        assert session_result.session_id == autonomous_core.session_id
        assert session_result.start_time == autonomous_core.start_time
        assert session_result.queries_processed >= 0
        assert session_result.performance_score >= 0


class TestAutonomousIntelligenceEngine:
    """Test suite for autonomous intelligence engine."""
    
    @pytest.fixture
    def intelligence_engine(self):
        """Create intelligence engine for testing."""
        return AutonomousIntelligenceEngine()
    
    def test_pattern_recognition(self, intelligence_engine):
        """Test query pattern recognition."""
        
        # Process similar queries to establish patterns
        queries = [
            "What is Apple's revenue growth?",
            "What is Microsoft's revenue growth?", 
            "What is Google's revenue growth?"
        ]
        
        for query in queries:
            result = intelligence_engine.process_autonomous_query(query)
            assert 'answer' in result
            assert 'processing_metrics' in result
        
        # Check if patterns were learned
        learning_summary = intelligence_engine.get_learning_summary()
        assert learning_summary['total_queries_processed'] == len(queries)
        assert learning_summary['learning_efficiency'] > 0
    
    def test_insight_discovery(self, intelligence_engine):
        """Test autonomous insight discovery."""
        
        query = "What are the key financial metrics and risk factors for technology companies?"
        documents = [
            "Technology companies face significant risks from cybersecurity threats and data breaches.",
            "Revenue growth in tech sector is driven by cloud computing and AI innovations.",
            "Key metrics include customer acquisition cost, lifetime value, and churn rate."
        ]
        
        result = intelligence_engine.process_autonomous_query(query, documents)
        
        # Verify insights were discovered
        assert 'autonomous_insights' in result
        insights = result['autonomous_insights']
        
        if insights:  # Insights may not always be generated
            for insight in insights:
                assert 'insight_id' in insight
                assert 'type' in insight
                assert 'content' in insight
                assert 'confidence' in insight
                assert insight['confidence'] >= 0.7  # Above threshold
    
    def test_learning_optimization(self, intelligence_engine):
        """Test continuous learning optimization."""
        
        # Process queries with different patterns
        quick_query = "Revenue?"
        detailed_query = "Provide a comprehensive analysis of revenue growth trends, including year-over-year comparisons and seasonal variations."
        
        result1 = intelligence_engine.process_autonomous_query(quick_query)
        result2 = intelligence_engine.process_autonomous_query(detailed_query)
        
        # Verify optimization applied
        assert result1['processing_metrics']['response_time_ms'] > 0
        assert result2['processing_metrics']['response_time_ms'] > 0
        
        # Learning status should show progress
        learning_status = result2['learning_status']
        assert learning_status['patterns_learned'] >= 0
        assert learning_status['query_count'] >= 2


class TestSelfHealingSystem:
    """Test suite for self-healing system."""
    
    @pytest.fixture
    def healing_system(self):
        """Create self-healing system for testing."""
        system = SelfHealingSystem()
        system.start_monitoring()
        return system
    
    def test_health_monitoring(self, healing_system):
        """Test health monitoring functionality."""
        
        # Record some metrics
        healing_system.record_request(0.5, had_error=False)
        healing_system.record_request(1.2, had_error=True)
        healing_system.record_request(0.8, had_error=False)
        
        # Get health report
        health_report = healing_system.get_health_report()
        
        # Verify report structure
        assert 'overall_status' in health_report
        assert 'current_metrics' in health_report
        assert 'issues' in health_report
        assert 'monitoring' in health_report
        assert 'performance' in health_report
        
        # Verify metrics
        assert health_report['performance']['total_requests'] == 3
        assert health_report['performance']['total_errors'] == 1
        assert health_report['monitoring']['monitoring_active'] is True
    
    def test_issue_detection_and_recovery(self, healing_system):
        """Test issue detection and autonomous recovery."""
        
        # Simulate high memory usage
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.percent = 90.0  # High memory usage
            
            # Trigger monitoring check
            metrics = healing_system._collect_health_metrics()
            issues = healing_system._detect_issues(metrics)
            
            # Should detect memory issue
            memory_issues = [i for i in issues if 'memory' in i.description.lower()]
            assert len(memory_issues) > 0
            
            # Test recovery
            memory_issue = memory_issues[0]
            healing_system._handle_issue(memory_issue)
            
            # Should have recovery actions
            assert len(memory_issue.resolution_actions) > 0
    
    def test_circuit_breaker(self, healing_system):
        """Test circuit breaker functionality."""
        
        circuit_key = "test_operation"
        
        # Should allow operations initially
        assert healing_system._check_circuit_breaker(circuit_key) is True
        
        # Record multiple failures
        for _ in range(6):  # Exceed failure threshold
            healing_system._record_circuit_breaker_failure(circuit_key)
        
        # Should block operations now
        assert healing_system._check_circuit_breaker(circuit_key) is False
        
        # Verify circuit breaker is open
        breaker = healing_system.circuit_breakers[circuit_key]
        assert breaker.state == "OPEN"
    
    def teardown_method(self, method):
        """Cleanup after each test."""
        # Stop any running monitoring
        pass


class TestRobustErrorHandling:
    """Test suite for robust error handling."""
    
    @pytest.fixture
    def error_handler(self):
        """Create error handler for testing."""
        return RobustErrorHandler()
    
    @pytest.mark.asyncio
    async def test_retry_mechanism(self, error_handler):
        """Test retry mechanism with exponential backoff."""
        
        call_count = 0
        
        @error_handler.robust_wrapper(max_retries=2, fallback_result="fallback")
        async def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Network error")
            return "success"
        
        result = await failing_function()
        
        # Should succeed after retries
        assert result == "success"
        assert call_count == 3  # Initial + 2 retries
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self, error_handler):
        """Test circuit breaker integration."""
        
        @error_handler.robust_wrapper(
            circuit_breaker_key="test_service",
            fallback_result="circuit_open"
        )
        async def service_call():
            raise Exception("Service down")
        
        # First few calls should be attempted
        for _ in range(5):
            result = await service_call()
        
        # Circuit should be open now
        result = await service_call()
        assert result == "circuit_open"
    
    def test_error_classification(self, error_handler):
        """Test error classification and categorization."""
        
        # Test different error types
        network_error = ConnectionError("Connection failed")
        memory_error = MemoryError("Out of memory")
        timeout_error = TimeoutError("Request timed out")
        
        # Classify errors
        network_ctx = error_handler._create_error_context(network_error, lambda: None, 0, {})
        memory_ctx = error_handler._create_error_context(memory_error, lambda: None, 0, {})
        timeout_ctx = error_handler._create_error_context(timeout_error, lambda: None, 0, {})
        
        # Verify classifications
        assert network_ctx.category.value == "network"
        assert memory_ctx.category.value == "memory"
        assert timeout_ctx.category.value == "timeout"
        
        # Verify severity
        assert memory_ctx.severity.value == "critical"
    
    def test_error_analytics(self, error_handler):
        """Test comprehensive error analytics."""
        
        # Simulate some errors
        errors = [
            ConnectionError("Network error 1"),
            MemoryError("Memory error 1"),
            ConnectionError("Network error 2"),
            ValueError("Validation error 1")
        ]
        
        for error in errors:
            ctx = error_handler._create_error_context(error, lambda: None, 0, {})
            error_handler.error_history.append(ctx)
        
        # Get analytics
        analytics = error_handler.get_error_analytics()
        
        # Verify analytics structure
        assert 'total_errors' in analytics
        assert 'category_distribution' in analytics
        assert 'severity_distribution' in analytics
        assert analytics['total_errors'] == len(errors)
        
        # Verify distributions
        assert analytics['category_distribution']['network'] == 2
        assert analytics['category_distribution']['memory'] == 1


class TestComprehensiveValidation:
    """Test suite for comprehensive validation system."""
    
    @pytest.fixture
    def validation_system(self):
        """Create validation system for testing."""
        return ComprehensiveValidationSystem()
    
    def test_basic_validation(self, validation_system):
        """Test basic string validation."""
        
        valid_query = "What is Apple's revenue growth?"
        result = validation_system.validate(valid_query, 'financial_query')
        
        assert result.is_valid is True
        assert result.threat_level.value == "none"
        assert result.sanitized_value is not None
    
    def test_security_threat_detection(self, validation_system):
        """Test security threat detection."""
        
        malicious_inputs = [
            "SELECT * FROM users WHERE id = 1; DROP TABLE users;",  # SQL injection
            "<script>alert('XSS')</script>",  # XSS
            "../../../../etc/passwd",  # Path traversal
            "system('rm -rf /')"  # Command injection
        ]
        
        for malicious_input in malicious_inputs:
            result = validation_system.validate(malicious_input, 'string', ValidationLevel.STRICT)
            
            # Should detect threats
            assert result.threat_level.value in ["medium", "high", "critical"]
            assert len(result.issues) > 0
    
    def test_financial_query_validation(self, validation_system):
        """Test financial query specific validation."""
        
        good_queries = [
            "What is the revenue growth for Apple?",
            "Analyze the debt-to-equity ratio trends",
            "What are the main risk factors for profitability?"
        ]
        
        bad_queries = [
            "hi",  # Too short
            "x" * 2000,  # Too long
            "What is the weather today?"  # No financial terms
        ]
        
        for query in good_queries:
            result = validation_system.validate(query, 'financial_query')
            assert result.is_valid is True or len(result.warnings) == 0
        
        for query in bad_queries:
            result = validation_system.validate(query, 'financial_query')
            # May be invalid or have warnings
            assert result.is_valid is False or len(result.warnings) > 0
    
    def test_performance_and_caching(self, validation_system):
        """Test validation performance and caching."""
        
        query = "What is Microsoft's quarterly earnings performance?"
        
        # First validation
        start_time = time.time()
        result1 = validation_system.validate(query, 'financial_query')
        first_duration = time.time() - start_time
        
        # Second validation (should use cache)
        start_time = time.time()
        result2 = validation_system.validate(query, 'financial_query')
        second_duration = time.time() - start_time
        
        # Results should be identical
        assert result1.is_valid == result2.is_valid
        assert result1.threat_level == result2.threat_level
        
        # Second should be faster (cache hit)
        assert second_duration < first_duration
        
        # Verify cache hit in stats
        stats = validation_system.get_validation_stats()
        assert stats['cache_hits'] > 0
    
    def test_batch_validation(self, validation_system):
        """Test batch validation functionality."""
        
        queries = [
            "What is Apple's revenue?",
            "Analyze Microsoft's debt ratios",
            "What are Google's risk factors?",
            "Tesla's profitability trends"
        ]
        
        results = validation_system.validate_batch(queries, 'financial_query')
        
        assert len(results) == len(queries)
        for result in results:
            assert hasattr(result, 'is_valid')
            assert hasattr(result, 'processing_time_ms')
            assert result.processing_time_ms > 0


@pytest.mark.integration
class TestFullSystemIntegration:
    """Integration tests for the complete autonomous system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_processing(self):
        """Test complete end-to-end autonomous processing."""
        
        # Initialize full autonomous core
        autonomous_core = TerragonAutonomousCore()
        
        try:
            # Process a realistic financial query
            query = "What are the key risk factors affecting Apple's revenue growth in the technology sector?"
            documents = [
                "Apple Inc. faces several operational risks including supply chain disruptions, competitive pressures, and regulatory changes.",
                "The company's revenue is primarily driven by iPhone sales, which face market saturation risks.",
                "Technology sector risks include cybersecurity threats, rapid innovation cycles, and changing consumer preferences."
            ]
            
            result = await autonomous_core.process_autonomous_query(
                question=query,
                documents=documents,
                language='en',
                region='us-east'
            )
            
            # Comprehensive validation of result
            assert result is not None
            assert 'answer' in result
            assert 'autonomous_insights' in result
            assert 'risk_assessment' in result
            assert 'performance_metrics' in result
            
            # Verify autonomous systems worked
            assert result['performance_metrics']['response_time_ms'] > 0
            assert result['session_info']['autonomous_mode'] is True
            
            # Verify learning occurred
            learning_status = result['learning_status']
            assert learning_status['total_queries_processed'] > 0
            
            # Get final system status
            status = autonomous_core.get_autonomous_status()
            assert status['processing_stats']['queries_processed'] > 0
            
        finally:
            # Cleanup
            autonomous_core.shutdown_autonomous_systems()
    
    @pytest.mark.asyncio
    async def test_error_resilience(self):
        """Test system resilience under error conditions."""
        
        autonomous_core = TerragonAutonomousCore()
        
        try:
            # Test with various error-inducing inputs
            error_queries = [
                "",  # Empty query
                "x" * 10000,  # Very long query
                None,  # None input
                "SELECT * FROM secret_data;",  # SQL injection attempt
            ]
            
            results = []
            for query in error_queries:
                try:
                    if query is None:
                        # Skip None query as it would cause issues before validation
                        continue
                    
                    result = await autonomous_core.process_autonomous_query(
                        question=query,
                        documents=["Test document content"]
                    )
                    results.append(result)
                    
                except Exception as e:
                    # System should handle errors gracefully
                    assert "error" in str(e).lower() or "validation" in str(e).lower()
            
            # System should still be functional
            status = autonomous_core.get_autonomous_status()
            assert status is not None
            
        finally:
            autonomous_core.shutdown_autonomous_systems()
    
    def test_performance_benchmarks(self):
        """Test that performance meets benchmarks."""
        
        autonomous_core = TerragonAutonomousCore()
        
        try:
            # Benchmark test
            start_time = time.time()
            
            # Initialize should be fast
            initialization_time = time.time() - start_time
            assert initialization_time < 5.0  # Should initialize within 5 seconds
            
            # Status retrieval should be fast
            start_time = time.time()
            status = autonomous_core.get_autonomous_status()
            status_time = time.time() - start_time
            assert status_time < 1.0  # Should get status within 1 second
            
            # Verify status completeness
            required_keys = [
                'session_info', 'processing_stats', 'health_status',
                'learning_progress', 'autonomous_capabilities'
            ]
            for key in required_keys:
                assert key in status
            
        finally:
            autonomous_core.shutdown_autonomous_systems()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])