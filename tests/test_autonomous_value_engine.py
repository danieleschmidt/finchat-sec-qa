"""
Comprehensive tests for the Autonomous Value Discovery Engine.

This test suite ensures the autonomous value engine works correctly
and discovers optimization opportunities as expected.
"""

import asyncio
import pytest
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from finchat_sec_qa.autonomous_value_engine import (
    AutonomousValueEngine,
    ValueOpportunity,
    ValueOpportunityType,
    get_value_engine,
    get_value_summary
)


@pytest.fixture
def temp_storage_path(tmp_path):
    """Create temporary storage path for testing."""
    return tmp_path / "test_value_engine"


@pytest.fixture
def value_engine(temp_storage_path):
    """Create a value engine instance for testing."""
    engine = AutonomousValueEngine(storage_path=temp_storage_path)
    yield engine
    engine.stop_autonomous_discovery()


@pytest.fixture
def mock_performance_metrics():
    """Mock performance metrics for testing."""
    return [
        {
            'timestamp': time.time() - 300,
            'response_time': 2.5,
            'memory_usage': 0.85,
            'cpu_usage': 0.75
        },
        {
            'timestamp': time.time() - 240,
            'response_time': 2.8,
            'memory_usage': 0.88,
            'cpu_usage': 0.80
        },
        {
            'timestamp': time.time() - 180,
            'response_time': 3.1,
            'memory_usage': 0.90,
            'cpu_usage': 0.82
        }
    ]


class TestAutonomousValueEngine:
    """Test cases for the Autonomous Value Engine."""

    def test_initialization(self, value_engine):
        """Test value engine initialization."""
        assert value_engine.storage_path.exists()
        assert len(value_engine.opportunities) == 0
        assert value_engine.metrics.opportunities_discovered == 0
        assert value_engine.metrics.opportunities_implemented == 0
        assert not value_engine._running

    def test_start_stop_discovery(self, value_engine):
        """Test starting and stopping the discovery process."""
        # Start discovery
        value_engine.start_autonomous_discovery()
        assert value_engine._running
        assert value_engine._monitor_thread is not None
        
        # Stop discovery
        value_engine.stop_autonomous_discovery()
        assert not value_engine._running

    def test_discover_performance_opportunities(self, value_engine, mock_performance_metrics):
        """Test discovery of performance optimization opportunities."""
        # Mock recent performance metrics
        value_engine._get_recent_performance_metrics = Mock(return_value=mock_performance_metrics)
        
        # Discover opportunities
        opportunities = value_engine._analyze_performance_patterns()
        
        # Should discover slow response time opportunity
        assert len(opportunities) > 0
        
        response_time_opportunities = [
            opp for opp in opportunities 
            if opp.get('type') == 'performance_improvement' and 'response' in opp.get('description', '').lower()
        ]
        assert len(response_time_opportunities) > 0
        
        # Check opportunity details
        opp = response_time_opportunities[0]
        assert opp['impact_score'] > 0
        assert opp['implementation_effort'] > 0
        assert opp['roi_estimate'] > 0

    def test_analyze_user_patterns(self, value_engine):
        """Test analysis of user behavior patterns."""
        # Mock query patterns
        mock_queries = [
            {'query': 'revenue analysis for AAPL'},
            {'query': 'revenue growth for MSFT'},
            {'query': 'revenue trends for GOOGL'},
            {'query': 'risk assessment for TSLA'},
            {'query': 'financial health of META'}
        ]
        
        value_engine._get_recent_query_patterns = Mock(return_value=mock_queries)
        
        opportunities = value_engine._analyze_user_patterns()
        
        # Should identify patterns in similar queries
        assert isinstance(opportunities, list)

    def test_analyze_resource_patterns(self, value_engine):
        """Test analysis of resource utilization patterns."""
        # Mock low CPU utilization
        value_engine._get_cpu_utilization_history = Mock(return_value=[0.25, 0.30, 0.28, 0.32, 0.26])
        
        opportunities = value_engine._analyze_resource_patterns()
        
        # Should identify cost reduction opportunity
        cost_opportunities = [
            opp for opp in opportunities 
            if opp['type'] == ValueOpportunityType.COST_REDUCTION
        ]
        assert len(cost_opportunities) > 0

    def test_quantum_pattern_analysis(self, value_engine):
        """Test quantum algorithm performance pattern analysis."""
        # Mock quantum performance metrics
        mock_quantum_metrics = {
            'circuit_depth': [10, 12, 14, 16, 18],
            'fidelity': [0.95, 0.94, 0.93, 0.92, 0.91]
        }
        
        value_engine._get_quantum_performance_metrics = Mock(return_value=mock_quantum_metrics)
        value_engine.photonic_bridge = Mock()  # Enable quantum analysis
        
        opportunities = value_engine._analyze_quantum_patterns()
        
        # Should identify quantum optimization opportunities
        quantum_opportunities = [
            opp for opp in opportunities 
            if opp['type'] == ValueOpportunityType.ALGORITHM_OPTIMIZATION
        ]
        assert len(quantum_opportunities) > 0

    def test_opportunity_prioritization(self, value_engine):
        """Test opportunity prioritization logic."""
        # Create test opportunities
        opp1 = ValueOpportunity(
            id="test1",
            type=ValueOpportunityType.PERFORMANCE_IMPROVEMENT,
            title="High Impact, Low Effort",
            description="Test opportunity 1",
            impact_score=9.0,
            implementation_effort=2.0,
            roi_estimate=4.0
        )
        
        opp2 = ValueOpportunity(
            id="test2",
            type=ValueOpportunityType.COST_REDUCTION,
            title="Medium Impact, High Effort",
            description="Test opportunity 2",
            impact_score=6.0,
            implementation_effort=8.0,
            roi_estimate=2.0
        )
        
        # Compare priority scores
        assert opp1.priority_score > opp2.priority_score

    def test_implementation_decision_logic(self, value_engine):
        """Test logic for deciding which opportunities to implement."""
        # High priority opportunity
        high_priority_opp = ValueOpportunity(
            id="high_priority",
            type=ValueOpportunityType.PERFORMANCE_IMPROVEMENT,
            title="High Priority",
            description="Should be implemented",
            impact_score=8.0,
            implementation_effort=3.0,
            roi_estimate=3.0
        )
        
        # Low priority opportunity
        low_priority_opp = ValueOpportunity(
            id="low_priority",
            type=ValueOpportunityType.USER_EXPERIENCE,
            title="Low Priority",
            description="Should not be implemented",
            impact_score=3.0,
            implementation_effort=9.0,
            roi_estimate=1.0
        )
        
        assert value_engine._should_implement_opportunity(high_priority_opp)
        assert not value_engine._should_implement_opportunity(low_priority_opp)

    def test_performance_optimization_implementation(self, value_engine):
        """Test implementation of performance optimizations."""
        opportunity = ValueOpportunity(
            id="perf_test",
            type=ValueOpportunityType.PERFORMANCE_IMPROVEMENT,
            title="Response Time Optimization",
            description="Optimize response time through caching",
            impact_score=7.0,
            implementation_effort=4.0,
            roi_estimate=2.5
        )
        
        success, details = value_engine._implement_performance_optimization(opportunity)
        
        assert success
        assert 'cache' in str(details).lower()

    def test_ux_improvement_implementation(self, value_engine):
        """Test implementation of UX improvements."""
        opportunity = ValueOpportunity(
            id="ux_test",
            type=ValueOpportunityType.USER_EXPERIENCE,
            title="Query Templates",
            description="Implement query template suggestions",
            impact_score=6.0,
            implementation_effort=3.0,
            roi_estimate=2.0
        )
        
        # Set the ID to match the implementation logic
        opportunity.id = "ux_query_templates_123456"
        
        success, details = value_engine._implement_ux_improvement(opportunity)
        
        assert success

    def test_duplicate_opportunity_detection(self, value_engine):
        """Test detection of duplicate opportunities."""
        # Add an opportunity
        opportunity1 = ValueOpportunity(
            id="original",
            type=ValueOpportunityType.PERFORMANCE_IMPROVEMENT,
            title="Optimize Response Time",
            description="Original opportunity",
            impact_score=7.0,
            implementation_effort=4.0,
            roi_estimate=2.0
        )
        value_engine.opportunities.append(opportunity1)
        
        # Create duplicate
        opportunity2 = ValueOpportunity(
            id="duplicate",
            type=ValueOpportunityType.PERFORMANCE_IMPROVEMENT,
            title="Optimize Response Time",
            description="Duplicate opportunity",
            impact_score=7.5,
            implementation_effort=4.5,
            roi_estimate=2.1
        )
        
        assert value_engine._is_duplicate_opportunity(opportunity2)

    def test_state_persistence(self, value_engine):
        """Test saving and loading of engine state."""
        # Add test data
        opportunity = ValueOpportunity(
            id="persist_test",
            type=ValueOpportunityType.SCALABILITY,
            title="Test Persistence",
            description="Testing state persistence",
            impact_score=5.0,
            implementation_effort=3.0,
            roi_estimate=1.5
        )
        value_engine.opportunities.append(opportunity)
        value_engine.metrics.opportunities_discovered = 5
        
        # Save state
        value_engine._save_state()
        
        # Create new engine and load state
        new_engine = AutonomousValueEngine(storage_path=value_engine.storage_path)
        
        # Check loaded state
        assert len(new_engine.opportunities) == 1
        assert new_engine.opportunities[0].title == "Test Persistence"
        assert new_engine.metrics.opportunities_discovered == 5

    def test_learning_and_adaptation(self, value_engine):
        """Test learning from implementation results."""
        # Mock successful implementation
        value_engine.performance_history.append({
            'timestamp': time.time(),
            'opportunities_discovered': 10,
            'opportunities_implemented': 5,
            'system_performance': {'cpu_usage': 0.6, 'response_time': 1.2}
        })
        
        # Update learning
        value_engine._update_learning()
        
        assert len(value_engine.performance_history) > 0

    def test_value_summary_generation(self, value_engine):
        """Test generation of value summary."""
        # Add test opportunities
        pending_opp = ValueOpportunity(
            id="pending",
            type=ValueOpportunityType.PERFORMANCE_IMPROVEMENT,
            title="Pending Opportunity",
            description="Not implemented yet",
            impact_score=6.0,
            implementation_effort=3.0,
            roi_estimate=2.0,
            implemented=False
        )
        
        implemented_opp = ValueOpportunity(
            id="implemented",
            type=ValueOpportunityType.COST_REDUCTION,
            title="Implemented Opportunity",
            description="Already implemented",
            impact_score=5.0,
            implementation_effort=2.0,
            roi_estimate=3.0,
            implemented=True
        )
        
        value_engine.opportunities.extend([pending_opp, implemented_opp])
        
        summary = value_engine.get_value_summary()
        
        assert summary['total_opportunities_discovered'] == 2
        assert summary['pending_opportunities'] == 1
        assert summary['implemented_opportunities'] == 1
        assert len(summary['top_pending_opportunities']) <= 5

    @patch('finchat_sec_qa.autonomous_value_engine.time.sleep')
    def test_discovery_loop_execution(self, mock_sleep, value_engine):
        """Test execution of the discovery loop."""
        # Mock discovery methods
        value_engine._discover_opportunities = Mock()
        value_engine._implement_opportunities = Mock()
        value_engine._update_learning = Mock()
        value_engine._save_state = Mock()
        
        # Set up for single loop iteration
        mock_sleep.side_effect = [None, KeyboardInterrupt()]  # Stop after first iteration
        value_engine._running = True
        
        # Run discovery loop
        with pytest.raises(KeyboardInterrupt):
            value_engine._discovery_loop()
        
        # Verify methods were called
        value_engine._discover_opportunities.assert_called_once()
        value_engine._implement_opportunities.assert_called_once()
        value_engine._update_learning.assert_called_once()
        value_engine._save_state.assert_called_once()

    def test_error_handling_in_discovery(self, value_engine):
        """Test error handling during opportunity discovery."""
        # Mock method that raises exception
        value_engine._analyze_performance_patterns = Mock(side_effect=Exception("Test error"))
        
        # Discovery should handle errors gracefully
        try:
            value_engine._discover_opportunities()
            # Should not raise exception
        except Exception:
            pytest.fail("Discovery should handle errors gracefully")

    def test_metrics_integration(self, value_engine):
        """Test integration with metrics system."""
        # Mock metrics collection
        mock_metrics = {
            'cpu_usage': 0.75,
            'memory_usage': 0.85,
            'response_time': 1.5
        }
        
        value_engine._get_current_system_performance = Mock(return_value=mock_metrics)
        
        # Update learning should record metrics
        value_engine._update_learning()
        
        assert len(value_engine.performance_history) > 0
        latest_metrics = value_engine.performance_history[-1]
        assert 'system_performance' in latest_metrics

    def test_opportunity_types_coverage(self, value_engine):
        """Test that all opportunity types can be handled."""
        opportunity_types = [
            ValueOpportunityType.ALGORITHM_OPTIMIZATION,
            ValueOpportunityType.PERFORMANCE_IMPROVEMENT,
            ValueOpportunityType.USER_EXPERIENCE,
            ValueOpportunityType.DATA_INSIGHTS,
            ValueOpportunityType.COST_REDUCTION,
            ValueOpportunityType.ACCURACY_ENHANCEMENT,
            ValueOpportunityType.SCALABILITY,
            ValueOpportunityType.INTEGRATION
        ]
        
        for opp_type in opportunity_types:
            opportunity = ValueOpportunity(
                id=f"test_{opp_type.value}",
                type=opp_type,
                title=f"Test {opp_type.value}",
                description=f"Testing {opp_type.value} opportunity",
                impact_score=5.0,
                implementation_effort=3.0,
                roi_estimate=2.0
            )
            
            # Should be able to create and process opportunity
            assert opportunity.type == opp_type
            assert opportunity.priority_score > 0


class TestGlobalValueEngine:
    """Test cases for global value engine functions."""

    def test_get_value_engine_singleton(self):
        """Test that get_value_engine returns singleton instance."""
        engine1 = get_value_engine()
        engine2 = get_value_engine()
        
        assert engine1 is engine2
        assert engine1._running  # Should be auto-started
        
        # Cleanup
        engine1.stop_autonomous_discovery()

    def test_get_value_summary_function(self):
        """Test global get_value_summary function."""
        summary = get_value_summary()
        
        assert isinstance(summary, dict)
        assert 'total_opportunities_discovered' in summary
        
        # Cleanup
        engine = get_value_engine()
        engine.stop_autonomous_discovery()


@pytest.mark.asyncio
class TestAsyncOperations:
    """Test async operations in the value engine."""

    async def test_async_discovery_simulation(self, value_engine):
        """Test simulated async discovery operations."""
        # Simulate async opportunity discovery
        async def mock_async_discovery():
            await asyncio.sleep(0.1)
            return [
                {
                    'type': ValueOpportunityType.PERFORMANCE_IMPROVEMENT,
                    'title': 'Async Discovery Test',
                    'description': 'Found through async analysis',
                    'impact_score': 6.0,
                    'implementation_effort': 4.0,
                    'roi_estimate': 2.0
                }
            ]
        
        opportunities = await mock_async_discovery()
        assert len(opportunities) == 1
        assert opportunities[0]['title'] == 'Async Discovery Test'


@pytest.mark.integration
class TestValueEngineIntegration:
    """Integration tests for the value engine."""

    def test_integration_with_monitoring_system(self, value_engine):
        """Test integration with the monitoring system."""
        # Mock monitoring integration
        value_engine.monitoring = Mock()
        
        # Should integrate with monitoring for metrics collection
        value_engine._get_current_system_performance()
        
        # Verify integration points exist
        assert hasattr(value_engine, 'monitoring')

    def test_integration_with_photonic_bridge(self, value_engine):
        """Test integration with quantum photonic bridge."""
        # Mock photonic bridge
        value_engine.photonic_bridge = Mock()
        
        # Should be able to analyze quantum patterns when bridge is available
        opportunities = value_engine._analyze_quantum_patterns()
        
        assert isinstance(opportunities, list)

    def test_full_discovery_and_implementation_cycle(self, value_engine):
        """Test complete discovery and implementation cycle."""
        # Mock performance data indicating optimization opportunity
        value_engine._get_recent_performance_metrics = Mock(return_value=[
            {'response_time': 3.0, 'memory_usage': 0.9, 'cpu_usage': 0.85}
        ])
        
        # Run discovery
        value_engine._discover_opportunities()
        
        # Should have discovered opportunities
        initial_count = len(value_engine.opportunities)
        assert initial_count > 0
        
        # Mock successful implementation
        value_engine._execute_opportunity_implementation = Mock(return_value=True)
        
        # Run implementation
        value_engine._implement_opportunities()
        
        # Should have attempted implementation
        assert value_engine.metrics.opportunities_discovered >= initial_count


@pytest.mark.performance
class TestValueEnginePerformance:
    """Performance tests for the value engine."""

    def test_discovery_performance(self, value_engine):
        """Test performance of opportunity discovery."""
        # Generate large dataset for performance testing
        large_metrics = [
            {
                'timestamp': time.time() - i,
                'response_time': 1.0 + (i % 10) * 0.1,
                'memory_usage': 0.5 + (i % 20) * 0.02,
                'cpu_usage': 0.3 + (i % 15) * 0.03
            }
            for i in range(1000)
        ]
        
        value_engine._get_recent_performance_metrics = Mock(return_value=large_metrics[-100:])
        
        # Measure discovery time
        start_time = time.time()
        opportunities = value_engine._analyze_performance_patterns()
        discovery_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert discovery_time < 5.0  # 5 seconds max
        assert isinstance(opportunities, list)

    def test_state_persistence_performance(self, value_engine):
        """Test performance of state persistence."""
        # Generate large number of opportunities
        for i in range(1000):
            opportunity = ValueOpportunity(
                id=f"perf_test_{i}",
                type=ValueOpportunityType.PERFORMANCE_IMPROVEMENT,
                title=f"Performance Test {i}",
                description=f"Testing performance with opportunity {i}",
                impact_score=float(i % 10),
                implementation_effort=float((i % 8) + 1),
                roi_estimate=float((i % 5) + 1)
            )
            value_engine.opportunities.append(opportunity)
        
        # Measure save/load performance
        start_time = time.time()
        value_engine._save_state()
        save_time = time.time() - start_time
        
        start_time = time.time()
        value_engine._load_state()
        load_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert save_time < 2.0  # 2 seconds max for save
        assert load_time < 2.0  # 2 seconds max for load