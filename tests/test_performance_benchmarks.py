"""Performance benchmark tests for the API endpoints."""
from __future__ import annotations

import pytest
import asyncio
import time
from unittest.mock import Mock, patch
from pathlib import Path
import sys

# Add scripts to path for imports
sys.path.append(str(Path(__file__).parent.parent / "scripts"))

try:
    from load_test import LoadTester, LoadTestResult
    from benchmark import PerformanceBenchmark
except ImportError:
    pytest.skip("Load testing dependencies not available", allow_module_level=True)


class TestPerformanceBenchmarks:
    """Test performance benchmarking functionality."""
    
    def test_load_test_result_creation(self):
        """Test LoadTestResult dataclass creation."""
        result = LoadTestResult(
            endpoint="/test",
            total_requests=100,
            successful_requests=95,
            failed_requests=5,
            avg_response_time=0.125,
            median_response_time=0.100,
            p95_response_time=0.250,
            p99_response_time=0.500,
            requests_per_second=45.5,
            error_rate=5.0,
            errors=["Timeout", "Connection error"]
        )
        
        assert result.endpoint == "/test"
        assert result.total_requests == 100
        assert result.error_rate == 5.0
        assert len(result.errors) == 2
    
    def test_performance_benchmark_initialization(self, tmp_path):
        """Test PerformanceBenchmark initialization."""
        benchmark_file = tmp_path / "test_benchmarks.json"
        benchmark = PerformanceBenchmark(str(benchmark_file))
        
        assert "benchmarks" in benchmark.benchmarks
        assert "baselines" in benchmark.benchmarks
        assert isinstance(benchmark.benchmarks["benchmarks"], list)
    
    def test_add_benchmark(self, tmp_path):
        """Test adding benchmark results."""
        benchmark_file = tmp_path / "test_benchmarks.json"
        benchmark = PerformanceBenchmark(str(benchmark_file))
        
        # Create test results
        results = [
            LoadTestResult(
                endpoint="/health",
                total_requests=100,
                successful_requests=100,
                failed_requests=0,
                avg_response_time=0.050,
                median_response_time=0.045,
                p95_response_time=0.080,
                p99_response_time=0.100,
                requests_per_second=200.0,
                error_rate=0.0,
                errors=[]
            )
        ]
        
        benchmark.add_benchmark("1.4.3", results, "Test benchmark")
        
        assert len(benchmark.benchmarks["benchmarks"]) == 1
        assert benchmark.benchmarks["benchmarks"][0]["version"] == "1.4.3"
        assert "/health" in benchmark.benchmarks["benchmarks"][0]["results"]
    
    def test_baseline_comparison(self, tmp_path):
        """Test benchmark comparison to baseline."""
        benchmark_file = tmp_path / "test_benchmarks.json"
        benchmark = PerformanceBenchmark(str(benchmark_file))
        
        # Add baseline
        baseline_results = [
            LoadTestResult(
                endpoint="/health",
                total_requests=100,
                successful_requests=100,
                failed_requests=0,
                avg_response_time=0.100,  # 100ms baseline
                median_response_time=0.090,
                p95_response_time=0.150,
                p99_response_time=0.200,
                requests_per_second=100.0,  # 100 RPS baseline
                error_rate=0.0,
                errors=[]
            )
        ]
        
        benchmark.add_benchmark("1.4.0", baseline_results, "Baseline")
        benchmark.set_baseline("1.4.0")
        
        # Add improved results
        improved_results = [
            LoadTestResult(
                endpoint="/health",
                total_requests=100,
                successful_requests=100,
                failed_requests=0,
                avg_response_time=0.050,  # 50ms - 50% improvement
                median_response_time=0.045,
                p95_response_time=0.075,
                p99_response_time=0.100,
                requests_per_second=200.0,  # 200 RPS - 100% improvement
                error_rate=0.0,
                errors=[]
            )
        ]
        
        comparison = benchmark.compare_to_baseline(improved_results)
        
        assert "/health" in comparison
        health_metrics = comparison["/health"]
        
        # Should show 100% improvement in RPS
        assert abs(health_metrics["rps_change_percent"] - 100.0) < 1.0
        
        # Should show ~50% reduction in latency (negative change)
        assert health_metrics["latency_change_percent"] < -40.0
    
    def test_load_tester_percentile_calculation(self):
        """Test percentile calculation in LoadTester."""
        data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        p50 = LoadTester._percentile(data, 50)
        p95 = LoadTester._percentile(data, 95)
        p99 = LoadTester._percentile(data, 99)
        
        assert p50 == 0.5  # Median
        assert p95 >= 0.9  # Should be close to 95th percentile
        assert p99 >= 0.9  # Should be near the top
    
    def test_performance_report_generation(self, tmp_path):
        """Test performance report generation."""
        benchmark_file = tmp_path / "test_benchmarks.json"
        benchmark = PerformanceBenchmark(str(benchmark_file))
        
        # Add test results
        results = [
            LoadTestResult(
                endpoint="/metrics",
                total_requests=200,
                successful_requests=200,
                failed_requests=0,
                avg_response_time=0.025,
                median_response_time=0.020,
                p95_response_time=0.050,
                p99_response_time=0.075,
                requests_per_second=400.0,
                error_rate=0.0,
                errors=[]
            )
        ]
        
        benchmark.add_benchmark("1.4.3", results, "Metrics endpoint optimization")
        
        report = benchmark.generate_report()
        
        assert "Performance Benchmark Report" in report
        assert "1.4.3" in report
        assert "/metrics" in report
        assert "400.00" in report  # RPS
        assert "25.0ms" in report  # Average latency


class MockAsyncResponse:
    """Mock async response for testing."""
    
    def __init__(self, status=200, text_content="OK"):
        self.status = status
        self._text = text_content
    
    async def text(self):
        return self._text
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class TestLoadTesterUnit:
    """Unit tests for LoadTester that don't require actual HTTP calls."""
    
    def test_analyze_results(self):
        """Test load test result analysis."""
        tester = LoadTester()
        
        # Simulate results: (response_time, success, error)
        results = [
            (0.1, True, None),   # Success
            (0.2, True, None),   # Success
            (0.3, True, None),   # Success
            (0.5, False, "Timeout"),  # Failure
            (0.1, True, None),   # Success
        ]
        
        analyzed = tester.analyze_results("/test", results, 1.0)  # 1 second duration
        
        assert analyzed.endpoint == "/test"
        assert analyzed.total_requests == 5
        assert analyzed.successful_requests == 4
        assert analyzed.failed_requests == 1
        assert analyzed.error_rate == 20.0
        assert analyzed.requests_per_second == 5.0
        assert "Timeout" in analyzed.errors
    
    @pytest.mark.asyncio
    async def test_make_request_success(self):
        """Test successful request simulation."""
        tester = LoadTester()
        
        # Mock the session
        mock_session = Mock()
        mock_response = MockAsyncResponse(200, "OK")
        mock_session.request.return_value = mock_response
        
        tester.session = mock_session
        
        response_time, success, error = await tester.make_request("GET", "/test")
        
        assert success is True
        assert error is None
        assert response_time > 0  # Should have some response time
        mock_session.request.assert_called_once()
    
    @pytest.mark.asyncio 
    async def test_make_request_failure(self):
        """Test failed request simulation."""
        tester = LoadTester()
        
        # Mock the session to raise an exception
        mock_session = Mock()
        mock_session.request.side_effect = Exception("Connection error")
        
        tester.session = mock_session
        
        response_time, success, error = await tester.make_request("GET", "/test")
        
        assert success is False
        assert "Connection error" in error
        assert response_time > 0


def test_scripts_exist():
    """Verify that the load testing scripts exist and are accessible."""
    repo_root = Path(__file__).parent.parent
    
    load_test_script = repo_root / "scripts" / "load_test.py"
    benchmark_script = repo_root / "scripts" / "benchmark.py"
    
    assert load_test_script.exists(), "load_test.py script not found"
    assert benchmark_script.exists(), "benchmark.py script not found"
    
    # Check that scripts have the expected entry points
    with open(load_test_script) as f:
        content = f.read()
        assert "class LoadTester" in content
        assert "async def run_load_tests" in content
    
    with open(benchmark_script) as f:
        content = f.read()
        assert "class PerformanceBenchmark" in content
        assert "async def run_benchmark" in content