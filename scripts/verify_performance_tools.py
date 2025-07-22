#!/usr/bin/env python3
"""Verify performance tools work correctly."""

import sys
import json
from pathlib import Path
from load_test import LoadTester, LoadTestResult
from benchmark import PerformanceBenchmark

def test_load_test_result():
    """Test LoadTestResult creation."""
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
        errors=["Timeout"]
    )
    
    assert result.endpoint == "/test"
    assert result.error_rate == 5.0
    print("✓ LoadTestResult creation works")

def test_percentile_calculation():
    """Test percentile calculations."""
    data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    p50 = LoadTester._percentile(data, 50)
    p95 = LoadTester._percentile(data, 95)
    
    assert p50 == 0.5
    assert p95 >= 0.9
    print("✓ Percentile calculation works")

def test_benchmark_functionality():
    """Test benchmark creation and storage."""
    import tempfile
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        benchmark_file = f.name
    
    try:
        benchmark = PerformanceBenchmark(benchmark_file)
        
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
        
        # Verify it was stored
        assert len(benchmark.benchmarks["benchmarks"]) == 1
        assert benchmark.benchmarks["benchmarks"][0]["version"] == "1.4.3"
        
        print("✓ Benchmark storage works")
        
        # Test baseline setting
        benchmark.set_baseline("1.4.3")
        assert benchmark.benchmarks["baselines"]
        print("✓ Baseline setting works")
        
        # Test comparison
        improved_results = [
            LoadTestResult(
                endpoint="/health",
                total_requests=100,
                successful_requests=100,
                failed_requests=0,
                avg_response_time=0.025,  # 50% improvement
                median_response_time=0.020,
                p95_response_time=0.040,
                p99_response_time=0.050,
                requests_per_second=400.0,  # 100% improvement
                error_rate=0.0,
                errors=[]
            )
        ]
        
        comparison = benchmark.compare_to_baseline(improved_results)
        assert "/health" in comparison
        assert comparison["/health"]["rps_change_percent"] > 90  # Should show ~100% improvement
        print("✓ Benchmark comparison works")
        
    finally:
        Path(benchmark_file).unlink(missing_ok=True)

def test_report_generation():
    """Test performance report generation."""
    import tempfile
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        benchmark_file = f.name
    
    try:
        benchmark = PerformanceBenchmark(benchmark_file)
        
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
        
        benchmark.add_benchmark("1.4.3", results, "Test report")
        report = benchmark.generate_report()
        
        assert "Performance Benchmark Report" in report
        assert "1.4.3" in report
        assert "/metrics" in report
        print("✓ Report generation works")
        
    finally:
        Path(benchmark_file).unlink(missing_ok=True)

def main():
    """Run all verification tests."""
    print("Running performance tools verification...")
    
    try:
        test_load_test_result()
        test_percentile_calculation()
        test_benchmark_functionality()
        test_report_generation()
        
        print("\n🎉 All performance tools verified successfully!")
        print("\nYou can now use:")
        print("  python scripts/load_test.py [url]           - Run load tests")
        print("  python scripts/benchmark.py <version> [url] - Run benchmarks")
        print("  python scripts/benchmark.py set-baseline <version> - Set baseline")
        print("  python scripts/benchmark.py report          - View latest report")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)