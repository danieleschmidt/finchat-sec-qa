#!/usr/bin/env python3
"""Performance benchmarking script to measure API improvements over time."""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Any
from load_test import LoadTester, LoadTestResult


class PerformanceBenchmark:
    """Track performance benchmarks over time."""
    
    def __init__(self, benchmark_file: str = "performance_benchmarks.json"):
        self.benchmark_file = Path(benchmark_file)
        self.benchmarks = self._load_benchmarks()
    
    def _load_benchmarks(self) -> Dict[str, Any]:
        """Load existing benchmarks from file."""
        if self.benchmark_file.exists():
            try:
                with open(self.benchmark_file) as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading benchmarks: {e}")
        
        return {
            "version": "1.0",
            "benchmarks": [],
            "baselines": {}
        }
    
    def _save_benchmarks(self):
        """Save benchmarks to file."""
        with open(self.benchmark_file, "w") as f:
            json.dump(self.benchmarks, f, indent=2)
    
    def add_benchmark(self, version: str, results: List[LoadTestResult], notes: str = ""):
        """Add a new benchmark result."""
        benchmark_data = {
            "version": version,
            "timestamp": time.time(),
            "notes": notes,
            "results": {}
        }
        
        for result in results:
            benchmark_data["results"][result.endpoint] = {
                "requests_per_second": result.requests_per_second,
                "avg_response_time": result.avg_response_time,
                "p95_response_time": result.p95_response_time,
                "error_rate": result.error_rate,
                "total_requests": result.total_requests
            }
        
        self.benchmarks["benchmarks"].append(benchmark_data)
        self._save_benchmarks()
        
        print(f"Benchmark added for version {version}")
    
    def set_baseline(self, version: str):
        """Set a benchmark as the baseline for comparison."""
        benchmark = self._find_benchmark(version)
        if benchmark:
            self.benchmarks["baselines"] = benchmark["results"]
            self._save_benchmarks()
            print(f"Baseline set to version {version}")
        else:
            print(f"Benchmark for version {version} not found")
    
    def _find_benchmark(self, version: str) -> Dict[str, Any]:
        """Find a benchmark by version."""
        for benchmark in self.benchmarks["benchmarks"]:
            if benchmark["version"] == version:
                return benchmark
        return None
    
    def compare_to_baseline(self, results: List[LoadTestResult]) -> Dict[str, Dict[str, float]]:
        """Compare current results to baseline."""
        if not self.benchmarks["baselines"]:
            print("No baseline set for comparison")
            return {}
        
        comparison = {}
        
        for result in results:
            endpoint = result.endpoint
            if endpoint in self.benchmarks["baselines"]:
                baseline = self.benchmarks["baselines"][endpoint]
                
                rps_change = (result.requests_per_second - baseline["requests_per_second"]) / baseline["requests_per_second"] * 100
                latency_change = (result.avg_response_time - baseline["avg_response_time"]) / baseline["avg_response_time"] * 100
                p95_change = (result.p95_response_time - baseline["p95_response_time"]) / baseline["p95_response_time"] * 100
                
                comparison[endpoint] = {
                    "rps_change_percent": rps_change,
                    "latency_change_percent": latency_change, 
                    "p95_change_percent": p95_change,
                    "current_rps": result.requests_per_second,
                    "baseline_rps": baseline["requests_per_second"],
                    "current_latency": result.avg_response_time * 1000,
                    "baseline_latency": baseline["avg_response_time"] * 1000
                }
        
        return comparison
    
    def print_comparison(self, comparison: Dict[str, Dict[str, float]]):
        """Print formatted comparison results."""
        print("\n=== Performance Comparison to Baseline ===")
        
        for endpoint, metrics in comparison.items():
            print(f"\n{endpoint}:")
            print(f"  RPS: {metrics['current_rps']:.1f} vs {metrics['baseline_rps']:.1f} "
                  f"({metrics['rps_change_percent']:+.1f}%)")
            print(f"  Avg Latency: {metrics['current_latency']:.1f}ms vs {metrics['baseline_latency']:.1f}ms "
                  f"({metrics['latency_change_percent']:+.1f}%)")
            print(f"  95th Percentile: {metrics['p95_change_percent']:+.1f}% change")
    
    def generate_report(self) -> str:
        """Generate a performance report."""
        if not self.benchmarks["benchmarks"]:
            return "No benchmarks available"
        
        latest = self.benchmarks["benchmarks"][-1]
        report = f"""
# Performance Benchmark Report

**Version:** {latest['version']}
**Date:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(latest['timestamp']))}
**Notes:** {latest.get('notes', 'N/A')}

## Results

"""
        
        for endpoint, metrics in latest["results"].items():
            report += f"""
### {endpoint}
- **Requests/second:** {metrics['requests_per_second']:.2f}
- **Average latency:** {metrics['avg_response_time'] * 1000:.1f}ms
- **95th percentile:** {metrics['p95_response_time'] * 1000:.1f}ms
- **Error rate:** {metrics['error_rate']:.1f}%
"""
        
        # Add historical comparison if multiple benchmarks exist
        if len(self.benchmarks["benchmarks"]) > 1:
            report += "\n## Historical Trends\n\n"
            
            endpoints = list(latest["results"].keys())
            for endpoint in endpoints:
                report += f"### {endpoint} Performance Over Time\n\n"
                report += "| Version | RPS | Avg Latency (ms) | P95 (ms) |\n"
                report += "|---------|-----|------------------|----------|\n"
                
                for benchmark in self.benchmarks["benchmarks"][-5:]:  # Last 5 benchmarks
                    if endpoint in benchmark["results"]:
                        metrics = benchmark["results"][endpoint]
                        report += f"| {benchmark['version']} | "
                        report += f"{metrics['requests_per_second']:.1f} | "
                        report += f"{metrics['avg_response_time'] * 1000:.1f} | "
                        report += f"{metrics['p95_response_time'] * 1000:.1f} |\n"
                
                report += "\n"
        
        return report


async def run_benchmark(version: str, base_url: str = "http://localhost:8000", notes: str = ""):
    """Run a complete performance benchmark."""
    print(f"Running performance benchmark for version {version}")
    
    benchmark = PerformanceBenchmark()
    
    async with LoadTester(base_url) as tester:
        # Run standardized test suite
        results = []
        
        print("Benchmarking health endpoint...")
        results.append(await tester.test_health_endpoint(requests=1000, concurrency=50))
        
        print("Benchmarking metrics endpoint...")
        results.append(await tester.test_metrics_endpoint(requests=1000, concurrency=50))
        
        print("Benchmarking risk endpoint...")
        results.append(await tester.test_risk_endpoint(requests=500, concurrency=25))
        
        # Light query test to avoid external API limits
        print("Benchmarking query endpoint...")  
        results.append(await tester.test_query_endpoint(requests=20, concurrency=4))
        
        # Add to benchmarks
        benchmark.add_benchmark(version, results, notes)
        
        # Compare to baseline if available
        comparison = benchmark.compare_to_baseline(results)
        if comparison:
            benchmark.print_comparison(comparison)
        
        # Generate and save report
        report = benchmark.generate_report()
        report_file = f"performance_report_{version.replace('.', '_')}.md"
        with open(report_file, "w") as f:
            f.write(report)
        
        print(f"\nBenchmark complete! Report saved to {report_file}")
        return results


def main():
    """Main benchmark runner."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python benchmark.py <version> [base_url] [notes]")
        print("       python benchmark.py set-baseline <version>")
        print("       python benchmark.py report")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "set-baseline":
        if len(sys.argv) < 3:
            print("Usage: python benchmark.py set-baseline <version>")
            sys.exit(1)
        
        version = sys.argv[2]
        benchmark = PerformanceBenchmark()
        benchmark.set_baseline(version)
    
    elif command == "report":
        benchmark = PerformanceBenchmark()
        print(benchmark.generate_report())
    
    else:
        version = command
        base_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000"
        notes = sys.argv[3] if len(sys.argv) > 3 else ""
        
        try:
            asyncio.run(run_benchmark(version, base_url, notes))
        except KeyboardInterrupt:
            print("\nBenchmark interrupted by user")
        except Exception as e:
            print(f"Benchmark failed: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()