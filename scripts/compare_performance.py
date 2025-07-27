#!/usr/bin/env python3
"""Compare performance between two benchmark runs."""

import json
import sys
from pathlib import Path


def load_benchmark_data(filepath):
    """Load benchmark data from JSON file."""
    if not Path(filepath).exists():
        return {}
    
    with open(filepath) as f:
        return json.load(f)


def compare_benchmarks(baseline, current):
    """Compare two benchmark datasets."""
    print("Performance Comparison Report")
    print("=" * 40)
    
    # Compare overall metrics
    baseline_avg = baseline.get("average_response_time", 0)
    current_avg = current.get("average_response_time", 0)
    
    if baseline_avg > 0:
        change_pct = ((current_avg - baseline_avg) / baseline_avg) * 100
        print(f"Average Response Time:")
        print(f"  Baseline: {baseline_avg:.2f}ms")
        print(f"  Current:  {current_avg:.2f}ms")
        print(f"  Change:   {change_pct:+.1f}%")
        
        if change_pct > 10:
            print("  ⚠️  PERFORMANCE REGRESSION DETECTED")
        elif change_pct < -5:
            print("  ✅ Performance improved")
        else:
            print("  ✅ Performance stable")
    else:
        print("No baseline data available for comparison")
    
    print("\nDetailed Metrics:")
    print("-" * 20)
    
    # Compare individual test metrics
    baseline_tests = baseline.get("tests", {})
    current_tests = current.get("tests", {})
    
    for test_name in set(baseline_tests.keys()) | set(current_tests.keys()):
        baseline_time = baseline_tests.get(test_name, {}).get("mean", 0)
        current_time = current_tests.get(test_name, {}).get("mean", 0)
        
        if baseline_time > 0 and current_time > 0:
            change = ((current_time - baseline_time) / baseline_time) * 100
            status = "⚠️" if change > 15 else "✅"
            print(f"{test_name}: {baseline_time:.1f}ms → {current_time:.1f}ms ({change:+.1f}%) {status}")
        elif current_time > 0:
            print(f"{test_name}: NEW TEST - {current_time:.1f}ms")
        else:
            print(f"{test_name}: MISSING DATA")


def main():
    """Main function."""
    if len(sys.argv) != 3:
        print("Usage: python compare_performance.py <baseline.json> <current.json>")
        sys.exit(1)
    
    baseline_file = sys.argv[1]
    current_file = sys.argv[2]
    
    baseline_data = load_benchmark_data(baseline_file)
    current_data = load_benchmark_data(current_file)
    
    compare_benchmarks(baseline_data, current_data)


if __name__ == "__main__":
    main()